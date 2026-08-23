"""Subprocess sandboxing for HELIX agent and evaluator commands."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import os
import shlex
import shutil
import subprocess
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Literal

from helix.backends import BACKEND_AUTH_COMMANDS, DEFAULT_BACKEND_IMAGES
from helix.config import EvaluatorSidecarConfig, SandboxConfig


logger = logging.getLogger(__name__)


HELIX_ARTIFACT_NAMES = {
    ".helix_backend_result.json",
    ".helix_backend_stdout.txt",
    ".helix_backend_stderr.txt",
    "helix_batch.json",
}


@dataclass(frozen=True)
class CandidateAuthVolume:
    """A credential-only volume created for exactly one agent candidate."""

    name: str
    backend: str
    labels: tuple[tuple[str, str], ...]


class CredentialCleanupError(RuntimeError):
    """A candidate credential volume could not be removed safely."""


_CANDIDATE_AUTH_PREFIX = "helix-candidate-auth-"
_CANDIDATE_AUTH_LABEL = "helix.auth.candidate"
_RUNNER_UID_GID = "1000:1000"

# Source paths are relative to the operator-facing login volume; targets are
# relative to the candidate-owned volume mounted at the destination below.
AUTH_CREDENTIAL_MANIFEST: dict[str, tuple[tuple[str, str], ...]] = {
    "claude": ((".claude/.credentials.json", ".credentials.json"),),
    "codex": ((".codex/auth.json", "auth.json"),),
    "cursor": ((".cursor/cli-config.json", "cli-config.json"),),
    "gemini": ((".gemini/oauth_creds.json", "oauth_creds.json"),),
    "opencode": ((".local/share/opencode/auth.json", "auth.json"),),
}

AUTH_MOUNT_DESTINATIONS: dict[str, str] = {
    "claude": "/home/node/.claude",
    "codex": "/home/node/.codex",
    "cursor": "/home/node/.cursor",
    "gemini": "/home/node/.gemini",
    "opencode": "/home/node/.local/share/opencode",
}

_AGENT_HOME = "/home/node"
_AGENT_UID, _AGENT_GID = _RUNNER_UID_GID.split(":")

# Claude's transcript bind always lands here -- three path components below
# the agent's tmpfs $HOME, nested under (but never equal to)
# AUTH_MOUNT_DESTINATIONS["claude"].
_CLAUDE_TRANSCRIPT_MOUNT_DESTINATION = f"{_AGENT_HOME}/.claude/projects/-workspace"


@dataclass(frozen=True)
class EvaluatorSidecarRuntime:
    network: str
    container_name: str
    endpoint: str


# Process-wide stack of active evaluator sidecar runtimes guarded by a lock.
#
# A stack (rather than a single global) lets nested ``evaluator_sidecar_runtime``
# context managers on the same thread restore the outer runtime correctly.
#
# We use a process-wide stack rather than ``threading.local`` because the
# evolution loop dispatches per-candidate evaluations to a
# ``ThreadPoolExecutor`` -- those worker threads must see the sidecar that the
# main thread just started. Running two overlapping evolution loops in the
# *same* process is not supported (it would also conflict on cwd, file locks,
# evaluator manifest writes, etc.); concurrent runs should use separate
# processes.
_sidecar_stack: list[EvaluatorSidecarRuntime] = []
_sidecar_stack_lock = threading.RLock()


def current_evaluator_sidecar_runtime() -> EvaluatorSidecarRuntime | None:
    """Return the most recently pushed evaluator sidecar runtime, or ``None``."""
    with _sidecar_stack_lock:
        return _sidecar_stack[-1] if _sidecar_stack else None


@contextmanager
def evaluator_sidecar_runtime(
    runtime: EvaluatorSidecarRuntime,
) -> Iterator[EvaluatorSidecarRuntime]:
    """Push *runtime* onto the active sidecar stack for the duration of the block.

    Nested ``with`` blocks are supported: the innermost runtime wins, and the
    outer runtime is restored on exit even if an exception is raised inside the
    block.
    """
    with _sidecar_stack_lock:
        _sidecar_stack.append(runtime)
    try:
        yield runtime
    finally:
        with _sidecar_stack_lock:
            # Remove the most recent occurrence of *runtime*; tolerate odd
            # stack states (e.g. exceptions during start) without raising in
            # ``finally``.
            for i in range(len(_sidecar_stack) - 1, -1, -1):
                if _sidecar_stack[i] is runtime:
                    del _sidecar_stack[i]
                    break


def _is_supported_workspace_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)


def resolve_sandbox_image(
    sandbox: SandboxConfig, agent_backend: str | None = None
) -> str:
    if sandbox.image:
        return sandbox.image
    if agent_backend is None:
        raise ValueError("agent_backend is required when sandbox.image is not set")
    try:
        return DEFAULT_BACKEND_IMAGES[agent_backend]
    except KeyError as exc:
        raise ValueError(
            f"No default sandbox image for backend: {agent_backend}"
        ) from exc


def _is_helix_artifact_name(name: str) -> bool:
    return name in HELIX_ARTIFACT_NAMES or name == ".agent_task_prompt.md"


def _ignore_for_copy(path: Path) -> bool:
    parts = path.parts
    return (
        ".git" in parts
        or any(part.startswith(".helix") for part in parts)
        or path.name == "helix.toml"
        or path.name == ".env"
        or path.name.startswith(".env.")
    )


def _ignore_for_sync(path: Path) -> bool:
    parts = path.parts
    return (
        ".git" in parts
        or any(part.startswith(".helix") for part in parts)
        or path.name == "helix.toml"
        or _is_helix_artifact_name(path.name)
        or ".agent_internal" in parts
        or path.name == ".env"
        or path.name.startswith(".env.")
    )


def _extract_session_id_from_json_output(stdout: str) -> str | None:
    """Best-effort session id extraction from backend structured stdout."""
    if not stdout.strip():
        return None
    payloads: list[object] = []
    try:
        payloads.append(json.loads(stdout))
    except json.JSONDecodeError:
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payloads.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    # Check top-level payloads first
    for payload in payloads:
        if isinstance(payload, dict):
            for key in ("session_id", "sessionId", "sessionID"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value

    def walk(obj: object) -> Iterator[object]:
        yield obj
        if isinstance(obj, dict):
            for value in obj.values():
                yield from walk(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from walk(value)

    for payload in payloads:
        for node in walk(payload):
            if not isinstance(node, dict) or node is payload:
                continue
            for key in ("session_id", "sessionId", "sessionID"):
                value = node.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _candidate_auth_volume_name(agent_backend: str) -> str:
    return f"{_CANDIDATE_AUTH_PREFIX}{agent_backend}-{uuid.uuid4().hex}"


def _create_candidate_auth_volume(agent_backend: str) -> CandidateAuthVolume:
    """Create a never-reused, labelled candidate credential volume."""
    if agent_backend not in AUTH_CREDENTIAL_MANIFEST:
        raise ValueError(f"No credential manifest for backend: {agent_backend}")
    name = _candidate_auth_volume_name(agent_backend)
    existing = _run_docker(["docker", "volume", "inspect", name], check=False)
    # Only a real inspect payload counts as proof the name is taken: a zero
    # exit code with no JSON object is not evidence a volume exists.
    if existing.returncode == 0 and '"Name"' in (existing.stdout or ""):
        raise RuntimeError(f"refusing to reuse existing candidate auth volume {name}")
    labels = ((_CANDIDATE_AUTH_LABEL, "true"), ("helix.auth.backend", agent_backend))
    args = ["docker", "volume", "create"]
    for key, value in labels:
        args.extend(["--label", f"{key}={value}"])
    args.append(name)
    _run_docker(args)
    return CandidateAuthVolume(name=name, backend=agent_backend, labels=labels)


def _seed_command(agent_backend: str) -> str:
    """Return a fixed allowlist-only copy command for the seed helper."""
    statements = ["set -eu", "umask 077"]
    source_paths: list[str] = []
    copy_statements: list[str] = []
    for source, target in AUTH_CREDENTIAL_MANIFEST[agent_backend]:
        source_path = f"/source/{source}"
        target_path = f"/destination/{target}"
        source_paths.append(source_path)
        statements.extend(
            [
                f"test -f {shlex.quote(source_path)}",
                f"test ! -L {shlex.quote(source_path)}",
                f"test -s {shlex.quote(source_path)}",
                f"test $(wc -c < {shlex.quote(source_path)}) -le 1048576",
                f"test $(stat -c %a {shlex.quote(source_path)}) = 600",
            ]
        )
        copy_statements.extend(
            [
                f"mkdir -p {shlex.quote(str(Path(target_path).parent))}",
                f"cp {shlex.quote(source_path)} {shlex.quote(target_path)}",
                f"chmod 600 {shlex.quote(target_path)}",
            ]
        )
    # Every manifest entry is a JSON credential record. Parsing before copying
    # fails closed on a malformed source without exposing its contents, and
    # without ever writing it into the candidate volume.
    statements.append(
        "python -c "
        + shlex.quote(
            "import json, pathlib, sys; "
            "records=[json.loads(pathlib.Path(item).read_text()) for item in sys.argv[1:]]; "
            "assert all(isinstance(record, dict) and record for record in records)"
        )
        + " "
        + " ".join(shlex.quote(path) for path in source_paths)
    )
    statements.extend(copy_statements)
    # Claude's transcript bind is deliberately nested below the auth mount.
    # Pre-creating it as node avoids Docker synthesising a root-owned parent.
    if agent_backend == "claude":
        statements.append("mkdir -p /destination/projects/-workspace")
    statements.append(f"chown -R {_RUNNER_UID_GID} /destination")
    return "; ".join(statements)


def _seed_candidate_auth_volume(volume: CandidateAuthVolume, image: str) -> None:
    """Copy the manifest's credential files from the login volume to *volume*.

    *image* is the backend's own runner image, resolved by the caller. The seed
    helper must run inside the trust boundary HELIX already accepts for that
    backend; it must never introduce a third-party image of its own.
    """
    args = [
        "docker",
        "run",
        "--rm",
        "--user",
        "root",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{sandbox_auth_volume_name(volume.backend)}:/source:ro",
        "-v",
        f"{volume.name}:/destination:rw",
        image,
        "sh",
        "-c",
        _seed_command(volume.backend),
    ]
    try:
        _run_docker(args)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            f"credential seed failed for backend {volume.backend}; agent was not started"
        ) from exc


def _remove_candidate_auth_volume(volume: CandidateAuthVolume) -> None:
    """Remove only a volume HELIX created and labelled as candidate-owned."""
    labels = dict(volume.labels)
    if (
        not volume.name.startswith(_CANDIDATE_AUTH_PREFIX)
        or volume.name.startswith("helix-auth-")
        or labels.get(_CANDIDATE_AUTH_LABEL) != "true"
    ):
        raise CredentialCleanupError(
            f"refusing credential cleanup for unsafe volume identifier {volume.name!r}"
        )
    result = _run_docker(["docker", "volume", "rm", volume.name], check=False)
    if result.returncode != 0:
        raise CredentialCleanupError(
            "credential cleanup failed; remove candidate volume manually: "
            f"docker volume rm {volume.name}"
        )


def _matches_omitted_path(path: Path, omitted: set[Path]) -> bool:
    """Return whether a relative path is equal to or under an omitted path."""
    for item in omitted:
        if path == item or item in path.parents:
            return True
    return False


def _copy_tree_contents(
    src: Path,
    dst: Path,
    *,
    for_sync: bool = False,
    skip_special_files: bool = True,
    omit_paths: set[Path] | None = None,
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    ignore = _ignore_for_sync if for_sync else _ignore_for_copy
    omitted = omit_paths or set()
    try:
        children = list(src.iterdir())
    except OSError as exc:
        logger.warning("skipping inaccessible sandbox directory %s: %s", src, exc)
        return
    for child in children:
        rel = child.relative_to(src)
        if ignore(rel) or _matches_omitted_path(rel, omitted):
            continue
        if skip_special_files and not _is_supported_workspace_file(child):
            continue
        target = dst / child.name
        try:
            child_is_symlink = child.is_symlink()
            child_is_dir = child.is_dir()
        except OSError as exc:
            logger.warning("skipping inaccessible sandbox path %s: %s", child, exc)
            continue
        if child_is_symlink:
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            os.symlink(os.readlink(child), target)
        elif child_is_dir:
            if target.exists() or target.is_symlink():
                if not target.is_dir() or target.is_symlink():
                    target.unlink()
            _copy_tree_contents(
                child,
                target,
                for_sync=for_sync,
                skip_special_files=skip_special_files,
                omit_paths={
                    path.relative_to(rel)
                    for path in omitted
                    if path != rel and rel in path.parents
                },
            )
        else:
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            try:
                shutil.copy2(child, target)
            except PermissionError as exc:
                logger.warning("skipping inaccessible sandbox file %s: %s", child, exc)


def _remove_extraneous_files(
    src: Path,
    dst: Path,
    *,
    skip_special_files: bool = True,
    omit_paths: set[Path] | None = None,
) -> None:
    omitted = omit_paths or set()
    try:
        children = list(dst.iterdir())
    except OSError as exc:
        logger.warning("skipping inaccessible destination directory %s: %s", dst, exc)
        return
    for child in children:
        rel = child.relative_to(dst)
        if _ignore_for_sync(rel) or _matches_omitted_path(rel, omitted):
            continue
        if skip_special_files and not _is_supported_workspace_file(child):
            continue
        source_child = src / rel
        try:
            source_exists = source_child.exists() or source_child.is_symlink()
        except OSError as exc:
            logger.warning(
                "keeping %s because sandbox source %s is inaccessible: %s",
                child,
                source_child,
                exc,
            )
            continue
        if not source_exists:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
            continue
        if child.is_dir() and not child.is_symlink():
            _remove_extraneous_files(
                src / rel,
                child,
                skip_special_files=skip_special_files,
                omit_paths={
                    path.relative_to(rel)
                    for path in omitted
                    if path != rel and rel in path.parents
                },
            )


def _safe_rmtree(path: Path, *, docker_image: str | None = None) -> None:
    try:
        shutil.rmtree(path)
        return
    except FileNotFoundError:
        return
    except OSError as first_exc:
        if docker_image:
            if host_owner := _host_owner():
                _docker_chown_workspace(path, docker_image, host_owner)
            else:
                _docker_relax_workspace_permissions(path, docker_image)
            try:
                shutil.rmtree(path)
                return
            except OSError as second_exc:
                logger.warning(
                    "failed to clean sandbox temp dir %s after permission recovery: %s",
                    path,
                    second_exc,
                )
                return
        logger.warning("failed to clean sandbox temp dir %s: %s", path, first_exc)


def _sync_back_workspace(
    src: Path,
    dst: Path,
    *,
    skip_special_files: bool = True,
    omit_paths: set[Path] | None = None,
) -> None:
    _remove_extraneous_files(
        src,
        dst,
        skip_special_files=skip_special_files,
        omit_paths=omit_paths,
    )
    _copy_tree_contents(
        src,
        dst,
        for_sync=True,
        skip_special_files=skip_special_files,
        omit_paths=omit_paths,
    )


def _sync_back_backend_transcripts(src: Path, dst: Path) -> None:
    """Preserve HELIX-owned transcript artifacts while hiding .helix* from agents."""
    source = src / ".helix_artifacts" / "backend_transcripts"
    try:
        if not source.exists():
            return
    except OSError as exc:
        logger.warning("skipping inaccessible backend transcript artifacts %s: %s", source, exc)
        return
    target = dst / ".helix_artifacts" / "backend_transcripts"
    target.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, target, dirs_exist_ok=True)
    except OSError as exc:
        logger.warning("skipping backend transcript artifact copy from %s: %s", source, exc)


def _init_synthetic_git_repo(workspace: Path) -> None:
    """Create local-only git metadata so agent CLIs can inspect status."""
    # Build a clean environment that ignores host-level git config and any
    # GIT_* overrides inherited from the parent process so the synthetic repo
    # is fully self-contained.
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_CONFIG_GLOBAL"] = os.devnull
    env["GIT_CONFIG_SYSTEM"] = os.devnull
    # Disable any user/system hooks that could veto commits.
    env["GIT_TERMINAL_PROMPT"] = "0"
    init_args = [
        "git",
        "-c",
        "init.defaultBranch=main",
        "-c",
        "core.hooksPath=/dev/null",
        "init",
    ]
    subprocess.run(init_args, cwd=workspace, check=True, capture_output=True, env=env)
    subprocess.run(
        ["git", "config", "user.name", "HELIX Sandbox"],
        cwd=workspace,
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        ["git", "config", "user.email", "helix-sandbox@noreply"],
        cwd=workspace,
        check=True,
        capture_output=True,
        env=env,
    )
    subprocess.run(
        ["git", "add", "-A"], cwd=workspace, check=True, capture_output=True, env=env
    )
    subprocess.run(
        [
            "git",
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--allow-empty",
            "-m",
            "helix: sandbox baseline",
        ],
        cwd=workspace,
        check=True,
        capture_output=True,
        env=env,
    )


def _run_workspace_helper(
    workspace: Path,
    image: str,
    sh_command: str,
    extra_args: list[str] | None = None,
    *,
    helper_name: str,
) -> None:
    """Run a one-shot ``docker run`` helper bind-mounting ``workspace``.

    All workspace-recovery helpers share the same security flags (root user,
    no network, no-new-privileges) and the same bind-mount, so factor the
    boilerplate here. Failures are logged but never raised: callers want
    best-effort recovery, and surfacing the exit status helps diagnose
    otherwise-confusing downstream sync/cleanup errors.
    """
    args = [
        "docker",
        "run",
        "--rm",
        "--workdir",
        "/workspace",
        "--user",
        "root",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{workspace}:/workspace:rw",
        image,
        "sh",
        "-c",
        sh_command,
    ]
    if extra_args:
        args.extend(extra_args)
    result = subprocess.run(
        args,
        check=False,
        capture_output=True,
        text=True,
        env=_docker_host_env(),
    )
    if result.returncode != 0:
        logger.warning(
            "docker workspace helper %s failed for %s: rc=%s stderr=%s",
            helper_name,
            workspace,
            result.returncode,
            (result.stderr or "").strip(),
        )


def _docker_chown_workspace(workspace: Path, image: str, owner: str) -> None:
    _run_workspace_helper(
        workspace,
        image,
        'find /workspace -path /workspace/.git -prune -o -exec chown -h "$0" {} +',
        extra_args=[owner],
        helper_name="chown",
    )


def _docker_relax_workspace_permissions(workspace: Path, image: str) -> None:
    """Make a sandbox workspace readable/writable by the host after userns runs.

    Rootless Docker and user-namespace remapping can make container-owned files
    appear on the host as unmapped high UIDs. In that mode chowning to the host
    UID from inside the container is not meaningful, but a root helper in the
    same namespace can still relax mode bits on the bind mount before host-side
    sync and cleanup. ``a+rwX`` (capital ``X``) preserves existing executable
    bits on regular files and keeps directories traversable, unlike a flat
    ``0666``/``0777`` pair.
    """
    _run_workspace_helper(
        workspace,
        image,
        "find /workspace -path /workspace/.git -prune -o -exec chmod a+rwX {} +",
        helper_name="relax",
    )


def _host_owner() -> str | None:
    """Return the ``UID:GID`` to chown the workspace back to after a container run.

    Returns ``None`` whenever container UIDs do not map directly to host UIDs --
    e.g. on macOS, Windows, Docker Desktop on Linux, rootless Docker, or any
    Docker context that uses a remote daemon or user-namespace remapping --
    in which case the caller should skip the chown step entirely.
    """
    import sys

    if sys.platform in ("darwin", "win32"):
        return None
    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        return None
    try:
        info = subprocess.run(
            [
                "docker",
                "info",
                "--format",
                "{{.OperatingSystem}}|{{.SecurityOptions}}|{{.Name}}",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=_docker_host_env(),
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        # If we cannot ask the daemon, do not guess: skipping the chown is
        # safer than chowning to a UID the host cannot map.
        return None
    if info.returncode != 0:
        return None
    out = (info.stdout or "").lower()
    # Rootless Docker, Docker Desktop (incl. Linux), Lima/Colima/OrbStack/podman
    # machine, and userns-remapped daemons all break the assumption that
    # container UIDs match host UIDs. Skip chown in any of these.
    indeterminate_markers = (
        "rootless",
        "docker desktop",
        "colima",
        "orbstack",
        "lima",
        "userns",
    )
    if any(marker in out for marker in indeterminate_markers):
        return None
    # Remote daemons (docker context with a non-local DOCKER_HOST) also break
    # local UID assumptions.
    docker_host = os.environ.get("DOCKER_HOST", "")
    if docker_host and not (
        docker_host.startswith("unix://")
        or docker_host.startswith("fd://")
        or docker_host == ""
    ):
        return None
    return f"{os.getuid()}:{os.getgid()}"


def _docker_host_env() -> dict[str, str]:
    env = {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}
    for k in os.environ:
        if k.startswith("DOCKER_"):
            env[k] = os.environ[k]
    return env


REDACTED_DOCKER_ENV_VALUE = "<redacted>"


def _redact_docker_argv(args: Sequence[str]) -> list[str]:
    """Return ``args`` with every literal Docker env *value* replaced.

    Docker accepts ``-e KEY=VALUE``, ``--env KEY=VALUE``, ``-eKEY=VALUE`` and
    ``--env=KEY=VALUE``.  Only the value is replaced; the key is preserved
    because it is what makes a rendered command diagnosable.

    This is a purely structural rewrite of the argv — it never inspects or
    rewrites captured stdout/stderr.  Substring-scrubbing evaluator output
    against every env value would also rewrite the non-secret variables HELIX
    itself injects (``HOME=/home/node``, ``PATH=...``), mangling container
    tracebacks and, on a non-zero exit, the output callers still parse.
    """
    redacted = list(args)
    index = 0
    while index < len(redacted):
        arg = redacted[index]
        if arg in {"-e", "--env"} and index + 1 < len(redacted):
            assignment = redacted[index + 1]
            if "=" in assignment:
                key, _ = assignment.split("=", 1)
                redacted[index + 1] = f"{key}={REDACTED_DOCKER_ENV_VALUE}"
            index += 1
        elif arg.startswith("--env=") and "=" in arg.removeprefix("--env="):
            key, _ = arg.removeprefix("--env=").split("=", 1)
            redacted[index] = f"--env={key}={REDACTED_DOCKER_ENV_VALUE}"
        elif arg.startswith("-e") and arg != "-e" and "=" in arg[2:]:
            key, _ = arg[2:].split("=", 1)
            redacted[index] = f"-e{key}={REDACTED_DOCKER_ENV_VALUE}"
        index += 1
    return redacted


def _redact_subprocess_exception(
    exc: subprocess.CalledProcessError | subprocess.TimeoutExpired,
    args: Sequence[str],
) -> None:
    """Strip Docker env values from a subprocess exception, in place.

    ``cmd`` and ``args`` are both rewritten: ``repr()`` and traceback rendering
    read ``args``, while HELIX's own error formatting reads ``cmd``.
    """
    safe_args = _redact_docker_argv(args)
    exc.cmd = safe_args
    if isinstance(exc, subprocess.CalledProcessError):
        exc.args = (exc.returncode, safe_args)
    else:
        exc.args = (safe_args, exc.timeout)


def _run_docker_process(
    args: list[str],
    *,
    check: bool = False,
    capture_output: bool = True,
    cwd: str | None = None,
    input_text: str | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Docker command, keeping env values out of its rendered argv."""
    try:
        result = subprocess.run(
            args,
            check=check,
            capture_output=capture_output,
            text=True,
            cwd=cwd,
            input=input_text,
            env=_docker_host_env(),
            timeout=timeout,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        _redact_subprocess_exception(exc, args)
        raise
    result.args = _redact_docker_argv(args)
    return result


def _run_docker(
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return _run_docker_process(args, check=check)


def _build_add_host_args(
    *,
    add_host_gateway: bool,
    extra_hosts: dict[str, str] | None,
) -> list[str]:
    args: list[str] = []
    if add_host_gateway:
        args.extend(["--add-host", "host.docker.internal:host-gateway"])
    for host, target in (extra_hosts or {}).items():
        args.extend(["--add-host", f"{host}:{target}"])
    return args


def _wait_for_container_running(container_name: str, timeout_seconds: int) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_stderr = ""
    while time.monotonic() < deadline:
        result = _run_docker(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Running}} {{.State.Status}}",
                container_name,
            ],
            check=False,
        )
        if result.returncode == 0:
            status = result.stdout.strip().split()
            if status[:1] == ["true"]:
                return
            if len(status) > 1 and status[1] in {"exited", "dead"}:
                logs = _run_docker(["docker", "logs", container_name], check=False)
                raise RuntimeError(
                    "Evaluator sidecar exited before it became ready.\n"
                    f"stdout:\n{logs.stdout}\nstderr:\n{logs.stderr}"
                )
        last_stderr = result.stderr
        time.sleep(0.25)
    raise TimeoutError(
        f"Evaluator sidecar did not become running within {timeout_seconds}s. "
        f"{last_stderr}".strip()
    )


def _default_sidecar_healthcheck_command() -> list[str]:
    return [
        "python",
        "-c",
        (
            "import os, sys, urllib.error, urllib.request\n"
            "url = os.environ['HELIX_EVALUATOR_ENDPOINT']\n"
            "try:\n"
            "    urllib.request.urlopen(url, timeout=2).close()\n"
            "except urllib.error.HTTPError:\n"
            "    sys.exit(0)\n"
            "except Exception as exc:\n"
            "    print(exc, file=sys.stderr)\n"
            "    sys.exit(1)\n"
        ),
    ]


def _healthcheck_docker_args(
    sidecar: EvaluatorSidecarConfig,
    *,
    network: str,
    extra_hosts: dict[str, str] | None = None,
) -> list[str]:
    command = (
        shlex.split(sidecar.healthcheck_command)
        if sidecar.healthcheck_command
        else _default_sidecar_healthcheck_command()
    )
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        network,
        "--security-opt",
        "no-new-privileges",
        *_build_add_host_args(add_host_gateway=False, extra_hosts=extra_hosts),
        "-e",
        f"HELIX_EVALUATOR_ENDPOINT={sidecar.endpoint}",
        sidecar.resolved_runner_image,
        *command,
    ]


def _wait_for_sidecar_service(
    sidecar: EvaluatorSidecarConfig,
    *,
    network: str,
    container_name: str,
    extra_hosts: dict[str, str] | None = None,
) -> None:
    deadline = time.monotonic() + sidecar.startup_timeout_seconds
    last_output = ""
    while time.monotonic() < deadline:
        status = _run_docker(
            [
                "docker",
                "inspect",
                "-f",
                "{{.State.Running}} {{.State.Status}}",
                container_name,
            ],
            check=False,
        )
        if status.returncode == 0:
            parts = status.stdout.strip().split()
            if len(parts) > 1 and parts[1] in {"exited", "dead"}:
                logs = _run_docker(["docker", "logs", container_name], check=False)
                raise RuntimeError(
                    "Evaluator sidecar exited before its endpoint became ready.\n"
                    f"stdout:\n{logs.stdout}\nstderr:\n{logs.stderr}"
                )
        result = _run_docker(
            _healthcheck_docker_args(
                sidecar,
                network=network,
                extra_hosts=extra_hosts,
            ),
            check=False,
        )
        if result.returncode == 0:
            return
        last_output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        time.sleep(0.5)
    raise TimeoutError(
        "Evaluator sidecar endpoint did not become reachable within "
        f"{sidecar.startup_timeout_seconds}s: {sidecar.endpoint}\n{last_output}"
    )


@contextmanager
def start_evaluator_sidecar(
    sidecar: EvaluatorSidecarConfig,
    *,
    passthrough_env: list[str] | None = None,
    fixed_env: dict[str, str] | None = None,
    extra_hosts: dict[str, str] | None = None,
) -> Iterator[EvaluatorSidecarRuntime]:
    suffix = uuid.uuid4().hex[:12]
    network = f"helix-eval-{suffix}"
    container_name = f"helix-evaluator-{suffix}"
    net_cmd = ["docker", "network", "create"]
    if sidecar.internal_network:
        net_cmd.append("--internal")
    net_cmd.append(network)
    _run_docker(net_cmd)
    try:
        args = [
            "docker",
            "run",
            "-d",
            "--name",
            container_name,
            "--network",
            network,
            "--network-alias",
            "helix-evaluator",
            "--security-opt",
            "no-new-privileges",
        ]
        args.extend(
            _build_add_host_args(add_host_gateway=False, extra_hosts=extra_hosts)
        )
        sidecar_env_names = list(
            dict.fromkeys([*(passthrough_env or []), *sidecar.passthrough_env])
        )
        for key in sidecar_env_names:
            if key in os.environ:
                args.extend(["-e", f"{key}={os.environ[key]}"])
        for key, value in (fixed_env or {}).items():
            args.extend(["-e", f"{key}={value}"])
        args.append(sidecar.image)
        args.extend(shlex.split(sidecar.command))
        _run_docker(args)
        _wait_for_container_running(container_name, sidecar.startup_timeout_seconds)
        _wait_for_sidecar_service(
            sidecar,
            network=network,
            container_name=container_name,
            extra_hosts=extra_hosts,
        )
        runtime = EvaluatorSidecarRuntime(
            network=network,
            container_name=container_name,
            endpoint=sidecar.endpoint,
        )
        with evaluator_sidecar_runtime(runtime):
            yield runtime
    finally:
        _run_docker(["docker", "rm", "-f", container_name], check=False)
        _run_docker(["docker", "network", "rm", network], check=False)


def sandbox_auth_volume_name(agent_backend: str) -> str:
    if agent_backend not in DEFAULT_BACKEND_IMAGES:
        raise ValueError(f"No default sandbox auth volume for backend: {agent_backend}")
    return f"helix-auth-{agent_backend}"


def _tmpfs_owned_by_agent(path: str) -> str:
    """A ``--tmpfs`` value mounting *path* writable by the agent's uid/gid."""
    return f"{path}:rw,uid={_AGENT_UID},gid={_AGENT_GID},mode=700"


def _synthesized_ancestor_tmpfs_args(mount_destinations: list[str]) -> list[str]:
    """``--tmpfs`` args pre-owning directories Docker would otherwise root-own.

    Docker synthesises every directory between the agent's tmpfs ``$HOME`` and
    a deeper ``-v`` mount destination itself -- as ``root:root 0755`` --
    regardless of ``$HOME``'s own ownership. Any destination nested more than
    one path component below ``$HOME`` (an ``AUTH_MOUNT_DESTINATIONS`` entry
    such as opencode's, or Claude's transcript bind) therefore leaves an
    intervening directory the uid-1000 agent cannot write into.

    *mount_destinations* is every path this docker invocation will mount
    directly under ``$HOME`` in this call -- derived by the caller from
    ``AUTH_MOUNT_DESTINATIONS`` and the transcript bind, never hardcoded here.
    For each, every ancestor directory strictly between ``$HOME`` and that
    destination needs pre-owning, except one already nested under a
    *different* entry in *mount_destinations*: that entry's own volume
    (seeded and chowned by ``_seed_command``) or tmpfs already owns everything
    beneath it, and stacking another tmpfs inside it would only shadow that
    ownership rather than extend it.
    """
    home = PurePosixPath(_AGENT_HOME)
    ordered: list[str] = []
    seen: set[str] = set()
    for destination in mount_destinations:
        ancestors: list[str] = []
        current = PurePosixPath(destination).parent
        while current != home:
            current_str = str(current)
            covered_by_another = any(
                other != destination
                and (current_str == other or current_str.startswith(other + "/"))
                for other in mount_destinations
            )
            if covered_by_another:
                break
            ancestors.append(current_str)
            current = current.parent
        for path in reversed(ancestors):  # shallowest first
            if path not in seen:
                seen.add(path)
                ordered.append(path)
    args: list[str] = []
    for path in ordered:
        args.extend(["--tmpfs", _tmpfs_owned_by_agent(path)])
    return args


def _transcript_bind_dir(
    sandbox: SandboxConfig,
    workspace: Path,
    scope: Literal["agent", "evaluator"],
    agent_backend: str | None,
) -> Path | None:
    """Host path bind-mounted over claude's in-container transcript directory.

    ``None`` when this run has no such bind. The directory must be created
    before the workspace is chowned to ``node:node`` so that chown covers it;
    creating it afterwards fails on native Linux whenever the host UID is not
    1000, and silently produces a bind directory the container user cannot
    write when the host is root.
    """
    if scope != "agent" or agent_backend != "claude":
        return None
    if not sandbox.preserve_backend_transcripts:
        return None
    return workspace / sandbox.transcript_artifact_dir / "claude"


def _docker_args(
    command: list[str],
    env: dict[str, str],
    workspace: Path,
    sandbox: SandboxConfig,
    scope: Literal["agent", "evaluator"],
    image: str,
    agent_backend: str | None,
    candidate_auth_volume: CandidateAuthVolume | None = None,
    network: str | None = None,
    container_name: str | None = None,
) -> list[str]:
    args = [
        "docker",
        "run",
        "--rm",
        "--workdir",
        "/workspace",
        "--user",
        "node",
        "--network",
        network or sandbox.network,
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{workspace}:/workspace:rw",
    ]
    if container_name:
        args.extend(["--name", container_name])
    if scope == "agent":
        if agent_backend is None:
            raise ValueError("agent_backend is required for sandboxed agent commands")
        args.extend(["--tmpfs", _tmpfs_owned_by_agent(_AGENT_HOME)])

        candidate_destination: str | None = None
        if sandbox.auth == "volume":
            if candidate_auth_volume is None:
                raise ValueError("volume auth requires a candidate credential volume")
            candidate_destination = AUTH_MOUNT_DESTINATIONS[agent_backend]

        transcript_dir = _transcript_bind_dir(sandbox, workspace, scope, agent_backend)
        transcript_destination = (
            _CLAUDE_TRANSCRIPT_MOUNT_DESTINATION if transcript_dir is not None else None
        )

        # Pre-own whatever ancestor directories the mounts below would
        # otherwise leave root-owned -- see F2/F3: under `auth = "env"` with
        # transcripts on, nothing mounts at AUTH_MOUNT_DESTINATIONS["claude"]
        # itself; under the default volume mode, opencode's destination is
        # three levels deep and its own parents are never mounted at all.
        args.extend(
            _synthesized_ancestor_tmpfs_args(
                [d for d in (candidate_destination, transcript_destination) if d is not None]
            )
        )

        if candidate_destination is not None:
            assert candidate_auth_volume is not None
            args.extend(
                ["-v", f"{candidate_auth_volume.name}:{candidate_destination}:rw"]
            )
        if transcript_dir is not None:
            args.extend(
                ["-v", f"{transcript_dir}:{_CLAUDE_TRANSCRIPT_MOUNT_DESTINATION}:rw"]
            )

    if sandbox.pids_limit is not None:
        args.extend(["--pids-limit", str(sandbox.pids_limit)])
    if sandbox.cpus is not None:
        args.extend(["--cpus", str(sandbox.cpus)])
    if sandbox.memory is not None:
        args.extend(["--memory", sandbox.memory])
    args.extend(
        _build_add_host_args(
            add_host_gateway=sandbox.add_host_gateway,
            extra_hosts=sandbox.extra_hosts,
        )
    )

    container_env = {
        key: value for key, value in env.items() if key not in {"HOME", "PATH"}
    }
    container_env["HOME"] = "/home/node"
    container_env["PATH"] = (
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    if scope == "agent" and agent_backend == "opencode":
        container_env["XDG_DATA_HOME"] = "/home/node/.local/share"

    for key, value in container_env.items():
        args.extend(["-e", f"{key}={value}"])

    args.append(image)
    args.extend(command)
    return args


def run_sandboxed_commands(
    commands: list[list[str]],
    *,
    cwd: str | Path,
    env: dict[str, str],
    sandbox: SandboxConfig,
    scope: Literal["agent", "evaluator"],
    sync_back: bool,
    image: str | None = None,
    agent_backend: str | None = None,
    input_text: str | None = None,
) -> list[subprocess.CompletedProcess[str]]:
    """Run commands in one Docker sandbox workspace copy."""
    if not commands:
        raise ValueError("at least one sandbox command is required")
    if input_text is not None and len(commands) != 1:
        raise ValueError("input_text is only supported for a single sandbox command")
    source = Path(cwd).resolve()
    docker_image = image or sandbox.image
    if docker_image is None:
        raise ValueError("sandbox image must be provided")
    tmp_path = Path(tempfile.mkdtemp(prefix="helix-sandbox-"))
    candidate_auth_volume: CandidateAuthVolume | None = None
    try:
        workspace = tmp_path / "workspace"
        omit_paths = (
            {Path(item) for item in sandbox.omit_from_agent}
            if scope == "agent"
            else set()
        )
        _copy_tree_contents(
            source,
            workspace,
            skip_special_files=sandbox.skip_special_files,
            omit_paths=omit_paths,
        )
        _init_synthetic_git_repo(workspace)
        if transcript_dir := _transcript_bind_dir(sandbox, workspace, scope, agent_backend):
            transcript_dir.mkdir(parents=True, exist_ok=True)
        _docker_chown_workspace(workspace, docker_image, "node:node")
        if scope == "agent" and sandbox.auth == "volume":
            if agent_backend is None:
                raise ValueError("agent_backend is required for volume auth")
            candidate_auth_volume = _create_candidate_auth_volume(agent_backend)
            _seed_candidate_auth_volume(candidate_auth_volume, docker_image)
        sidecar_runtime = (
            current_evaluator_sidecar_runtime() if scope == "evaluator" else None
        )
        command_env = dict(env)
        if sidecar_runtime is not None:
            command_env["HELIX_EVALUATOR_ENDPOINT"] = sidecar_runtime.endpoint
        results = []
        try:
            for command in commands:
                container_name = f"helix-cmd-{uuid.uuid4().hex[:12]}"
                docker_cmd = _docker_args(
                    command,
                    command_env,
                    workspace,
                    sandbox,
                    scope,
                    docker_image,
                    agent_backend,
                    candidate_auth_volume,
                    sidecar_runtime.network if sidecar_runtime is not None else None,
                    container_name=container_name,
                )
                try:
                    results.append(
                        _run_docker_process(
                            docker_cmd,
                            cwd=str(source),
                            input_text=input_text,
                            timeout=sandbox.timeout_seconds,
                        )
                    )
                finally:
                    _run_docker(["docker", "rm", "-f", container_name], check=False)
        finally:
            if host_owner := _host_owner():
                _docker_chown_workspace(workspace, docker_image, host_owner)
            else:
                _docker_relax_workspace_permissions(workspace, docker_image)
        if sync_back:
            _sync_back_workspace(
                workspace,
                source,
                skip_special_files=sandbox.skip_special_files,
                omit_paths=omit_paths,
            )
            if scope == "agent":
                _sync_back_backend_transcripts(workspace, source)
        return results
    finally:
        try:
            if candidate_auth_volume is not None:
                _remove_candidate_auth_volume(candidate_auth_volume)
        finally:
            _safe_rmtree(tmp_path, docker_image=docker_image)


def run_sandboxed_command(
    command: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str],
    sandbox: SandboxConfig,
    scope: Literal["agent", "evaluator"],
    sync_back: bool,
    image: str | None = None,
    agent_backend: str | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run one command in a Docker sandbox using a copy of *cwd* as workspace."""
    return run_sandboxed_commands(
        [command],
        cwd=cwd,
        env=env,
        sandbox=sandbox,
        scope=scope,
        sync_back=sync_back,
        image=image,
        agent_backend=agent_backend,
        input_text=input_text,
    )[0]


def sandbox_auth_docker_args(
    agent_backend: str,
    *,
    image: str,
    action: Literal["login", "status", "logout"],
    network: str = "bridge",
    add_host_gateway: bool = False,
    extra_hosts: dict[str, str] | None = None,
    interactive: bool = False,
) -> list[str]:
    try:
        command = BACKEND_AUTH_COMMANDS[agent_backend][action]
    except KeyError as exc:
        raise ValueError(
            f"No sandbox auth {action!r} command for backend: {agent_backend}"
        ) from exc

    args = [
        "docker",
        "run",
        "--rm",
        "--workdir",
        "/workspace",
        "--user",
        "node",
        "--network",
        network,
        "--security-opt",
        "no-new-privileges",
        *_build_add_host_args(
            add_host_gateway=add_host_gateway, extra_hosts=extra_hosts
        ),
        "-v",
        f"{sandbox_auth_volume_name(agent_backend)}:/home/node:rw",
        "-e",
        "HOME=/home/node",
        "-e",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    ]
    if interactive:
        args.insert(2, "-it")
    args.append(image)
    args.extend(command)
    return args


def run_sandbox_auth_command(
    agent_backend: str,
    *,
    action: Literal["login", "status", "logout"],
    image: str | None = None,
    network: str = "bridge",
    add_host_gateway: bool = False,
    extra_hosts: dict[str, str] | None = None,
    interactive: bool = False,
) -> subprocess.CompletedProcess[str]:
    docker_image = image or resolve_sandbox_image(
        SandboxConfig(enabled=True), agent_backend
    )
    args = sandbox_auth_docker_args(
        agent_backend,
        image=docker_image,
        action=action,
        network=network,
        add_host_gateway=add_host_gateway,
        extra_hosts=extra_hosts,
        interactive=interactive,
    )
    if interactive:
        return subprocess.run(args, text=True)
    return subprocess.run(args, capture_output=True, text=True)


def run_command(
    command: list[str],
    *,
    cwd: str | Path,
    env: dict[str, str],
    sandbox: SandboxConfig | None,
    scope: Literal["agent", "evaluator"],
    sync_back: bool = False,
    image: str | None = None,
    agent_backend: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a HELIX subprocess either directly or through the configured sandbox."""
    if sandbox is not None and sandbox.enabled:
        return run_sandboxed_command(
            command,
            cwd=cwd,
            env=env,
            sandbox=sandbox,
            scope=scope,
            sync_back=sync_back,
            image=image,
            agent_backend=agent_backend,
        )
    return subprocess.run(
        command,
        shell=False,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
