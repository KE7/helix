"""Subprocess sandboxing for HELIX agent and evaluator commands."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import parse_qsl, unquote, urlsplit

from helix.backends import BACKEND_AUTH_COMMANDS, DEFAULT_BACKEND_IMAGES
from helix.config import EvaluatorSidecarConfig, SandboxConfig
from helix.envpolicy import EnvGrant
from helix.exceptions import (
    SandboxAuthImageError,
    SharedHomeMountError,
    VolumeModeUnsupportedError,
)
from helix.transcripts import capture_claude_transcript
from helix.backend_layout import assert_layout_is_isolatable, layout_for
from helix.sandbox_home import (
    CONTAINER_TRANSCRIPT_PARENT,
    NODE_GID,
    NODE_UID,
    ensure_transcript_host_dir,
    private_home_tmpfs_arg,
    transcript_bind_arg,
    transcript_host_dir,
)


logger = logging.getLogger(__name__)

# Injectable Docker entry point.  The preflight and volume-lifecycle helpers
# take a runner so unit tests never start a container.
DockerRunner = Callable[..., "subprocess.CompletedProcess[str]"]


_REDACTED_DOCKER_ENV_VALUE = "<redacted>"
_SHORT_REDACTION_VALUE_MAX_LENGTH = 3


@dataclass(frozen=True)
class _DiagnosticRedactionPolicy:
    """Secrets grouped by the matching policy their provenance permits."""

    substring_secrets: tuple[str, ...] = ()
    boundary_secrets: tuple[str, ...] = ()


HELIX_ARTIFACT_NAMES = {
    ".helix_backend_result.json",
    ".helix_backend_stdout.txt",
    ".helix_backend_stderr.txt",
    "helix_batch.json",
}


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
        logger.warning(
            "skipping inaccessible backend transcript artifacts %s: %s", source, exc
        )
        return
    target = dst / ".helix_artifacts" / "backend_transcripts"
    target.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copytree(source, target, dirs_exist_ok=True)
    except OSError as exc:
        logger.warning(
            "skipping backend transcript artifact copy from %s: %s", source, exc
        )


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
    result = _run_docker_process(args, check=False)
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
        info = _run_docker_process(
            [
                "docker",
                "info",
                "--format",
                "{{.OperatingSystem}}|{{.SecurityOptions}}|{{.Name}}",
            ],
            check=False,
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


def _docker_env_assignments(args: Sequence[str]) -> list[tuple[str, str]]:
    """Return literal ``KEY=VALUE`` assignments passed to ``docker run``.

    Docker accepts environment values as ``-e KEY=VALUE``, ``--env KEY=VALUE``,
    ``-eKEY=VALUE``, and ``--env=KEY=VALUE``.  Keeping this parser next to the
    subprocess boundary ensures every diagnostic path applies the same policy.
    """
    assignments: list[tuple[str, str]] = []
    index = 0
    while index < len(args):
        arg = args[index]
        assignment: str | None = None
        if arg in {"-e", "--env"} and index + 1 < len(args):
            assignment = args[index + 1]
            index += 1
        elif arg.startswith("--env="):
            assignment = arg.removeprefix("--env=")
        elif arg.startswith("-e") and arg != "-e":
            assignment = arg[2:]
        if assignment is not None and "=" in assignment:
            key, value = assignment.split("=", 1)
            assignments.append((key, value))
        index += 1
    return assignments


def _docker_diagnostic_redaction_policy(
    args: Sequence[str],
    explicit_values: Sequence[str] = (),
    inherited_policy: _DiagnosticRedactionPolicy | None = None,
) -> _DiagnosticRedactionPolicy:
    """Classify Docker secrets by their required diagnostic matching policy.

    Any explicit ``KEY=VALUE`` passed to Docker can carry sensitive data,
    regardless of how the key is named.  Callers may also supply values from
    an earlier Docker invocation when later output (such as ``docker logs``)
    is rendered as part of the same failure. Literal env and explicit values
    are always substring-matched. Only short alphanumeric endpoint-derived
    components use boundaries, unless literal provenance overrides that rule.
    """
    assignments = _docker_env_assignments(args)
    substring_secrets = {value for _key, value in assignments if value}
    substring_secrets.update(value for value in explicit_values if value)
    derived_secrets: set[str] = set()
    for key, value in assignments:
        if key == "HELIX_EVALUATOR_ENDPOINT":
            derived_secrets.update(_endpoint_component_redaction_values(value))
    if inherited_policy is not None:
        substring_secrets.update(inherited_policy.substring_secrets)
        derived_secrets.update(inherited_policy.boundary_secrets)

    derived_secrets.difference_update(substring_secrets)
    boundary_secrets = {
        secret
        for secret in derived_secrets
        if len(secret) <= _SHORT_REDACTION_VALUE_MAX_LENGTH and secret.isalnum()
    }
    substring_secrets.update(derived_secrets - boundary_secrets)
    return _DiagnosticRedactionPolicy(
        substring_secrets=tuple(
            sorted(substring_secrets, key=lambda secret: (-len(secret), secret))
        ),
        boundary_secrets=tuple(
            sorted(boundary_secrets, key=lambda secret: (-len(secret), secret))
        ),
    )


def _endpoint_component_redaction_values(endpoint: str) -> set[str]:
    """Return sensitive structured pieces of an evaluator endpoint URL.

    A failed URL client may print only userinfo or an isolated raw/decoded
    query or fragment field.  Scrub complete structured fields, ``&``
    segments, and both sides of key/value pairs without hiding the harmless
    scheme, host, or path text that remains useful in diagnostics.
    """
    try:
        parsed = urlsplit(endpoint)
        values = {
            value
            for value in (
                parsed.username,
                parsed.password,
                parsed.query,
                parsed.fragment,
            )
            if value
        }
        for field in (parsed.query, parsed.fragment):
            for segment in field.split("&"):
                if not segment:
                    continue
                values.add(segment)
                raw_key, separator, raw_value = segment.partition("=")
                if raw_key:
                    values.add(raw_key)
                if separator and raw_value:
                    values.add(raw_value)
            values.update(
                component
                for key, value in parse_qsl(field, keep_blank_values=True)
                for component in (key, value)
                if component
            )
        values.update(unquote(value) for value in tuple(values))
    except (UnicodeError, ValueError):
        return set()
    return {value for value in values if value}


def _redact_diagnostic_output(
    value: Any,
    secrets: Sequence[str],
    *,
    boundary_secrets: Sequence[str] = (),
) -> Any:
    """Replace known Docker env secrets in text/bytes diagnostic output.

    ``secrets`` are literal env or explicit values and are always replaced as
    substrings. ``boundary_secrets`` are endpoint-derived values; only short
    alphanumeric members use word boundaries so a query key such as ``q`` does
    not corrupt harmless diagnostic words, hostnames, or paths.
    """
    if isinstance(value, str):
        for secret in secrets:
            value = value.replace(secret, _REDACTED_DOCKER_ENV_VALUE)
        for secret in boundary_secrets:
            if len(secret) <= _SHORT_REDACTION_VALUE_MAX_LENGTH and secret.isalnum():
                value = re.sub(
                    rf"(?<!\w){re.escape(secret)}(?!\w)",
                    _REDACTED_DOCKER_ENV_VALUE,
                    value,
                )
            else:
                value = value.replace(secret, _REDACTED_DOCKER_ENV_VALUE)
    elif isinstance(value, bytes):
        replacement = _REDACTED_DOCKER_ENV_VALUE.encode()
        for secret in secrets:
            value = value.replace(secret.encode(), replacement)
        for secret in boundary_secrets:
            encoded_secret = secret.encode()
            if len(secret) <= _SHORT_REDACTION_VALUE_MAX_LENGTH and secret.isalnum():
                value = re.sub(
                    rb"(?<!\w)" + re.escape(encoded_secret) + rb"(?!\w)",
                    replacement,
                    value,
                )
            else:
                value = value.replace(encoded_secret, replacement)
    return value


def _redact_docker_argv(
    args: Sequence[str],
    *,
    redaction_values: Sequence[str] = (),
    redaction_policy: _DiagnosticRedactionPolicy | None = None,
) -> list[str]:
    """Render Docker argv safely while preserving environment key context.

    Every literal Docker environment value is replaced, including values whose
    key does not look secret.  This avoids heuristic gaps in command/exception
    rendering while retaining the key names needed to diagnose configuration.
    Known values are also scrubbed wherever safely identifiable in another
    argv token.
    """
    policy = _docker_diagnostic_redaction_policy(
        args, redaction_values, redaction_policy
    )
    redacted = list(args)
    index = 0
    while index < len(redacted):
        arg = redacted[index]
        if arg in {"-e", "--env"} and index + 1 < len(redacted):
            assignment = redacted[index + 1]
            if "=" in assignment:
                key, _value = assignment.split("=", 1)
                redacted[index + 1] = f"{key}={_REDACTED_DOCKER_ENV_VALUE}"
            index += 1
        elif arg.startswith("--env="):
            assignment = arg.removeprefix("--env=")
            if "=" in assignment:
                key, _value = assignment.split("=", 1)
                redacted[index] = f"--env={key}={_REDACTED_DOCKER_ENV_VALUE}"
        elif arg.startswith("-e") and arg != "-e":
            assignment = arg[2:]
            if "=" in assignment:
                key, _value = assignment.split("=", 1)
                redacted[index] = f"-e{key}={_REDACTED_DOCKER_ENV_VALUE}"
        index += 1
    for index, arg in enumerate(redacted):
        redacted[index] = _redact_diagnostic_output(
            arg,
            policy.substring_secrets,
            boundary_secrets=policy.boundary_secrets,
        )
    return redacted


def _redact_subprocess_exception(
    exc: subprocess.CalledProcessError | subprocess.TimeoutExpired,
    args: Sequence[str],
    *,
    redaction_values: Sequence[str] = (),
    redaction_policy: _DiagnosticRedactionPolicy | None = None,
) -> None:
    """Sanitize a subprocess exception in place, including indirect rendering."""
    policy = _docker_diagnostic_redaction_policy(
        args, redaction_values, redaction_policy
    )
    safe_args = _redact_docker_argv(args, redaction_policy=policy)
    exc.cmd = safe_args
    if isinstance(exc, subprocess.CalledProcessError):
        exc.args = (exc.returncode, safe_args)
    else:
        exc.args = (safe_args, exc.timeout)
    exc.output = _redact_diagnostic_output(
        exc.output,
        policy.substring_secrets,
        boundary_secrets=policy.boundary_secrets,
    )
    exc.stderr = _redact_diagnostic_output(
        exc.stderr,
        policy.substring_secrets,
        boundary_secrets=policy.boundary_secrets,
    )


def _run_docker_process(
    args: list[str],
    *,
    check: bool = False,
    capture_output: bool = True,
    cwd: str | None = None,
    input_text: str | None = None,
    timeout: float | None = None,
    redaction_values: Sequence[str] = (),
    redaction_policy: _DiagnosticRedactionPolicy | None = None,
    diagnostic_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run Docker, sanitizing argv and any output used as a diagnostic.

    Successful functional stdout/stderr is preserved by default.  Failure
    output is always scrubbed, and callers that intentionally consume a
    successful command's output as diagnostics can opt in with
    ``diagnostic_output=True``.
    """
    policy = _docker_diagnostic_redaction_policy(
        args, redaction_values, redaction_policy
    )
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
        _redact_subprocess_exception(exc, args, redaction_policy=policy)
        raise
    result.args = _redact_docker_argv(args, redaction_policy=policy)
    if result.returncode != 0 or diagnostic_output:
        result.stdout = _redact_diagnostic_output(
            result.stdout,
            policy.substring_secrets,
            boundary_secrets=policy.boundary_secrets,
        )
        result.stderr = _redact_diagnostic_output(
            result.stderr,
            policy.substring_secrets,
            boundary_secrets=policy.boundary_secrets,
        )
    return result


def _run_docker(
    args: list[str],
    *,
    check: bool = True,
    redaction_values: Sequence[str] = (),
    redaction_policy: _DiagnosticRedactionPolicy | None = None,
    diagnostic_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    return _run_docker_process(
        args,
        check=check,
        redaction_values=redaction_values,
        redaction_policy=redaction_policy,
        diagnostic_output=diagnostic_output,
    )


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


def _wait_for_container_running(
    container_name: str,
    timeout_seconds: int,
    *,
    redaction_policy: _DiagnosticRedactionPolicy | None = None,
) -> None:
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
            redaction_policy=redaction_policy,
        )
        if result.returncode == 0:
            status = result.stdout.strip().split()
            if status[:1] == ["true"]:
                return
            if len(status) > 1 and status[1] in {"exited", "dead"}:
                logs = _run_docker(
                    ["docker", "logs", container_name],
                    check=False,
                    redaction_policy=redaction_policy,
                    diagnostic_output=True,
                )
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
    redaction_policy: _DiagnosticRedactionPolicy | None = None,
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
            redaction_policy=redaction_policy,
        )
        if status.returncode == 0:
            parts = status.stdout.strip().split()
            if len(parts) > 1 and parts[1] in {"exited", "dead"}:
                logs = _run_docker(
                    ["docker", "logs", container_name],
                    check=False,
                    redaction_policy=redaction_policy,
                    diagnostic_output=True,
                )
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
            redaction_policy=redaction_policy,
        )
        if result.returncode == 0:
            return
        last_output = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        time.sleep(0.5)
    raise TimeoutError(
        "Evaluator sidecar endpoint did not become reachable within "
        f"{sidecar.startup_timeout_seconds}s: "
        f"HELIX_EVALUATOR_ENDPOINT={_REDACTED_DOCKER_ENV_VALUE}\n{last_output}"
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
        for key in passthrough_env or []:
            if key in os.environ:
                args.extend(["-e", f"{key}={os.environ[key]}"])
        for key, value in (fixed_env or {}).items():
            args.extend(["-e", f"{key}={value}"])
        args.append(sidecar.image)
        args.extend(shlex.split(sidecar.command))
        redaction_policy = _docker_diagnostic_redaction_policy(
            _healthcheck_docker_args(
                sidecar,
                network=network,
                extra_hosts=extra_hosts,
            ),
            inherited_policy=_docker_diagnostic_redaction_policy(args),
        )
        _run_docker(args)
        _wait_for_container_running(
            container_name,
            sidecar.startup_timeout_seconds,
            redaction_policy=redaction_policy,
        )
        _wait_for_sidecar_service(
            sidecar,
            network=network,
            container_name=container_name,
            extra_hosts=extra_hosts,
            redaction_policy=redaction_policy,
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


# ---------------------------------------------------------------------------
# Auth volume lifecycle (R9) and runtime identity (R2/R8)
# ---------------------------------------------------------------------------

# HELIX-owned provenance stamp.  Deliberately NOT a field inside
# ``.credentials.json``: the backend CLI owns that file and rewrites the
# ``claudeAiOauth`` object *wholesale* on every successful refresh, so anything
# HELIX writes inside it is destroyed by the next refresh.  A sibling file in a
# path the CLI never touches travels with the credential and survives refresh.
AUTH_MANIFEST_CONTAINER_PATH = "/home/node/.helix-auth-meta.json"
AUTH_MANIFEST_SCHEMA = 1


@dataclass(frozen=True)
class AuthVolumeManifest:
    """HELIX-authored provenance stamp for a backend auth volume.

    Contains no credential material — backend name, CLI version, image
    reference and a timestamp only.  This is a *skew detector*, never a
    security control: anything able to write the volume can write the
    manifest (and can equally write the credentials themselves).
    """

    backend: str
    cli_version: str
    image: str
    written_at: str
    helix_version: str = ""
    schema: int = AUTH_MANIFEST_SCHEMA

    def to_json(self) -> str:
        return json.dumps(
            {
                "backend": self.backend,
                "cli_version": self.cli_version,
                "image": self.image,
                "written_at": self.written_at,
                "helix_version": self.helix_version,
                "schema": self.schema,
            },
            sort_keys=True,
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> AuthVolumeManifest:
        return cls(
            backend=str(data.get("backend", "")),
            cli_version=str(data.get("cli_version", "")),
            image=str(data.get("image", "")),
            written_at=str(data.get("written_at", "")),
            helix_version=str(data.get("helix_version", "")),
            schema=int(data.get("schema", 0)),
        )


def production_docker_runner() -> DockerRunner:
    """Return the real Docker entry point.

    Exists so that production call sites name their Docker dependency
    explicitly.  Helpers that touch the auth volume take a REQUIRED runner —
    a defaulted one lets a non-production caller reach Docker by omission,
    which is how a real authenticated probe against the shared credential
    volume became reachable from the unit test suite.
    """
    return _run_docker


def docker_volume_exists(volume: str, *, runner: DockerRunner) -> bool:
    """Return True iff the named Docker volume already exists.

    Uses ``docker volume inspect`` and **never** ``docker run -v``.  This is
    load-bearing: ``docker run -v <name>:/path`` *silently creates* a missing
    named volume, which makes "is the volume mounted" true by construction on
    every host and makes observing the volume indistinguishable from
    provisioning it.  Any existence check routed through a container start is
    not an existence check.
    """
    run = runner
    result = run(["docker", "volume", "inspect", volume], check=False)
    return result.returncode == 0


def _auth_image_resolution_error(
    agent_backend: str,
    *,
    searched_dir: Path | None,
    volume_exists: bool | None,
    manifest: AuthVolumeManifest | None,
) -> SandboxAuthImageError:
    """Build the hard, actionable refusal for an undeterminable runner image.

    Every fact here is available without touching the volume's contents, and
    the last one is the operator's one-line fix.
    """
    volume = sandbox_auth_volume_name(agent_backend)
    declined = DEFAULT_BACKEND_IMAGES.get(agent_backend, "<none>")
    lines = [
        "cannot determine which runner image to authenticate against.",
        "",
        "  `helix sandbox login` must use the same image your runs use, or the",
        "  credentials it writes may not be readable by the CLI version your runs",
        "  consume.",
        "",
        f"  backend:          {agent_backend}",
        f"  auth volume:      {volume}",
    ]
    if searched_dir is not None:
        lines.append(f"  no helix.toml in: {searched_dir}")
    if volume_exists is not None:
        lines.append(
            f"  volume exists:    {'yes' if volume_exists else 'no (not provisioned)'}"
        )
    lines.append(
        f"  NOT used:         {declined} "
        "(default tag — would risk producer/consumer CLI skew)"
    )
    if manifest is not None and manifest.image:
        lines.extend(
            [
                "",
                f"  This volume was last written by {manifest.image}"
                + (f" (CLI {manifest.cli_version})" if manifest.cli_version else ""),
                f"  Remedy: helix sandbox login {agent_backend} "
                f"--image {manifest.image}",
            ]
        )
        suggestion = f"helix sandbox login {agent_backend} --image {manifest.image}"
    else:
        suggestion = (
            "Run from a project directory whose helix.toml sets sandbox.image, "
            "or pass --image <ref> explicitly."
        )
        lines.extend(
            [
                "",
                "  Remedy: run this from a project directory containing helix.toml",
                "          with sandbox.image set, or pass --image <ref> explicitly.",
            ]
        )
    return SandboxAuthImageError(
        "\n".join(lines),
        operation="resolve sandbox auth runtime image",
        suggestion=suggestion,
    )


def resolve_auth_runtime_image(
    agent_backend: str,
    *,
    explicit_image: str | None = None,
    sandbox: SandboxConfig | None = None,
    searched_dir: Path | None = None,
    volume_exists: bool | None = None,
    manifest: AuthVolumeManifest | None = None,
) -> str:
    """Resolve the runner image for login/status/logout — never silently.

    R2/R8: ``login``, ``status``, ``logout`` and the run preflight must all
    resolve to the **exact same** image the run will use.  Previously
    ``run_sandbox_auth_command`` constructed a fresh ``SandboxConfig(enabled=True)``
    whose ``image`` is ``None`` by definition, so it always fell through to
    ``DEFAULT_BACKEND_IMAGES[backend]`` (``:latest``) and *could not* use the
    project's pinned runner even in principle.

    There is deliberately **no** ``:latest`` fallback.  On this project's
    release host ``:latest`` is *older* than the pinned digest, so a silent
    default is worse than a version lottery: it writes credentials with one CLI
    for a runner that executes another.
    """
    if explicit_image:
        return explicit_image
    if sandbox is not None and sandbox.image:
        return sandbox.image
    raise _auth_image_resolution_error(
        agent_backend,
        searched_dir=searched_dir,
        volume_exists=volume_exists,
        manifest=manifest,
    )


def read_auth_manifest(
    agent_backend: str,
    *,
    image: str,
    runner: DockerRunner,
) -> AuthVolumeManifest | None:
    """Read the HELIX provenance stamp from an existing auth volume.

    Returns ``None`` when the stamp is absent or unparseable, which means
    **unknown provenance**.  Unknown is never valid and must never be
    silently promoted to valid — but it also must not hard-fail, or this
    change would brick every volume provisioned before stamps existed.

    Mounted ``:ro``: this is *observation*, not authentication.  (The Stage 2
    sufficiency probe is the opposite case and must be ``:rw`` — see
    :func:`preflight_auth`.)  Callers must establish existence with
    :func:`docker_volume_exists` first, since mounting creates.
    """
    run = runner
    volume = sandbox_auth_volume_name(agent_backend)
    result = run(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "node",
            "--network",
            "none",
            "--security-opt",
            "no-new-privileges",
            "-v",
            f"{volume}:/helix-auth-probe:ro",
            image,
            "cat",
            f"/helix-auth-probe/{Path(AUTH_MANIFEST_CONTAINER_PATH).name}",
        ],
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout or "")
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return AuthVolumeManifest.from_mapping(data)


BACKEND_VERSION_COMMANDS: dict[str, list[str]] = {
    "claude": ["claude", "--version"],
    "codex": ["codex", "--version"],
    "cursor": ["cursor-agent", "--version"],
    "gemini": ["gemini", "--version"],
    "opencode": ["opencode", "--version"],
}


def probe_backend_cli_version(
    agent_backend: str,
    *,
    image: str,
    runner: DockerRunner,
) -> str:
    """Return the backend CLI version string reported by ``image``.

    No volume is mounted and the network is disabled: this reads the image,
    not the credential.  Returns ``""`` when the version cannot be determined,
    which downstream is treated as unknown rather than as a match.
    """
    command = BACKEND_VERSION_COMMANDS.get(agent_backend)
    if command is None:
        return ""
    run = runner
    result = run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--security-opt",
            "no-new-privileges",
            image,
            *command,
        ],
        check=False,
    )
    if result.returncode != 0:
        return ""
    return (result.stdout or "").strip().splitlines()[0] if result.stdout else ""


def auth_manifest_write_args(
    agent_backend: str,
    *,
    image: str,
    manifest: AuthVolumeManifest,
) -> list[str]:
    """Docker argv that writes the provenance stamp into the auth volume."""
    volume = sandbox_auth_volume_name(agent_backend)
    name = Path(AUTH_MANIFEST_CONTAINER_PATH).name
    payload = manifest.to_json()
    return [
        "docker",
        "run",
        "--rm",
        "--user",
        "node",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{volume}:/home/node:rw",
        image,
        "sh",
        "-c",
        f"cat > /home/node/{shlex.quote(name)} <<'HELIX_EOF'\n{payload}\nHELIX_EOF",
    ]


def _assert_env_is_granted(
    env: dict[str, str],
    grants: list[EnvGrant] | None,
    scope: Literal["agent", "evaluator"],
) -> None:
    """Second, INDEPENDENT check that every emitted variable carries a grant.

    This is defence in depth, not belt-and-braces duplication.  The original
    bug was created by a *new call site* (``_add_backend_auth_env``) added
    downstream of the control everyone was asserting on: the scrubber reported
    a clean environment while the credential still reached the container.  A
    check that lives at the point of emission — where ``-e KEY=VALUE`` is
    actually constructed — cannot be bypassed by adding another upstream
    mutation, because there is nowhere downstream left to hide.

    Agent scope REQUIRES grants.  A caller that reaches this function with a
    bare dict for an agent container has bypassed the resolver, and that is
    precisely the shape of the regression this guards against, so it is an
    error rather than a permissive fallback.
    """
    if scope == "agent" and grants is None:
        raise ValueError(
            "sandboxed agent environment was built without provenance grants. "
            "Every variable entering a mutation-agent container must be "
            "resolved through helix.envpolicy.resolve_env_grants so its origin "
            "and scope are recorded. Constructing the environment dict "
            "directly bypasses the credential policy."
        )
    if grants is None:
        return
    granted = {g.name for g in grants if g.authorizes(scope)}
    # HOME and PATH are set by _docker_args itself, to container-fixed values.
    ungranted = sorted(set(env) - granted - {"HOME", "PATH"})
    if ungranted:
        raise ValueError(
            f"refusing to pass ungranted environment variable(s) to a "
            f"{scope} container: {', '.join(ungranted)}. "
            "No EnvGrant authorizes these names for this scope."
        )


# Allowed mount DESTINATIONS for an agent container, by exact path.
#
# ``/home/node`` itself is allowed ONLY as a tmpfs (the private per-run HOME).
# The transcript bind is the one non-tmpfs mount permitted beneath HOME.
_AGENT_PRIVATE_HOME = "/home/node"
_AGENT_ALLOWED_HOME_BINDS = (CONTAINER_TRANSCRIPT_PARENT,)


def _mount_destinations(args: list[str]) -> list[tuple[str, str]]:
    """Return ``(destination, kind)`` for every mount-bearing token in *args*.

    A FRESH parser, deliberately not the one inherited from the earlier audit.
    That one took its ``":" in spec`` branch only when ``"=" not in spec``, and
    every tmpfs spec contains ``uid=1000`` -- so the private HOME tmpfs was
    INVISIBLE to it, and its "no mount lands on HOME" result was much narrower
    than it read. This parser handles all four syntaxes explicitly.
    """
    found: list[tuple[str, str]] = []
    for index, token in enumerate(args[:-1]):
        spec = args[index + 1]
        if token == "--tmpfs":
            # ``/dst`` or ``/dst:opt=val,...``
            found.append((spec.split(":", 1)[0], "tmpfs"))
        elif token in {"-v", "--volume"}:
            # ``src:dst[:mode]`` -- destination is the SECOND colon field, and
            # a Windows-style drive letter is not a concern on these hosts.
            parts = spec.split(":")
            if len(parts) >= 2:
                found.append((parts[1], "bind"))
        elif token == "--mount":
            fields = dict(item.split("=", 1) for item in spec.split(",") if "=" in item)
            destination = (
                fields.get("dst") or fields.get("destination") or fields.get("target")
            )
            if destination:
                kind = "tmpfs" if fields.get("type") == "tmpfs" else "bind"
                found.append((destination, kind))
    return found


def _assert_no_shared_home_mount(
    args: list[str], *, allowed_auth_dir: str | None = None
) -> None:
    """Reject ANY mount that could make the agent's HOME shared.

    This is the mount-side counterpart of ``_assert_env_is_granted``, and it
    exists for the same reason that one does: the original defect was created
    by a NEW MOUNT being added, and every existing guard asked "is the AUTH
    VOLUME on HOME?" -- a NAME-scoped question. The property is
    DESTINATION-scoped: *nothing* shared may land on HOME, whatever it is
    called. A rogue bind or a differently-named volume at ``/home/node``
    reproduces the original cross-candidate defect exactly while passing every
    auth-volume assertion.

    Agent scope only. ``helix sandbox login`` legitimately mounts the auth
    volume at HOME -- that is what the volume is for -- and it builds its argv
    without going through this function at all.
    """
    seen_private_home = 0
    for destination, kind in _mount_destinations(args):
        normalised = destination.rstrip("/") or "/"
        if normalised == _AGENT_PRIVATE_HOME:
            if kind != "tmpfs":
                raise SharedHomeMountError(
                    f"a non-tmpfs mount targets the agent's HOME: {destination!r}.\n"
                    "  Whatever its source is called, this makes HOME shared "
                    "across candidates -- the original cross-run defect.\n"
                    "  Only the private per-run tmpfs may target /home/node."
                )
            seen_private_home += 1
            continue
        if normalised in {"/", "/home"}:
            raise SharedHomeMountError(
                f"a mount targets an ANCESTOR of the agent's HOME: "
                f"{destination!r}, which shares HOME transitively."
            )
        if normalised.startswith(_AGENT_PRIVATE_HOME + "/") and kind != "tmpfs":
            # The backend's own auth directory is the ONE shared mount volume
            # mode is allowed, and only at the exact path the registry declares
            # -- a nested mount anywhere else is the S3 shape.
            permitted = set(_AGENT_ALLOWED_HOME_BINDS)
            if allowed_auth_dir:
                permitted.add(allowed_auth_dir.rstrip("/"))
            if normalised not in permitted:
                raise SharedHomeMountError(
                    f"an undeclared non-tmpfs mount targets a path inside the "
                    f"agent's HOME: {destination!r}.\n"
                    "  Only the candidate-keyed transcript bind "
                    f"({CONTAINER_TRANSCRIPT_PARENT}) is permitted there."
                )
    if seen_private_home != 1:
        raise SharedHomeMountError(
            f"expected exactly ONE private tmpfs at {_AGENT_PRIVATE_HOME}; "
            f"found {seen_private_home}. Zero means the agent inherits the "
            f"image's shared HOME; more than one is ambiguous."
        )


def _docker_args(
    command: list[str],
    env: dict[str, str],
    workspace: Path,
    sandbox: SandboxConfig,
    scope: Literal["agent", "evaluator"],
    image: str,
    agent_backend: str | None,
    network: str | None = None,
    container_name: str | None = None,
    grants: list[EnvGrant] | None = None,
) -> list[str]:
    _assert_env_is_granted(env, grants, scope)
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
    class3_env: dict[str, str] = {}
    if scope == "agent":
        if agent_backend is None:
            raise ValueError("agent_backend is required for sandboxed agent commands")
        if sandbox.resolved_auth() == "env":
            # ENV MODE: mount NO auth volume at all.
            #
            # This previously mounted the volume ``:ro``, reasoning that env
            # mode cannot refresh the token so a writable mount was
            # unnecessary.  That reasoning addressed the wrong risk.  A
            # read-only mount over the whole HOME still exposes every prior
            # run's transcripts, sessions and caches for READING, which is
            # precisely the cross-candidate channel -- read access is the
            # defect, not write access.
            #
            # Env mode therefore gets a private per-run HOME and no shared
            # store whatsoever.  The cross-run channel does not exist in this
            # mode rather than being masked, which is why its isolation claim
            # does not depend on any denylist being complete.
            args.extend(private_home_tmpfs_arg())
            args.extend(transcript_bind_arg(transcript_host_dir(workspace)))
        else:
            # VOLUME MODE IS RETIRED FOR AGENT EXECUTION IN 0.3.0, ALL
            # BACKENDS. Not "not recommended", not "not the default" --
            # UNSUPPORTED.
            #
            # EA ruling: "models_cache.json is persistent application-level
            # state in the deliberately shared, agent-visible auth store. One
            # run writes the remote catalogue; a later run reads
            # presence/version and CHANGES CONTROL FLOW by skipping/refetching
            # network work. That is causal cross-run influence under the
            # project's carrying definition EVEN IF catalogue contents are
            # account-wide."
            #
            # Per backend, precisely -- and codex's entry matters because it is
            # the one that was nearly certified:
            #   claude, gemini  the CLI keeps per-run state beside the
            #                   credential with no relocation knob;
            #   cursor          a config/data split is plausible but was never
            #                   verified, and plausible is not proven;
            #   opencode        its session database sits beside the credential
            #                   and the only knob moves both;
            #   codex           ITS AGENT MEMORY DATABASES ISOLATE CORRECTLY.
            #                   Measured across three clean runs under the full
            #                   production layout: nothing created or mutated in
            #                   the shared dir (shared=1, being only an
            #                   untouched stale seed), redirect=6.
            #                   CODEX_SQLITE_HOME WORKS. Codex fails on
            #                   models_cache.json ALONE -- which is materially
            #                   different from, and more honest than, "codex
            #                   leaks agent memory".
            #
            # And even a fully classified backend could not claim candidate
            # independence: the auth dir must stay writable for OAuth rotation,
            # so an agent can create an unenumerated file the next candidate
            # reads.
            #
            # `helix sandbox login` / `status` / `logout` are UNAFFECTED and
            # still use the volume -- that is what it is for.
            raise VolumeModeUnsupportedError(
                'sandbox.auth = "volume" is not supported for agent execution '
                "in HELIX 0.3.0.\n"
                f"  backend: {agent_backend}\n\n"
                "  The persistent auth store is shared ACROSS RUNS and every "
                "supported CLI keeps per-run state inside it that HELIX cannot "
                "relocate, so a later candidate can be causally influenced by "
                "an earlier one. HELIX refuses rather than report an isolated "
                "run that is not isolated.\n\n"
                '  Remedy: set sandbox.auth = "env" with a non-empty '
                "sandbox.auth_env_allow. Env mode mounts NO persistent store, "
                "so the cross-run channel does not exist rather than being "
                "masked.\n"
                "  Disclosed tradeoff: the named host credential is present "
                "inside the agent container, and OAuth refresh is suppressed.\n\n"
                "  `helix sandbox login` and `status` are unaffected."
            )

            # --- RETIRED BELOW; unreachable. Retained only until the layout
            # --- registry is removed, so the diff shows what was withdrawn.
            # VOLUME MODE: private per-run HOME, with ONLY the backend's auth
            # directory coming from the persistent store.
            #
            # The whole-HOME mount this replaces made the shared volume BE the
            # container HOME, so every candidate saw every prior candidate's
            # transcripts, sessions and caches.
            #
            # Fail closed first: a backend whose per-run state cannot be
            # relocated off the shared store must not run here at all, rather
            # than run and be reported as isolated.
            layout = layout_for(agent_backend)
            assert_layout_is_isolatable(layout)

            args.extend(private_home_tmpfs_arg())
            # Class 1: the auth directory only, via volume-subpath. Writable --
            # OAuth rotation renames a temp file over the credential inside its
            # own directory, so this cannot be :ro and cannot be a per-file
            # bind (EBUSY on rename-over and unlink).
            args.extend(
                [
                    "--mount",
                    f"type=volume,src={sandbox_auth_volume_name(agent_backend)},"
                    f"dst={layout.auth_dir},"
                    f"volume-subpath={layout.volume_subpath}",
                ]
            )
            # Class 2: per-run overlays on the directories inside the auth dir
            # that carry per-run state. Transcripts get the candidate-keyed
            # host bind instead, so capture keeps working.
            for subdir in layout.ephemeral_subdirs:
                target = f"{layout.auth_dir}/{subdir}"
                if target == CONTAINER_TRANSCRIPT_PARENT:
                    continue
                args.extend(
                    [
                        "--tmpfs",
                        f"{target}:rw,uid={NODE_UID},gid={NODE_GID},mode=0700",
                    ]
                )
            # Class 3: regular files inside the auth dir that carry per-run
            # state. NO mount can isolate these -- an overlay works on
            # directories only, and a per-file bind is EBUSY on rename-over --
            # so they are relocated wholesale by the backend's own env knob.
            #
            # Each redirect target also gets a tmpfs below, which guarantees the
            # directory EXISTS (a missing target can silently send the files
            # back to the shared directory) and makes it per-run.
            class3_env = dict(layout.env_redirects)
            for target in sorted(set(class3_env.values())):
                args.extend(
                    [
                        "--tmpfs",
                        f"{target}:rw,uid={NODE_UID},gid={NODE_GID},mode=0700",
                    ]
                )
            args.extend(transcript_bind_arg(transcript_host_dir(workspace)))

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
    # Class-3 relocation knobs. Set here, alongside HOME and PATH, because they
    # are HELIX-internal non-credential RUNTIME settings rather than host
    # environment values -- the same category as the two above, and subject to
    # the same rule: they are HELIX's to set and never carry user data.
    #
    # Load-bearing: without these the registry would DECLARE a backend
    # isolatable while the argv left its per-run files in the shared auth
    # directory. That is the exact shape of a control that reports success
    # while the property is false, so the argv is asserted to carry them.
    container_env.update(class3_env)
    container_env["PATH"] = (
        "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )

    for key, value in container_env.items():
        args.extend(["-e", f"{key}={value}"])

    args.append(image)
    args.extend(command)

    # Independent RE-CHECK of the assembled argv, mirroring
    # ``_assert_env_is_granted`` above. Both exist because a later-added call
    # site is how the original defects were introduced, and a construction-time
    # convention cannot catch that -- only a check over the FINAL artifact can.
    if scope == "agent":
        auth_dir = (
            layout_for(agent_backend).auth_dir
            if agent_backend and sandbox.resolved_auth() == "volume"
            else None
        )
        _assert_no_shared_home_mount(args, allowed_auth_dir=auth_dir)
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
    grants: list[EnvGrant] | None = None,
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
        _docker_chown_workspace(workspace, docker_image, "node:node")
        sidecar_runtime = (
            current_evaluator_sidecar_runtime() if scope == "evaluator" else None
        )
        command_env = dict(env)
        command_grants = list(grants) if grants is not None else None
        if sidecar_runtime is not None:
            command_env["HELIX_EVALUATOR_ENDPOINT"] = sidecar_runtime.endpoint
            if command_grants is not None:
                command_grants.append(
                    EnvGrant(
                        name="HELIX_EVALUATOR_ENDPOINT",
                        value=sidecar_runtime.endpoint,
                        origin="helix_internal",
                        scopes=frozenset({"agent", "evaluator", "sidecar"}),
                    )
                )
        # Create the candidate transcript directory before any container
        # starts.  Docker auto-creates a missing bind source as ``root:root``,
        # which would hand the agent a transcript directory its own uid cannot
        # write -- silently, and only for transcripts.
        if scope == "agent":
            ensure_transcript_host_dir(transcript_host_dir(workspace))

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
                    sidecar_runtime.network if sidecar_runtime is not None else None,
                    container_name=container_name,
                    grants=command_grants,
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
                if scope == "agent":
                    # Capture reads the candidate-keyed HOST BIND, so it runs
                    # no container and cannot re-mount the auth volume.  The
                    # outcome is typed and a genuine failure RAISES rather
                    # than being swallowed by ``check=False``.
                    transcript_outcome = capture_claude_transcript(
                        workspace=workspace,
                        artifact_dir=sandbox.transcript_artifact_dir,
                        session_id=_extract_session_id_from_json_output(
                            results[-1].stdout or ""
                        ),
                        enabled=sandbox.preserve_backend_transcripts,
                        backend=agent_backend or "",
                    )
                    logger.debug(
                        "transcript capture: %s (%s)",
                        transcript_outcome.status,
                        transcript_outcome.detail or transcript_outcome.artifact,
                    )
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
    grants: list[EnvGrant] | None = None,
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
        grants=grants,
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
    image: str,
    network: str = "bridge",
    add_host_gateway: bool = False,
    extra_hosts: dict[str, str] | None = None,
    interactive: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a backend auth command against the auth volume.

    ``image`` is **required** and must already be resolved by
    :func:`resolve_auth_runtime_image`.  It used to be optional, defaulting via
    a freshly constructed ``SandboxConfig(enabled=True)`` that could never
    carry the project's pinned image — the structural cause of
    producer/consumer CLI skew.  Making it required means the fallback cannot
    be reintroduced by accident.
    """
    args = sandbox_auth_docker_args(
        agent_backend,
        image=image,
        action=action,
        network=network,
        add_host_gateway=add_host_gateway,
        extra_hosts=extra_hosts,
        interactive=interactive,
    )
    if interactive:
        return _run_docker_process(args, capture_output=False)
    return _run_docker_process(args)


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
    grants: list[EnvGrant] | None = None,
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
            grants=grants,
        )
    return subprocess.run(
        command,
        shell=False,
        cwd=cwd,
        capture_output=True,
        text=True,
        env=env,
    )
