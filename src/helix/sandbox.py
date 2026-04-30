"""Subprocess sandboxing for HELIX agent and evaluator commands."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import os
import shlex
import shutil
import subprocess
import stat
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from helix.backends import BACKEND_AUTH_COMMANDS, DEFAULT_BACKEND_IMAGES
from helix.config import EvaluatorSidecarConfig, SandboxConfig


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


_EVALUATOR_SIDECAR_RUNTIME: EvaluatorSidecarRuntime | None = None
_EVALUATOR_SIDECAR_RUNTIME_LOCK = threading.RLock()


def current_evaluator_sidecar_runtime() -> EvaluatorSidecarRuntime | None:
    with _EVALUATOR_SIDECAR_RUNTIME_LOCK:
        return _EVALUATOR_SIDECAR_RUNTIME


@contextmanager
def evaluator_sidecar_runtime(
    runtime: EvaluatorSidecarRuntime,
) -> Iterator[EvaluatorSidecarRuntime]:
    global _EVALUATOR_SIDECAR_RUNTIME
    with _EVALUATOR_SIDECAR_RUNTIME_LOCK:
        previous = _EVALUATOR_SIDECAR_RUNTIME
        _EVALUATOR_SIDECAR_RUNTIME = runtime
    try:
        yield runtime
    finally:
        with _EVALUATOR_SIDECAR_RUNTIME_LOCK:
            _EVALUATOR_SIDECAR_RUNTIME = previous


def _is_supported_workspace_file(path: Path) -> bool:
    try:
        mode = path.lstat().st_mode
    except OSError:
        return False
    return stat.S_ISREG(mode) or stat.S_ISDIR(mode) or stat.S_ISLNK(mode)


def resolve_sandbox_image(sandbox: SandboxConfig, agent_backend: str | None = None) -> str:
    if sandbox.image:
        return sandbox.image
    if agent_backend is None:
        raise ValueError("agent_backend is required when sandbox.image is not set")
    try:
        return DEFAULT_BACKEND_IMAGES[agent_backend]
    except KeyError as exc:
        raise ValueError(f"No default sandbox image for backend: {agent_backend}") from exc


def _is_helix_artifact_name(name: str) -> bool:
    return (
        name in HELIX_ARTIFACT_NAMES
        or name == ".agent_task_prompt.md"
    )


def _ignore_for_copy(path: Path) -> bool:
    parts = path.parts
    return (
        ".git" in parts
        or path.name == "helix.toml"
        or path.name == ".env"
        or path.name.startswith(".env.")
    )


def _ignore_for_sync(path: Path) -> bool:
    parts = path.parts
    return (
        ".git" in parts
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
            if not isinstance(node, dict):
                continue
            for key in ("session_id", "sessionId", "sessionID"):
                value = node.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _copy_claude_transcript_from_auth_volume(
    *,
    workspace: Path,
    image: str,
    agent_backend: str | None,
    sandbox: SandboxConfig,
    stdout: str,
) -> None:
    if (
        agent_backend != "claude"
        or not sandbox.preserve_backend_transcripts
    ):
        return
    session_id = _extract_session_id_from_json_output(stdout)
    if not session_id:
        return
    rel_dir = Path(sandbox.transcript_artifact_dir) / "claude"
    rel_file = rel_dir / f"{session_id}.jsonl"
    source = Path(sandbox.claude_transcript_root) / f"{session_id}.jsonl"
    command = (
        "set -eu; "
        f"src={shlex.quote(str(source))}; "
        f"dst={shlex.quote('/workspace/' + str(rel_file))}; "
        '[ -f "$src" ] || exit 0; '
        'mkdir -p "$(dirname "$dst")"; '
        'cp "$src" "$dst"'
    )
    args = [
        "docker",
        "run",
        "--rm",
        "--workdir",
        "/workspace",
        "--user",
        "node",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{workspace}:/workspace:rw",
        "-v",
        f"{sandbox_auth_volume_name(agent_backend)}:/home/node:ro",
        "-e",
        "HOME=/home/node",
        image,
        "sh",
        "-c",
        command,
    ]
    _run_docker(args, check=False)


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
    for child in src.iterdir():
        rel = child.relative_to(src)
        if ignore(rel) or _matches_omitted_path(rel, omitted):
            continue
        if skip_special_files and not _is_supported_workspace_file(child):
            continue
        target = dst / child.name
        if child.is_symlink():
            if target.exists() or target.is_symlink():
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            os.symlink(os.readlink(child), target)
        elif child.is_dir():
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
            shutil.copy2(child, target)


def _remove_extraneous_files(
    src: Path,
    dst: Path,
    *,
    skip_special_files: bool = True,
    omit_paths: set[Path] | None = None,
) -> None:
    omitted = omit_paths or set()
    for child in list(dst.iterdir()):
        rel = child.relative_to(dst)
        if _ignore_for_sync(rel) or _matches_omitted_path(rel, omitted):
            continue
        if skip_special_files and not _is_supported_workspace_file(child):
            continue
        if not (src / rel).exists() and not (src / rel).is_symlink():
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


def _init_synthetic_git_repo(workspace: Path) -> None:
    """Create local-only git metadata so agent CLIs can inspect status."""
    subprocess.run(["git", "init"], cwd=workspace, check=False, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "HELIX Sandbox"],
        cwd=workspace,
        check=False,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "helix-sandbox@noreply"],
        cwd=workspace,
        check=False,
        capture_output=True,
    )
    subprocess.run(["git", "add", "-A"], cwd=workspace, check=False, capture_output=True)
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "helix: sandbox baseline"],
        cwd=workspace,
        check=False,
        capture_output=True,
    )


def _docker_chown_workspace(workspace: Path, image: str, owner: str) -> None:
    subprocess.run(
        [
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
            'find /workspace -path /workspace/.git -prune -o -exec chown -h "$0" {} +',
            owner,
        ],
        check=True,
        capture_output=True,
        text=True,
        env={k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ},
    )


def _host_owner() -> str | None:
    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        return None
    try:
        info = subprocess.run(
            ["docker", "info", "--format", "{{json .SecurityOptions}}"],
            check=False,
            capture_output=True,
            text=True,
            env={k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ},
        )
        if info.returncode == 0 and "rootless" in (info.stdout or ""):
            # In rootless Docker, container root maps back to the host user.
            # Chowning to the numeric host uid from inside the namespace maps
            # to an unmapped subuid and leaves the host unable to clean up.
            return "root:root"
    except Exception:
        pass
    return f"{os.getuid()}:{os.getgid()}"


def _docker_host_env() -> dict[str, str]:
    return {k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ}


def _run_docker(
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        capture_output=True,
        text=True,
        env=_docker_host_env(),
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
            ["docker", "inspect", "-f", "{{.State.Running}} {{.State.Status}}", container_name],
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
    _run_docker(["docker", "network", "create", "--internal", network])
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
        args.extend(_build_add_host_args(add_host_gateway=False, extra_hosts=extra_hosts))
        for key in passthrough_env or []:
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


def _docker_args(
    command: list[str],
    env: dict[str, str],
    workspace: Path,
    sandbox: SandboxConfig,
    scope: Literal["agent", "evaluator"],
    image: str,
    agent_backend: str | None,
    network: str | None = None,
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
    if scope == "agent":
        if agent_backend is None:
            raise ValueError("agent_backend is required for sandboxed agent commands")
        args.extend(["-v", f"{sandbox_auth_volume_name(agent_backend)}:/home/node:rw"])

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
        key: value
        for key, value in env.items()
        if key not in {"HOME", "PATH"}
    }
    container_env["HOME"] = "/home/node"
    container_env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

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
    with tempfile.TemporaryDirectory(prefix="helix-sandbox-") as tmp:
        workspace = Path(tmp) / "workspace"
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
        sidecar_runtime = current_evaluator_sidecar_runtime() if scope == "evaluator" else None
        command_env = dict(env)
        if sidecar_runtime is not None:
            command_env["HELIX_EVALUATOR_ENDPOINT"] = sidecar_runtime.endpoint
        results = []
        try:
            for command in commands:
                docker_cmd = _docker_args(
                    command,
                    command_env,
                    workspace,
                    sandbox,
                    scope,
                    docker_image,
                    agent_backend,
                    sidecar_runtime.network if sidecar_runtime is not None else None,
                )
                results.append(
                    subprocess.run(
                        docker_cmd,
                        cwd=str(source),
                        capture_output=True,
                        text=True,
                        input=input_text,
                        env={k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ},
                        timeout=sandbox.timeout_seconds,
                    )
                )
                if scope == "agent":
                    _copy_claude_transcript_from_auth_volume(
                        workspace=workspace,
                        image=docker_image,
                        agent_backend=agent_backend,
                        sandbox=sandbox,
                        stdout=results[-1].stdout or "",
                    )
        finally:
            if host_owner := _host_owner():
                _docker_chown_workspace(workspace, docker_image, host_owner)
        if sync_back:
            _sync_back_workspace(
                workspace,
                source,
                skip_special_files=sandbox.skip_special_files,
                omit_paths=omit_paths,
            )
    return results


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
        raise ValueError(f"No sandbox auth {action!r} command for backend: {agent_backend}") from exc

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
        *_build_add_host_args(add_host_gateway=add_host_gateway, extra_hosts=extra_hosts),
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
    docker_image = image or resolve_sandbox_image(SandboxConfig(enabled=True), agent_backend)
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
