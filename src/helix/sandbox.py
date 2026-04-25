"""Subprocess sandboxing for HELIX agent and evaluator commands."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import stat
from pathlib import Path
from typing import Literal

from helix.backends import BACKEND_AUTH_COMMANDS, DEFAULT_BACKEND_IMAGES
from helix.config import SandboxConfig


HELIX_ARTIFACT_NAMES = {
    ".helix_mutation_prompt.md",
    ".helix_backend_result.json",
    ".helix_backend_stdout.txt",
    ".helix_backend_stderr.txt",
    "helix_batch.json",
}


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


def _ignore_for_copy(path: Path) -> bool:
    parts = path.parts
    return ".git" in parts or path.name == ".env" or path.name.startswith(".env.")


def _ignore_for_sync(path: Path) -> bool:
    parts = path.parts
    return (
        ".git" in parts
        or path.name in HELIX_ARTIFACT_NAMES
        or path.name == ".env"
        or path.name.startswith(".env.")
    )


def _copy_tree_contents(
    src: Path,
    dst: Path,
    *,
    for_sync: bool = False,
    skip_special_files: bool = True,
) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    ignore = _ignore_for_sync if for_sync else _ignore_for_copy
    for child in src.iterdir():
        if ignore(child.relative_to(src)):
            continue
        if skip_special_files and not _is_supported_workspace_file(child):
            continue
        target = dst / child.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        if child.is_symlink():
            os.symlink(os.readlink(child), target)
        elif child.is_dir():
            _copy_tree_contents(
                child,
                target,
                for_sync=for_sync,
                skip_special_files=skip_special_files,
            )
        else:
            shutil.copy2(child, target)


def _remove_extraneous_files(src: Path, dst: Path, *, skip_special_files: bool = True) -> None:
    for child in list(dst.iterdir()):
        rel = child.relative_to(dst)
        if _ignore_for_sync(rel):
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
            _remove_extraneous_files(src / rel, child, skip_special_files=skip_special_files)


def _sync_back_workspace(src: Path, dst: Path, *, skip_special_files: bool = True) -> None:
    _remove_extraneous_files(src, dst, skip_special_files=skip_special_files)
    _copy_tree_contents(src, dst, for_sync=True, skip_special_files=skip_special_files)


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
            "chown",
            "-R",
            "-h",
            owner,
            "/workspace",
        ],
        check=True,
        capture_output=True,
        text=True,
        env={k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ},
    )


def _host_owner() -> str | None:
    if not hasattr(os, "getuid") or not hasattr(os, "getgid"):
        return None
    return f"{os.getuid()}:{os.getgid()}"


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
        sandbox.network,
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
    if sandbox.add_host_gateway:
        args.extend(["--add-host", "host.docker.internal:host-gateway"])

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
) -> list[subprocess.CompletedProcess[str]]:
    """Run commands in one Docker sandbox workspace copy."""
    if not commands:
        raise ValueError("at least one sandbox command is required")
    source = Path(cwd).resolve()
    docker_image = image or sandbox.image
    if docker_image is None:
        raise ValueError("sandbox image must be provided")
    with tempfile.TemporaryDirectory(prefix="helix-sandbox-") as tmp:
        workspace = Path(tmp) / "workspace"
        _copy_tree_contents(source, workspace, skip_special_files=sandbox.skip_special_files)
        _init_synthetic_git_repo(workspace)
        _docker_chown_workspace(workspace, docker_image, "node:node")
        results = []
        try:
            for command in commands:
                docker_cmd = _docker_args(
                    command,
                    env,
                    workspace,
                    sandbox,
                    scope,
                    docker_image,
                    agent_backend,
                )
                results.append(
                    subprocess.run(
                        docker_cmd,
                        cwd=str(workspace),
                        capture_output=True,
                        text=True,
                        env={k: os.environ[k] for k in ("PATH", "HOME") if k in os.environ},
                        timeout=sandbox.timeout_seconds,
                    )
                )
        finally:
            if host_owner := _host_owner():
                _docker_chown_workspace(workspace, docker_image, host_owner)
        if sync_back:
            _sync_back_workspace(workspace, source, skip_special_files=sandbox.skip_special_files)
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
    )[0]


def sandbox_auth_docker_args(
    agent_backend: str,
    *,
    image: str,
    action: Literal["login", "status", "logout"],
    network: str = "bridge",
    add_host_gateway: bool = False,
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
        "-v",
        f"{sandbox_auth_volume_name(agent_backend)}:/home/node:rw",
        "-e",
        "HOME=/home/node",
        "-e",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    ]
    if interactive:
        args.insert(2, "-it")
    if add_host_gateway:
        args.extend(["--add-host", "host.docker.internal:host-gateway"])
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
    interactive: bool = False,
) -> subprocess.CompletedProcess[str]:
    docker_image = image or resolve_sandbox_image(SandboxConfig(enabled=True), agent_backend)
    args = sandbox_auth_docker_args(
        agent_backend,
        image=docker_image,
        action=action,
        network=network,
        add_host_gateway=add_host_gateway,
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
