from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helix.config import SandboxConfig
from helix.sandbox import (
    resolve_sandbox_image,
    run_sandboxed_command,
    run_sandboxed_commands,
    sandbox_auth_docker_args,
    sandbox_auth_volume_name,
)


def _is_workspace_chown(args: list[str]) -> bool:
    return args[:2] == ["docker", "run"] and "chown" in args and args[-1] == "/workspace"


def test_resolve_sandbox_image_defaults_from_backend():
    cfg = SandboxConfig(enabled=True)
    assert resolve_sandbox_image(cfg, "claude") == "ghcr.io/ke7/helix-evo-runner-claude:latest"
    assert resolve_sandbox_image(cfg, "codex") == "ghcr.io/ke7/helix-evo-runner-codex:latest"
    assert resolve_sandbox_image(cfg, "cursor") == "ghcr.io/ke7/helix-evo-runner-cursor:latest"
    assert resolve_sandbox_image(cfg, "gemini") == "ghcr.io/ke7/helix-evo-runner-gemini:latest"
    assert resolve_sandbox_image(cfg, "opencode") == "ghcr.io/ke7/helix-evo-runner-opencode:latest"


def test_resolve_sandbox_image_honors_override():
    cfg = SandboxConfig(enabled=True, image="custom:latest")
    assert resolve_sandbox_image(cfg, "claude") == "custom:latest"


def test_docker_command_mounts_only_workspace_and_auth_volume(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    cfg = SandboxConfig(
        enabled=True,
        image="helix-test:latest",
        network="none",
        cpus=1.5,
        memory="2g",
        timeout_seconds=10,
        add_host_gateway=True,
    )

    run_sandboxed_command(
        ["codex", "exec", "prompt"],
        cwd=source,
        env={"HELIX_DEBUG": "1"},
        sandbox=cfg,
        scope="agent",
        sync_back=True,
        agent_backend="codex",
    )

    docker_call = next(call for call in calls if call[:2] == ["docker", "run"] and "--user" in call and call[call.index("--user") + 1] == "node")
    joined = " ".join(docker_call)
    assert "--network none" in joined
    assert "--user node" in joined
    assert "--cpus 1.5" in joined
    assert "--memory 2g" in joined
    assert "--add-host host.docker.internal:host-gateway" in joined
    assert "helix-test:latest" in docker_call
    assert "-e" in docker_call
    assert "HOME=/home/node" in docker_call
    assert "helix-auth-codex:/home/node:rw" in docker_call
    assert f"{tmp_path}:" not in joined
    assert "/workspace:rw" in joined
    chown_calls = [call for call in calls if _is_workspace_chown(call)]
    assert len(chown_calls) == 2
    assert "node:node" in chown_calls[0]


def test_evaluator_scope_does_not_mount_agent_auth(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")

    mock_run = mocker.patch(
        "helix.sandbox.subprocess.run",
        return_value=MagicMock(stdout="", stderr="", returncode=0),
    )

    cfg = SandboxConfig(enabled=True)
    run_sandboxed_command(
        ["python", "evaluate.py"],
        cwd=source,
        env={},
        sandbox=cfg,
        scope="evaluator",
        sync_back=False,
        image="helix-test:latest",
    )

    docker_call = next(call.args[0] for call in mock_run.call_args_list if call.args[0][:2] == ["docker", "run"])
    assert "helix-auth-codex:/home/node:rw" not in docker_call


def test_agent_syncs_changes_back_but_excludes_git_and_artifacts(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "keep.py").write_text("old\n")
    (source / "delete.py").write_text("bye\n")
    (source / ".env").write_text("SECRET=value\n")

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert not (workspace / ".env").exists()
            (workspace / "keep.py").write_text("new\n")
            (workspace / "delete.py").unlink()
            (workspace / "added.py").write_text("added\n")
            (workspace / ".env.local").write_text("NEW_SECRET=value\n")
            (workspace / ".helix_backend_stdout.txt").write_text("artifact\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    cfg = SandboxConfig(enabled=True)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=cfg,
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "keep.py").read_text() == "new\n"
    assert not (source / "delete.py").exists()
    assert (source / "added.py").read_text() == "added\n"
    assert (source / ".env").read_text() == "SECRET=value\n"
    assert not (source / ".env.local").exists()
    assert not (source / ".git").exists()
    assert not (source / ".helix_backend_stdout.txt").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is unavailable on this platform")
def test_agent_sync_skips_special_files_by_default(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            os.mkfifo(workspace / "agent.pipe")
            (workspace / "regular.txt").write_text("ok\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "regular.txt").read_text() == "ok\n"
    assert not (source / "agent.pipe").exists()


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is unavailable on this platform")
def test_sync_preserves_existing_host_special_files_when_skipped(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    os.mkfifo(source / "existing.pipe")

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert not (workspace / "existing.pipe").exists()
            (workspace / "regular.txt").write_text("ok\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "existing.pipe").exists()
    assert not (source / "existing.pipe").is_file()
    assert (source / "regular.txt").read_text() == "ok\n"


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="mkfifo is unavailable on this platform")
def test_special_file_skip_can_be_disabled(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            os.mkfifo(workspace / "agent.pipe")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    with pytest.raises(OSError):
        run_sandboxed_command(
            ["claude", "-p", "prompt"],
            cwd=source,
            env={},
            sandbox=SandboxConfig(enabled=True, skip_special_files=False),
            scope="agent",
            sync_back=True,
            image="helix-test:latest",
            agent_backend="claude",
        )


def test_evaluator_does_not_sync_changes_back(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")
    (source / "helix_batch.json").write_text('["0"]\n')

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert (workspace / "helix_batch.json").read_text() == '["0"]\n'
            (workspace / "main.py").write_text("mutated\n")
            (workspace / "helix_batch.json").write_text('["changed"]\n')
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    cfg = SandboxConfig(enabled=True)

    run_sandboxed_command(
        ["python", "evaluate.py"],
        cwd=source,
        env={},
        sandbox=cfg,
        scope="evaluator",
        sync_back=False,
        image="helix-test:latest",
    )

    assert (source / "main.py").read_text() == "old\n"
    assert (source / "helix_batch.json").read_text() == '["0"]\n'


def test_sandboxed_command_sequence_reuses_workspace(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    seen_workspaces: list[Path] = []

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            seen_workspaces.append(workspace)
            if args[-1] == "write":
                (workspace / "result.txt").write_text("ok\n")
            elif args[-1] == "read":
                assert (workspace / "result.txt").read_text() == "ok\n"
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_commands(
        [["sh", "-c", "write"], ["sh", "-c", "read"]],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="evaluator",
        sync_back=False,
        image="helix-test:latest",
    )

    assert len(seen_workspaces) == 2
    assert seen_workspaces[0] == seen_workspaces[1]
    assert not (source / "result.txt").exists()


def test_sandbox_auth_volume_name_is_backend_specific():
    assert sandbox_auth_volume_name("claude") == "helix-auth-claude"
    assert sandbox_auth_volume_name("cursor") == "helix-auth-cursor"


def test_sandbox_auth_login_command_uses_persistent_volume():
    args = sandbox_auth_docker_args(
        "cursor",
        image="helix-cursor:latest",
        action="login",
        interactive=True,
    )

    assert args[:3] == ["docker", "run", "-it"]
    assert "helix-auth-cursor:/home/node:rw" in args
    assert "helix-cursor:latest" in args
    assert args[-2:] == ["cursor-agent", "login"]


def test_sandbox_auth_status_command_uses_backend_command():
    args = sandbox_auth_docker_args(
        "claude",
        image="helix-claude:latest",
        action="status",
    )

    assert "helix-auth-claude:/home/node:rw" in args
    assert args[-4:] == ["claude", "auth", "status", "--text"]


def test_sandbox_auth_claude_login_uses_setup_token_flow():
    args = sandbox_auth_docker_args(
        "claude",
        image="helix-claude:latest",
        action="login",
        interactive=True,
    )

    assert "helix-auth-claude:/home/node:rw" in args
    assert args[-2:] == ["claude", "setup-token"]


def test_sandbox_auth_codex_login_uses_device_auth_flow():
    args = sandbox_auth_docker_args(
        "codex",
        image="helix-codex:latest",
        action="login",
        interactive=True,
    )

    assert "helix-auth-codex:/home/node:rw" in args
    assert args[-3:] == ["codex", "login", "--device-auth"]


def test_sandbox_auth_gemini_login_skips_workspace_trust_prompt():
    args = sandbox_auth_docker_args(
        "gemini",
        image="helix-gemini:latest",
        action="login",
        interactive=True,
    )

    assert "helix-auth-gemini:/home/node:rw" in args
    assert args[-2:] == ["gemini", "--skip-trust"]


def test_sandbox_auth_opencode_login_uses_full_setup_tui():
    args = sandbox_auth_docker_args(
        "opencode",
        image="helix-opencode:latest",
        action="login",
        interactive=True,
    )

    assert "helix-auth-opencode:/home/node:rw" in args
    assert args[-1:] == ["opencode"]
