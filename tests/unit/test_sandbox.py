from __future__ import annotations

import os
import subprocess
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helix.config import EvaluatorSidecarConfig, SandboxConfig
from helix.sandbox import (
    EvaluatorSidecarRuntime,
    _healthcheck_docker_args,
    current_evaluator_sidecar_runtime,
    evaluator_sidecar_runtime,
    resolve_sandbox_image,
    run_sandboxed_command,
    run_sandboxed_commands,
    sandbox_auth_docker_args,
    sandbox_auth_volume_name,
    start_evaluator_sidecar,
)


def _is_workspace_chown(args: list[str]) -> bool:
    return (
        args[:2] == ["docker", "run"]
        and any("find /workspace -path /workspace/.git -prune" in item for item in args)
    )


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
        extra_hosts={"env-endpoint": "host-gateway", "local-service": "127.0.0.1"},
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
    assert "--add-host" in joined
    assert "--add-host host.docker.internal:host-gateway" in joined
    assert "--add-host env-endpoint:host-gateway" in joined
    assert "--add-host local-service:127.0.0.1" in joined
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


def test_sidecar_runtime_switches_evaluator_to_private_network(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")

    seen_calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        seen_calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    runtime = EvaluatorSidecarRuntime(
        network="helix-eval-private",
        container_name="helix-evaluator-test",
        endpoint="http://helix-evaluator:8080/evaluate",
    )
    with evaluator_sidecar_runtime(runtime):
        run_sandboxed_command(
            ["python", "/runner/evaluate.py"],
            cwd=source,
            env={},
            sandbox=SandboxConfig(enabled=True),
            scope="evaluator",
            sync_back=False,
            image="helix-test:latest",
        )

    docker_call = next(
        call
        for call in seen_calls
        if call[:2] == ["docker", "run"] and not _is_workspace_chown(call)
    )
    assert docker_call[docker_call.index("--network") + 1] == "helix-eval-private"
    assert "HELIX_EVALUATOR_ENDPOINT=http://helix-evaluator:8080/evaluate" in docker_call
    assert "helix-auth-codex:/home/node:rw" not in docker_call


def test_sidecar_runtime_is_visible_to_worker_threads():
    runtime = EvaluatorSidecarRuntime(
        network="helix-eval-private",
        container_name="helix-evaluator-test",
        endpoint="http://helix-evaluator:8080/evaluate",
    )
    seen: list[EvaluatorSidecarRuntime | None] = []

    with evaluator_sidecar_runtime(runtime):
        thread = threading.Thread(target=lambda: seen.append(current_evaluator_sidecar_runtime()))
        thread.start()
        thread.join()

    assert seen == [runtime]


def test_sidecar_healthcheck_uses_runner_image_and_endpoint():
    sidecar = EvaluatorSidecarConfig(
        image="eval-service:latest",
        runner_image="eval-runner:latest",
        command="python -m server",
        endpoint="http://helix-evaluator:8080/evaluate",
        healthcheck_command="python /runner/healthcheck.py",
    )

    args = _healthcheck_docker_args(sidecar, network="helix-eval-private")

    assert args[args.index("--network") + 1] == "helix-eval-private"
    assert "HELIX_EVALUATOR_ENDPOINT=http://helix-evaluator:8080/evaluate" in args
    assert "eval-runner:latest" in args
    assert args[-2:] == ["python", "/runner/healthcheck.py"]


def test_start_evaluator_sidecar_injects_fixed_env(mocker):
    calls: list[list[str]] = []

    def fake_run_docker(args, *, check=True):
        calls.append(args)
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(args, 0, stdout="true running\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox._run_docker", side_effect=fake_run_docker)

    sidecar = EvaluatorSidecarConfig(
        image="eval-service:latest",
        command="python -m server",
        endpoint="http://helix-evaluator:8080/evaluate",
    )

    with start_evaluator_sidecar(
        sidecar,
        fixed_env={"EVALUATOR_BASE_URL": "http://qwen-vllm-endpoint:8003"},
    ):
        pass

    docker_run = next(call for call in calls if call[:3] == ["docker", "run", "-d"])
    assert "EVALUATOR_BASE_URL=http://qwen-vllm-endpoint:8003" in docker_run


def test_agent_syncs_changes_back_but_excludes_git_and_artifacts(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "keep.py").write_text("old\n")
    (source / "delete.py").write_text("bye\n")
    (source / ".env").write_text("SECRET=value\n")
    (source / ".helix").mkdir()
    (source / ".helix" / "state.json").write_text("{}\n")
    (source / ".helix_artifacts").mkdir()
    (source / ".helix_artifacts" / "old.txt").write_text("old artifact\n")
    (source / "helix.toml").write_text("[evaluator.sidecar]\nendpoint = 'private'\n")

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert not (workspace / ".env").exists()
            assert not (workspace / ".helix").exists()
            assert not (workspace / ".helix_artifacts").exists()
            assert not (workspace / "helix.toml").exists()
            (workspace / "keep.py").write_text("new\n")
            (workspace / "delete.py").unlink()
            (workspace / "added.py").write_text("added\n")
            (workspace / "helix.toml").write_text("tampered\n")
            (workspace / ".env.local").write_text("NEW_SECRET=value\n")
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text("tampered\n")
            (workspace / ".helix_artifacts").mkdir()
            (workspace / ".helix_artifacts" / "new.txt").write_text("new artifact\n")
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
    assert (source / "helix.toml").read_text() == "[evaluator.sidecar]\nendpoint = 'private'\n"
    assert (source / ".helix" / "state.json").read_text() == "{}\n"
    assert (source / ".helix_artifacts" / "old.txt").read_text() == "old artifact\n"
    assert not (source / ".helix_artifacts" / "new.txt").exists()
    assert not (source / ".helix_artifacts" / "backend_transcripts").exists()
    assert not (source / ".env.local").exists()
    assert not (source / ".git").exists()
    assert not (source / ".helix_backend_stdout.txt").exists()


def test_agent_copies_claude_transcript_from_auth_volume(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            if args[-3:] and args[-3] == "sh" and "sess_123.jsonl" in args[-1]:
                transcript = (
                    workspace
                    / ".helix_artifacts"
                    / "backend_transcripts"
                    / "claude"
                    / "sess_123.jsonl"
                )
                transcript.parent.mkdir(parents=True)
                transcript.write_text('{"message":"saved"}\n')
            else:
                (workspace / "main.py").write_text("new\n")
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"type":"result","session_id":"sess_123"}\n',
            stderr="",
        )

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

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

    assert (source / "main.py").read_text() == "new\n"
    transcript = source / ".helix_artifacts" / "backend_transcripts" / "claude" / "sess_123.jsonl"
    assert transcript.read_text() == '{"message":"saved"}\n'
    copy_call = next(call for call in calls if call[:2] == ["docker", "run"] and "sess_123.jsonl" in " ".join(call))
    assert "helix-auth-claude:/home/node:ro" in copy_call


def test_agent_sync_back_honors_omitted_paths(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")
    (source / "private").mkdir()
    (source / "private" / "token.txt").write_text("host secret\n")

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert not (workspace / "private" / "token.txt").exists()
            (workspace / "main.py").write_text("new\n")
            (workspace / "private").mkdir(exist_ok=True)
            (workspace / "private" / "token.txt").write_text("agent secret\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True, omit_from_agent=["private/token.txt"]),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert (source / "main.py").read_text() == "new\n"
    assert (source / "private" / "token.txt").read_text() == "host secret\n"


def test_agent_sync_back_does_not_create_omitted_paths(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"] and not _is_workspace_chown(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace / "private").mkdir()
            (workspace / "private" / "token.txt").write_text("agent secret\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    run_sandboxed_command(
        ["claude", "-p", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True, omit_from_agent=["private"]),
        scope="agent",
        sync_back=True,
        image="helix-test:latest",
        agent_backend="claude",
    )

    assert not (source / "private").exists()


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
        extra_hosts={"local-service": "127.0.0.1"},
        interactive=True,
    )

    assert args[:3] == ["docker", "run", "-it"]
    assert "helix-auth-cursor:/home/node:rw" in args
    assert "helix-cursor:latest" in args
    assert args[-2:] == ["cursor-agent", "login"]
    assert "--add-host local-service:127.0.0.1" in " ".join(args)


def test_sandbox_auth_status_command_uses_backend_command():
    args = sandbox_auth_docker_args(
        "claude",
        image="helix-claude:latest",
        action="status",
    )

    assert "helix-auth-claude:/home/node:rw" in args
    assert args[-4:] == ["claude", "auth", "status", "--text"]


def test_sandbox_auth_claude_login_uses_claudeai_flow():
    args = sandbox_auth_docker_args(
        "claude",
        image="helix-claude:latest",
        action="login",
        interactive=True,
    )

    assert "helix-auth-claude:/home/node:rw" in args
    assert args[-4:] == ["claude", "auth", "login", "--claudeai"]


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
