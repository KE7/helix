from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import helix.sandbox as sandbox_module

from helix.config import EvaluatorSidecarConfig, SandboxConfig
from helix.sandbox import (
    _SEED_IMAGE,
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
    """True for either workspace-recovery helper (chown or permission-relax).

    Existing call sites use this predicate to filter out housekeeping
    containers from the primary agent ``docker run``; both helpers share the
    same ``find /workspace`` boilerplate. Use :func:`_is_workspace_chown_only`
    or :func:`_is_workspace_permission_relax` when you need to distinguish
    them.
    """
    return args[:2] == ["docker", "run"] and any(
        "find /workspace -path /workspace/.git -prune" in item for item in args
    )


def _is_workspace_chown_only(args: list[str]) -> bool:
    return args[:2] == ["docker", "run"] and any(
        "chown" in item and "find /workspace" in item for item in args
    )


def _is_workspace_permission_relax(args: list[str]) -> bool:
    return args[:2] == ["docker", "run"] and any(
        "chmod a+rwX" in item for item in args
    )


def _is_agent_container(args: list[str]) -> bool:
    return (
        args[:2] == ["docker", "run"]
        and "--user" in args
        and args[args.index("--user") + 1] == "node"
    )


def test_resolve_sandbox_image_defaults_from_backend():
    cfg = SandboxConfig(enabled=True)
    assert (
        resolve_sandbox_image(cfg, "claude")
        == "ghcr.io/ke7/helix-evo-runner-claude:latest"
    )
    assert (
        resolve_sandbox_image(cfg, "codex")
        == "ghcr.io/ke7/helix-evo-runner-codex:latest"
    )
    assert (
        resolve_sandbox_image(cfg, "cursor")
        == "ghcr.io/ke7/helix-evo-runner-cursor:latest"
    )
    assert (
        resolve_sandbox_image(cfg, "gemini")
        == "ghcr.io/ke7/helix-evo-runner-gemini:latest"
    )
    assert (
        resolve_sandbox_image(cfg, "opencode")
        == "ghcr.io/ke7/helix-evo-runner-opencode:latest"
    )


def test_resolve_sandbox_image_honors_override():
    cfg = SandboxConfig(enabled=True, image="custom:latest")
    assert resolve_sandbox_image(cfg, "claude") == "custom:latest"


def test_docker_command_mounts_private_home_and_candidate_auth_volume(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("print('hi')\n")

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value="1000:1000")

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

    docker_call = next(
        call
        for call in calls
        if call[:2] == ["docker", "run"]
        and "--user" in call
        and call[call.index("--user") + 1] == "node"
    )
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
    assert "--tmpfs" in docker_call
    assert "/home/node:rw,uid=1000,gid=1000,mode=700" in docker_call
    assert not any("helix-auth-codex" in item for item in docker_call)
    assert any(
        item.startswith("helix-candidate-auth-codex-")
        and item.endswith(":/home/node/.codex:rw")
        for item in docker_call
    )
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

    docker_call = next(
        call.args[0]
        for call in mock_run.call_args_list
        if call.args[0][:2] == ["docker", "run"]
    )
    assert "helix-auth-codex:/home/node:rw" not in docker_call


def test_volume_auth_seeds_a_private_allowlisted_volume_and_never_mounts_source(
    tmp_path: Path, mocker
):
    source = tmp_path / "candidate"
    source.mkdir()
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    run_sandboxed_command(
        ["codex", "exec", "prompt"],
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True),
        scope="agent",
        sync_back=False,
        image="helix-test:latest",
        agent_backend="codex",
    )

    seed = next(call for call in calls if _SEED_IMAGE in call)
    agent = next(call for call in calls if _is_agent_container(call))
    candidate_mount = next(
        item
        for item in agent
        if item.startswith("helix-candidate-auth-codex-")
    )
    assert "helix-auth-codex:/source:ro" in seed
    assert "cp /source/.codex/auth.json /destination/auth.json" in seed[-1]
    assert "cp -R" not in seed[-1]
    assert "helix-auth-codex" not in " ".join(agent)
    assert candidate_mount.endswith(":/home/node/.codex:rw")
    assert any(
        call[:3] == ["docker", "volume", "rm"]
        and call[-1] == candidate_mount.split(":", 1)[0]
        for call in calls
    )


def test_parallel_volume_candidates_use_distinct_private_auth_volumes(tmp_path: Path, mocker):
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    def run_one(index: int) -> None:
        source = tmp_path / f"candidate-{index}"
        source.mkdir()
        run_sandboxed_command(
            ["codex", "exec", "prompt"],
            cwd=source,
            env={},
            sandbox=SandboxConfig(enabled=True),
            scope="agent",
            sync_back=False,
            image="helix-test:latest",
            agent_backend="codex",
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run_one, range(2)))

    agent_mounts = {
        item.split(":", 1)[0]
        for call in calls
        if _is_agent_container(call)
        for item in call
        if item.startswith("helix-candidate-auth-codex-")
    }
    assert len(agent_mounts) == 2
    assert all("helix-auth-codex" not in " ".join(call) for call in calls if _is_agent_container(call))


def test_candidate_volume_cleanup_failure_is_loud_and_actionable(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if args[:3] == ["docker", "volume", "rm"]:
            return subprocess.CompletedProcess(args, 1, stdout="", stderr="busy")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    with pytest.raises(RuntimeError, match="credential cleanup failed") as exc:
        run_sandboxed_command(
            ["codex", "exec", "prompt"],
            cwd=source,
            env={},
            sandbox=SandboxConfig(enabled=True),
            scope="agent",
            sync_back=False,
            image="helix-test:latest",
            agent_backend="codex",
        )
    assert "docker volume rm helix-candidate-auth-codex-" in str(exc.value)


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
    assert (
        "HELIX_EVALUATOR_ENDPOINT=http://helix-evaluator:8080/evaluate" in docker_call
    )
    assert "helix-auth-codex:/home/node:rw" not in docker_call


def test_sidecar_runtime_is_visible_to_worker_threads():
    import concurrent.futures

    runtime = EvaluatorSidecarRuntime(
        network="helix-eval-private",
        container_name="helix-evaluator-test",
        endpoint="http://helix-evaluator:8080/evaluate",
    )
    seen: list[EvaluatorSidecarRuntime | None] = []

    with evaluator_sidecar_runtime(runtime):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                lambda: seen.append(current_evaluator_sidecar_runtime())
            )
            future.result()

    assert seen == [runtime]
    # Stack must be empty after the context manager exits.
    assert current_evaluator_sidecar_runtime() is None


def test_sidecar_runtime_nested_same_runtime_restores_outer():
    """Nested ``with`` blocks must restore the outer runtime, not blank it."""
    outer = EvaluatorSidecarRuntime(
        network="net-outer", container_name="c-outer", endpoint="http://outer/"
    )
    inner = EvaluatorSidecarRuntime(
        network="net-inner", container_name="c-inner", endpoint="http://inner/"
    )
    with evaluator_sidecar_runtime(outer):
        assert current_evaluator_sidecar_runtime() is outer
        with evaluator_sidecar_runtime(inner):
            assert current_evaluator_sidecar_runtime() is inner
        assert current_evaluator_sidecar_runtime() is outer
    assert current_evaluator_sidecar_runtime() is None


def test_sidecar_runtime_accepts_unhashable_helixconfig_like_object():
    """Regression: prior design used a WeakKeyDictionary keyed on HelixConfig,
    which is a Pydantic ``BaseModel`` (unhashable). The new lock+stack design
    must not key on the config at all, so even an unhashable sentinel works.
    """

    class Unhashable:
        __hash__ = None  # type: ignore[assignment]

    runtime = EvaluatorSidecarRuntime(
        network="net-x", container_name="c-x", endpoint="http://x/"
    )
    # Just exercising the API; unhashable config objects must not crash.
    Unhashable()  # constructed but never used as a key
    with evaluator_sidecar_runtime(runtime):
        assert current_evaluator_sidecar_runtime() is runtime
    assert current_evaluator_sidecar_runtime() is None


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
            return subprocess.CompletedProcess(
                args, 0, stdout="true running\n", stderr=""
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox._run_docker", side_effect=fake_run_docker)

    sidecar = EvaluatorSidecarConfig(
        image="eval-service:latest",
        command="python -m server",
        endpoint="http://helix-evaluator:8080/evaluate",
    )

    with start_evaluator_sidecar(
        sidecar,
        fixed_env={"EVALUATOR_BASE_URL": "https://model-service.example.invalid/v1"},
    ):
        pass

    docker_run = next(call for call in calls if call[:3] == ["docker", "run", "-d"])
    assert "EVALUATOR_BASE_URL=https://model-service.example.invalid/v1" in docker_run


def test_start_evaluator_sidecar_injects_its_own_passthrough_env(mocker, monkeypatch):
    calls: list[list[str]] = []

    def fake_run_docker(args, *, check=True):
        calls.append(args)
        if args[:2] == ["docker", "inspect"]:
            return subprocess.CompletedProcess(
                args, 0, stdout="true running\n", stderr=""
            )
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox._run_docker", side_effect=fake_run_docker)
    monkeypatch.setenv("OPENAI_API_KEY", "sidecar-only-test-key")
    sidecar = EvaluatorSidecarConfig(
        image="eval-service:latest",
        command="python -m server",
        endpoint="http://helix-evaluator:8080/evaluate",
        passthrough_env=["OPENAI_API_KEY"],
    )

    with start_evaluator_sidecar(sidecar):
        pass

    docker_run = next(call for call in calls if call[:3] == ["docker", "run", "-d"])
    assert "OPENAI_API_KEY=sidecar-only-test-key" in docker_run


def test_agent_syncs_changes_back_but_excludes_git_and_artifacts(
    tmp_path: Path, mocker
):
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
        if _is_agent_container(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            assert not (workspace / ".env").exists()
            assert not (workspace / ".helix").exists()
            # Claude's nested transcript bind creates this candidate-local
            # host directory before the agent starts.
            assert not (workspace / "helix.toml").exists()
            (workspace / "keep.py").write_text("new\n")
            (workspace / "delete.py").unlink()
            (workspace / "added.py").write_text("added\n")
            (workspace / "helix.toml").write_text("tampered\n")
            (workspace / ".env.local").write_text("NEW_SECRET=value\n")
            (workspace / ".helix").mkdir()
            (workspace / ".helix" / "state.json").write_text("tampered\n")
            (workspace / ".helix_artifacts").mkdir(exist_ok=True)
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
    assert (
        source / "helix.toml"
    ).read_text() == "[evaluator.sidecar]\nendpoint = 'private'\n"
    assert (source / ".helix" / "state.json").read_text() == "{}\n"
    assert (source / ".helix_artifacts" / "old.txt").read_text() == "old artifact\n"
    assert not (source / ".helix_artifacts" / "new.txt").exists()
    assert (source / ".helix_artifacts" / "backend_transcripts").exists()
    assert not (source / ".env.local").exists()
    assert not (source / ".git").exists()
    assert not (source / ".helix_backend_stdout.txt").exists()


def test_agent_sync_tolerates_inaccessible_workspace_paths(
    tmp_path: Path, mocker, monkeypatch
):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "keep.py").write_text("old\n")
    (source / ".gitignore").write_text("*.tmp\n")

    workspace_ready = False
    real_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if workspace_ready and path.name == ".gitignore" and "/workspace/" in str(path):
            raise PermissionError("permission denied")
        return real_exists(path)

    def fake_run(args, **kwargs):
        nonlocal workspace_ready
        if _is_agent_container(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace / "keep.py").write_text("new\n")
            workspace_ready = True
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    monkeypatch.setattr(Path, "exists", fake_exists)
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

    assert (source / "keep.py").read_text() == "new\n"
    assert (source / ".gitignore").read_text() == "*.tmp\n"


def test_agent_transcript_bind_does_not_remount_login_volume(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if _is_agent_container(args):
            workspace = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace / "main.py").write_text("new\n")
            transcript = (
                workspace
                / ".helix_artifacts"
                / "backend_transcripts"
                / "claude"
                / "sess_123.jsonl"
            )
            transcript.parent.mkdir(parents=True, exist_ok=True)
            transcript.write_text('{"message":"saved"}\n')
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
    transcript = (
        source
        / ".helix_artifacts"
        / "backend_transcripts"
        / "claude"
        / "sess_123.jsonl"
    )
    assert transcript.read_text() == '{"message":"saved"}\n'
    agent_calls = [call for call in calls if _is_agent_container(call)]
    assert len(agent_calls) == 1
    assert not any("helix-auth-claude" in item for call in agent_calls for item in call)


def test_agent_sync_tolerates_inaccessible_backend_transcripts(
    tmp_path: Path, mocker, monkeypatch
):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")

    workspace_root: Path | None = None
    real_exists = Path.exists

    def fake_exists(path: Path) -> bool:
        if (
            workspace_root is not None
            and path == workspace_root / ".helix_artifacts" / "backend_transcripts"
        ):
            raise PermissionError("permission denied")
        return real_exists(path)

    def fake_run(args, **kwargs):
        nonlocal workspace_root
        if _is_agent_container(args):
            workspace_root = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace_root / "main.py").write_text("new\n")
            blocked = workspace_root / ".helix_artifacts" / "backend_transcripts"
            blocked.mkdir(parents=True, exist_ok=True)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout='{"type":"result","session_id":"sess_123"}\n',
            stderr="",
        )

    monkeypatch.setattr(Path, "exists", fake_exists)
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
    assert not (source / ".helix_artifacts" / "backend_transcripts").exists()


def test_agent_sync_recovers_rootless_workspace_permissions(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")

    calls: list[list[str]] = []
    workspace_root: Path | None = None

    def fake_run(args, **kwargs):
        nonlocal workspace_root
        calls.append(args)
        if (
            _is_agent_container(args)
            and not _is_workspace_permission_relax(args)
        ):
            workspace_root = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace_root / "main.py").write_text("new\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

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

    assert workspace_root is not None
    assert (source / "main.py").read_text() == "new\n"
    relax_calls = [call for call in calls if _is_workspace_permission_relax(call)]
    assert relax_calls
    # The relax helper must run after the agent container exits and before
    # host-side sync-back reads the workspace; verify ordering relative to the
    # primary agent docker run.
    agent_run_idx = next(
        i
        for i, call in enumerate(calls)
        if call[:2] == ["docker", "run"]
        and not _is_workspace_chown(call)
        and not _is_workspace_permission_relax(call)
    )
    assert calls.index(relax_calls[0]) > agent_run_idx


def test_agent_sync_skips_relax_when_host_owner_available(tmp_path: Path, mocker):
    """When ``_host_owner`` returns a value, the chown path runs and the relax
    helper must NOT be invoked (the two recovery branches are mutually
    exclusive)."""
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")

    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if (
            _is_agent_container(args)
            and not _is_workspace_permission_relax(args)
        ):
            workspace_root = Path(args[args.index("-v") + 1].split(":", 1)[0])
            (workspace_root / "main.py").write_text("new\n")
        return subprocess.CompletedProcess(args, 0, stdout="{}", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
    mocker.patch("helix.sandbox._host_owner", return_value="1000:1000")

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
    assert not [call for call in calls if _is_workspace_permission_relax(call)]
    assert [call for call in calls if _is_workspace_chown_only(call)]


def test_safe_rmtree_uses_relax_helper_when_host_owner_missing(
    tmp_path: Path, mocker
):
    """``_safe_rmtree`` must invoke the relax helper (not chown) when the host
    owner is unavailable, then retry the rmtree."""
    from helix.sandbox import _safe_rmtree

    target = tmp_path / "doomed"
    target.mkdir()
    (target / "f").write_text("x")

    rmtree_calls: list[Path] = []
    original_rmtree = __import__("shutil").rmtree

    def fake_rmtree(path, *args, **kwargs):
        rmtree_calls.append(Path(path))
        if len(rmtree_calls) == 1:
            raise OSError("permission denied")
        return original_rmtree(path, *args, **kwargs)

    mocker.patch("helix.sandbox.shutil.rmtree", side_effect=fake_rmtree)
    mocker.patch("helix.sandbox._host_owner", return_value=None)

    docker_calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        docker_calls.append(args)
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)

    _safe_rmtree(target, docker_image="helix-test:latest")

    assert any(_is_workspace_permission_relax(call) for call in docker_calls)
    assert not any(_is_workspace_chown_only(call) for call in docker_calls)
    assert len(rmtree_calls) == 2  # initial failure + retry


def test_agent_sync_back_honors_omitted_paths(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()
    (source / "main.py").write_text("old\n")
    (source / "private").mkdir()
    (source / "private" / "token.txt").write_text("host secret\n")

    def fake_run(args, **kwargs):
        if _is_agent_container(args):
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
        if _is_agent_container(args):
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


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="mkfifo is unavailable on this platform"
)
def test_agent_sync_skips_special_files_by_default(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if _is_agent_container(args):
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


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="mkfifo is unavailable on this platform"
)
def test_sync_preserves_existing_host_special_files_when_skipped(
    tmp_path: Path, mocker
):
    source = tmp_path / "candidate"
    source.mkdir()
    os.mkfifo(source / "existing.pipe")

    def fake_run(args, **kwargs):
        if _is_agent_container(args):
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


@pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="mkfifo is unavailable on this platform"
)
def test_special_file_skip_can_be_disabled(tmp_path: Path, mocker):
    source = tmp_path / "candidate"
    source.mkdir()

    def fake_run(args, **kwargs):
        if _is_agent_container(args):
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
        if _is_agent_container(args):
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
        if _is_agent_container(args):
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
    assert args[-3:-1] == ["sh", "-lc"]
    script = args[-1]
    assert script.startswith("set -eu; ")
    assert "claude auth status --text" in script
    # Robust check: requires the on-disk credential file to exist and be
    # non-empty so the exit code is meaningful even when the CLI itself
    # exits 0 for an unauthenticated user.
    assert 'test -s "${HOME:-/home/node}/.claude/.credentials.json"' in script


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


# ---------------------------------------------------------------------------
# Tests: Docker env values never appear in rendered diagnostics
# ---------------------------------------------------------------------------


class TestDockerEnvRedaction:
    """`-e KEY=VALUE` can carry a sensitive value; failures must not echo it.

    `subprocess.CalledProcessError` keeps the full argv on both `.cmd` and
    `.args`, so an unredacted failure puts the key into logs, tracebacks and
    any crash reporter that renders the exception.
    """

    SENSITIVE_VALUE = "value-that-must-not-appear"

    def _args(self) -> list[str]:
        return [
            "docker", "run", "--rm",
            "-e", f"KEY={self.SENSITIVE_VALUE}",
            "-e", "HOME=/home/node",
            "helix-test-image:local", "true",
        ]

    def test_argv_redaction_keeps_keys_and_drops_values(self) -> None:
        redacted = sandbox_module._redact_docker_argv(self._args())

        assert self.SENSITIVE_VALUE not in " ".join(redacted)
        assert "KEY=<redacted>" in redacted, (
            "the key must survive so a rendered command stays diagnosable"
        )
        assert redacted[0:3] == ["docker", "run", "--rm"]
        assert redacted[-2:] == ["helix-test-image:local", "true"]

    @pytest.mark.parametrize(
        "form",
        [
            ["-e", "K=SEKRET"],
            ["--env", "K=SEKRET"],
            ["-eK=SEKRET"],
            ["--env=K=SEKRET"],
        ],
    )
    def test_every_docker_env_spelling_is_redacted(self, form: list[str]) -> None:
        redacted = sandbox_module._redact_docker_argv(["docker", "run", *form, "img"])
        assert "SEKRET" not in " ".join(redacted)
        assert "K=<redacted>" in " ".join(redacted)

    def test_called_process_error_is_redacted_on_both_cmd_and_args(
        self, mocker
    ) -> None:
        args = self._args()
        mocker.patch(
            "helix.sandbox.subprocess.run",
            side_effect=subprocess.CalledProcessError(
                returncode=125, cmd=args, output="", stderr="no such image"
            ),
        )

        with pytest.raises(subprocess.CalledProcessError) as excinfo:
            sandbox_module._run_docker(args, check=True)

        exc = excinfo.value
        assert self.SENSITIVE_VALUE not in repr(exc)
        assert self.SENSITIVE_VALUE not in str(exc.cmd)
        assert self.SENSITIVE_VALUE not in str(exc.args)
        assert exc.returncode == 125
        assert exc.stderr == "no such image"

    def test_timeout_expired_is_redacted(self, mocker) -> None:
        args = self._args()
        mocker.patch(
            "helix.sandbox.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=args, timeout=1.5),
        )

        with pytest.raises(subprocess.TimeoutExpired) as excinfo:
            sandbox_module._run_docker(args, check=False)

        exc = excinfo.value
        assert self.SENSITIVE_VALUE not in repr(exc)
        assert self.SENSITIVE_VALUE not in str(exc.cmd)
        assert self.SENSITIVE_VALUE not in str(exc.args)
        assert exc.timeout == 1.5

    def test_completed_process_args_are_redacted(self, mocker) -> None:
        args = self._args()
        mocker.patch(
            "helix.sandbox.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=args, returncode=0, stdout="ok", stderr=""
            ),
        )

        result = sandbox_module._run_docker(args, check=False)

        assert self.SENSITIVE_VALUE not in str(result.args)

    def test_captured_output_is_left_verbatim(self, mocker) -> None:
        """Container output is not scrubbed — deliberately.

        Substring-matching output against every env value would also rewrite
        the non-secret variables HELIX injects (``HOME=/home/node``), turning
        container tracebacks into `<redacted>` soup and corrupting output that
        callers still parse on a non-zero exit.
        """
        args = self._args()
        traceback_text = 'File "/home/node/app/run.py", line 3\n'
        mocker.patch(
            "helix.sandbox.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=args, returncode=1, stdout="", stderr=traceback_text
            ),
        )

        result = sandbox_module._run_docker(args, check=False)

        assert result.stderr == traceback_text
