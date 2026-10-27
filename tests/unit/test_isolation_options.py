"""Isolation knobs used by leak-free sandboxed runs.

Covers ``sandbox.network`` naming an existing Docker network,
``sandbox.auth_volume`` (a dedicated ``/home/node`` volume per run) and the
``xhigh``/``max`` effort levels for the ``claude`` backend.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from pydantic import ValidationError

import helix.sandbox as sandbox_module
from helix.config import (
    AgentConfig,
    EvaluatorConfig,
    HelixConfig,
    SandboxConfig,
)
from helix.sandbox import (
    run_sandboxed_command,
    sandbox_auth_volume_name,
)


def _primary_docker_run(calls: list[list[str]]) -> list[str]:
    return next(
        call
        for call in calls
        if call[:2] == ["docker", "run"]
        and not any("find /workspace" in item for item in call)
        and "--user" in call
        and call[call.index("--user") + 1] == "node"
        and "/workspace:rw" in " ".join(call)
        and not any(".helix_artifacts" in item for item in call)
    )


# ---------------------------------------------------------------------------
# claude effort levels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_claude_accepts_xhigh_and_max_without_warning(effort: str, recwarn) -> None:
    HelixConfig(
        objective="x",
        evaluator=EvaluatorConfig(command="echo 1"),
        agent=AgentConfig(backend="claude", effort=effort),
    )
    assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []


def test_agy_still_warns_on_xhigh(recwarn) -> None:
    HelixConfig(
        objective="x",
        evaluator=EvaluatorConfig(command="echo 1"),
        agent=AgentConfig(backend="agy", effort="xhigh"),
    )
    messages = [str(w.message) for w in recwarn.list if w.category is UserWarning]
    assert any("not a recognized value" in m for m in messages), messages


# ---------------------------------------------------------------------------
# sandbox.network / sandbox.auth_volume
# ---------------------------------------------------------------------------


class TestSandboxNames:
    @pytest.mark.parametrize(
        "network", ["bridge", "none", "host", "helix-circle-clean-agent", "a.b_c-1"]
    )
    def test_network_accepts_builtin_and_named_networks(self, network: str) -> None:
        assert SandboxConfig(network=network).network == network

    @pytest.mark.parametrize(
        "network", ["", "-rm", "--privileged", "bad name", "net;rm", "../x"]
    )
    def test_network_rejects_option_like_or_malformed_names(self, network: str) -> None:
        with pytest.raises(ValidationError, match="sandbox.network"):
            SandboxConfig(network=network)

    @pytest.mark.parametrize("volume", ["", "-v", "a:b", "x y", "/abs/path"])
    def test_auth_volume_rejects_malformed_names(self, volume: str) -> None:
        with pytest.raises(ValidationError, match="sandbox.auth_volume"):
            SandboxConfig(auth_volume=volume)

    def test_auth_volume_name_defaults_to_shared_login_volume(self) -> None:
        assert sandbox_auth_volume_name("claude") == "helix-auth-claude"
        assert (
            sandbox_auth_volume_name("claude", SandboxConfig(enabled=True))
            == "helix-auth-claude"
        )
        assert (
            sandbox_auth_volume_name(
                "claude", SandboxConfig(enabled=True, auth_volume="run-home")
            )
            == "run-home"
        )

    def test_agent_container_uses_named_network_and_dedicated_volume(
        self, tmp_path: Path, mocker
    ) -> None:
        source = tmp_path / "candidate"
        source.mkdir()
        (source / "solve.py").write_text("x = 1\n")
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(
                args, 0, stdout='{"session_id":"sess_1"}', stderr=""
            )

        mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
        mocker.patch("helix.sandbox._host_owner", return_value=None)
        cfg = SandboxConfig(
            enabled=True,
            image="agent:test",
            network="helix-circle-clean-agent",
            auth_volume="helix-circle-clean-claude-home",
        )
        run_sandboxed_command(
            ["claude", "-p", "prompt"],
            cwd=source,
            env={},
            sandbox=cfg,
            scope="agent",
            sync_back=True,
            agent_backend="claude",
        )
        agent = _primary_docker_run(calls)
        assert agent[agent.index("--network") + 1] == "helix-circle-clean-agent"
        assert "helix-circle-clean-claude-home:/home/node:rw" in agent
        assert not any("helix-auth-claude" in item for item in agent)
        # The transcript helper reads the same dedicated volume.
        copy_call = next(
            call for call in calls if any("sess_1.jsonl" in item for item in call)
        )
        assert "helix-circle-clean-claude-home:/home/node:ro" in copy_call
        assert not any("helix-auth-claude" in item for c in calls for item in c)

    def test_state_reset_targets_the_given_volume(self, mocker) -> None:
        calls: list[list[str]] = []

        def fake_run(args, **kwargs):
            calls.append(args)
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        mocker.patch("helix.sandbox.subprocess.run", side_effect=fake_run)
        sandbox_module.reset_sandbox_agent_state(
            "claude", image="agent:test", generation=1,
            sandbox=SandboxConfig(enabled=True, auth_volume="run-home"),
        )
        [args] = calls
        assert "run-home:/home/node:rw" in args
        assert not any("helix-auth-claude" in item for item in args)

    def test_generation_reset_uses_configured_volume(self, mocker) -> None:
        from helix import evolution

        reset = mocker.patch("helix.evolution.reset_sandbox_agent_state")
        config = HelixConfig(
            objective="x",
            evaluator=EvaluatorConfig(command="echo 1"),
            sandbox=SandboxConfig(
                enabled=True, image="agent:test", auth_volume="run-home"
            ),
        )
        evolution._reset_sandbox_agent_state_for_generation(config, 4)
        assert reset.call_args.kwargs["sandbox"].auth_volume == "run-home"
        assert reset.call_args.kwargs["generation"] == 4
