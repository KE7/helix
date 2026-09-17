"""Isolation knobs used by leak-free sandboxed runs.

Covers ``agent.disallowed_tools`` (Claude Code ``--disallowedTools``),
``sandbox.network`` naming an existing Docker network, ``sandbox.auth_volume``
(a dedicated ``/home/node`` volume per run), ``evaluator.sidecar.volumes``
(private mounts for the sidecar service only) and the ``xhigh``/``max`` effort
levels for the ``claude`` backend.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

import helix.sandbox as sandbox_module
from helix.config import (
    AgentConfig,
    EvaluatorConfig,
    EvaluatorSidecarConfig,
    HelixConfig,
    SandboxConfig,
)
from helix.mutator import _build_backend_args, invoke_claude_code
from helix.sandbox import (
    run_sandboxed_command,
    sandbox_auth_volume_name,
    start_evaluator_sidecar,
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
# agent.disallowed_tools
# ---------------------------------------------------------------------------


class TestDisallowedTools:
    def test_default_is_empty_and_adds_no_flag(self) -> None:
        config = AgentConfig()
        assert config.disallowed_tools == []
        args = _build_backend_args("/workspace", config, ".agent_task_prompt.md")
        assert not any(a.startswith("--disallowedTools") for a in args)

    def test_claude_args_carry_one_bound_flag_before_the_prompt(self) -> None:
        config = AgentConfig(
            backend="claude",
            model="claude-sonnet-5",
            effort="xhigh",
            max_turns=7,
            disallowed_tools=["WebSearch", "WebFetch"],
        )
        args = _build_backend_args("/workspace", config, ".agent_task_prompt.md")
        flags = [a for a in args if a.startswith("--disallowedTools")]
        # One element using the ``=`` form: Claude Code's option is variadic,
        # so a separate value token could swallow the trailing prompt.
        assert flags == ["--disallowedTools=WebSearch,WebFetch"]
        assert args.index(flags[0]) == len(args) - 2
        assert ".agent_task_prompt.md" in args[-1]
        assert args[args.index("--effort") + 1] == "xhigh"
        assert args[args.index("--max-turns") + 1] == "7"

    def test_flag_reaches_the_subprocess(self, mocker) -> None:
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(stdout="{}", stderr="", returncode=0)
        config = AgentConfig(disallowed_tools=["WebSearch"])
        invoke_claude_code("/tmp/wt", "prompt", config)
        argv = mock_run.call_args[0][0]
        assert "--disallowedTools=WebSearch" in argv

    @pytest.mark.parametrize("backend", ["agy", "codex", "cursor", "opencode"])
    def test_other_backends_reject_the_option_at_load(self, backend: str) -> None:
        with pytest.raises(ValidationError, match="only supported for backend='claude'"):
            AgentConfig(backend=backend, disallowed_tools=["WebSearch"])  # type: ignore[arg-type]

    def test_backend_override_cannot_bypass_the_check(self) -> None:
        # ``helix evolve --backend`` applies overrides with ``model_copy``,
        # which skips validation; the argv builder must still refuse.
        config = AgentConfig(disallowed_tools=["WebSearch"]).model_copy(
            update={"backend": "codex"}
        )
        with pytest.raises(ValueError, match="only supported for backend='claude'"):
            _build_backend_args("/workspace", config, ".agent_task_prompt.md")

    @pytest.mark.parametrize("bad", ["", "   ", "WebSearch,WebFetch"])
    def test_rejects_empty_or_comma_entries(self, bad: str) -> None:
        with pytest.raises(ValidationError, match="disallowed_tools entries"):
            AgentConfig(disallowed_tools=[bad])

    def test_loads_from_toml(self, tmp_path: Path) -> None:
        from helix.config import load_config

        toml = tmp_path / "helix.toml"
        toml.write_text(
            'objective = "x"\n'
            "[evaluator]\n"
            'command = "echo 1"\n'
            "[agent]\n"
            'backend = "claude"\n'
            'disallowed_tools = ["WebSearch", "WebFetch"]\n'
        )
        assert load_config(toml).agent.disallowed_tools == ["WebSearch", "WebFetch"]


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
            "claude", image="agent:test", generation=1, volume="run-home"
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
        assert reset.call_args.kwargs["volume"] == "run-home"
        assert reset.call_args.kwargs["generation"] == 4


# ---------------------------------------------------------------------------
# evaluator.sidecar.volumes
# ---------------------------------------------------------------------------


class TestSidecarVolumes:
    def _sidecar(self, volumes: list[str]) -> EvaluatorSidecarConfig:
        return EvaluatorSidecarConfig(
            image="eval:test",
            command="python /srv/server.py",
            endpoint="http://helix-evaluator:8080/evaluate",
            volumes=volumes,
        )

    @pytest.mark.parametrize(
        "spec",
        ["/host/ledger:/ledger", "/host/ledger:/ledger:rw", "named-vol:/data:ro"],
    )
    def test_accepts_mount_specs(self, spec: str) -> None:
        assert self._sidecar([spec]).volumes == [spec]

    @pytest.mark.parametrize(
        "spec", ["/host/only", ":/x", "/h:relative", "/h:/c:rx", "/a:/b:ro:extra"]
    )
    def test_rejects_malformed_specs(self, spec: str) -> None:
        with pytest.raises(ValidationError, match="sidecar.volumes"):
            self._sidecar([spec])

    def test_mounts_apply_to_sidecar_service_only(self, tmp_path: Path, mocker) -> None:
        calls: list[list[str]] = []

        def fake_run_docker(args, *, check=True):
            calls.append(args)
            if args[:2] == ["docker", "inspect"]:
                return subprocess.CompletedProcess(
                    args, 0, stdout="true running\n", stderr=""
                )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        mocker.patch("helix.sandbox._run_docker", side_effect=fake_run_docker)
        sidecar = self._sidecar(["/host/ledger:/ledger:rw"])
        with start_evaluator_sidecar(sidecar):
            pass
        service = next(call for call in calls if call[:3] == ["docker", "run", "-d"])
        assert service[service.index("-v") + 1] == "/host/ledger:/ledger:rw"
        healthchecks = [
            call for call in calls if call[:3] == ["docker", "run", "--rm"]
        ]
        assert healthchecks
        assert all("/host/ledger:/ledger:rw" not in call for call in healthchecks)
