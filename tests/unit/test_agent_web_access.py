"""``[agent].web_access = false``: argv, env and validation per backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from helix.cli import cli
from helix.config import AgentConfig, SandboxConfig
from helix.mutator import invoke_claude_code

NO_WEB_ARGS = {
    "claude": ["--disallowedTools=WebSearch,WebFetch"],
    "codex": ["-c", 'web_search="disabled"', "-c", "features.apps=false"],
    "opencode": [],
}
NO_WEB_FLAGS = {a for flags in NO_WEB_ARGS.values() for a in flags if a != "-c"}
OPENCODE_PERMISSION = '{"webfetch": "deny", "websearch": "deny"}'


def _invoke(mocker: Any, config: AgentConfig, **kwargs: Any) -> MagicMock:
    run = mocker.patch(
        "helix.mutator.run_sandboxed_command"
        if "sandbox" in kwargs
        else "helix.mutator.subprocess.run"
    )
    run.return_value = MagicMock(stdout="{}", stderr="", returncode=0)
    invoke_claude_code("/tmp/wt", "the prompt", config, **kwargs)
    return run


@pytest.mark.parametrize("backend", sorted(NO_WEB_ARGS))
@pytest.mark.parametrize("web_access", [True, False])
def test_web_access_argv_and_env(mocker: Any, backend: str, web_access: bool) -> None:
    run = _invoke(mocker, AgentConfig(backend=backend, web_access=web_access))
    args, env = run.call_args[0][0], run.call_args[1]["env"]
    assert ".agent_task_prompt.md" in args[-1]
    # User config is never skipped: web_access only removes web tools.
    assert "--ignore-user-config" not in args
    assert "--strict-mcp-config" not in args
    assert "XDG_CONFIG_HOME" not in env
    if web_access:
        assert not NO_WEB_FLAGS & set(args)
        assert "OPENCODE_PERMISSION" not in env
        return
    expected = NO_WEB_ARGS[backend]
    # The flags sit right before the prompt, which stays the last argument.
    assert args[-1 - len(expected) : -1] == expected
    if backend == "opencode":
        assert env["OPENCODE_PERMISSION"] == OPENCODE_PERMISSION
    else:
        assert "OPENCODE_PERMISSION" not in env


def test_opencode_web_deny_reaches_the_sandbox(mocker: Any) -> None:
    run = _invoke(
        mocker,
        AgentConfig(backend="opencode", web_access=False),
        sandbox=SandboxConfig(enabled=True),
    )
    assert run.call_args[1]["env"]["OPENCODE_PERMISSION"] == OPENCODE_PERMISSION


@pytest.mark.parametrize("backend", ["agy", "cursor"])
def test_backends_without_a_switch_reject_web_access_false(backend: str) -> None:
    with pytest.raises(ValueError, match="agent.web_access = false"):
        AgentConfig(backend=backend, web_access=False)  # type: ignore[arg-type]
    AgentConfig(backend=backend)  # type: ignore[arg-type]


@pytest.mark.parametrize(("backend", "exit_code"), [("cursor", 2), ("codex", 0)])
def test_evolve_validates_backend_override(
    tmp_path: Path, mocker: Any, backend: str, exit_code: int
) -> None:
    (tmp_path / "helix.toml").write_text(
        'objective = "test"\n\n[evaluator]\ncommand = "true"\n\n'
        "[agent]\nweb_access = false\n"
    )
    run = mocker.patch("helix.evolution.run_evolution")
    result = CliRunner().invoke(
        cli, ["evolve", "--dir", str(tmp_path), "--backend", backend]
    )
    assert result.exit_code == exit_code, result.output
    if exit_code:
        assert "agent.web_access = false" in result.output
        run.assert_not_called()
    else:
        assert run.call_args[0][0].agent.backend == backend


def test_evolve_override_error_is_not_read_as_rich_markup(
    tmp_path: Path, mocker: Any
) -> None:
    # Unescaped, Rich reads pydantic's ``[type=value_error, ...]`` as a tag
    # and drops it (and a ``[/x]`` in an echoed value raises MarkupError).
    (tmp_path / "helix.toml").write_text(
        'objective = "test"\n\n[evaluator]\ncommand = "true"\n\n'
        "[agent]\nweb_access = false\n"
    )
    mocker.patch("helix.evolution.run_evolution")
    result = CliRunner().invoke(
        cli, ["evolve", "--dir", str(tmp_path), "--backend", "agy"]
    )
    assert result.exit_code == 2, result.output
    output = " ".join(result.output.split())
    assert "[type=value_error" in output


@pytest.mark.parametrize(
    ("saved", "current", "rejected"),
    [
        (True, True, False),
        (False, False, False),
        (False, True, True),
        (True, False, True),
    ],
)
def test_resume_pins_web_access(saved: bool, current: bool, rejected: bool) -> None:
    from helix.config import EvaluatorConfig, HelixConfig
    from helix.evolution import _resume_semantics, _validate_resume_semantics
    from helix.exceptions import ResumeIncompatibleError
    from helix.state import BudgetState, EvolutionState

    def config(web_access: bool) -> HelixConfig:
        return HelixConfig(
            objective="test",
            evaluator=EvaluatorConfig(command="true"),
            agent=AgentConfig(web_access=web_access),
        )

    state = EvolutionState(
        generation=1,
        frontier=[],
        instance_scores={},
        budget=BudgetState(),
        config_hash="x",
        resume_semantics=_resume_semantics(config(saved)),
    )
    if rejected:
        with pytest.raises(ResumeIncompatibleError, match="agent"):
            _validate_resume_semantics(state, config(current))
    else:
        _validate_resume_semantics(state, config(current))


def test_resume_semantics_unchanged_for_web_access_default() -> None:
    from helix.config import EvaluatorConfig, HelixConfig
    from helix.evolution import _resume_semantics

    # States saved before ``web_access`` existed have no ``agent`` key.
    semantics = _resume_semantics(
        HelixConfig(objective="t", evaluator=EvaluatorConfig(command="true"))
    )
    assert "agent" not in semantics
