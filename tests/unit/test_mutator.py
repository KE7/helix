"""Unit tests for helix.mutator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from helix.population import Candidate, EvalResult
from helix.config import ClaudeConfig, HelixConfig, EvaluatorConfig
from helix.mutator import (
    MutationError,
    AUTONOMOUS_SYSTEM_PROMPT,
    MUTATION_PROMPT_TEMPLATE,
    build_mutation_prompt,
    invoke_claude_code,
    mutate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_eval_result(
    candidate_id: str = "g0-s0",
    scores: dict | None = None,
    asi: dict | None = None,
    instance_scores: dict | None = None,
) -> EvalResult:
    return EvalResult(
        candidate_id=candidate_id,
        scores=scores if scores is not None else {"pass_rate": 0.5},
        asi=asi or {"stdout": "2 passed, 2 failed", "stderr": ""},
        instance_scores=instance_scores or {"test_a": 1.0, "test_b": 0.0},
    )


def make_candidate(
    cid: str = "g0-s0",
    worktree_path: str = "/tmp/fake-wt",
) -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=worktree_path,
        branch_name=f"helix/{cid}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def make_config(objective: str = "Pass all tests") -> HelixConfig:
    return HelixConfig(
        objective=objective,
        evaluator=EvaluatorConfig(command="pytest -q"),
    )


# ---------------------------------------------------------------------------
# Tests: build_mutation_prompt
# ---------------------------------------------------------------------------


class TestBuildMutationPrompt:
    def test_contains_objective(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("Optimise sorting", er)
        assert "Optimise sorting" in prompt

    def test_contains_scores(self):
        er = make_eval_result(scores={"pass_rate": 0.75})
        prompt = build_mutation_prompt("goal", er)
        assert "pass_rate" in prompt
        assert "0.75" in prompt

    def test_contains_stdout(self):
        er = make_eval_result(asi={"stdout": "unique_stdout_xyz", "stderr": ""})
        prompt = build_mutation_prompt("goal", er)
        assert "unique_stdout_xyz" in prompt

    def test_contains_stderr(self):
        er = make_eval_result(asi={"stdout": "", "stderr": "unique_stderr_abc"})
        prompt = build_mutation_prompt("goal", er)
        assert "unique_stderr_abc" in prompt

    def test_contains_extra_asi(self):
        er = make_eval_result(asi={"stdout": "", "stderr": "", "extra_0": "coverage: 80%"})
        prompt = build_mutation_prompt("goal", er)
        assert "coverage: 80%" in prompt

    def test_contains_background(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("goal", er, background="special context here")
        assert "special context here" in prompt

    def test_default_background_when_none(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("goal", er, background=None)
        assert "no additional background" in prompt

    def test_contains_mutation_complete_marker(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("goal", er)
        assert "[MUTATION COMPLETE]" in prompt

    def test_contains_autonomous_rules(self):
        er = make_eval_result()
        prompt = build_mutation_prompt("goal", er)
        assert "NEVER ask for human input" in prompt

    def test_no_scores_fallback(self):
        er = make_eval_result(scores={})
        prompt = build_mutation_prompt("goal", er)
        assert "no scores recorded" in prompt


class TestPerExampleDiagnostics:
    """``build_mutation_prompt`` renders ``eval_result.per_example_side_info``
    as the Diagnostics section under the new GEPA O.A. contract
    (``optimize_anything_adapter.py:524-553`` + ``format_samples``
    at ``gepa/strategies/instruction_proposal.py:54-95``).  The legacy
    batch-level ``side_info`` rendering is used only when
    ``per_example_side_info`` is absent.
    """

    def _make(
        self,
        per_example_side_info: list[dict] | None,
        side_info: dict | None = None,
        instance_scores: dict | None = None,
    ) -> EvalResult:
        return EvalResult(
            candidate_id="g0-s0",
            scores={"pass_rate": 0.5},
            asi={"stdout": "", "stderr": ""},
            instance_scores=instance_scores or {"cube_lifting__0": 1.0, "cube_lifting__1": 0.0},
            side_info=side_info,
            per_example_side_info=per_example_side_info,
        )

    def test_renders_per_example_headers_from_instance_scores(self):
        er = self._make(
            per_example_side_info=[
                {"trajectory": "fell over", "scores": {"success": 1.0, "affordance": 0.8}},
                {"trajectory": "stuck", "scores": {"success": 0.0, "affordance": 0.2}},
            ],
            instance_scores={"cube_lifting__0": 1.0, "cube_lifting__1": 0.0},
        )
        prompt = build_mutation_prompt("goal", er)
        # Per-example ids from instance_scores become "# Example <id>" headers.
        assert "# Example cube_lifting__0" in prompt
        assert "# Example cube_lifting__1" in prompt

    def test_reserved_scores_key_renamed_for_mutator(self):
        er = self._make(
            per_example_side_info=[
                {"scores": {"success": 1.0, "affordance": 0.8}},
            ],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        # GEPA parity: the reserved "scores" key renders under a
        # friendly header.  Raw "scores:" label should not appear.
        assert "## Scores (Higher is Better)" in prompt
        # Values are exposed with their axis names.
        assert "success: 1.0" in prompt
        assert "affordance: 0.8" in prompt

    def test_other_side_info_keys_render_verbatim(self):
        er = self._make(
            per_example_side_info=[
                {"trajectory": "unique_traj_marker", "loss": 0.42},
            ],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "## trajectory" in prompt
        assert "unique_traj_marker" in prompt
        assert "## loss" in prompt
        assert "0.42" in prompt

    def test_diagnostics_section_header_present(self):
        er = self._make(
            per_example_side_info=[{"scores": {"s": 1.0}}],
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "## Diagnostics" in prompt

    def test_empty_per_example_slots_still_render_headers(self):
        """When every per-example side_info is ``{}`` (bare-score
        evaluator), the section still shows per-example headers so the
        mutator knows the data was present but empty — rather than
        silently dropping the section."""
        er = self._make(
            per_example_side_info=[{}, {}],
            instance_scores={"ex_0": 1.0, "ex_1": 0.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "# Example ex_0" in prompt
        assert "# Example ex_1" in prompt
        assert "no per-example side_info" in prompt

    def test_legacy_side_info_used_when_per_example_absent(self):
        """Fallback: ``per_example_side_info=None`` + batch-level
        ``side_info={...}`` keeps the legacy rendering (non-helix_result
        paths that still populate the batch-level field)."""
        er = self._make(
            per_example_side_info=None,
            side_info={"legacy_key": "legacy_val"},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "## Diagnostics" in prompt
        assert "legacy_key" in prompt
        assert "legacy_val" in prompt
        # Per-example markers must NOT appear on the legacy path.
        assert "# Example" not in prompt

    def test_no_diagnostics_when_both_absent(self):
        er = self._make(per_example_side_info=None, side_info=None)
        prompt = build_mutation_prompt("goal", er)
        assert "## Diagnostics" not in prompt

    def test_per_example_takes_precedence_over_legacy(self):
        """When both fields are populated (hybrid evaluator path),
        per-example wins — the legacy batch-level dict is dropped
        from the rendered prompt."""
        er = self._make(
            per_example_side_info=[{"per_example_marker": "YES"}],
            side_info={"legacy_marker": "NO"},
            instance_scores={"ex_0": 1.0},
        )
        prompt = build_mutation_prompt("goal", er)
        assert "per_example_marker" in prompt
        assert "YES" in prompt
        assert "legacy_marker" not in prompt
        assert "NO" not in prompt


# ---------------------------------------------------------------------------
# Tests: invoke_claude_code
# ---------------------------------------------------------------------------


class TestInvokeClaudeCode:
    def test_returns_parsed_json_on_success(self, mocker):
        payload = {"result": "ok", "turns": 3}
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout=json.dumps(payload),
            stderr="",
            returncode=0,
        )
        config = ClaudeConfig()
        result = invoke_claude_code("/tmp/wt", "do something", config)
        assert result == payload

    def test_raises_on_nonzero_returncode(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="fatal error",
            returncode=1,
        )
        config = ClaudeConfig()
        with pytest.raises(MutationError, match="exited with code 1"):
            invoke_claude_code("/tmp/wt", "do something", config)

    def test_raises_on_invalid_json(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="not valid json {{{{",
            stderr="",
            returncode=0,
        )
        config = ClaudeConfig()
        with pytest.raises(MutationError, match="Failed to parse"):
            invoke_claude_code("/tmp/wt", "do something", config)

    def test_cli_args_include_required_flags(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(stdout="{}", stderr="", returncode=0)
        config = ClaudeConfig(allowed_tools=["Read", "Edit"])
        invoke_claude_code("/tmp/wt", "the prompt", config)

        call_args = mock_run.call_args
        args_list = call_args[0][0]

        assert "claude" in args_list
        assert "--dangerously-skip-permissions" in args_list
        assert "--print" in args_list
        assert "--output-format" in args_list
        assert "json" in args_list
        assert "--allowedTools" in args_list
        assert "Read,Edit" in args_list
        assert "--max-turns" not in args_list
        assert "-p" in args_list
        assert "the prompt" in args_list

    def test_uses_correct_cwd(self, mocker):
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(stdout="{}", stderr="", returncode=0)
        config = ClaudeConfig()
        invoke_claude_code("/specific/path", "prompt", config)
        assert mock_run.call_args[1]["cwd"] == "/specific/path"

    def test_no_timeout_in_subprocess(self, mocker):
        """subprocess.run should not have a timeout — let Claude run forever."""
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(stdout="{}", stderr="", returncode=0)
        config = ClaudeConfig()
        invoke_claude_code("/tmp/wt", "prompt", config)
        assert "timeout" not in mock_run.call_args[1]


# ---------------------------------------------------------------------------
# Tests: mutate
# ---------------------------------------------------------------------------


class TestMutate:
    def test_returns_candidate_on_success(self, mocker):
        parent = make_candidate("g0-s0", "/tmp/parent")
        er = make_eval_result("g0-s0")
        config = make_config()

        child = make_candidate("g1-s0", "/tmp/g1-s0")
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch("helix.mutator.invoke_claude_code", return_value={"result": "ok"})
        mocker.patch("helix.mutator.snapshot_candidate", return_value="abc123")
        mocker.patch("helix.mutator.remove_worktree")

        result = mutate(parent, er, "g1-s0", config, Path("/tmp"))

        assert result is child

    def test_sets_operation_to_mutate(self, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child = make_candidate("g1-s0")
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch("helix.mutator.invoke_claude_code", return_value={})
        mocker.patch("helix.mutator.snapshot_candidate", return_value="sha")
        mocker.patch("helix.mutator.remove_worktree")

        result = mutate(parent, er, "g1-s0", config, Path("/tmp"))
        assert result.operation == "mutate"

    def test_returns_none_on_mutation_error(self, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child = make_candidate("g1-s0")
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch(
            "helix.mutator.invoke_claude_code",
            side_effect=MutationError("timeout"),
        )
        mock_remove = mocker.patch("helix.mutator.remove_worktree")
        mocker.patch("helix.mutator.snapshot_candidate")

        result = mutate(parent, er, "g1-s0", config, Path("/tmp"))

        assert result is None
        mock_remove.assert_called_once_with(child)

    def test_removes_worktree_on_failure(self, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child = make_candidate("g1-s0")
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch(
            "helix.mutator.invoke_claude_code",
            side_effect=MutationError("bad json"),
        )
        mock_remove = mocker.patch("helix.mutator.remove_worktree")
        mocker.patch("helix.mutator.snapshot_candidate")

        mutate(parent, er, "g1-s0", config, Path("/tmp"))

        mock_remove.assert_called_once_with(child)

    def test_snapshot_not_called_by_mutate_on_success(self, mocker):
        """mutate() must NOT call snapshot_candidate — the caller owns that step.

        Callers (evolution.py) must call save_state() BEFORE snapshot_candidate()
        so that state is persisted even if the commit step crashes.
        """
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child = make_candidate("g1-s0")
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch("helix.mutator.invoke_claude_code", return_value={})
        mock_snapshot = mocker.patch("helix.mutator.snapshot_candidate", return_value="sha")
        mocker.patch("helix.mutator.remove_worktree")

        result = mutate(parent, er, "g1-s0", config, Path("/tmp"))

        # mutate() returns the child but does NOT snapshot internally
        assert result is child
        mock_snapshot.assert_not_called()

    def test_snapshot_not_called_on_failure(self, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child = make_candidate("g1-s0")
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch(
            "helix.mutator.invoke_claude_code",
            side_effect=MutationError("timeout"),
        )
        mock_snapshot = mocker.patch("helix.mutator.snapshot_candidate")
        mocker.patch("helix.mutator.remove_worktree")

        mutate(parent, er, "g1-s0", config, Path("/tmp"))

        mock_snapshot.assert_not_called()

    def test_passes_background_to_prompt(self, mocker):
        parent = make_candidate("g0-s0")
        er = make_eval_result()
        config = make_config()

        child = make_candidate("g1-s0")
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mock_invoke = mocker.patch("helix.mutator.invoke_claude_code", return_value={})
        mocker.patch("helix.mutator.snapshot_candidate", return_value="sha")
        mocker.patch("helix.mutator.remove_worktree")

        mutate(parent, er, "g1-s0", config, Path("/tmp"), background="special context")

        prompt_arg = mock_invoke.call_args[0][1]
        assert "special context" in prompt_arg
