"""Tests for HELIX_RESULT= evaluator output support."""

from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pytest

from helix.population import Candidate, EvalResult
from helix.config import HelixConfig, EvaluatorConfig
from helix.executor import run_evaluator
from helix.mutator import build_mutation_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidate(worktree_path: str = "/tmp/fake-worktree") -> Candidate:
    return Candidate(
        id="cand-001",
        worktree_path=worktree_path,
        branch_name="helix/cand-001",
        generation=1,
        parent_id=None,
        parent_ids=[],
        operation="mutate",
    )


def make_config(
    command: str = "python eval.py",
    score_parser: str = "exitcode",
) -> HelixConfig:
    evaluator = EvaluatorConfig(
        command=command,
        score_parser=score_parser,
        include_stdout=True,
        include_stderr=True,
        extra_commands=[],
    )
    return HelixConfig(objective="test objective", evaluator=evaluator)


def _mock_subprocess_result(stdout: str, stderr: str = "", returncode: int = 0):
    """Create a mock subprocess.CompletedProcess."""
    result = MagicMock()
    result.stdout = stdout
    result.stderr = stderr
    result.returncode = returncode
    return result


# ---------------------------------------------------------------------------
# Tests: HELIX_RESULT= parsing in executor
# ---------------------------------------------------------------------------


class TestHelixResultParsing:
    """Test that HELIX_RESULT= lines are parsed from evaluator stdout."""

    @patch("helix.executor.subprocess.run")
    def test_helix_result_parsed(self, mock_run):
        """HELIX_RESULT=[score, side_info] should override parser score and set side_info."""
        side_info = {"accuracy": 0.95, "loss": 0.12, "note": "converged fast"}
        helix_line = f"HELIX_RESULT={json.dumps([0.95, side_info])}"
        stdout = f"Some evaluator output\n{helix_line}\nMore output\n"
        mock_run.return_value = _mock_subprocess_result(stdout, returncode=0)

        config = make_config()
        candidate = make_candidate()
        result = run_evaluator(candidate, config, split="val")

        assert result.scores["success"] == 0.95
        assert result.side_info == side_info

    @patch("helix.executor.subprocess.run")
    def test_no_helix_result_backward_compat(self, mock_run):
        """Without HELIX_RESULT=, fallback to parser (backward compatible)."""
        stdout = "Regular evaluator output\nNo special lines\n"
        mock_run.return_value = _mock_subprocess_result(stdout, returncode=0)

        config = make_config()
        candidate = make_candidate()
        result = run_evaluator(candidate, config, split="val")

        # exitcode parser: returncode 0 -> success=1.0
        assert result.scores["success"] == 1.0
        assert result.side_info is None

    @patch("helix.executor.subprocess.run")
    def test_helix_result_malformed_json_falls_back(self, mock_run):
        """Malformed HELIX_RESULT= line should not crash; side_info stays None."""
        stdout = "HELIX_RESULT=not-valid-json\nOther output\n"
        mock_run.return_value = _mock_subprocess_result(stdout, returncode=0)

        config = make_config()
        candidate = make_candidate()
        result = run_evaluator(candidate, config, split="val")

        # Falls back to parser path
        assert result.side_info is None

    @patch("helix.executor.subprocess.run")
    def test_helix_result_multiple_lines_takes_first(self, mock_run):
        """When multiple HELIX_RESULT= lines exist, only the first is used."""
        side_info_1 = {"source": "first"}
        side_info_2 = {"source": "second"}
        line1 = f"HELIX_RESULT={json.dumps([0.9, side_info_1])}"
        line2 = f"HELIX_RESULT={json.dumps([0.1, side_info_2])}"
        stdout = f"preamble\n{line1}\nmiddle\n{line2}\nend\n"
        mock_run.return_value = _mock_subprocess_result(stdout, returncode=0)

        config = make_config()
        candidate = make_candidate()
        result = run_evaluator(candidate, config, split="val")

        assert result.scores["success"] == 0.9
        assert result.side_info == side_info_1

    @patch("helix.executor.subprocess.run")
    def test_helix_result_non_dict_side_info_ignored(self, mock_run):
        """When payload[1] is not a dict, score is used but side_info stays None."""
        helix_line = f"HELIX_RESULT={json.dumps([0.75, 'just a string'])}"
        stdout = f"output\n{helix_line}\n"
        mock_run.return_value = _mock_subprocess_result(stdout, returncode=0)

        config = make_config()
        candidate = make_candidate()
        result = run_evaluator(candidate, config, split="val")

        assert result.scores["success"] == 0.75
        assert result.side_info is None

    @patch("helix.executor.subprocess.run")
    def test_helix_result_stdout_still_in_asi(self, mock_run):
        """Non-HELIX_RESULT stdout lines should still flow into ASI."""
        side_info = {"metric": 42}
        helix_line = f"HELIX_RESULT={json.dumps([0.8, side_info])}"
        stdout = f"line1\n{helix_line}\nline2\n"
        mock_run.return_value = _mock_subprocess_result(stdout, returncode=0)

        config = make_config()
        candidate = make_candidate()
        result = run_evaluator(candidate, config, split="val")

        # Full stdout (including HELIX_RESULT line) is in ASI
        assert "line1" in result.asi["stdout"]
        assert "line2" in result.asi["stdout"]


# ---------------------------------------------------------------------------
# Tests: side_info in mutation prompt
# ---------------------------------------------------------------------------


class TestSideInfoInMutationPrompt:
    """Test that side_info appears as a Diagnostics section in the mutation prompt."""

    def test_diagnostics_section_present(self):
        """When side_info is set, mutation prompt should have ## Diagnostics."""
        eval_result = EvalResult(
            candidate_id="cand-001",
            scores={"success": 0.8},
            asi={"stdout": "output", "stderr": ""},
            instance_scores={"ex1": 0.8},
            side_info={"accuracy": 0.95, "loss": 0.12},
        )
        prompt = build_mutation_prompt("improve accuracy", eval_result)
        assert "## Diagnostics" in prompt
        assert "accuracy: 0.95" in prompt
        assert "loss: 0.12" in prompt

    def test_no_diagnostics_without_side_info(self):
        """When side_info is None, no Diagnostics section should appear."""
        eval_result = EvalResult(
            candidate_id="cand-001",
            scores={"success": 0.5},
            asi={"stdout": "output", "stderr": ""},
            instance_scores={"ex1": 0.5},
        )
        prompt = build_mutation_prompt("improve accuracy", eval_result)
        assert "## Diagnostics" not in prompt

    def test_diagnostics_separate_from_scores(self):
        """Diagnostics section should be separate from Current Evaluation Scores."""
        eval_result = EvalResult(
            candidate_id="cand-001",
            scores={"success": 0.7},
            asi={"stdout": "out", "stderr": ""},
            instance_scores={},
            side_info={"hint": "try batch norm"},
        )
        prompt = build_mutation_prompt("optimize", eval_result)
        scores_idx = prompt.index("## Current Evaluation Scores")
        diag_idx = prompt.index("## Diagnostics")
        assert diag_idx != scores_idx


# ---------------------------------------------------------------------------
# Tests: EvalResult serialization round-trip
# ---------------------------------------------------------------------------


class TestEvalResultSerialization:
    """Test that side_info survives to_dict / from_dict round-trip."""

    def test_round_trip_with_side_info(self):
        er = EvalResult(
            candidate_id="c1",
            scores={"s": 1.0},
            asi={},
            instance_scores={"i1": 1.0},
            side_info={"k": "v"},
        )
        d = er.to_dict()
        assert d["side_info"] == {"k": "v"}
        er2 = EvalResult.from_dict(d)
        assert er2.side_info == {"k": "v"}

    def test_round_trip_without_side_info(self):
        er = EvalResult(
            candidate_id="c1",
            scores={"s": 1.0},
            asi={},
            instance_scores={"i1": 1.0},
        )
        d = er.to_dict()
        assert "side_info" not in d
        er2 = EvalResult.from_dict(d)
        assert er2.side_info is None
