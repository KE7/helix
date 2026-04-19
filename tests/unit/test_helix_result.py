"""Executor-level tests for ``score_parser="helix_result"`` — now the
GEPA-parity per-example contract.

BREAKING (pre-1.0): ``helix_result`` previously accepted one
``[score, side_info_dict]`` blob per batch; it now takes one
``[score, side_info]`` pair **per example**, positional to the ids in
``helix_batch.json``.  HELIX zips them into id-keyed
``instance_scores`` (for the minibatch gate) and stores the raw
per-example side_info list on ``EvalResult`` for the reflection
prompt.  See :mod:`helix.parsers.helix_result` and
``/tmp/gepa_audit_report.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from helix.population import Candidate, EvalResult
from helix.config import HelixConfig, EvaluatorConfig
from helix.exceptions import EvaluatorError
from helix.executor import run_evaluator
from helix.mutator import build_mutation_prompt


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidate(worktree_path: str | Path = "/tmp/fake-worktree") -> Candidate:
    return Candidate(
        id="cand-001",
        worktree_path=str(worktree_path),
        branch_name="helix/cand-001",
        generation=1,
        parent_id=None,
        parent_ids=[],
        operation="mutate",
    )


def make_config(
    command: str = "python eval.py",
    score_parser: str = "helix_result",
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


def _write_batch(worktree: Path, ids: list[str]) -> None:
    (worktree / "helix_batch.json").write_text(json.dumps(ids))


def _pairs(scores: list[float], side_infos: list[dict] | None = None) -> list[list]:
    if side_infos is None:
        side_infos = [{} for _ in scores]
    return [[s, si] for s, si in zip(scores, side_infos)]


# ---------------------------------------------------------------------------
# Tests: HELIX_RESULT= per-example parsing via run_evaluator
# ---------------------------------------------------------------------------


class TestHelixResultParsing:
    """End-to-end: helix_result parser dispatch + per-example side_info
    plumbing onto ``EvalResult``."""

    @patch("helix.executor.subprocess.run")
    def test_per_example_pairs_become_id_keyed(self, mock_run, tmp_path: Path):
        """Each ``[score, side_info]`` pair is positional to the id list
        in ``helix_batch.json``.  HELIX zips them into
        ``instance_scores`` and stores the side_info list on
        ``EvalResult``."""
        ids = ["task__0", "task__1", "task__2"]
        _write_batch(tmp_path, ids)
        side_infos = [
            {"note": "example 0", "loss": 0.1},
            {"note": "example 1", "loss": 0.2},
            {"note": "example 2"},
        ]
        helix_line = f"HELIX_RESULT={json.dumps(_pairs([1.0, 0.0, 0.5], side_infos))}"
        stdout = f"Some evaluator output\n{helix_line}\nMore output\n"
        mock_run.return_value = _mock_subprocess_result(stdout, returncode=0)

        result = run_evaluator(make_candidate(tmp_path), make_config(), split="val")

        assert result.instance_scores == {
            "task__0": 1.0, "task__1": 0.0, "task__2": 0.5,
        }
        # Aggregate is mean(scores).
        assert result.scores["success"] == pytest.approx(0.5)
        # Per-example side_info captured in id order.
        assert result.per_example_side_info == side_infos
        # Legacy batch-level side_info is not populated by this parser.
        assert result.side_info is None

    @patch("helix.executor.subprocess.run")
    def test_missing_helix_result_raises_evaluator_error(
        self, mock_run, tmp_path: Path
    ):
        """No ``HELIX_RESULT=`` line on stdout → strict parser raises."""
        _write_batch(tmp_path, ["task__0"])
        mock_run.return_value = _mock_subprocess_result(
            "Regular evaluator output\nNo special lines\n", returncode=0,
        )

        with pytest.raises(EvaluatorError, match="no HELIX_RESULT"):
            run_evaluator(make_candidate(tmp_path), make_config(), split="val")

    @patch("helix.executor.subprocess.run")
    def test_malformed_helix_result_raises(self, mock_run, tmp_path: Path):
        """Malformed ``HELIX_RESULT=`` line → strict parser raises rather
        than falling back (the fallback was the old footgun)."""
        _write_batch(tmp_path, ["task__0"])
        mock_run.return_value = _mock_subprocess_result(
            "HELIX_RESULT=not-valid-json\nOther output\n", returncode=0,
        )

        with pytest.raises(EvaluatorError, match="JSON-decode"):
            run_evaluator(make_candidate(tmp_path), make_config(), split="val")

    @patch("helix.executor.subprocess.run")
    def test_multiple_helix_result_lines_raises(self, mock_run, tmp_path: Path):
        """The executor guards against multiple ``HELIX_RESULT=`` lines
        and raises before the parser runs."""
        _write_batch(tmp_path, ["task__0"])
        line1 = f"HELIX_RESULT={json.dumps(_pairs([0.9]))}"
        line2 = f"HELIX_RESULT={json.dumps(_pairs([0.1]))}"
        stdout = f"preamble\n{line1}\nmiddle\n{line2}\nend\n"
        mock_run.return_value = _mock_subprocess_result(stdout, returncode=0)

        with pytest.raises(RuntimeError, match="Multiple HELIX_RESULT= lines found"):
            run_evaluator(make_candidate(tmp_path), make_config(), split="val")

    @patch("helix.executor.subprocess.run")
    def test_stdout_still_in_asi(self, mock_run, tmp_path: Path):
        """Non-``HELIX_RESULT=`` stdout lines still flow into ASI for
        the mutation prompt."""
        _write_batch(tmp_path, ["task__0"])
        helix_line = f"HELIX_RESULT={json.dumps([[0.8, {'metric': 42}]])}"
        stdout = f"line1\n{helix_line}\nline2\n"
        mock_run.return_value = _mock_subprocess_result(stdout, returncode=0)

        result = run_evaluator(make_candidate(tmp_path), make_config(), split="val")

        # Full stdout (including HELIX_RESULT line) is in ASI.
        assert "line1" in result.asi["stdout"]
        assert "line2" in result.asi["stdout"]

    @patch("helix.executor.subprocess.run")
    def test_instance_ids_subset_post_filter_matches_batch(
        self, mock_run, tmp_path: Path
    ):
        """When HELIX requests a subset via ``instance_ids``, it writes
        exactly those ids to ``helix_batch.json`` pre-invocation, so the
        parser's output already matches and the post-filter is a no-op."""
        ids = ["task__0", "task__1", "task__2"]
        _write_batch(tmp_path, ids)
        helix_line = f"HELIX_RESULT={json.dumps(_pairs([0.1, 0.2, 0.3]))}"
        mock_run.return_value = _mock_subprocess_result(
            helix_line + "\n", returncode=0,
        )

        result = run_evaluator(
            make_candidate(tmp_path), make_config(), split="train", instance_ids=ids,
        )

        assert result.instance_scores == {"task__0": 0.1, "task__1": 0.2, "task__2": 0.3}

    @patch("helix.executor.subprocess.run")
    def test_nonzero_returncode_zeros_aggregate_keeps_instance_scores(
        self, mock_run, tmp_path: Path
    ):
        """Non-zero exit: ``scores["success"]`` drops to 0.0 but per-id
        scores are preserved for diagnostics."""
        _write_batch(tmp_path, ["task__0", "task__1"])
        helix_line = f"HELIX_RESULT={json.dumps(_pairs([1.0, 1.0]))}"
        mock_run.return_value = _mock_subprocess_result(
            helix_line + "\n", returncode=2,
        )

        result = run_evaluator(make_candidate(tmp_path), make_config(), split="val")

        assert result.scores["success"] == 0.0
        assert result.instance_scores == {"task__0": 1.0, "task__1": 1.0}


# ---------------------------------------------------------------------------
# Tests: side_info in mutation prompt (legacy batch-level path — unchanged)
# ---------------------------------------------------------------------------


class TestSideInfoInMutationPrompt:
    """The legacy batch-level ``EvalResult.side_info`` dict still
    renders as a Diagnostics section in the mutation prompt.  The new
    ``per_example_side_info`` is NOT yet wired into the prompt — that's
    a follow-up PR.  These tests pin the existing rendering so it keeps
    working for any non-``helix_result`` path that populates the
    legacy field.
    """

    def test_diagnostics_section_present(self):
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
        eval_result = EvalResult(
            candidate_id="cand-001",
            scores={"success": 0.5},
            asi={"stdout": "output", "stderr": ""},
            instance_scores={"ex1": 0.5},
        )
        prompt = build_mutation_prompt("improve accuracy", eval_result)
        assert "## Diagnostics" not in prompt

    def test_diagnostics_separate_from_scores(self):
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
# Tests: EvalResult serialization round-trip (incl. new per-example fields)
# ---------------------------------------------------------------------------


class TestEvalResultSerialization:
    """side_info and the new per-example fields survive
    to_dict / from_dict round-trip."""

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
        # New per-example field is also omitted when None.
        assert "per_example_side_info" not in d
        er2 = EvalResult.from_dict(d)
        assert er2.side_info is None
        assert er2.per_example_side_info is None

    def test_round_trip_with_per_example_side_info(self):
        er = EvalResult(
            candidate_id="c1",
            scores={"success": 0.5},
            asi={},
            instance_scores={"a": 1.0, "b": 0.0},
            per_example_side_info=[{"traj": "A"}, {"traj": "B"}],
        )
        d = er.to_dict()
        assert d["per_example_side_info"] == [{"traj": "A"}, {"traj": "B"}]
        er2 = EvalResult.from_dict(d)
        assert er2.per_example_side_info == [{"traj": "A"}, {"traj": "B"}]
