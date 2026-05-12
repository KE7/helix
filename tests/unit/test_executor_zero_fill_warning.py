"""Tests for the post-filter applied by :func:`helix.executor.run_evaluator`.

When HELIX asks the evaluator for per-example scores via ``instance_ids``,
the executor maps the parser's ``instance_scores`` to only the requested ids.

``exitcode`` is a global pass/fail parser: it broadcasts the global
``"success"`` score to every requested example id rather than zero-filling
(which was the pre-fix behaviour and caused spurious rejections when the
evaluator exited 0).

Every other parser goes through the existing post-filter; if an id is
missing the executor zero-fills it with 0.0 and emits a warning so the
mismatch is visible.  Historically that silence hid the id-key-mismatch
footgun — 113 generations of evolution once failed silently because the old
``helix_result`` contract required an id-keyed ``side_info["scores"]`` dict
and the evaluator keyed it by aggregate metric names instead.

The new per-example ``helix_result`` contract eliminates that specific
footgun at the parser level (it raises on length mismatch), but the
warning still has value as defense in depth for every other parser.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helix.config import EvaluatorConfig, HelixConfig
from helix.executor import run_evaluator
from helix.population import Candidate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(worktree_path: Path) -> Candidate:
    return Candidate(
        id="cand-zfw",
        worktree_path=str(worktree_path),
        branch_name="helix/cand-zfw",
        generation=1,
        parent_id=None,
        parent_ids=[],
        operation="mutate",
    )


def _make_config(score_parser: str = "exitcode") -> HelixConfig:
    return HelixConfig(
        objective="test",
        evaluator=EvaluatorConfig(
            command="python eval.py",
            score_parser=score_parser,
            include_stdout=True,
            include_stderr=True,
            extra_commands=[],
        ),
    )


def _mock_subprocess(stdout: str, returncode: int = 0, stderr: str = "") -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.stderr = stderr
    m.returncode = returncode
    return m


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestZeroFillWarning:
    def test_exitcode_broadcasts_success_to_all_instance_ids(
        self, tmp_path: Path, mocker, caplog: pytest.LogCaptureFixture
    ) -> None:
        # ``exitcode`` is a global pass/fail parser; when ``instance_ids``
        # are requested the executor broadcasts the global "success" score
        # to every id instead of zero-filling.  Exit 0 → 1.0 for all ids.
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=_mock_subprocess("ok\n", returncode=0),
        )

        candidate = _make_candidate(tmp_path)
        instance_ids = [f"task__{i}" for i in range(4)]

        with caplog.at_level(logging.WARNING, logger="helix.executor"):
            result = run_evaluator(
                candidate,
                _make_config(score_parser="exitcode"),
                split="train",
                instance_ids=instance_ids,
            )

        # All requested ids present, all receiving the broadcasted 1.0.
        assert set(result.instance_scores.keys()) == set(instance_ids)
        assert all(v == 1.0 for v in result.instance_scores.values()), (
            "exitcode exit-0 should broadcast 1.0 to all requested ids"
        )

        # No "missing instance_scores" warning — broadcast is intentional.
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "missing instance_scores" in r.getMessage()
        ]
        assert not warnings, (
            "exitcode broadcast must not emit a 'missing instance_scores' warning; "
            f"got: {[r.getMessage() for r in warnings]}"
        )

    def test_exitcode_broadcasts_failure_to_all_instance_ids(
        self, tmp_path: Path, mocker
    ) -> None:
        # Exit non-zero → 0.0 broadcast to all ids.
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=_mock_subprocess("FAIL\n", returncode=1),
        )

        candidate = _make_candidate(tmp_path)
        instance_ids = ["ex0", "ex1", "ex2"]

        result = run_evaluator(
            candidate,
            _make_config(score_parser="exitcode"),
            split="train",
            instance_ids=instance_ids,
        )

        assert set(result.instance_scores.keys()) == set(instance_ids)
        assert all(v == 0.0 for v in result.instance_scores.values()), (
            "exitcode exit-nonzero should broadcast 0.0 to all requested ids"
        )

    def test_no_warning_when_parser_covers_all_ids(
        self, tmp_path: Path, mocker, caplog: pytest.LogCaptureFixture
    ) -> None:
        ids = ["task__0", "task__1", "task__2"]
        (tmp_path / "helix_batch.json").write_text(json.dumps(ids))
        # Per-example payload: one [score, side_info] pair per id.
        payload = [[1.0, {}], [0.5, {}], [0.0, {}]]
        helix_line = f"HELIX_RESULT={json.dumps(payload)}"
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=_mock_subprocess(helix_line + "\n", returncode=0),
        )

        candidate = _make_candidate(tmp_path)

        with caplog.at_level(logging.WARNING, logger="helix.executor"):
            run_evaluator(
                candidate,
                _make_config(score_parser="helix_result"),
                split="train",
                instance_ids=ids,
            )

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "missing instance_scores" in r.getMessage()
        ]
        assert not warnings, (
            "No warning expected when every requested id is present in the "
            f"parser's returned instance_scores; got: {[r.getMessage() for r in warnings]}"
        )

    def test_warning_sample_truncates_to_five_for_non_exitcode_parsers(
        self, tmp_path: Path, mocker, caplog: pytest.LogCaptureFixture
    ) -> None:
        # json_accuracy returns an empty instance_scores when the JSON has no
        # "instance_scores" key.  All 8 requested ids go missing → warning
        # must cap the sample list at 5 and surface the overflow count.
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=_mock_subprocess('{"accuracy": 0.5}\n', returncode=0),
        )

        candidate = _make_candidate(tmp_path)
        instance_ids = [f"task__{i}" for i in range(8)]

        with caplog.at_level(logging.WARNING, logger="helix.executor"):
            run_evaluator(
                candidate,
                _make_config(score_parser="json_accuracy"),
                split="train",
                instance_ids=instance_ids,
            )

        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "missing instance_scores" in r.getMessage()
        ]
        assert warnings
        msg = warnings[0].getMessage()
        assert "8/8" in msg
        # First five ids shown (task__0..task__4), and "+3 more" overflow marker present.
        assert "task__0" in msg and "task__4" in msg
        assert "+3 more" in msg
