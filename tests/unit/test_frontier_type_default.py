"""Regression: the shipped ``evolution.frontier_type`` default must not
fail closed on the seed evaluation.

The multi-axis frontier modes (``"objective"``, ``"hybrid"``,
``"cartesian"``) need per-example objective scores.  Those can only come
from the ``helix_result`` parser, and only when the evaluator author
opts in by emitting a ``"scores"`` sub-dict inside each per-example
``side_info``.

When the default was ``"hybrid"``, any project without that opt-in died
with :class:`~helix.population.MissingObjectiveScoresError` on the *seed*
evaluation — before generation 1 ever started.  The default is now
``"instance"``, and explicit hybrid mode tolerates missing objective scores
by staying on its instance axis. Objective/cartesian modes retain a typed
selection error when no objective axis exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from helix.config import EvaluatorConfig, EvolutionConfig, HelixConfig
from helix.executor import run_evaluator
from helix.population import (
    Candidate,
    EvalResult,
    MissingObjectiveScoresError,
    ParetoFrontier,
)

SEED_ID = "g0-s0"


def _seed_candidate(worktree_path: Path) -> Candidate:
    return Candidate(
        id=SEED_ID,
        worktree_path=str(worktree_path),
        branch_name=f"helix/{SEED_ID}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def _helix_result_config() -> HelixConfig:
    return HelixConfig(
        objective="test",
        evaluator=EvaluatorConfig(
            command="python evaluate.py",
        ),
    )


def _mock_subprocess(stdout: str) -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = 0
    return m


def _evaluate_without_objective_scores(
    tmp_path: Path, mocker: Any
) -> EvalResult:
    """Run the ``helix_result`` parser over an evaluator that returns
    per-example ``[score, side_info]`` pairs with **no** ``"scores"`` key.

    This is the shape the overwhelming majority of evaluators emit: a
    score plus freeform diagnostics, with no per-objective breakdown.
    """
    ids = ["i0", "i1"]
    (tmp_path / "helix_batch.json").write_text(json.dumps(ids))
    payload = [
        [1.0, {"trajectory": "solved i0"}],
        [0.5, {"trajectory": "partially solved i1"}],
    ]
    mocker.patch(
        "helix.executor.subprocess.run",
        return_value=_mock_subprocess("HELIX_RESULT=" + json.dumps(payload)),
    )
    result = run_evaluator(
        _seed_candidate(tmp_path), _helix_result_config(), split="val"
    )
    # Precondition for everything below: the parser ran fine and produced
    # real instance scores, but every objective slot came back empty.
    assert result.instance_scores == {"i0": 1.0, "i1": 0.5}
    assert result.objective_scores == [{}, {}]
    return result


class TestDefaultFrontierTypeIsUsable:
    def test_default_is_instance(self) -> None:
        assert EvolutionConfig().frontier_type == "instance"

    def test_seed_evaluation_without_objective_scores_is_accepted(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        """The reproducer.  Under the old ``"hybrid"`` default this
        raised ``MissingObjectiveScoresError`` for candidate ``g0-s0``
        and the run never reached generation 1.
        """
        result = _evaluate_without_objective_scores(tmp_path, mocker)

        frontier = ParetoFrontier(
            frontier_type=EvolutionConfig().frontier_type
        )
        frontier.add(_seed_candidate(tmp_path), result)

        assert frontier.get_non_dominated() == {SEED_ID}
        assert frontier._per_key_best == {"i0": {SEED_ID}, "i1": {SEED_ID}}

    def test_default_frontier_accepts_missing_objective_scores(
        self, tmp_path: Path
    ) -> None:
        """The instance frontier accepts a scalar-only evaluation."""
        result = EvalResult(
            candidate_id=SEED_ID,
            scores={"pass_rate": 0.75},
            asi={},
            instance_scores={"i0": 1.0, "i1": 0.5},
            objective_scores=None,
        )

        frontier = ParetoFrontier(
            frontier_type=EvolutionConfig().frontier_type
        )
        frontier.add(_seed_candidate(tmp_path), result)

        assert frontier.get_non_dominated() == {SEED_ID}


class TestExplicitMultiAxisMissingScores:
    """Explicit modes warn on missing scores and diverge only at selection.

    Hybrid can use its instance axis. Objective/cartesian cannot select a
    parent without an objective axis, so they keep a typed actionable error.
    """

    @pytest.mark.parametrize(
        "frontier_type", ["objective", "hybrid", "cartesian"]
    )
    def test_explicit_multi_axis_selection_behavior(
        self, tmp_path: Path, mocker: Any, frontier_type: str
    ) -> None:
        result = _evaluate_without_objective_scores(tmp_path, mocker)

        frontier = ParetoFrontier(frontier_type=frontier_type)  # type: ignore[arg-type]
        frontier.add(_seed_candidate(tmp_path), result)
        if frontier_type == "hybrid":
            assert frontier.select_parent().id == SEED_ID
        else:
            with pytest.raises(MissingObjectiveScoresError, match="no objective axis"):
                frontier.select_parent()
