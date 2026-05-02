"""End-to-end: toy evaluator → helix_result parser → EvalResult → ParetoFrontier.

Exercises the full pipeline for this branch's feature set:

  1. An evaluator emits a ``HELIX_RESULT=[[score, {"scores": {obj: v}}], ...]``
     per-example payload (the shape introduced by the ``helix_result``
     reshape in commit A).
  2. :func:`helix.executor.run_evaluator` parses it, populates
     :attr:`helix.population.EvalResult.instance_scores`,
     :attr:`EvalResult.per_example_side_info`, and the new
     :attr:`EvalResult.objective_scores` harvest.
  3. A :class:`helix.population.ParetoFrontier` constructed with each
     ``frontier_type`` variant tracks the expected per-axis state when
     the ``EvalResult`` is added.

The acceptance gate is **not** exercised here — it remains positional
on ``scores_list`` regardless of ``frontier_type``, and the e2e story
is about the multi-axis retention path.

GEPA cross-references:
  * Evaluator contract — per-example ``(score, side_info)``:
    ``src/gepa/optimize_anything.py:387-438``.
  * ``side_info["scores"]`` → ``objective_scores`` harvest:
    ``optimize_anything_adapter.py:260-272``.
  * ``FrontierType`` literal:
    ``src/gepa/core/state.py:22-23``.
  * O.A. default ``"hybrid"``:
    ``src/gepa/optimize_anything.py:476``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from helix.config import EvaluatorConfig, HelixConfig
from helix.executor import run_evaluator
from helix.population import Candidate, ParetoFrontier


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(cid: str, worktree_path: Path) -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=str(worktree_path),
        branch_name=f"branch-{cid}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="mutation",
    )


def _make_config() -> HelixConfig:
    return HelixConfig(
        objective="test",
        evaluator=EvaluatorConfig(
            command="python eval.py",
            score_parser="helix_result",
            include_stdout=True,
            include_stderr=True,
            extra_commands=[],
        ),
    )


def _mock_subprocess(stdout: str, returncode: int = 0) -> MagicMock:
    m = MagicMock()
    m.stdout = stdout
    m.stderr = ""
    m.returncode = returncode
    return m


def _write_batch(worktree: Path, ids: list[str]) -> None:
    (worktree / "helix_batch.json").write_text(json.dumps(ids))


def _toy_helix_result(
    scores: list[float], objective_dicts: list[dict[str, float]] | None = None
) -> str:
    """Build a per-example ``HELIX_RESULT=`` line.

    Each inner pair is ``[score_i, {"scores": objective_dicts[i]}]`` —
    the GEPA O.A. user-facing evaluator shape.
    """
    if objective_dicts is None:
        objective_dicts = [{} for _ in scores]
    assert len(scores) == len(objective_dicts)
    payload = [
        [s, {"scores": obj} if obj else {}]
        for s, obj in zip(scores, objective_dicts)
    ]
    return "HELIX_RESULT=" + json.dumps(payload)


# ---------------------------------------------------------------------------
# E2E: each frontier_type tracks the expected state
# ---------------------------------------------------------------------------


class TestE2EMultiAxisFrontier:
    """A toy evaluator emits per-example ``[score, {"scores": {obj: v}}]``
    pairs for three candidates with complementary strengths.  We then
    check each ``frontier_type`` path tracks the expected state.
    """

    def _setup_candidates(
        self, tmp_path: Path, mocker
    ) -> dict[str, Any]:  # type: ignore[valid-type]
        """Build three candidates (a, b, c) each evaluated on the same
        2-example valset.

        Scores:
          a — instance [1.0, 0.0], obj_fast={1.0, 0.0}, obj_safe={0.0, 1.0}
          b — instance [0.0, 1.0], obj_fast={0.0, 1.0}, obj_safe={1.0, 0.0}
          c — instance [0.5, 0.5], obj_fast={0.3, 0.3}, obj_safe={0.5, 0.5}

        Per-axis winners:
          * instance i0 → a (1.0); instance i1 → b (1.0) — c is dominated
            on instance axis.
          * obj_fast mean: a=0.5, b=0.5, c=0.3 — a and b tie.
          * obj_safe mean: a=0.5, b=0.5, c=0.5 — all three tie.

        Returns a dict of ``{cid: EvalResult}`` with
        ``objective_scores`` populated by the parser via
        ``side_info["scores"]`` harvest.
        """
        ids = ["i0", "i1"]
        _write_batch(tmp_path, ids)

        def _eval_for(
            scores_list: list[float],
            fast: list[float],
            safe: list[float],
        ):
            obj_dicts = [
                {"obj_fast": f, "obj_safe": s} for f, s in zip(fast, safe)
            ]
            return _toy_helix_result(scores_list, obj_dicts)

        per_candidate: dict[str, str] = {
            "a": _eval_for([1.0, 0.0], [1.0, 0.0], [0.0, 1.0]),
            "b": _eval_for([0.0, 1.0], [0.0, 1.0], [1.0, 0.0]),
            "c": _eval_for([0.5, 0.5], [0.3, 0.3], [0.5, 0.5]),
        }

        results: dict[str, Any] = {}  # type: ignore[valid-type]
        for cid, stdout in per_candidate.items():
            mocker.patch(
                "helix.executor.subprocess.run",
                return_value=_mock_subprocess(stdout, returncode=0),
            )
            cand = _make_candidate(cid, tmp_path)
            results[cid] = run_evaluator(
                cand, _make_config(), split="val",
            )
        return results

    def test_instance_path_tracks_per_example_winners(
        self, tmp_path: Path, mocker
    ):
        results = self._setup_candidates(tmp_path, mocker)

        frontier = ParetoFrontier(frontier_type="instance")
        for cid, r in results.items():
            frontier.add(_make_candidate(cid, tmp_path), r)

        # i0 best = 1.0 (a); i1 best = 1.0 (b); c is dominated on instance.
        assert frontier._per_key_best["i0"] == {"a"}
        assert frontier._per_key_best["i1"] == {"b"}
        assert frontier.get_non_dominated() == {"a", "b"}
        assert frontier.is_dominated("c")

    def test_objective_path_tracks_mean_across_valset(
        self, tmp_path: Path, mocker
    ):
        results = self._setup_candidates(tmp_path, mocker)

        frontier = ParetoFrontier(frontier_type="objective")
        for cid, r in results.items():
            frontier.add(_make_candidate(cid, tmp_path), r)

        # obj_fast means: a=0.5, b=0.5, c=0.3 — a and b tie for best.
        assert frontier._objective_best["obj_fast"] == {"a", "b"}
        assert frontier._objective_best_score["obj_fast"] == pytest.approx(0.5)
        # obj_safe means: all three at 0.5.
        assert frontier._objective_best["obj_safe"] == {"a", "b", "c"}

    def test_hybrid_path_unions_instance_and_objective(
        self, tmp_path: Path, mocker
    ):
        results = self._setup_candidates(tmp_path, mocker)

        frontier = ParetoFrontier(frontier_type="hybrid")
        for cid, r in results.items():
            frontier.add(_make_candidate(cid, tmp_path), r)

        # Active frontier is the union with prefixed keys.
        active = frontier._active_frontier()
        assert "inst::i0" in active and "inst::i1" in active
        assert "obj::obj_fast" in active and "obj::obj_safe" in active

        # Coverage-based dominance: c appears only on ``obj::obj_safe``
        # (tied with a and b).  Both a and b are also on that front, so
        # ``{a, b}`` covers every front c participates in → c is
        # dominated.  (Ties alone don't protect from coverage dominance
        # when the tying dominators cover c's only front.)  Non-dominated
        # set = {a, b}.  Under ``"instance"`` alone the result would be
        # the same; hybrid adds the objective keys but doesn't rescue c
        # here because a and b still cover every c-participating key.
        non_dom = frontier.get_non_dominated()
        assert non_dom == {"a", "b"}

    def test_cartesian_path_tracks_per_cell_winners(
        self, tmp_path: Path, mocker
    ):
        results = self._setup_candidates(tmp_path, mocker)

        frontier = ParetoFrontier(frontier_type="cartesian")
        for cid, r in results.items():
            frontier.add(_make_candidate(cid, tmp_path), r)

        # (i0, obj_fast): a=1.0, b=0.0, c=0.3 → a wins.
        assert frontier._cartesian_best["i0::obj_fast"] == {"a"}
        # (i1, obj_fast): a=0.0, b=1.0, c=0.3 → b wins.
        assert frontier._cartesian_best["i1::obj_fast"] == {"b"}
        # (i0, obj_safe): a=0.0, b=1.0, c=0.5 → b wins.
        assert frontier._cartesian_best["i0::obj_safe"] == {"b"}
        # (i1, obj_safe): a=1.0, b=0.0, c=0.5 → a wins.
        assert frontier._cartesian_best["i1::obj_safe"] == {"a"}

    def test_parser_wires_all_data_onto_eval_result(
        self, tmp_path: Path, mocker
    ):
        """End-to-end sanity: per-example side_info dicts (full raw),
        objective_scores harvest (just the "scores" sub-dict), and
        instance_scores (positional-to-ids zip) are all populated on
        the ``EvalResult`` returned by :func:`run_evaluator`."""
        ids = ["i0", "i1"]
        _write_batch(tmp_path, ids)
        payload = [
            [1.0, {"trajectory": "fast path", "scores": {"speed": 0.9, "cost": 0.1}}],
            [0.5, {"trajectory": "slow path", "scores": {"speed": 0.3, "cost": 0.5}}],
        ]
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=_mock_subprocess("HELIX_RESULT=" + json.dumps(payload)),
        )
        result = run_evaluator(
            _make_candidate("a", tmp_path), _make_config(), split="val",
        )

        assert result.instance_scores == {"i0": 1.0, "i1": 0.5}
        assert result.per_example_side_info == [
            {"trajectory": "fast path", "scores": {"speed": 0.9, "cost": 0.1}},
            {"trajectory": "slow path", "scores": {"speed": 0.3, "cost": 0.5}},
        ]
        assert result.objective_scores == [
            {"speed": 0.9, "cost": 0.1},
            {"speed": 0.3, "cost": 0.5},
        ]
