"""Unit tests for the multi-axis ``ParetoFrontier`` — GEPA ``FrontierType`` parity.

Pins the per-axis state accumulation (``_per_key_best``,
``_objective_best``, ``_cartesian_best``) and the dispatch-on-
``frontier_type`` behaviour of :meth:`ParetoFrontier._active_frontier`,
:meth:`get_non_dominated`, and :meth:`select_parent`.

Cross-references:
  * GEPA's ``FrontierType`` literal, defined in ``core/state.py``.
  * ``_update_objective_pareto_front`` in ``core/state.py``.
  * ``_update_pareto_front_for_cartesian`` in ``core/state.py``.
  * O.A. default ``frontier_type="hybrid"``: GEPA's
    ``EngineConfig.frontier_type`` in ``gepa_launcher.py`` — rationale
    for HELIX's own default (``evolution.frontier_type``).

The acceptance gate is **not** tested here: it remains positional on
raw score lists regardless of ``frontier_type`` (GEPA's
``StrictImprovementAcceptance.should_accept`` in ``strategies/acceptance.py``).
"""

from __future__ import annotations

import random

import pytest

from helix.population import (
    Candidate,
    EvalResult,
    MissingObjectiveScoresError,
    ParetoFrontier,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(cid: str) -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=f"/tmp/{cid}",
        branch_name=f"branch-{cid}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="mutation",
    )


def _make_result(
    cid: str,
    instance_scores: dict[str, float],
    objective_scores: list[dict[str, float]] | None = None,
) -> EvalResult:
    return EvalResult(
        candidate_id=cid,
        scores={},
        asi={},
        instance_scores=instance_scores,
        objective_scores=objective_scores,
    )


# ---------------------------------------------------------------------------
# Default frontier_type back-compat
# ---------------------------------------------------------------------------


class TestDefaultFrontierTypeBackCompat:
    """The default ``ParetoFrontier(rng=...)`` path stays
    ``frontier_type="instance"`` to preserve every existing test
    exercising ``_per_key_best`` directly."""

    def test_default_is_instance(self):
        frontier = ParetoFrontier()
        assert frontier.frontier_type == "instance"

    def test_instance_active_frontier_is_per_key_best(self):
        frontier = ParetoFrontier()
        frontier.add(_make_candidate("a"), _make_result("a", {"i1": 1.0}))
        # Active frontier for "instance" is the raw _per_key_best dict
        # (same object), so legacy code reading _per_key_best keeps working.
        assert frontier._active_frontier() is frontier._per_key_best


# ---------------------------------------------------------------------------
# Objective-axis accumulation + active frontier
# ---------------------------------------------------------------------------


class TestObjectiveFrontier:
    def test_objective_best_tracks_mean_across_valset(self):
        """``_objective_best_score[obj]`` = max over candidates of
        mean(obj-score across valset), mirroring GEPA's
        ``_update_objective_pareto_front``.
        """
        frontier = ParetoFrontier(frontier_type="objective")
        # Candidate a: obj_alpha mean = (0.8+0.2)/2 = 0.5
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i0": 1.0, "i1": 0.0},
                objective_scores=[
                    {"obj_alpha": 0.8}, {"obj_alpha": 0.2},
                ],
            ),
        )
        # Candidate b: obj_alpha mean = (0.7+0.7)/2 = 0.7 → beats a.
        frontier.add(
            _make_candidate("b"),
            _make_result(
                "b", {"i0": 0.5, "i1": 0.5},
                objective_scores=[
                    {"obj_alpha": 0.7}, {"obj_alpha": 0.7},
                ],
            ),
        )

        assert frontier._objective_best_score["obj_alpha"] == pytest.approx(0.7)
        assert frontier._objective_best["obj_alpha"] == {"b"}

    def test_objective_best_tie_expands_set(self):
        frontier = ParetoFrontier(frontier_type="objective")
        frontier.add(
            _make_candidate("a"),
            _make_result("a", {"i0": 1.0}, objective_scores=[{"obj": 0.5}]),
        )
        frontier.add(
            _make_candidate("b"),
            _make_result("b", {"i0": 1.0}, objective_scores=[{"obj": 0.5}]),
        )
        assert frontier._objective_best["obj"] == {"a", "b"}

    def test_multiple_objectives_tracked_independently(self):
        frontier = ParetoFrontier(frontier_type="objective")
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i0": 1.0, "i1": 0.0},
                objective_scores=[
                    {"latency": 40.0, "accuracy": 0.9},
                    {"latency": 60.0, "accuracy": 0.1},
                ],
            ),
        )
        frontier.add(
            _make_candidate("b"),
            _make_result(
                "b", {"i0": 0.5, "i1": 0.5},
                objective_scores=[
                    {"latency": 20.0, "accuracy": 0.5},
                    {"latency": 20.0, "accuracy": 0.5},
                ],
            ),
        )
        # All axes are higher-is-better (GEPA parity: the framework
        # maximizes).  latency means: a=50, b=20 → a wins.
        # accuracy means: a=0.5, b=0.5 → tie.
        assert frontier._objective_best["latency"] == {"a"}
        assert frontier._objective_best["accuracy"] == {"a", "b"}

    @pytest.mark.parametrize("frontier_type", ["objective", "hybrid", "cartesian"])
    def test_missing_objective_scores_warns_and_is_tolerated(
        self, frontier_type, caplog
    ):
        frontier = ParetoFrontier(frontier_type=frontier_type)
        frontier.add(
            _make_candidate("a"),
            _make_result("a", {"i0": 1.0}, objective_scores=None),
        )
        assert "no objective axes" in caplog.text

    @pytest.mark.parametrize("frontier_type", ["objective", "hybrid", "cartesian"])
    def test_all_empty_objective_scores_exercise_selection(
        self, frontier_type, caplog
    ):
        """Measure the post-validation behavior for every non-instance mode.

        Hybrid retains its instance-axis parent pool. Objective/cartesian
        have no coherent fallback and retain HELIX's typed actionable error.
        """
        frontier = ParetoFrontier(frontier_type=frontier_type, rng=random.Random(0))
        frontier.add(
            _make_candidate("a"),
            _make_result("a", {"i0": 1.0}, objective_scores=[{}]),
        )
        assert "no objective axes" in caplog.text

        if frontier_type == "hybrid":
            assert frontier.active_frontier_snapshot() == {"inst::i0": ["a"]}
            assert frontier.select_parent().id == "a"
            assert frontier.get_non_dominated() == {"a"}
        else:
            assert frontier.active_frontier_snapshot() == {}
            assert frontier.get_non_dominated() == set()
            with pytest.raises(
                MissingObjectiveScoresError,
                match=(
                    rf"frontier_type={frontier_type!r} has no objective axis "
                    "and cannot select a parent"
                ),
            ):
                frontier.select_parent()

    def test_active_frontier_is_objective_dict(self):
        frontier = ParetoFrontier(frontier_type="objective")
        frontier.add(
            _make_candidate("a"),
            _make_result("a", {"i0": 1.0}, objective_scores=[{"obj": 0.5}]),
        )
        assert frontier._active_frontier() is frontier._objective_best

    def test_get_non_dominated_uses_objective_axis(self):
        """Candidate a dominates on obj_alpha only; candidate b on
        obj_beta only; both survive under frontier_type='objective'."""
        frontier = ParetoFrontier(frontier_type="objective", rng=random.Random(0))
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i": 1.0},
                objective_scores=[{"obj_alpha": 1.0, "obj_beta": 0.0}],
            ),
        )
        frontier.add(
            _make_candidate("b"),
            _make_result(
                "b", {"i": 1.0},
                objective_scores=[{"obj_alpha": 0.0, "obj_beta": 1.0}],
            ),
        )
        non_dom = frontier.get_non_dominated()
        assert non_dom == {"a", "b"}


# ---------------------------------------------------------------------------
# Cartesian-axis accumulation
# ---------------------------------------------------------------------------


class TestCartesianFrontier:
    def test_cartesian_keys_encode_val_id_and_objective(self):
        frontier = ParetoFrontier(frontier_type="cartesian")
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i0": 1.0, "i1": 0.5},
                objective_scores=[
                    {"obj_alpha": 0.8, "obj_beta": 0.1},
                    {"obj_alpha": 0.2, "obj_beta": 0.9},
                ],
            ),
        )
        # Keys are "{val_id}::{obj_name}".
        assert frontier._cartesian_best["i0::obj_alpha"] == {"a"}
        assert frontier._cartesian_best["i0::obj_beta"] == {"a"}
        assert frontier._cartesian_best["i1::obj_alpha"] == {"a"}
        assert frontier._cartesian_best["i1::obj_beta"] == {"a"}
        assert frontier._cartesian_best_score["i1::obj_beta"] == pytest.approx(0.9)

    def test_cartesian_per_cell_winner(self):
        """Different candidates can win different (val_id, obj) cells."""
        frontier = ParetoFrontier(frontier_type="cartesian")
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i0": 1.0, "i1": 0.5},
                objective_scores=[{"obj": 0.9}, {"obj": 0.1}],
            ),
        )
        frontier.add(
            _make_candidate("b"),
            _make_result(
                "b", {"i0": 0.5, "i1": 1.0},
                objective_scores=[{"obj": 0.1}, {"obj": 0.9}],
            ),
        )
        assert frontier._cartesian_best["i0::obj"] == {"a"}
        assert frontier._cartesian_best["i1::obj"] == {"b"}

    def test_length_mismatch_fails_loudly(self):
        """Defensive: if ``objective_scores`` length ≠ ``instance_scores``
        length (should not happen on the helix_result path), raise."""
        frontier = ParetoFrontier(frontier_type="cartesian")
        with pytest.raises(MissingObjectiveScoresError, match="length"):
            frontier.add(
                _make_candidate("a"),
                _make_result(
                    "a", {"i0": 1.0, "i1": 0.0},
                    objective_scores=[{"obj": 0.5}],  # len 1 vs ids len 2
                ),
            )

    def test_empty_objective_slots_are_tolerated(self, caplog):
        frontier = ParetoFrontier(frontier_type="hybrid")
        frontier.add(
            _make_candidate("a"),
            _make_result("a", {"i0": 1.0}, objective_scores=[{}]),
        )
        assert "no objective axes" in caplog.text

    def test_empty_objective_list_is_a_cartesian_noop(self, caplog):
        frontier = ParetoFrontier(frontier_type="cartesian")
        frontier.add(
            _make_candidate("a"),
            _make_result("a", {"i0": 1.0}, objective_scores=[]),
        )
        assert frontier.active_frontier_snapshot() == {}
        assert "no objective axes" in caplog.text

    def test_active_frontier_is_cartesian_dict(self):
        frontier = ParetoFrontier(frontier_type="cartesian")
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i0": 1.0},
                objective_scores=[{"obj": 0.5}],
            ),
        )
        assert frontier._active_frontier() is frontier._cartesian_best


# ---------------------------------------------------------------------------
# Hybrid — union of instance ∪ objective keyspaces
# ---------------------------------------------------------------------------


class TestHybridFrontier:
    def test_active_frontier_prefixes_both_keyspaces(self):
        frontier = ParetoFrontier(frontier_type="hybrid")
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i0": 1.0, "i1": 0.5},
                objective_scores=[{"obj": 0.8}, {"obj": 0.2}],
            ),
        )
        merged = frontier._active_frontier()
        # Instance keys are namespaced "inst::", objective keys "obj::".
        assert "inst::i0" in merged
        assert "inst::i1" in merged
        assert "obj::obj" in merged

    def test_active_frontier_snapshot_is_distinct_from_candidates(self):
        frontier = ParetoFrontier(frontier_type="hybrid")
        frontier.add(
            _make_candidate("a"),
            _make_result("a", {"i0": 1.0}, objective_scores=[{"obj": 0.1}]),
        )
        frontier.add(
            _make_candidate("b"),
            _make_result("b", {"i0": 0.0}, objective_scores=[{"obj": 0.9}]),
        )

        assert list(frontier._candidates) == ["a", "b"]
        assert frontier.active_frontier_snapshot() == {
            "inst::i0": ["a"],
            "obj::obj": ["b"],
        }

    def test_hybrid_survives_on_either_axis(self):
        """Candidate a wins an instance key but is dominated on objective;
        candidate b wins an objective but is dominated on instance.  Under
        frontier_type="hybrid" both survive — the union keyspace puts them
        on different fronts."""
        frontier = ParetoFrontier(frontier_type="hybrid", rng=random.Random(0))
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i0": 1.0, "i1": 0.0},
                objective_scores=[{"obj": 0.1}, {"obj": 0.1}],  # mean 0.1
            ),
        )
        frontier.add(
            _make_candidate("b"),
            _make_result(
                "b", {"i0": 0.0, "i1": 0.0},
                objective_scores=[{"obj": 0.9}, {"obj": 0.9}],  # mean 0.9
            ),
        )
        non_dom = frontier.get_non_dominated()
        # a wins inst::i0, b wins obj::obj — both survive.
        assert non_dom == {"a", "b"}

    def test_hybrid_dominates_on_both_axes(self):
        """A candidate that loses every instance key AND every objective
        is eliminated under hybrid."""
        frontier = ParetoFrontier(frontier_type="hybrid", rng=random.Random(0))
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i0": 1.0, "i1": 1.0},
                objective_scores=[{"obj": 1.0}, {"obj": 1.0}],
            ),
        )
        frontier.add(
            _make_candidate("b"),
            _make_result(
                "b", {"i0": 0.0, "i1": 0.0},
                objective_scores=[{"obj": 0.0}, {"obj": 0.0}],
            ),
        )
        assert frontier.get_non_dominated() == {"a"}
        assert frontier.is_dominated("b")


# ---------------------------------------------------------------------------
# select_parent dispatches on frontier_type
# ---------------------------------------------------------------------------


class TestSelectParentRespectsFrontierType:
    def test_objective_only_parent_pool(self):
        """With frontier_type='objective' and no surviving instance
        winners in the active frontier, ``select_parent`` picks from the
        objective-axis winners rather than falling back to
        instance-axis."""
        frontier = ParetoFrontier(frontier_type="objective", rng=random.Random(0))
        # Candidate a wins on instance but loses objective.
        frontier.add(
            _make_candidate("a"),
            _make_result("a", {"i0": 1.0}, objective_scores=[{"obj": 0.1}]),
        )
        # Candidate b is the sole objective-axis winner.
        frontier.add(
            _make_candidate("b"),
            _make_result(
                "b", {"i0": 0.0},
                objective_scores=[{"obj": 0.9}],
            ),
        )
        parent = frontier.select_parent()
        assert parent.id == "b"

    def test_rebuild_preserves_all_axes(self):
        """``update_scores`` triggers a rebuild that regenerates every
        axis from the (possibly updated) results."""
        frontier = ParetoFrontier(frontier_type="hybrid")
        frontier.add(
            _make_candidate("a"),
            _make_result(
                "a", {"i0": 0.5},
                objective_scores=[{"obj": 0.5}],
            ),
        )
        # Update a's result — bump both instance and objective scores.
        new_result = _make_result(
            "a", {"i0": 1.0},
            objective_scores=[{"obj": 1.0}],
        )
        frontier.update_scores(new_result)
        assert frontier._per_key_best_score["i0"] == 1.0
        assert frontier._objective_best_score["obj"] == 1.0
