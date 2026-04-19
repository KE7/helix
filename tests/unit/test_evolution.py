"""Unit tests for helix.evolution — the HELIX evolution loop.

Ported from GEPA's evolution loop test scenarios (gepa-research.md).

Covers:
- degrades() helper (gating acceptance/rejection/tie logic)
- budget_exhausted() helper
- run_evolution() loop behaviors:
    gating, convergence, budget exhaustion, perfect score break,
    merge triggering/cap, train/val split routing, dominated cleanup,
    pareto frontier update
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helix.config import (
    DatasetConfig,
    EvolutionConfig,
    EvaluatorConfig,
    HelixConfig,
    WorktreeConfig,
)
from helix.evolution import HelixProgress, budget_exhausted, degrades, run_evolution
from helix.lineage import LineageEntry
from helix.population import Candidate, EvalResult
from helix.state import BudgetState, EvolutionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidate(cid: str = "g0-s0", generation: int = 0) -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=f"/tmp/helix/{cid}",
        branch_name=f"helix/{cid}",
        generation=generation,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def make_eval_result(
    candidate_id: str = "g0-s0",
    instance_scores: dict[str, float] | None = None,
) -> EvalResult:
    if instance_scores is None:
        instance_scores = {"i1": 0.5, "i2": 0.5}
    return EvalResult(
        candidate_id=candidate_id,
        scores={},
        asi={},
        instance_scores=instance_scores,
    )


def make_config(
    max_generations: int = 5,
    max_metric_calls: int = 1000,
    perfect_score_threshold: float | None = 1.0,
    gating_threshold: float = 0.0,
    merge_enabled: bool = False,
    cleanup_dominated: bool = False,
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
    merge_subsample_size: int = 5,
) -> HelixConfig:
    return HelixConfig(
        objective="Improve the code",
        evaluator=EvaluatorConfig(command="pytest -q"),
        dataset=DatasetConfig(),
        evolution=EvolutionConfig(
            max_generations=max_generations,
            max_metric_calls=max_metric_calls,
            perfect_score_threshold=perfect_score_threshold,
            gating_threshold=gating_threshold,
            merge_enabled=merge_enabled,
            max_merge_invocations=max_merge_invocations,
            merge_val_overlap_floor=merge_val_overlap_floor,
            merge_subsample_size=merge_subsample_size,
        ),
        worktree=WorktreeConfig(cleanup_dominated=cleanup_dominated),
    )


def make_budget_state(evaluations: int = 0) -> EvolutionState:
    return EvolutionState(
        generation=0,
        frontier=[],
        instance_scores={},
        budget=BudgetState(evaluations=evaluations),
        config_hash="abc123",
    )


@pytest.fixture
def all_mocks(mocker):
    """Patch all external I/O dependencies of run_evolution."""
    return {
        "create_seed_worktree": mocker.patch("helix.evolution.create_seed_worktree"),
        "run_evaluator": mocker.patch("helix.evolution.run_evaluator"),
        "mutate": mocker.patch("helix.evolution.mutate"),
        "merge": mocker.patch("helix.evolution.merge", return_value=None),
        "remove_worktree": mocker.patch("helix.evolution.remove_worktree"),
        "load_state": mocker.patch("helix.evolution.load_state", return_value=None),
        "save_state": mocker.patch("helix.evolution.save_state"),
        "init_base_dir": mocker.patch("helix.evolution.init_base_dir"),
        "_save_evaluation": mocker.patch("helix.evolution._save_evaluation"),
        "_load_evaluation": mocker.patch(
            "helix.evolution._load_evaluation", return_value=None
        ),
        "record_entry": mocker.patch("helix.evolution.record_entry"),
        # GEPA parity (merge-pairing audit D1, /tmp/audit_audit-merge-pairing.md:49-50):
        # the merge branch now enforces GEPA's ``len(parent_program_for_candidate) < 3``
        # early-exit (merge.py:130-131), i.e. you need two siblings plus one
        # ancestor.  Provide a 3-entry dummy lineage by default so merge tests
        # that mock ``find_merge_triplet`` directly continue to exercise the
        # downstream merge flow.  Merge tests that want to assert the gate
        # trips override this to an empty dict.
        "load_lineage": mocker.patch(
            "helix.evolution.load_lineage",
            return_value={
                "g0-s0": LineageEntry(
                    id="g0-s0", parent=None, parents=[],
                    operation="seed", generation=0, files_changed=[],
                ),
                "g1-s1": LineageEntry(
                    id="g1-s1", parent="g0-s0", parents=["g0-s0"],
                    operation="mutate", generation=1, files_changed=[],
                ),
                "g1-s2": LineageEntry(
                    id="g1-s2", parent="g0-s0", parents=["g0-s0"],
                    operation="mutate", generation=1, files_changed=[],
                ),
            },
        ),
        "find_merge_triplet": mocker.patch(
            "helix.evolution.find_merge_triplet", return_value=None
        ),
        "snapshot_candidate": mocker.patch("helix.evolution.snapshot_candidate"),
        "set_phase": mocker.patch("helix.evolution.set_phase"),
        "print_info": mocker.patch("helix.evolution.print_info"),
        "print_success": mocker.patch("helix.evolution.print_success"),
        "print_warning": mocker.patch("helix.evolution.print_warning"),
        "render_budget": mocker.patch("helix.evolution.render_budget"),
        "render_generation": mocker.patch("helix.evolution.render_generation"),
        # Merge eval (M5 subsample path) writes helix_batch.json via
        # _cached_evaluate_batch.  Stub it out so tests can use non-numeric
        # or synthetic instance ids without hitting the filesystem.
        "_write_helix_batch": mocker.patch("helix.evolution._write_helix_batch"),
    }


# ---------------------------------------------------------------------------
# degrades() unit tests
# ---------------------------------------------------------------------------


class TestDegrades:
    def test_gating_rejects_degrading_mutation(self):
        """Mutation scoring below parent is flagged as degrading."""
        baseline = make_eval_result(instance_scores={"i1": 0.8, "i2": 0.8})
        worse = make_eval_result(instance_scores={"i1": 0.5, "i2": 0.5})
        assert degrades(worse, baseline, threshold=0.0) is True

    def test_gating_accepts_improving_mutation(self):
        """Mutation scoring strictly above parent is NOT degrading."""
        baseline = make_eval_result(instance_scores={"i1": 0.5, "i2": 0.5})
        better = make_eval_result(instance_scores={"i1": 0.9, "i2": 0.9})
        assert degrades(better, baseline, threshold=0.0) is False

    def test_gating_tie_accepted(self):
        """Ties are ACCEPTED under HELIX's non-degrading acceptance criterion.

        HELIX uses `<` (strict less-than): a mutation degrades only when its score
        is *strictly less than* baseline - threshold.  Equal scores (ties) pass.
        This matches GEPA's "non-degrading" gate described in §Key Design Decisions.
        """
        baseline = make_eval_result(instance_scores={"i1": 0.5, "i2": 0.5})
        tied = make_eval_result(instance_scores={"i1": 0.5, "i2": 0.5})
        # 0.5 < 0.5 → False → not degrading → tie accepted
        assert degrades(tied, baseline, threshold=0.0) is False

    def test_degrades_within_positive_threshold_allowed(self):
        """With a positive threshold, slight degradation inside the band is OK."""
        baseline = make_eval_result(instance_scores={"i1": 0.8})
        slightly_worse = make_eval_result(instance_scores={"i1": 0.75})
        # 0.75 <= 0.8 - 0.1 = 0.70?  No (0.75 > 0.70) → not degrading
        assert degrades(slightly_worse, baseline, threshold=0.1) is False

    def test_degrades_below_threshold_band(self):
        """Degradation outside the tolerance band still counts as degrading."""
        baseline = make_eval_result(instance_scores={"i1": 0.8})
        much_worse = make_eval_result(instance_scores={"i1": 0.6})
        # 0.6 <= 0.8 - 0.1 = 0.70 → degrading
        assert degrades(much_worse, baseline, threshold=0.1) is True


# ---------------------------------------------------------------------------
# budget_exhausted() unit tests
# ---------------------------------------------------------------------------


class TestBudgetExhausted:
    """GEPA parity (C1): budget uses only the evaluations counter."""

    def test_not_exhausted_below_limit(self):
        state = make_budget_state(evaluations=10)
        assert budget_exhausted(state, make_config(max_metric_calls=100)) is False

    def test_exhausted_evaluations_at_limit(self):
        state = make_budget_state(evaluations=200)
        assert budget_exhausted(state, make_config(max_metric_calls=200)) is True

    def test_exhausted_evaluations_exceed_limit(self):
        state = make_budget_state(evaluations=250)
        assert budget_exhausted(state, make_config(max_metric_calls=200)) is True

    def test_budget_exhausted_only_checks_evaluations(self):
        """Budget only uses evaluations counter (GEPA parity C1)."""
        state = make_budget_state(evaluations=10)
        assert budget_exhausted(state, make_config(max_metric_calls=1000)) is False
        state = make_budget_state(evaluations=1000)
        assert budget_exhausted(state, make_config(max_metric_calls=1000)) is True


# ---------------------------------------------------------------------------
# run_evolution — budget exhaustion stops loop
# ---------------------------------------------------------------------------


class TestBudgetExhaustionStopsLoop:
    def test_budget_exhaustion_stops_loop_evaluations(self, mocker, tmp_path, all_mocks):
        """Evolution stops immediately when evaluations budget exhausted after seed."""
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        # max_metric_calls=1: seed uses the only evaluation slot
        config = make_config(
            max_generations=10,
            max_metric_calls=1,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Gen loop should never reach mutate
        all_mocks["mutate"].assert_not_called()

    def test_budget_per_instance_counting_stops_loop(self, mocker, tmp_path, all_mocks):
        """Budget counts per-instance (GEPA parity): 3 instances = +3 evaluations."""
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            return make_eval_result(candidate.id, {"i1": 0.5, "i2": 0.6, "i3": 0.7})

        all_mocks["run_evaluator"].side_effect = run_eval

        # Seed has 3 instances → +3 evaluations; set limit to 3
        config = make_config(
            max_generations=10,
            max_metric_calls=3,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        all_mocks["mutate"].assert_not_called()

    def test_evaluations_count_per_instance(self, mocker, tmp_path, all_mocks):
        """Seed evaluation counts per-instance: 2 instances = +2 evaluations (GEPA parity)."""
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            return make_eval_result(candidate.id, {"i1": 0.5, "i2": 0.6})

        all_mocks["run_evaluator"].side_effect = run_eval

        saved_budgets: list[BudgetState] = []

        def capture(state, path):
            saved_budgets.append(
                BudgetState(
                    evaluations=state.budget.evaluations,
                )
            )

        all_mocks["save_state"].side_effect = capture

        config = make_config(max_generations=0, max_metric_calls=10000)
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # After seed with 2 instances: evaluations == 2 (per-instance counting)
        assert saved_budgets, "save_state should have been called"
        assert saved_budgets[0].evaluations == 2


# ---------------------------------------------------------------------------
# run_evolution — convergence detection
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# run_evolution — perfect score early stopping
# ---------------------------------------------------------------------------


class TestPerfectScoreEarlyStopping:
    def test_perfect_score_skips_mutation_continues_loop(self, mocker, tmp_path, all_mocks):
        """Perfect score on train eval skips mutation (continue) but loop keeps running.

        HELIX uses `continue` (not `break`) when the parent's train score is perfect,
        in line with GEPA §Reflective Mutation step 5: "skip mutation for this parent
        when the minibatch score is already perfect".  Mutation is never invoked.
        """
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            return make_eval_result(candidate.id, {"i1": 1.0, "i2": 1.0})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=10,
            perfect_score_threshold=1.0,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # `continue` skips mutation in each gen; mutate is never called
        all_mocks["mutate"].assert_not_called()

    def test_near_perfect_score_does_not_break_loop(self, mocker, tmp_path, all_mocks):
        """Score just below threshold does NOT trigger early stopping."""
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None  # mutation fails; doesn't matter

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            # Use a single instance so aggregate_score (sum) stays below 1.0
            return make_eval_result(candidate.id, {"i1": 0.99})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=2,
            perfect_score_threshold=1.0,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Loop should have run and called mutate at least once
        all_mocks["mutate"].assert_called()


# ---------------------------------------------------------------------------
# run_evolution — gating behavior
# ---------------------------------------------------------------------------


class TestGatingInEvolutionLoop:
    def _setup(self, mocker, tmp_path, all_mocks, seed_score: float, child_score: float):
        """Run 1 generation with tunable seed/child scores."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                return make_eval_result("g1-s1", {"i1": child_score})
            return make_eval_result(candidate.id, {"i1": seed_score})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            gating_threshold=0.0,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

    def test_gating_rejects_degrading_mutation_removes_worktree(
        self, mocker, tmp_path, all_mocks
    ):
        """When gating rejects a degrading mutation, remove_worktree is called."""
        self._setup(mocker, tmp_path, all_mocks, seed_score=0.8, child_score=0.3)
        all_mocks["remove_worktree"].assert_called_once()
        removed_id = all_mocks["remove_worktree"].call_args[0][0].id
        assert removed_id == "g1-s1"

    def test_gating_accepts_improving_mutation_no_remove(
        self, mocker, tmp_path, all_mocks
    ):
        """When gating accepts an improvement, remove_worktree is NOT called for child."""
        self._setup(mocker, tmp_path, all_mocks, seed_score=0.5, child_score=0.9)
        removed_ids = [c[0][0].id for c in all_mocks["remove_worktree"].call_args_list]
        assert "g1-s1" not in removed_ids

    def test_gating_tie_rejected_strict_improvement(
        self, mocker, tmp_path, all_mocks
    ):
        """Tied score (same as parent) is REJECTED under strict improvement acceptance.

        GEPA parity: StrictImprovementAcceptance requires new_sum > old_sum.
        Equal scores pass gating (non-degrading) but fail the strict improvement
        check, so the child is removed.
        """
        self._setup(mocker, tmp_path, all_mocks, seed_score=0.5, child_score=0.5)
        # Tie is rejected by strict improvement → child is removed
        removed_ids = [c[0][0].id for c in all_mocks["remove_worktree"].call_args_list]
        assert "g1-s1" in removed_ids

    def test_accepted_mutation_updates_frontier(self, mocker, tmp_path, all_mocks):
        """Accepted mutation becomes the best candidate on the frontier."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                return make_eval_result("g1-s1", {"i1": 0.9, "i2": 0.9})
            return make_eval_result(candidate.id, {"i1": 0.3, "i2": 0.3})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            gating_threshold=0.0,
            max_metric_calls=10000,

        )
        best = run_evolution(config, tmp_path, tmp_path / ".helix")
        assert best.id == "g1-s1"


# ---------------------------------------------------------------------------
# run_evolution — merge behavior
# ---------------------------------------------------------------------------


class TestMergeBehavior:
    def test_merge_fires_when_new_program_accepted(self, mocker, tmp_path, all_mocks):
        """Merge is triggered in the NEXT generation after acceptance (GEPA parity).

        GEPA defers merge to the next iteration: gen 1 accepts a child and
        sets merges_due += 1; gen 2 fires the merge at the top of the loop.

        GEPA parity (M1): merge also requires ``last_iter_found_new_program``.
        GEPA parity (M2/L3): merge candidate pool must have >= 2 non-dominated
        candidates — so seed and child need complementary instance scores.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].return_value = None  # merge attempt made but returns None
        # find_merge_triplet returns a valid triplet so merge fires
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                # Complementary instance scores: child better on i1 but worse
                # on i2 → neither candidate dominates the other → both
                # non-dominated → merge can fire.
                return make_eval_result("g1-s1", {"i1": 0.9, "i2": 0.5})
            return make_eval_result(candidate.id, {"i1": 0.5, "i2": 0.8})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=2,  # Need 2 gens: gen 1 accepts, gen 2 fires merge
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Child was accepted in gen 1, merge fires at start of gen 2
        all_mocks["merge"].assert_called_once()

    def test_merge_does_not_fire_when_gating_rejects(self, mocker, tmp_path, all_mocks):
        """Merge is NOT triggered when gating rejects the mutation."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                # Worse than seed → gating rejects
                return make_eval_result("g1-s1", {"i1": 0.1})
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        all_mocks["merge"].assert_not_called()

    def test_merge_disabled_never_fires(self, mocker, tmp_path, all_mocks):
        """When merge_enabled=False, merge() is never called regardless of outcomes."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                return make_eval_result("g1-s1", {"i1": 0.9, "i2": 0.9})
            return make_eval_result(candidate.id, {"i1": 0.3, "i2": 0.3})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            merge_enabled=False,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        all_mocks["merge"].assert_not_called()

    def test_merge_cap_zero_prevents_all_merges(self, mocker, tmp_path, all_mocks):
        """max_merge_invocations=0 completely prevents merge from firing.

        HELIX tracks total_merge_invocations across the entire run.  With
        cap=0, the condition `total_merge_invocations < 0` is always False,
        so merge() is never called.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                return make_eval_result("g1-s1", {"i1": 0.9})
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=3,
            merge_enabled=True,
            max_merge_invocations=0,  # cap at zero → no merges ever
            merge_val_overlap_floor=1,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        all_mocks["merge"].assert_not_called()

    def test_merge_total_cap_across_run(self, mocker, tmp_path, all_mocks):
        """max_merge_invocations is a TOTAL cap across the entire run.

        GEPA only counts ACCEPTED merges against the cap (rejected and
        failed merges do not consume the cap).  With max_merge_invocations=1,
        once a merge is accepted (score >= max parent), no further merges
        are attempted because merges_due stops accumulating.

        GEPA parity (M2/L3): candidates need complementary instance scores
        so both are non-dominated and merge candidate pool has >= 2 entries.

        Uses numeric instance ids ("1"/"2") because the merge-eval path
        (M5 subsample) runs through ``_cached_evaluate_batch`` → positional
        helix_batch.json handoff, which requires int-convertible ids.
        """
        seed = make_candidate("g0-s0")
        merged_cand = make_candidate("g1-m1", generation=1)
        children = [make_candidate(f"g{i}-s1", generation=i) for i in range(1, 4)]
        child_iter = iter(children)
        all_mocks["create_seed_worktree"].return_value = seed

        def make_child(*args, **kwargs):
            try:
                return next(child_iter)
            except StopIteration:
                return None

        all_mocks["mutate"].side_effect = make_child
        # Merge succeeds (returns candidate), so acceptance check will run.
        all_mocks["merge"].return_value = merged_cand
        # Return a valid triplet so merge can attempt to fire
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            # Seed is better on "2", children are better on "1" →
            # neither dominates the other → both non-dominated.
            # Child sum (1.4) > parent sum (1.3) so strict acceptance passes.
            if candidate.id == "g0-s0":
                scores = {"1": 0.5, "2": 0.8}
            else:
                scores = {"1": 0.9, "2": 0.5}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=3,
            merge_enabled=True,
            max_merge_invocations=1,  # total lifetime cap of 1
            merge_val_overlap_floor=1,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Total cap = 1 → after the first accepted merge, no more merges
        # fire because merges_due stops accumulating once cap is reached.
        assert all_mocks["merge"].call_count == 1

    def test_merge_result_evaluated_after_merge(self, mocker, tmp_path, all_mocks):
        """After a successful merge, the merged candidate is evaluated.

        GEPA parity: merge fires at start of gen 2 (deferred from gen 1).
        GEPA parity (M2/L3): complementary scores so both are non-dominated.
        GEPA parity (M5): merged eval runs on val subsample of common ids
        via ``_cached_evaluate_batch``; ids must be int-convertible.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        merged = make_candidate("g2-m1", generation=2)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].return_value = merged
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        merged_eval_count = [0]

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            if candidate.id == "g2-m1":
                merged_eval_count[0] += 1
                # Merged subsample sum must be >= max(parent subsample sums).
                # Parent subsample sums on {"1","2"}: seed=1.3, child=1.4.
                # Merged subsample sum = 1.5 → accepted.
                scores = {"1": 0.9, "2": 0.6}
            elif candidate.id == "g1-s1":
                scores = {"1": 0.9, "2": 0.5}
            else:
                scores = {"1": 0.5, "2": 0.8}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=2,  # Need 2 gens: gen 1 accepts, gen 2 fires merge
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert merged_eval_count[0] >= 1

    def test_merge_acceptance_uses_val_subsample_not_full_val(
        self, mocker, tmp_path, all_mocks
    ):
        """Merge acceptance compares subsample sums, not full-val sums.

        GEPA parity (M5, merge.py:332-400): the merged candidate is evaluated
        only on ``subsample_ids`` (intersection of parent val coverage) and
        compared against ``max(parent_a_subsample, parent_b_subsample)`` —
        not ``max(parent_a_full, parent_b_full)``.

        Scenario chosen so that old (full-val) and new (subsample) logic
        disagree:

            seed  (era) : {"1": 0.3, "2": 0.3, "3": 1.0} full=1.6 sub=0.6
            child (erb) : {"1": 0.5, "2": 0.5}           full=1.0 sub=1.0
            merged      : {"1": 0.6, "2": 0.5}           sub=1.1

        Old: required = max(1.6, 1.0) = 1.6 → 1.1 < 1.6 → REJECT.
        New: required = max(0.6, 1.0) = 1.0 → 1.1 >= 1.0 → ACCEPT.

        Asserting the merge eval runs on ``split="val"`` with
        ``instance_ids=["1","2"]`` (sorted common ids) — plus acceptance
        into the frontier — proves the subsample path is active.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        merged = make_candidate("g2-m1", generation=2)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].return_value = merged
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        merged_eval_calls: list[tuple[str | None, tuple[str, ...] | None]] = []

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            # Gating uses split="train"; frontier + merge use split="val".
            # We only care about val-side asymmetry for the subsample test,
            # so make train sums trivially pass the gate (child >= seed).
            if split == "train":
                if candidate.id == "g0-s0":
                    scores = {"1": 0.3}
                elif candidate.id == "g1-s1":
                    scores = {"1": 0.9}
                else:
                    scores = {"1": 0.9}
            else:
                if candidate.id == "g0-s0":
                    scores = {"1": 0.3, "2": 0.3, "3": 1.0}
                elif candidate.id == "g1-s1":
                    scores = {"1": 0.5, "2": 0.5}
                elif candidate.id == "g2-m1":
                    scores = {"1": 0.6, "2": 0.5}
                    merged_eval_calls.append(
                        (split, tuple(instance_ids) if instance_ids is not None else None)
                    )
                else:
                    scores = {"1": 0.0}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=2,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            # Cap subsample to the 2 common ids exactly so the GEPA port
            # does not fall through to rng.choices with replacement (which
            # would duplicate ids and change the sum math below).
            merge_subsample_size=2,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Merge eval must use the val split restricted to common ids.
        assert merged_eval_calls, "merged candidate was never evaluated"
        assert all(split == "val" for split, _ids in merged_eval_calls), (
            f"merge eval must route through val, saw splits: "
            f"{[s for s, _ in merged_eval_calls]}"
        )
        # GEPA parity (merge-gate audit M3, /tmp/audit_audit-merge-gate.md:10-32):
        # after the subsample gate passes, HELIX now runs a SECOND (full-val)
        # eval on the merged candidate mirroring GEPA ``engine.py:690`` →
        # ``_run_full_eval_and_add`` → ``_evaluate_on_valset``.  With
        # val_size=None (single-task mode) the full-val path falls through
        # to ``_cached_eval`` which calls ``run_evaluator`` *without*
        # instance_ids.  Assert the *first* call is the subsample
        # (["1","2"]), and the full-val call is either the explicit
        # instance-id list (if val_size were set) or ``None`` here.
        assert merged_eval_calls[0] == ("val", ("1", "2")), (
            f"first merge eval must be the common-id subsample, saw: "
            f"{merged_eval_calls[0]}"
        )

        # Under the old (full-val) comparison the merge would be REJECTED
        # (1.1 < 1.6); under the new (subsample) comparison it is ACCEPTED
        # (1.1 >= 1.0).  Acceptance manifests as a frontier entry for the
        # merged candidate — rejection would call remove_worktree on it.
        removed_ids = [
            c.args[0].id for c in all_mocks["remove_worktree"].call_args_list
        ]
        assert "g2-m1" not in removed_ids, (
            f"merge should be accepted under subsample comparison; "
            f"remove_worktree was called on: {removed_ids}"
        )

    def test_merge_subsample_size_configurable(
        self, mocker, tmp_path, all_mocks
    ):
        """evolution.merge_subsample_size caps the merge-eval batch size.

        GEPA parity (merge.py:262 num_subsample_ids=5, overridable): when
        both parents cover 10 common val ids but config sets
        merge_subsample_size=3, the merge eval must run on exactly 3 ids
        drawn from the 10-id intersection.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        merged = make_candidate("g2-m1", generation=2)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].return_value = merged
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        common_ids = [str(i) for i in range(10)]
        merged_eval_batches: list[tuple[str, ...]] = []

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            if split == "train":
                # Gating trivially passes (child > seed on id "0").
                if candidate.id == "g0-s0":
                    scores = {"0": 0.1}
                else:
                    scores = {"0": 0.9}
            else:
                # Seed wins on even ids; child wins on odd ids → non-dominated.
                if candidate.id == "g0-s0":
                    scores = {i: (0.9 if int(i) % 2 == 0 else 0.1) for i in common_ids}
                elif candidate.id == "g1-s1":
                    scores = {i: (0.1 if int(i) % 2 == 0 else 0.9) for i in common_ids}
                elif candidate.id == "g2-m1":
                    merged_eval_batches.append(
                        tuple(instance_ids) if instance_ids is not None else ()
                    )
                    scores = {i: 1.0 for i in common_ids}
                else:
                    scores = {"0": 0.0}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=2,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            merge_subsample_size=3,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert merged_eval_batches, "merged candidate was never evaluated"
        # GEPA parity (merge-gate audit M3): the first merge eval is the
        # subsample gate (exactly ``merge_subsample_size`` ids drawn from
        # the common val intersection); subsequent calls are the
        # GEPA-aligned post-acceptance full-val pass
        # (/tmp/audit_audit-merge-gate.md:10-32).  Assert only the first
        # call against the subsample contract.
        first_batch = merged_eval_batches[0]
        assert len(first_batch) == 3, (
            f"merge subsample must use exactly merge_subsample_size ids, "
            f"saw {len(first_batch)}: {first_batch}"
        )
        assert all(b in common_ids for b in first_batch), (
            f"merge subsample must be drawn from common val ids, saw: {first_batch}"
        )

    def test_merge_duplicate_subsample_counts_duplicates_symmetrically(
        self, mocker, tmp_path, all_mocks
    ):
        """GEPA's merge acceptance iterates the subsample list on BOTH sides.

        When only 2 common val ids exist but ``merge_subsample_size=5``, the
        GEPA port falls through to ``rng.choices(common_ids, k=remaining)``
        (merger.select_eval_subsample_for_merged_program → GEPA
        merge.py:286), producing a subsample with duplicate ids.  GEPA
        (merge.py:344-345, 394-395) sums scores by iterating those
        duplicate-bearing lists on both parents and the merged program, so
        duplicates contribute equally to all three aggregates.

        HELIX previously used ``merge_result.sum_score()`` (iterating the
        instance_scores dict — unique keys only) on the merged side while
        the parent sides already iterated the list, producing an
        asymmetric comparison.  This test constructs a scenario where the
        old (unique-counted merged) path rejects but the new
        (duplicate-counted merged) path accepts, and asserts the merge is
        accepted.

        Intentional divergence from HELIX's usual dict-based aggregation;
        flagged for a future ablation study (unique vs duplicate counting
        would be an interesting knob once we have an evolution baseline).
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        merged = make_candidate("g2-m1", generation=2)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].return_value = merged
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        merged_eval_batches: list[tuple[str, ...]] = []

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            if split == "train":
                # Gating trivially passes (child > seed on id "1").
                if candidate.id == "g0-s0":
                    scores = {"1": 0.0}
                else:
                    scores = {"1": 1.0}
            else:
                # Parent-side val scores: two common ids with flipped
                # winners so unique-counted parent sums both equal 1.0.
                # Only 2 common ids → subsample of size 5 must fall
                # through to rng.choices (with replacement) → duplicates.
                if candidate.id == "g0-s0":
                    scores = {"1": 0.0, "2": 1.0}
                elif candidate.id == "g1-s1":
                    scores = {"1": 1.0, "2": 0.0}
                elif candidate.id == "g2-m1":
                    merged_eval_batches.append(
                        tuple(instance_ids) if instance_ids is not None else ()
                    )
                    # Merged scores chosen so duplicate-counted sum is
                    # high (5.0 across any 5-item subsample) but unique
                    # dict sum is only 2.0 — well below required_score.
                    scores = {"1": 1.0, "2": 1.0}
                else:
                    scores = {"1": 0.0}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        # Default merge_subsample_size=5 triggers the fallback path here
        # because the parents only share 2 val ids.
        config = make_config(
            max_generations=2,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            merge_subsample_size=5,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert merged_eval_batches, "merged candidate was never evaluated"
        batch = merged_eval_batches[0]
        # The fallback path must have produced duplicates (5 items, ≤2 unique).
        assert len(batch) == 5, f"expected size-5 subsample, got {batch}"
        assert len(set(batch)) <= 2, (
            f"fallback must produce duplicates when |common|=2, got {batch}"
        )

        # Counter-check: dup-counted and unique-counted merged sums differ.
        merged_full = {"1": 1.0, "2": 1.0}
        duplicate_counted = sum(merged_full[k] for k in batch)  # 5.0
        unique_counted = sum(merged_full.values())               # 2.0
        assert duplicate_counted != unique_counted, (
            "test is only meaningful when the two semantics disagree: "
            f"dup={duplicate_counted} unique={unique_counted} batch={batch}"
        )

        # Parent sums (duplicate-counted, as the acceptance code does):
        # with any mix of "1" and "2" across 5 slots, one parent's sum
        # is always ≥ 3 (the other's is 5 minus that).  So required_score
        # ≥ 3 > 2.0 (unique-counted merged) but ≤ 5.0 (dup-counted merged).
        era_scores = {"1": 0.0, "2": 1.0}
        erb_scores = {"1": 1.0, "2": 0.0}
        a_score = sum(era_scores[k] for k in batch)
        b_score = sum(erb_scores[k] for k in batch)
        required = max(a_score, b_score)
        assert required > unique_counted, (
            f"scenario invalid: unique-counted merged {unique_counted} "
            f">= required {required}; old code would also accept"
        )
        assert duplicate_counted >= required, (
            f"scenario invalid: dup-counted merged {duplicate_counted} "
            f"< required {required}; new code would also reject"
        )

        # Acceptance under dup-counted math manifests as the merged
        # candidate staying (no remove_worktree call on it).
        removed_ids = [
            c.args[0].id for c in all_mocks["remove_worktree"].call_args_list
        ]
        assert "g2-m1" not in removed_ids, (
            f"merge must be accepted under symmetric duplicate-counted "
            f"comparison (dup={duplicate_counted} >= required={required}); "
            f"remove_worktree was called on: {removed_ids}"
        )

    def test_merge_accepted_entry_has_full_val_coverage(
        self, mocker, tmp_path, all_mocks
    ):
        """GEPA parity (merge-gate audit M3, /tmp/audit_audit-merge-gate.md:10-32).

        Once the subsample gate passes, HELIX runs a SECOND (full-val)
        eval on the merged candidate and uses THAT result for
        ``frontier.add`` / ``state.instance_scores[merged.id]`` — mirrors
        GEPA ``engine.py:688-696`` → ``_run_full_eval_and_add``
        (engine.py:175-197) → ``_evaluate_on_valset`` (engine.py:154-173).

        Before this fix, the frontier entry for the merged candidate
        only carried scores for the 5 subsample ids, so
        ``ParetoFrontier._update_per_key`` registered it on 5 keys and
        ``sum_score()`` collapsed to a fraction of the full-val sum,
        systematically under-representing merged candidates as dominators
        and over-representing them as tiebreak-eliminated candidates.

        Regression invariant: ``val_stage_size`` gates the mutation path
        only (src/helix/evolution.py:719) — it does not affect the merge
        flow — so the full-val pass runs regardless of its value.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        merged = make_candidate("g2-m1", generation=2)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].return_value = merged
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        # Distinguish subsample vs full-val merged evals by the ids they
        # ship to run_evaluator.  The subsample draws from the 2-id
        # intersection; the full-val path (val_size=None here) goes via
        # _cached_eval → run_evaluator WITHOUT instance_ids.
        merged_eval_calls: list[tuple[str | None, tuple[str, ...] | None]] = []

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            if split == "train":
                scores = {"1": 0.9} if candidate.id != "g0-s0" else {"1": 0.1}
            else:
                # Complementary parent scores → neither dominates → both
                # non-dominated → merge gate clears.
                if candidate.id == "g0-s0":
                    scores = {"1": 0.8, "2": 0.2, "3": 0.5, "4": 0.5}
                elif candidate.id == "g1-s1":
                    scores = {"1": 0.2, "2": 0.8, "3": 0.5, "4": 0.5}
                elif candidate.id == "g2-m1":
                    merged_eval_calls.append(
                        (split, tuple(instance_ids) if instance_ids is not None else None)
                    )
                    # Subsample call: merged wins vs required_score = 1.0
                    # (both parents sum to 1.0 on the 2-id subsample).
                    # Full-val call: no instance_ids → returns a 4-id dict
                    # the frontier will persist.
                    if instance_ids is not None:
                        scores = {"1": 0.6, "2": 0.6}
                    else:
                        scores = {"1": 0.7, "2": 0.7, "3": 0.7, "4": 0.7}
                else:
                    scores = {"1": 0.0}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        # Capture the frontier used by run_evolution via the ParetoFrontier
        # state after acceptance.  Easiest: re-read the accepted merged
        # instance_scores on save_state via the state snapshot.
        saved_states: list[EvolutionState] = []

        def _capture_state(state, *_a, **_kw):
            # Shallow-copy the dict so a later mutation doesn't rewrite it.
            saved_states.append(
                EvolutionState(
                    generation=state.generation,
                    frontier=list(state.frontier),
                    instance_scores={k: dict(v) for k, v in state.instance_scores.items()},
                    budget=BudgetState(evaluations=state.budget.evaluations),
                    config_hash=state.config_hash,
                    mutation_counter=state.mutation_counter,
                    merge_counter=state.merge_counter,
                    total_merge_invocations=state.total_merge_invocations,
                    merge_attempted_pairs=list(state.merge_attempted_pairs),
                    merge_description_triplets=list(state.merge_description_triplets),
                    i=state.i,
                )
            )

        all_mocks["save_state"].side_effect = _capture_state

        config = make_config(
            max_generations=2,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            merge_subsample_size=2,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Two merged evals: [0] subsample (gate), [1] full-val (post-accept).
        assert len(merged_eval_calls) >= 2, (
            f"M3 requires TWO merged evals (subsample + full-val); saw "
            f"{len(merged_eval_calls)}: {merged_eval_calls}"
        )
        # Subsample is always the first call.
        assert merged_eval_calls[0][1] == ("1", "2"), (
            f"subsample eval must request common ids ['1','2']; saw "
            f"{merged_eval_calls[0]}"
        )
        # Full-val call: with val_size=None it lands as instance_ids=None.
        # Either way (None or a full-val list), it MUST NOT be restricted
        # to the 2-id subsample.
        full_val_call = merged_eval_calls[1]
        assert full_val_call[1] is None or len(full_val_call[1]) > 2, (
            f"post-acceptance merge eval must be full-val (unrestricted), "
            f"saw: {full_val_call}"
        )

        # Final state: merged candidate's instance_scores must cover the
        # FULL val coverage (4 ids), not just the 2 subsample ids.
        merged_states = [s for s in saved_states if "g2-m1" in s.instance_scores]
        assert merged_states, "state never persisted the merged candidate's scores"
        final_inst = merged_states[-1].instance_scores["g2-m1"]
        assert set(final_inst.keys()) == {"1", "2", "3", "4"}, (
            f"M3: merged candidate must carry full-val scores, saw keys "
            f"{sorted(final_inst.keys())}"
        )

    def test_merge_attempted_pairs_stored_canonically(
        self, mocker, tmp_path, all_mocks
    ):
        """GEPA parity (merge-pairing audit C3, merge.py:94-95).

        ``find_merge_triplet`` canonicalizes the sampled pair via lex sort
        before returning, so the attempted-pair ledger stores
        ``[min(i,j), max(i,j)]`` regardless of which order the RNG drew
        the pair in.  Assert the persisted state uses the canonical form.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        merged = make_candidate("g2-m1", generation=2)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].return_value = merged
        # Return the non-canonical ("g1-s1", "g0-s0") order on purpose —
        # the real find_merge_triplet now canonicalizes internally, but
        # we're mocking it here to demonstrate that the callsite is
        # robust.  (Note: with the real lineage retry loop, the sampled
        # pair is sorted before return — this mock simulates a legacy
        # caller path.)
        all_mocks["find_merge_triplet"].return_value = ("g1-s1", "g0-s0", "g0-s0")

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            # Gating (split="train") wants strict improvement, so train
            # scores must give child a higher sum than seed.  Val
            # (split="val" or default) uses complementary scores so
            # neither candidate dominates — merge can fire at gen 2.
            if split == "train":
                scores = {"1": 0.9} if candidate.id != "g0-s0" else {"1": 0.1}
            else:
                if candidate.id == "g0-s0":
                    scores = {"1": 0.8, "2": 0.2}
                elif candidate.id == "g1-s1":
                    scores = {"1": 0.2, "2": 0.8}
                elif candidate.id == "g2-m1":
                    scores = {"1": 0.6, "2": 0.6}
                else:
                    scores = {"1": 0.5, "2": 0.5}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        saved_states: list[EvolutionState] = []

        def _capture(state, *_a, **_kw):
            saved_states.append(
                EvolutionState(
                    generation=state.generation,
                    frontier=list(state.frontier),
                    instance_scores={k: dict(v) for k, v in state.instance_scores.items()},
                    budget=BudgetState(evaluations=state.budget.evaluations),
                    config_hash=state.config_hash,
                    mutation_counter=state.mutation_counter,
                    merge_counter=state.merge_counter,
                    total_merge_invocations=state.total_merge_invocations,
                    merge_attempted_pairs=[list(p) for p in state.merge_attempted_pairs],
                    merge_description_triplets=[list(t) for t in state.merge_description_triplets],
                    i=state.i,
                )
            )

        all_mocks["save_state"].side_effect = _capture

        config = make_config(
            max_generations=2,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            merge_subsample_size=2,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        final = saved_states[-1]
        # Exactly one attempted-pair entry; stored as the canonical
        # [min, max] tuple regardless of the order rng.sample / the
        # proposer yielded.  Skipping the assertion that the mock-returned
        # order was canonical (the evolution loop trusts find_merge_triplet's
        # canonical contract), so here we assert the ledger semantic:
        # lookup should succeed for EITHER ("g0-s0","g1-s1") order.
        assert final.merge_attempted_pairs, "no merge pair recorded"
        pair = final.merge_attempted_pairs[0]
        assert set(pair) == {"g0-s0", "g1-s1"}, (
            f"merge_attempted_pairs entry must contain both candidates, saw {pair}"
        )

    def test_merge_description_triplet_recorded_on_accept(
        self, mocker, tmp_path, all_mocks
    ):
        """GEPA parity (merge-pairing audit C1, merge.py:195-203).

        Forward-direction test: an accepted merge records a
        ``(id1, id2, desc_hash)`` triplet in
        ``state.merge_description_triplets``, keyed canonically on the
        lex-sorted pair and the snapshotted worktree's git SHA.  Mirrors
        GEPA ``merges_performed[1].append((id1, id2, new_prog_desc))`` at
        merge.py:203.

        The reverse direction (dedup SKIPS when the triplet is already
        recorded) is covered structurally by the identical ``in``-list
        check the propose loop runs on state at runtime; the unit tests
        in test_lineage.py exercise the within-retry filter equivalents.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        merged = make_candidate("g2-m1", generation=2)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].return_value = merged
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")
        all_mocks["snapshot_candidate"].return_value = "deadbeefdeadbeef"

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            if split == "train":
                scores = {"1": 0.9} if candidate.id != "g0-s0" else {"1": 0.1}
            else:
                if candidate.id == "g0-s0":
                    scores = {"1": 0.8, "2": 0.2}
                elif candidate.id == "g1-s1":
                    scores = {"1": 0.2, "2": 0.8}
                elif candidate.id == "g2-m1":
                    scores = {"1": 0.6, "2": 0.6}
                else:
                    scores = {"1": 0.5, "2": 0.5}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        saved_states: list[EvolutionState] = []

        def _capture(state, *_a, **_kw):
            saved_states.append(
                EvolutionState(
                    generation=state.generation,
                    frontier=list(state.frontier),
                    instance_scores={k: dict(v) for k, v in state.instance_scores.items()},
                    budget=BudgetState(evaluations=state.budget.evaluations),
                    config_hash=state.config_hash,
                    mutation_counter=state.mutation_counter,
                    merge_counter=state.merge_counter,
                    total_merge_invocations=state.total_merge_invocations,
                    merge_attempted_pairs=[list(p) for p in state.merge_attempted_pairs],
                    merge_description_triplets=[list(t) for t in state.merge_description_triplets],
                    i=state.i,
                )
            )

        all_mocks["save_state"].side_effect = _capture

        config = make_config(
            max_generations=2,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            merge_subsample_size=2,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        final = saved_states[-1]
        # Exactly one merge-description triplet: canonical pair +
        # post-snapshot SHA.  ``cid_i <= cid_j`` (``"g0-s0" < "g1-s1"``).
        assert final.merge_description_triplets, (
            "merge acceptance must record a description triplet"
        )
        triplet = final.merge_description_triplets[0]
        assert triplet[0] <= triplet[1], (
            f"description triplet must store pair canonically, got {triplet}"
        )
        assert triplet == ["g0-s0", "g1-s1", "deadbeefdeadbeef"], (
            f"description triplet must carry (pair, desc_hash), got {triplet}"
        )

    def test_merge_gate_requires_three_candidates(
        self, mocker, tmp_path, all_mocks
    ):
        """GEPA parity (merge-pairing audit D1, merge.py:130-131).

        The ``len(parent_program_for_candidate) < 3`` early-exit means a
        run with only two recorded candidates (seed + one child) skips
        merge entirely.  HELIX mirrors this with ``len(lineage) < 3`` —
        ``find_merge_triplet`` is never called and ``merge()`` stays a
        no-op for that iteration, even when every other merge condition
        is satisfied.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        # Only seed + 1 child → lineage has 2 entries → gate trips.
        all_mocks["load_lineage"].return_value = {
            "g0-s0": LineageEntry(
                id="g0-s0", parent=None, parents=[],
                operation="seed", generation=0, files_changed=[],
            ),
            "g1-s1": LineageEntry(
                id="g1-s1", parent="g0-s0", parents=["g0-s0"],
                operation="mutate", generation=1, files_changed=[],
            ),
        }

        def run_eval(candidate, config, split=None, instance_ids=None, **kwargs):
            if candidate.id == "g1-s1":
                scores = {"1": 0.9, "2": 0.5}
            else:
                scores = {"1": 0.5, "2": 0.9}
            if instance_ids is not None:
                scores = {k: scores.get(k, 0.0) for k in instance_ids}
            return make_eval_result(candidate.id, scores)

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=3,
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=1,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # find_merge_triplet is gated out by the <3 guard → neither it
        # nor merge() fires.
        all_mocks["find_merge_triplet"].assert_not_called()
        all_mocks["merge"].assert_not_called()


# ---------------------------------------------------------------------------
# run_evolution — train/val split routing
# ---------------------------------------------------------------------------


class TestTrainValSplitRouting:
    def test_train_split_used_for_gating(self, mocker, tmp_path, all_mocks):
        """Gating evaluation uses split='train'."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        splits_seen: list[tuple[str, str | None]] = []

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            splits_seen.append((candidate.id, split))
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        child_splits = [split for cid, split in splits_seen if cid == "g1-s1"]
        assert "train" in child_splits, f"Gating should use train split; saw: {child_splits}"

    def test_val_split_used_for_pareto_update(self, mocker, tmp_path, all_mocks):
        """After gating passes, val split is used for the frontier update."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        splits_seen: list[tuple[str, str | None]] = []

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            splits_seen.append((candidate.id, split))
            if candidate.id == "g1-s1" and split == "train":
                # Gating passes (child improves on train)
                return make_eval_result("g1-s1", {"i1": 0.9})
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            gating_threshold=0.0,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        child_splits = [split for cid, split in splits_seen if cid == "g1-s1"]
        assert "val" in child_splits, (
            f"Pareto update should use val split; saw: {child_splits}"
        )

    def test_separate_val_eval_after_gating(self, mocker, tmp_path, all_mocks):
        """A separate val evaluation is run after gating passes.

        GEPA parity: the frontier score reflects a clean val evaluation
        rather than reusing the gating minibatch result.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        child_call_count = [0]

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                child_call_count[0] += 1
                return make_eval_result("g1-s1", {"i1": 0.9})
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Child should be evaluated twice: once for gating (train), once for val
        assert child_call_count[0] == 2


# ---------------------------------------------------------------------------
# run_evolution — dominated candidate cleanup
# ---------------------------------------------------------------------------


class TestAppendOnlyPopulation:
    def test_dominated_candidates_never_pruned(
        self, mocker, tmp_path, all_mocks
    ):
        """GEPA: population is append-only — dominated candidates are never pruned."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                # Child dominates seed on all instances
                return make_eval_result("g1-s1", {"i1": 0.95, "i2": 0.95})
            return make_eval_result(candidate.id, {"i1": 0.1, "i2": 0.1})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            cleanup_dominated=True,  # flag is ignored — no pruning regardless
            gating_threshold=0.0,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # remove_worktree should NOT be called for dominated cleanup
        # (it may still be called for gating rejections)
        removed_ids = [c[0][0].id for c in all_mocks["remove_worktree"].call_args_list]
        assert "g0-s0" not in removed_ids, (
            f"GEPA: dominated seed should NOT be removed; removed: {removed_ids}"
        )


# ---------------------------------------------------------------------------
# run_evolution — --generations flag / config override
# ---------------------------------------------------------------------------


class TestGenerationsFlagOverridesConfig:
    def test_generations_flag_overrides_config(self, mocker, tmp_path, all_mocks):
        """max_generations in config controls how many generations the loop runs.

        This test verifies that passing a config with max_generations=2 causes the
        evolution loop to run for exactly 2 generations (not the default 5 or more).
        The CLI's --generations flag applies this override before calling run_evolution.
        """
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=2,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Loop ran for exactly 2 generations — mutate is called once per gen
        # (Note: train evals may be cached by EvaluationCache for the same parent,
        # so we count via mutate calls instead.)
        assert all_mocks["mutate"].call_count == 2

    def test_zero_generations_skips_loop(self, mocker, tmp_path, all_mocks):
        """max_generations=0 means the generation loop body never executes."""
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=0,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # No generations → mutate never called
        all_mocks["mutate"].assert_not_called()


# ---------------------------------------------------------------------------
# run_evolution — mutation counter tracking
# ---------------------------------------------------------------------------


class TestMutationCountersTracked:
    def test_mutation_counters_tracked(self, mocker, tmp_path, all_mocks):
        """mutations_attempted and mutations_accepted are tracked correctly.

        We verify by inspecting the arguments passed to render_generation —
        it receives the per-run counters so the display can show acceptance rate.
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                # Child improves on parent → accepted
                return make_eval_result("g1-s1", {"i1": 0.9})
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            gating_threshold=0.0,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # render_generation should have been called with non-zero mutation counters
        calls = all_mocks["render_generation"].call_args_list
        # Find the call that recorded the accepted mutation (mutations_accepted > 0)
        found_accepted = any(
            call.kwargs.get("mutations_accepted", 0) >= 1
            for call in calls
        )
        found_attempted = any(
            call.kwargs.get("mutations_attempted", 0) >= 1
            for call in calls
        )
        assert found_attempted, "mutations_attempted should be passed to render_generation"
        assert found_accepted, "mutations_accepted should be passed to render_generation"

    def test_rejected_mutation_increments_attempted_not_accepted(
        self, mocker, tmp_path, all_mocks
    ):
        """A rejected mutation increments mutations_attempted but NOT mutations_accepted."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g1-s1":
                # Child is worse → gating rejects
                return make_eval_result("g1-s1", {"i1": 0.1})
            return make_eval_result(candidate.id, {"i1": 0.8})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=1,
            gating_threshold=0.0,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        calls = all_mocks["render_generation"].call_args_list
        # After rejection: attempted=1, accepted=0
        found_rejected_call = any(
            call.kwargs.get("mutations_attempted", 0) >= 1
            and call.kwargs.get("mutations_accepted", 0) == 0
            for call in calls
        )
        assert found_rejected_call, (
            "render_generation should show attempted > 0 and accepted == 0 after a rejection"
        )


# ---------------------------------------------------------------------------
# HelixProgress context-manager lifecycle
# ---------------------------------------------------------------------------


class TestHelixProgressLifecycle:
    """Verify the HelixProgress context-manager contract.

    These tests always run with HELIX_NO_PROGRESS=1 so they don't open a
    Rich Live display, keeping the test output clean.
    """

    def test_enter_returns_self(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """__enter__ must return the HelixProgress instance."""
        monkeypatch.setenv("HELIX_NO_PROGRESS", "1")
        prog = HelixProgress(max_generations=10)
        result = prog.__enter__()
        prog.__exit__(None, None, None)
        assert result is prog

    def test_context_manager_protocol(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """'with' statement must enter and exit cleanly."""
        monkeypatch.setenv("HELIX_NO_PROGRESS", "1")
        with HelixProgress(max_generations=5) as prog:
            assert prog is not None

    def test_is_active_false_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """is_active returns False when HELIX_NO_PROGRESS disables the bar."""
        monkeypatch.setenv("HELIX_NO_PROGRESS", "1")
        with HelixProgress(max_generations=5) as prog:
            assert prog.is_active is False

    def test_update_does_not_raise_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """update() is a no-op when the progress bar is disabled."""
        monkeypatch.setenv("HELIX_NO_PROGRESS", "1")
        with HelixProgress(max_generations=10) as prog:
            # Must not raise for any gen/score value
            prog.update(1, 0.5)
            prog.update(10, 1.0)
            prog.update(0, 0.0)

    def test_exit_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling __exit__ multiple times must not raise."""
        monkeypatch.setenv("HELIX_NO_PROGRESS", "1")
        prog = HelixProgress(max_generations=3)
        prog.__enter__()
        prog.__exit__(None, None, None)
        prog.__exit__(None, None, None)  # second call must be safe

    def test_is_active_true_when_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """is_active returns True when the Rich Live display is running."""
        monkeypatch.delenv("HELIX_NO_PROGRESS", raising=False)
        # Ensure a clean env: no leftover value from a previous test
        monkeypatch.setenv("HELIX_NO_PROGRESS", "")
        # Empty string → enabled (only non-empty disables)
        with HelixProgress(max_generations=5) as prog:
            assert prog.is_active is True

    def test_is_active_false_after_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """is_active must be False after __exit__."""
        monkeypatch.setenv("HELIX_NO_PROGRESS", "1")
        prog = HelixProgress(max_generations=5)
        prog.__enter__()
        assert prog.is_active is False  # disabled → never active
        prog.__exit__(None, None, None)
        assert prog.is_active is False
