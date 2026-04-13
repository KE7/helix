"""Unit tests for helix.evolution — the HELIX evolution loop.

Ported from GEPA's evolution loop test scenarios (gepa-research.md).

Covers:
- degrades() helper (gating acceptance/rejection/tie logic)
- budget_exhausted() helper
- run_evolution() loop behaviors:
    gating, convergence, budget exhaustion, perfect score break,
    merge triggering/cap, dev/val split routing, dominated cleanup,
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
    convergence_patience: int = 5,
    perfect_score_threshold: float | None = 1.0,
    gating_threshold: float = 0.0,
    merge_enabled: bool = False,
    cleanup_dominated: bool = False,
    max_merge_invocations: int = 5,
    merge_val_overlap_floor: int = 5,
) -> HelixConfig:
    return HelixConfig(
        objective="Improve the code",
        evaluator=EvaluatorConfig(command="pytest -q"),
        dataset=DatasetConfig(),
        evolution=EvolutionConfig(
            max_generations=max_generations,
            max_metric_calls=max_metric_calls,
            convergence_patience=convergence_patience,
            perfect_score_threshold=perfect_score_threshold,
            gating_threshold=gating_threshold,
            merge_enabled=merge_enabled,
            max_merge_invocations=max_merge_invocations,
            merge_val_overlap_floor=merge_val_overlap_floor,
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
        "load_lineage": mocker.patch("helix.evolution.load_lineage", return_value={}),
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


class TestConvergenceDetection:
    def test_convergence_patience_stops_loop(self, mocker, tmp_path, all_mocks):
        """Loop stops after convergence_patience stagnant generations."""
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        # Mutate returns None every time → no frontier change
        all_mocks["mutate"].return_value = None

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            return make_eval_result(candidate.id, {"i1": 0.5, "i2": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=20,
            convergence_patience=3,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        warning_msgs = " ".join(str(c) for c in all_mocks["print_warning"].call_args_list)
        assert "Converged" in warning_msgs or "converged" in warning_msgs.lower()

    def test_convergence_count_does_not_exceed_patience(self, mocker, tmp_path, all_mocks):
        """Loop never runs more than convergence_patience extra generations after stagnation."""
        seed = make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None

        gen_count = [0]

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if split == "dev" or (split is None and candidate.id != "g0-s0"):
                gen_count[0] += 1
            return make_eval_result(candidate.id, {"i1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        patience = 3
        config = make_config(
            max_generations=50,
            convergence_patience=patience,
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Loop cannot run more than patience+1 generations (off-by-one in check)
        assert gen_count[0] <= patience + 2  # generous upper bound


# ---------------------------------------------------------------------------
# run_evolution — perfect score early stopping
# ---------------------------------------------------------------------------


class TestPerfectScoreEarlyStopping:
    def test_perfect_score_skips_mutation_continues_loop(self, mocker, tmp_path, all_mocks):
        """Perfect score on dev eval skips mutation (continue) but loop keeps running.

        HELIX uses `continue` (not `break`) when the parent's dev score is perfect,
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
            convergence_patience=3,
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
            convergence_patience=10,
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
            convergence_patience=10,
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
            merge_val_overlap_floor=0,
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
            merge_val_overlap_floor=0,
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
            merge_val_overlap_floor=0,
            convergence_patience=10,
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

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            # Seed is better on i2, children are better on i1 →
            # neither dominates the other → both non-dominated.
            # Child sum (1.4) > parent sum (1.3) so strict acceptance passes.
            if candidate.id == "g0-s0":
                return make_eval_result(candidate.id, {"i1": 0.5, "i2": 0.8})
            return make_eval_result(candidate.id, {"i1": 0.9, "i2": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=3,
            merge_enabled=True,
            max_merge_invocations=1,  # total lifetime cap of 1
            merge_val_overlap_floor=0,
            convergence_patience=10,
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
        """
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        merged = make_candidate("g2-m1", generation=2)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child
        all_mocks["merge"].return_value = merged
        all_mocks["find_merge_triplet"].return_value = ("g0-s0", "g1-s1", "g0-s0")

        merged_eval_count = [0]

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            if candidate.id == "g2-m1":
                merged_eval_count[0] += 1
                # Score must be >= max parent sum to be accepted.
                # max(seed_sum=1.3, child_sum=1.4) = 1.4; merged sum = 1.5
                return make_eval_result("g2-m1", {"i1": 0.9, "i2": 0.6})
            if candidate.id == "g1-s1":
                # Complementary: child better on i1, seed better on i2
                return make_eval_result("g1-s1", {"i1": 0.9, "i2": 0.5})
            return make_eval_result(candidate.id, {"i1": 0.5, "i2": 0.8})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = make_config(
            max_generations=2,  # Need 2 gens: gen 1 accepts, gen 2 fires merge
            merge_enabled=True,
            max_merge_invocations=5,
            merge_val_overlap_floor=0,
            max_metric_calls=10000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert merged_eval_count[0] >= 1


# ---------------------------------------------------------------------------
# run_evolution — dev/val split routing
# ---------------------------------------------------------------------------


class TestDevValSplitRouting:
    def test_dev_split_used_for_gating(self, mocker, tmp_path, all_mocks):
        """Gating evaluation uses split='dev'."""
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
        assert "dev" in child_splits, f"Gating should use dev split; saw: {child_splits}"

    def test_val_split_used_for_pareto_update(self, mocker, tmp_path, all_mocks):
        """After gating passes, val split is used for the frontier update."""
        seed = make_candidate("g0-s0")
        child = make_candidate("g1-s1", generation=1)
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        splits_seen: list[tuple[str, str | None]] = []

        def run_eval(candidate, config, split=None, instances=None, **kwargs):
            splits_seen.append((candidate.id, split))
            if candidate.id == "g1-s1" and split == "dev":
                # Gating passes (child improves on dev)
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

        # Child should be evaluated twice: once for gating (dev), once for val
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
            convergence_patience=100,  # prevent early convergence stop
            max_metric_calls=10000,

        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Loop ran for exactly 2 generations — mutate is called once per gen
        # (Note: dev evals may be cached by EvaluationCache for the same parent,
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
