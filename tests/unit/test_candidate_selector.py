"""Unit tests for helix.candidate_selector — GEPA-parity selection strategies.

Companion to tests/unit/test_population.py (which exercises the pre-existing
``pareto`` strategy via ``ParetoFrontier.select_parent`` directly and is left
untouched by this PR).
"""

from __future__ import annotations

import random

import pytest

from helix.candidate_selector import (
    select_candidate,
    select_current_best,
    select_epsilon_greedy,
    select_top_k_pareto,
)
from helix.population import Candidate, EvalResult, ParetoFrontier


# ---------------------------------------------------------------------------
# Helpers (mirrors tests/unit/test_population.py's fixtures)
# ---------------------------------------------------------------------------


def make_candidate(cid: str, generation: int = 0) -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=f"/tmp/{cid}",
        branch_name=f"branch-{cid}",
        generation=generation,
        parent_id=None,
        parent_ids=[],
        operation="mutation",
    )


def make_result(cid: str, instance_scores: dict[str, float]) -> EvalResult:
    return EvalResult(
        candidate_id=cid,
        scores={},
        asi={},
        instance_scores=instance_scores,
    )


def add(frontier: ParetoFrontier, cid: str, instance_scores: dict[str, float]) -> None:
    frontier.add(make_candidate(cid), make_result(cid, instance_scores))


# ---------------------------------------------------------------------------
# select_current_best
# ---------------------------------------------------------------------------


class TestCurrentBest:
    def test_empty_frontier_raises(self):
        with pytest.raises(ValueError):
            select_current_best(ParetoFrontier())

    def test_single_candidate(self):
        frontier = ParetoFrontier()
        add(frontier, "only", {"i1": 0.4})
        assert select_current_best(frontier).id == "only"

    def test_picks_highest_aggregate_score(self):
        frontier = ParetoFrontier()
        add(frontier, "weak", {"i1": 0.2, "i2": 0.2})
        add(frontier, "strong", {"i1": 0.9, "i2": 0.9})
        assert select_current_best(frontier).id == "strong"

    def test_uses_aggregate_not_sum_score(self):
        """current_best's HELIX analog is aggregate_score() (mean), not
        sum_score(). X has fewer, better-average instances; Y has more,
        worse-average instances but a higher sum. current_best must pick X.
        """
        frontier = ParetoFrontier()
        add(frontier, "X", {"i1": 0.9})  # sum=0.9, agg=0.9
        add(frontier, "Y", {"i1": 0.5, "i2": 0.5, "i3": 0.5, "i4": 0.5})  # sum=2.0, agg=0.5
        assert select_current_best(frontier).id == "X"
        # Sanity: sum_score would have picked Y, proving the two quantities
        # genuinely disagree on this pool.
        result_x = frontier.get_result("X")
        result_y = frontier.get_result("Y")
        assert result_x is not None and result_y is not None
        assert result_y.sum_score() > result_x.sum_score()

    def test_tie_break_is_earliest_discovered(self):
        """Ties resolve to the earliest-added candidate (GEPA idxmax parity:
        ``lst.index(max(lst))`` returns the FIRST index achieving the max).
        """
        frontier = ParetoFrontier()
        add(frontier, "first", {"i1": 1.0})
        add(frontier, "second", {"i1": 1.0})
        add(frontier, "third", {"i1": 0.1})
        assert select_current_best(frontier).id == "first"

    def test_tie_break_survives_a_later_strict_improvement(self):
        frontier = ParetoFrontier()
        add(frontier, "a", {"i1": 0.5})
        add(frontier, "b", {"i1": 0.5})
        add(frontier, "c", {"i1": 0.9})
        assert select_current_best(frontier).id == "c"

    def test_deterministic_across_repeated_calls(self):
        frontier = ParetoFrontier()
        add(frontier, "a", {"i1": 0.5})
        add(frontier, "b", {"i1": 0.5})
        add(frontier, "c", {"i1": 0.2})
        results = {select_current_best(frontier).id for _ in range(20)}
        assert results == {"a"}


# ---------------------------------------------------------------------------
# select_epsilon_greedy
# ---------------------------------------------------------------------------


class TestEpsilonGreedy:
    def test_empty_frontier_raises(self):
        with pytest.raises(ValueError):
            select_epsilon_greedy(ParetoFrontier(), random.Random(0), 0.5)

    def test_epsilon_zero_is_always_current_best(self):
        frontier = ParetoFrontier()
        add(frontier, "weak", {"i1": 0.1})
        add(frontier, "strong", {"i1": 0.9})
        rng = random.Random(123)
        for _ in range(25):
            assert select_epsilon_greedy(frontier, rng, 0.0).id == "strong"

    def test_epsilon_one_is_always_uniform_over_whole_pool(self):
        """epsilon=1.0 always takes the random branch (rng.random() < 1.0 is
        always true) and draws from the WHOLE pool, not the Pareto frontier.
        Build a pool where a candidate is dominated on every key (so it
        would never be drawn by "pareto") and confirm it can still be
        picked here.
        """
        frontier = ParetoFrontier()
        add(frontier, "dominator", {"i1": 1.0, "i2": 1.0})
        add(frontier, "dominated", {"i1": 0.1, "i2": 0.1})
        rng = random.Random(7)
        seen = {select_epsilon_greedy(frontier, rng, 1.0).id for _ in range(200)}
        assert seen == {"dominator", "dominated"}

    def test_epsilon_one_never_consumes_extra_randomness_on_greedy_path(self):
        """Greedy branch (else) must not draw from rng — only the coin-flip
        draw is consumed when the random branch isn't taken."""
        frontier = ParetoFrontier()
        add(frontier, "only", {"i1": 0.5})
        rng_a = random.Random(99)
        rng_b = random.Random(99)
        select_epsilon_greedy(frontier, rng_a, 0.0)
        rng_b.random()  # manually consume exactly one draw (the coin flip)
        assert rng_a.random() == rng_b.random()


# ---------------------------------------------------------------------------
# select_top_k_pareto
# ---------------------------------------------------------------------------


class TestTopKPareto:
    def test_empty_frontier_raises(self):
        with pytest.raises(ValueError):
            select_top_k_pareto(ParetoFrontier(), random.Random(0), 1)

    def test_single_candidate(self):
        frontier = ParetoFrontier()
        add(frontier, "only", {"i1": 0.4})
        assert select_top_k_pareto(frontier, random.Random(0), 1).id == "only"
        # k larger than the pool must degrade cleanly too.
        assert select_top_k_pareto(frontier, random.Random(0), 99).id == "only"

    def test_k_greater_equal_pool_matches_pareto(self):
        """k >= pool size makes the top-k filter a mathematical no-op, so
        this must reduce exactly to ``pareto`` (frontier.select_parent()),
        including its exact rng draw for a given seed."""
        frontier_a = ParetoFrontier(rng=random.Random(42))
        frontier_b = ParetoFrontier(rng=random.Random(42))
        for f in (frontier_a, frontier_b):
            add(f, "a", {"i1": 1.0, "i2": 0.2})
            add(f, "b", {"i1": 0.2, "i2": 1.0})
            add(f, "c", {"i1": 0.5, "i2": 0.5})

        expected = frontier_a.select_parent().id
        got = select_top_k_pareto(frontier_b, frontier_b._rng, 3)
        assert got.id == expected

        # k strictly larger than the pool degrades the same way.
        frontier_c = ParetoFrontier(rng=random.Random(42))
        add(frontier_c, "a", {"i1": 1.0, "i2": 0.2})
        add(frontier_c, "b", {"i1": 0.2, "i2": 1.0})
        add(frontier_c, "c", {"i1": 0.5, "i2": 0.5})
        assert select_top_k_pareto(frontier_c, frontier_c._rng, 100).id == expected

    def test_ranks_by_mean_score_and_intersects_per_key_fronts(self):
        """Hand-built frontier: A/B/D each own one key; only A/B survive a
        top-2 filter by aggregate_score, so D's key must be dropped and the
        draw must never return D."""
        frontier = ParetoFrontier()
        add(frontier, "a", {"i1": 1.0})  # sum=1.0, wins i1
        add(frontier, "b", {"i2": 1.0})  # sum=1.0, wins i2
        add(frontier, "d", {"i3": 0.05})  # sum=0.05, wins i3 (sole entrant)
        rng = random.Random(3)
        seen = {select_top_k_pareto(frontier, rng, 2).id for _ in range(50)}
        assert seen == {"a", "b"}
        assert "d" not in seen

    def test_empty_filtered_mapping_falls_back_to_mean_score_argmax(self):
        """Y has the highest MEAN but wins no per-key front, so a top-1
        filter empties every front and forces the fallback.

        A owns key i1 outright (1.0) but carries a 0.0 on i2 that drags its
        mean to 0.5; B takes i2 with 0.6; Y scores 0.9 on i1 alone — a clean
        loss on the only key it touches, but the best mean in the pool.

        Both the ranking and the fallback must read aggregate_score(). A
        sum_score() implementation returns "a" twice over: sums are
        a=1.0 > y=0.9 > b=0.6, so top-1-by-sum is A, whose front is
        non-empty, and the draw returns A without ever reaching the
        fallback. Asserting "y" therefore pins mean semantics end to end.
        """
        frontier = ParetoFrontier()
        add(frontier, "a", {"i1": 1.0, "i2": 0.0})  # agg=0.5 sum=1.0, wins i1
        add(frontier, "b", {"i2": 0.6})  # agg=0.6 sum=0.6, takes i2
        add(frontier, "y", {"i1": 0.9})  # agg=0.9 sum=0.9, wins nothing
        rng = random.Random(0)
        assert select_top_k_pareto(frontier, rng, 1).id == "y"

    def test_ranking_uses_mean_not_sum_when_cardinality_differs(self):
        """Regression guard for the sum-vs-mean mapping defect.

        Upstream ranks top-K by ``per_program_tracked_scores``, which is a
        MEAN (``get_program_average_val_subset`` -> ``sum(...)/num_samples``).
        Sum and mean only disagree when candidates have unequal numbers of
        scored instances, so this pool gives Y two instances and everyone
        else one:

            a/b/d  agg=1.00  sum=1.00   (each owns one key)
            x      agg=0.99  sum=0.99   (owns its own key)
            y      agg=0.90  sum=1.80   (wins nothing)

        Top-1 by MEAN is "a" (first among the three tied at 1.0), whose
        front survives the intersection, so the draw returns "a".
        Top-1 by SUM would be "y", which owns no key — that empties the
        filtered mapping and the sum fallback yields "y". The two rankings
        disagree on the returned candidate, which is exactly the defect.
        """
        frontier = ParetoFrontier()
        add(frontier, "a", {"i1": 1.0})
        add(frontier, "b", {"i2": 1.0})
        add(frontier, "d", {"i3": 1.0})
        add(frontier, "x", {"i9": 0.99})
        add(frontier, "y", {"i1": 0.9, "i2": 0.9})
        rng = random.Random(0)
        assert select_top_k_pareto(frontier, rng, 1).id == "a"


# ---------------------------------------------------------------------------
# select_candidate dispatcher
# ---------------------------------------------------------------------------


class TestSelectCandidateDispatch:
    def test_pareto_delegates_to_select_parent(self):
        frontier = ParetoFrontier(rng=random.Random(1))
        add(frontier, "only", {"i1": 0.5})
        assert select_candidate("pareto", frontier, random.Random(1)).id == "only"

    def test_current_best_dispatch(self):
        frontier = ParetoFrontier()
        add(frontier, "weak", {"i1": 0.1})
        add(frontier, "strong", {"i1": 0.9})
        assert (
            select_candidate("current_best", frontier, random.Random(0)).id
            == "strong"
        )

    def test_unknown_strategy_raises(self):
        frontier = ParetoFrontier()
        add(frontier, "only", {"i1": 0.5})
        with pytest.raises(ValueError):
            select_candidate("bogus", frontier, random.Random(0))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Seeded reproducibility
# ---------------------------------------------------------------------------


class TestSeededReproducibility:
    def _build_pool(self, frontier: ParetoFrontier) -> None:
        add(frontier, "a", {"i1": 1.0, "i2": 0.2, "i3": 0.5})
        add(frontier, "b", {"i1": 0.2, "i2": 1.0, "i3": 0.4})
        add(frontier, "c", {"i1": 0.5, "i2": 0.5, "i3": 1.0})
        add(frontier, "d", {"i1": 0.3, "i2": 0.3, "i3": 0.3})

    @pytest.mark.parametrize("strategy", ["epsilon_greedy", "top_k_pareto"])
    def test_same_seed_same_sequence(self, strategy):
        kwargs = {"epsilon": 0.5} if strategy == "epsilon_greedy" else {"top_k": 2}

        def run(seed: int) -> list[str]:
            frontier = ParetoFrontier()
            self._build_pool(frontier)
            rng = random.Random(seed)
            return [
                select_candidate(strategy, frontier, rng, **kwargs).id
                for _ in range(10)
            ]

        seq_a = run(2024)
        seq_b = run(2024)
        assert seq_a == seq_b
        # Sanity: a different seed is allowed to (and, with this pool,
        # does) diverge — otherwise the test would trivially pass even if
        # the rng were ignored entirely.
        assert run(999) != seq_a
