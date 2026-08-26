"""Unit tests for helix.candidate_selector — GEPA-parity selection strategies.

Companion to tests/unit/test_population.py, which covers the ``pareto``
strategy through ``ParetoFrontier.select_parent`` directly.
"""

from __future__ import annotations

import os
import random
import subprocess
import sys
import textwrap

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
        """current_best ranks by aggregate_score() (mean), not sum_score().

        X has fewer, better-average instances; Y has more, worse-average
        instances but a higher sum, so the two rankings disagree here.
        """
        frontier = ParetoFrontier()
        add(frontier, "X", {"i1": 0.9})  # sum=0.9, agg=0.9
        add(frontier, "Y", {"i1": 0.5, "i2": 0.5, "i3": 0.5, "i4": 0.5})  # sum=2.0, agg=0.5
        assert select_current_best(frontier).id == "X"
        # sum_score would have picked Y — the two quantities disagree here.
        result_x = frontier.get_result("X")
        result_y = frontier.get_result("Y")
        assert result_x is not None and result_y is not None
        assert result_y.sum_score() > result_x.sum_score()

    def test_tie_break_is_earliest_discovered(self):
        """Ties resolve to the earliest-added candidate, matching GEPA's
        ``idxmax`` (``lst.index(max(lst))`` returns the first index at the max).
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
        """epsilon=1.0 always takes the random branch, which draws from the
        whole pool rather than the Pareto frontier.

        "dominated" loses on every key, so the "pareto" strategy would never
        draw it; epsilon-greedy exploration still can.
        """
        frontier = ParetoFrontier()
        add(frontier, "dominator", {"i1": 1.0, "i2": 1.0})
        add(frontier, "dominated", {"i1": 0.1, "i2": 0.1})
        rng = random.Random(7)
        seen = {select_epsilon_greedy(frontier, rng, 1.0).id for _ in range(200)}
        assert seen == {"dominator", "dominated"}

    def test_epsilon_zero_never_consumes_extra_randomness_on_greedy_path(self):
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

    def test_k_greater_equal_pool_still_ranks_by_mean_not_sum(self):
        """``k >= len(frontier)`` makes the top-k membership filter a no-op,
        so this pins that the ranking step itself — separate from that
        filter — stays mean-based even when it is the only thing left doing
        any work.

        This no longer distinguishes this function from
        ``ParetoFrontier.select_parent``: that method has also ranked by
        ``aggregate_score()`` since ``fix/default-parent-selection``, and
        with the membership filter a no-op the two run dominance removal
        over the identical mapping with the identical scores, producing an
        identical eligible set with identical frequency weights (verified
        directly, not just by matching outcomes over repeated draws;
        ``select_parent`` draws from its own internal RNG rather than the
        one passed here, so exact draw sequences do not compare). This test
        now only guards against a regression to sum-based ranking within
        this function's own ranking step.

            a  {i1: 1.0}                agg=1.00  sum=1.00  (wins i1)
            b  {i1: 1.0, i2: 0.9}       agg=0.95  sum=1.90  (wins nothing)
            c  {i2: 1.0}                agg=1.00  sum=1.00  (wins i2)

        By mean, "b" is dominated by "a" on i1's front and never survives
        dominance removal, leaving only "a" and "c". By sum, "b" would
        survive and even displace "a" from i1's cleaned front, so "b"
        appearing in the draw would mean the ranking regressed to sum.
        """
        frontier = ParetoFrontier()
        add(frontier, "a", {"i1": 1.0})
        add(frontier, "b", {"i1": 1.0, "i2": 0.9})
        add(frontier, "c", {"i2": 1.0})
        rng = random.Random(5)
        seen = {select_top_k_pareto(frontier, rng, 3).id for _ in range(200)}
        assert seen == {"a", "c"}
        assert "b" not in seen

        # k strictly larger than the pool must behave identically.
        frontier2 = ParetoFrontier()
        add(frontier2, "a", {"i1": 1.0})
        add(frontier2, "b", {"i1": 1.0, "i2": 0.9})
        add(frontier2, "c", {"i2": 1.0})
        rng2 = random.Random(5)
        seen2 = {select_top_k_pareto(frontier2, rng2, 100).id for _ in range(200)}
        assert seen2 == {"a", "c"}
        assert "b" not in seen2

    def test_k_greater_equal_pool_mean_ranking_holds_under_seed_1(self):
        """Seed-specific companion to
        ``test_k_greater_equal_pool_still_ranks_by_mean_not_sum``.

        "b" is dominated out of the sampling list under mean ranking
        regardless of RNG seed — a sum-based ranking would instead let "b"
        survive and become drawable — so this pins that guarantee under a
        second, independently chosen seed. Pinned separately because a
        single seed's draws could coincidentally avoid "b" even if a
        sum-based regression made it selectable again.
        """
        frontier = ParetoFrontier()
        add(frontier, "a", {"i1": 1.0})
        add(frontier, "b", {"i1": 1.0, "i2": 0.9})
        add(frontier, "c", {"i2": 1.0})
        rng = random.Random(1)
        seen = {select_top_k_pareto(frontier, rng, 3).id for _ in range(50)}
        assert "b" not in seen
        assert seen == {"a", "c"}

    def test_ranks_by_mean_score_and_intersects_per_key_fronts(self):
        """A/B/D each own one key, but only A/B survive a top-2 filter by
        aggregate_score, so D's key is dropped and D is never drawn."""
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

        Both the ranking and the fallback read aggregate_score(). Under
        sum_score() the sums are a=1.0 > y=0.9 > b=0.6, so top-1-by-sum is A,
        whose front is non-empty, and the draw returns A without reaching the
        fallback. Asserting "y" therefore pins mean semantics end to end.
        """
        frontier = ParetoFrontier()
        add(frontier, "a", {"i1": 1.0, "i2": 0.0})  # agg=0.5 sum=1.0, wins i1
        add(frontier, "b", {"i2": 0.6})  # agg=0.6 sum=0.6, takes i2
        add(frontier, "y", {"i1": 0.9})  # agg=0.9 sum=0.9, wins nothing
        rng = random.Random(0)
        assert select_top_k_pareto(frontier, rng, 1).id == "y"

    def test_ranking_uses_mean_not_sum_when_cardinality_differs(self):
        """Top-k ranking reads the mean, not the sum.

        Upstream ranks top-K by ``per_program_tracked_scores``, a mean
        (``get_program_average_val_subset`` -> ``sum(...)/num_samples``). Sum
        and mean only disagree when candidates have unequal numbers of scored
        instances, so this pool gives Y two instances and everyone else one:

            a/b/d  agg=1.00  sum=1.00   (each owns one key)
            x      agg=0.99  sum=0.99   (owns its own key)
            y      agg=0.90  sum=1.80   (wins nothing)

        Top-1 by mean is "a" (first among the three tied at 1.0), whose front
        survives the intersection, so the draw returns "a". Top-1 by sum would
        be "y", which owns no key — that empties the filtered mapping and the
        fallback yields "y". The two rankings return different candidates.
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
# Empty-result floor (-inf, not 0.0)
# ---------------------------------------------------------------------------


class TestEmptyResultFloor:
    """An unscored candidate (``instance_scores == {}``) ranks below every
    scored candidate, including one with a negative mean — matching upstream
    ``get_program_average_val_subset``'s ``float("-inf")`` for a program with
    no recorded subscores. ``EvalResult.aggregate_score()`` returns ``0.0``
    there, by design and for other callers, so an unfloored selector would
    rank "unscored" above "scored but bad".
    """

    def test_current_best_prefers_negative_mean_over_empty(self):
        """empty:{} vs negative:{'x': -0.25}. The upstream-equivalent scores
        are [-inf, -0.25], so 'negative' wins.
        """
        frontier = ParetoFrontier()
        add(frontier, "empty", {})
        add(frontier, "negative", {"x": -0.25})
        assert select_current_best(frontier).id == "negative"

    def test_top_k_pareto_ranking_never_lets_empty_bump_a_negative_score(self):
        """'negative' and 'positive' each own a distinct per-key front;
        'empty' owns none. A top-2 ranking that floors 'empty' at -inf keeps
        both real candidates' fronts alive. An unfloored ranking (0.0 beats
        -0.25) instead lets 'empty' displace 'negative' out of the top-2,
        dropping negative's only front and leaving 'positive' as the sole
        winner.
        """
        frontier = ParetoFrontier()
        add(frontier, "negative", {"i1": -0.25})
        add(frontier, "positive", {"i2": 0.5})
        add(frontier, "empty", {})
        rng = random.Random(11)
        seen = {select_top_k_pareto(frontier, rng, 2).id for _ in range(100)}
        assert seen == {"negative", "positive"}
        assert "empty" not in seen

    def test_top_k_pareto_fallback_prefers_negative_mean_over_empty(self):
        """Forces the empty-filtered-mapping fallback (neither the
        top-ranked real candidate nor 'empty' owns a front: 'dominator' has
        the worse mean but wins the only key either of them touches), then
        asserts the fallback argmax floors 'empty' below 'negative'.
        """
        frontier = ParetoFrontier()
        add(frontier, "negative", {"i1": -0.1})
        add(frontier, "dominator", {"i1": 0.0, "i2": -0.9})  # wins i1, mean -0.45
        add(frontier, "empty", {})
        assert select_top_k_pareto(frontier, random.Random(0), 1).id == "negative"

    def test_all_empty_pool_returns_first_discovered_not_none_or_raise(self):
        """``_first_argmax`` takes its first candidate unconditionally: once
        every candidate floors to the same -inf, a pure strict-improvement
        scan never satisfies ``score > best_score`` and would leave
        ``best_id`` unset.
        """
        frontier = ParetoFrontier()
        add(frontier, "first", {})
        add(frontier, "second", {})
        add(frontier, "third", {})
        assert select_current_best(frontier).id == "first"


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
        # A different seed must diverge on this pool; otherwise the test
        # would pass even if the rng were ignored entirely.
        assert run(999) != seq_a

    def test_top_k_pareto_same_seed_across_processes(self):
        """String-hash randomization cannot alter a seeded top-k draw."""
        selection_script = textwrap.dedent(
            """
            import random

            from helix.candidate_selector import select_top_k_pareto
            from helix.population import Candidate, EvalResult, ParetoFrontier

            frontier = ParetoFrontier()
            for candidate_id in ("alpha", "bravo", "charlie", "delta"):
                frontier.add(
                    Candidate(
                        id=candidate_id,
                        worktree_path="",
                        branch_name="",
                        generation=0,
                        parent_id=None,
                        parent_ids=[],
                        operation="mutation",
                    ),
                    EvalResult(
                        candidate_id=candidate_id,
                        scores={},
                        asi={},
                        instance_scores={"instance": 1.0},
                    ),
                )

            print(select_top_k_pareto(frontier, random.Random(2024), 4).id)
            """
        )
        selections = {
            subprocess.run(
                [sys.executable, "-c", selection_script],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
            ).stdout.strip()
            for hash_seed in range(1, 9)
        }

        assert len(selections) == 1
