"""Unit tests for helix.population — GEPA coverage-based dominance, selection, and frontier logic."""

from __future__ import annotations

import random

import pytest

from helix.population import Candidate, EvalResult, ParetoFrontier


# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Per-key best tracking
# ---------------------------------------------------------------------------

class TestPerKeyBestTracking:
    def test_single_candidate_owns_all_keys(self):
        """A single candidate should appear in every key's best set."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.5, "i2": 0.7}))
        assert frontier._per_key_best["i1"] == {"a"}
        assert frontier._per_key_best["i2"] == {"a"}

    def test_better_score_replaces_best_set(self):
        """When a new candidate beats the best score on a key, it replaces the set."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.5}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.9}))
        assert frontier._per_key_best["i1"] == {"b"}

    def test_tied_score_expands_best_set(self):
        """When a new candidate ties the best score, it joins the set."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.5}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.5}))
        assert frontier._per_key_best["i1"] == {"a", "b"}

    def test_worse_score_does_not_enter_best_set(self):
        """A candidate with a worse score on a key should not join the set."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.9}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.3}))
        assert frontier._per_key_best["i1"] == {"a"}

    def test_different_keys_tracked_independently(self):
        """Each validation key tracks its best candidates independently."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.9, "i2": 0.2}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.3, "i2": 0.8}))
        assert frontier._per_key_best["i1"] == {"a"}
        assert frontier._per_key_best["i2"] == {"b"}

    def test_update_scores_rebuilds_tracking(self):
        """Calling update_scores rebuilds per-key tracking from scratch."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.9}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.5}))
        assert frontier._per_key_best["i1"] == {"a"}

        # Now update b to beat a
        frontier.update_scores(make_result("b", {"i1": 0.95}))
        assert frontier._per_key_best["i1"] == {"b"}


# ---------------------------------------------------------------------------
# Coverage-based dominance (GEPA)
# ---------------------------------------------------------------------------

class TestCoverageBasedDominance:
    def test_dominated_when_not_in_any_frontier_key(self):
        """A candidate not in any per-key best set is dominated."""
        frontier = ParetoFrontier()
        # 'strong' beats 'weak' on both instances -> weak not in any key
        frontier.add(make_candidate("weak"), make_result("weak", {"i1": 0.5, "i2": 0.4}))
        frontier.add(make_candidate("strong"), make_result("strong", {"i1": 0.9, "i2": 0.8}))

        assert frontier.is_dominated("weak") is True

    def test_not_dominated_with_unique_key_coverage(self):
        """A candidate with unique coverage on at least one key is NOT dominated."""
        frontier = ParetoFrontier()
        # 'a' wins i1, 'b' wins i2 -- each has unique coverage
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.9, "i2": 0.2}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.3, "i2": 0.8}))

        assert frontier.is_dominated("a") is False
        assert frontier.is_dominated("b") is False

    def test_single_candidate_never_dominated(self):
        """A lone candidate cannot be dominated by anyone."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("solo"), make_result("solo", {"i1": 0.5}))

        assert frontier.is_dominated("solo") is False

    def test_get_dominated_returns_correct_ids(self):
        """get_dominated() returns exactly the dominated candidate IDs."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("weak"), make_result("weak", {"i1": 0.1, "i2": 0.1}))
        frontier.add(make_candidate("mid"), make_result("mid", {"i1": 0.5, "i2": 0.5}))
        frontier.add(make_candidate("strong"), make_result("strong", {"i1": 0.9, "i2": 0.9}))

        dominated = frontier.get_dominated()
        assert "weak" in dominated
        assert "mid" in dominated
        assert "strong" not in dominated

    def test_tied_candidates_one_is_redundant(self):
        """GEPA coverage: when two candidates tie on all keys, one provides
        redundant coverage and is considered dominated."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.5, "i2": 0.5}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.5, "i2": 0.5}))

        non_dom = frontier.get_non_dominated()
        # Exactly one should be non-dominated (the first processed)
        assert len(non_dom) == 1
        # The other is dominated
        dominated = frontier.get_dominated()
        assert len(dominated) == 1

    def test_unknown_candidate_not_dominated(self):
        """An unknown candidate ID returns False (not dominated)."""
        frontier = ParetoFrontier()
        assert frontier.is_dominated("ghost") is False

    def test_three_way_coverage_split(self):
        """Three candidates each winning a unique key: all non-dominated."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.9, "i2": 0.1, "i3": 0.1}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.1, "i2": 0.9, "i3": 0.1}))
        frontier.add(make_candidate("c"), make_result("c", {"i1": 0.1, "i2": 0.1, "i3": 0.9}))

        non_dom = frontier.get_non_dominated()
        assert non_dom == {"a", "b", "c"}

    def test_dominated_with_partial_key_overlap(self):
        """A candidate whose keys are all covered by non-dominated others is dominated."""
        frontier = ParetoFrontier()
        # a is best at i1, b is best at i2, c ties on both -> covered by a+b
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.9, "i2": 0.5}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.5, "i2": 0.9}))
        frontier.add(make_candidate("c"), make_result("c", {"i1": 0.5, "i2": 0.5}))

        non_dom = frontier.get_non_dominated()
        assert "a" in non_dom
        assert "b" in non_dom
        assert "c" not in non_dom

    def test_population_is_append_only(self):
        """Population storage is never pruned by dominance operations."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("weak"), make_result("weak", {"i1": 0.1}))
        frontier.add(make_candidate("strong"), make_result("strong", {"i1": 0.9}))

        # Even though weak is dominated, it's still in storage
        assert "weak" in frontier._candidates
        assert "weak" in frontier._results
        assert len(frontier) == 2


# ---------------------------------------------------------------------------
# Weighted parent selection (GEPA)
# ---------------------------------------------------------------------------

class TestWeightedParentSelection:
    def test_more_coverage_higher_selection_probability(self):
        """Candidates with more frontier key coverage should be selected more often."""
        random.seed(42)
        frontier = ParetoFrontier()
        # 'dominant' wins 9 instances; 'weak' wins 1
        dominant_scores = {f"i{k}": 0.9 for k in range(9)}
        dominant_scores["i9"] = 0.9
        weak_scores = {f"i{k}": 0.1 for k in range(9)}
        weak_scores["i9"] = 0.95  # weak wins only i9

        frontier.add(make_candidate("dominant"), make_result("dominant", dominant_scores))
        frontier.add(make_candidate("weak"), make_result("weak", weak_scores))

        selections = {"dominant": 0, "weak": 0}
        for _ in range(1000):
            selected = frontier.select_parent()
            selections[selected.id] += 1

        # dominant should be selected significantly more often
        assert selections["dominant"] > selections["weak"] * 3, (
            f"Expected dominant to be selected 3x more; got {selections}"
        )

    def test_select_parent_single_candidate(self):
        """With one candidate, it is always selected."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("only"), make_result("only", {"i1": 0.5}))

        for _ in range(10):
            assert frontier.select_parent().id == "only"

    def test_select_parent_raises_on_empty(self):
        with pytest.raises(ValueError, match="empty"):
            ParetoFrontier().select_parent()

    def test_select_parent_only_from_non_dominated(self):
        """Parent selection should only return non-dominated candidates."""
        random.seed(123)
        frontier = ParetoFrontier()
        frontier.add(make_candidate("strong"), make_result("strong", {"i1": 0.9, "i2": 0.9}))
        frontier.add(make_candidate("weak"), make_result("weak", {"i1": 0.1, "i2": 0.1}))

        for _ in range(50):
            assert frontier.select_parent().id == "strong"


# ---------------------------------------------------------------------------
# Complementary pair selection
# ---------------------------------------------------------------------------

class TestSelectComplementaryPair:
    def test_returns_candidates_with_different_strengths(self):
        """Pair should consist of candidates that win on different instances."""
        frontier = ParetoFrontier()
        # 'left' is strong on i0-i4; 'right' is strong on i5-i9
        left_scores = {f"i{k}": (0.9 if k < 5 else 0.1) for k in range(10)}
        right_scores = {f"i{k}": (0.1 if k < 5 else 0.9) for k in range(10)}
        # 'mediocre' is average everywhere
        mediocre_scores = {f"i{k}": 0.5 for k in range(10)}

        frontier.add(make_candidate("left"), make_result("left", left_scores))
        frontier.add(make_candidate("right"), make_result("right", right_scores))
        frontier.add(make_candidate("mediocre"), make_result("mediocre", mediocre_scores))

        a, b = frontier.select_complementary_pair()
        pair = {a.id, b.id}

        # The most complementary pair should be left+right (0 overlap in wins)
        assert pair == {"left", "right"}, (
            f"Expected {{left, right}} but got {pair}"
        )

    def test_complementary_pair_raises_with_one_candidate(self):
        frontier = ParetoFrontier()
        frontier.add(make_candidate("solo"), make_result("solo", {"i1": 0.5}))
        with pytest.raises(ValueError, match="at least 2"):
            frontier.select_complementary_pair()

    def test_complementary_pair_two_candidates_returns_both(self):
        """With exactly two candidates the pair is always those two."""
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.9, "i2": 0.1}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.1, "i2": 0.9}))

        a, b = frontier.select_complementary_pair()
        assert {a.id, b.id} == {"a", "b"}


# ---------------------------------------------------------------------------
# Best / aggregate score
# ---------------------------------------------------------------------------

class TestBestAndAggregateScore:
    def test_best_returns_highest_aggregate(self):
        frontier = ParetoFrontier()
        frontier.add(make_candidate("low"), make_result("low", {"i1": 0.1, "i2": 0.2}))
        frontier.add(make_candidate("high"), make_result("high", {"i1": 0.8, "i2": 0.9}))
        assert frontier.best().id == "high"

    def test_aggregate_score_is_mean_of_instance_scores(self):
        result = make_result("c", {"i1": 0.6, "i2": 0.4})
        assert result.aggregate_score() == pytest.approx(0.5)

    def test_aggregate_score_empty(self):
        result = make_result("c", {})
        assert result.aggregate_score() == 0.0


# ---------------------------------------------------------------------------
# Signature / convergence
# ---------------------------------------------------------------------------

class TestSignature:
    def test_same_state_same_signature(self):
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.5}))
        sig1 = frontier.signature()
        sig2 = frontier.signature()
        assert sig1 == sig2

    def test_different_state_different_signature(self):
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.5}))
        sig1 = frontier.signature()
        frontier.update_scores(make_result("a", {"i1": 0.9}))
        sig2 = frontier.signature()
        assert sig1 != sig2


# ---------------------------------------------------------------------------
# Non-dominated set
# ---------------------------------------------------------------------------

class TestGetNonDominated:
    def test_all_non_dominated_when_each_wins_unique_key(self):
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.9, "i2": 0.1}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.1, "i2": 0.9}))
        assert frontier.get_non_dominated() == {"a", "b"}

    def test_single_candidate_is_non_dominated(self):
        frontier = ParetoFrontier()
        frontier.add(make_candidate("solo"), make_result("solo", {"i1": 0.5}))
        assert frontier.get_non_dominated() == {"solo"}

    def test_completely_dominated_excluded(self):
        frontier = ParetoFrontier()
        frontier.add(make_candidate("a"), make_result("a", {"i1": 0.9, "i2": 0.9}))
        frontier.add(make_candidate("b"), make_result("b", {"i1": 0.1, "i2": 0.1}))
        non_dom = frontier.get_non_dominated()
        assert "a" in non_dom
        assert "b" not in non_dom
