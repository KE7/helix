"""Unit tests for helix.eval_policy."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from helix.eval_policy import (
    FullEvaluationPolicy,
    ImprovementOrEqualAcceptance,
    StrictImprovementAcceptance,
)


class FakeLoader:
    def __init__(self, ids: list[Any]) -> None:
        self._ids = list(ids)

    def all_ids(self) -> list[Any]:
        return list(self._ids)


@dataclass
class FakeState:
    prog_candidate_val_subscores: list[dict[Any, float]] = field(default_factory=list)


@dataclass
class FakeProposal:
    subsample_scores_before: list[float] | None
    subsample_scores_after: list[float] | None


# ---------------------------------------------------------------------------
# FullEvaluationPolicy.get_eval_batch
# ---------------------------------------------------------------------------


def test_get_eval_batch_returns_all_ids_regardless_of_state() -> None:
    policy = FullEvaluationPolicy()
    loader = FakeLoader(["a", "b", "c", "d"])
    state = FakeState(prog_candidate_val_subscores=[{"a": 0.1}, {"b": 0.9}])

    assert policy.get_eval_batch(loader, state) == ["a", "b", "c", "d"]
    assert policy.get_eval_batch(loader, state, target_program_idx=1) == [
        "a",
        "b",
        "c",
        "d",
    ]
    # Empty state must still return full ids.
    empty_state = FakeState(prog_candidate_val_subscores=[])
    assert policy.get_eval_batch(loader, empty_state) == ["a", "b", "c", "d"]


# ---------------------------------------------------------------------------
# FullEvaluationPolicy.get_best_program
# ---------------------------------------------------------------------------


def test_get_best_program_picks_highest_mean() -> None:
    policy = FullEvaluationPolicy()
    state = FakeState(
        prog_candidate_val_subscores=[
            {"a": 0.1, "b": 0.2},   # mean 0.15
            {"a": 0.9, "b": 0.9},   # mean 0.90  <-- best
            {"a": 0.5, "b": 0.5},   # mean 0.50
        ]
    )
    assert policy.get_best_program(state) == 1


def test_get_best_program_tiebreak_by_coverage() -> None:
    policy = FullEvaluationPolicy()
    state = FakeState(
        prog_candidate_val_subscores=[
            {"a": 0.5},                       # mean 0.5, coverage 1
            {"a": 0.5, "b": 0.5, "c": 0.5},   # mean 0.5, coverage 3  <-- best
            {"a": 0.5, "b": 0.5},             # mean 0.5, coverage 2
        ]
    )
    assert policy.get_best_program(state) == 1


def test_get_best_program_empty_state_returns_minus_one() -> None:
    policy = FullEvaluationPolicy()
    state = FakeState(prog_candidate_val_subscores=[])
    assert policy.get_best_program(state) == -1


def test_get_valset_score_mean_and_empty() -> None:
    policy = FullEvaluationPolicy()
    state = FakeState(
        prog_candidate_val_subscores=[
            {"a": 0.2, "b": 0.8},
            {},
        ]
    )
    assert policy.get_valset_score(0, state) == 0.5
    assert policy.get_valset_score(1, state) == float("-inf")


# ---------------------------------------------------------------------------
# StrictImprovementAcceptance
# ---------------------------------------------------------------------------


def test_strict_acceptance_improvement_true() -> None:
    acc = StrictImprovementAcceptance()
    proposal = FakeProposal(
        subsample_scores_before=[0.1, 0.2],
        subsample_scores_after=[0.3, 0.3],
    )
    assert acc.should_accept(proposal) is True


def test_strict_acceptance_equal_false() -> None:
    acc = StrictImprovementAcceptance()
    proposal = FakeProposal(
        subsample_scores_before=[0.5, 0.5],
        subsample_scores_after=[0.6, 0.4],
    )
    assert acc.should_accept(proposal) is False


def test_strict_acceptance_none_lists_treated_as_empty() -> None:
    acc = StrictImprovementAcceptance()
    # Both None -> 0 > 0 is False
    p1 = FakeProposal(subsample_scores_before=None, subsample_scores_after=None)
    assert acc.should_accept(p1) is False
    # Before None, after positive -> accept
    p2 = FakeProposal(subsample_scores_before=None, subsample_scores_after=[0.1])
    assert acc.should_accept(p2) is True
    # Before positive, after None -> reject
    p3 = FakeProposal(subsample_scores_before=[0.1], subsample_scores_after=None)
    assert acc.should_accept(p3) is False


# ---------------------------------------------------------------------------
# ImprovementOrEqualAcceptance
# ---------------------------------------------------------------------------


def test_improvement_or_equal_equal_true() -> None:
    acc = ImprovementOrEqualAcceptance()
    proposal = FakeProposal(
        subsample_scores_before=[0.5, 0.5],
        subsample_scores_after=[0.6, 0.4],
    )
    assert acc.should_accept(proposal) is True


def test_improvement_or_equal_strict_drop_false() -> None:
    acc = ImprovementOrEqualAcceptance()
    proposal = FakeProposal(
        subsample_scores_before=[0.5, 0.5],
        subsample_scores_after=[0.4, 0.5],
    )
    assert acc.should_accept(proposal) is False
