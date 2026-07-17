"""GEPA-parity tests for HELIX proposal planning and selection."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from typing import cast
from unittest.mock import MagicMock

import pytest

from helix.eval_policy import StrictImprovementAcceptance
from helix.population import Candidate, EvalResult
from helix.proposals import (
    DEFAULT_SAMPLING_STRATEGY,
    DEFAULT_SELECTION_STRATEGY,
    AllImprovements,
    BestImprovement,
    EvaluatedProposal,
    FailedProposal,
    IndependentSampling,
    ProposalAcceptanceCriterion,
    ProposalTask,
    PxNSampling,
    SameParentSampling,
    SelectedProposal,
    SingleMutationSampling,
    SkippedProposal,
    TamperedProposal,
    TopKImprovements,
)


def make_candidate(candidate_id: str) -> Candidate:
    return Candidate(
        id=candidate_id,
        worktree_path=f"/tmp/{candidate_id}",
        branch_name=f"helix/{candidate_id}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="mutation",
    )


def make_eval(candidate_id: str, score: float) -> EvalResult:
    return EvalResult(
        candidate_id=candidate_id,
        scores={"score": score},
        asi={},
        instance_scores={"example": score},
    )


def make_evaluated(index: int, before: float, after: float) -> EvaluatedProposal:
    parent = make_candidate(f"parent-{index}")
    child = make_candidate(f"child-{index}")
    task = ProposalTask(
        batch_index=index,
        parent_group_index=index,
        mutation_index=0,
        parent_candidate=parent,
        minibatch_ids=(f"example-{index}",),
        reserved_child_id=child.id,
    )
    return EvaluatedProposal(
        task=task,
        parent_eval_result=make_eval(parent.id, before),
        child_candidate=child,
        child_eval_result=make_eval(child.id, after),
    )


def strict_criterion() -> ProposalAcceptanceCriterion:
    # The existing HELIX criterion consumes the two GEPA-compatible score
    # properties exposed by EvaluatedProposal.  The cast only bridges the
    # independently declared structural protocols for static type checkers.
    return cast(ProposalAcceptanceCriterion, StrictImprovementAcceptance())


def sampling_callbacks(
    parents: list[Candidate],
) -> tuple[MagicMock, MagicMock, MagicMock]:
    parent_index = 0
    minibatch_index = 0
    child_index = 0

    def select_parent() -> Candidate:
        nonlocal parent_index
        parent = parents[parent_index]
        parent_index += 1
        return parent

    def sample_minibatch() -> list[str]:
        nonlocal minibatch_index
        minibatch = [f"mb-{minibatch_index}"]
        minibatch_index += 1
        return minibatch

    def reserve_child_id() -> str:
        nonlocal child_index
        child_id = f"g1-s{child_index}"
        child_index += 1
        return child_id

    return (
        MagicMock(side_effect=select_parent),
        MagicMock(side_effect=sample_minibatch),
        MagicMock(side_effect=reserve_child_id),
    )


class TestSamplingStrategies:
    """Adapt GEPA's four proposal-sampling cases to HELIX candidates."""

    def test_single_mutation(self) -> None:
        parent = make_candidate("parent")
        select, minibatch, reserve = sampling_callbacks([parent])

        tasks = SingleMutationSampling().sample_tasks(
            select_parent=select,
            sample_minibatch=minibatch,
            reserve_child_id=reserve,
        )

        assert tasks == [
            ProposalTask(0, 0, 0, parent, ("mb-0",), "g1-s0")
        ]
        with pytest.raises(FrozenInstanceError):
            setattr(tasks[0], "batch_index", 10)

    def test_same_parent_sampling(self) -> None:
        parent = make_candidate("parent")
        select, minibatch, reserve = sampling_callbacks([parent])

        tasks = SameParentSampling(n=3).sample_tasks(
            select_parent=select,
            sample_minibatch=minibatch,
            reserve_child_id=reserve,
        )

        assert select.call_count == 1
        assert [task.parent_candidate for task in tasks] == [parent, parent, parent]
        assert [task.mutation_index for task in tasks] == [0, 1, 2]
        assert [task.minibatch_ids for task in tasks] == [
            ("mb-0",),
            ("mb-1",),
            ("mb-2",),
        ]

    def test_independent_sampling(self) -> None:
        parents = [make_candidate(f"parent-{index}") for index in range(4)]
        select, minibatch, reserve = sampling_callbacks(parents)

        tasks = IndependentSampling(n=4).sample_tasks(
            select_parent=select,
            sample_minibatch=minibatch,
            reserve_child_id=reserve,
        )

        assert select.call_count == 4
        assert [task.parent_candidate for task in tasks] == parents
        assert [task.parent_group_index for task in tasks] == [0, 1, 2, 3]
        assert [task.mutation_index for task in tasks] == [0, 0, 0, 0]

    def test_pxn_sampling_is_parent_major_and_allows_replacement(self) -> None:
        # Returning the same object twice demonstrates that parent selection is
        # with replacement; the strategy neither deduplicates nor reorders it.
        parent = make_candidate("parent")
        select, minibatch, reserve = sampling_callbacks([parent, parent])

        tasks = PxNSampling(p=2, n=3).sample_tasks(
            select_parent=select,
            sample_minibatch=minibatch,
            reserve_child_id=reserve,
        )

        assert len(tasks) == 6
        assert select.call_count == 2
        assert [task.batch_index for task in tasks] == list(range(6))
        assert [task.parent_group_index for task in tasks] == [0, 0, 0, 1, 1, 1]
        assert [task.mutation_index for task in tasks] == [0, 1, 2, 0, 1, 2]
        assert [task.reserved_child_id for task in tasks] == [
            "g1-s0",
            "g1-s1",
            "g1-s2",
            "g1-s3",
            "g1-s4",
            "g1-s5",
        ]
        assert sorted(reversed(tasks)) == tasks


class TestSelectionStrategies:
    """Adapt GEPA's four proposal-selection cases to EvalResult scores."""

    def test_all_improvements(self) -> None:
        proposals = [
            make_evaluated(0, 0.5, 0.8),
            make_evaluated(1, 0.5, 0.3),
            make_evaluated(2, 0.5, 0.6),
        ]

        selected = AllImprovements().select(proposals, strict_criterion())

        assert [item.proposal.task.batch_index for item in selected] == [0, 2]

    def test_best_improvement_is_stable_on_ties(self) -> None:
        proposals = [
            make_evaluated(0, 0.5, 0.9),
            make_evaluated(1, 0.4, 0.8),  # equal +0.4; first proposal wins
            make_evaluated(2, 0.5, 0.6),
        ]

        selected = BestImprovement().select(proposals, strict_criterion())

        assert len(selected) == 1
        assert selected[0].proposal.task.batch_index == 0
        assert selected[0].improvement == pytest.approx(0.4)

    def test_best_improvement_none_pass(self) -> None:
        proposals = [make_evaluated(0, 0.5, 0.3)]

        selected = BestImprovement().select(proposals, strict_criterion())

        assert selected == []

    def test_top_k_improvements_is_ranked_and_stable_on_ties(self) -> None:
        proposals = [
            make_evaluated(0, 0.5, 0.9),
            make_evaluated(1, 0.5, 0.6),
            make_evaluated(2, 0.4, 0.8),  # ties task 0 at +0.4
            make_evaluated(3, 0.5, 0.3),
        ]

        selected = TopKImprovements(k=2).select(proposals, strict_criterion())

        assert [item.proposal.task.batch_index for item in selected] == [0, 2]


class TestSequentialPlanningFallback:
    """Adapt GEPA's sequential fallback test to HELIX task preparation."""

    def test_sampling_callbacks_execute_sequentially(self) -> None:
        events: list[str] = []
        parent = make_candidate("parent")

        def select_parent() -> Candidate:
            events.append("parent")
            return parent

        def sample_minibatch() -> list[str]:
            events.append("minibatch")
            return [str(events.count("minibatch"))]

        def reserve_child_id() -> str:
            events.append("reserve")
            return f"child-{events.count('reserve')}"

        tasks = PxNSampling(p=2, n=2).sample_tasks(
            select_parent=select_parent,
            sample_minibatch=sample_minibatch,
            reserve_child_id=reserve_child_id,
        )

        assert len(tasks) == 4
        assert events == [
            "parent",
            "minibatch",
            "reserve",
            "minibatch",
            "reserve",
            "parent",
            "minibatch",
            "reserve",
            "minibatch",
            "reserve",
        ]


class TestDefaultStrategiesRetainBehavior:
    """Adapt GEPA's two default-strategy contract tests."""

    def test_single_mutation_is_default_sampling(self) -> None:
        assert isinstance(DEFAULT_SAMPLING_STRATEGY, SingleMutationSampling)
        assert hasattr(DEFAULT_SAMPLING_STRATEGY, "sample_tasks")

    def test_all_improvements_is_default_selection(self) -> None:
        assert isinstance(DEFAULT_SELECTION_STRATEGY, AllImprovements)
        assert hasattr(DEFAULT_SELECTION_STRATEGY, "select")


class TestTypedOutcomes:
    def test_outcomes_retain_immutable_task_and_typed_results(self) -> None:
        evaluated = make_evaluated(0, 0.5, 0.8)
        task = evaluated.task
        parent_result = evaluated.parent_eval_result
        child = evaluated.child_candidate

        skipped = SkippedProposal(task, parent_result, "perfect_parent")
        failed = FailedProposal(task, "mutation", "agent failed", parent_result)
        tampered = TamperedProposal(task, parent_result, child, ("evaluate.py",))
        selected = SelectedProposal(evaluated, evaluated.improvement)

        assert skipped.task is task
        assert failed.task is task
        assert tampered.task is task
        assert evaluated.task is task
        assert selected.proposal.task is task


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SameParentSampling(0),
        lambda: IndependentSampling(-1),
        lambda: PxNSampling(0, 1),
        lambda: PxNSampling(1, 0),
        lambda: TopKImprovements(0),
    ],
)
def test_strategy_widths_must_be_positive(factory: Callable[[], object]) -> None:
    with pytest.raises(ValueError, match="must be >= 1"):
        factory()
