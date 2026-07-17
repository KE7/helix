"""Typed proposal sampling, outcomes, and selection strategies.

Proposal planning is intentionally sequential.  A sampling strategy records
all parent choices, minibatches, and child IDs before evolution starts any
concurrent work.  The explicit ``batch_index`` on each task then provides the
canonical order for evaluation alignment, selection ties, and state updates.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

from helix.population import Candidate, EvalResult


MinibatchIds: TypeAlias = tuple[str, ...] | None
ProposalFailureStage: TypeAlias = Literal[
    "parent_evaluation",
    "mutation",
    "child_evaluation",
    "full_validation",
]


@dataclass(frozen=True, slots=True, order=True)
class ProposalTask:
    """One immutable, pre-reserved child proposal in a P-by-N batch.

    ``batch_index`` is the task's parent-major position.  The parent candidate
    is excluded from generated comparisons because ``Candidate`` is mutable
    and not orderable; all scheduling metadata remains part of the ordering.
    Minibatch IDs are stored as a tuple so neither completion order nor caller
    mutation can change the planned evaluation.
    """

    batch_index: int
    parent_group_index: int
    mutation_index: int
    parent_candidate: Candidate = field(compare=False)
    minibatch_ids: MinibatchIds = field(compare=False)
    reserved_child_id: str

    def __post_init__(self) -> None:
        for name, value in (
            ("batch_index", self.batch_index),
            ("parent_group_index", self.parent_group_index),
            ("mutation_index", self.mutation_index),
        ):
            if value < 0:
                raise ValueError(f"{name} must be >= 0 (got {value})")
        if not self.reserved_child_id:
            raise ValueError("reserved_child_id must be non-empty")


@dataclass(frozen=True, slots=True)
class SkippedProposal:
    """A proposal skipped before a child candidate was produced."""

    task: ProposalTask
    parent_eval_result: EvalResult
    reason: str
    parent_n_uncached: int = 0


@dataclass(frozen=True, slots=True)
class FailedProposal:
    """A proposal whose isolated work failed without invalidating siblings."""

    task: ProposalTask
    stage: ProposalFailureStage
    message: str
    parent_eval_result: EvalResult | None = None
    child_candidate: Candidate | None = None
    parent_n_uncached: int = 0
    child_n_uncached: int = 0


@dataclass(frozen=True, slots=True)
class TamperedProposal:
    """A child rejected for modifying protected evaluator files."""

    task: ProposalTask
    parent_eval_result: EvalResult
    child_candidate: Candidate
    tampered_paths: tuple[str, ...]
    parent_n_uncached: int = 0


@dataclass(frozen=True, slots=True)
class EvaluatedProposal:
    """A child and its parent evaluated on the task's minibatch."""

    task: ProposalTask
    parent_eval_result: EvalResult
    child_candidate: Candidate
    child_eval_result: EvalResult
    parent_n_uncached: int = 0
    child_n_uncached: int = 0

    def _scores_for_task(self, result: EvalResult) -> list[float]:
        """Return scores in task order, retaining padded ID multiplicity.

        ``EvalResult.instance_scores`` is a mapping and therefore has one
        entry per unique example ID.  The epoch sampler may pad a minibatch by
        repeating an ID, so the task's immutable ID tuple is the authoritative
        positional view for acceptance and ranking.
        """

        if self.task.minibatch_ids is None:
            return list(result.instance_scores.values())
        return [result.instance_scores[eid] for eid in self.task.minibatch_ids]

    @property
    def subsample_scores_before(self) -> list[float]:
        """GEPA-compatible view used by HELIX acceptance criteria."""

        return self._scores_for_task(self.parent_eval_result)

    @property
    def subsample_scores_after(self) -> list[float]:
        """GEPA-compatible view used by HELIX acceptance criteria."""

        return self._scores_for_task(self.child_eval_result)

    @property
    def improvement(self) -> float:
        """Return the child's minibatch score gain over its parent."""

        return sum(self.subsample_scores_after) - sum(self.subsample_scores_before)


@dataclass(frozen=True, slots=True)
class SelectedProposal:
    """An evaluated proposal selected for the full-validation phase."""

    proposal: EvaluatedProposal
    improvement: float


TerminalProposalOutcome: TypeAlias = (
    SkippedProposal | FailedProposal | TamperedProposal | EvaluatedProposal
)
ProposalOutcome: TypeAlias = TerminalProposalOutcome | SelectedProposal


ParentSelector: TypeAlias = Callable[[], Candidate]
MinibatchSampler: TypeAlias = Callable[[], Sequence[str] | None]
ChildIdReservation: TypeAlias = Callable[[], str]


class ProposalSamplingStrategy(Protocol):
    """Plan a deterministic sequence of proposal tasks."""

    def sample_tasks(
        self,
        *,
        select_parent: ParentSelector,
        sample_minibatch: MinibatchSampler,
        reserve_child_id: ChildIdReservation,
    ) -> list[ProposalTask]: ...


def _require_positive(name: str, value: int) -> None:
    if value < 1:
        raise ValueError(f"{name} must be >= 1 (got {value})")


def _sample_pxn(
    p: int,
    n: int,
    *,
    select_parent: ParentSelector,
    sample_minibatch: MinibatchSampler,
    reserve_child_id: ChildIdReservation,
) -> list[ProposalTask]:
    """Sample P parent groups with N tasks each in parent-major order."""

    tasks: list[ProposalTask] = []
    for parent_group_index in range(p):
        # Selection is deliberately inside the group loop: repeated Candidate
        # objects are valid and represent frontier sampling with replacement.
        parent = select_parent()
        for mutation_index in range(n):
            minibatch = sample_minibatch()
            reserved_child_id = reserve_child_id()
            tasks.append(
                ProposalTask(
                    batch_index=len(tasks),
                    parent_group_index=parent_group_index,
                    mutation_index=mutation_index,
                    parent_candidate=parent,
                    minibatch_ids=None if minibatch is None else tuple(minibatch),
                    reserved_child_id=reserved_child_id,
                )
            )
    return tasks


class SingleMutationSampling:
    """Plan one mutation from one selected parent (the default shape)."""

    def sample_tasks(
        self,
        *,
        select_parent: ParentSelector,
        sample_minibatch: MinibatchSampler,
        reserve_child_id: ChildIdReservation,
    ) -> list[ProposalTask]:
        return _sample_pxn(
            1,
            1,
            select_parent=select_parent,
            sample_minibatch=sample_minibatch,
            reserve_child_id=reserve_child_id,
        )


@dataclass(frozen=True, slots=True)
class SameParentSampling:
    """Plan N mutations from one selected parent."""

    n: int

    def __post_init__(self) -> None:
        _require_positive("n", self.n)

    def sample_tasks(
        self,
        *,
        select_parent: ParentSelector,
        sample_minibatch: MinibatchSampler,
        reserve_child_id: ChildIdReservation,
    ) -> list[ProposalTask]:
        return _sample_pxn(
            1,
            self.n,
            select_parent=select_parent,
            sample_minibatch=sample_minibatch,
            reserve_child_id=reserve_child_id,
        )


@dataclass(frozen=True, slots=True)
class IndependentSampling:
    """Plan one mutation for each of N independently selected parents."""

    n: int

    def __post_init__(self) -> None:
        _require_positive("n", self.n)

    def sample_tasks(
        self,
        *,
        select_parent: ParentSelector,
        sample_minibatch: MinibatchSampler,
        reserve_child_id: ChildIdReservation,
    ) -> list[ProposalTask]:
        return _sample_pxn(
            self.n,
            1,
            select_parent=select_parent,
            sample_minibatch=sample_minibatch,
            reserve_child_id=reserve_child_id,
        )


@dataclass(frozen=True, slots=True)
class PxNSampling:
    """Plan P parent selections by N mutations in parent-major order."""

    p: int
    n: int

    def __post_init__(self) -> None:
        _require_positive("p", self.p)
        _require_positive("n", self.n)

    def sample_tasks(
        self,
        *,
        select_parent: ParentSelector,
        sample_minibatch: MinibatchSampler,
        reserve_child_id: ChildIdReservation,
    ) -> list[ProposalTask]:
        return _sample_pxn(
            self.p,
            self.n,
            select_parent=select_parent,
            sample_minibatch=sample_minibatch,
            reserve_child_id=reserve_child_id,
        )


class ProposalAcceptanceCriterion(Protocol):
    """Acceptance gate applied before ranking evaluated proposals."""

    def should_accept(self, proposal: EvaluatedProposal) -> bool: ...


class ProposalSelectionStrategy(Protocol):
    """Filter and rank evaluated proposals for full validation."""

    def select(
        self,
        proposals: Sequence[EvaluatedProposal],
        acceptance_criterion: ProposalAcceptanceCriterion,
    ) -> list[SelectedProposal]: ...


def _selected(proposal: EvaluatedProposal) -> SelectedProposal:
    return SelectedProposal(proposal=proposal, improvement=proposal.improvement)


class AllImprovements:
    """Select every accepted proposal, preserving input order."""

    def select(
        self,
        proposals: Sequence[EvaluatedProposal],
        acceptance_criterion: ProposalAcceptanceCriterion,
    ) -> list[SelectedProposal]:
        return [
            _selected(proposal)
            for proposal in proposals
            if acceptance_criterion.should_accept(proposal)
        ]


class BestImprovement:
    """Select the accepted proposal with the greatest score gain."""

    def select(
        self,
        proposals: Sequence[EvaluatedProposal],
        acceptance_criterion: ProposalAcceptanceCriterion,
    ) -> list[SelectedProposal]:
        best: EvaluatedProposal | None = None
        best_improvement = float("-inf")
        for proposal in proposals:
            if not acceptance_criterion.should_accept(proposal):
                continue
            improvement = proposal.improvement
            # Strict comparison is intentional: an exact tie retains the
            # first proposal in sampled order.
            if improvement > best_improvement:
                best = proposal
                best_improvement = improvement
        return [_selected(best)] if best is not None else []


@dataclass(frozen=True, slots=True)
class TopKImprovements:
    """Select up to K accepted proposals by descending score gain."""

    k: int

    def __post_init__(self) -> None:
        _require_positive("k", self.k)

    def select(
        self,
        proposals: Sequence[EvaluatedProposal],
        acceptance_criterion: ProposalAcceptanceCriterion,
    ) -> list[SelectedProposal]:
        passing = [
            proposal
            for proposal in proposals
            if acceptance_criterion.should_accept(proposal)
        ]
        # Python's sort is stable, so equally improving proposals retain their
        # sampled order and completion timing cannot affect the tie-break.
        passing.sort(key=lambda proposal: proposal.improvement, reverse=True)
        return [_selected(proposal) for proposal in passing[: self.k]]


DEFAULT_SAMPLING_STRATEGY: ProposalSamplingStrategy = SingleMutationSampling()
DEFAULT_SELECTION_STRATEGY: ProposalSelectionStrategy = AllImprovements()
