"""Typed results of one proposal attempt.

A *proposal* is one (parent, minibatch) slot: evaluate the parent, reflect,
mutate, check the result for evaluator tampering, then evaluate the child.
Each attempt terminates in exactly one of the four results below.

These are a sealed union expressed as a class hierarchy rather than one
``kind``-discriminated dataclass: it gives ``mypy --strict`` the ``isinstance``
narrowing it needs to verify field access in the apply phase without asserts
or ``TypeGuard``s.

Budget charging, lineage writes and frontier updates are deliberately absent —
producing a proposal is pure with respect to run state, so the attempts can run
concurrently while the apply phase stays sequential and deterministic.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias

from helix.display import UsageStats
from helix.population import Candidate, EvalResult


# ``(parent, parent_frontier_result, subsample_ids, reserved_child_id)`` — the
# sequentially sampled context every attempt starts from.
ProposalContext: TypeAlias = tuple[
    Candidate, EvalResult | None, list[str] | None, str
]


@dataclass
class SkippedProposal:
    """Skip-perfect fired — the parent already scores perfectly; no LLM call."""

    presample_ctx: ProposalContext
    parent_eval_result: EvalResult
    parent_n_uncached: int = 0


@dataclass
class MutationFailedProposal:
    """``mutate()`` raised or returned None; ``parent_eval_result`` may be None."""

    presample_ctx: ProposalContext
    parent_eval_result: EvalResult | None
    parent_n_uncached: int = 0


@dataclass
class TamperedProposal:
    """The child modified protected evaluator files and must be rejected."""

    presample_ctx: ProposalContext
    parent_eval_result: EvalResult
    child: Candidate
    tampered_paths: list[str]
    parent_n_uncached: int = 0
    child_usage: UsageStats | None = None


@dataclass
class EvaluatedProposal:
    """Every step completed; ``child_eval_result`` is None on the no-minibatch path."""

    presample_ctx: ProposalContext
    parent_eval_result: EvalResult
    child: Candidate
    child_eval_result: EvalResult | None = None
    parent_n_uncached: int = 0
    child_n_uncached: int = 0
    child_usage: UsageStats | None = None


ProposalResult: TypeAlias = (
    SkippedProposal | MutationFailedProposal | TamperedProposal | EvaluatedProposal
)


# ---------------------------------------------------------------------------
# Acceptance: one judgement per proposal
# ---------------------------------------------------------------------------


@dataclass
class ScoreVectors:
    """The two score vectors an acceptance criterion compares.

    Field names and mutability match the ``_Proposal`` protocol in
    ``helix.eval_policy``, which this is passed to.  It replaces the
    ``SimpleNamespace`` the gate used to build, which satisfied that
    protocol only at runtime.
    """

    subsample_scores_before: list[float] | None
    subsample_scores_after: list[float] | None


class AcceptanceCriterion(Protocol):
    """Structural type of ``StrictImprovement`` / ``ImprovementOrEqual``."""

    def should_accept(self, proposal: ScoreVectors, /) -> bool: ...


@dataclass(frozen=True)
class AcceptanceJudgement:
    """What the criterion decided about one proposal, and by what margin."""

    accepted: bool
    before: list[float]
    after: list[float]

    @property
    def improvement(self) -> float:
        """``sum(after) - sum(before)`` — the margin the gate judged on.

        Selection ranks on this rather than on the child's absolute score so
        that a proposal from a weak parent competes on what it added, which
        is the same quantity the acceptance criterion tested.
        """
        return sum(self.after) - sum(self.before)


class AcceptanceMemo:
    """Memoises one acceptance judgement per proposal slot.

    ``best_improvement`` and ``top_k`` have to know the gate outcome for
    every proposal in a batch before they can promote any of it, so the
    criterion gets consulted in one place and the selected proposals get
    re-examined in another.  Routing every consultation through this memo
    makes the number of underlying ``should_accept`` calls a function of the
    batch size alone — a selection strategy cannot change what the criterion
    decided about a proposal by asking about it more or less often.
    """

    def __init__(self, criterion: AcceptanceCriterion) -> None:
        self._criterion = criterion
        self._judgements: dict[int, AcceptanceJudgement] = {}
        # Number of times the underlying criterion actually ran.  Asserted on
        # by the tests that pin the one-judgement-per-proposal property.
        self.criterion_calls = 0

    def judge(
        self, order: int, before: list[float], after: list[float]
    ) -> AcceptanceJudgement:
        """Return the judgement for slot ``order``, computing it at most once."""
        cached = self._judgements.get(order)
        if cached is not None:
            return cached
        self.criterion_calls += 1
        judgement = AcceptanceJudgement(
            accepted=self._criterion.should_accept(
                ScoreVectors(
                    subsample_scores_before=before,
                    subsample_scores_after=after,
                )
            ),
            before=before,
            after=after,
        )
        self._judgements[order] = judgement
        return judgement


# ---------------------------------------------------------------------------
# Selection: which gated proposals reach the frontier
# ---------------------------------------------------------------------------


SelectionStrategy: TypeAlias = Literal[
    "all_improvements", "best_improvement", "top_k"
]


@dataclass
class GatedProposal:
    """An evaluated proposal that cleared its acceptance gate.

    ``order`` is the slot's index in *sampled* order, not completion order.
    Every tie-break and the final apply order key off it, so a batch's
    outcome is a function of the seed alone.
    """

    order: int
    proposal: EvaluatedProposal
    judgement: AcceptanceJudgement
    gating_result: EvalResult
    # Ids the gate compared on; None on the no-minibatch path.
    subsample_ids: list[str] | None = field(default=None)

    @property
    def child(self) -> Candidate:
        return self.proposal.child

    @property
    def improvement(self) -> float:
        return self.judgement.improvement


def select_all_improvements(gated: list[GatedProposal]) -> list[GatedProposal]:
    """Promote every proposal that cleared the gate — the historical behaviour."""
    return list(gated)


def select_best_improvement(gated: list[GatedProposal]) -> list[GatedProposal]:
    """Promote only the largest improvement, earliest-sampled winning ties.

    ``max`` returns the first maximal element, so the winner is the first in
    sampled order and never the one whose worker happened to finish first.
    """
    if not gated:
        return []
    return [max(gated, key=lambda g: g.improvement)]


def select_top_k(gated: list[GatedProposal], k: int) -> list[GatedProposal]:
    """Promote the ``k`` largest improvements, earliest-sampled winning ties.

    The sort is stable over a list already in sampled order, so equal
    improvements resolve to the earlier slot.  Survivors stay in *ranked*
    order rather than being restored to sampled order: the apply phase can
    still stop early on budget exhaustion, and applying best-first means
    what survives a truncated batch is what the ranking preferred.
    """
    return sorted(gated, key=lambda g: -g.improvement)[:k]


def select_proposals(
    gated: list[GatedProposal],
    *,
    strategy: SelectionStrategy,
    top_k: int | None = None,
) -> list[GatedProposal]:
    """Apply the configured selection strategy to the gated proposals."""
    if strategy == "all_improvements":
        return select_all_improvements(gated)
    if strategy == "best_improvement":
        return select_best_improvement(gated)
    if top_k is None:  # pragma: no cover - config validation rejects this
        raise ValueError("proposal_selection='top_k' requires proposal_top_k")
    return select_top_k(gated, top_k)


def dedupe_children(
    selected: list[GatedProposal], content_key: Callable[[Candidate], str]
) -> tuple[list[GatedProposal], list[GatedProposal]]:
    """Split ``selected`` into first-seen children and byte-identical repeats.

    Two proposals in a batch can produce identical children — most easily
    two of the N siblings of one parent, which start from the same worktree
    and the same reflection.  Inserting both would put two frontier entries
    on one point, double-charge a full validation for a score already known,
    and skew parent selection toward whatever that point happens to be.

    Sampled order decides which copy survives, so the survivor does not
    depend on completion timing.
    """
    seen: dict[str, GatedProposal] = {}
    unique: list[GatedProposal] = []
    duplicates: list[GatedProposal] = []
    for candidate_proposal in selected:
        key = content_key(candidate_proposal.child)
        if key in seen:
            duplicates.append(candidate_proposal)
            continue
        seen[key] = candidate_proposal
        unique.append(candidate_proposal)
    return unique, duplicates
