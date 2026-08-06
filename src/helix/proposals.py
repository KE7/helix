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

from dataclasses import dataclass
from typing import TypeAlias

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
