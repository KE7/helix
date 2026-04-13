"""HELIX evaluation policy — GEPA parity (design §3, §5).

Line-for-line port of GEPA's full-eval policy and acceptance criteria.
Kept stdlib-only and mypy --strict compatible.
"""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class _Loader(Protocol):
    def all_ids(self) -> list[Any]: ...


@runtime_checkable
class _PolicyState(Protocol):
    # list of dict[example_id, score] per program candidate
    prog_candidate_val_subscores: list[dict[Any, float]]


@runtime_checkable
class EvaluationPolicy(Protocol):
    def get_eval_batch(
        self,
        loader: _Loader,
        state: _PolicyState,
        target_program_idx: int | None = None,
    ) -> list[Any]: ...

    def get_best_program(self, state: _PolicyState) -> int: ...

    def get_valset_score(self, program_idx: int, state: _PolicyState) -> float: ...


class FullEvaluationPolicy:
    """Always return the full valset — the only policy used by optimize_anything.

    GEPA §4.2: ``get_eval_batch`` returns ``list(loader.all_ids())`` and
    ignores both ``state`` and ``target_program_idx``.
    """

    def get_eval_batch(
        self,
        loader: _Loader,
        state: _PolicyState,
        target_program_idx: int | None = None,
    ) -> list[Any]:
        return list(loader.all_ids())

    def get_best_program(self, state: _PolicyState) -> int:
        best_idx: int = -1
        best_score: float = float("-inf")
        best_coverage: int = -1
        for program_idx, scores in enumerate(state.prog_candidate_val_subscores):
            coverage = len(scores)
            if coverage == 0:
                avg = float("-inf")
            else:
                avg = sum(scores.values()) / coverage
            if avg > best_score or (avg == best_score and coverage > best_coverage):
                best_score = avg
                best_idx = program_idx
                best_coverage = coverage
        return best_idx

    def get_valset_score(self, program_idx: int, state: _PolicyState) -> float:
        scores = state.prog_candidate_val_subscores[program_idx]
        if not scores:
            return float("-inf")
        return sum(scores.values()) / len(scores)


@runtime_checkable
class _Proposal(Protocol):
    subsample_scores_before: list[float] | None
    subsample_scores_after: list[float] | None


class StrictImprovementAcceptance:
    """``sum(new) > sum(old)`` on the SAME minibatch — GEPA §5.1."""

    def should_accept(self, proposal: _Proposal) -> bool:
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        return new_sum > old_sum


class ImprovementOrEqualAcceptance:
    """``sum(new) >= sum(old)`` on the SAME minibatch — GEPA §5.1 (relaxed)."""

    def should_accept(self, proposal: _Proposal) -> bool:
        old_sum = sum(proposal.subsample_scores_before or [])
        new_sum = sum(proposal.subsample_scores_after or [])
        return new_sum >= old_sum
