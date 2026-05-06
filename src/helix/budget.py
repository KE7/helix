"""Central budget and progress accounting helpers for HELIX evolution."""

from __future__ import annotations

from typing import Any

from helix.config import HelixConfig
from helix.state import EvolutionState
from helix.trace import TRACE, EventType


def budget_exhausted(state: EvolutionState, config: HelixConfig) -> bool:
    """Return True if the configured evaluation budget is exhausted."""
    cap = config.evolution.max_evaluations
    return cap > 0 and state.budget.evaluations >= cap


def evaluation_budget_units(
    *, num_actual_examples: int | None = None, was_cached: bool = False
) -> int:
    """Return evaluation budget units for an evaluation attempt."""
    if was_cached:
        return 0
    if num_actual_examples is not None:
        return max(0, int(num_actual_examples))
    return 1


def charge_evaluation(
    state: EvolutionState,
    *,
    num_actual_examples: int | None = None,
    was_cached: bool = False,
    candidate_id: str | None = None,
    split: str | None = None,
    source: str | None = None,
) -> int:
    """Charge metric-call/evaluation budget using HELIX's existing semantics."""
    units = evaluation_budget_units(
        num_actual_examples=num_actual_examples,
        was_cached=was_cached,
    )
    state.budget.evaluations += units
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        candidate_id=candidate_id,
        split=split,
        decision=source,
        budget_delta=units,
        budget_evaluations=state.budget.evaluations,
    )
    return units


def charge_llm_usage(
    state: EvolutionState,
    usage: dict[str, Any] | None,
    *,
    candidate_id: str | None = None,
    source: str | None = None,
) -> None:
    """Charge token and cost counters from a backend usage payload."""
    if not usage:
        return
    input_delta = int(usage.get("input_tokens", 0))
    output_delta = int(usage.get("output_tokens", 0))
    cost_delta = float(usage.get("cost_usd", 0.0))
    state.budget.input_tokens += input_delta
    state.budget.output_tokens += output_delta
    state.budget.cost_usd += cost_delta
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        candidate_id=candidate_id,
        decision=source,
        input_tokens_delta=input_delta,
        output_tokens_delta=output_delta,
        cost_usd_delta=cost_delta,
        input_tokens=state.budget.input_tokens,
        output_tokens=state.budget.output_tokens,
        cost_usd=state.budget.cost_usd,
    )


def start_generation(state: EvolutionState, generation: int) -> None:
    """Record the active generation."""
    state.generation = generation
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        decision="generation",
        generation=state.generation,
    )


def advance_proposal_counter(state: EvolutionState, *, source: str) -> int:
    """Advance HELIX/GEPA's monotonic proposal counter."""
    state.i += 1
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        decision=source,
        proposal_index=state.i,
    )
    return state.i


def next_mutation_id(state: EvolutionState, generation: int) -> str:
    """Allocate the next mutation candidate id."""
    state.mutation_counter += 1
    mutation_id = f"g{generation}-s{state.mutation_counter}"
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        candidate_id=mutation_id,
        decision="mutation_id",
        mutation_counter=state.mutation_counter,
    )
    return mutation_id


def next_merge_id(state: EvolutionState, generation: int) -> str:
    """Allocate the next merge candidate id."""
    state.merge_counter += 1
    merge_id = f"g{generation}-m{state.merge_counter}"
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        candidate_id=merge_id,
        decision="merge_id",
        merge_counter=state.merge_counter,
    )
    return merge_id


def record_merge_invocation(state: EvolutionState) -> None:
    """Record one accepted merge invocation."""
    state.total_merge_invocations += 1
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        decision="merge_invocation",
        merge_invocations=state.total_merge_invocations,
    )


def record_discovery_budget(state: EvolutionState, candidate_id: str) -> None:
    """Stamp the evaluation budget at which a candidate entered the frontier."""
    state.num_metric_calls_by_discovery[candidate_id] = state.budget.evaluations
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        candidate_id=candidate_id,
        decision="discovery_budget",
        budget_evaluations=state.budget.evaluations,
    )
