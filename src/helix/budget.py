"""Central budget and progress accounting helpers for HELIX evolution.

Threading model
---------------
None of these helpers lock ``state``; callers serialize access by calling
them only from the main evolution thread.  Today this invariant is
maintained by ``evolution.py``: every ``budget.*`` call site is in the
main loop, the presample/iteration loop, or a ``concurrent.futures.
as_completed`` consumer (which runs serially on the submitting thread).
The functions submitted to the parallel ``ThreadPoolExecutor``s
(``_eval_parent``, ``_do_mutate``) are state-free — they neither read
nor mutate ``EvolutionState``.

If you ever move a ``budget.*`` call into a worker function body,
introduce a ``threading.Lock`` here first; otherwise concurrent
``next_mutation_id`` / ``next_merge_id`` calls can produce duplicate
``g{gen}-s{n}`` ids and concurrent ``+=`` updates can drop charges.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

from helix.config import HelixConfig
from helix.display import UsageStats
from helix.state import EvolutionState
from helix.trace import TRACE, EventType


@dataclass(frozen=True)
class BatchBudgetGuard:
    """The explicit budget contract for one admitted proposal batch.

    A batch uses ``P = evolution.num_parallel_proposals`` parents sampled with
    replacement and ``N = evolution.mutations_per_parent`` reflective
    mutations per selected parent, yielding up to ``P*N`` proposal slots.

    Let ``U = max_in_flight_evaluations`` be the conservative number of
    uncached evaluation units that the batch can still consume after it has
    crossed a dispatch boundary. It is deliberately a unit bound, not a
    worker count: one evaluator invocation can account for several minibatch
    examples.

    """

    evaluations_before: int
    max_evaluations: int
    max_in_flight_evaluations: int
    maximum_overshoot: int


def begin_batch_budget_guard(
    state: EvolutionState,
    *,
    max_evaluations: int,
    max_in_flight_evaluations: int,
) -> BatchBudgetGuard:
    """Record the admissible in-flight work and overshoot for a batch.

    With a positive cap, a batch starts only below that cap. Once admitted,
    its in-flight work is allowed to drain. Thus a batch with ``U`` units can
    overshoot by at most ``max(0, U - 1)`` in the worst case (it starts one
    unit below the cap). A non-positive cap is unlimited.

    Raises:
        ValueError: if the caller supplies a negative in-flight unit bound.
    """
    if max_in_flight_evaluations < 0:
        raise ValueError(
            "Batch in-flight evaluation bound must be non-negative "
            f"(actual {max_in_flight_evaluations}, permitted >= 0)"
        )

    evaluations_before = state.budget.evaluations
    maximum_overshoot = (
        0
        if max_evaluations <= 0
        else max(0, evaluations_before + max_in_flight_evaluations - max_evaluations)
    )
    return BatchBudgetGuard(
        evaluations_before=evaluations_before,
        max_evaluations=max_evaluations,
        max_in_flight_evaluations=max_in_flight_evaluations,
        maximum_overshoot=maximum_overshoot,
    )


def enforce_batch_budget_guard(
    state: EvolutionState,
    guard: BatchBudgetGuard,
    *,
    actual_in_flight_evaluations: int,
) -> None:
    """Reject a batch whose observed work exceeds its declared budget bounds.

    Both failures name the observed and allowed values so an accounting
    regression cannot silently turn a documented bound into best-effort
    behaviour.
    """
    if actual_in_flight_evaluations > guard.max_in_flight_evaluations:
        raise ValueError(
            "Batch in-flight evaluation units exceeded their bound: "
            f"actual {actual_in_flight_evaluations}, permitted "
            f"{guard.max_in_flight_evaluations}"
        )

    if guard.max_evaluations <= 0:
        return

    actual_overshoot = max(0, state.budget.evaluations - guard.max_evaluations)
    if actual_overshoot > guard.maximum_overshoot:
        raise ValueError(
            "Batch evaluation-budget overshoot exceeded its bound: "
            f"actual {actual_overshoot}, permitted {guard.maximum_overshoot}"
        )


def budget_exhausted(state: EvolutionState, config: HelixConfig) -> bool:
    """Return True iff the configured evaluation budget cap is set and reached.

    A non-positive ``config.evolution.max_evaluations`` (``<= 0``)
    disables the cap entirely; HELIX then runs until ``max_generations``
    alone and this predicate always returns ``False``.  Mirrors GEPA's
    metric-call cap (``optimize_anything.py`` ``max_metric_calls``).
    """
    cap = config.evolution.max_evaluations
    return cap > 0 and state.budget.evaluations >= cap


def evaluation_budget_units(
    *, num_actual_examples: int | None = None, was_cached: bool = False
) -> int:
    """Return evaluation budget units for an evaluation attempt.

    Cached results charge 0 units; minibatch evals charge one unit per
    uncached example (clamped at 0); single-task / no-example paths charge 1
    (one evaluator invocation).
    """
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
    """Charge metric-call/evaluation budget using HELIX's existing semantics.

    No-op charges (``units == 0`` from a cache hit or a 0-example
    minibatch) update neither the counter nor the trace stream — this
    keeps ``BUDGET_UPDATE`` events focused on actual budget consumption
    and avoids inflating long-run trace files.  Use the ``source``
    argument to label the call site (e.g., ``"seed_val"``,
    ``"merge_subsample"``, ``"mutation_minibatch_gate"``); it is
    forwarded to ``Event.reason``.
    """
    units = evaluation_budget_units(
        num_actual_examples=num_actual_examples,
        was_cached=was_cached,
    )
    if units == 0:
        return 0
    state.budget.evaluations += units
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        candidate_id=candidate_id,
        split=split,
        reason=source,
        budget_delta=units,
        budget_evaluations=state.budget.evaluations,
    )
    return units


def charge_llm_usage(
    state: EvolutionState,
    usage: UsageStats | None,
    *,
    candidate_id: str | None = None,
    source: str | None = None,
) -> None:
    """Charge token and cost counters from a ``UsageStats`` payload.

    Safe to call with ``None``; short-circuits without touching counters
    or emitting a trace event.  A non-None ``UsageStats`` with all-zero
    deltas still emits one ``BUDGET_UPDATE`` event (the call indicates a
    real backend response was processed).
    """
    if usage is None:
        return
    input_delta = usage.input_tokens
    output_delta = usage.output_tokens
    cached_input_delta = usage.cached_input_tokens
    cache_creation_delta = usage.cache_creation_input_tokens
    cache_read_delta = usage.cache_read_input_tokens
    reasoning_delta = usage.reasoning_tokens
    cost_delta = usage.cost_usd
    state.budget.input_tokens += input_delta
    state.budget.output_tokens += output_delta
    state.budget.cached_input_tokens += cached_input_delta
    state.budget.cache_creation_input_tokens += cache_creation_delta
    state.budget.cache_read_input_tokens += cache_read_delta
    state.budget.reasoning_tokens += reasoning_delta
    state.budget.cost_usd += cost_delta
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        candidate_id=candidate_id,
        reason=source,
        input_tokens_delta=input_delta,
        output_tokens_delta=output_delta,
        cost_usd_delta=cost_delta,
        input_tokens=state.budget.input_tokens,
        output_tokens=state.budget.output_tokens,
        cost_usd=state.budget.cost_usd,
    )


def set_generation(state: EvolutionState, generation: int) -> None:
    """Record the active generation."""
    state.generation = generation
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        reason="generation",
        generation=state.generation,
    )


def advance_proposal_counter(state: EvolutionState, *, source: str) -> int:
    """Advance HELIX/GEPA's monotonic proposal counter.

    ``EvolutionState.i`` defaults to ``-1`` (GEPA ``core/state.py``
    parity); the first call therefore yields ``0``.
    """
    state.i += 1
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        reason=source,
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
        reason="mutation_id",
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
        reason="merge_id",
        merge_counter=state.merge_counter,
    )
    return merge_id


def record_merge_invocation(state: EvolutionState) -> None:
    """Record one accepted merge invocation."""
    state.total_merge_invocations += 1
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        reason="merge_invocation",
        merge_invocations=state.total_merge_invocations,
    )


def record_discovery_budget(state: EvolutionState, candidate_id: str) -> None:
    """Stamp the evaluation budget at which a candidate entered the frontier.

    Warns (without raising) when ``candidate_id`` has already been
    recorded.  A duplicate stamp overwrites the original discovery
    budget and almost always indicates a real bug in the accept path —
    each accepted candidate should hit this exactly once.  Resume
    paths that legitimately re-process a candidate should skip this
    call rather than rely on overwrite semantics.
    """
    if candidate_id in state.num_metric_calls_by_discovery:
        warnings.warn(
            f"discovery budget already recorded for {candidate_id!r}: "
            f"overwriting "
            f"{state.num_metric_calls_by_discovery[candidate_id]} "
            f"-> {state.budget.evaluations}",
            stacklevel=2,
        )
    state.num_metric_calls_by_discovery[candidate_id] = state.budget.evaluations
    TRACE.emit(
        EventType.BUDGET_UPDATE,
        candidate_id=candidate_id,
        reason="discovery_budget",
        budget_evaluations=state.budget.evaluations,
    )
