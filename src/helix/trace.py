"""HELIX TraceBus — lightweight runtime event stream for differential testing.

Zero overhead when disabled: ``TRACE.emit(...)`` short-circuits on a single
attribute check (``self.enabled``) before building any event payload.

Enable via the ``record()`` context manager — which flips the flag, yields an
in-memory ``events`` list, and restores the flag (and event buffer) on exit.

Event points are sprinkled throughout ``evolution.py``, ``eval_cache.py``,
``executor.py``, ``batch_sampler.py``, and ``mutator.py``.  The GEPA
differential harness consumes these events to assert runtime parity against
the GEPA reference engine.
"""

from __future__ import annotations

import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from helix.state import ProposalBatchRecord


class EventType(str, Enum):
    OPT_START = "OPT_START"
    ITER_START = "ITER_START"
    SAMPLE_MINIBATCH = "SAMPLE_MINIBATCH"
    EVAL_START = "EVAL_START"
    CACHE_GET = "CACHE_GET"
    CACHE_PUT = "CACHE_PUT"
    EVAL_END = "EVAL_END"
    ACCEPT_DECISION = "ACCEPT_DECISION"
    FRONTIER_UPDATE = "FRONTIER_UPDATE"
    ITER_END = "ITER_END"
    OPT_END = "OPT_END"
    BUDGET_UPDATE = "BUDGET_UPDATE"
    PROPOSAL_BATCH_START = "PROPOSAL_BATCH_START"
    PROPOSAL_TASK_TERMINAL = "PROPOSAL_TASK_TERMINAL"
    PROPOSAL_BATCH_END = "PROPOSAL_BATCH_END"


@dataclass
class Event:
    type: EventType
    candidate_id: str | None = None
    example_ids: list[Any] | None = None
    split: str | None = None
    hit_ids: list[Any] | None = None
    miss_ids: list[Any] | None = None
    decision: str | None = None
    # Human-readable label for *why* an event was emitted (e.g., the
    # ``charge_evaluation`` source: "seed_val", "merge_subsample",
    # "mutation_minibatch_gate", ...).  Distinct from ``decision``,
    # which carries iteration-level accept/reject text, and from
    # ``source`` below, which is reserved for the ``"file:line"`` stack
    # frame captured by ``inspect`` when enabled.
    reason: str | None = None
    score: float | None = None
    budget_delta: int | None = None
    budget_evaluations: int | None = None
    input_tokens_delta: int | None = None
    output_tokens_delta: int | None = None
    cost_usd_delta: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None
    generation: int | None = None
    proposal_index: int | None = None
    batch_id: str | None = None
    p: int | None = None
    n: int | None = None
    task_index: int | None = None
    parent_group: int | None = None
    mutation_index: int | None = None
    parent_id: str | None = None
    child_id: str | None = None
    status: str | None = None
    score_delta: float | None = None
    selection: str | None = None
    cleanup: str | None = None
    mutation_counter: int | None = None
    merge_counter: int | None = None
    merge_invocations: int | None = None
    source: str | None = None  # "file:line" — captured via inspect when enabled


class TraceBus:
    """Module-level singleton — see ``TRACE`` below."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.events: list[Event] = []

    def emit(self, type: EventType, **fields: Any) -> None:
        if not self.enabled:
            return
        # Capture caller file:line so divergence messages can point at the
        # exact guilty source location.
        frame = inspect.stack()[1]
        source = f"{frame.filename}:{frame.lineno}"
        self.events.append(Event(type=type, source=source, **fields))

    def emit_proposal_batch_terminal(self, batch: ProposalBatchRecord) -> None:
        """Emit one ordered terminal event for every planned P*N slot.

        A partial batch is rejected instead of producing a misleading trace.
        Consumers can rely on the interval between ``PROPOSAL_BATCH_START``
        and ``PROPOSAL_BATCH_END`` containing exactly ``P*N`` terminal task
        events in parent-major order.
        """
        if not self.enabled:
            return
        batch.validate_plan()
        nonterminal = [
            task.task_index for task in batch.tasks if not task.is_terminal()
        ]
        if nonterminal:
            raise ValueError(
                f"Cannot trace proposal batch {batch.batch_id!r}; nonterminal slots: "
                + ", ".join(str(index) for index in nonterminal)
            )
        self.emit(
            EventType.PROPOSAL_BATCH_START,
            batch_id=batch.batch_id,
            p=batch.p,
            n=batch.n,
            generation=batch.generation,
        )
        for task in batch.tasks:
            self.emit(
                EventType.PROPOSAL_TASK_TERMINAL,
                batch_id=task.batch_id,
                p=task.p,
                n=task.n,
                task_index=task.task_index,
                proposal_index=task.task_index,
                parent_group=task.parent_group,
                mutation_index=task.mutation_index,
                parent_id=task.parent_id,
                candidate_id=task.child_id,
                child_id=task.child_id,
                status=task.status,
                score_delta=task.score_delta,
                selection=task.selection,
                cleanup=task.cleanup,
                generation=batch.generation,
            )
        self.emit(
            EventType.PROPOSAL_BATCH_END,
            batch_id=batch.batch_id,
            p=batch.p,
            n=batch.n,
            generation=batch.generation,
            decision=batch.phase,
        )

    @contextmanager
    def record(self) -> Iterator[list[Event]]:
        """Enable the bus, yield the in-memory event list, restore on exit."""
        prev_enabled = self.enabled
        prev_events = self.events
        self.enabled = True
        self.events = []
        try:
            yield self.events
        finally:
            self.enabled = prev_enabled
            self.events = prev_events


TRACE = TraceBus()
