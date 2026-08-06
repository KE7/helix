"""HELIX TraceBus — lightweight runtime event stream for differential testing.

Zero overhead when disabled: ``TRACE.emit(...)`` short-circuits on a single
attribute check (``self.enabled``) before building any event payload.

There are two ways to turn the bus on:

``record()``
    Flips the flag, yields an in-memory ``events`` list, and restores the flag
    (and event buffer) on exit.  Used by the differential-testing harness and
    the unit tests, which want the events as objects.

``write_jsonl(path)``
    Flips the flag and streams every event to *path* as JSON Lines, one object
    per line, flushed as it is emitted.  This is what ``helix evolve --trace``
    uses.  Streaming (rather than buffering and dumping at the end) keeps a
    long run's memory bounded and leaves a usable trace behind even if the run
    crashes or is killed.

Event points are sprinkled throughout ``evolution.py``, ``eval_cache.py``,
``executor.py``, ``batch_sampler.py``, and ``mutator.py``.  The GEPA
differential harness consumes these events to assert runtime parity against
the GEPA reference engine.
"""
from __future__ import annotations

import json
import sys
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from enum import Enum
from pathlib import Path
from typing import Any, IO, Iterator


class EventType(str, Enum):
    OPT_START = "OPT_START"
    ITER_START = "ITER_START"
    SAMPLE_MINIBATCH = "SAMPLE_MINIBATCH"
    EVAL_START = "EVAL_START"
    CACHE_GET = "CACHE_GET"
    CACHE_PUT = "CACHE_PUT"
    EVAL_END = "EVAL_END"
    MUTATE_START = "MUTATE_START"
    MUTATE_END = "MUTATE_END"
    ACCEPT_DECISION = "ACCEPT_DECISION"
    FRONTIER_UPDATE = "FRONTIER_UPDATE"
    ITER_END = "ITER_END"
    OPT_END = "OPT_END"
    BUDGET_UPDATE = "BUDGET_UPDATE"


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
    # frame captured when enabled.
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
    mutation_counter: int | None = None
    merge_counter: int | None = None
    merge_invocations: int | None = None
    source: str | None = None  # "file:line" — captured from the caller frame

    # --- timing / provenance -------------------------------------------------
    # Two clocks, deliberately.  ``wall_time`` is Unix epoch seconds from
    # ``time.time()``: the only value that can be lined up against an external
    # log, a container's stdout, or a wall-clock incident timeline.  It is also
    # the only one that may go *backwards* — NTP slew and DST-adjacent system
    # clock corrections both do that — so it must never be subtracted to get a
    # duration.  ``monotonic`` is ``time.monotonic()``: meaningless as an
    # absolute instant (its epoch is arbitrary and process-local) but
    # guaranteed non-decreasing, so START/END deltas computed from it are
    # trustworthy.  Recording both costs 16 bytes and removes the choice
    # between "correlatable" and "correct".
    wall_time: float = field(default_factory=time.time)
    monotonic: float = field(default_factory=time.monotonic)
    # Emits arrive from proposal-worker threads (``run_evaluator`` emits
    # EVAL_START/EVAL_END, and it is reached from a ThreadPoolExecutor when
    # more than one proposal runs at a time).  Events therefore land in the
    # file in *completion* order, and a START can be separated from its END by
    # any number of a sibling's events.  The thread id is what makes a pair
    # re-associable with its own worker after the fact; without it, per-slot
    # durations are unrecoverable whenever more than one worker is live.
    thread_id: int = field(default_factory=threading.get_ident)


def _event_to_dict(event: Event) -> dict[str, Any]:
    """JSON-ready mapping for *event*, omitting fields left at ``None``.

    Dropping the unset optionals matters: ``Event`` carries ~25 mostly-None
    fields and a long run emits a lot of them.  Consumers should treat a
    missing key as None.
    """
    out: dict[str, Any] = {}
    for f in dataclass_fields(event):
        value = getattr(event, f.name)
        if value is None:
            continue
        out[f.name] = value.value if isinstance(value, EventType) else value
    return out


class TraceBus:
    """Module-level singleton — see ``TRACE`` below."""

    def __init__(self) -> None:
        self.enabled: bool = False
        self.events: list[Event] = []
        # When False, emitted events are not retained in ``self.events``.
        # ``write_jsonl`` clears it so a long run does not accumulate every
        # event it ever emitted in memory; ``record()`` leaves it True.
        self._collect: bool = True
        self._sink: IO[str] | None = None
        # Guards the sink.  ``list.append`` is atomic under the GIL so
        # ``self.events`` needs no lock, but a file write is not: two threads
        # writing concurrently can interleave partial lines and corrupt the
        # JSONL.
        self._sink_lock = threading.Lock()

    def emit(self, type: EventType, **fields: Any) -> None:
        if not self.enabled:
            return
        # Capture caller file:line so divergence messages can point at the
        # exact guilty source location.  ``sys._getframe(1)`` reads one frame
        # in O(1); ``inspect.stack()`` walks the entire stack and materialises
        # a FrameInfo (including source-line lookups) for every level of it,
        # which costs hundreds of microseconds per call in a deep call stack.
        # That was tolerable when tracing was test-only; it is not once
        # ``--trace`` can be switched on for a real run.  The rendered string
        # keeps the same ``"file:line"`` shape, and the same value: what
        # inspect reported as ``FrameInfo.filename`` for an ordinary .py
        # module is exactly ``f_code.co_filename``.  Existing consumers of
        # ``source`` see no change.
        frame = sys._getframe(1)
        source = f"{frame.f_code.co_filename}:{frame.f_lineno}"
        event = Event(type=type, source=source, **fields)
        if self._collect:
            self.events.append(event)
        if self._sink is not None:
            # Serialise outside the lock; hold it only for the write.
            line = json.dumps(_event_to_dict(event), default=str)
            with self._sink_lock:
                # Re-check under the lock: ``write_jsonl`` clears the sink
                # here before closing the file, so a thread that read a
                # non-None sink a moment ago cannot write to a closed handle.
                if self._sink is not None:
                    self._sink.write(line + "\n")
                    self._sink.flush()

    @contextmanager
    def record(self) -> Iterator[list[Event]]:
        """Enable the bus, yield the in-memory event list, restore on exit."""
        prev_enabled = self.enabled
        prev_events = self.events
        prev_collect = self._collect
        self.enabled = True
        self._collect = True
        self.events = []
        try:
            yield self.events
        finally:
            self.enabled = prev_enabled
            self.events = prev_events
            self._collect = prev_collect

    @contextmanager
    def write_jsonl(self, path: str | Path) -> Iterator[Path]:
        """Enable the bus and stream events to *path* as JSON Lines.

        Each event is written and flushed as it is emitted, so a run that is
        killed part-way still leaves every event it got to on disk.  Yields
        the resolved path.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        prev_enabled = self.enabled
        prev_events = self.events
        prev_collect = self._collect
        prev_sink = self._sink
        handle = target.open("w", encoding="utf-8")
        self.events = []
        self._collect = False
        with self._sink_lock:
            self._sink = handle
        self.enabled = True
        try:
            yield target
        finally:
            self.enabled = prev_enabled
            with self._sink_lock:
                self._sink = prev_sink
            handle.close()
            self._collect = prev_collect
            self.events = prev_events


TRACE = TraceBus()
