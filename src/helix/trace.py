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
from typing import Any, Iterator


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


@dataclass
class Event:
    type: EventType
    candidate_id: str | None = None
    example_ids: list[Any] | None = None
    split: str | None = None
    hit_ids: list[Any] | None = None
    miss_ids: list[Any] | None = None
    decision: str | None = None
    score: float | None = None
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
