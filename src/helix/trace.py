"""HELIX TraceBus — lightweight runtime event stream for differential testing.

The second half of this module (``traced`` / ``enable``) is the JSONL timing
trace behind ``helix evolve --trace PATH``; it shares nothing with the bus
except this file.

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

import functools
import inspect
import itertools
import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterator, ParamSpec, TextIO, TypeVar

logger = logging.getLogger(__name__)


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


# ---------------------------------------------------------------------------
# JSONL timing spans — ``helix evolve --trace PATH`` / ``HELIX_TRACE=PATH``
# ---------------------------------------------------------------------------
#
# One JSON object per line, appended and flushed per record.  A span is one
# whole decorated function call: a ``start`` record on entry, an ``end``
# record on exit carrying ``duration_seconds`` and ``outcome``.  The last
# line of a complete trace is the ``end`` record of the ``run`` span; a
# file whose last line is anything else was cut short.

P = ParamSpec("P")
R = TypeVar("R")
AttrsFn = Callable[[tuple[Any, ...], dict[str, Any]], dict[str, Any]]

_enabled = False
_sink: TextIO | None = None
_sink_lock = threading.Lock()
_span_ids = itertools.count(1)
_sink_failed = False

# Cheap identity attrs per span, keyed by span name.  Each receives the
# decorated call's ``(args, kwargs)``; anything it raises is dropped.
_ATTRS: dict[str, AttrsFn] = {
    "evaluate": lambda a, k: {
        "candidate_id": a[0].id,
        "split": k.get("split", a[2] if len(a) > 2 else "val"),
        "evaluation_phase": k.get("evaluation_phase"),
    },
    "validate": lambda a, k: {"candidate_id": a[0].id},
    "proposal": lambda a, k: {"candidate_id": a[0][3], "generation": k.get("gen")},
    "agent": lambda a, k: {"prompt_artifact": k.get("prompt_artifact_name")},
}


def enable(path: str | os.PathLike[str] | None = None) -> bool:
    """Open *path* (or ``$HELIX_TRACE``) for appending and turn spans on.

    Returns whether tracing is now enabled.  Raises ``OSError`` when the
    file cannot be opened, so the caller can fail loudly up front rather
    than run untraced by accident.
    """
    global _enabled, _sink, _sink_failed
    target = path if path is not None else os.environ.get("HELIX_TRACE")
    if not target:
        return False
    with _sink_lock:
        if _sink is not None:
            _sink.close()
        _sink = open(target, "a", encoding="utf-8")
        _sink_failed = False
        _enabled = True
    return True


def disable() -> None:
    """Turn spans off and close the sink (tests; not needed at process exit)."""
    global _enabled, _sink
    with _sink_lock:
        _enabled = False
        if _sink is not None:
            _sink.close()
            _sink = None


def _write(record: dict[str, Any]) -> None:
    """Append one record; a failing sink is reported once and never raises."""
    global _sink_failed
    try:
        line = json.dumps(record, default=str)
        with _sink_lock:
            if _sink is not None:
                _sink.write(line + "\n")
                _sink.flush()
    except Exception as exc:
        if not _sink_failed:
            _sink_failed = True
            logger.error("Trace unavailable: %s: %s", type(exc).__name__, exc)


def traced(
    span: str, attrs: AttrsFn | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Record a ``start``/``end`` span around every call of the decorated function.

    Signature-preserving passthrough (``functools.wraps``); when tracing is
    disabled the only cost is one global check.  An exception raised by the
    wrapped function is classified (``outcome="error"``, ``error_type``)
    and re-raised untouched — ``KeyboardInterrupt`` included — and a broken
    sink can never mask it.
    """
    extract = attrs if attrs is not None else _ATTRS.get(span)

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if not _enabled:
                return fn(*args, **kwargs)
            span_id = next(_span_ids)
            extracted: dict[str, Any] = {}
            if extract is not None:
                try:
                    extracted = extract(args, kwargs)
                except Exception:
                    pass
            started = time.monotonic()
            record: dict[str, Any] = {
                "event": "start",
                "span": span,
                "span_id": span_id,
                "wall_time": time.time(),
                "monotonic": started,
                "thread_id": threading.get_ident(),
                "attrs": extracted,
            }
            _write(record)
            outcome, error_type = "ok", None
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:
                outcome, error_type = "error", type(exc).__name__
                raise
            finally:
                now = time.monotonic()
                record.update(
                    event="end",
                    wall_time=time.time(),
                    monotonic=now,
                    duration_seconds=now - started,
                    outcome=outcome,
                )
                if error_type is not None:
                    record["error_type"] = error_type
                _write(record)

        return wrapper

    return decorate
