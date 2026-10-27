"""Always-on timing spans for a HELIX run, written to ``.helix/timing.jsonl``.

Every run records where its wall clock went, the way it always records its
budget: :func:`run_evolution` opens the sink with :func:`recording`, and the
functions that do the slow work are decorated with :func:`traced`.  Outside a
recording (unit tests, library code calling a helper directly) every span is a
no-op.

One JSON object per line, appended and flushed as written, so a killed run
keeps every record it got to.  A span is a ``start`` record on entry and an
``end`` record on exit::

    {"event": "start", "span": "evaluate", "span_id": "...",
     "parent_span_id": "...", "run_id": "...", "wall_time": 1757400000.1,
     "monotonic": 1234.5, "thread_id": 6199, "attrs": {...}}
    {"event": "end", ..., "duration_seconds": 12.2, "outcome": "ok"}

``parent_span_id`` is the span that was open when this one started, so the
records form a tree rooted at the run.  It is carried in a context variable;
code that hands work to a thread pool must submit through
``contextvars.copy_context().run`` for the child spans to keep their parent.
A span whose parent was lost is still recorded, with ``parent_span_id: null``.

Spans are written whatever becomes of the candidate they worked on, the same
way its tokens are charged whatever its fate.  :func:`decision` then records
that fate once per candidate, so time can be split into what went to accepted
and to rejected candidates.

Unrelated to :mod:`helix.trace`, the in-memory event bus used for GEPA
differential testing.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, ParamSpec, TextIO, TypeVar
from uuid import uuid4

logger = logging.getLogger(__name__)

#: File name of the timing trace inside the run's ``.helix`` directory.
TIMING_FILE_NAME = "timing.jsonl"

P = ParamSpec("P")
R = TypeVar("R")
#: Receives the decorated call's arguments by parameter name (defaults
#: applied) and returns the span's ``attrs``.
AttrsFn = Callable[[dict[str, Any]], dict[str, Any]]

_lock = threading.Lock()
_sink: TextIO | None = None
_run_id: str | None = None
_write_failed = False
_parent: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "helix_timing_parent", default=None
)
_attrs_failed: set[str] = set()

#: Every ``@traced`` function with its span name and attrs extractor, so tests
#: can check each extractor against its function's current signature.
TRACED: list[tuple[str, Callable[..., Any], AttrsFn | None]] = []


@contextmanager
def recording(path: Path) -> Iterator[None]:
    """Append every span opened inside this block to *path*.

    A trace that cannot be opened is logged and the run continues untraced:
    timing is telemetry and must never be the reason a run fails.
    """
    global _sink, _run_id, _write_failed
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        sink: TextIO | None = open(path, "a", encoding="utf-8")
    except OSError as exc:
        logger.warning("Timing trace disabled: cannot open %s: %s", path, exc)
        sink = None
    with _lock:
        previous = (_sink, _run_id, _write_failed)
        _sink, _run_id, _write_failed = sink, uuid4().hex, False
    try:
        yield
    finally:
        with _lock:
            _sink, _run_id, _write_failed = previous
        if sink is not None:
            sink.close()


def _write(record: dict[str, Any]) -> None:
    """Append one record; a failing sink is reported once and never raises."""
    global _write_failed
    try:
        line = json.dumps(record, default=str)
        with _lock:
            if _sink is not None:
                _sink.write(line + "\n")
                _sink.flush()
    except Exception as exc:
        if not _write_failed:
            _write_failed = True
            logger.warning("Timing trace write failed: %s: %s", type(exc).__name__, exc)


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[None]:
    """Record one ``start``/``end`` pair around the block.

    An exception leaving the block is classified (``outcome="error"``,
    ``error_type`` = its class name, never its message) and re-raised
    untouched, ``KeyboardInterrupt`` included.
    """
    if _sink is None:
        yield
        return
    span_id = uuid4().hex
    record: dict[str, Any] = {
        "event": "start",
        "span": name,
        "span_id": span_id,
        "parent_span_id": _parent.get(),
        "run_id": _run_id,
        "wall_time": time.time(),
        "monotonic": time.monotonic(),
        "thread_id": threading.get_ident(),
        "attrs": attrs,
    }
    _write(record)
    started = record["monotonic"]
    token = _parent.set(span_id)
    outcome, error_type = "ok", None
    try:
        yield
    except BaseException as exc:
        outcome, error_type = "error", type(exc).__name__
        raise
    finally:
        try:
            _parent.reset(token)
        except ValueError:
            # Exited in a different context than it was entered in; the
            # parent pointer there was never this span's to restore.
            pass
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


def decision(candidate_id: str, *, accepted: bool, reason: str) -> None:
    """Record a candidate's fate: kept on the frontier, or discarded and why."""
    if _sink is None:
        return
    _write(
        {
            "event": "decision",
            "candidate_id": candidate_id,
            "accepted": accepted,
            "reason": reason,
            "parent_span_id": _parent.get(),
            "run_id": _run_id,
            "wall_time": time.time(),
            "monotonic": time.monotonic(),
            "thread_id": threading.get_ident(),
        }
    )


def traced(
    name: str, attrs: AttrsFn | None = None
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Run every call of the decorated function inside ``span(name, ...)``.

    *attrs* receives the call's arguments by name, so it keeps working when a
    caller switches between positional and keyword arguments.  If it raises,
    the span is still recorded, with empty ``attrs``, and the failure is
    logged once per span name.
    """

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(fn)
        TRACED.append((name, fn, attrs))

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if _sink is None:
                return fn(*args, **kwargs)
            span_attrs: dict[str, Any] = {}
            if attrs is not None:
                try:
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    span_attrs = attrs(bound.arguments)
                except Exception as exc:
                    if name not in _attrs_failed:
                        _attrs_failed.add(name)
                        logger.warning(
                            "Timing attrs for span %r failed: %s: %s",
                            name, type(exc).__name__, exc,
                        )
            with span(name, **span_attrs):
                return fn(*args, **kwargs)

        return wrapper

    return decorate
