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
     "parent_span_id": "...", "run_id": "...", "generation": 3,
     "candidate": "g3-s9", "wall_time": 1757400000.1, "thread_id": 6199,
     "attrs": {...}}
    {"event": "end", ..., "duration_seconds": 12.2, "outcome": "ok"}

``parent_span_id`` is the span that was open when this one started, so the
records form a tree rooted at the run.  ``generation`` and ``candidate`` say
what the work was for: the outermost enclosing span that names one decides it
(a proposal's evaluation of its *parent* is still work for the child), so every
question about a generation or a candidate is a flat group-by.  All three are
carried in a context variable; code that hands work to a thread pool must
submit through ``contextvars.copy_context().run`` for them to follow.  A span
that lost its context is still recorded, with those fields ``null``.

Spans are written whatever becomes of the candidate they worked on, the same
way its tokens are charged whatever its fate.  :func:`decision` then records
that fate once per candidate, so time can be split into what went to accepted
and to rejected candidates.

The ``helix.timed`` section is the other side: the API that code HELIX runs
(the evaluator, scripts the agent runs) uses to report its own operations.

Unrelated to :mod:`helix.trace`, the in-memory event bus used for GEPA
differential testing.
"""

from __future__ import annotations

import contextvars
import functools
import inspect
import json
import logging
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, ParamSpec, TextIO, TypeVar
from uuid import uuid4

from helix.asi import append_line

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
#: What the open span hands to the spans inside it: its id (their parent),
#: and the generation and candidate the work is for.
_context: contextvars.ContextVar[tuple[str | None, int | None, str | None]] = (
    contextvars.ContextVar("helix_timing_context", default=(None, None, None))
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
    parent, generation, candidate = _context.get()
    if generation is None:
        generation = attrs.get("generation")
    if candidate is None:
        candidate = attrs.get("candidate_id")
    record: dict[str, Any] = {
        "event": "start",
        "span": name,
        "span_id": span_id,
        "parent_span_id": parent,
        "run_id": _run_id,
        "generation": generation,
        "candidate": candidate,
        "wall_time": time.time(),
        "thread_id": threading.get_ident(),
        "attrs": attrs,
    }
    _write(record)
    # Durations come from the monotonic clock, which never jumps; its raw
    # value means nothing outside this process, so it is not recorded.
    started = time.monotonic()
    token = _context.set((span_id, generation, candidate))
    outcome, error_type = "ok", None
    try:
        yield
    except BaseException as exc:
        outcome, error_type = "error", type(exc).__name__
        raise
    finally:
        _context.reset(token)
        record.update(
            event="end",
            wall_time=time.time(),
            duration_seconds=time.monotonic() - started,
            outcome=outcome,
        )
        if error_type is not None:
            record["error_type"] = error_type
        _write(record)


def decision(candidate_id: str, *, accepted: bool, reason: str) -> None:
    """Record a candidate's fate: kept on the frontier, or discarded and why."""
    if _sink is None:
        return
    parent, generation, _ = _context.get()
    _write(
        {
            "event": "decision",
            "candidate": candidate_id,
            "accepted": accepted,
            "reason": reason,
            "parent_span_id": parent,
            "run_id": _run_id,
            "generation": generation,
            "wall_time": time.time(),
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


def detail(within: str, source: str, total: dict[str, Any]) -> None:
    """Record time spent inside the open *within* span, totalled per name.

    *source* is ``"transcript"`` (read from the agent's own transcript:
    ``model``, ``tool:<name>``, ``other``) or ``"timed"`` (``helix.timed``
    calls).  *total* holds ``name``, ``example``, ``count``, ``seconds``
    and ``max_seconds``.
    """
    if _sink is None:
        return
    parent, generation, candidate = _context.get()
    _write(
        {
            "event": "detail",
            "within": within,
            "source": source,
            **total,
            "parent_span_id": parent,
            "run_id": _run_id,
            "generation": generation,
            "candidate": candidate,
            "wall_time": time.time(),
        }
    )


def evaluation_attrs(ids: str | None) -> AttrsFn:
    """Attrs for an evaluation span, reading the example ids from parameter
    *ids* (``None``: the function always evaluates the whole split).

    ``examples`` and ``evaluation_phase`` are left out when there are none.
    """

    def attrs(a: dict[str, Any]) -> dict[str, Any]:
        examples = None if ids is None else a[ids]
        extra = {
            "examples": None if examples is None else len(examples),
            "evaluation_phase": a.get("evaluation_phase"),
        }
        return {
            "candidate_id": a["candidate"].id,
            "split": a["split"],
            **{key: value for key, value in extra.items() if value is not None},
        }

    return attrs


# ---------------------------------------------------------------------------
# ``helix.timed``: code HELIX runs times its own operations
# ---------------------------------------------------------------------------
#
# The evaluator, its ``extra_commands`` and scripts the mutating agent runs can
# time their own operations::
#
#     with timed("sam3", example=example_id):
#         masks = sam3(frame)
#
#     @timed("molmo")                 # sync or async functions
#     def point(...): ...
#
#     timed.record("owl-vit", 0.41)   # a duration measured elsewhere
#
# Each call appends one JSON line, ``{"name", "seconds", "example"}``, to the
# per-invocation file named by ``HELIX_TIMING_LOG``; any language can write
# the same line.  HELIX totals the file per ``(name, example)`` after the call
# returns and records the totals as ``detail`` records.  Outside a HELIX
# invocation this is a no-op, so the same code runs unchanged locally.

HELIX_TIMING_LOG_ENV = "HELIX_TIMING_LOG"

#: A longer line is dropped, so one ``write`` stays atomic (POSIX ``PIPE_BUF``)
#: across the threads and processes appending to the same file.
_MAX_LINE_BYTES = 4096
#: HELIX reads at most this much of one invocation's file.
MAX_LOG_BYTES = 64 * 1024 * 1024

F = TypeVar("F", bound=Callable[..., Any])


def _record(
    name: str, seconds: float, *, example: object = None
) -> None:
    """Append one timing line; best-effort, never raises."""
    if not os.environ.get(HELIX_TIMING_LOG_ENV):
        return
    entry: dict[str, Any] = {"name": str(name), "seconds": float(seconds)}
    if example is not None:
        entry["example"] = str(example)
    line = json.dumps(entry) + "\n"
    if len(line.encode()) <= _MAX_LINE_BYTES:
        append_line(HELIX_TIMING_LOG_ENV, line)


class timed:
    """Time a block (``with``) or every call of a function (decorator)."""

    record = staticmethod(_record)

    def __init__(self, name: str, *, example: object = None) -> None:
        self.name = name
        self.example = example
        self._started = 0.0

    def __enter__(self) -> timed:
        self._started = time.perf_counter()
        return self

    def __exit__(self, *_: object) -> None:
        _record(self.name, time.perf_counter() - self._started, example=self.example)

    def __call__(self, fn: F) -> F:
        # A fresh instance per call, so concurrent calls never share a start.
        name, example = self.name, self.example
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with timed(name, example=example):
                    return await fn(*args, **kwargs)

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timed(name, example=example):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


def timed_log_name() -> str:
    """A fresh per-invocation log file name (git-ignored in worktrees)."""
    return f".helix_timing_{uuid4().hex}.jsonl"


def timed_totals(raw: str) -> list[dict[str, Any]]:
    """Total a timing log per ``(name, example)``; malformed lines are skipped."""
    by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    for line in raw.splitlines():
        try:
            entry = json.loads(line)
            name, seconds = str(entry["name"]), float(entry["seconds"])
        except (ValueError, TypeError, KeyError):
            continue
        example = entry.get("example")
        key = (name, None if example is None else str(example))
        total = by_key.setdefault(
            key,
            {"name": key[0], "example": key[1], "count": 0, "seconds": 0.0,
             "max_seconds": 0.0},
        )
        total["count"] += 1
        total["seconds"] += seconds
        total["max_seconds"] = max(total["max_seconds"], seconds)
    return list(by_key.values())


def read_timed_log(path: Path) -> str:
    """The first :data:`MAX_LOG_BYTES` of *path*, or ``""`` if it is missing."""
    try:
        with path.open("rb") as fh:
            return fh.read(MAX_LOG_BYTES).decode("utf-8", "replace")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Reading a trace (``helix timing``)
# ---------------------------------------------------------------------------

#: Spans that never contain one another, so their durations add up to the
#: work done without counting anything twice.
WORK_KINDS = {
    "agent": "agent",
    "evaluate": "evaluator",
    "clone": "git",
    "snapshot": "git",
    "remove": "git",
    "save_state": "state save",
    "sandbox_reset": "sandbox reset",
}


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every record in *path*, skipping a line torn by a killed run."""
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _fates(records: list[dict[str, Any]]) -> dict[str, str]:
    return {
        r["candidate"]: "accepted" if r["accepted"] else "rejected"
        for r in records
        if r["event"] == "decision"
    }


@dataclass
class Summary:
    """Where a trace's time went, as ``helix timing`` reports it."""

    #: kind (a :data:`WORK_KINDS` value) -> fate of the candidate it was for
    #: (``accepted``, ``rejected``, ``undecided``, or ``unattributed`` for
    #: work done for no candidate) -> seconds.
    work: dict[str, dict[str, float]] = field(default_factory=dict)
    #: fate -> number of candidates.
    candidates: dict[str, int] = field(default_factory=dict)
    #: Seconds of every run (``evolve`` plus each ``resume``), leaving out time
    #: the machine slept like every other duration; a run still going counts
    #: the wall-clock time up to its latest record.
    run_seconds: float = 0.0
    #: generation -> wall seconds.  A generation redone after a resume adds up.
    generation_seconds: dict[int, float] = field(default_factory=dict)
    #: ``(within, source, name)`` of :func:`detail` records -> ``count``,
    #: ``max_seconds``, and ``seconds`` by fate.
    details: dict[tuple[str, str, str], dict[str, Any]] = field(default_factory=dict)


def summarize(records: list[dict[str, Any]]) -> Summary:
    """Total each kind of work by the fate of the candidate it was for."""
    fates = _fates(records)
    work: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    generations: dict[int, float] = defaultdict(float)
    details: dict[tuple[str, str, str], dict[str, Any]] = {}
    seen: set[str] = set()
    run_started: dict[str, float] = {}
    run_latest: dict[str, float] = defaultdict(float)
    run_finished: dict[str, float] = {}

    def fate_of(candidate: str | None) -> str:
        return "unattributed" if candidate is None else fates.get(candidate, "undecided")

    for r in records:
        candidate = r.get("candidate")
        run_latest[r["run_id"]] = max(run_latest[r["run_id"]], r["wall_time"])
        if r["event"] == "start" and r["span"] == "run":
            run_started[r["run_id"]] = r["wall_time"]
        if r["event"] == "detail":
            total = details.setdefault(
                (r["within"], r["source"], r["name"]),
                {"count": 0, "max_seconds": 0.0, "seconds": defaultdict(float)},
            )
            total["count"] += r["count"]
            total["max_seconds"] = max(total["max_seconds"], r["max_seconds"])
            total["seconds"][fate_of(candidate)] += r["seconds"]
        if r["event"] != "end":
            continue
        if candidate is not None:
            seen.add(candidate)
        if r["span"] == "run":
            run_finished[r["run_id"]] = r["duration_seconds"]
        elif r["span"] == "generation":
            generations[r["attrs"]["generation"]] += r["duration_seconds"]
        kind = WORK_KINDS.get(r["span"])
        if kind is not None:
            work[kind][fate_of(candidate)] += r["duration_seconds"]
    candidates: dict[str, int] = defaultdict(int)
    for candidate in seen | set(fates):
        candidates[fates.get(candidate, "undecided")] += 1
    return Summary(
        work={kind: dict(by_fate) for kind, by_fate in work.items()},
        candidates=dict(candidates),
        run_seconds=sum(
            run_finished.get(run, run_latest[run] - start)
            for run, start in run_started.items()
        ),
        generation_seconds=dict(generations),
        details={key: {**total, "seconds": dict(total["seconds"])} for key, total in details.items()},
    )


def to_perfetto(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Chrome trace JSON for ui.perfetto.dev: one process per run, one track
    per thread, fates as instant events.  Wall clock, so an ``evolve`` and its
    ``resume`` line up in real time.

    Each slice runs from its ``start`` record's wall time to its ``end``
    record's.  ``duration_seconds`` leaves out time the machine slept, so a
    slice drawn from it would start late and its children would spill out.
    """
    fates = _fates(records)
    started = {r["span_id"]: r["wall_time"] for r in records if r["event"] == "start"}
    # Detail totals are shown on the slice they were recorded inside.
    details: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for r in records:
        if r["event"] == "detail":
            label = r["name"] if r["example"] is None else f"{r['name']}[{r['example']}]"
            details[r["parent_span_id"]][label] += r["seconds"]
    pids: dict[str, int] = {}
    events: list[dict[str, Any]] = []
    for r in records:
        pid = pids.setdefault(r["run_id"], len(pids) + 1)
        candidate = r.get("candidate")
        if r["event"] == "end":
            start = started.get(r["span_id"], r["wall_time"] - r["duration_seconds"])
            events.append({
                "name": r["span"],
                "ph": "X",
                "ts": start * 1e6,
                "dur": (r["wall_time"] - start) * 1e6,
                "pid": pid,
                "tid": r["thread_id"],
                "args": {
                    **r["attrs"],
                    "generation": r.get("generation"),
                    "candidate": candidate,
                    "fate": None if candidate is None else fates.get(candidate, "undecided"),
                    "outcome": r["outcome"],
                    **({"details": dict(details[r["span_id"]])} if r["span_id"] in details else {}),
                },
            })
        elif r["event"] == "decision":
            events.append({
                "name": "accepted" if r["accepted"] else "rejected",
                "ph": "i",
                "s": "t",
                "ts": r["wall_time"] * 1e6,
                "pid": pid,
                "tid": r["thread_id"],
                "args": {"candidate": candidate, "reason": r["reason"]},
            })
    return {"traceEvents": events}
