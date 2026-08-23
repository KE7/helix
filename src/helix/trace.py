"""HELIX TraceBus — lightweight runtime event stream for differential testing.

Disabled by default, and disabled costs one attribute check: ``TRACE.emit(...)``
reads ``self.enabled`` and returns before building any event payload.

There are two ways to turn the bus on:

``record()``
    Flips the flag, yields an in-memory ``events`` list, and restores the flag
    (and event buffer) on exit.  Used by the differential-testing harness and
    the unit tests, which want the events as objects.

``write_jsonl(path)``
    Flips the flag and streams every event to *path* as JSON Lines, one object
    per line.  This is what ``helix evolve --trace`` uses.  Streaming rather
    than buffering at the end keeps a long run's memory bounded.

Traces are written to be shared: an operator attaches one to a bug report
without reading it first.  A record may therefore carry only ids, numbers,
clocks and code-controlled tags.  Exception messages, command lines,
environment and absolute filesystem paths must never reach one — hence
``error_type`` holding the exception *class* and never its text, and ``source``
being repository-relative and dropped entirely for a caller outside the tree.

A JSONL trace is framed, and the framing is what makes it safe to compute
timings from:

* the **first** line is a header record — ``{"record": "helix.trace.header",
  ...}`` — carrying the schema version, a per-run id, and the unit of every
  timestamp in the file;
* every middle line is one event;
* the **last** line is ``{"record": "helix.trace.run_complete", ...}``, written
  only after every accepted event has reached the file and no write error was
  recorded.

The footer is the point of the framing: the failure being designed against is a
truncated trace that still *looks* complete, because a killed run leaves a
prefix of perfectly well-formed event lines.  :func:`load_jsonl_trace` enforces
both ends and raises :class:`TraceIncompleteError`, so a consumer cannot
silently total that prefix.

Every record is written and ``flush()``ed inline, under one lock, by the thread
that emitted it: once ``emit`` returns, its record is on disk.  Nothing is
buffered and there is no writer thread, so a process killed at any instant
leaves behind exactly the records it had already emitted.  Do not reintroduce a
queue or a background writer — that trades this guarantee for throughput the
trace does not need, at tens of records per generation against minutes of
agent-backend and evaluator wall clock.

It is a ``flush()``, not an ``fsync()``: a machine-level power loss can still
lose the tail.  Either way the footer is absent unless the run finished
cleanly, which is what makes the loss detectable.

All timestamps in the file are **seconds** (floating point).  ``wall_time`` is
Unix epoch seconds and ``monotonic`` is a process-local monotonic reading;
durations must be computed from ``monotonic`` only.

Every ``*_END`` event carries ``outcome``, and ``"ok"`` is the single spelling
of success across all of them; any other value is a failure, narrowed by
``error_type`` where the exception class is known.  Keep it that way: a
consumer generalises the vocabulary it sees on one span to the rest, and a
second spelling of success turns that into a silently wrong total.

Spans nest: a ``PROPOSAL_START``/``PROPOSAL_END`` pair contains the
``MUTATE_*`` and ``EVAL_*`` pairs of the slot it covers, a
``PROPOSAL_BATCH_*`` pair contains every ``PROPOSAL_*`` pair of its
generation, and a ``VALIDATE_*`` pair contains the ``EVAL_*`` pairs of the
evaluator runs it drives.  Totalling spans of different levels therefore
double-counts; compare one level at a time.  ``PROPOSAL_BATCH_*`` (concurrent)
and ``VALIDATE_*`` (sequential) are siblings and never overlap, so their totals
partition a generation rather than double-counting it.

Event points are sprinkled throughout ``evolution.py``, ``eval_cache.py``,
``executor.py``, ``batch_sampler.py``, and ``mutator.py``.  The GEPA
differential harness consumes these events to assert runtime parity against
the GEPA reference engine.
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
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
    PROPOSAL_BATCH_START = "PROPOSAL_BATCH_START"
    PROPOSAL_START = "PROPOSAL_START"
    PROPOSAL_END = "PROPOSAL_END"
    PROPOSAL_BATCH_END = "PROPOSAL_BATCH_END"
    VALIDATE_START = "VALIDATE_START"
    VALIDATE_END = "VALIDATE_END"


#: Bumped whenever the on-disk record shape changes incompatibly.
TRACE_SCHEMA_VERSION = 1
#: ``record`` discriminator on the first line of a JSONL trace.
HEADER_RECORD = "helix.trace.header"
#: ``record`` discriminator on the last line of a *complete* JSONL trace.
RUN_COMPLETE_RECORD = "helix.trace.run_complete"
#: Unit of ``wall_time``, ``monotonic`` and every derived duration.
TIME_UNIT = "seconds"


class TraceWriteError(RuntimeError):
    """The trace cannot be complete, so it must not be used as evidence."""


class TraceIncompleteError(RuntimeError):
    """A trace read back from disk is not framed as a complete run.

    Raised by :func:`load_jsonl_trace`.  Such a trace must be discarded, not
    trimmed and used with care: an unknown number of events is missing, so
    every span total taken from it under-reports by an unknown amount.
    """


@dataclass
class Event:
    type: EventType
    candidate_id: str | None = None
    example_ids: list[Any] | None = None
    split: str | None = None
    hit_ids: list[Any] | None = None
    miss_ids: list[Any] | None = None
    decision: str | None = None
    # Code-controlled tag for *why* an event was emitted (e.g., the
    # ``charge_evaluation`` source: "seed_val", "merge_subsample",
    # "mutation_minibatch_gate", ...).  Never free text.  Distinct from
    # ``decision``, which carries the iteration-level accept/reject label, and
    # from ``source`` below, which holds the emitting ``"file:line"``.
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
    n_proposals: int | None = None
    # Terminal status of a ``*_END`` span: "ok" on success, anything else a
    # failure.  ``error_type`` narrows a failure to an exception class name and
    # nothing more -- never the exception's message, which routinely carries an
    # evaluator command line or its output.
    outcome: str | None = None
    error_type: str | None = None
    source: str | None = None  # repository-relative "file:line"

    # --- timing / provenance -------------------------------------------------
    # Two clocks, deliberately.  ``wall_time`` is Unix epoch seconds from
    # ``time.time()``: the only value that can be lined up against an external
    # log, a container's stdout, or a wall-clock incident timeline.  It is also
    # the only one that may go *backwards* — NTP slew and DST-adjacent system
    # clock corrections both do that — so it must never be subtracted to get a
    # duration.  ``monotonic`` is ``time.monotonic()``: meaningless as an
    # absolute instant (its epoch is arbitrary and process-local) but
    # guaranteed non-decreasing, so START/END deltas computed from it are
    # trustworthy.  Both are recorded so a consumer never has to choose between
    # "correlatable" and "correct".
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


def score_or_none(result: Any) -> float | None:
    """``result.aggregate_score()`` if it can be had, otherwise ``None``.

    The END side of every span is emitted from a ``finally`` block, where an
    exception raised while *building* the event would replace the exception the
    run is already unwinding with.  Instrumentation is not allowed to do that,
    so a score that cannot be computed is simply absent from the trace.
    """
    if result is None:
        return None
    try:
        return float(result.aggregate_score())
    except Exception:  # noqa: BLE001 - instrumentation must never raise
        return None


def _helix_version() -> str:
    """Installed HELIX version, for pinning a trace to the code that wrote it."""
    try:
        from importlib.metadata import version  # noqa: PLC0415

        return version("helix-evo")
    except Exception:  # noqa: BLE001 - provenance is optional, tracing is not
        return "unknown"


def _event_to_dict(event: Event) -> dict[str, Any]:
    """JSON-ready mapping for *event*, omitting fields left at ``None``.

    A consumer must read a missing key as ``None``: 25 of ``Event``'s 29 fields
    are optional and most are unset on any given event, so emitting them would
    be almost all of the file.
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
        # Held across write-and-flush so a record reaches the file whole even
        # when several proposal workers emit at the same instant, and across
        # the header and footer writes so neither can interleave with an event.
        self._sink_lock = threading.Lock()
        # Number of event records (not framing records) written to ``_sink``.
        # Mutated only under ``_sink_lock``; becomes the footer's event_count.
        self._written = 0
        # Set by whichever thread first fails to record an event -- a write
        # error, a malformed emit -- and read at close to decide whether the
        # run may be stamped complete.  Keep the lock: bare attribute
        # assignment would let a later, less informative error win the race.
        self._error_lock = threading.Lock()
        self._writer_error: TraceWriteError | None = None
        self._repo_root = Path(__file__).resolve().parents[2]
        # A call site is stable for the life of the process.  Cache its safe
        # relative filename so hot cache/budget events do not resolve paths.
        self._source_bases: dict[Any, str | None] = {}

    def _record_error(self, message: str) -> None:
        """Mark the in-flight trace unusable, keeping the first cause."""
        with self._error_lock:
            if self._writer_error is None:
                self._writer_error = TraceWriteError(message)

    def _take_error(self) -> TraceWriteError | None:
        with self._error_lock:
            return self._writer_error

    def _source_for_caller(self) -> str | None:
        """Return a stable repository-relative caller location, if known.

        Called only from :meth:`emit`, and the frame depth below is counted
        from there: a helper inserted between the two would silently start
        reporting the wrong line.
        """
        frame = sys._getframe(2)
        code = frame.f_code
        source_base = self._source_bases.get(code)
        if source_base is None and code not in self._source_bases:
            try:
                filename = Path(code.co_filename).resolve()
                source_base = filename.relative_to(self._repo_root).as_posix()
            except (OSError, ValueError):
                # A caller outside the repository -- a plugin, an ad-hoc
                # evaluator -- would only resolve to a machine-local absolute
                # path, which must not reach a file operators share.  Provenance
                # is optional; leaking someone's home directory is not.
                source_base = None
            self._source_bases[code] = source_base
        if source_base is None:
            return None
        return f"{source_base}:{frame.f_lineno}"

    def _write_record(self, sink: IO[str], payload: dict[str, Any]) -> bool:
        """Write one whole line and push it to the OS.  True if it landed.

        Flushing per record is what bounds the loss from an abrupt kill to the
        one record still in flight; everything ``emit`` has already returned
        from is on disk.  Callers hold ``_sink_lock``.
        """
        try:
            sink.write(json.dumps(payload, default=str) + "\n")
            sink.flush()
            return True
        except Exception as exc:  # noqa: BLE001 - never let tracing kill a run
            self._record_error(
                "Trace write failed; the trace is incomplete and cannot be used: "
                f"{type(exc).__name__}: {exc}"
            )
            return False

    def emit(self, type: EventType, **fields: Any) -> None:
        """Record one event.  Never raises while a JSONL sink is attached.

        Instrumentation must not be able to end a run it is only observing, so
        with a sink attached any failure here — an unknown field name, a full
        disk — becomes a recorded error: the run carries on, and the trace is
        denied its ``RUN_COMPLETE`` footer so nobody can time from it.  With no
        sink (the in-memory :meth:`record` path) the exception propagates
        instead: there is no file to reject, and a silently dropped event would
        surface only as a mysterious assertion failure.
        """
        if not self.enabled:
            return
        sink = self._sink
        try:
            event = Event(type=type, source=self._source_for_caller(), **fields)
            if self._collect:
                self.events.append(event)
            # Skip the file once the trace is already void: it will be refused
            # either way, and a failing disk should not be retried per event.
            if sink is not None and self._writer_error is None:
                payload = _event_to_dict(event)
                with self._sink_lock:
                    if self._write_record(sink, payload):
                        self._written += 1
        except Exception as exc:  # noqa: BLE001 - never let tracing kill a run
            if sink is None:
                raise
            # ``exc.__class__``, not ``type(exc)``: ``type`` is this method's
            # own EventType parameter.
            self._record_error(
                "Trace event could not be recorded; the trace is incomplete and "
                f"cannot be used: {exc.__class__.__name__}: {exc}"
            )

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

        Writes the header, then the events, then a ``RUN_COMPLETE`` footer —
        the last only if no write failed.  A write failure raises
        :class:`TraceWriteError` at close.  Yields the requested path.

        The footer is written on the way out, so it is absent whenever this
        exit does not run at all: a killed process, a downed machine.
        :func:`load_jsonl_trace` refuses such a file.
        """
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = target.open("w", encoding="utf-8")
        except OSError as exc:
            raise TraceWriteError(f"Cannot write trace to {target}: {exc}") from exc
        run_id = uuid.uuid4().hex
        prev_enabled = self.enabled
        prev_events = self.events
        prev_collect = self._collect
        prev_sink = self._sink
        self.events = []
        self._collect = False
        with self._sink_lock:
            self._sink = handle
            self._written = 0
            with self._error_lock:
                self._writer_error = None
            self._write_record(
                handle,
                {
                    "record": HEADER_RECORD,
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "run_id": run_id,
                    "helix_version": _helix_version(),
                    "pid": os.getpid(),
                    "time_unit": TIME_UNIT,
                    "wall_time": time.time(),
                    "monotonic": time.monotonic(),
                },
            )
        self.enabled = True
        try:
            yield target
        finally:
            # Disable first: no further emit may reach a sink that is about to
            # be stamped complete.
            self.enabled = prev_enabled
            with self._sink_lock:
                if self._take_error() is None:
                    self._write_record(
                        handle,
                        {
                            "record": RUN_COMPLETE_RECORD,
                            "schema_version": TRACE_SCHEMA_VERSION,
                            "run_id": run_id,
                            "event_count": self._written,
                            "time_unit": TIME_UNIT,
                            "wall_time": time.time(),
                            "monotonic": time.monotonic(),
                        },
                    )
                self._sink = prev_sink
            handle.close()
            self._collect = prev_collect
            self.events = prev_events
            pending_error = self._take_error()
            with self._error_lock:
                self._writer_error = None
            if pending_error is not None:
                raise pending_error


def load_jsonl_trace(path: str | Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read a framed JSONL trace, refusing anything that is not complete.

    Returns ``(header, events)`` where *events* excludes the header and the
    ``RUN_COMPLETE`` footer.  Raises :class:`TraceIncompleteError` if the file
    is empty, does not begin with a header record, does not end with a
    ``RUN_COMPLETE`` footer, carries a schema version this code does not
    understand, or contains a line that is not valid JSON.

    A consumer that wants a *usable* trace should call this rather than reading
    the lines itself: a killed run leaves a prefix of perfectly well-formed
    event lines, and totalling that prefix silently under-reports.
    """
    target = Path(path)
    try:
        raw = target.read_text(encoding="utf-8")
    except OSError as exc:
        raise TraceIncompleteError(f"Cannot read trace {target}: {exc}") from exc

    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise TraceIncompleteError(f"Trace {target} is empty.")

    records: list[dict[str, Any]] = []
    for lineno, line in enumerate(lines, 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise TraceIncompleteError(
                f"Trace {target} line {lineno} is not valid JSON — the writer was "
                f"cut off mid-record: {exc}"
            ) from exc

    header = records[0]
    if header.get("record") != HEADER_RECORD:
        raise TraceIncompleteError(
            f"Trace {target} does not start with a {HEADER_RECORD!r} record, "
            "so it cannot be read as a HELIX trace."
        )
    if header.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise TraceIncompleteError(
            f"Trace {target} has schema_version {header.get('schema_version')!r}; "
            f"this build reads {TRACE_SCHEMA_VERSION}."
        )

    footer = records[-1]
    if footer.get("record") != RUN_COMPLETE_RECORD:
        raise TraceIncompleteError(
            f"Trace {target} has no {RUN_COMPLETE_RECORD!r} footer: the run was "
            "killed or its trace writer failed, so an unknown number of events "
            "is missing. Discard it rather than totalling the prefix."
        )
    if footer.get("run_id") != header.get("run_id"):
        raise TraceIncompleteError(
            f"Trace {target} footer belongs to a different run than its header."
        )

    events = records[1:-1]
    expected = footer.get("event_count")
    if isinstance(expected, int) and expected != len(events):
        raise TraceIncompleteError(
            f"Trace {target} declares {expected} events but carries {len(events)}."
        )
    return header, events


TRACE = TraceBus()
