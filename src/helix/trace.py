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
    per line.  This is what ``helix evolve --trace`` uses.  Streaming (rather
    than buffering and dumping at the end) keeps a long run's memory bounded.

A JSONL trace is framed, and the framing is what makes it safe to compute
timings from:

* the **first** line is a header record — ``{"record": "helix.trace.header",
  ...}`` — carrying the schema version, a per-run id, and the unit of every
  timestamp in the file;
* every middle line is one event;
* the **last** line is ``{"record": "helix.trace.run_complete", ...}``, written
  only after every accepted event has reached the file and no write error was
  recorded.

The writer thread calls ``flush()`` after *every* record, so a process killed
at any instant leaves on disk exactly the records it had already emitted, whole
— the flush is what stops a kill from truncating a line mid-way.  It is a
``flush()``, not an ``fsync()``: a machine-level power loss can still lose the
tail.  Either way the footer is the authority, and the footer is absent unless
the run drained cleanly.  :func:`load_jsonl_trace` enforces both ends and
raises :class:`TraceIncompleteError` on a trace that was cut short, so a
consumer cannot silently total a truncated prefix.

All timestamps in the file are **seconds** (floating point).  ``wall_time`` is
Unix epoch seconds and ``monotonic`` is a process-local monotonic reading;
durations must be computed from ``monotonic`` only.

Spans nest: a ``PROPOSAL_START``/``PROPOSAL_END`` pair contains the
``MUTATE_*`` and ``EVAL_*`` pairs of the slot it covers, a
``PROPOSAL_BATCH_*`` pair contains every ``PROPOSAL_*`` pair of its
generation, and a ``VALIDATE_*`` pair contains the ``EVAL_*`` pairs of the
evaluator runs it drives.  Totalling spans of different levels therefore
double-counts; compare one level at a time.  ``PROPOSAL_BATCH_*`` (concurrent)
and ``VALIDATE_*`` (sequential) are siblings and never overlap, which is what
makes "how much of the generation is the sequential validation stage?" a
question this trace can answer.

Event points are sprinkled throughout ``evolution.py``, ``eval_cache.py``,
``executor.py``, ``batch_sampler.py``, and ``mutator.py``.  The GEPA
differential harness consumes these events to assert runtime parity against
the GEPA reference engine.
"""
from __future__ import annotations

import json
import os
import queue
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
    """A trace read back from disk is missing its header or its footer.

    Raised by :func:`load_jsonl_trace`.  A trace that hits this is not a
    slightly-short trace to be used with care: the run it came from was killed
    or its writer failed, so any span total taken from it is missing an unknown
    amount of time and must be discarded.
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
    n_proposals: int | None = None
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
        # The writer owns all file I/O.  Producers only enqueue Event objects,
        # so cache/budget instrumentation never contends on a filesystem lock.
        self._sink_lock = threading.Lock()
        self._queue: queue.Queue[Event | None] | None = None
        self._writer: threading.Thread | None = None
        # ``_writer_error`` is written by producer threads (queue overflow, a
        # malformed emit) and by the writer thread (an OSError on write), and
        # read by the writer thread before it decides whether the run may be
        # stamped complete.  It is guarded rather than left to attribute
        # atomicity so "first error wins" is actually first-error-wins.
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
        """Return a stable repository-relative caller location, if known."""
        frame = sys._getframe(2)
        code = frame.f_code
        source_base = self._source_bases.get(code)
        if source_base is None and code not in self._source_bases:
            try:
                filename = Path(code.co_filename).resolve()
                source_base = filename.relative_to(self._repo_root).as_posix()
            except (OSError, ValueError):
                # Never serialize a machine-local path from a plugin or an
                # ad-hoc evaluator.  The source field is optional provenance.
                source_base = None
            self._source_bases[code] = source_base
        if source_base is None:
            return None
        return f"{source_base}:{frame.f_lineno}"

    def _write_record(self, sink: IO[str], payload: dict[str, Any]) -> bool:
        """Write one whole line and push it to the OS.  True if it landed.

        The ``flush()`` is per record, deliberately.  The producers never touch
        the file — they hand Events to a queue — so the flush cost is paid on
        the writer thread and never inside a span being measured.  What it buys
        is that a SIGKILL cannot leave a half-written line behind, and cannot
        swallow a batch of records that a consumer would then never know were
        missing.  ``flush()`` reaches the OS, not the platter: this survives the
        process dying, not the machine dying.
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

    def _write_events(
        self,
        sink: IO[str],
        event_queue: queue.Queue[Event | None],
        header: dict[str, Any],
    ) -> None:
        """Write complete JSONL lines in one dedicated background thread.

        The header goes out first, then one line per event, then — and only if
        nothing has been recorded as failed — the ``RUN_COMPLETE`` footer.  A
        reader that finds no footer knows the run did not finish draining, and
        must reject the file rather than total what it can see.

        Every step inside the loop swallows its own exceptions.  This thread
        dying would leave ``write_jsonl``'s drain waiting on a queue nobody is
        consuming — a hang in the middle of a real run — so once it is running
        it always reaches the sentinel, even after it has given up writing.
        """
        written = 0
        failed = not self._write_record(sink, header)
        while True:
            event = event_queue.get()
            try:
                if event is None:
                    break
                if failed:
                    continue
                try:
                    payload = _event_to_dict(event)
                except Exception as exc:  # noqa: BLE001 - see docstring
                    self._record_error(
                        "Trace event could not be serialized; the trace is "
                        f"incomplete and cannot be used: {type(exc).__name__}: {exc}"
                    )
                    failed = True
                    continue
                if self._write_record(sink, payload):
                    written += 1
                else:
                    failed = True
            finally:
                event_queue.task_done()
        # Re-check the shared error: a producer may have overflowed the queue
        # or failed to build an event while the writer was idle, and such a run
        # is missing records even though every line the writer saw was fine.
        if failed or self._take_error() is not None:
            return
        self._write_record(
            sink,
            {
                "record": RUN_COMPLETE_RECORD,
                "schema_version": TRACE_SCHEMA_VERSION,
                "run_id": header["run_id"],
                "event_count": written,
                "time_unit": TIME_UNIT,
                "wall_time": time.time(),
                "monotonic": time.monotonic(),
            },
        )

    def emit(self, type: EventType, **fields: Any) -> None:
        """Record one event.  Never raises while a JSONL sink is attached.

        Instrumentation must not be able to end a real run, so with a sink
        attached any failure here — an unknown field name, a queue overflow —
        is turned into a recorded error: the run continues, and the trace is
        denied its ``RUN_COMPLETE`` footer so nobody can time from it.  With no
        sink (the in-memory :meth:`record` path used by the differential
        harness and the unit tests) the exception is left to propagate, because
        there is no file to reject and a silently dropped event would only show
        up as a mysterious assertion failure.
        """
        if not self.enabled:
            return
        event_queue = self._queue
        try:
            event = Event(type=type, source=self._source_for_caller(), **fields)
            if self._collect:
                self.events.append(event)
            if event_queue is not None:
                # Never delay the operation being measured.  If the writer
                # cannot keep up, invalidate the trace instead of fabricating
                # spans whose starts include tracing backpressure.
                event_queue.put_nowait(event)
        except queue.Full:
            self._record_error(
                "Trace queue overflowed; the trace is incomplete and cannot be used. "
                "Reduce trace volume or increase writer capacity."
            )
        except Exception as exc:  # noqa: BLE001 - never let tracing kill a run
            if event_queue is None:
                raise
            # ``exc.__class__``, not ``type(exc)``: ``type`` is this method's
            # own EventType parameter.
            self._record_error(
                "Trace event could not be recorded; the trace is incomplete and "
                f"cannot be used: {exc.__class__.__name__}: {exc}"
            )

    def drain(self) -> None:
        """Block until every event emitted so far has reached the file.

        A no-op unless a :meth:`write_jsonl` sink is attached.  The orderly
        exit of ``write_jsonl`` does this implicitly; this is the checkpoint
        for a caller that wants to know, mid-run, that the trace on disk is
        caught up with what has been emitted.
        """
        event_queue = self._queue
        if event_queue is not None:
            event_queue.join()

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

        A bounded background writer performs JSON encoding and file I/O.  The
        producer never blocks on disk I/O: an overflow or write failure marks
        the stream incomplete and raises :class:`TraceWriteError` at close.
        Timestamps are therefore captured at the operation boundary rather
        than before a locked write/flush delay.  Yields the requested path.

        The file is framed: a header record first, then the events, then a
        ``RUN_COMPLETE`` footer written only when the drain succeeded.  If this
        context manager never runs its exit — the process is killed, the
        machine goes down — the footer is simply absent, and
        :func:`load_jsonl_trace` refuses the file.  That is the whole point:
        the failure mode being designed against is a truncated trace that still
        *looks* complete.
        """
        target = Path(path)
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            handle = target.open("w", encoding="utf-8")
        except OSError as exc:
            raise TraceWriteError(f"Cannot write trace to {target}: {exc}") from exc
        run_id = uuid.uuid4().hex
        header = {
            "record": HEADER_RECORD,
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": run_id,
            "helix_version": _helix_version(),
            "pid": os.getpid(),
            "time_unit": TIME_UNIT,
            "wall_time": time.time(),
            "monotonic": time.monotonic(),
        }
        prev_enabled = self.enabled
        prev_events = self.events
        prev_collect = self._collect
        prev_sink = self._sink
        # 65,536 records bound memory while absorbing short bursts from a
        # proposal pool.  Overflow fails the run's trace rather than delaying
        # the operation being measured or silently dropping span endpoints.
        event_queue: queue.Queue[Event | None] = queue.Queue(maxsize=65_536)
        self.events = []
        self._collect = False
        with self._sink_lock:
            self._sink = handle
            self._queue = event_queue
            with self._error_lock:
                self._writer_error = None
            self._writer = threading.Thread(
                target=self._write_events,
                args=(handle, event_queue, header),
                name="helix-trace-writer",
                daemon=True,
            )
            self._writer.start()
        self.enabled = True
        try:
            yield target
        finally:
            self.enabled = prev_enabled
            # Finish accepted records before closing the file.  ``None`` is
            # enqueued after the drain so the single writer emits whole lines.
            event_queue.join()
            event_queue.put(None)
            event_queue.join()
            writer = self._writer
            if writer is not None:
                writer.join()
            with self._sink_lock:
                self._sink = prev_sink
                self._queue = None
                self._writer = None
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
    understand, or contains a line that is not valid JSON — every way a run
    that was killed mid-write shows up on disk.

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
            f"Trace {target} does not start with a {HEADER_RECORD!r} record; it "
            "was not written by this version of HELIX and cannot be trusted."
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
            "is missing. Timings taken from it would be wrong."
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
