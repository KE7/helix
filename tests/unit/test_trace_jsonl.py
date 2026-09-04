"""Unit tests for ``helix evolve --trace`` and the TraceBus timing fields.

Covers the operator-facing question the flag exists to answer: for a finished
run, how much wall clock went into the agent backend versus the evaluator
subprocess?  That needs (a) a way to turn the bus on outside a test, (b) a
clock on every event, and (c) enough provenance to re-pair a START with its
own END when several proposal workers are live at once.

Nothing here talks to a real backend or a real evaluator subprocess: the agent
call and the evaluator are both replaced with local fakes that sleep.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from helix.cli import cli
from helix.config import EvaluatorConfig, HelixConfig
from helix.population import Candidate, EvalResult
from helix.trace import (
    HEADER_RECORD,
    RUN_COMPLETE_RECORD,
    TIME_UNIT,
    TRACE,
    TRACE_SCHEMA_VERSION,
    Event,
    EventType,
    TraceIncompleteError,
    load_jsonl_trace,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_project(tmp_path: Path) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "helix.toml").write_text(
        'objective = "test"\n\n[evaluator]\ncommand = "true"\n'
    )
    return tmp_path


def _make_candidate(cid: str = "g0-s0", worktree_path: str = "/tmp/fake-wt") -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=worktree_path,
        branch_name=f"helix/{cid}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def _make_eval_result(candidate_id: str = "g0-s0") -> EvalResult:
    return EvalResult(
        candidate_id=candidate_id,
        scores={"pass_rate": 0.5},
        asi={"stdout": "", "stderr": ""},
        instance_scores={"test_a": 1.0},
    )


def _make_config() -> HelixConfig:
    return HelixConfig(
        objective="Pass all tests",
        evaluator=EvaluatorConfig(command="true"),
    )


def _current_line() -> int:
    """Line number of the caller — used to pin down emit()'s ``source``."""
    import sys

    return sys._getframe(1).f_lineno


def _read_jsonl(path: Path) -> list[dict]:
    """Return the *event* records of a complete trace at *path*.

    Goes through the shipped reader on purpose: every test that reads a trace
    then also asserts, implicitly, that the file passed the header/footer
    completeness check a real consumer applies.
    """
    try:
        _header, events = load_jsonl_trace(path)
    except TraceIncompleteError as exc:  # pragma: no cover - failure path
        pytest.fail(f"{path} did not read back as a complete trace: {exc}")
    return events


def _read_raw_lines(path: Path) -> list[dict]:
    """Parse every line of *path*, framing included, failing on malformed JSON."""
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{path}:{lineno} is not valid JSON ({exc}): {line!r}")
    return records


def _retag(path: Path, index: int, **changes: object) -> None:
    """Rewrite one framing record of *path* in place, keeping its other keys."""
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[index] = json.dumps({**json.loads(lines[index]), **changes})
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _drop_last_line(path: Path) -> None:
    lines = path.read_text(encoding="utf-8").splitlines()[:-1]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pair_durations(
    records: list[dict], start: str, end: str
) -> dict[tuple[int, str], float]:
    """Match START/END events per (thread_id, candidate_id) and return deltas."""
    open_starts: dict[tuple[int, str], float] = {}
    durations: dict[tuple[int, str], float] = {}
    for rec in records:
        key = (rec["thread_id"], rec.get("candidate_id", ""))
        if rec["type"] == start:
            open_starts[key] = rec["monotonic"]
        elif rec["type"] == end and key in open_starts:
            durations[key] = rec["monotonic"] - open_starts.pop(key)
    return durations


# ---------------------------------------------------------------------------
# The flag itself
# ---------------------------------------------------------------------------


class TestTraceFlag:
    def test_evolve_trace_writes_jsonl_to_the_given_path(self, tmp_path, mocker):
        project = _make_project(tmp_path / "proj")
        trace_file = tmp_path / "out" / "run.jsonl"

        def fake_run_evolution(config, project_root, base_dir):
            TRACE.emit(EventType.OPT_START)
            TRACE.emit(EventType.ITER_START, decision="0")
            TRACE.emit(EventType.OPT_END, candidate_id="g0-s0")

        mocker.patch("helix.evolution.run_evolution", fake_run_evolution)

        result = CliRunner().invoke(
            cli, ["evolve", "--dir", str(project), "--trace", str(trace_file)]
        )

        assert result.exit_code == 0, result.output
        # The parent directory did not exist beforehand; the flag creates it.
        assert trace_file.exists()
        records = _read_jsonl(trace_file)
        assert [r["type"] for r in records] == [
            "OPT_START",
            "ITER_START",
            "OPT_END",
        ]
        assert records[1]["decision"] == "0"
        assert records[2]["candidate_id"] == "g0-s0"

    def test_trace_events_carry_both_clocks_and_a_thread_id(self, tmp_path, mocker):
        project = _make_project(tmp_path / "proj")
        trace_file = tmp_path / "run.jsonl"
        before = time.time()

        def fake_run_evolution(config, project_root, base_dir):
            TRACE.emit(EventType.OPT_START)
            TRACE.emit(EventType.OPT_END)

        mocker.patch("helix.evolution.run_evolution", fake_run_evolution)

        result = CliRunner().invoke(
            cli, ["evolve", "--dir", str(project), "--trace", str(trace_file)]
        )
        assert result.exit_code == 0, result.output

        after = time.time()
        records = _read_jsonl(trace_file)
        assert records
        for rec in records:
            assert before <= rec["wall_time"] <= after
            assert isinstance(rec["monotonic"], float)
            assert rec["thread_id"] == threading.get_ident()
            assert rec["source"].startswith("tests/unit/")
            assert not Path(rec["source"].rsplit(":", 1)[0]).is_absolute()
        # ... and the bus is handed back the way it was found.
        assert TRACE.enabled is False
        assert TRACE.events == []

    def test_bus_stays_disabled_without_the_flag(self, tmp_path, mocker):
        project = _make_project(tmp_path / "proj")
        seen: list[bool] = []

        def fake_run_evolution(config, project_root, base_dir):
            seen.append(TRACE.enabled)
            TRACE.emit(EventType.OPT_START)

        mocker.patch("helix.evolution.run_evolution", fake_run_evolution)

        result = CliRunner().invoke(cli, ["evolve", "--dir", str(project)])

        assert result.exit_code == 0, result.output
        assert seen == [False]
        assert TRACE.enabled is False
        assert TRACE.events == []

    def test_streaming_does_not_buffer_events_in_memory(self, tmp_path):
        """A long run must not accumulate every event it ever emitted."""
        trace_file = tmp_path / "run.jsonl"
        with TRACE.write_jsonl(trace_file):
            for i in range(50):
                TRACE.emit(EventType.CACHE_GET, candidate_id=f"c{i}")
            assert TRACE.events == []
        assert len(_read_jsonl(trace_file)) == 50

    def test_run_that_raises_still_drains_and_stamps_the_trace_complete(
        self, tmp_path
    ):
        """A run that raises still exits cleanly, so its trace is complete.

        Raising inside the context manager runs its ``finally``: the footer is
        written and the trace is legitimately complete.  The opposite case,
        where no cleanup runs at all, is ``TestAbruptExit`` below.
        """
        trace_file = tmp_path / "run.jsonl"
        with pytest.raises(RuntimeError):
            with TRACE.write_jsonl(trace_file):
                TRACE.emit(EventType.OPT_START)
                TRACE.emit(EventType.ITER_START, decision="0")
                raise RuntimeError("run failed mid-generation")
        assert [r["type"] for r in _read_jsonl(trace_file)] == [
            "OPT_START",
            "ITER_START",
        ]
        assert _read_raw_lines(trace_file)[-1]["record"] == RUN_COMPLETE_RECORD

    def test_unwritable_trace_destination_is_a_clear_cli_error(self, tmp_path):
        project = _make_project(tmp_path / "proj")
        not_a_directory = tmp_path / "not-a-directory"
        not_a_directory.write_text("blocker")

        result = CliRunner().invoke(
            cli,
            [
                "evolve",
                "--dir",
                str(project),
                "--trace",
                str(not_a_directory / "run.jsonl"),
            ],
        )

        assert result.exit_code == 2
        assert "Cannot write trace" in result.output
        assert "FileExistsError" not in result.output


# ---------------------------------------------------------------------------
# The timing split: agent mutation vs evaluator subprocess
# ---------------------------------------------------------------------------


_SLEEP = 0.02


class TestTimingSplit:
    def test_eval_start_end_pair_yields_a_positive_duration(self, mocker):
        from helix import executor

        def fake_evaluator(candidate, split, instance_ids):
            time.sleep(_SLEEP)
            return _make_eval_result(candidate.id)

        mocker.patch.object(executor, "_EVALUATOR_OVERRIDE", fake_evaluator)

        with TRACE.record() as events:
            executor.run_evaluator(_make_candidate(), _make_config(), split="val")

        types = [e.type for e in events]
        assert EventType.EVAL_START in types
        assert EventType.EVAL_END in types
        start = next(e for e in events if e.type is EventType.EVAL_START)
        end = next(e for e in events if e.type is EventType.EVAL_END)
        assert end.monotonic - start.monotonic >= _SLEEP
        # ``wall_time`` is asserted present but never asserted ordered: NTP
        # slew can move it backwards, which is why durations come from
        # ``monotonic``.  Do not add an ordering assertion here.
        assert isinstance(end.wall_time, float)

    def test_failed_evaluator_still_closes_its_interval(self, mocker):
        from helix import executor

        def raising_evaluator(candidate, split, instance_ids):
            raise TimeoutError("simulated evaluator timeout")

        mocker.patch.object(executor, "_EVALUATOR_OVERRIDE", raising_evaluator)

        with TRACE.record() as events, pytest.raises(TimeoutError):
            executor.run_evaluator(_make_candidate(), _make_config(), split="val")

        assert [event.type for event in events] == [
            EventType.EVAL_START,
            EventType.EVAL_END,
        ]
        end = events[-1]
        assert end.outcome == "error"
        assert end.error_type == "TimeoutError"
        assert end.score is None

    def test_mutate_start_end_pair_yields_a_positive_duration(self, tmp_path, mocker):
        from helix.mutator import mutate

        child_path = tmp_path / "g1-s0"
        child_path.mkdir()
        child = _make_candidate("g1-s0", str(child_path))

        def fake_backend(*args, **kwargs):
            time.sleep(_SLEEP)
            return ({"result": "ok"}, {})

        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch("helix.mutator.invoke_claude_code", side_effect=fake_backend)
        mocker.patch("helix.mutator.snapshot_candidate", return_value="sha")
        mocker.patch("helix.mutator.remove_worktree")

        with TRACE.record() as events:
            assert (
                mutate(
                    _make_candidate("g0-s0"),
                    _make_eval_result(),
                    "g1-s0",
                    _make_config(),
                    tmp_path,
                )
                is child
            )

        start = next(e for e in events if e.type is EventType.MUTATE_START)
        end = next(e for e in events if e.type is EventType.MUTATE_END)
        assert start.candidate_id == "g1-s0"
        assert end.candidate_id == "g1-s0"
        assert end.outcome == "ok"
        assert end.monotonic - start.monotonic >= _SLEEP

    def test_failed_mutation_still_closes_its_interval(self, tmp_path, mocker):
        from helix.exceptions import MutationError
        from helix.mutator import mutate

        child_path = tmp_path / "g1-s0"
        child_path.mkdir()
        child = _make_candidate("g1-s0", str(child_path))

        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch(
            "helix.mutator.invoke_claude_code", side_effect=MutationError("boom")
        )
        mocker.patch("helix.mutator.snapshot_candidate")
        mocker.patch("helix.mutator.remove_worktree")

        with TRACE.record() as events:
            assert (
                mutate(
                    _make_candidate("g0-s0"),
                    _make_eval_result(),
                    "g1-s0",
                    _make_config(),
                    tmp_path,
                )
                is None
            )

        ends = [e for e in events if e.type is EventType.MUTATE_END]
        assert len(ends) == 1
        assert ends[0].outcome == "mutation_error"


# ---------------------------------------------------------------------------
# Concurrency: emits arrive from proposal-worker threads
# ---------------------------------------------------------------------------


class TestConcurrentEmits:
    def test_concurrent_emits_produce_wellformed_jsonl(self, tmp_path):
        """Real threads, no mocks: no interleaved or truncated lines."""
        trace_file = tmp_path / "run.jsonl"
        n_threads, per_thread = 8, 250
        barrier = threading.Barrier(n_threads)

        def worker(i: int) -> None:
            barrier.wait()  # maximise the overlap
            for j in range(per_thread):
                TRACE.emit(
                    EventType.EVAL_START,
                    candidate_id=f"g{i}-s{j}",
                    split="val",
                    # A long payload widens the window for a torn write.
                    example_ids=[f"example-{i}-{j}-{k}" for k in range(20)],
                )

        with TRACE.write_jsonl(trace_file):
            threads = [
                threading.Thread(target=worker, args=(i,)) for i in range(n_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        records = _read_jsonl(trace_file)
        assert len(records) == n_threads * per_thread
        assert {r["candidate_id"] for r in records} == {
            f"g{i}-s{j}" for i in range(n_threads) for j in range(per_thread)
        }
        assert len({r["thread_id"] for r in records}) == n_threads
        for rec in records:
            assert len(rec["example_ids"]) == 20

    def test_thread_id_repairs_pairs_that_the_file_order_scrambles(self, tmp_path):
        """Under P>1 the file is in completion order; thread_id restores pairing."""
        trace_file = tmp_path / "run.jsonl"
        n_threads = 6
        barrier = threading.Barrier(n_threads)

        def worker(i: int) -> None:
            barrier.wait()
            TRACE.emit(EventType.MUTATE_START, candidate_id=f"g1-s{i}")
            # Staggered sleeps guarantee the ENDs do not come back in the same
            # order the STARTs went out.
            time.sleep(0.01 * (n_threads - i))
            TRACE.emit(EventType.MUTATE_END, candidate_id=f"g1-s{i}", outcome="ok")

        with TRACE.write_jsonl(trace_file):
            threads = [
                threading.Thread(target=worker, args=(i,)) for i in range(n_threads)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        records = _read_jsonl(trace_file)
        # Sanity: the raw file really is interleaved, not neatly START/END.
        assert [r["type"] for r in records] != [
            t for i in range(n_threads) for t in ("MUTATE_START", "MUTATE_END")
        ]

        durations = _pair_durations(records, "MUTATE_START", "MUTATE_END")
        assert len(durations) == n_threads
        for (_thread_id, candidate_id), delta in durations.items():
            expected = 0.01 * (n_threads - int(candidate_id.split("s")[1]))
            assert delta >= expected * 0.5
            # Each pair belongs to exactly one worker: a mispaired interval
            # would inherit a sibling's much longer sleep.
            assert delta < expected + 0.5


# ---------------------------------------------------------------------------
# emit() internals
# ---------------------------------------------------------------------------


class TestEmitSource:
    def test_source_is_caller_file_and_line_in_the_original_format(self):
        with TRACE.record() as events:
            TRACE.emit(EventType.CACHE_GET)
            expected_line = _current_line() - 1

        source = events[0].source
        assert source is not None
        filename, _, lineno = source.rpartition(":")
        assert filename == "tests/unit/test_trace_jsonl.py"
        assert not Path(filename).is_absolute()
        assert int(lineno) == expected_line

    def test_source_outside_the_repository_is_omitted(self, tmp_path):
        external_module = tmp_path / "external_emitter.py"
        external_module.write_text(
            "from helix.trace import TRACE, EventType\n"
            "def emit():\n"
            "    TRACE.emit(EventType.CACHE_GET)\n"
        )
        namespace: dict[str, object] = {}
        exec(compile(external_module.read_text(), str(external_module), "exec"), namespace)

        with TRACE.record() as events:
            namespace["emit"]()

        assert events[0].source is None

    def test_record_still_yields_the_in_memory_event_list(self):
        with TRACE.record() as events:
            TRACE.emit(EventType.CACHE_PUT, candidate_id="c1")
            assert len(events) == 1
            assert isinstance(events[0], Event)
        assert TRACE.enabled is False


# ---------------------------------------------------------------------------
# Framing: a trace is only evidence if it says so at both ends
# ---------------------------------------------------------------------------


class TestFraming:
    def test_a_complete_trace_is_bracketed_by_a_header_and_a_footer(self, tmp_path):
        trace_file = tmp_path / "run.jsonl"
        with TRACE.write_jsonl(trace_file):
            TRACE.emit(EventType.OPT_START)
            TRACE.emit(EventType.OPT_END)

        raw = _read_raw_lines(trace_file)
        header, footer = raw[0], raw[-1]
        assert header["record"] == HEADER_RECORD
        assert header["schema_version"] == TRACE_SCHEMA_VERSION
        assert header["time_unit"] == TIME_UNIT
        assert header["pid"] == os.getpid()
        assert isinstance(header["run_id"], str) and header["run_id"]
        assert isinstance(header["helix_version"], str)

        assert footer["record"] == RUN_COMPLETE_RECORD
        assert footer["run_id"] == header["run_id"]
        assert footer["event_count"] == 2
        assert len(raw) == 2 + footer["event_count"]

        loaded_header, events = load_jsonl_trace(trace_file)
        assert loaded_header == header
        assert [e["type"] for e in events] == ["OPT_START", "OPT_END"]

    def test_two_runs_get_distinct_run_ids(self, tmp_path):
        ids = []
        for name in ("a.jsonl", "b.jsonl"):
            path = tmp_path / name
            with TRACE.write_jsonl(path):
                TRACE.emit(EventType.OPT_START)
            ids.append(_read_raw_lines(path)[0]["run_id"])
        assert ids[0] != ids[1]

    #: Ways a trace that must not be totalled shows up on disk, each paired
    #: with the part of the refusal that tells an operator which one it was.
    #: Every case is applied to a trace that was complete a moment earlier, so
    #: a case that stops corrupting anything fails instead of silently passing.
    @pytest.mark.parametrize(
        ("corrupt", "expected"),
        [
            (_drop_last_line, RUN_COMPLETE_RECORD),
            (lambda p: p.write_text(p.read_text(encoding="utf-8")[:-20], encoding="utf-8"), "not valid JSON"),
            (lambda p: p.write_text('{"type": "OPT_START"}\n', encoding="utf-8"), HEADER_RECORD),
            (lambda p: p.write_text("", encoding="utf-8"), "is empty"),
            (lambda p: _retag(p, 0, schema_version=TRACE_SCHEMA_VERSION + 1), "schema_version"),
            (lambda p: _retag(p, -1, run_id="0" * 32), "different run"),
            (lambda p: _retag(p, -1, event_count=99), "declares 99 events"),
        ],
        ids=[
            "footer-dropped-by-a-killed-writer",
            "final-line-torn-mid-record",
            "no-header-record-at-all",
            "empty-file",
            "schema-version-this-build-cannot-read",
            "footer-belonging-to-a-different-run",
            "footer-event-count-disagrees-with-the-body",
        ],
    )
    def test_an_incomplete_or_tampered_trace_is_rejected(
        self, tmp_path, corrupt, expected
    ):
        trace_file = tmp_path / "run.jsonl"
        with TRACE.write_jsonl(trace_file):
            TRACE.emit(EventType.OPT_START)
        corrupt(trace_file)

        with pytest.raises(TraceIncompleteError) as excinfo:
            load_jsonl_trace(trace_file)
        assert expected in str(excinfo.value)

    def test_a_dropped_event_withholds_the_footer(self, tmp_path):
        """A record that never landed means the run may not be stamped complete."""
        trace_file = tmp_path / "run.jsonl"
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - asserted below
            with TRACE.write_jsonl(trace_file):
                TRACE.emit(EventType.OPT_START)
                # Simulate a write that failed after the sink was attached.
                TRACE._record_error("Trace write failed; simulated in test.")
        assert "Trace write failed" in str(excinfo.value)
        assert all(
            rec.get("record") != RUN_COMPLETE_RECORD
            for rec in _read_raw_lines(trace_file)
        )
        with pytest.raises(TraceIncompleteError):
            load_jsonl_trace(trace_file)

    def test_a_bad_emit_never_kills_a_traced_run(self, tmp_path):
        """Instrumentation must not be able to end a run it is only observing."""
        trace_file = tmp_path / "run.jsonl"
        reached_the_end = []
        with pytest.raises(Exception) as excinfo:  # noqa: PT011 - asserted below
            with TRACE.write_jsonl(trace_file):
                TRACE.emit(EventType.OPT_START)
                TRACE.emit(EventType.ITER_START, no_such_field="boom")
                reached_the_end.append(True)
        # The run carried on past the bad emit ...
        assert reached_the_end == [True]
        # ... and the trace was refused rather than quietly shortened.
        assert "no_such_field" in str(excinfo.value)
        with pytest.raises(TraceIncompleteError):
            load_jsonl_trace(trace_file)

    def test_a_bad_emit_still_raises_on_the_in_memory_path(self):
        """``record()`` has no file to reject, so the mistake must surface."""
        with pytest.raises(TypeError):
            with TRACE.record():
                TRACE.emit(EventType.ITER_START, no_such_field="boom")


# ---------------------------------------------------------------------------
# Abrupt exit: SIGKILL a real child mid-run
# ---------------------------------------------------------------------------


#: The child emits ``_KILL_CHILD_EVENTS`` events, announces on stdout that they
#: are all on disk, and then keeps emitting forever so the SIGKILL genuinely
#: lands mid-write.  ``emit`` writes and flushes inline, so returning from the
#: loop is itself the guarantee that all ``_KILL_CHILD_EVENTS`` have landed --
#: there is nothing buffered anywhere to drain.  The child deliberately
#: installs no signal handler and no ``atexit`` hook, since SIGKILL would run
#: neither.
_KILL_CHILD_EVENTS = 200

_KILL_CHILD = """
import sys, time
from helix.trace import TRACE, EventType

path, n = sys.argv[1], int(sys.argv[2])
with TRACE.write_jsonl(path):
    for i in range(n):
        TRACE.emit(EventType.EVAL_START, candidate_id="g0-s%d" % i)
    sys.stdout.write("drained\\n")
    sys.stdout.flush()
    i = n
    while True:
        TRACE.emit(EventType.EVAL_START, candidate_id="g0-s%d" % i)
        i += 1
"""


class TestAbruptExit:
    def test_sigkilled_run_leaves_a_trace_that_is_unmistakably_rejectable(
        self, tmp_path
    ):
        """SIGKILL a real child; the trace it leaves must not read as complete.

        SIGKILL cannot be caught, so no ``finally``, no ``atexit`` and no
        context-manager exit runs in the child — exactly the case the orderly
        drain does not cover. What must survive is the *framing*: a header, a
        prefix of whole event lines, and no ``RUN_COMPLETE``.
        """
        trace_file = tmp_path / "killed.jsonl"
        proc = subprocess.Popen(
            [sys.executable, "-c", _KILL_CHILD, str(trace_file), str(_KILL_CHILD_EVENTS)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            assert proc.stdout is not None
            # Blocks until the child says its first ``_KILL_CHILD_EVENTS``
            # events have reached the file.  Everything it wrote up to that
            # point must therefore still be there after the kill — that is the
            # per-record flush being tested, not just the missing footer.
            handshake = proc.stdout.readline()
            assert handshake == "drained\n", (
                f"child never reached its handshake (stdout {handshake!r}, "
                f"stderr {proc.stderr.read() if proc.stderr else ''!r})"
            )
            proc.kill()
            proc.wait(timeout=30)
        finally:
            if proc.poll() is None:  # pragma: no cover - cleanup path
                proc.kill()
                proc.wait(timeout=30)
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()

        assert proc.returncode == -signal.SIGKILL, (
            f"child was not SIGKILLed (returncode {proc.returncode})"
        )

        raw = _read_raw_lines(trace_file)  # every line whole: no torn record
        assert raw[0]["record"] == HEADER_RECORD
        assert all(rec.get("record") != RUN_COMPLETE_RECORD for rec in raw), (
            "a killed run must never be stamped complete"
        )
        # Nothing acknowledged before the kill was lost in a write buffer: the
        # ids the child emitted before its handshake are all on disk.
        ids = [rec.get("candidate_id") for rec in raw if rec.get("type") == "EVAL_START"]
        assert ids[:_KILL_CHILD_EVENTS] == [
            f"g0-s{i}" for i in range(_KILL_CHILD_EVENTS)
        ]

        with pytest.raises(TraceIncompleteError) as excinfo:
            load_jsonl_trace(trace_file)
        assert RUN_COMPLETE_RECORD in str(excinfo.value)
        assert "killed" in str(excinfo.value)
