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
import threading
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from helix.cli import cli
from helix.config import EvaluatorConfig, HelixConfig
from helix.population import Candidate, EvalResult
from helix.trace import TRACE, Event, EventType


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
    """Parse *path* as JSONL, failing loudly on any malformed line."""
    records = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as exc:  # pragma: no cover - failure path
            pytest.fail(f"{path}:{lineno} is not valid JSON ({exc}): {line!r}")
    return records


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
            assert ":" in rec["source"]

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

    def test_bus_is_restored_after_the_traced_run(self, tmp_path, mocker):
        project = _make_project(tmp_path / "proj")
        trace_file = tmp_path / "run.jsonl"

        def fake_run_evolution(config, project_root, base_dir):
            TRACE.emit(EventType.OPT_START)

        mocker.patch("helix.evolution.run_evolution", fake_run_evolution)
        CliRunner().invoke(
            cli, ["evolve", "--dir", str(project), "--trace", str(trace_file)]
        )

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

    def test_trace_survives_a_crashing_run(self, tmp_path):
        """Events emitted before an exception are already on disk."""
        trace_file = tmp_path / "run.jsonl"
        with pytest.raises(RuntimeError):
            with TRACE.write_jsonl(trace_file):
                TRACE.emit(EventType.OPT_START)
                TRACE.emit(EventType.ITER_START, decision="0")
                raise RuntimeError("killed mid-run")
        assert [r["type"] for r in _read_jsonl(trace_file)] == [
            "OPT_START",
            "ITER_START",
        ]


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
        assert end.wall_time >= start.wall_time

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
        assert end.reason == "ok"
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
        assert ends[0].reason == "mutation_error"


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
            TRACE.emit(EventType.MUTATE_END, candidate_id=f"g1-s{i}", reason="ok")

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
        assert Path(filename).name == Path(__file__).name
        assert int(lineno) == expected_line

    def test_record_still_yields_the_in_memory_event_list(self):
        with TRACE.record() as events:
            TRACE.emit(EventType.CACHE_PUT, candidate_id="c1")
            assert len(events) == 1
            assert isinstance(events[0], Event)
        assert TRACE.enabled is False
