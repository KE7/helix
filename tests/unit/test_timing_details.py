"""Tests for timing inside agent and evaluator calls: ``helix.timed`` and the
per-backend agent breakdown (``helix.agent_timing``)."""
from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from helix import agent_timing, timed, timing
from helix.cli import cli
from helix.config import AgentConfig, EvaluatorSidecarConfig, SandboxConfig
from helix.executor import run_evaluator
from helix.mutator import invoke_claude_code
from helix.timing import recording, span
from tests.unit.test_executor import (  # type: ignore[import-untyped]
    _prepare_batch,
    _result_line,
    make_candidate,
    make_config,
)


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _details(path: Path) -> list[dict[str, Any]]:
    return [r for r in _read(path) if r["event"] == "detail"]


@pytest.fixture
def timing_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / timing.timed_log_name()
    monkeypatch.setenv(timing.HELIX_TIMING_LOG_ENV, str(path))
    return path


def _lines(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# helix.timed
# ---------------------------------------------------------------------------


class TestTimed:
    def test_block_records_name_seconds_and_example(self, timing_log: Path) -> None:
        with timed("sam3", example=7):
            pass

        (line,) = _lines(timing_log)
        assert line["name"] == "sam3"
        assert line["example"] == "7"
        assert line["seconds"] >= 0

    def test_a_raising_block_is_still_timed_and_reraises(
        self, timing_log: Path
    ) -> None:
        with pytest.raises(ValueError):
            with timed("molmo"):
                raise ValueError("boom")

        assert _lines(timing_log)[0]["name"] == "molmo"

    def test_decorator_times_sync_and_async_calls(self, timing_log: Path) -> None:
        @timed("sync")
        def f(x: int) -> int:
            return x + 1

        @timed("async")
        async def g(x: int) -> int:
            await asyncio.sleep(0.01)
            return x * 2

        assert f(1) == 2
        assert asyncio.run(g(2)) == 4
        lines = {line["name"]: line for line in _lines(timing_log)}
        assert set(lines) == {"sync", "async"}
        assert lines["async"]["seconds"] >= 0.01
        assert f.__name__ == "f" and g.__name__ == "g"

    def test_decorated_calls_from_threads_do_not_share_a_start(
        self, timing_log: Path
    ) -> None:
        @timed("work")
        def work() -> None:
            pass

        threads = [threading.Thread(target=work) for _ in range(16)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        lines = _lines(timing_log)
        assert len(lines) == 16
        assert all(0 <= line["seconds"] < 1 for line in lines)

    def test_child_processes_append_to_the_same_log(self, timing_log: Path) -> None:
        code = "from helix import timed; timed.record('child', 0.5, example='e1')"
        for _ in range(3):
            subprocess.run([sys.executable, "-c", code], check=True)

        assert [line["name"] for line in _lines(timing_log)] == ["child"] * 3

    def test_record_takes_a_duration_measured_elsewhere(self, timing_log: Path) -> None:
        timed.record("owl-vit", 0.41, example="12")

        assert _lines(timing_log) == [{"name": "owl-vit", "seconds": 0.41, "example": "12"}]

    def test_outside_helix_it_is_a_no_op(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv(timing.HELIX_TIMING_LOG_ENV, raising=False)
        with timed("x"):
            pass
        timed.record("y", 1.0)
        assert list(tmp_path.iterdir()) == []

    def test_an_oversized_line_is_dropped(self, timing_log: Path) -> None:
        timed.record("x" * 5000, 1.0)
        assert not timing_log.exists()

    def test_totals_group_by_name_and_example_and_skip_bad_lines(self) -> None:
        raw = "\n".join([
            json.dumps({"name": "sam3", "seconds": 1.0, "example": "a"}),
            json.dumps({"name": "sam3", "seconds": 3.0, "example": "a"}),
            json.dumps({"name": "sam3", "seconds": 2.0, "example": "b"}),
            json.dumps({"name": "molmo", "seconds": 0.5}),
            "not json",
            json.dumps({"seconds": 1.0}),
            '{"name": "torn", "seco',
        ])

        totals = {(t["name"], t["example"]): t for t in timing.timed_totals(raw)}

        assert set(totals) == {("sam3", "a"), ("sam3", "b"), ("molmo", None)}
        assert totals[("sam3", "a")] == {
            "name": "sam3", "example": "a", "count": 2, "seconds": 4.0,
            "max_seconds": 3.0,
        }

    def test_read_is_empty_for_a_missing_file(self, tmp_path: Path) -> None:
        assert timing.read_timed_log(tmp_path / "missing.jsonl") == ""


class TestEvaluatorChannel:
    def test_timed_calls_in_the_evaluator_become_detail_records(
        self, tmp_path: Path
    ) -> None:
        worktree = tmp_path / "wt"
        _prepare_batch(worktree)
        command = (
            f"{sys.executable} -c "
            '"from helix import timed; '
            "timed.record('sam3', 1.5, example='example-0'); "
            "timed.record('sam3', 0.5, example='example-0'); "
            "print('" + _result_line([0.7]) + "')" + '"'
        )
        trace = tmp_path / "t.jsonl"
        with recording(trace), span("generation", generation=2):
            run_evaluator(make_candidate(str(worktree)), make_config(command=command))

        (detail,) = _details(trace)
        evaluate = next(r for r in _read(trace) if r.get("span") == "evaluate")
        assert detail["within"] == "evaluate" and detail["source"] == "timed"
        assert (detail["name"], detail["example"], detail["count"]) == ("sam3", "example-0", 2)
        assert detail["seconds"] == pytest.approx(2.0)
        assert detail["parent_span_id"] == evaluate["span_id"]
        assert (detail["generation"], detail["candidate"]) == (2, "cand-001")
        assert not list(worktree.glob(".helix_timing_*"))


    def test_a_sidecar_evaluator_reports_through_the_workspace_file(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        """Inside the evaluator sidecar the log lives at ``/workspace/...``
        and comes back after a marker line on the one capture command that
        also returns the ``helix.log`` notes, so it costs no extra container."""
        timed_line = json.dumps({"name": "molmo", "seconds": 0.25, "example": "3"})

        def sidecar(commands: list[list[str]], **kwargs: Any) -> list[MagicMock]:
            marker = re.search(r"echo (HELIX_TIMING_LOG_\w+)", commands[-1][2]).group(1)
            capture = '{"message": "note"}\n' + f"\n{marker}\n{timed_line}\n"
            return [
                MagicMock(stdout=_result_line([1.0]), stderr="", returncode=0),
                MagicMock(stdout=capture, stderr="", returncode=0),
            ]

        run = mocker.patch("helix.executor.run_sandboxed_commands", side_effect=sidecar)
        mocker.patch("helix.executor.current_evaluator_sidecar_runtime", return_value=MagicMock())
        config = make_config(command="python /runner/evaluate.py")
        config.evaluator.sidecar = EvaluatorSidecarConfig(
            image="eval:latest",
            runner_image="eval-runner:latest",
            command="python -m server",
            endpoint="http://helix-evaluator:8080/evaluate",
        )
        config = config.model_copy(update={"sandbox": SandboxConfig(enabled=True, evaluator=True)})
        worktree = tmp_path / "wt"
        _prepare_batch(worktree)

        trace = tmp_path / "t.jsonl"
        with recording(trace):
            result = run_evaluator(make_candidate(str(worktree)), config)

        commands = run.call_args.args[0]
        assert len(commands) == 2  # the evaluator and one capture command
        log_path = run.call_args.kwargs["env"][timing.HELIX_TIMING_LOG_ENV]
        assert log_path.startswith("/workspace/.helix_timing_")
        assert log_path in commands[-1][2]
        assert "note" in result.asi["log"]
        assert "HELIX_TIMING_LOG_" not in result.asi["log"]
        (detail,) = _details(trace)
        assert (detail["name"], detail["example"], detail["seconds"]) == ("molmo", "3", 0.25)


# ---------------------------------------------------------------------------
# Agent breakdown
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _claude_transcript(path: Path) -> Path:
    def at(seconds: float) -> str:
        return f"2026-10-27T00:00:{seconds:06.3f}Z"

    return _write_jsonl(path, [
        {"type": "user", "timestamp": at(0), "message": {"content": "do it"}},
        # Two tool calls in one message run in parallel.
        {"type": "assistant", "timestamp": at(10), "message": {"content": [
            {"type": "tool_use", "id": "a", "name": "Bash"},
            {"type": "tool_use", "id": "b", "name": "Read"},
        ]}},
        {"type": "user", "timestamp": at(12), "message": {"content": [
            {"type": "tool_result", "tool_use_id": "b"}]}},
        {"type": "user", "timestamp": at(20), "message": {"content": [
            {"type": "tool_result", "tool_use_id": "a"}]}},
        # A subagent's own calls sit inside its Task call and are skipped.
        {"type": "assistant", "isSidechain": True, "timestamp": at(21), "message": {
            "content": [{"type": "tool_use", "id": "s", "name": "Grep"}]}},
        {"type": "assistant", "timestamp": at(30), "message": {"content": [
            {"type": "text", "text": "done"}]}},
    ])


def _by_name(totals: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {t["name"]: t for t in totals}


class TestAgentBreakdown:
    def test_claude_pairs_tool_use_with_tool_result(self, tmp_path: Path) -> None:
        totals = _by_name(
            agent_timing.breakdown("claude", _claude_transcript(tmp_path / "c.jsonl"), 35.0)
        )

        assert set(totals) == {"model", "other", "tool:Bash", "tool:Read"}
        assert totals["tool:Bash"]["seconds"] == pytest.approx(10.0)
        assert totals["tool:Read"]["seconds"] == pytest.approx(2.0)
        # Window 30 s, tools cover 10 s of it (they overlap), wall 35 s.
        assert totals["model"]["seconds"] == pytest.approx(20.0)
        assert totals["other"]["seconds"] == pytest.approx(5.0)

    def test_codex_pairs_calls_with_outputs_by_call_id(self, tmp_path: Path) -> None:
        def row(seconds: int, **payload: Any) -> dict[str, Any]:
            return {"timestamp": f"2026-10-27T00:00:{seconds:02d}.000Z",
                    "type": "response_item", "payload": payload}

        path = _write_jsonl(tmp_path / "x.jsonl", [
            {"timestamp": "2026-10-27T00:00:00.000Z", "type": "session_meta", "payload": {}},
            row(5, type="function_call", name="exec_command", call_id="1"),
            row(9, type="function_call_output", call_id="1"),
            row(12, type="custom_tool_call", name="apply_patch", call_id="2"),
            row(13, type="custom_tool_call_output", call_id="2"),
            row(20, type="message"),
        ])

        totals = _by_name(agent_timing.breakdown("codex", path, 20.0))

        assert totals["tool:exec_command"]["seconds"] == pytest.approx(4.0)
        assert totals["tool:apply_patch"]["seconds"] == pytest.approx(1.0)
        assert totals["model"]["seconds"] == pytest.approx(15.0)
        assert totals["other"]["seconds"] == pytest.approx(0.0)

    def test_opencode_reads_tool_part_times(self, tmp_path: Path) -> None:
        def row(table: str, data: dict[str, Any]) -> dict[str, Any]:
            return {"table": table, "row": {"data": json.dumps(data)}}

        path = _write_jsonl(tmp_path / "o.jsonl", [
            {"table": "session", "row": {"id": "s"}},
            row("message", {"role": "assistant", "time": {"created": 1_000, "completed": 9_000}}),
            row("part", {"type": "tool", "tool": "bash",
                         "state": {"time": {"start": 2_000, "end": 5_000}}}),
            row("part", {"type": "reasoning", "time": {"start": 1_000, "end": 2_000}}),
        ])

        totals = _by_name(agent_timing.breakdown("opencode", path, 10.0))

        assert totals["tool:bash"]["seconds"] == pytest.approx(3.0)
        assert totals["model"]["seconds"] == pytest.approx(5.0)
        assert totals["other"]["seconds"] == pytest.approx(2.0)

    @pytest.mark.parametrize("backend", ["cursor", "agy"])
    def test_backends_without_usable_timestamps_are_unavailable(
        self, backend: str, tmp_path: Path
    ) -> None:
        (total,) = agent_timing.breakdown(backend, tmp_path / "t.jsonl", 12.0)
        assert total["name"] == "unavailable"
        assert total["seconds"] == 12.0
        assert backend in total["reason"]

    def test_a_missing_or_unreadable_transcript_is_unavailable(self, tmp_path: Path) -> None:
        garbage = tmp_path / "g.jsonl"
        garbage.write_text('{"table": "part", "row": {"data": "{not json"}}\n')
        for path in (None, tmp_path / "missing.jsonl", garbage):
            (total,) = agent_timing.breakdown("opencode", path, 3.0)
            assert total["name"] == "unavailable"

    def test_clock_skew_never_makes_time_negative(self, tmp_path: Path) -> None:
        totals = _by_name(
            agent_timing.breakdown("claude", _claude_transcript(tmp_path / "c.jsonl"), 25.0)
        )
        assert totals["other"]["seconds"] == 0.0
        assert all(t["seconds"] >= 0 for t in totals.values())


class TestAgentChannel:
    def test_agent_call_records_its_breakdown_and_its_scripts_timed_calls(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        worktree = tmp_path / "wt"
        worktree.mkdir()
        _claude_transcript(worktree / "transcript.jsonl")
        mocker.patch(
            "helix.mutator._collect_backend_transcript_artifacts",
            return_value=[{"available": True, "path": "transcript.jsonl"}],
        )

        def agent(args: list[str], **kwargs: Any) -> MagicMock:
            # A script the agent ran reports its own timing.
            log = Path(kwargs["env"][timing.HELIX_TIMING_LOG_ENV])
            log.write_text(json.dumps({"name": "sam3", "seconds": 2.5}) + "\n")
            return MagicMock(stdout=json.dumps({"result": "ok"}), stderr="", returncode=0)

        mocker.patch("helix.mutator.subprocess.run", side_effect=agent)
        trace = tmp_path / "t.jsonl"
        with recording(trace), span("mutate", candidate_id="g1-s1"):
            invoke_claude_code(str(worktree), "do it", AgentConfig())

        details = _details(trace)
        agent_span = next(r for r in _read(trace) if r.get("span") == "agent")
        assert all(d["parent_span_id"] == agent_span["span_id"] for d in details)
        assert all(d["within"] == "agent" and d["candidate"] == "g1-s1" for d in details)
        sources = {(d["source"], d["name"]) for d in details}
        assert ("timed", "sam3") in sources
        assert {("transcript", "model"), ("transcript", "tool:Bash")} <= sources
        assert not list(worktree.glob(".helix_timing_*"))


# ---------------------------------------------------------------------------
# Reading: helix timing and Perfetto
# ---------------------------------------------------------------------------


def _trace_with_details(path: Path) -> Path:
    with recording(path), span("run"):
        with span("generation", generation=1):
            with span("proposal", candidate_id="g1-s1"), span("agent"):
                timing.detail("agent", "transcript", {
                    "name": "tool:Bash", "example": None, "count": 3, "seconds": 9.0,
                    "max_seconds": 5.0})
            with span("validate", candidate_id="g1-s1"), span("evaluate"):
                timing.detail("evaluate", "timed", {
                    "name": "sam3", "example": "e1", "count": 4, "seconds": 2.0,
                    "max_seconds": 1.0})
            timing.decision("g1-s1", accepted=False, reason="r")
    return path


class TestReadingDetails:
    def test_summarize_totals_details_by_fate(self, tmp_path: Path) -> None:
        summary = timing.summarize(timing.read_records(_trace_with_details(tmp_path / "t.jsonl")))

        assert summary.details[("agent", "transcript", "tool:Bash")] == {
            "count": 3, "max_seconds": 5.0, "seconds": {"rejected": 9.0}}
        assert summary.details[("evaluate", "timed", "sam3")]["seconds"] == {"rejected": 2.0}

    def test_perfetto_shows_details_on_their_slice(self, tmp_path: Path) -> None:
        records = timing.read_records(_trace_with_details(tmp_path / "t.jsonl"))
        slices = {e["name"]: e for e in timing.to_perfetto(records)["traceEvents"] if e["ph"] == "X"}

        assert slices["agent"]["args"]["details"] == {"tool:Bash": 9.0}
        assert slices["evaluate"]["args"]["details"] == {"sam3[e1]": 2.0}
        assert "details" not in slices["run"]["args"]

    def test_cli_prints_the_inside_table(self, tmp_path: Path) -> None:
        helix_dir = tmp_path / ".helix"
        helix_dir.mkdir()
        _trace_with_details(helix_dir / timing.TIMING_FILE_NAME)

        result = CliRunner().invoke(cli, ["timing", "--dir", str(tmp_path)], terminal_width=150)

        assert result.exit_code == 0, result.output
        assert "Inside agent and evaluator calls" in result.output
        assert "tool:Bash" in result.output
        assert "evaluate (timed)" in result.output
