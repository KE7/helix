"""Tests for the always-on timing trace (``helix.timing``)."""
from __future__ import annotations

import contextvars
import inspect
import json
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

# Imported for their ``@traced`` decorations, so ``timing.TRACED`` is complete.
import helix.evolution  # noqa: F401
import helix.merger  # noqa: F401
from helix import executor, mutator, timing
from helix.cli import cli
from helix.config import (
    DatasetConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    WorktreeConfig,
)
from helix.display import UsageStats
from helix.evolution import run_evolution
from helix.population import Candidate, EvalResult
from helix.timing import recording, span, traced


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def _ends(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in records if r["event"] == "end"]


class _BrokenSink:
    """A sink whose every write fails: stands in for a full disk."""

    def write(self, _data: str) -> int:
        raise OSError("disk full")

    def flush(self) -> None:
        raise OSError("disk full")

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# span / recording
# ---------------------------------------------------------------------------


class TestSpan:
    def test_start_and_end_pair_up_with_duration_and_attrs(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        with recording(path), span("work", candidate_id="g1-s0"):
            pass

        start, end = _read(path)
        assert start["event"] == "start" and end["event"] == "end"
        assert start["span_id"] == end["span_id"]
        assert start["run_id"] == end["run_id"]
        assert end["attrs"] == {"candidate_id": "g1-s0"}
        assert end["outcome"] == "ok" and "error_type" not in end
        assert end["duration_seconds"] >= 0
        assert end["wall_time"] >= start["wall_time"]

    def test_nested_spans_link_to_their_parent(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        with recording(path), span("outer"):
            with span("inner"):
                pass
            with span("sibling"):
                pass

        ends = {r["span"]: r for r in _ends(_read(path))}
        assert ends["outer"]["parent_span_id"] is None
        assert ends["inner"]["parent_span_id"] == ends["outer"]["span_id"]
        assert ends["sibling"]["parent_span_id"] == ends["outer"]["span_id"]

    @pytest.mark.parametrize("exc_type", [ValueError, KeyboardInterrupt])
    def test_error_records_class_name_only_and_propagates(
        self, tmp_path: Path, exc_type: type[BaseException]
    ) -> None:
        path = tmp_path / "t.jsonl"
        with pytest.raises(exc_type, match="secret detail"):
            with recording(path), span("work"):
                raise exc_type("secret detail")

        (end,) = _ends(_read(path))
        assert end["outcome"] == "error"
        assert end["error_type"] == exc_type.__name__
        assert "secret detail" not in path.read_text()

    def test_outside_a_recording_spans_are_no_ops(self, tmp_path: Path) -> None:
        with span("work"):
            pass

        @traced("fn")
        def fn(x: int) -> int:
            return x + 1

        assert fn(1) == 2
        assert list(tmp_path.iterdir()) == []

    def test_unopenable_trace_warns_and_the_block_still_runs(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        blocker = tmp_path / "file"
        blocker.write_text("")
        ran = []
        with caplog.at_level(logging.WARNING, logger="helix.timing"):
            with recording(blocker / "t.jsonl"), span("work"):
                ran.append(True)

        assert ran == [True]
        assert "Timing trace disabled" in caplog.text

    def test_broken_sink_never_masks_the_exception_in_flight(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        with recording(tmp_path / "t.jsonl"):
            timing._sink = _BrokenSink()  # type: ignore[assignment]
            with caplog.at_level(logging.WARNING, logger="helix.timing"):
                with pytest.raises(KeyboardInterrupt):
                    with span("a"), span("b"):
                        raise KeyboardInterrupt

        assert caplog.text.count("Timing trace write failed") == 1

    def test_broken_sink_on_a_successful_block_returns_normally(
        self, tmp_path: Path
    ) -> None:
        @traced("fn")
        def fn() -> str:
            return "value"

        with recording(tmp_path / "t.jsonl"):
            timing._sink = _BrokenSink()  # type: ignore[assignment]
            assert fn() == "value"

    def test_each_recording_appends_under_its_own_run_id(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        for _ in range(2):
            with recording(path), span("run"), span("child"):
                pass

        records = _read(path)
        assert len(records) == 8
        assert len({r["run_id"] for r in records}) == 2
        assert len({r["span_id"] for r in records}) == 4

    def test_recording_restores_the_enclosing_sink(self, tmp_path: Path) -> None:
        outer, inner = tmp_path / "outer.jsonl", tmp_path / "inner.jsonl"
        with recording(outer):
            with recording(inner), span("inner"):
                pass
            with span("outer"):
                pass

        assert {r["span"] for r in _read(inner)} == {"inner"}
        assert {r["span"] for r in _read(outer)} == {"outer"}
        assert timing._sink is None

    def test_exit_stack_closes_one_span_per_iteration(self, tmp_path: Path) -> None:
        """The pattern ``run_evolution`` uses for its ``generation`` span."""
        path = tmp_path / "t.jsonl"
        with recording(path), ExitStack() as stack:
            for gen in (1, 2, 3):
                stack.close()
                stack.enter_context(span("generation", generation=gen))
                if gen == 2:
                    continue
                with span("child"):
                    pass

        records = _read(path)
        gens = [r for r in _ends(records) if r["span"] == "generation"]
        assert [g["attrs"]["generation"] for g in gens] == [1, 2, 3]
        children = [r for r in _ends(records) if r["span"] == "child"]
        assert [c["parent_span_id"] for c in children] == [
            gens[0]["span_id"],
            gens[2]["span_id"],
        ]
        # Generations are siblings, not nested in each other.
        assert all(g["parent_span_id"] is None for g in gens)


class TestStamping:
    def test_outermost_generation_and_candidate_win(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        with recording(path), span("run"):
            with span("generation", generation=3):
                with span("proposal", candidate_id="g3-s1", generation=3):
                    # The proposal's eval of its parent is work for the child.
                    with span("cached_evaluate", candidate_id="g2-s0"):
                        with span("evaluate", candidate_id="g2-s0"):
                            pass
                    with span("agent"):
                        pass
                timing.decision("g3-s1", accepted=False, reason="rejected")

        by_span = {r["span"]: r for r in _read(path) if r["event"] == "end"}
        assert (by_span["run"]["generation"], by_span["run"]["candidate"]) == (None, None)
        assert by_span["generation"]["generation"] == 3
        assert by_span["generation"]["candidate"] is None
        for name in ("proposal", "cached_evaluate", "evaluate", "agent"):
            assert by_span[name]["generation"] == 3
            assert by_span[name]["candidate"] == "g3-s1"
        assert by_span["evaluate"]["attrs"]["candidate_id"] == "g2-s0"
        (mark,) = [r for r in _read(path) if r["event"] == "decision"]
        assert (mark["generation"], mark["candidate"]) == (3, "g3-s1")

    def test_a_thread_pool_inherits_through_a_copied_context(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "t.jsonl"
        with recording(path), span("generation", generation=5):
            with ThreadPoolExecutor(max_workers=2) as pool:
                def agent() -> None:
                    with span("agent"):
                        pass

                pool.submit(contextvars.copy_context().run, agent).result()

        agent = next(r for r in _read(path) if r["span"] == "agent")
        assert agent["generation"] == 5


class TestReading:
    def _trace(self, tmp_path: Path) -> list[dict[str, Any]]:
        path = tmp_path / "t.jsonl"
        with recording(path), span("run"):
            for gen, child, accepted in ((1, "g1-s1", True), (2, "g2-s2", False)):
                with span("generation", generation=gen):
                    with span("proposal", candidate_id=child):
                        with span("agent"):
                            time.sleep(0.01)
                        with span("evaluate", candidate_id=child):
                            pass
                    with span("save_state"):
                        pass
                    timing.decision(child, accepted=accepted, reason="r")
            with span("generation", generation=3), span("proposal", candidate_id="g3-s3"):
                with span("agent"):
                    pass
        with path.open("a") as f:
            f.write('{"event": "end", "span": "tor')  # killed mid-write
        return timing.read_records(path)

    def test_read_records_skips_a_torn_last_line(self, tmp_path: Path) -> None:
        records = self._trace(tmp_path)
        assert records[-1]["span"] == "run"

    def test_summarize_splits_work_by_fate(self, tmp_path: Path) -> None:
        summary = timing.summarize(self._trace(tmp_path))
        assert set(summary.work) == {"agent", "evaluator", "state save"}
        assert set(summary.work["agent"]) == {"accepted", "rejected", "undecided"}
        assert summary.work["agent"]["accepted"] >= 0.01
        assert set(summary.work["state save"]) == {"unattributed"}
        assert summary.candidates == {"accepted": 1, "rejected": 1, "undecided": 1}
        assert sorted(summary.generation_seconds) == [1, 2, 3]
        assert summary.run_seconds >= sum(summary.generation_seconds.values())

    def test_perfetto_export_has_one_slice_per_span_and_the_fates(
        self, tmp_path: Path
    ) -> None:
        records = self._trace(tmp_path)
        events = timing.to_perfetto(records)["traceEvents"]
        slices = [e for e in events if e["ph"] == "X"]
        assert len(slices) == sum(r["event"] == "end" for r in records)
        assert {e["name"] for e in events if e["ph"] == "i"} == {"accepted", "rejected"}
        agent = next(e for e in slices if e["name"] == "agent")
        assert agent["args"]["candidate"] == "g1-s1"
        assert agent["args"]["fate"] == "accepted"
        assert agent["dur"] >= 10_000  # microseconds
        json.dumps(events)

    def test_cli_prints_the_summary_and_writes_perfetto(self, tmp_path: Path) -> None:
        helix_dir = tmp_path / ".helix"
        helix_dir.mkdir()
        trace_dir = tmp_path / "src"
        trace_dir.mkdir()
        self._trace(trace_dir)
        (trace_dir / "t.jsonl").rename(helix_dir / timing.TIMING_FILE_NAME)
        out = tmp_path / "trace.json"

        result = CliRunner().invoke(
            cli, ["timing", "--dir", str(tmp_path), "--perfetto", str(out)]
        )

        assert result.exit_code == 0, result.output
        assert "Work by kind and candidate fate" in result.output
        assert "1 accepted, 1 rejected, 1 undecided" in result.output
        assert json.loads(out.read_text())["traceEvents"]

    def test_cli_without_a_trace_says_so(self, tmp_path: Path) -> None:
        result = CliRunner().invoke(cli, ["timing", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No timing trace found" in result.output


class TestDecision:
    def test_records_the_fate_under_the_open_span(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        with recording(path), span("generation"):
            timing.decision("g1-s0", accepted=False, reason="train-gate-rejected")

        start, mark, _end = _read(path)
        assert mark["event"] == "decision"
        assert mark["candidate"] == "g1-s0"
        assert mark["accepted"] is False
        assert mark["reason"] == "train-gate-rejected"
        assert mark["parent_span_id"] == start["span_id"]
        assert mark["run_id"] == start["run_id"]

    def test_outside_a_recording_is_a_no_op(self) -> None:
        timing.decision("g1-s0", accepted=True, reason="mutation")


class TestThreads:
    def test_copied_context_keeps_the_parent_and_lines_stay_whole(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "t.jsonl"

        @traced("task")
        def task(i: int) -> int:
            with span("inner", i=i):
                return i

        with recording(path), span("dispatch"):
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [
                    pool.submit(contextvars.copy_context().run, task, i)
                    for i in range(64)
                ]
                assert sorted(f.result() for f in futures) == list(range(64))

        records = _read(path)  # every line parses
        assert len(records) == 2 * (1 + 64 + 64)
        ends = _ends(records)
        dispatch = next(r for r in ends if r["span"] == "dispatch")
        tasks = {r["span_id"] for r in ends if r["span"] == "task"}
        assert all(
            r["parent_span_id"] == dispatch["span_id"]
            for r in ends
            if r["span"] == "task"
        )
        assert all(r["parent_span_id"] in tasks for r in ends if r["span"] == "inner")

    def test_a_thread_without_the_context_still_records(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"
        with recording(path), span("dispatch"):
            def orphan() -> None:
                with span("orphan"):
                    pass

            worker = threading.Thread(target=orphan)
            worker.start()
            worker.join()

        orphan = next(r for r in _read(path) if r["span"] == "orphan")
        assert orphan["parent_span_id"] is None


# ---------------------------------------------------------------------------
# traced
# ---------------------------------------------------------------------------


class TestTraced:
    def test_passes_args_kwargs_and_return_through(self, tmp_path: Path) -> None:
        @traced("fn")
        def fn(a: int, b: int = 0, *, c: int = 0) -> tuple[int, int, int]:
            return a, b, c

        with recording(tmp_path / "t.jsonl"):
            assert fn(1, 2, c=3) == (1, 2, 3)
        assert fn.__name__ == "fn"
        assert list(inspect.signature(fn).parameters) == ["a", "b", "c"]

    def test_attrs_see_arguments_by_name_with_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "t.jsonl"

        @traced("fn", lambda a: {"x": a["x"], "split": a["split"]})
        def fn(x: int, split: str = "val") -> None:
            pass

        with recording(path):
            fn(1)
            fn(2, "train")
            fn(x=3, split="train")

        assert [r["attrs"] for r in _ends(_read(path))] == [
            {"x": 1, "split": "val"},
            {"x": 2, "split": "train"},
            {"x": 3, "split": "train"},
        ]

    def test_attrs_failure_keeps_the_span_and_warns_once(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "t.jsonl"

        @traced("broken_attrs", lambda a: {"x": a["missing"]})
        def fn() -> int:
            return 7

        with caplog.at_level(logging.WARNING, logger="helix.timing"):
            with recording(path):
                assert fn() == 7
                assert fn() == 7

        assert [r["attrs"] for r in _ends(_read(path))] == [{}, {}]
        assert caplog.text.count("Timing attrs for span 'broken_attrs'") == 1

    @pytest.mark.parametrize(
        "name,fn,attrs",
        [entry for entry in timing.TRACED if entry[2] is not None],
        ids=lambda v: v if isinstance(v, str) else "",
    )
    def test_every_attrs_extractor_matches_its_function_signature(
        self, name: str, fn: Any, attrs: Any
    ) -> None:
        """A renamed parameter fails here instead of silently emptying attrs."""
        signature = inspect.signature(fn)
        bound = signature.bind_partial(
            **{
                p.name: MagicMock()
                for p in signature.parameters.values()
                if p.default is inspect.Parameter.empty
            }
        )
        bound.apply_defaults()
        assert isinstance(attrs(bound.arguments), dict)


# ---------------------------------------------------------------------------
# End to end through run_evolution
# ---------------------------------------------------------------------------


def _candidate(cid: str, root: Path, generation: int = 0) -> Candidate:
    wt = root / cid
    wt.mkdir(parents=True, exist_ok=True)
    return Candidate(
        id=cid,
        worktree_path=str(wt),
        branch_name=f"helix/{cid}",
        generation=generation,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def _eval(cand: Candidate, split: str, ids: list[str] | None) -> EvalResult:
    scores = {i: 0.5 for i in (ids or ["i1", "i2"])}
    return EvalResult(
        candidate_id=cand.id,
        scores={},
        asi={},
        instance_scores=scores,
        objective_scores=[{"quality": s} for s in scores.values()],
    )


class TestEndToEnd:
    # 1 slot runs in the dispatching thread; 3 run in the thread pool, where
    # the spans only nest under their generation if the context was copied.
    @pytest.mark.parametrize("slots", [1, 3])
    def test_every_run_writes_a_span_tree_to_the_helix_dir(
        self,
        tmp_path: Path,
        mocker: Any,
        caplog: pytest.LogCaptureFixture,
        slots: int,
    ) -> None:
        worktrees = tmp_path / "wt"
        seed = _candidate("g0-s0", worktrees)

        # Real ``run_evaluator`` / ``mutate`` / ``invoke_claude_code`` run
        # through the differential-testing hooks; only git and persistence
        # are stubbed.
        mocker.patch.object(executor, "_EVALUATOR_OVERRIDE", _eval)
        mocker.patch.object(
            mutator,
            "_MUTATOR_OVERRIDE",
            lambda wt, prompt, cfg: ({"result": "ok"}, UsageStats()),
        )
        mocker.patch(
            "helix.mutator.clone_candidate",
            side_effect=lambda parent, new_id, base: _candidate(new_id, worktrees, 1),
        )
        mocker.patch("helix.mutator.snapshot_candidate", return_value="abc123")
        mocker.patch("helix.mutator.remove_worktree")
        mocker.patch("helix.evaluator_manifest.snapshot_candidate")
        for name in (
            "remove_worktree",
            "save_state",
            "init_base_dir",
            "_save_evaluation",
            "record_entry",
            "snapshot_candidate",
            "HelixLiveDisplay",
        ):
            mocker.patch(f"helix.evolution.{name}")
        mocker.patch("helix.evolution.create_seed_worktree", return_value=seed)
        mocker.patch("helix.evolution.load_state", return_value=None)
        mocker.patch("helix.evolution._load_evaluation", return_value=None)
        mocker.patch("helix.evolution.load_lineage", return_value={})

        config = HelixConfig(
            objective="Improve the code",
            evaluator=EvaluatorConfig(command="pytest -q"),
            dataset=DatasetConfig(),
            evolution=EvolutionConfig(
                max_generations=2,
                max_evaluations=1000,
                perfect_score_threshold=None,
                merge_enabled=False,
                frontier_type="instance",
                num_parallel_proposals=slots,
                max_workers=slots,
            ),
            worktree=WorktreeConfig(),
        )

        base_dir = tmp_path / ".helix"
        with caplog.at_level(logging.WARNING, logger="helix.timing"):
            run_evolution(config, tmp_path, base_dir)
        assert "Timing" not in caplog.text

        records = _read(base_dir / timing.TIMING_FILE_NAME)
        assert records[0]["span"] == "run" and records[0]["event"] == "start"
        assert records[-1]["span"] == "run" and records[-1]["event"] == "end"
        assert len({r["run_id"] for r in records}) == 1

        ends = _ends(records)
        assert all(r["outcome"] == "ok" for r in ends)
        assert {r["span_id"] for r in records if r["event"] == "start"} == {
            r["span_id"] for r in ends
        }
        assert {
            "run", "generation", "validate", "proposal", "mutate", "agent",
            "cached_evaluate", "evaluate", "save_state",
        } <= {r["span"] for r in ends}

        # One tree: every span but the run has a recorded parent.
        by_id = {r["span_id"]: r for r in ends}
        run = next(r for r in ends if r["span"] == "run")
        assert all(
            r["parent_span_id"] in by_id for r in ends if r is not run
        )

        # Rejected candidates are timed like accepted ones, and every
        # candidate the run worked on ends with exactly one decision.
        decisions = [r for r in records if r["event"] == "decision"]
        decided = [d["candidate"] for d in decisions]
        assert len(decided) == len(set(decided))
        worked_on = {
            r["attrs"]["candidate_id"]
            for r in ends
            if r["span"] in {"proposal", "merge"}
        }
        assert worked_on | {seed.id} == set(decided)
        fates = {d["candidate"]: d for d in decisions}
        assert fates[seed.id]["accepted"] is True
        rejected = {c for c, d in fates.items() if not d["accepted"]}
        assert rejected, "the stub evaluator never improves, so children are rejected"
        for cid in rejected:
            assert {"proposal", "mutate", "cached_evaluate"} <= {
                r["span"] for r in ends if r["attrs"].get("candidate_id") == cid
            }

        # Stamped fields make every question a flat group-by: everything
        # inside a generation knows it, and an agent call knows the child it
        # was for.
        def ancestors(r: dict[str, Any]) -> list[dict[str, Any]]:
            chain = []
            while r["parent_span_id"] is not None:
                r = by_id[r["parent_span_id"]]
                chain.append(r)
            return chain

        for r in ends:
            gen = next(
                (a["attrs"]["generation"] for a in ancestors(r) if a["span"] == "generation"),
                None,
            )
            if gen is not None:
                assert r["generation"] == gen
        proposals = {r["span_id"]: r for r in ends if r["span"] == "proposal"}
        for r in ends:
            if r["span"] == "agent":
                owner = by_id[by_id[r["parent_span_id"]]["parent_span_id"]]
                assert r["candidate"] == owner["attrs"]["candidate_id"]
                assert owner["span_id"] in proposals

        def parent(r: dict[str, Any]) -> str:
            return str(by_id[r["parent_span_id"]]["span"])

        assert sum(r["span"] == "proposal" for r in ends) == 2 * slots
        gens = [r for r in ends if r["span"] == "generation"]
        assert [g["attrs"]["generation"] for g in gens] == [1, 2]
        assert all(parent(g) == "run" for g in gens)
        for r in ends:
            if r["span"] == "proposal":
                assert parent(r) == "generation"
                assert r["attrs"]["candidate_id"].startswith(
                    f"g{r['generation']}-"
                )
            elif r["span"] == "mutate":
                assert parent(r) == "proposal"
            elif r["span"] == "agent":
                assert parent(r) == "mutate"
                assert r["attrs"]["backend"] == config.agent.backend
            elif r["span"] == "save_state":
                assert parent(r) in {"run", "generation"}
            elif r["span"] == "evaluate":
                assert parent(r) == "cached_evaluate"
                assert r["attrs"]["split"] in {"train", "val"}


class TestFormatting:
    @pytest.mark.parametrize(
        "seconds,text",
        [(0.04, "0.0s"), (59.94, "59.9s"), (59.96, "1m00s"), (119.7, "2m00s"),
         (3599.6, "1h00m"), (7322.0, "2h02m")],
    )
    def test_fmt_seconds_never_shows_60(self, seconds: float, text: str) -> None:
        from helix.cli import _fmt_seconds

        assert _fmt_seconds(seconds) == text

    def test_runs_do_not_grow_the_traced_registry(self) -> None:
        # Runs after TestEndToEnd in this file: a per-run closure decorated
        # with @traced would be registered once per run_evolution call.
        assert "save_state" not in [name for name, _, _ in timing.TRACED]
