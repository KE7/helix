"""Tests for the JSONL timing trace (``helix.trace.traced`` / ``enable``)."""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from helix import executor, mutator, trace
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
from helix.trace import traced


def _read(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


@pytest.fixture
def trace_file(tmp_path: Path) -> Any:
    path = tmp_path / "trace.jsonl"
    trace.enable(path)
    try:
        yield path
    finally:
        trace.disable()


@pytest.fixture(autouse=True)
def _trace_off_after_test() -> Any:
    yield
    trace.disable()


class _BrokenSink:
    """A sink whose every write fails — stands in for a full or closed disk."""

    def write(self, _data: str) -> int:
        raise OSError("disk full")

    def flush(self) -> None:
        raise OSError("disk full")

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Decorator unit tests
# ---------------------------------------------------------------------------


class TestTracedDecorator:
    def test_passes_args_kwargs_and_return_through(self, trace_file: Path) -> None:
        @traced("t")
        def add(a: int, b: int, *, scale: int = 1) -> int:
            return (a + b) * scale

        assert add(1, 2, scale=10) == 30
        assert add.__name__ == "add"
        assert add.__wrapped__(1, 2) == 3  # type: ignore[attr-defined]

    def test_start_and_end_records_pair_up(self, trace_file: Path) -> None:
        @traced("t")
        def work() -> None:
            pass

        work()
        start, end = _read(trace_file)
        assert (start["event"], end["event"]) == ("start", "end")
        assert start["span"] == end["span"] == "t"
        assert start["span_id"] == end["span_id"]
        assert start["thread_id"] == end["thread_id"] == threading.get_ident()
        assert end["outcome"] == "ok" and "error_type" not in end
        assert end["duration_seconds"] >= 0
        assert end["monotonic"] >= start["monotonic"]
        assert end["duration_seconds"] == pytest.approx(
            end["monotonic"] - start["monotonic"]
        )
        assert isinstance(start["wall_time"], float)

    def test_error_outcome_records_class_name_and_propagates(
        self, trace_file: Path
    ) -> None:
        @traced("t")
        def boom() -> None:
            raise ValueError("secret message")

        with pytest.raises(ValueError, match="secret message"):
            boom()
        _start, end = _read(trace_file)
        assert end["outcome"] == "error"
        assert end["error_type"] == "ValueError"
        assert "secret message" not in trace_file.read_text()

    def test_writer_failure_never_masks_the_exception_in_flight(
        self, trace_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        @traced("t")
        def interrupted() -> None:
            raise KeyboardInterrupt

        trace._sink = _BrokenSink()  # type: ignore[assignment]
        with caplog.at_level(logging.ERROR, logger="helix.trace"):
            with pytest.raises(KeyboardInterrupt):
                interrupted()
            with pytest.raises(KeyboardInterrupt):
                interrupted()
        unavailable = [r for r in caplog.records if "Trace unavailable" in r.message]
        assert len(unavailable) == 1

    def test_writer_failure_on_a_successful_call_returns_normally(
        self, trace_file: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        @traced("t")
        def fine() -> str:
            return "ok"

        trace._sink = _BrokenSink()  # type: ignore[assignment]
        with caplog.at_level(logging.ERROR, logger="helix.trace"):
            assert fine() == "ok"
        assert any("Trace unavailable" in r.message for r in caplog.records)

    def test_thread_pool_writes_one_well_formed_record_per_line(
        self, trace_file: Path
    ) -> None:
        @traced("t")
        def work(i: int) -> int:
            return i

        with ThreadPoolExecutor(max_workers=8) as pool:
            assert sorted(pool.map(work, range(200))) == list(range(200))
        records = _read(trace_file)
        assert len(records) == 400
        by_id: dict[int, list[str]] = {}
        for r in records:
            by_id.setdefault(r["span_id"], []).append(r["event"])
        assert len(by_id) == 200
        assert all(sorted(v) == ["end", "start"] for v in by_id.values())
        assert len({r["thread_id"] for r in records}) > 1

    def test_disabled_mode_is_a_passthrough_that_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        calls: list[int] = []

        @traced("t", attrs=lambda a, k: calls.append(1) or {})
        def work(x: int) -> int:
            return x * 2

        assert not trace._enabled
        assert work(21) == 42
        assert calls == []  # attrs extraction is skipped entirely
        assert not list(tmp_path.iterdir())

    def test_explicit_attrs_callable_receives_args_and_kwargs(
        self, trace_file: Path
    ) -> None:
        @traced("t", attrs=lambda a, k: {"first": a[0], "flag": k.get("flag")})
        def work(x: str, *, flag: bool = False) -> None:
            pass

        work("a", flag=True)
        start, end = _read(trace_file)
        assert start["attrs"] == {"first": "a", "flag": True}
        assert end["attrs"] == start["attrs"]

    def test_attrs_extraction_failure_is_dropped_not_raised(
        self, trace_file: Path
    ) -> None:
        @traced("t", attrs=lambda a, k: {"x": a[5]})
        def work() -> str:
            return "ran"

        assert work() == "ran"
        assert _read(trace_file)[0]["attrs"] == {}

    def test_registry_attrs_for_the_five_spans(self) -> None:
        cand = Candidate(
            id="g1-s1",
            worktree_path="/tmp/x",
            branch_name="b",
            generation=1,
            parent_id=None,
            parent_ids=[],
            operation="mutate",
        )
        pre_ctx = (cand, None, ["i1"], "g2-s3")
        assert trace._ATTRS["proposal"]((pre_ctx,), {"gen": 2}) == {
            "candidate_id": "g2-s3",
            "generation": 2,
        }
        assert trace._ATTRS["evaluate"](
            (cand, None), {"split": "train", "evaluation_phase": "staged"}
        ) == {"candidate_id": "g1-s1", "split": "train", "evaluation_phase": "staged"}
        assert trace._ATTRS["evaluate"]((cand, None), {})["split"] == "val"
        assert trace._ATTRS["validate"]((cand,), {}) == {"candidate_id": "g1-s1"}
        assert trace._ATTRS["agent"]((), {"prompt_artifact_name": ".p.md"}) == {
            "prompt_artifact": ".p.md"
        }


class TestEnable:
    def test_env_var_is_the_fallback_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "env.jsonl"
        monkeypatch.setenv("HELIX_TRACE", str(path))
        assert trace.enable() is True

        @traced("t")
        def work() -> None:
            pass

        work()
        assert len(_read(path)) == 2

    def test_nothing_to_enable_stays_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("HELIX_TRACE", raising=False)
        assert trace.enable() is False
        assert not trace._enabled

    def test_unopenable_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises(OSError):
            trace.enable(tmp_path / "missing-dir" / "trace.jsonl")

    def test_records_are_appended_and_flushed_immediately(
        self, trace_file: Path
    ) -> None:
        @traced("t")
        def work() -> None:
            assert len(_read(trace_file)) == 1  # start is on disk mid-call

        work()
        assert len(_read(trace_file)) == 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


_TOML = 'objective = "improve"\n\n[evaluator]\ncommand = "pytest -q"\n'


class TestCliFlag:
    @pytest.mark.parametrize("command", ["evolve", "resume"])
    def test_trace_option_enables_the_sink_before_evolution(
        self, tmp_path: Path, mocker: Any, command: str
    ) -> None:
        (tmp_path / "helix.toml").write_text(_TOML)
        seen: list[bool] = []
        mocker.patch(
            "helix.evolution.run_evolution",
            side_effect=lambda *a, **k: seen.append(trace._enabled),
        )
        path = tmp_path / "out" / "trace.jsonl"
        path.parent.mkdir()
        result = CliRunner().invoke(
            cli, [command, "--dir", str(tmp_path), "--trace", str(path)]
        )
        assert result.exit_code == 0, result.output
        assert seen == [True]
        assert path.exists()

    def test_unopenable_trace_path_exits_2(self, tmp_path: Path, mocker: Any) -> None:
        (tmp_path / "helix.toml").write_text(_TOML)
        run = mocker.patch("helix.evolution.run_evolution")
        result = CliRunner().invoke(
            cli,
            ["evolve", "--dir", str(tmp_path), "--trace", str(tmp_path / "no" / "t")],
        )
        assert result.exit_code == 2
        assert "Cannot open trace file" in result.output
        run.assert_not_called()


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
    def test_a_real_run_emits_every_span_and_ends_with_the_run_end(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        trace_path = tmp_path / "trace.jsonl"
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
                max_generations=1,
                max_evaluations=1000,
                perfect_score_threshold=None,
                merge_enabled=False,
                frontier_type="instance",
            ),
            worktree=WorktreeConfig(),
        )

        trace.enable(trace_path)
        run_evolution(config, tmp_path, tmp_path / ".helix")
        trace.disable()

        records = _read(trace_path)
        assert records[0]["span"] == "run" and records[0]["event"] == "start"
        assert records[-1]["span"] == "run" and records[-1]["event"] == "end"
        assert records[-1]["outcome"] == "ok"

        ends = [r for r in records if r["event"] == "end"]
        spans = {r["span"] for r in ends}
        assert {"run", "seed", "validate", "proposal", "evaluate", "agent"} - spans == {
            "seed"  # non-seedless run: no seed generation
        }
        assert all(r["outcome"] == "ok" for r in ends)

        starts = {r["span_id"] for r in records if r["event"] == "start"}
        assert starts == {r["span_id"] for r in ends}

        proposal = next(r for r in ends if r["span"] == "proposal")
        assert proposal["attrs"]["candidate_id"].startswith("g1-")
        agent = next(r for r in ends if r["span"] == "agent")
        assert agent["attrs"]["prompt_artifact"] == mutator.MUTATION_PROMPT_ARTIFACT_NAME
        evaluate = next(r for r in ends if r["span"] == "evaluate")
        assert evaluate["attrs"]["split"] in {"train", "val"}
