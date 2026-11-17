"""Tests for ``usage`` records: each budget charge of agent tokens and cost,
written to the timing trace by ``helix.budget.charge_llm_usage``."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner

from helix import budget, timing
from helix.cli import cli
from helix.display import UsageStats
from helix.state import BudgetState, EvolutionState
from helix.timing import recording, span

USAGE_KEYS = [
    "event", "candidate", "source",
    "input_tokens", "output_tokens", "cached_input_tokens",
    "cache_creation_input_tokens", "cache_read_input_tokens", "reasoning_tokens",
    "cost_usd", "parent_span_id", "run_id", "generation", "wall_time", "thread_id",
]


def _state() -> EvolutionState:
    return EvolutionState(
        generation=0, frontier=[], instance_scores={}, budget=BudgetState(), config_hash="h"
    )


def _stats(scale: int = 1) -> UsageStats:
    return UsageStats(
        input_tokens=100 * scale,
        output_tokens=20 * scale,
        cached_input_tokens=30 * scale,
        cache_creation_input_tokens=5 * scale,
        cache_read_input_tokens=7 * scale,
        reasoning_tokens=9 * scale,
        num_turns=3,
        tool_names=["exec"],
        cost_usd=0.25 * scale,
    )


def _usage(path: Path) -> list[dict[str, Any]]:
    return [r for r in timing.read_records(path) if r["event"] == "usage"]


def test_a_charge_writes_one_record_of_the_exact_shape(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    with recording(path), span("generation", generation=4):
        budget.charge_llm_usage(_state(), _stats(), candidate_id="g4-s2", source="mutation")

    start, record, _end = timing.read_records(path)
    assert list(record) == USAGE_KEYS
    assert record["candidate"] == "g4-s2"
    assert record["source"] == "mutation"
    assert [record[k] for k in timing.USAGE_TOKEN_FIELDS] == [100, 20, 30, 5, 7, 9]
    assert record["cost_usd"] == 0.25
    assert record["generation"] == 4
    assert record["parent_span_id"] == start["span_id"]
    assert record["run_id"] == start["run_id"]
    assert isinstance(record["wall_time"], float)
    assert isinstance(record["thread_id"], int)


def test_records_add_up_to_the_budget_counters(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    state = _state()
    with recording(path), span("generation", generation=1):
        budget.charge_llm_usage(state, _stats(1), candidate_id="g1-s1", source="mutation")
        budget.charge_llm_usage(state, _stats(2), candidate_id="g1-m1", source="merge_failed")

    records = _usage(path)
    assert [r["source"] for r in records] == ["mutation", "merge_failed"]
    assert sum(r["input_tokens"] for r in records) == state.budget.input_tokens
    assert sum(r["cost_usd"] for r in records) == state.budget.cost_usd


def test_seed_generation_outside_a_generation_counts_as_generation_0(
    tmp_path: Path,
) -> None:
    path = tmp_path / "t.jsonl"
    with recording(path), span("run"):
        budget.charge_llm_usage(_state(), _stats(), candidate_id="g0-s0", source="seed_generation")
        budget.charge_llm_usage(_state(), _stats(), candidate_id=None, source="other")

    seed, other = _usage(path)
    assert seed["generation"] == 0
    assert other["generation"] is None and other["candidate"] is None


def test_no_record_outside_a_recording_or_for_no_usage(tmp_path: Path) -> None:
    state = _state()
    budget.charge_llm_usage(state, _stats(), candidate_id="g1-s1", source="mutation")
    assert state.budget.input_tokens == 100  # charged, just not recorded
    assert list(tmp_path.iterdir()) == []

    path = tmp_path / "t.jsonl"
    with recording(path):
        budget.charge_llm_usage(state, None, candidate_id="g1-s1", source="mutation")
    assert path.read_text() == ""


def _trace_with_usage(path: Path) -> None:
    state = _state()
    with recording(path), span("run"):
        budget.charge_llm_usage(state, _stats(), candidate_id="g0-s0", source="seed_generation")
        timing.decision("g0-s0", accepted=True, reason="seed")
        with span("generation", generation=1), span("proposal", candidate_id="g1-s1"):
            with span("agent"):
                pass
            budget.charge_llm_usage(state, _stats(), candidate_id="g1-s1", source="mutation")
            timing.decision("g1-s1", accepted=False, reason="r")


def test_readers_ignore_usage_records(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    _trace_with_usage(path)
    records = timing.read_records(path)
    assert len(_usage(path)) == 2

    summary = timing.summarize(records)
    assert summary.candidates == {"accepted": 1, "rejected": 1}
    assert set(summary.work) == {"agent"}
    assert sorted(summary.generation_seconds) == [1]

    events = timing.to_perfetto(records)["traceEvents"]
    assert len([e for e in events if e["ph"] == "X"]) == sum(
        r["event"] == "end" for r in records
    )
    json.dumps(events)


def test_helix_timing_cli_reads_a_trace_with_usage_records(tmp_path: Path) -> None:
    helix_dir = tmp_path / ".helix"
    _trace_with_usage(helix_dir / timing.TIMING_FILE_NAME)
    out = tmp_path / "trace.json"

    result = CliRunner().invoke(
        cli, ["timing", "--dir", str(tmp_path), "--perfetto", str(out)]
    )

    assert result.exit_code == 0, result.output
    assert "1 accepted, 1 rejected" in result.output
    assert json.loads(out.read_text())["traceEvents"]
