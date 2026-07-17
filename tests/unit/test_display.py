"""Unit tests for helix.display — UsageStats dataclass and helpers."""

from __future__ import annotations

import json

import pytest
from rich.console import Console

from helix.display import UsageStats, render_proposal_batch_table
from helix.population import Candidate, CandidateSummary, EvalResult
from helix.state import ProposalBatchRecord, ProposalTaskRecord
from helix.trace import TRACE, EventType


def _terminal_batch() -> ProposalBatchRecord:
    batch_id = "g2-b0"
    tasks = [
        ProposalTaskRecord(
            batch_id=batch_id,
            p=2,
            n=2,
            task_index=index,
            parent_group=index // 2,
            mutation_index=index % 2,
            parent_id="parent-a" if index < 2 else "parent-b",
            child_id=f"child-{index}",
        )
        for index in range(4)
    ]
    terminal = [
        ("applied", "selected", "not_required", 0.25, True),
        ("rejected", "not_selected", "removed", -0.1, False),
        ("failed", "not_applicable", "missing", None, False),
        ("interrupted", "not_applicable", "failed", None, False),
    ]
    for task, (status, selection, cleanup, delta, applied) in zip(
        tasks, terminal, strict=True
    ):
        task.status = status  # type: ignore[assignment]
        task.selection = selection  # type: ignore[assignment]
        task.cleanup = cleanup  # type: ignore[assignment]
        task.score_delta = delta
        task.applied = applied
    return ProposalBatchRecord(
        batch_id=batch_id,
        generation=2,
        p=2,
        n=2,
        tasks=tasks,
        phase="interrupted",
    )


# ---------------------------------------------------------------------------
# UsageStats.add
# ---------------------------------------------------------------------------


class TestUsageStatsAdd:
    """Verify UsageStats.add accumulates all fields from another UsageStats."""

    def test_add_accumulates_integer_fields(self) -> None:
        stats = UsageStats()
        stats.add(
            UsageStats(
                input_tokens=11,
                output_tokens=7,
                cached_input_tokens=3,
                cache_creation_input_tokens=5,
                cache_read_input_tokens=7,
                reasoning_tokens=13,
                num_turns=2,
                tool_event_count=1,
            )
        )
        assert stats.input_tokens == 11
        assert stats.output_tokens == 7
        assert stats.cached_input_tokens == 3
        assert stats.cache_creation_input_tokens == 5
        assert stats.cache_read_input_tokens == 7
        assert stats.reasoning_tokens == 13
        assert stats.num_turns == 2
        assert stats.tool_event_count == 1

    def test_add_accumulates_cost_usd(self) -> None:
        stats = UsageStats(cost_usd=1.0)
        stats.add(UsageStats(cost_usd=0.31))
        assert stats.cost_usd == pytest.approx(1.31)

    def test_add_accumulates_tool_names(self) -> None:
        stats = UsageStats(tool_names=["Bash"])
        stats.add(UsageStats(tool_names=["Read", "Edit"]))
        assert stats.tool_names == ["Bash", "Read", "Edit"]

    def test_add_twice_doubles_values(self) -> None:
        base = UsageStats(input_tokens=10, output_tokens=5, cost_usd=0.1)
        base.add(UsageStats(input_tokens=10, output_tokens=5, cost_usd=0.1))
        assert base.input_tokens == 20
        assert base.output_tokens == 10
        assert base.cost_usd == pytest.approx(0.2)

    def test_add_empty_stats_is_noop(self) -> None:
        stats = UsageStats(input_tokens=42, cost_usd=3.14)
        stats.add(UsageStats())
        assert stats.input_tokens == 42
        assert stats.cost_usd == pytest.approx(3.14)

    def test_add_does_not_share_tool_names_list(self) -> None:
        """add must not alias the other.tool_names list into self."""
        other = UsageStats(tool_names=["Bash"])
        stats = UsageStats()
        stats.add(other)
        # Mutating other afterward must not affect stats
        other.tool_names.append("EXTRA")
        assert "EXTRA" not in stats.tool_names


# ---------------------------------------------------------------------------
# UsageStats.to_dict / from_dict round-trip
# ---------------------------------------------------------------------------


class TestUsageStatsSerialization:
    """Verify to_dict/from_dict round-trips all 11 fields losslessly."""

    def _full_stats(self) -> UsageStats:
        return UsageStats(
            input_tokens=100,
            output_tokens=50,
            cached_input_tokens=20,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=30,
            reasoning_tokens=5,
            num_turns=3,
            tool_event_count=7,
            tool_names=["Bash", "Read", "Edit"],
            cost_usd=1.23,
            session_id="sess-abc123",
        )

    def test_to_dict_contains_all_11_fields(self) -> None:
        d = self._full_stats().to_dict()
        assert set(d.keys()) == {
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
            "reasoning_tokens",
            "num_turns",
            "tool_event_count",
            "tool_names",
            "cost_usd",
            "session_id",
        }

    def test_round_trip_all_fields(self) -> None:
        original = self._full_stats()
        restored = UsageStats.from_dict(original.to_dict())
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.cached_input_tokens == original.cached_input_tokens
        assert (
            restored.cache_creation_input_tokens == original.cache_creation_input_tokens
        )
        assert restored.cache_read_input_tokens == original.cache_read_input_tokens
        assert restored.reasoning_tokens == original.reasoning_tokens
        assert restored.num_turns == original.num_turns
        assert restored.tool_event_count == original.tool_event_count
        assert restored.tool_names == original.tool_names
        assert restored.cost_usd == pytest.approx(original.cost_usd)
        assert restored.session_id == original.session_id

    def test_to_dict_tool_names_is_fresh_copy(self) -> None:
        """to_dict must return a copy of tool_names, not the live reference."""
        stats = UsageStats(tool_names=["Bash"])
        d = stats.to_dict()
        d["tool_names"].append("MUTATED")
        assert stats.tool_names == ["Bash"], "to_dict must not alias the live list"

    def test_round_trip_is_json_serializable(self) -> None:
        d = self._full_stats().to_dict()
        # Must not raise
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert decoded["input_tokens"] == 100
        assert decoded["session_id"] == "sess-abc123"

    def test_from_dict_tolerates_missing_keys(self) -> None:
        """from_dict must use dataclass defaults for absent keys."""
        stats = UsageStats.from_dict({})
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0
        assert stats.cached_input_tokens == 0
        assert stats.cache_creation_input_tokens == 0
        assert stats.cache_read_input_tokens == 0
        assert stats.reasoning_tokens == 0
        assert stats.num_turns == 0
        assert stats.tool_event_count == 0
        assert stats.tool_names == []
        assert stats.cost_usd == 0.0
        assert stats.session_id is None

    def test_from_dict_tolerates_non_list_tool_names(self) -> None:
        """from_dict must fall back to [] when tool_names is not a list."""
        stats = UsageStats.from_dict({"tool_names": "Bash"})
        assert stats.tool_names == []
        stats2 = UsageStats.from_dict({"tool_names": None})
        assert stats2.tool_names == []
        stats3 = UsageStats.from_dict({"tool_names": 42})
        assert stats3.tool_names == []

    def test_from_dict_coerces_numeric_types(self) -> None:
        """from_dict must coerce int/float defensively."""
        stats = UsageStats.from_dict(
            {
                "input_tokens": "11",  # str → int
                "cost_usd": "0.31",  # str → float
                "output_tokens": 7.9,  # float → int (truncates)
            }
        )
        assert stats.input_tokens == 11
        assert stats.cost_usd == pytest.approx(0.31)
        assert stats.output_tokens == 7

    def test_from_dict_session_id_none_when_absent(self) -> None:
        stats = UsageStats.from_dict({"input_tokens": 5})
        assert stats.session_id is None

    def test_from_dict_session_id_preserved(self) -> None:
        stats = UsageStats.from_dict({"session_id": "abc"})
        assert stats.session_id == "abc"


# ---------------------------------------------------------------------------
# Regression: CandidateSummary.to_dict() → json.dumps (critical risk #2)
# ---------------------------------------------------------------------------


class TestCandidateSummaryJsonSerialization:
    """Regression guard for the TypeError that would occur pre-flip.

    Pre-flip: Candidate.usage was dict[str, Any], passed directly as
    "usage": self.candidate.usage in CandidateSummary.to_dict().
    json.dumps accepted it transparently.

    Post-flip: Candidate.usage is UsageStats; CandidateSummary.to_dict()
    calls .to_dict() on it, so json.dumps sees a plain dict again.
    This test would TypeError on the pre-flip code if Candidate.usage
    were a UsageStats without the .to_dict() adapter.
    """

    def _make_candidate_summary(self) -> CandidateSummary:
        candidate = Candidate(
            id="g1-s1",
            worktree_path="/tmp/helix/g1-s1",
            branch_name="helix/g1-s1",
            generation=1,
            parent_id="g0-s0",
            parent_ids=["g0-s0"],
            operation="mutate",
            usage=UsageStats(
                input_tokens=42,
                output_tokens=17,
                cost_usd=0.07,
                session_id="sess-xyz",
            ),
        )
        eval_result = EvalResult(
            candidate_id="g1-s1",
            scores={"train": 0.8},
            instance_scores={"i1": 0.8},
            asi={},
        )
        return CandidateSummary(
            candidate=candidate,
            aggregate_score=eval_result.aggregate_score(),
            sum_score=eval_result.sum_score(),
            scores=eval_result.scores,
            instance_scores=eval_result.instance_scores,
            objective_scores=None,
            parents=["g0-s0"],
            operation="mutate",
            generation=1,
        )

    def test_candidate_summary_to_dict_is_json_serializable(self) -> None:
        """CandidateSummary.to_dict() must not raise TypeError at json.dumps."""
        summary = self._make_candidate_summary()
        d = summary.to_dict()
        # Pre-flip this line would TypeError because UsageStats is not JSON-able
        encoded = json.dumps(d)
        decoded = json.loads(encoded)
        assert decoded["id"] == "g1-s1"
        assert decoded["usage"]["input_tokens"] == 42
        assert decoded["usage"]["cost_usd"] == pytest.approx(0.07)
        assert decoded["usage"]["session_id"] == "sess-xyz"

    def test_usage_field_in_summary_is_serialized_dict(self) -> None:
        """The 'usage' key in CandidateSummary.to_dict() must be a plain dict."""
        summary = self._make_candidate_summary()
        d = summary.to_dict()
        assert isinstance(d["usage"], dict), (
            "CandidateSummary.to_dict() must call .to_dict() on usage; "
            f"got {type(d['usage'])!r}"
        )


# ---------------------------------------------------------------------------
# P-by-N proposal observability
# ---------------------------------------------------------------------------


def test_batch_table_renders_every_terminal_slot_in_parent_major_order() -> None:
    batch = _terminal_batch()
    table = render_proposal_batch_table(batch)
    render_console = Console(record=True, width=140)
    render_console.print(table)
    rendered = render_console.export_text()

    assert "2×2, 4 slots" in rendered
    positions = [rendered.index(f"child-{index}") for index in range(4)]
    assert positions == sorted(positions)
    for status in ("applied", "rejected", "failed", "interrupted"):
        assert status in rendered
    for cleanup in ("not_required", "removed", "missing"):
        assert cleanup in rendered


def test_completed_batch_table_rejects_nonterminal_slot() -> None:
    batch = _terminal_batch()
    batch.tasks[2].status = "running"
    batch.tasks[2].cleanup = "pending"
    with pytest.raises(ValueError, match="nonterminal slots: 2"):
        render_proposal_batch_table(batch)


def test_trace_emits_exactly_one_terminal_event_per_batch_slot() -> None:
    batch = _terminal_batch()
    with TRACE.record() as events:
        TRACE.emit_proposal_batch_terminal(batch)

    terminal_events = [
        event for event in events if event.type is EventType.PROPOSAL_TASK_TERMINAL
    ]
    assert len(terminal_events) == batch.p * batch.n
    assert [event.task_index for event in terminal_events] == [0, 1, 2, 3]
    assert [event.child_id for event in terminal_events] == [
        "child-0",
        "child-1",
        "child-2",
        "child-3",
    ]
    assert all(event.status is not None for event in terminal_events)
    assert events[0].type is EventType.PROPOSAL_BATCH_START
    assert events[-1].type is EventType.PROPOSAL_BATCH_END


def test_trace_rejects_partial_batch_instead_of_omitting_slot() -> None:
    batch = _terminal_batch()
    batch.tasks[3].status = "evaluated"
    batch.tasks[3].cleanup = "pending"
    with TRACE.record():
        with pytest.raises(ValueError, match="nonterminal slots: 3"):
            TRACE.emit_proposal_batch_terminal(batch)
