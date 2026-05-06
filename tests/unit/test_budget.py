"""Tests for centralized HELIX budget/progress accounting."""

from __future__ import annotations

import re
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helix.backends import BACKENDS
from helix import budget
from helix.config import AgentConfig, EvolutionConfig, EvaluatorConfig, HelixConfig
from helix.mutator import invoke_claude_code
from helix.state import BudgetState, EvolutionState
from helix.trace import TRACE, EventType


def make_state(evaluations: int = 0) -> EvolutionState:
    return EvolutionState(
        generation=0,
        frontier=[],
        instance_scores={},
        budget=BudgetState(evaluations=evaluations),
        config_hash="test",
    )


def make_config(max_evaluations: int) -> HelixConfig:
    return HelixConfig(
        objective="Improve the code",
        evaluator=EvaluatorConfig(command="pytest -q"),
        evolution=EvolutionConfig(max_evaluations=max_evaluations),
    )


def test_evaluation_units_preserve_existing_semantics() -> None:
    assert budget.evaluation_budget_units(num_actual_examples=5) == 5
    assert budget.evaluation_budget_units(num_actual_examples=0) == 0
    assert budget.evaluation_budget_units(was_cached=True) == 0
    assert budget.evaluation_budget_units() == 1


def test_budget_exhausted_preserves_evaluation_cap_semantics() -> None:
    assert budget.budget_exhausted(make_state(10), make_config(100)) is False
    assert budget.budget_exhausted(make_state(200), make_config(200)) is True
    assert budget.budget_exhausted(make_state(250), make_config(200)) is True
    assert budget.budget_exhausted(make_state(250), make_config(-1)) is False


def test_charge_evaluation_updates_counter_and_emits_event() -> None:
    state = make_state(3)

    with TRACE.record() as events:
        charged = budget.charge_evaluation(
            state,
            num_actual_examples=2,
            candidate_id="g1-s1",
            split="train",
            source="mutation_minibatch_gate",
        )

    assert charged == 2
    assert state.budget.evaluations == 5
    assert len(events) == 1
    event = events[0]
    assert event.type is EventType.BUDGET_UPDATE
    assert event.candidate_id == "g1-s1"
    assert event.split == "train"
    assert event.decision == "mutation_minibatch_gate"
    assert event.budget_delta == 2
    assert event.budget_evaluations == 5


@pytest.mark.parametrize(
    ("backend", "stdout", "expected"),
    [
        (
            "claude",
            json.dumps(
                {
                    "type": "result",
                    "session_id": "claude-session",
                    "usage": {"input_tokens": 11, "output_tokens": 7},
                    "total_cost_usd": 0.31,
                }
            ),
            (11, 7, 0.31),
        ),
        (
            "codex",
            "\n".join(
                [
                    '{"type":"session.started","session_id":"codex-session"}',
                    (
                        '{"type":"turn","usage":{"prompt_tokens":12,'
                        '"completion_tokens":8,"total_cost_usd":0.32}}'
                    ),
                ]
            ),
            (12, 8, 0.32),
        ),
        (
            "cursor",
            "\n".join(
                [
                    '{"type":"system","sessionId":"cursor-session"}',
                    (
                        '{"type":"assistant","usage":{"inputTokens":13,'
                        '"outputTokens":9,"costUsd":0.33}}'
                    ),
                ]
            ),
            (13, 9, 0.33),
        ),
        (
            "gemini",
            "\n".join(
                [
                    "MCP advisory preamble tolerated by the Gemini parser.",
                    '{"type":"init","session_id":"gemini-session"}',
                    (
                        '{"type":"result","usageMetadata":{"prompt_tokens":14,'
                        '"completion_tokens":10},"cost":0.34}'
                    ),
                ]
            ),
            (14, 10, 0.34),
        ),
        (
            "opencode",
            "\n".join(
                [
                    '{"type":"step_start","sessionID":"opencode-session"}',
                    (
                        '{"type":"step_finish","part":{"tokens":{"input":15,'
                        '"output":11},"cost":0.35}}'
                    ),
                ]
            ),
            (15, 11, 0.35),
        ),
    ],
)
def test_backend_usage_parsing_charges_llm_budget(
    backend: str,
    stdout: str,
    expected: tuple[int, int, float],
    tmp_path: Path,
    mocker,
) -> None:
    assert backend in BACKENDS
    worktree = tmp_path / backend
    worktree.mkdir()
    mock_run = mocker.patch("helix.mutator.subprocess.run")
    mock_run.return_value = MagicMock(stdout=stdout, stderr="", returncode=0)
    state = make_state()

    _parsed, usage = invoke_claude_code(
        str(worktree),
        "read the prompt artifact",
        AgentConfig(backend=backend),
    )
    budget.charge_llm_usage(
        state,
        usage,
        candidate_id=f"{backend}-candidate",
        source=backend,
    )

    expected_input, expected_output, expected_cost = expected
    assert state.budget.input_tokens == expected_input
    assert state.budget.output_tokens == expected_output
    assert state.budget.cost_usd == pytest.approx(expected_cost)


def test_progress_counters_update_through_budget_api() -> None:
    state = make_state()

    budget.start_generation(state, 4)
    first_i = budget.advance_proposal_counter(state, source="iteration")
    mutation_id = budget.next_mutation_id(state, 4)
    merge_id = budget.next_merge_id(state, 4)
    budget.record_merge_invocation(state)
    state.budget.evaluations = 9
    budget.record_discovery_budget(state, mutation_id)

    assert state.generation == 4
    assert first_i == 0
    assert state.i == 0
    assert mutation_id == "g4-s1"
    assert state.mutation_counter == 1
    assert merge_id == "g4-m1"
    assert state.merge_counter == 1
    assert state.total_merge_invocations == 1
    assert state.num_metric_calls_by_discovery == {"g4-s1": 9}


def test_charge_llm_usage_none_is_silent_noop() -> None:
    """Passing ``usage=None`` must not touch counters or emit a trace event."""
    state = make_state()
    state.budget.input_tokens = 7
    state.budget.output_tokens = 11
    state.budget.cost_usd = 1.25

    with TRACE.record() as events:
        budget.charge_llm_usage(state, None, candidate_id="c", source="s")
        # The empty-dict case is the same falsy short-circuit; pin it too.
        budget.charge_llm_usage(state, {}, candidate_id="c", source="s")

    assert state.budget.input_tokens == 7
    assert state.budget.output_tokens == 11
    assert state.budget.cost_usd == 1.25
    assert events == []


def test_charge_llm_usage_zero_delta_still_emits_event() -> None:
    """A non-empty usage dict with all-zero deltas should still emit one event.

    Pins current behavior so a future "skip on zero" optimization is a
    deliberate, test-visible decision rather than an accidental change.
    """
    state = make_state()
    usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    with TRACE.record() as events:
        budget.charge_llm_usage(state, usage, candidate_id="c", source="mutation")

    assert state.budget.input_tokens == 0
    assert state.budget.output_tokens == 0
    assert state.budget.cost_usd == 0.0
    assert len(events) == 1
    event = events[0]
    assert event.type is EventType.BUDGET_UPDATE
    assert event.decision == "mutation"
    assert event.input_tokens_delta == 0
    assert event.output_tokens_delta == 0
    assert event.cost_usd_delta == 0.0


def test_cached_charge_at_cap_does_not_overshoot_budget_exhausted() -> None:
    """A ``was_cached=True`` charge adds 0 units; the cap predicate must agree."""
    config = make_config(max_evaluations=100)

    # One unit shy of the cap: cached charge keeps us shy, not exhausted.
    state = make_state(99)
    charged = budget.charge_evaluation(state, was_cached=True, source="test")
    assert charged == 0
    assert state.budget.evaluations == 99
    assert budget.budget_exhausted(state, config) is False

    # Exactly at the cap: cached charge keeps us at the cap, exhausted.
    state = make_state(100)
    charged = budget.charge_evaluation(state, was_cached=True, source="test")
    assert charged == 0
    assert state.budget.evaluations == 100
    assert budget.budget_exhausted(state, config) is True


def test_advance_proposal_counter_increments_monotonically() -> None:
    """Pin GEPA-parity initial value (-1) and per-call increment semantics."""
    state = make_state()
    # ``EvolutionState.i`` defaults to -1 (GEPA core/state.py parity); the
    # first ``advance`` should yield 0, the second 1, and so on.
    assert state.i == -1
    assert budget.advance_proposal_counter(state, source="iteration") == 0
    assert state.i == 0
    assert budget.advance_proposal_counter(state, source="iteration") == 1
    assert state.i == 1
    assert budget.advance_proposal_counter(state, source="parallel_proposal") == 2
    assert state.i == 2


def test_evolution_counter_mutations_route_through_budget_api() -> None:
    evolution_py = Path(__file__).parents[2] / "src" / "helix" / "evolution.py"
    source = evolution_py.read_text()

    forbidden_patterns = [
        r"state\.budget\.(evaluations|input_tokens|output_tokens|cost_usd)\s*\+=",
        r"state\.generation\s*=",
        r"state\.i\s*\+=",
        r"state\.(mutation_counter|merge_counter|total_merge_invocations)\s*\+=",
        r"state\.num_metric_calls_by_discovery\[[^\]]+\]\s*=",
    ]
    for pattern in forbidden_patterns:
        assert re.search(pattern, source) is None
