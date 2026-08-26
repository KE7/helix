"""Tests for centralized HELIX budget/progress accounting."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest

from helix import budget
from helix.backends import BACKENDS
from helix.config import (
    AgentConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    SandboxConfig,
)
from helix.display import UsageStats
from helix.mutator import invoke_claude_code
from helix.state import BudgetState, EvolutionState
from helix.trace import TRACE, EventType


# Mapping from backend name to the executable that ``_build_backend_args``
# emits as args[0].  Used by parser-coverage tests to assert that
# ``invoke_claude_code`` actually dispatched to the requested backend CLI
# (rather than short-circuiting via the differential-testing override).
_BACKEND_EXECUTABLE = {
    "agy": "agy",
    "claude": "claude",
    "codex": "codex",
    "cursor": "cursor",
    "opencode": "opencode",
}


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


def test_batch_budget_guard_pins_the_documented_maximum_overshoot() -> None:
    """An eight-unit batch admitted at 19/20 may finish at 27, not 28."""
    state = make_state(19)

    guard = budget.begin_batch_budget_guard(
        state, max_evaluations=20, max_in_flight_evaluations=8
    )

    assert guard.maximum_overshoot == 7
    state.budget.evaluations = 27
    budget.enforce_batch_budget_guard(
        state, guard, actual_in_flight_evaluations=8
    )


def test_batch_budget_guard_rejects_excess_in_flight_work() -> None:
    state = make_state(19)
    guard = budget.begin_batch_budget_guard(
        state, max_evaluations=20, max_in_flight_evaluations=8
    )

    with pytest.raises(
        ValueError,
        match=r"in-flight.*actual 9, permitted 8",
    ):
        budget.enforce_batch_budget_guard(
            state, guard, actual_in_flight_evaluations=9
        )


def test_batch_budget_guard_rejects_excess_overshoot() -> None:
    state = make_state(19)
    guard = budget.begin_batch_budget_guard(
        state, max_evaluations=20, max_in_flight_evaluations=8
    )
    # Simulate an accounting defect: reported in-flight work remains within
    # the declared bound but the ledger advanced one extra unit.
    state.budget.evaluations = 28

    with pytest.raises(
        ValueError,
        match=r"overshoot.*actual 8, permitted 7",
    ):
        budget.enforce_batch_budget_guard(
            state, guard, actual_in_flight_evaluations=8
        )


def test_batch_budget_guard_rejects_a_negative_in_flight_bound() -> None:
    with pytest.raises(
        ValueError,
        match=r"actual -1, permitted >= 0",
    ):
        budget.begin_batch_budget_guard(
            make_state(), max_evaluations=20, max_in_flight_evaluations=-1
        )


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
    assert event.reason == "mutation_minibatch_gate"
    assert event.budget_delta == 2
    assert event.budget_evaluations == 5


@pytest.mark.parametrize(
    ("backend", "stdout", "expected"),
    [
        pytest.param(
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
            id="claude",
        ),
        pytest.param(
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
            id="codex",
        ),
        pytest.param(
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
            id="cursor",
        ),
        pytest.param(
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
            id="opencode",
        ),
        pytest.param(
            # agy's confirmed dispatch is the same single-JSON-object branch
            # claude uses (see _parse_backend_output); its real field names
            # are not yet confirmed (needs a real --print run, out of scope
            # for this migration), so this envelope is illustrative -- it
            # exercises the generic _normalise_usage_stats walk, not a
            # verified agy transcript shape.
            "agy",
            json.dumps(
                {
                    "session_id": "agy-session",
                    "usage": {"input_tokens": 16, "output_tokens": 12},
                    "total_cost_usd": 0.36,
                }
            ),
            (16, 12, 0.36),
            id="agy",
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
    # Use the real CompletedProcess type so attribute typos in the production
    # path (e.g. ``result.exit_code`` vs ``returncode``) surface as test
    # failures rather than being silently absorbed by ``MagicMock``.
    mock_run.return_value = subprocess.CompletedProcess(
        args=[_BACKEND_EXECUTABLE[backend]],
        returncode=0,
        stdout=stdout,
        stderr="",
    )
    state = make_state()

    _parsed, usage = invoke_claude_code(
        str(worktree),
        "read the prompt artifact",
        AgentConfig(backend=backend),
    )

    # Guard against future refactors that bypass ``_build_backend_args`` or
    # leave ``_MUTATOR_OVERRIDE`` set globally — the test must observe the
    # real backend dispatch, not a short-circuited override.
    assert mock_run.call_count == 1
    invoked_args = mock_run.call_args.args[0]
    assert invoked_args[0] == _BACKEND_EXECUTABLE[backend]

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


@pytest.mark.parametrize(
    ("backend", "stdout", "expected"),
    [
        pytest.param(
            "claude",
            json.dumps(
                {
                    "type": "result",
                    "session_id": "claude-session",
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "cache_creation_input_tokens": 13,
                        "cache_read_input_tokens": 17,
                        "cached_input_tokens": 19,
                    },
                    "reasoning_tokens": 5,
                    "total_cost_usd": 0.31,
                }
            ),
            {
                "input_tokens": 11,
                "output_tokens": 7,
                "cache_creation_input_tokens": 13,
                "cache_read_input_tokens": 17,
                "cached_input_tokens": 19,
                "reasoning_tokens": 5,
            },
            id="claude",
        ),
        pytest.param(
            "codex",
            "\n".join(
                [
                    '{"type":"session.started","session_id":"codex-session"}',
                    (
                        '{"type":"turn","usage":{"prompt_tokens":12,'
                        '"completion_tokens":8,"cached_input_tokens":21,'
                        '"reasoning_tokens":6,"total_cost_usd":0.32}}'
                    ),
                ]
            ),
            {
                "input_tokens": 12,
                "output_tokens": 8,
                "cached_input_tokens": 21,
                "reasoning_tokens": 6,
            },
            id="codex",
        ),
        pytest.param(
            "cursor",
            "\n".join(
                [
                    '{"type":"system","sessionId":"cursor-session"}',
                    (
                        '{"type":"assistant","usage":{"inputTokens":13,'
                        '"outputTokens":9,"cachedTokens":22,'
                        '"reasoningTokens":7,"costUsd":0.33}}'
                    ),
                ]
            ),
            {
                "input_tokens": 13,
                "output_tokens": 9,
                "cached_input_tokens": 22,
                "reasoning_tokens": 7,
            },
            id="cursor",
        ),
        pytest.param(
            "opencode",
            "\n".join(
                [
                    '{"type":"step_start","sessionID":"opencode-session"}',
                    (
                        '{"type":"step_finish","part":{"tokens":{"input":15,'
                        '"output":11,"cached":24,"thoughts":9},"cost":0.35}}'
                    ),
                ]
            ),
            {
                "input_tokens": 15,
                "output_tokens": 11,
                "cached_input_tokens": 24,
                "reasoning_tokens": 9,
            },
            id="opencode",
        ),
        pytest.param(
            # Illustrative envelope on agy's confirmed dispatch path (the
            # same single-JSON-object branch claude uses); real field names
            # are not yet confirmed, see the "agy" case above.
            "agy",
            json.dumps(
                {
                    "session_id": "agy-session",
                    "usage": {
                        "input_tokens": 16,
                        "output_tokens": 12,
                        "cached_input_tokens": 25,
                        "reasoning_tokens": 10,
                    },
                }
            ),
            {
                "input_tokens": 16,
                "output_tokens": 12,
                "cached_input_tokens": 25,
                "reasoning_tokens": 10,
            },
            id="agy",
        ),
    ],
)
def test_backend_usage_parsing_charges_extended_llm_budget(
    backend: str,
    stdout: str,
    expected: dict[str, int],
    tmp_path: Path,
    mocker,
) -> None:
    assert backend in BACKENDS
    worktree = tmp_path / backend
    worktree.mkdir()
    mock_run = mocker.patch("helix.mutator.subprocess.run")
    mock_run.return_value = subprocess.CompletedProcess(
        args=[_BACKEND_EXECUTABLE[backend]],
        returncode=0,
        stdout=stdout,
        stderr="",
    )
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

    for key, value in expected.items():
        assert getattr(state.budget, key) == value


def test_backend_usage_parsing_under_sandbox_charges_llm_budget(
    tmp_path: Path, mocker
) -> None:
    """Cover the ``run_sandboxed_command`` branch of ``invoke_claude_code``.

    The parametrized test above exercises the direct ``subprocess.run`` path.
    This variant flips ``SandboxConfig.enabled=True`` so the dispatch goes
    through ``run_sandboxed_command`` instead, ensuring sandboxed invocations
    still produce a usage payload that flows into ``charge_llm_usage``.
    """
    worktree = tmp_path / "claude"
    worktree.mkdir()
    stdout = json.dumps(
        {
            "type": "result",
            "session_id": "sandbox-session",
            "usage": {"input_tokens": 17, "output_tokens": 9},
            "total_cost_usd": 0.45,
        }
    )
    mock_sandboxed = mocker.patch("helix.mutator.run_sandboxed_command")
    mock_sandboxed.return_value = subprocess.CompletedProcess(
        args=["claude"], returncode=0, stdout=stdout, stderr=""
    )
    mocker.patch(
        "helix.mutator.resolve_sandbox_image",
        return_value="ghcr.io/example/image:latest",
    )
    # Belt-and-suspenders: the direct path must NOT be exercised when sandbox
    # is enabled, so leave ``subprocess.run`` patched to fail loudly.
    mock_run = mocker.patch(
        "helix.mutator.subprocess.run",
        side_effect=AssertionError(
            "subprocess.run must not be called when sandbox is enabled"
        ),
    )
    state = make_state()

    _parsed, usage = invoke_claude_code(
        str(worktree),
        "read the prompt artifact",
        AgentConfig(backend="claude"),
        sandbox=SandboxConfig(enabled=True),
    )

    assert mock_sandboxed.call_count == 1
    assert mock_run.call_count == 0

    budget.charge_llm_usage(
        state, usage, candidate_id="claude-candidate", source="claude"
    )

    assert state.budget.input_tokens == 17
    assert state.budget.output_tokens == 9
    assert state.budget.cost_usd == pytest.approx(0.45)


def test_backend_usage_parsing_extracts_cached_and_reasoning_tokens(
    tmp_path: Path, mocker
) -> None:
    """Lock in parser support for cached_input_tokens and reasoning_tokens."""
    worktree = tmp_path / "claude"
    worktree.mkdir()
    stdout = json.dumps(
        {
            "type": "result",
            "session_id": "claude-session",
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                # ``cacheReadInputTokens`` is the alias the parser actually
                # recognises (see ``_normalise_usage_stats`` aliases).
                "cacheReadInputTokens": 3,
            },
            "reasoning_tokens": 5,
            "total_cost_usd": 0.31,
        }
    )
    mock_run = mocker.patch("helix.mutator.subprocess.run")
    mock_run.return_value = subprocess.CompletedProcess(
        args=["claude"], returncode=0, stdout=stdout, stderr=""
    )

    _parsed, usage = invoke_claude_code(
        str(worktree),
        "read the prompt artifact",
        AgentConfig(backend="claude"),
    )

    assert usage.input_tokens == 11
    assert usage.output_tokens == 7
    assert usage.cached_input_tokens == 3
    assert usage.reasoning_tokens == 5
    assert usage.cost_usd == pytest.approx(0.31)
    state = make_state()
    budget.charge_llm_usage(state, usage, candidate_id="c", source="claude")
    assert state.budget.cached_input_tokens == 3
    assert state.budget.reasoning_tokens == 5


def test_charge_llm_usage_handles_none_and_partial_payloads() -> None:
    """Regression guard for the None short-circuit and partial UsageStats paths."""
    state = make_state()

    # ``None`` usage: short-circuit, no mutation, no trace event.
    budget.charge_llm_usage(state, None, candidate_id="c", source="x")
    assert state.budget.input_tokens == 0
    assert state.budget.output_tokens == 0
    assert state.budget.cost_usd == 0.0

    # Partial payload: only provided fields matter; absent fields default to 0.
    budget.charge_llm_usage(
        state,
        UsageStats(input_tokens=7),
        candidate_id="c",
        source="x",
    )
    assert state.budget.input_tokens == 7
    assert state.budget.output_tokens == 0
    assert state.budget.cached_input_tokens == 0
    assert state.budget.cache_creation_input_tokens == 0
    assert state.budget.cache_read_input_tokens == 0
    assert state.budget.reasoning_tokens == 0
    assert state.budget.cost_usd == 0.0


def test_charge_llm_usage_emits_budget_update_event() -> None:
    """Lock in the BUDGET_UPDATE trace contract for charge_llm_usage."""
    state = make_state()

    with TRACE.record() as events:
        budget.charge_llm_usage(
            state,
            UsageStats(
                input_tokens=11,
                output_tokens=7,
                cached_input_tokens=3,
                cache_creation_input_tokens=5,
                cache_read_input_tokens=7,
                reasoning_tokens=13,
                cost_usd=0.31,
            ),
            candidate_id="g1-s1",
            source="mutation",
        )

    assert len(events) == 1
    event = events[0]
    assert event.type is EventType.BUDGET_UPDATE
    assert event.candidate_id == "g1-s1"
    assert event.reason == "mutation"
    assert event.input_tokens_delta == 11
    assert event.output_tokens_delta == 7
    assert event.cost_usd_delta == pytest.approx(0.31)
    assert event.input_tokens == 11
    assert event.output_tokens == 7
    assert state.budget.cached_input_tokens == 3
    assert state.budget.cache_creation_input_tokens == 5
    assert state.budget.cache_read_input_tokens == 7
    assert state.budget.reasoning_tokens == 13
    assert event.cost_usd == pytest.approx(0.31)


def test_progress_counters_update_through_budget_api() -> None:
    state = make_state()

    budget.set_generation(state, 4)
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

    assert state.budget.input_tokens == 7
    assert state.budget.output_tokens == 11
    assert state.budget.cost_usd == 1.25
    assert events == []


def test_charge_llm_usage_zero_delta_still_emits_event() -> None:
    """A UsageStats with all-zero deltas should still emit one event.

    Pins current behavior so a future "skip on zero" optimization is a
    deliberate, test-visible decision rather than an accidental change.
    """
    state = make_state()
    usage = UsageStats(input_tokens=0, output_tokens=0, cost_usd=0.0)

    with TRACE.record() as events:
        budget.charge_llm_usage(state, usage, candidate_id="c", source="mutation")

    assert state.budget.input_tokens == 0
    assert state.budget.output_tokens == 0
    assert state.budget.cost_usd == 0.0
    assert len(events) == 1
    event = events[0]
    assert event.type is EventType.BUDGET_UPDATE
    assert event.reason == "mutation"
    assert event.input_tokens_delta == 0
    assert event.output_tokens_delta == 0
    assert event.cost_usd_delta == 0.0


def test_cached_charge_at_cap_does_not_overshoot_budget_exhausted() -> None:
    """A ``was_cached=True`` charge adds 0 units; the cap predicate must agree.

    Also pins the new "skip emit on units==0" behavior: a cached charge
    must not produce a ``BUDGET_UPDATE`` event, since no budget was
    actually consumed.
    """
    config = make_config(max_evaluations=100)

    # One unit shy of the cap: cached charge keeps us shy, not exhausted,
    # and emits no trace event.
    state = make_state(99)
    with TRACE.record() as events:
        charged = budget.charge_evaluation(state, was_cached=True, source="test")
    assert charged == 0
    assert state.budget.evaluations == 99
    assert events == []
    assert budget.budget_exhausted(state, config) is False

    # Exactly at the cap: cached charge keeps us at the cap, exhausted.
    state = make_state(100)
    with TRACE.record() as events:
        charged = budget.charge_evaluation(state, was_cached=True, source="test")
    assert charged == 0
    assert state.budget.evaluations == 100
    assert events == []
    assert budget.budget_exhausted(state, config) is True

    # Zero-example minibatch is the same units==0 short-circuit.
    state = make_state(50)
    with TRACE.record() as events:
        charged = budget.charge_evaluation(
            state, num_actual_examples=0, source="empty_minibatch"
        )
    assert charged == 0
    assert state.budget.evaluations == 50
    assert events == []


def test_record_discovery_budget_warns_on_duplicate_id() -> None:
    """Duplicate stamps overwrite but emit a warning so bugs are visible."""
    import warnings as _warnings

    state = make_state(5)
    budget.record_discovery_budget(state, "g1-s1")
    assert state.num_metric_calls_by_discovery == {"g1-s1": 5}

    state.budget.evaluations = 9
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        budget.record_discovery_budget(state, "g1-s1")

    assert len(caught) == 1
    assert "g1-s1" in str(caught[0].message)
    assert state.num_metric_calls_by_discovery == {"g1-s1": 9}


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
