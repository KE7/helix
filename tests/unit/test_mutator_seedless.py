"""Unit tests for seedless-mode additions to helix.mutator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest

from helix.config import ClaudeConfig, EvaluatorConfig, HelixConfig, SeedlessConfig
from helix.exceptions import MutationError
from helix.mutator import (
    SEEDLESS_INIT_PROMPT_TEMPLATE,
    build_seed_generation_prompt,
    generate_seed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(objective: str = "Maximise test coverage") -> HelixConfig:
    return HelixConfig(
        objective=objective,
        evaluator=EvaluatorConfig(command="pytest -q"),
        seedless=SeedlessConfig(enabled=True),
    )


# ---------------------------------------------------------------------------
# build_seed_generation_prompt
# ---------------------------------------------------------------------------


class TestBuildSeedGenerationPrompt:
    def test_contains_objective(self):
        """The prompt must contain the objective text."""
        objective = "Generate a Python solver for the N-queens problem"
        prompt = build_seed_generation_prompt(objective=objective)
        assert objective in prompt

    def test_contains_background_when_provided(self):
        """Background text must appear in the prompt when given."""
        background = "Use only the standard library, no external packages."
        prompt = build_seed_generation_prompt(
            objective="Write a sorting algorithm",
            background=background,
        )
        assert background in prompt

    def test_no_background_section_when_absent(self):
        """When background=None, the prompt should not contain the background header."""
        prompt = build_seed_generation_prompt(
            objective="Write a sorting algorithm",
            background=None,
        )
        assert "Domain Context" not in prompt

    def test_contains_evaluator_cmd_when_provided(self):
        """evaluator_cmd should appear in the prompt when given."""
        evaluator_cmd = "python evaluate.py --split train"
        prompt = build_seed_generation_prompt(
            objective="Generate a solver",
            evaluator_cmd=evaluator_cmd,
        )
        assert evaluator_cmd in prompt

    def test_no_evaluator_section_when_absent(self):
        """When evaluator_cmd=None, no evaluator section should appear."""
        prompt = build_seed_generation_prompt(
            objective="Generate a solver",
            evaluator_cmd=None,
        )
        assert "Evaluator" not in prompt

    def test_completion_signal_in_prompt(self):
        """The completion signal string must appear in the prompt."""
        prompt = build_seed_generation_prompt(objective="Test objective")
        assert "[SEED GENERATION COMPLETE]" in prompt

    def test_returns_string(self):
        """build_seed_generation_prompt must return a str."""
        result = build_seed_generation_prompt(objective="Test")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# generate_seed
# ---------------------------------------------------------------------------


class TestGenerateSeed:
    def test_calls_invoke_claude_code_exactly_once(self):
        """generate_seed must call invoke_claude_code exactly once (no retry loop)."""
        config = make_config()
        mock_result = {"result": "ok", "subtype": "success"}

        with patch("helix.mutator.invoke_claude_code", return_value=mock_result) as mock_invoke:
            generate_seed("/tmp/fake-wt", "some prompt", config)

        mock_invoke.assert_called_once()

    def test_passes_worktree_path_to_invoke(self):
        """generate_seed must pass the worktree_path as first arg to invoke_claude_code."""
        config = make_config()
        worktree_path = "/tmp/my-seed-worktree"

        with patch("helix.mutator.invoke_claude_code", return_value={}) as mock_invoke:
            generate_seed(worktree_path, "prompt text", config)

        args, kwargs = mock_invoke.call_args
        assert args[0] == worktree_path

    def test_passes_prompt_to_invoke(self):
        """generate_seed must pass the prompt as second arg to invoke_claude_code."""
        config = make_config()
        prompt = "my seed generation prompt"

        with patch("helix.mutator.invoke_claude_code", return_value={}) as mock_invoke:
            generate_seed("/tmp/wt", prompt, config)

        args, kwargs = mock_invoke.call_args
        assert args[1] == prompt

    def test_passes_claude_config_to_invoke(self):
        """generate_seed must pass config.claude as third arg to invoke_claude_code."""
        config = make_config()

        with patch("helix.mutator.invoke_claude_code", return_value={}) as mock_invoke:
            generate_seed("/tmp/wt", "prompt", config)

        args, kwargs = mock_invoke.call_args
        assert args[2] is config.claude

    def test_raises_on_failure_immediately(self):
        """If invoke_claude_code raises MutationError, generate_seed propagates it immediately."""
        config = make_config()
        exc = MutationError(
            "Claude Code exited with code 1",
            operation="seed generation",
            exit_code=1,
        )

        with patch("helix.mutator.invoke_claude_code", side_effect=exc):
            with pytest.raises(MutationError) as exc_info:
                generate_seed("/tmp/wt", "prompt", config)

        # Must be the exact same exception (no wrapping, no retry)
        assert exc_info.value is exc

    def test_no_retry_on_failure(self):
        """invoke_claude_code must only be called ONCE even when it fails (no retry)."""
        config = make_config()
        exc = MutationError("fail", exit_code=1)
        call_count = 0

        def failing_invoke(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            raise exc

        with patch("helix.mutator.invoke_claude_code", side_effect=failing_invoke):
            with pytest.raises(MutationError):
                generate_seed("/tmp/wt", "prompt", config)

        assert call_count == 1, f"Expected exactly 1 call, got {call_count} (retry loop detected!)"

    def test_returns_none_on_success(self):
        """generate_seed returns None on success (side-effect: files written to worktree)."""
        config = make_config()

        with patch("helix.mutator.invoke_claude_code", return_value={"subtype": "success"}):
            result = generate_seed("/tmp/wt", "prompt", config)

        assert result is None
