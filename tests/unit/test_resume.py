"""Unit tests for crash recovery and resume functionality.

Covers:
- test_state_saved_before_crash: state.json is written when an exception occurs mid-generation
- test_rate_limit_triggers_clean_exit: RateLimitError raised from invoke_claude_code on rate-limit errors
- test_resume_skips_missing_worktrees: missing worktrees are dropped from frontier with a warning
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from helix.exceptions import RateLimitError
from helix.mutator import MutationError, invoke_claude_code, _looks_like_rate_limit
from helix.state import BudgetState, EvolutionState, load_state, save_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_state(
    generation: int = 2,
    frontier: list[str] | None = None,
) -> EvolutionState:
    return EvolutionState(
        generation=generation,
        frontier=frontier if frontier is not None else ["g0-s0", "g1-m1"],
        instance_scores={},
        budget=BudgetState(evaluations=5),
        config_hash="abc123",
    )


# ---------------------------------------------------------------------------
# test_state_saved_before_crash
# ---------------------------------------------------------------------------


def test_state_saved_before_crash(tmp_path: Path) -> None:
    """State.json must be written before an exception propagates out of run_evolution."""
    from helix.config import (
        ClaudeConfig,
        DatasetConfig,
        EvolutionConfig,
        EvaluatorConfig,
        HelixConfig,
        WorktreeConfig,
    )
    from helix.evolution import run_evolution
    from helix.population import Candidate, EvalResult

    project_root = tmp_path / "proj"
    project_root.mkdir()
    base_dir = project_root / ".helix"

    # Minimal config with 3 generations max
    config = HelixConfig(
        objective="test",
        evaluator=EvaluatorConfig(command="echo 1"),
        evolution=EvolutionConfig(max_generations=3),
        claude=ClaudeConfig(),
        dataset=DatasetConfig(),
        worktree=WorktreeConfig(),
    )

    seed_candidate = Candidate(
        id="g0-s0",
        worktree_path=str(tmp_path / "wt" / "g0-s0"),
        branch_name="helix/g0-s0",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )
    seed_result = EvalResult(
        candidate_id="g0-s0",
        scores={"score": 0.5},
        instance_scores={},
        asi={},
    )

    boom_error = RuntimeError("simulated crash mid-generation")

    # Patch create_seed_worktree and run_evaluator so we control when to crash.
    call_count = [0]

    def fake_run_evaluator(candidate, cfg, split=None, instances=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            # First call = seed evaluation — succeed
            return seed_result
        # Second call (gen 1 train eval) — crash
        raise boom_error

    # Make HelixProgress a no-op context manager
    mock_ctx = MagicMock()
    mock_ctx.__enter__ = MagicMock(return_value=mock_ctx)
    mock_ctx.__exit__ = MagicMock(return_value=False)

    with (
        patch("helix.evolution.create_seed_worktree", return_value=seed_candidate),
        patch("helix.evolution.run_evaluator", side_effect=fake_run_evaluator),
        patch("helix.evolution.HelixProgress", return_value=mock_ctx),
    ):
        with pytest.raises(RuntimeError, match="simulated crash"):
            run_evolution(config, project_root, base_dir)

    # State file must have been written even though an exception propagated
    state_file = project_root / ".helix" / "state.json"
    assert state_file.exists(), "state.json was not written before crash propagated"

    loaded = load_state(project_root)
    assert loaded is not None, "Could not load state after crash"
    # The seed was evaluated successfully so it should be in the frontier
    assert "g0-s0" in loaded.frontier, "Seed should be in frontier after successful seed eval"


# ---------------------------------------------------------------------------
# test_rate_limit_triggers_clean_exit
# ---------------------------------------------------------------------------


def test_looks_like_rate_limit_keywords() -> None:
    """_looks_like_rate_limit should detect all expected keywords."""
    assert _looks_like_rate_limit("You have hit a rate limit")
    assert _looks_like_rate_limit("Claude is overloaded right now")
    assert _looks_like_rate_limit("HTTP 529")
    assert _looks_like_rate_limit("You have exceeded your usage limit")
    assert _looks_like_rate_limit("You have used your extra usage")
    assert not _looks_like_rate_limit("Everything is fine")


def test_rate_limit_raises_rate_limit_error(tmp_path: Path) -> None:
    """invoke_claude_code raises RateLimitError when subprocess returns rate-limit stderr."""
    from helix.config import ClaudeConfig

    config = ClaudeConfig()

    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stderr = "Error: You have hit a rate limit. Please try again later."
    fake_result.stdout = ""

    with patch("subprocess.run", return_value=fake_result):
        with pytest.raises(RateLimitError):
            invoke_claude_code(str(tmp_path), "test prompt", config)


def test_rate_limit_not_raised_for_normal_errors(tmp_path: Path) -> None:
    """invoke_claude_code raises MutationError (not RateLimitError) for ordinary failures."""
    from helix.config import ClaudeConfig

    config = ClaudeConfig()

    fake_result = MagicMock()
    fake_result.returncode = 1
    fake_result.stderr = "Error: file not found"
    fake_result.stdout = ""

    with patch("subprocess.run", return_value=fake_result):
        with pytest.raises(MutationError):
            invoke_claude_code(str(tmp_path), "test prompt", config)


def test_rate_limit_in_json_result(tmp_path: Path) -> None:
    """invoke_claude_code raises RateLimitError when JSON result contains overload error."""
    from helix.config import ClaudeConfig

    config = ClaudeConfig()

    json_payload = json.dumps({
        "is_error": True,
        "error": "Claude is overloaded",
        "result": "",
    })
    fake_result = MagicMock()
    fake_result.returncode = 0
    fake_result.stderr = ""
    fake_result.stdout = json_payload

    with patch("subprocess.run", return_value=fake_result):
        with pytest.raises(RateLimitError):
            invoke_claude_code(str(tmp_path), "test prompt", config)


# ---------------------------------------------------------------------------
# test_resume_skips_missing_worktrees
# ---------------------------------------------------------------------------


def test_resume_skips_missing_worktrees(tmp_path: Path, capsys) -> None:
    """The resume command drops frontier candidates whose worktrees no longer exist."""
    from click.testing import CliRunner
    from helix.cli import cli

    project_root = tmp_path / "proj"
    project_root.mkdir()
    helix_dir = project_root / ".helix"
    helix_dir.mkdir()
    (helix_dir / "worktrees").mkdir()

    # Create a helix.toml so the CLI can load config
    (project_root / "helix.toml").write_text(
        'objective = "test"\n\n[evaluator]\ncommand = "echo 1"\n'
    )

    # Create state with two candidates: g0-s0 (worktree present) and g1-m1 (worktree missing)
    existing_id = "g0-s0"
    missing_id = "g1-m1"
    wt_existing = helix_dir / "worktrees" / existing_id
    wt_existing.mkdir(parents=True)

    state = make_state(generation=1, frontier=[existing_id, missing_id])
    save_state(state, project_root)

    # Patch run_evolution so we don't actually try to evolve
    # (imported locally in the resume() function, so patch it at the source module)
    with patch("helix.evolution.run_evolution") as mock_evolve:
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--dir", str(project_root)])

    # The missing worktree warning should appear
    assert missing_id in result.output, (
        f"Expected warning about missing worktree {missing_id!r} in output:\n{result.output}"
    )

    # State should have been updated to drop the missing candidate
    updated_state = load_state(project_root)
    assert updated_state is not None
    assert missing_id not in updated_state.frontier, (
        f"Missing candidate {missing_id!r} should have been removed from frontier"
    )
    assert existing_id in updated_state.frontier, (
        f"Existing candidate {existing_id!r} should still be in frontier"
    )


def test_resume_keyboard_interrupt_preserves_state_and_prints_clean_hint(tmp_path: Path) -> None:
    """CLI resume should exit cleanly on Ctrl+C and guide the user toward resume/clean."""
    from click.testing import CliRunner
    from helix.cli import cli

    project_root = tmp_path / "proj"
    project_root.mkdir()
    helix_dir = project_root / ".helix"
    helix_dir.mkdir()
    (helix_dir / "worktrees").mkdir()
    (project_root / "helix.toml").write_text(
        'objective = "test"\n\n[evaluator]\ncommand = "echo 1"\n'
    )
    save_state(make_state(generation=1, frontier=[]), project_root)

    with patch("helix.evolution.run_evolution", side_effect=KeyboardInterrupt):
        runner = CliRunner()
        result = runner.invoke(cli, ["resume", "--dir", str(project_root)])

    assert result.exit_code == 130, result.output
    assert "helix resume" in result.output.lower(), result.output
    assert "helix clean" in result.output.lower(), result.output


def test_evolve_success_prints_clean_hint(tmp_path: Path) -> None:
    """CLI evolve should remind the user that HELIX state/worktrees remain on disk."""
    from click.testing import CliRunner
    from helix.cli import cli

    project_root = tmp_path / "proj"
    project_root.mkdir()
    (project_root / "helix.toml").write_text(
        'objective = "test"\n\n[evaluator]\ncommand = "echo 1"\n'
    )

    with patch("helix.evolution.run_evolution", return_value=MagicMock()) as mock_evolve:
        runner = CliRunner()
        result = runner.invoke(cli, ["evolve", "--dir", str(project_root)])

    assert result.exit_code == 0, result.output
    assert "helix clean" in result.output.lower(), result.output

    # run_evolution should still have been called
    assert mock_evolve.called, "run_evolution should be called even after dropping worktrees"
