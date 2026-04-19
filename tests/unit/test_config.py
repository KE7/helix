"""Unit tests for helix.config — loading, validation, and defaults."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from helix.config import (
    ClaudeConfig,
    DatasetConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    WorktreeConfig,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "helix.toml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# Valid TOML loads correctly
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_minimal_valid_toml(self, tmp_path):
        """A minimal config with only required fields should load with defaults."""
        toml = write_toml(tmp_path, """
            objective = "Maximise test coverage"
            seed = "."

            [evaluator]
            command = "pytest tests/"
        """)

        cfg = load_config(toml)

        assert cfg.objective == "Maximise test coverage"
        assert cfg.seed == "."
        assert cfg.evaluator.command == "pytest tests/"
        assert cfg.evaluator.score_parser == "pytest"
        assert cfg.evaluator.include_stdout is True
        assert cfg.evaluator.include_stderr is True
        assert cfg.evaluator.extra_commands == []

    def test_full_config_toml(self, tmp_path):
        """All fields specified in TOML should be parsed correctly."""
        toml = write_toml(tmp_path, """
            objective = "Pass all benchmarks"
            seed = "/repo"

            [evaluator]
            command = "make test"
            score_parser = "exitcode"
            include_stdout = false
            include_stderr = false
            extra_commands = ["make lint", "make typecheck"]

            [evolution]
            max_generations = 20
            gating_threshold = 0.1
            perfect_score_threshold = 0.95
            convergence_patience = 3
            max_metric_calls = 500
            parallel_eval = false
            merge_enabled = false
            max_merge_invocations = 2

            [claude]
            model = "claude-opus-4-5-20250514"
            allowed_tools = ["Read", "Write"]
            background = "You are a coding expert."

            [worktree]
            base_dir = "/tmp/worktrees"
            cleanup_dominated = false
        """)

        cfg = load_config(toml)

        assert cfg.objective == "Pass all benchmarks"
        assert cfg.seed == "/repo"

        assert cfg.evaluator.command == "make test"
        assert cfg.evaluator.score_parser == "exitcode"
        assert cfg.evaluator.include_stdout is False
        assert cfg.evaluator.include_stderr is False
        assert cfg.evaluator.extra_commands == ["make lint", "make typecheck"]

        assert cfg.evolution.max_generations == 20
        assert cfg.evolution.merge_enabled is False
        assert cfg.evolution.max_merge_invocations == 2
        assert cfg.claude.model == "claude-opus-4-5-20250514"
        assert cfg.claude.allowed_tools == ["Read", "Write"]
        assert cfg.claude.background == "You are a coding expert."

        assert cfg.worktree.base_dir == "/tmp/worktrees"
        assert cfg.worktree.cleanup_dominated is False

    def test_defaults_applied_for_nested_sections(self, tmp_path):
        """Omitted nested sections should use their default values."""
        toml = write_toml(tmp_path, """
            objective = "Improve score"

            [evaluator]
            command = "pytest"
        """)

        cfg = load_config(toml)

        # DatasetConfig defaults (empty after Fix 8 — dead fields removed)
        assert cfg.dataset is not None

        # EvolutionConfig defaults
        assert cfg.evolution.max_generations == 10
        assert cfg.evolution.gating_threshold == pytest.approx(0.0)
        assert cfg.evolution.perfect_score_threshold is None
        assert cfg.evolution.convergence_patience == 5
        assert cfg.evolution.max_metric_calls == 200
        assert cfg.evolution.merge_enabled is False  # GEPA parity: off by default
        assert cfg.evolution.max_merge_invocations == 5

        # ClaudeConfig defaults
        assert cfg.claude.model == "sonnet"
        assert "Read" in cfg.claude.allowed_tools
        assert cfg.claude.background is None

        # WorktreeConfig defaults
        assert cfg.worktree.base_dir == ".helix/worktrees"
        assert cfg.worktree.cleanup_dominated is False  # deprecated: GEPA append-only


# ---------------------------------------------------------------------------
# Missing required fields raise clear errors
# ---------------------------------------------------------------------------

class TestMissingRequiredFields:
    def test_missing_objective_raises(self, tmp_path):
        """objective is required — omitting it should exit with friendly error."""
        toml = write_toml(tmp_path, """
            [evaluator]
            command = "pytest"
        """)
        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)
        assert exc_info.value.code == 1

    def test_missing_evaluator_command_raises(self, tmp_path):
        """evaluator.command is required."""
        toml = write_toml(tmp_path, """
            objective = "do something"

            [evaluator]
            score_parser = "exitcode"
        """)
        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)
        assert exc_info.value.code == 1

    def test_missing_evaluator_section_raises(self, tmp_path):
        """evaluator section itself is required."""
        toml = write_toml(tmp_path, """
            objective = "do something"
        """)
        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)
        assert exc_info.value.code == 1

    def test_invalid_score_parser_raises(self, tmp_path):
        """score_parser must be 'pytest' or 'exitcode'."""
        toml = write_toml(tmp_path, """
            objective = "X"

            [evaluator]
            command = "run"
            score_parser = "unknown_parser"
        """)
        with pytest.raises(SystemExit):
            load_config(toml)


# ---------------------------------------------------------------------------
# Direct model construction
# ---------------------------------------------------------------------------

class TestDirectModelConstruction:
    def test_helix_config_requires_objective_and_evaluator(self):
        with pytest.raises(ValidationError):
            HelixConfig()  # missing objective and evaluator

    def test_evaluator_config_requires_command(self):
        with pytest.raises(ValidationError):
            EvaluatorConfig()  # missing command

    def test_dataset_config_empty(self):
        """DatasetConfig can be constructed with no arguments (all fields removed)."""
        cfg = DatasetConfig()
        assert cfg is not None

    def test_evolution_config_defaults(self):
        cfg = EvolutionConfig()
        assert cfg.max_generations == 10

    def test_merge_subsample_size_default_is_5(self) -> None:
        """Pin default to 5 per GEPA merge.py:262.

        Changing this default without intent should be a conscious act — the
        constant is algorithmically load-bearing (stratification math uses
        `ceil(5/3) = 2` per bucket across 3 buckets).  An ablation study
        would vary the config field, not the default.
        """
        cfg = EvolutionConfig()
        assert cfg.merge_subsample_size == 5, (
            "Default must match GEPA's num_subsample_ids=5 constant "
            "(gepa/src/gepa/proposer/merge.py:262).  If you are intentionally "
            "changing this default, update this test AND the comment in "
            "config.py that cites the GEPA line."
        )

    def test_claude_config_default_tools(self):
        cfg = ClaudeConfig()
        assert "Read" in cfg.allowed_tools
        assert "Edit" in cfg.allowed_tools
        assert "Bash" in cfg.allowed_tools

    def test_worktree_config_defaults(self):
        cfg = WorktreeConfig()
        assert cfg.base_dir == ".helix/worktrees"
        assert cfg.cleanup_dominated is False  # deprecated: GEPA append-only
