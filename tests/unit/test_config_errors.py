"""Unit tests for helix.config error handling — user-friendly TOML and validation errors."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from helix.config import load_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_toml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "helix.toml"
    p.write_text(textwrap.dedent(content))
    return p


# ---------------------------------------------------------------------------
# TOML parsing errors produce friendly messages and exit
# ---------------------------------------------------------------------------

class TestTOMLParsingErrors:
    def test_invalid_toml_syntax_exits(self, tmp_path, capsys):
        """Invalid TOML syntax should print a friendly error and exit with code 1."""
        toml = write_toml(tmp_path, """
            objective = "test
            # missing closing quote
            [evaluator]
            command = "pytest"
        """)

        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error parsing TOML file" in captured.err
        assert str(toml) in captured.err
        assert "TOML syntax" in captured.err

    def test_invalid_toml_duplicate_key_exits(self, tmp_path, capsys):
        """Duplicate keys in TOML should print a friendly error and exit."""
        toml = write_toml(tmp_path, """
            objective = "test1"
            objective = "test2"

            [evaluator]
            command = "pytest"
        """)

        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error parsing TOML file" in captured.err

    def test_invalid_toml_bad_array_exits(self, tmp_path, capsys):
        """Malformed arrays in TOML should print a friendly error and exit."""
        toml = write_toml(tmp_path, """
            objective = "test"

            [evaluator]
            command = "pytest"
            extra_commands = ["cmd1", "cmd2",]
            # trailing comma is actually valid in TOML, let's use a different error
        """)
        # This actually parses fine in TOML, let me use a real error
        toml.write_text('objective = "test"\n[evaluator\ncommand = "pytest"')

        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error parsing TOML file" in captured.err


# ---------------------------------------------------------------------------
# Validation errors produce friendly messages and exit
# ---------------------------------------------------------------------------

class TestValidationErrors:
    def test_missing_objective_exits(self, tmp_path, capsys):
        """Missing objective field should print a friendly error and exit with code 1."""
        toml = write_toml(tmp_path, """
            [evaluator]
            command = "pytest"
        """)

        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Configuration validation error" in captured.err
        assert "objective" in captured.err
        assert "Hint:" in captured.err or "hint" in captured.err.lower()

    def test_missing_evaluator_command_exits(self, tmp_path, capsys):
        """Missing evaluator.command should print a friendly error and exit."""
        toml = write_toml(tmp_path, """
            objective = "do something"

            [evaluator]
            score_parser = "exitcode"
        """)

        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Configuration validation error" in captured.err
        assert "command" in captured.err

    def test_missing_evaluator_section_exits(self, tmp_path, capsys):
        """Missing evaluator section should print a friendly error and exit."""
        toml = write_toml(tmp_path, """
            objective = "do something"
        """)

        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Configuration validation error" in captured.err
        assert "evaluator" in captured.err

    def test_invalid_score_parser_exits(self, tmp_path, capsys):
        """Invalid score_parser value should print a friendly error and exit."""
        toml = write_toml(tmp_path, """
            objective = "X"

            [evaluator]
            command = "run"
            score_parser = "unknown_parser"
        """)

        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Configuration validation error" in captured.err
        # Should mention the field path
        assert "score_parser" in captured.err

    def test_invalid_type_exits(self, tmp_path, capsys):
        """Wrong type for a field should print a friendly error with type hint."""
        toml = write_toml(tmp_path, """
            objective = "test"

            [evaluator]
            command = "pytest"

            [evolution]
            max_generations = "not_a_number"
        """)

        with pytest.raises(SystemExit) as exc_info:
            load_config(toml)

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Configuration validation error" in captured.err
        assert "max_generations" in captured.err


# ---------------------------------------------------------------------------
# Valid TOML still works fine
# ---------------------------------------------------------------------------

class TestValidTOMLStillWorks:
    def test_minimal_valid_toml_succeeds(self, tmp_path):
        """A minimal valid config should load successfully (no exit)."""
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

    def test_full_valid_toml_succeeds(self, tmp_path):
        """A full valid config should load successfully."""
        toml = write_toml(tmp_path, """
            objective = "Pass all benchmarks"
            seed = "/repo"

            [evaluator]
            command = "make test"
            score_parser = "exitcode"

            [evolution]
            max_generations = 20

            [claude]
            model = "sonnet"
        """)

        cfg = load_config(toml)

        assert cfg.objective == "Pass all benchmarks"
        assert cfg.evaluator.command == "make test"
        assert cfg.evolution.max_generations == 20
