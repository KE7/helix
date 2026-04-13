"""Unit tests for --dir option on HELIX CLI commands."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from helix.cli import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_project(tmp_path: Path) -> Path:
    """Create a minimal project directory that helix CLI can point --dir at."""
    (tmp_path / "helix.toml").write_text(
        'objective = "test"\n\n[evaluator]\ncommand = "true"\n'
    )
    return tmp_path


# ---------------------------------------------------------------------------
# --dir is accepted (option parsing / Path(exists=True) validation)
# ---------------------------------------------------------------------------

class TestDirOptionAccepted:
    """Each command should accept --dir <path> without raising a usage error."""

    def test_log_accepts_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["log", "--dir", str(tmp_path)])
        # The command may warn "no log entries", but --dir itself must not cause
        # a "No such option" or "Path does not exist" error.
        assert result.exit_code in (0, 1), result.output
        assert "no such option" not in result.output.lower()
        assert "path" not in result.output.lower() or "project" in result.output.lower()

    def test_best_accepts_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["best", "--dir", str(tmp_path)])
        assert result.exit_code in (0, 1), result.output
        assert "no such option" not in result.output.lower()

    def test_frontier_accepts_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["frontier", "--dir", str(tmp_path)])
        assert result.exit_code in (0, 1), result.output
        assert "no such option" not in result.output.lower()

    def test_history_accepts_dir(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--dir", str(tmp_path)])
        assert result.exit_code in (0, 1), result.output
        assert "no such option" not in result.output.lower()

    def test_evolve_accepts_dir(self, tmp_path):
        """evolve --dir should accept the path (fails fast on missing helix.toml if not present)."""
        _make_project(tmp_path)
        runner = CliRunner()
        # We only check option parsing; do NOT actually run evolution.
        result = runner.invoke(cli, ["evolve", "--help"])
        assert "--dir" in result.output
        assert "no such option" not in result.output.lower()


# ---------------------------------------------------------------------------
# --dir resolves the correct project_root
# ---------------------------------------------------------------------------

class TestDirResolvesRoot:
    """Commands should use the --dir path, not cwd, to locate .helix/."""

    def test_log_uses_dir_not_cwd(self, tmp_path, monkeypatch):
        """log --dir <empty_dir> should NOT look in cwd for .helix/log."""
        other = tmp_path / "other"
        other.mkdir()
        # Put a .helix/log in cwd but NOT in other/
        helix_log = tmp_path / ".helix" / "log"
        helix_log.mkdir(parents=True)
        (helix_log / "g0.json").write_text(
            '{"generation": 0, "candidate_id": "g0-s0", "operation": "seed", '
            '"timestamp": "2024-01-01", "summary": {}}'
        )
        monkeypatch.chdir(tmp_path)  # cwd has .helix/log entries

        runner = CliRunner()
        result = runner.invoke(cli, ["log", "--dir", str(other)])
        # Pointing at `other/` which has no .helix/log → should warn "No log entries"
        assert "no log entries" in result.output.lower(), result.output

    def test_best_uses_dir_not_cwd(self, tmp_path, monkeypatch):
        """best --dir <empty_dir> should not find evolution state."""
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["best", "--dir", str(other)])
        assert "no evolution state" in result.output.lower(), result.output

    def test_frontier_uses_dir_not_cwd(self, tmp_path, monkeypatch):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["frontier", "--dir", str(other)])
        assert "no evolution state" in result.output.lower(), result.output

    def test_history_uses_dir_not_cwd(self, tmp_path, monkeypatch):
        other = tmp_path / "other"
        other.mkdir()
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["history", "--dir", str(other)])
        assert "no lineage data" in result.output.lower(), result.output


# ---------------------------------------------------------------------------
# --dir with a nonexistent path produces a Click error
# ---------------------------------------------------------------------------

class TestDirNonexistentPath:
    def test_log_rejects_missing_dir(self, tmp_path):
        runner = CliRunner()
        missing = str(tmp_path / "does_not_exist")
        result = runner.invoke(cli, ["log", "--dir", missing])
        assert result.exit_code != 0
        # Click Path(exists=True) should report the path doesn't exist
        assert "does not exist" in result.output.lower() or "invalid" in result.output.lower()

    def test_best_rejects_missing_dir(self, tmp_path):
        runner = CliRunner()
        missing = str(tmp_path / "nope")
        result = runner.invoke(cli, ["best", "--dir", missing])
        assert result.exit_code != 0

    def test_frontier_rejects_missing_dir(self, tmp_path):
        runner = CliRunner()
        missing = str(tmp_path / "nope")
        result = runner.invoke(cli, ["frontier", "--dir", missing])
        assert result.exit_code != 0

    def test_history_rejects_missing_dir(self, tmp_path):
        runner = CliRunner()
        missing = str(tmp_path / "nope")
        result = runner.invoke(cli, ["history", "--dir", missing])
        assert result.exit_code != 0

    def test_clean_rejects_missing_dir(self, tmp_path):
        runner = CliRunner()
        missing = str(tmp_path / "nope")
        result = runner.invoke(cli, ["clean", "--dir", missing])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Help output includes --dir for all relevant commands
# ---------------------------------------------------------------------------

class TestHelpContainsDir:
    @pytest.mark.parametrize("command", ["log", "best", "frontier", "history", "resume", "clean", "evolve"])
    def test_help_shows_dir(self, command):
        runner = CliRunner()
        result = runner.invoke(cli, [command, "--help"])
        assert result.exit_code == 0, result.output
        assert "--dir" in result.output, f"--dir missing from `helix {command} --help`"

    def test_evolve_help_shows_model(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["evolve", "--help"])
        assert "--model" in result.output
        assert "--effort" in result.output
        assert "--dir" in result.output

    def test_log_help_describes_trajectory(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["log", "--help"])
        # Should mention trajectory or parent lineage in the docstring
        assert any(word in result.output.lower() for word in ("trajectory", "lineage", "parent"))

    def test_best_help_shows_export(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["best", "--help"])
        assert "--export" in result.output
        assert "--dir" in result.output
