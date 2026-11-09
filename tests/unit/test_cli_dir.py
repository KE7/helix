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
        """log --dir <empty_dir> should NOT read the run in cwd."""
        other = tmp_path / "other"
        other.mkdir()
        # Put a run's lineage in cwd but NOT in other/
        helix_dir = tmp_path / ".helix"
        helix_dir.mkdir()
        (helix_dir / "lineage.json").write_text(
            '[{"id": "g0-s0", "parent": null, "parents": [], '
            '"operation": "seed", "generation": 0, "files_changed": []}]'
        )
        monkeypatch.chdir(tmp_path)

        runner = CliRunner()
        result = runner.invoke(cli, ["log", "--dir", str(other)])
        assert "no candidates found" in result.output.lower(), result.output

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


def test_evolve_with_relative_dir_puts_the_seed_worktree_in_that_project(
    tmp_path, monkeypatch, mocker
):
    """``--dir proj`` from the parent directory.  The worktree path is handed
    to a subprocess running in the project, so a relative root would put the
    seed at proj/proj/.helix/... while HELIX evaluates proj/.helix/...."""
    from helix.worktree import create_seed_worktree

    project = tmp_path / "proj"
    project.mkdir()
    _make_project(project)
    seen = {}

    def fake_run_evolution(config, project_root, base_dir):
        seen["root"] = project_root
        seed = create_seed_worktree(project_root, base_dir / "worktrees")
        seen["seed"] = Path(seed.worktree_path)

    mocker.patch("helix.evolution.run_evolution", side_effect=fake_run_evolution)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(cli, ["evolve", "--dir", "proj"])

    assert result.exit_code == 0, result.output
    assert seen["root"].is_absolute()
    assert seen["seed"].is_absolute()
    expected = project.resolve() / ".helix" / "worktrees" / "g0-s0"
    assert (expected / "helix.toml").exists()
    assert not (project / "proj").exists()


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
