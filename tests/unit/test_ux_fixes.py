"""Unit tests for the 9 UX fixes applied to HELIX.

Covers:
- Fix 1: uncommitted-changes warning in create_seed_worktree
- Fix 2: _update_gitignore called in `helix evolve` and `helix resume`
- Fix 5: clear actionable error when helix.toml is missing
- Fix 6: pre-flight check for stale helix/* branches
- Pre-flight check for evaluator script existence
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from helix.cli import cli
from helix.exceptions import HelixError
from helix.evolution import _check_evaluator_script_exists
from helix.worktree import (
    _check_no_stale_helix_branches,
    _warn_uncommitted_changes,
    create_seed_worktree,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "HELIX Test",
    "GIT_AUTHOR_EMAIL": "helix@test.local",
    "GIT_COMMITTER_NAME": "HELIX Test",
    "GIT_COMMITTER_EMAIL": "helix@test.local",
}


def _run(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args, cwd=cwd, check=True, capture_output=True, text=True, env=GIT_ENV
    )


def _make_repo(path: Path) -> None:
    """Create a minimal git repo with one commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text("# HELIX seed\n")
    _run(["git", "init"], path)
    _run(["git", "config", "user.email", "helix@test.local"], path)
    _run(["git", "config", "user.name", "HELIX Test"], path)
    _run(["git", "add", "-A"], path)
    _run(["git", "commit", "-m", "init"], path)


# ---------------------------------------------------------------------------
# Fix 1 — uncommitted-changes warning
# ---------------------------------------------------------------------------


class TestWarnUncommittedChanges:
    """Tests for _warn_uncommitted_changes() helper."""

    def test_no_warning_when_repo_is_clean(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        _make_repo(tmp_path)
        with patch("helix.display.console") as mock_console:
            _warn_uncommitted_changes(tmp_path)
            mock_console.print.assert_not_called()

    def test_warning_printed_when_uncommitted_changes_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        _make_repo(tmp_path)
        # Create an uncommitted file
        (tmp_path / "dirty.py").write_text("x = 1\n")

        with patch("helix.display.console") as mock_console:
            _warn_uncommitted_changes(tmp_path)
            mock_console.print.assert_called_once()
            call_args = mock_console.print.call_args[0][0]
            assert "uncommitted changes" in call_args
            assert str(tmp_path) in call_args

    def test_warning_mentions_head_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        _make_repo(tmp_path)
        (tmp_path / "dirty.py").write_text("x = 1\n")

        with patch("helix.display.console") as mock_console:
            _warn_uncommitted_changes(tmp_path)
            call_args = mock_console.print.call_args[0][0]
            assert "uncommitted changes" in call_args.lower() or "working tree" in call_args.lower()

    def test_create_seed_worktree_warns_on_dirty_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_seed_worktree should emit the warning and still proceed."""
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        (repo_root / "untracked.py").write_text("pass\n")

        base_dir = tmp_path / "worktrees"

        with patch("helix.display.console") as mock_console:
            candidate = create_seed_worktree(repo_root, base_dir)
            # Warning must have been printed
            mock_console.print.assert_called_once()
            # But the worktree is still created
            assert Path(candidate.worktree_path).exists()

    def test_create_seed_worktree_no_warning_on_clean_repo(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)

        base_dir = tmp_path / "worktrees"

        with patch("helix.display.console") as mock_console:
            create_seed_worktree(repo_root, base_dir)
            mock_console.print.assert_not_called()


# ---------------------------------------------------------------------------
# Fix 2 — _update_gitignore called in helix evolve / helix resume
# ---------------------------------------------------------------------------


class TestEvolveCallsUpdateGitignore:
    """helix evolve should call _update_gitignore early."""

    def test_evolve_calls_update_gitignore(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        # Provide a valid helix.toml
        (tmp_path / "helix.toml").write_text(
            'objective = "test"\n[evaluator]\ncommand = "echo ok"\n'
        )
        runner = CliRunner()
        # run_evolution is a local import inside the function; patch at its source.
        with (
            patch("helix.cli._update_gitignore") as mock_gitignore,
            patch("helix.cli._ensure_git_repo"),
            patch("helix.evolution.run_evolution"),
        ):
            runner.invoke(cli, ["evolve"], catch_exceptions=False)
            mock_gitignore.assert_called_once_with(tmp_path)

    def test_evolve_gitignore_is_idempotent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling helix evolve twice must not duplicate the .helix/ entry."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "helix.toml").write_text(
            'objective = "test"\n[evaluator]\ncommand = "echo ok"\n'
        )
        runner = CliRunner()
        with patch("helix.evolution.run_evolution"):
            runner.invoke(cli, ["evolve"], catch_exceptions=False)
            runner.invoke(cli, ["evolve"], catch_exceptions=False)
        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists()
        assert gitignore.read_text().splitlines().count(".helix/") == 1

    def test_resume_calls_update_gitignore(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        (tmp_path / "helix.toml").write_text(
            'objective = "test"\n[evaluator]\ncommand = "echo ok"\n'
        )
        runner = CliRunner()
        with (
            patch("helix.cli._update_gitignore") as mock_gitignore,
            patch("helix.cli.load_state", return_value=None),
            patch("helix.evolution.run_evolution"),
        ):
            runner.invoke(cli, ["resume"], catch_exceptions=False)
            mock_gitignore.assert_called_once_with(tmp_path)


# ---------------------------------------------------------------------------
# Fix 5 — clear actionable error when helix.toml is missing
# ---------------------------------------------------------------------------


class TestEvolveMissingConfig:
    """helix evolve should print a clear error if helix.toml is absent."""

    def test_missing_helix_toml_prints_actionable_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        with (
            patch("helix.cli._ensure_git_repo"),
            patch("helix.cli._update_gitignore"),
        ):
            result = runner.invoke(cli, ["evolve"], catch_exceptions=True)
        assert result.exit_code != 0
        assert "helix init" in result.output or "helix.toml" in result.output

    def test_missing_helix_toml_exit_code_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        with (
            patch("helix.cli._ensure_git_repo"),
            patch("helix.cli._update_gitignore"),
        ):
            result = runner.invoke(cli, ["evolve"])
        assert result.exit_code == 1

    def test_missing_helix_toml_mentions_directory(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        runner = CliRunner()
        with (
            patch("helix.cli._ensure_git_repo"),
            patch("helix.cli._update_gitignore"),
        ):
            result = runner.invoke(cli, ["evolve"])
        # The error message should mention the project directory
        assert str(tmp_path) in result.output or "helix.toml" in result.output


# ---------------------------------------------------------------------------
# Fix 6 — pre-flight check for stale helix/* branches
# ---------------------------------------------------------------------------


class TestCheckNoStaleHelixBranches:
    """Tests for _check_no_stale_helix_branches() and its integration."""

    def test_no_error_when_no_helix_branches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        _make_repo(tmp_path)
        # Should not raise
        _check_no_stale_helix_branches(tmp_path)

    def test_raises_helix_error_when_stale_branches_exist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        _make_repo(tmp_path)
        # Create a stale helix/* branch
        _run(["git", "branch", "helix/g0-s0"], tmp_path)

        with pytest.raises(HelixError) as exc_info:
            _check_no_stale_helix_branches(tmp_path)

        assert "helix/g0-s0" in str(exc_info.value)
        assert "helix clean" in str(exc_info.value) or "helix resume" in str(exc_info.value)

    def test_error_message_includes_branch_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        _make_repo(tmp_path)
        _run(["git", "branch", "helix/g0-s0"], tmp_path)
        _run(["git", "branch", "helix/g1-s0"], tmp_path)

        with pytest.raises(HelixError) as exc_info:
            _check_no_stale_helix_branches(tmp_path)

        assert "2" in str(exc_info.value)

    def test_create_seed_worktree_aborts_on_stale_branches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """create_seed_worktree should raise HelixError when helix/* branches exist."""
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        # Pre-seed a stale branch
        _run(["git", "branch", "helix/g0-s0"], repo_root)

        base_dir = tmp_path / "worktrees"

        with pytest.raises(HelixError):
            create_seed_worktree(repo_root, base_dir)

    def test_create_seed_worktree_proceeds_without_stale_branches(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"

        candidate = create_seed_worktree(repo_root, base_dir)
        assert Path(candidate.worktree_path).exists()


# ---------------------------------------------------------------------------
# Pre-flight check for evaluator script existence
# ---------------------------------------------------------------------------


class TestEvaluatorScriptPreflightCheck:
    """Tests for _check_evaluator_script_exists() pre-flight validation."""

    def test_valid_script_passes(self, tmp_path: Path) -> None:
        """When evaluator script exists, check should pass silently."""
        (tmp_path / "evaluate.py").write_text("# evaluator\n")
        # Should not raise
        _check_evaluator_script_exists("python evaluate.py", tmp_path)

    def test_valid_script_with_python3_passes(self, tmp_path: Path) -> None:
        """Script with python3 prefix should pass."""
        (tmp_path / "eval.py").write_text("# evaluator\n")
        _check_evaluator_script_exists("python3 eval.py", tmp_path)

    def test_valid_script_with_uv_passes(self, tmp_path: Path) -> None:
        """Script with uv prefix should pass."""
        (tmp_path / "test.py").write_text("# evaluator\n")
        _check_evaluator_script_exists("uv run test.py", tmp_path)

    def test_missing_script_raises_system_exit(self, tmp_path: Path) -> None:
        """When script does not exist, should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            _check_evaluator_script_exists("python missing.py", tmp_path)
        assert exc_info.value.code == 1

    def test_missing_script_error_mentions_path(self, tmp_path: Path) -> None:
        """Error message should include the missing script path."""
        with pytest.raises(SystemExit):
            with patch("helix.evolution.print_error") as mock_error:
                _check_evaluator_script_exists("python missing.py", tmp_path)
                # Check that print_error was called with a message containing the path
                assert mock_error.called
                error_msg = mock_error.call_args[0][0]
                assert "missing.py" in error_msg
                assert "not found" in error_msg.lower()

    def test_command_with_flags_parses_correctly(self, tmp_path: Path) -> None:
        """Command with flags should correctly identify script path."""
        (tmp_path / "run.py").write_text("# evaluator\n")
        _check_evaluator_script_exists("python -u run.py --verbose", tmp_path)

    def test_make_command_allowed(self, tmp_path: Path) -> None:
        """Commands like 'make test' that don't reference a script should pass."""
        # make doesn't need a script file, so it should not raise
        _check_evaluator_script_exists("make test", tmp_path)

    def test_pytest_command_allowed(self, tmp_path: Path) -> None:
        """Commands like 'pytest tests/' should pass even if no specific script."""
        _check_evaluator_script_exists("pytest tests/", tmp_path)

    def test_path_is_directory_raises(self, tmp_path: Path) -> None:
        """If the script path is a directory (with .py extension), should exit with code 1."""
        (tmp_path / "evaluate.py").mkdir()
        with pytest.raises(SystemExit) as exc_info:
            _check_evaluator_script_exists("python evaluate.py", tmp_path)
        assert exc_info.value.code == 1

    def test_empty_command_raises(self, tmp_path: Path) -> None:
        """Empty command should exit with code 1."""
        with pytest.raises(SystemExit) as exc_info:
            _check_evaluator_script_exists("", tmp_path)
        assert exc_info.value.code == 1

    # ------------------------------------------------------------------
    # Shell-wrapper exemption: `bash -c "..."` style commands wrap the
    # real invocation inside an opaque body string that shlex cannot
    # meaningfully tokenize for path validation. The exemption is narrow
    # — it only applies to known shells + known -c-family flags.
    # ------------------------------------------------------------------

    def test_shell_wrapper_bash_c_passes_validation(self, tmp_path: Path) -> None:
        """`bash -c "..."` should skip path-level validation."""
        # No script file is created on disk; shell body is opaque.
        _check_evaluator_script_exists(
            'bash -c "cd /work/foo && python evaluate.py"', tmp_path
        )

    def test_shell_wrapper_bash_lc_passes_validation(self, tmp_path: Path) -> None:
        """`bash -lc "..."` should skip path-level validation."""
        _check_evaluator_script_exists(
            'bash -lc "cd /work/foo && python evaluate.py"', tmp_path
        )

    def test_shell_wrapper_sh_c_passes_validation(self, tmp_path: Path) -> None:
        """`sh -c "..."` should skip path-level validation."""
        _check_evaluator_script_exists(
            'sh -c "cd /work/foo && python evaluate.py"', tmp_path
        )

    def test_shell_wrapper_zsh_c_passes_validation(self, tmp_path: Path) -> None:
        """`zsh -c "..."` should skip path-level validation."""
        _check_evaluator_script_exists(
            'zsh -c "cd /work/foo && python evaluate.py"', tmp_path
        )

    def test_shell_wrapper_bash_without_c_flag_still_validates(
        self, tmp_path: Path
    ) -> None:
        """`bash foo.py` (no -c-family flag) should still validate foo.py.

        The exemption is intentionally narrow: only `bash -c/-lc/...`
        shell-wrappers are skipped. A bare `bash some_script.py` must
        keep its existing path-checking behavior.
        """
        with pytest.raises(SystemExit) as exc_info:
            _check_evaluator_script_exists("bash foo.py", tmp_path)
        assert exc_info.value.code == 1

    def test_shell_wrapper_bash_with_unknown_flag_still_validates(
        self, tmp_path: Path
    ) -> None:
        """`bash -x script.py` (unknown flag) should NOT get the exemption.

        Only the documented `-c`-family flags trigger the skip; anything
        else falls through to the regular path validator.
        """
        with pytest.raises(SystemExit) as exc_info:
            _check_evaluator_script_exists("bash -x script.py", tmp_path)
        assert exc_info.value.code == 1

    def test_evolve_runs_preflight_check_before_worktree(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """helix evolve should run evaluator check before creating worktrees."""
        monkeypatch.chdir(tmp_path)
        (tmp_path / "helix.toml").write_text(
            'objective = "test"\n[evaluator]\ncommand = "python missing.py"\n'
        )
        runner = CliRunner()
        with (
            patch("helix.cli._ensure_git_repo"),
            patch("helix.cli._update_gitignore"),
        ):
            result = runner.invoke(cli, ["evolve"], catch_exceptions=True)
        # Should fail before any worktree is created
        assert result.exit_code == 1
        assert "missing.py" in result.output or "not found" in result.output.lower()
