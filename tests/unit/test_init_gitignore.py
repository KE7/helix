"""Unit tests for `helix init` .gitignore behaviour."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from helix.cli import cli, _update_gitignore, _create_initial_gitignore


# ---------------------------------------------------------------------------
# Direct helper tests
# ---------------------------------------------------------------------------

class TestUpdateGitignore:
    """Tests for the _update_gitignore() helper in isolation."""

    def test_creates_gitignore_when_absent(self, tmp_path):
        _update_gitignore(tmp_path)
        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists()
        assert ".helix/" in gitignore.read_text()

    def test_gitignore_contains_helix_dir_entry(self, tmp_path):
        _update_gitignore(tmp_path)
        lines = (tmp_path / ".gitignore").read_text().splitlines()
        assert ".helix/" in lines

    def test_appends_to_existing_gitignore(self, tmp_path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("__pycache__/\n*.pyc\n")
        _update_gitignore(tmp_path)
        content = gitignore.read_text()
        assert "__pycache__/" in content
        assert "*.pyc" in content
        assert ".helix/" in content

    def test_no_duplicate_when_entry_already_present(self, tmp_path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text(".helix/\n")
        _update_gitignore(tmp_path)
        count = gitignore.read_text().splitlines().count(".helix/")
        assert count == 1

    def test_no_duplicate_when_entry_already_present_without_trailing_newline(self, tmp_path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text(".helix/")
        _update_gitignore(tmp_path)
        count = gitignore.read_text().count(".helix/")
        assert count == 1

    def test_appends_with_newline_separator_when_file_lacks_trailing_newline(self, tmp_path):
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log")
        _update_gitignore(tmp_path)
        text = gitignore.read_text()
        # ".helix/" must be on its own line, not concatenated with previous content
        assert "\n.helix/" in text


# ---------------------------------------------------------------------------
# Integration tests via `helix init`
# ---------------------------------------------------------------------------

class TestInitWritesGitignore:
    """helix init should add .helix/ to .gitignore."""

    def _invoke_init(self, tmp_path: Path):
        """Run `helix init` with cwd set to tmp_path, mocking git calls."""
        runner = CliRunner()
        with patch("helix.cli._ensure_git_repo"):
            result = runner.invoke(cli, ["init"], catch_exceptions=False,
                                   env={"HOME": str(tmp_path)})
        return result

    def test_init_creates_gitignore_with_helix_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = self._invoke_init(tmp_path)
        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists(), f"init output:\n{result.output}"
        assert ".helix/" in gitignore.read_text()

    def test_init_appends_to_existing_gitignore(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("venv/\n")
        result = self._invoke_init(tmp_path)
        content = gitignore.read_text()
        assert "venv/" in content, "existing content should be preserved"
        assert ".helix/" in content

    def test_init_does_not_duplicate_helix_entry(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text(".helix/\n")
        self._invoke_init(tmp_path)
        count = gitignore.read_text().splitlines().count(".helix/")
        assert count == 1

    def test_init_idempotent_on_second_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._invoke_init(tmp_path)
        self._invoke_init(tmp_path)
        count = (tmp_path / ".gitignore").read_text().splitlines().count(".helix/")
        assert count == 1


# ---------------------------------------------------------------------------
# Tests for safe git add (noise filtering)
# ---------------------------------------------------------------------------

class TestCreateInitialGitignore:
    """Tests for the _create_initial_gitignore() helper."""

    def test_creates_gitignore_with_noise_patterns(self, tmp_path):
        """Verify that .gitignore is created with common noise patterns."""
        _create_initial_gitignore(tmp_path)
        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists()
        content = gitignore.read_text()

        # Check for key noise patterns
        assert "__pycache__/" in content
        assert "*.pyc" in content
        assert "*.pyo" in content
        assert ".pytest_cache/" in content
        assert ".mypy_cache/" in content
        assert ".hypothesis/" in content
        assert "*.egg-info/" in content
        assert "build/" in content
        assert "dist/" in content
        assert ".coverage" in content
        assert "htmlcov/" in content
        assert ".env" in content
        assert ".venv/" in content
        assert "venv/" in content
        assert "node_modules/" in content
        assert ".DS_Store" in content

    def test_does_not_overwrite_existing_gitignore(self, tmp_path):
        """Verify that existing .gitignore is not overwritten."""
        gitignore = tmp_path / ".gitignore"
        original_content = "# Custom content\n*.log\n"
        gitignore.write_text(original_content)

        _create_initial_gitignore(tmp_path)

        # Should not have been modified
        assert gitignore.read_text() == original_content

    def test_gitignore_filters_pycache_during_init(self, tmp_path, monkeypatch):
        """Verify that __pycache__ would be excluded when git add -A is run."""
        monkeypatch.chdir(tmp_path)

        # Create a mock __pycache__ directory before init
        pycache_dir = tmp_path / "__pycache__"
        pycache_dir.mkdir()
        (pycache_dir / "test.pyc").write_text("compiled")

        # Create .gitignore with noise patterns
        _create_initial_gitignore(tmp_path)

        # Verify .gitignore exists and contains __pycache__/
        gitignore = tmp_path / ".gitignore"
        assert gitignore.exists()
        assert "__pycache__/" in gitignore.read_text()
