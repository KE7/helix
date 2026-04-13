"""Unit tests for create_empty_seed_worktree (seedless mode)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from helix.worktree import create_empty_seed_worktree


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
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=GIT_ENV,
    )


def _make_empty_repo(path: Path) -> None:
    """Create a git repo with an empty initial commit at *path*."""
    path.mkdir(parents=True, exist_ok=True)
    _run(["git", "init"], path)
    _run(["git", "config", "user.email", "helix@test.local"], path)
    _run(["git", "config", "user.name", "HELIX Test"], path)
    _run(
        ["git", "commit", "--allow-empty", "-m", "helix: empty seed (seedless mode)"],
        path,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCreateEmptySeedWorktree:
    def test_create_empty_seed_worktree_returns_candidate(self, tmp_path):
        """create_empty_seed_worktree returns a Candidate with id='g0-s0', generation=0."""
        repo_root = tmp_path / "repo"
        _make_empty_repo(repo_root)

        base_dir = tmp_path / "worktrees"
        candidate = create_empty_seed_worktree(repo_root, base_dir)

        assert candidate.id == "g0-s0"
        assert candidate.generation == 0
        assert candidate.parent_id is None
        assert candidate.parent_ids == []
        assert candidate.operation == "seed"

    def test_create_empty_seed_worktree_path(self, tmp_path):
        """The candidate's worktree_path points to <base_dir>/g0-s0."""
        repo_root = tmp_path / "repo"
        _make_empty_repo(repo_root)

        base_dir = tmp_path / "worktrees"
        candidate = create_empty_seed_worktree(repo_root, base_dir)

        expected_path = base_dir / "g0-s0"
        assert Path(candidate.worktree_path) == expected_path

    def test_create_empty_seed_worktree_creates_git_repo(self, tmp_path):
        """The created worktree is a valid git repository."""
        repo_root = tmp_path / "repo"
        _make_empty_repo(repo_root)

        base_dir = tmp_path / "worktrees"
        candidate = create_empty_seed_worktree(repo_root, base_dir)

        wt_path = Path(candidate.worktree_path)
        assert wt_path.exists()

        # Verify it's a git repo
        result = subprocess.run(
            ["git", "rev-parse", "--git-dir"],
            cwd=wt_path,
            capture_output=True,
            text=True,
            env=GIT_ENV,
        )
        assert result.returncode == 0, f"Not a git repo: {result.stderr}"

    def test_create_empty_seed_worktree_no_project_files(self, tmp_path):
        """The empty worktree starts with no project files (only git metadata)."""
        repo_root = tmp_path / "repo"
        _make_empty_repo(repo_root)

        base_dir = tmp_path / "worktrees"
        candidate = create_empty_seed_worktree(repo_root, base_dir)

        wt_path = Path(candidate.worktree_path)
        # Only the hidden .git (or .git file for linked worktrees) should exist
        visible_files = [f for f in wt_path.iterdir() if not f.name.startswith(".")]
        assert visible_files == [], f"Expected empty worktree, found: {visible_files}"

    def test_create_empty_seed_worktree_initialises_repo_if_needed(self, tmp_path):
        """If repo_root is not yet a git repo, create_empty_seed_worktree initialises one."""
        repo_root = tmp_path / "fresh_repo"
        repo_root.mkdir()
        # Do NOT initialise git — let create_empty_seed_worktree do it.

        # We need to set git identity so the auto-init commit works.
        env_backup = os.environ.copy()
        os.environ.update({
            "GIT_AUTHOR_NAME": "HELIX Test",
            "GIT_AUTHOR_EMAIL": "helix@test.local",
            "GIT_COMMITTER_NAME": "HELIX Test",
            "GIT_COMMITTER_EMAIL": "helix@test.local",
        })
        try:
            # Patch subprocess calls to use our git identity
            import subprocess as _sp
            orig_run = _sp.run

            def patched_run(args, **kwargs):
                kwargs.setdefault("env", GIT_ENV)
                return orig_run(args, **kwargs)

            import helix.worktree as _wt_mod
            orig_wt_run = _wt_mod._run

            def patched_wt_run(args, cwd=None, check=True, operation="", env=None):
                result = orig_run(args, cwd=cwd, check=False, capture_output=True, text=True, env=GIT_ENV)
                if check and result.returncode != 0:
                    from helix.exceptions import GitError
                    raise GitError(
                        f"Git command failed: {' '.join(args)}",
                        operation=operation,
                        stderr=result.stderr,
                        exit_code=result.returncode,
                    )
                return result

            _wt_mod._run = patched_wt_run
            try:
                base_dir = tmp_path / "worktrees"
                candidate = create_empty_seed_worktree(repo_root, base_dir)
                assert candidate.id == "g0-s0"
            finally:
                _wt_mod._run = orig_wt_run
        finally:
            os.environ.clear()
            os.environ.update(env_backup)
