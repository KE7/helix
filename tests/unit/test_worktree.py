"""Unit tests for helix.worktree using pytest tmp_path fixtures.

Each test creates a fresh temporary directory to avoid cross-test
interference.  We configure a minimal git identity so commits work in
environments where no global git config exists.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from helix.worktree import (
    _ensure_git_repo,
    clone_candidate,
    create_seed_worktree,
    get_diff,
    helix_git_env,
    remove_worktree,
    snapshot_candidate,
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
        args,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        env=GIT_ENV,
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
# Tests
# ---------------------------------------------------------------------------


class TestCreateSeedWorktree:
    def test_creates_worktree_from_non_git_dir(self, tmp_path: Path, monkeypatch) -> None:
        """create_seed_worktree should initialise a repo when none exists."""
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        repo_root.mkdir()
        (repo_root / "main.py").write_text("print('hello')\n")

        base_dir = tmp_path / "worktrees"

        candidate = create_seed_worktree(repo_root, base_dir)

        assert candidate.id == "g0-s0"
        assert candidate.generation == 0
        assert candidate.parent_id is None
        assert candidate.parent_ids == []
        assert candidate.operation == "seed"
        assert candidate.branch_name == "helix/g0-s0"

        wt_path = Path(candidate.worktree_path)
        assert wt_path.exists()
        assert wt_path.is_dir()

    def test_creates_worktree_from_existing_repo(self, tmp_path: Path, monkeypatch) -> None:
        """create_seed_worktree should work on an existing git repo."""
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)

        base_dir = tmp_path / "worktrees"
        candidate = create_seed_worktree(repo_root, base_dir)

        assert (base_dir / "g0-s0").is_dir()
        assert candidate.worktree_path == str(base_dir / "g0-s0")

    def test_seed_worktree_contains_source_files(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)

        base_dir = tmp_path / "worktrees"
        candidate = create_seed_worktree(repo_root, base_dir)

        wt_path = Path(candidate.worktree_path)
        assert (wt_path / "README.md").exists()

    def test_seed_worktree_snapshots_dirty_tree_and_clones_from_snapshot(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        (repo_root / "tracked.txt").write_text("tracked\n")
        _run(["git", "add", "tracked.txt"], repo_root)
        _run(["git", "commit", "-m", "add tracked file"], repo_root)

        # Dirty working tree state that should become the run seed.
        (repo_root / "README.md").write_text("# dirty snapshot\n")
        (repo_root / "tracked.txt").unlink()
        (repo_root / "local.txt").write_text("untracked\n")

        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        wt_path = Path(seed.worktree_path)

        assert (wt_path / "README.md").read_text() == "# dirty snapshot\n"
        assert not (wt_path / "tracked.txt").exists()
        assert (wt_path / "local.txt").read_text() == "untracked\n"

        repo_head = _run(["git", "rev-parse", "HEAD"], repo_root).stdout.strip()
        seed_head = _run(["git", "rev-parse", "HEAD"], wt_path).stdout.strip()
        assert seed_head != repo_head, "Dirty snapshot should be committed in the seed worktree"

        child = clone_candidate(seed, "g1-s0", base_dir)
        child_path = Path(child.worktree_path)
        assert (child_path / "README.md").read_text() == "# dirty snapshot\n"
        assert not (child_path / "tracked.txt").exists()
        assert (child_path / "local.txt").read_text() == "untracked\n"


class TestCloneCandidate:
    def _seed(self, tmp_path: Path, monkeypatch) -> tuple[Path, Path, object]:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        return repo_root, base_dir, seed

    def test_clone_creates_new_worktree(self, tmp_path: Path, monkeypatch) -> None:
        _, base_dir, seed = self._seed(tmp_path, monkeypatch)

        child = clone_candidate(seed, "g1-s0", base_dir)

        assert child.id == "g1-s0"
        assert child.generation == 1
        assert child.parent_id == seed.id
        assert child.parent_ids == [seed.id]
        assert child.operation == "clone"
        assert child.branch_name == "helix/g1-s0"

        wt_path = Path(child.worktree_path)
        assert wt_path.exists()
        assert wt_path.is_dir()

    def test_clone_worktree_contains_parent_files(self, tmp_path: Path, monkeypatch) -> None:
        _, base_dir, seed = self._seed(tmp_path, monkeypatch)
        child = clone_candidate(seed, "g1-s1", base_dir)
        assert (Path(child.worktree_path) / "README.md").exists()

    def test_clone_has_correct_path(self, tmp_path: Path, monkeypatch) -> None:
        _, base_dir, seed = self._seed(tmp_path, monkeypatch)
        child = clone_candidate(seed, "g1-s2", base_dir)
        assert child.worktree_path == str(base_dir / "g1-s2")


class TestSnapshotCandidate:
    def test_snapshot_creates_commit(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s0", base_dir)

        # Mutate a file
        wt = Path(child.worktree_path)
        (wt / "mutated.py").write_text("x = 42\n")

        sha = snapshot_candidate(child, "helix: mutation g1-s0")

        assert isinstance(sha, str)
        assert len(sha) == 40  # full SHA

        # Verify the commit exists
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            cwd=wt,
            capture_output=True,
            text=True,
        )
        assert sha[:7] in result.stdout

    def test_snapshot_returns_sha(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s0", base_dir)

        wt = Path(child.worktree_path)
        (wt / "new_file.py").write_text("pass\n")

        sha = snapshot_candidate(child, "test commit")
        assert sha and sha.isalnum()


class TestRemoveWorktree:
    def test_remove_cleans_up_directory(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s0", base_dir)

        wt_path = Path(child.worktree_path)
        assert wt_path.exists()

        remove_worktree(child)

        assert not wt_path.exists()

    def test_remove_deletes_branch(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s0", base_dir)

        remove_worktree(child)

        result = subprocess.run(
            ["git", "branch", "--list", "helix/g1-s0"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert result.stdout.strip() == ""

    def test_remove_tolerates_missing_branch(self, tmp_path: Path, monkeypatch) -> None:
        """remove_worktree should not raise if the branch was already deleted."""
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s0", base_dir)

        # Pre-delete the branch (detach HEAD first so deletion succeeds)
        wt = Path(child.worktree_path)
        sha_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "checkout", "--detach", sha_result.stdout.strip()],
            cwd=wt,
            capture_output=True,
        )
        subprocess.run(
            ["git", "branch", "-D", "helix/g1-s0"],
            cwd=repo_root,
            capture_output=True,
        )

        # Should not raise
        remove_worktree(child)


class TestGetDiff:
    def test_diff_is_empty_for_identical_candidates(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child_a = clone_candidate(seed, "g1-s0", base_dir)
        child_b = clone_candidate(seed, "g1-s1", base_dir)

        diff = get_diff(child_a, child_b)
        assert diff == ""

    def test_diff_shows_changes(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child_a = clone_candidate(seed, "g1-s0", base_dir)
        child_b = clone_candidate(seed, "g1-s1", base_dir)

        # Mutate child_b and snapshot it
        wt_b = Path(child_b.worktree_path)
        (wt_b / "delta.py").write_text("ANSWER = 42\n")
        snapshot_candidate(child_b, "add delta.py")

        diff = get_diff(child_a, child_b)
        assert "+ANSWER = 42" in diff


# ---------------------------------------------------------------------------
# HELIX identity tests
# ---------------------------------------------------------------------------


class TestHelixGitEnv:
    def test_returns_four_expected_keys(self) -> None:
        env = helix_git_env()
        assert env["GIT_AUTHOR_NAME"] == "HELIX"
        assert env["GIT_AUTHOR_EMAIL"] == "helix@noreply"
        assert env["GIT_COMMITTER_NAME"] == "HELIX"
        assert env["GIT_COMMITTER_EMAIL"] == "helix@noreply"

    def test_returns_dict_of_strings(self) -> None:
        env = helix_git_env()
        assert isinstance(env, dict)
        assert all(isinstance(k, str) and isinstance(v, str) for k, v in env.items())


class TestEnsureGitRepoIdentity:
    def test_local_git_config_set_to_helix(self, tmp_path: Path) -> None:
        """_ensure_git_repo must set user.name=HELIX and user.email=helix@noreply
        in the local config of the freshly created repo."""
        repo_root = tmp_path / "fixture"
        repo_root.mkdir()
        (repo_root / "main.py").write_text("x = 1\n")

        _ensure_git_repo(repo_root)

        name = subprocess.run(
            ["git", "config", "user.name"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        email = subprocess.run(
            ["git", "config", "user.email"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        assert name == "HELIX"
        assert email == "helix@noreply"

    def test_initial_commit_author_is_helix(self, tmp_path: Path) -> None:
        """The initial commit created by _ensure_git_repo must be authored
        as HELIX <helix@noreply>."""
        repo_root = tmp_path / "fixture"
        repo_root.mkdir()
        (repo_root / "main.py").write_text("x = 1\n")

        _ensure_git_repo(repo_root)

        log = subprocess.run(
            ["git", "log", "--format=%an <%ae>", "-1"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        assert log == "HELIX <helix@noreply>"

    def test_no_global_git_config_still_succeeds(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_ensure_git_repo must succeed even when HOME has no .gitconfig at all
        (e.g. pristine CI environments)."""
        fake_home = tmp_path / "home_no_config"
        fake_home.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(fake_home / ".gitconfig"))

        repo_root = tmp_path / "fixture"
        repo_root.mkdir()
        (repo_root / "hello.py").write_text("print('hi')\n")

        # Must not raise even with no global identity.
        _ensure_git_repo(repo_root)

        log = subprocess.run(
            ["git", "log", "--format=%an <%ae>", "-1"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert log == "HELIX <helix@noreply>"

    def test_global_identity_does_not_bleed_into_fixture(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the user's global git config has a different identity (e.g. Alice),
        the fixture repo must still use HELIX identity, not Alice's."""
        fake_home = tmp_path / "home_alice"
        fake_home.mkdir()
        gitconfig = fake_home / ".gitconfig"
        gitconfig.write_text("[user]\n\tname = Alice\n\temail = alice@example.com\n")
        monkeypatch.setenv("HOME", str(fake_home))
        monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(gitconfig))

        repo_root = tmp_path / "fixture"
        repo_root.mkdir()
        (repo_root / "main.py").write_text("pass\n")

        _ensure_git_repo(repo_root)

        # Local config must show HELIX, not Alice.
        name = subprocess.run(
            ["git", "config", "--local", "user.name"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        email = subprocess.run(
            ["git", "config", "--local", "user.email"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert name == "HELIX"
        assert email == "helix@noreply"

        # Commit author must be HELIX.
        log = subprocess.run(
            ["git", "log", "--format=%an <%ae>", "-1"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert log == "HELIX <helix@noreply>"


class TestSnapshotCandidateIdentity:
    def test_snapshot_commit_author_is_helix(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """snapshot_candidate must author commits as HELIX <helix@noreply>."""
        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s0", base_dir)

        wt = Path(child.worktree_path)
        (wt / "mutated.py").write_text("x = 99\n")
        snapshot_candidate(child, "helix: mutation")

        log = subprocess.run(
            ["git", "log", "--format=%an <%ae>", "-1"],
            cwd=wt,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        assert log == "HELIX <helix@noreply>"
