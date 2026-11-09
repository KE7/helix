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

from helix.population import Candidate
from helix.worktree import (
    _ensure_git_repo,
    changed_files,
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

    def test_seed_children_start_with_helix_artifacts_ignored(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """A child's diff shows only the agent's work, not HELIX's .gitignore."""
        from helix.mutator import _ignore_helix_artifacts

        monkeypatch.setenv("GIT_AUTHOR_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_AUTHOR_EMAIL", "helix@test.local")
        monkeypatch.setenv("GIT_COMMITTER_NAME", "HELIX Test")
        monkeypatch.setenv("GIT_COMMITTER_EMAIL", "helix@test.local")
        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)

        child = clone_candidate(seed, "g1-s0", base_dir)
        _ignore_helix_artifacts(Path(child.worktree_path))

        status = _run(["git", "status", "--porcelain"], Path(child.worktree_path))
        assert status.stdout == ""

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

    def test_clone_reclaims_a_leftover_branch_without_a_worktree(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # An interrupted cleanup removed g1-s2's worktree but not its branch
        # (seen when Ctrl-C hit a pi mutation's cleanup); resume reuses the id.
        repo_root, base_dir, seed = self._seed(tmp_path, monkeypatch)
        stale = clone_candidate(seed, "g1-s2", base_dir)
        (Path(stale.worktree_path) / "README.md").write_text("stale edit\n")
        _run(["git", "commit", "-qam", "stale"], cwd=Path(stale.worktree_path))
        _run(["git", "worktree", "remove", "--force", stale.worktree_path], cwd=repo_root)
        assert _run(["git", "branch", "--list", "helix/g1-s2"], cwd=repo_root).stdout.strip()

        child = clone_candidate(seed, "g1-s2", base_dir)

        seed_head = _run(["git", "rev-parse", "HEAD"], cwd=Path(seed.worktree_path)).stdout
        child_head = _run(["git", "rev-parse", "HEAD"], cwd=Path(child.worktree_path)).stdout
        assert child_head == seed_head
        assert (Path(child.worktree_path) / "README.md").read_text() != "stale edit\n"

    def test_clone_never_takes_a_branch_a_live_worktree_uses(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        # A retained candidate keeps its worktree; its branch must survive.
        from helix.exceptions import GitError

        _, base_dir, seed = self._seed(tmp_path, monkeypatch)
        kept = clone_candidate(seed, "g1-s3", base_dir)

        with pytest.raises(GitError):
            clone_candidate(seed, "g1-s3", tmp_path / "elsewhere")
        assert Path(kept.worktree_path).is_dir()

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

    def test_diff_against_the_detached_seed(self, tmp_path: Path, monkeypatch) -> None:
        """The seed worktree is detached, so its ``helix/g0-s0`` branch never
        exists; a merge whose common ancestor is the seed diffs against it."""
        for key in ("AUTHOR", "COMMITTER"):
            monkeypatch.setenv(f"GIT_{key}_NAME", "HELIX Test")
            monkeypatch.setenv(f"GIT_{key}_EMAIL", "helix@test.local")
        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s1", base_dir)
        (Path(child.worktree_path) / "delta.py").write_text("ANSWER = 42\n")
        snapshot_candidate(child, "add delta.py")

        assert "+ANSWER = 42" in get_diff(seed, child)

    def test_diff_leaves_out_excluded_paths(self, tmp_path: Path, monkeypatch) -> None:
        for key in ("AUTHOR", "COMMITTER"):
            monkeypatch.setenv(f"GIT_{key}_NAME", "HELIX Test")
            monkeypatch.setenv(f"GIT_{key}_EMAIL", "helix@test.local")
        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s1", base_dir)
        wt = Path(child.worktree_path)
        (wt / "solve.py").write_text("ANSWER = 42\n")
        (wt / "evaluate.py").write_text("SECRET_SCORER = 1\n")
        (wt / "data" / "val").mkdir(parents=True)
        (wt / "data" / "val" / "case.json").write_text('"SECRET_ANSWER"\n')
        snapshot_candidate(child, "add files")

        diff = get_diff(seed, child, exclude=["evaluate.py", "data/val"])

        assert "+ANSWER = 42" in diff
        assert "SECRET" not in diff

    def test_diff_failure_raises(self, tmp_path: Path, monkeypatch) -> None:
        from helix.exceptions import GitError

        for key in ("AUTHOR", "COMMITTER"):
            monkeypatch.setenv(f"GIT_{key}_NAME", "HELIX Test")
            monkeypatch.setenv(f"GIT_{key}_EMAIL", "helix@test.local")
        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        gone = clone_candidate(seed, "g1-s1", base_dir)
        remove_worktree(gone)

        with pytest.raises(GitError):
            get_diff(seed, gone)


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


class TestConcurrentWorktreeMetadata:
    """``git worktree`` common-metadata operations must not interleave.

    ``git worktree add`` walks every entry under ``.git/worktrees/`` while it
    registers a new one, and ``git worktree remove`` deletes those entries.
    Run concurrently they race, and ``add`` aborts with
    ``fatal: failed to read .git/worktrees/<other>/commondir``.
    """

    @staticmethod
    def _is_metadata_command(args: object) -> bool:
        if not isinstance(args, list) or len(args) < 2 or args[0] != "git":
            return False
        return args[1] == "worktree" or (args[1] == "branch" and "-D" in args)

    def test_metadata_commands_never_run_concurrently(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import threading
        import time

        from helix import worktree as worktree_module

        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        base_dir.mkdir()

        parent = Candidate(
            id="g0-s0",
            worktree_path=str(repo_root),
            branch_name="main",
            generation=0,
            parent_id=None,
            parent_ids=[],
            operation="seed",
        )

        probe = threading.Lock()
        active = 0
        max_active = 0
        real_run = worktree_module.subprocess.run

        def instrumented_run(args, *a, **kw):  # type: ignore[no-untyped-def]
            nonlocal active, max_active
            if not self._is_metadata_command(args):
                return real_run(args, *a, **kw)
            with probe:
                active += 1
                max_active = max(max_active, active)
            try:
                # Widen the window so an unserialised implementation is
                # observed deterministically rather than probabilistically.
                time.sleep(0.02)
                return real_run(args, *a, **kw)
            finally:
                with probe:
                    active -= 1

        monkeypatch.setattr(worktree_module.subprocess, "run", instrumented_run)

        errors: list[BaseException] = []

        def churn(index: int) -> None:
            try:
                child = clone_candidate(parent, f"g1-s{index}", base_dir)
                remove_worktree(child)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=churn, args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert not errors, f"concurrent worktree churn failed: {errors!r}"
        assert max_active == 1, (
            "git worktree common-metadata commands ran concurrently "
            f"(peak={max_active}); they must be serialised"
        )


class TestChangedFiles:
    def test_lists_everything_the_snapshot_will_commit(self, tmp_path: Path) -> None:
        """Committed, unstaged, untracked and deleted paths; ignored ones are not."""
        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        (repo_root / "solve.py").write_text("x = 1\n")
        (repo_root / "old.py").write_text("y = 1\n")
        (repo_root / ".gitignore").write_text("*.log\n")
        _run(["git", "add", "-A"], repo_root)
        _run(["git", "commit", "-m", "files"], repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s1", base_dir)
        wt = Path(child.worktree_path)

        # The agent commits one change itself, then leaves others uncommitted.
        (wt / "solve.py").write_text("x = 2\n")
        _run(["git", "commit", "-am", "agent commit"], wt)
        (wt / "README.md").write_text("# changed\n")
        (wt / "old.py").unlink()
        (wt / "new.py").write_text("z = 1\n")
        (wt / "run.log").write_text("ignored\n")

        assert changed_files(child, seed) == ["README.md", "new.py", "old.py", "solve.py"]

    def test_lists_both_sides_of_a_rename_and_non_ascii_names(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        (repo_root / "a.py").write_text("x = 1\n" * 20)
        _run(["git", "add", "-A"], repo_root)
        _run(["git", "commit", "-m", "a"], repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s1", base_dir)
        wt = Path(child.worktree_path)

        _run(["git", "mv", "a.py", "b.py"], wt)
        _run(["git", "commit", "-m", "rename"], wt)
        (wt / "é.py").write_text("y = 1\n")

        assert changed_files(child, seed) == ["a.py", "b.py", "é.py"]

    def test_unchanged_child_has_no_files(self, tmp_path: Path) -> None:
        repo_root = tmp_path / "project"
        _make_repo(repo_root)
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(repo_root, base_dir)
        child = clone_candidate(seed, "g1-s1", base_dir)
        assert changed_files(child, seed) == []

    def test_missing_worktree_gives_no_files_instead_of_raising(self, tmp_path: Path) -> None:
        gone = Candidate(
            id="g1-s1", worktree_path=str(tmp_path / "gone"), branch_name="helix/g1-s1",
            generation=1, parent_id="g0-s0", parent_ids=["g0-s0"], operation="mutate",
        )
        assert changed_files(gone, gone) == []


def test_snapshot_is_traced_and_changed_files_is_not() -> None:
    """The ``snapshot`` timing span belongs to snapshot_candidate alone."""
    import helix.worktree as worktree

    assert hasattr(worktree.snapshot_candidate, "__wrapped__")
    assert not hasattr(worktree.changed_files, "__wrapped__")


def test_export_copies_the_candidate_without_helix_artifacts(
    tmp_path: Path, monkeypatch
) -> None:
    """``helix best --export`` writes the candidate's committed tree, not the
    prompt, backend logs, batch file and caches HELIX left in its worktree."""
    from types import SimpleNamespace

    from click.testing import CliRunner

    from helix.cli import cli

    for key in ("GIT_AUTHOR_NAME", "GIT_COMMITTER_NAME"):
        monkeypatch.setenv(key, "HELIX Test")
    for key in ("GIT_AUTHOR_EMAIL", "GIT_COMMITTER_EMAIL"):
        monkeypatch.setenv(key, "helix@test.local")
    repo_root = tmp_path / "project"
    _make_repo(repo_root)
    base_dir = tmp_path / "worktrees"
    child = clone_candidate(create_seed_worktree(repo_root, base_dir), "g1-s0", base_dir)
    wt = Path(child.worktree_path)
    (wt / "pkg").mkdir()
    (wt / "pkg" / "solve.py").write_text("x = 1\n")
    os.symlink("pkg/solve.py", wt / "link.py")
    snapshot_candidate(child, "helix: mutate g1-s0")
    # What HELIX and the evaluator leave behind after the snapshot.
    (wt / ".agent_task_prompt.md").write_text("prompt")
    (wt / ".helix_backend_stdout.txt").write_text("log")
    (wt / "helix_batch.json").write_text("[]")
    (wt / ".helix_artifacts" / "backend_transcripts").mkdir(parents=True)
    (wt / "pkg" / "__pycache__").mkdir()
    (wt / "pkg" / "__pycache__" / "solve.pyc").write_bytes(b"\0")

    frontier = SimpleNamespace(
        _candidates={child.id: child}, _results={}, best=lambda: child
    )
    monkeypatch.setattr("helix.cli.load_state", lambda root: object())
    monkeypatch.setattr(
        "helix.cli._reconstruct_frontier", lambda base, state: (frontier, {}, [])
    )
    dest = tmp_path / "export"
    result = CliRunner().invoke(
        cli, ["best", "--dir", str(repo_root), "--export", str(dest)]
    )
    assert result.exit_code == 0, result.output

    exported = sorted(
        str(p.relative_to(dest)) for p in dest.rglob("*") if not p.is_dir()
    )
    # .gitignore holds the HELIX artifact names the seed commits.
    assert exported == [".gitignore", "README.md", "link.py", "pkg/solve.py"]
    assert os.readlink(dest / "link.py") == "pkg/solve.py"
