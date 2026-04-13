"""HELIX git worktree management.

Each evolutionary candidate lives in its own isolated git worktree so that
mutations, evaluations and rollbacks are fully independent.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from helix.exceptions import GitError, HelixError, print_helix_error
from helix.population import Candidate


# ---------------------------------------------------------------------------
# HELIX git identity
# ---------------------------------------------------------------------------


def helix_git_env() -> dict[str, str]:
    """Environment variables that stamp every HELIX-authored git commit.

    Pass ``env={**os.environ, **helix_git_env()}`` to any subprocess.run()
    call that creates a commit inside a HELIX worktree.  The values override
    whatever the ambient git config or GIT_* variables say, keeping the
    authorship scoped to that one subprocess without touching any config file.
    """
    return {
        "GIT_AUTHOR_NAME": "HELIX",
        "GIT_AUTHOR_EMAIL": "helix@noreply",
        "GIT_COMMITTER_NAME": "HELIX",
        "GIT_COMMITTER_EMAIL": "helix@noreply",
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _git_suggestion(args: list[str], cwd: Path | None, stderr: str) -> str:
    """Return a human-readable recovery suggestion based on the failed command and stderr."""
    cmd_str = " ".join(args)
    suggestions: list[str] = []

    # Worktree-related suggestions
    if "worktree" in cmd_str and "add" in cmd_str:
        if "already exists" in stderr.lower():
            suggestions.append(
                "A worktree already exists at this path. "
                "Try: git worktree remove --force <path>, or: git worktree prune"
            )
        if "already checked out" in stderr.lower():
            suggestions.append(
                "This branch is already checked out in another worktree. "
                "Try: git worktree prune, then retry."
            )

    # Branch-related suggestions
    if "branch" in cmd_str or "-b" in args:
        if "already exists" in stderr.lower():
            # Extract branch name from args if possible
            branch_name = ""
            for i, a in enumerate(args):
                if a == "-b" and i + 1 < len(args):
                    branch_name = args[i + 1]
                    break
            if branch_name:
                suggestions.append(
                    f"Stale branch exists. Try: git branch -D {branch_name}, "
                    f"then: git worktree prune"
                )
            else:
                suggestions.append(
                    "Stale branch exists. Try: git worktree prune, "
                    "then delete the stale branch with: git branch -D <branch>"
                )

    # Lock-related suggestions
    if "is locked" in stderr.lower():
        suggestions.append("Worktree is locked. Try: git worktree unlock <path>")

    # Generic worktree prune suggestion
    if "worktree" in cmd_str and not suggestions:
        suggestions.append("Try: git worktree prune, then retry the operation.")

    if not suggestions:
        cwd_str = str(cwd) if cwd else "(unknown)"
        suggestions.append(
            f"Check that the git repository at {cwd_str} is in a clean state. "
            f"Try: git status, git worktree list, git worktree prune"
        )

    return " | ".join(suggestions)


def _run(
    args: list[str],
    cwd: Path | None = None,
    check: bool = True,
    operation: str = "",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a git command, returning the CompletedProcess.

    On failure (when *check* is True), raises a :class:`GitError` with
    the full command, working directory, complete stderr (never truncated),
    and a suggested fix.

    Parameters
    ----------
    env:
        Optional environment dict for the subprocess.  Pass
        ``{**os.environ, **helix_git_env()}`` to stamp a commit with the
        HELIX author/committer identity without modifying any config file.
    """
    try:
        return subprocess.run(
            args,
            cwd=cwd,
            check=check,
            capture_output=True,
            text=True,
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        cmd_str = " ".join(args)
        cwd_str = str(cwd) if cwd else "(not set)"
        suggestion = _git_suggestion(args, cwd, exc.stderr or "")

        raise GitError(
            f"Git command failed: {cmd_str}",
            operation=operation or cmd_str,
            phase="git subprocess",
            command=cmd_str,
            cwd=cwd_str,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            exit_code=exc.returncode,
            suggestion=suggestion,
        ) from exc


def _is_git_repo(path: Path) -> bool:
    """Return True only when *path* itself has a .git entry (not a parent repo).

    We intentionally do NOT traverse upward: HELIX fixture directories that
    live inside a larger repo (e.g. the helix development repo itself) must
    get their own .git initialised by ``_ensure_git_repo`` so that git
    worktree operations are scoped to the fixture root, not the outer repo.
    """
    return (path / ".git").exists()


def _create_initial_gitignore(repo_root: Path) -> None:
    """Create a .gitignore with common noise patterns if it doesn't exist."""
    gitignore_path = repo_root / ".gitignore"

    # Common noise patterns to exclude
    noise_patterns = [
        "# Python",
        "__pycache__/",
        "*.pyc",
        "*.pyo",
        ".pytest_cache/",
        ".mypy_cache/",
        ".hypothesis/",
        "*.egg-info/",
        "",
        "# Build artifacts",
        "build/",
        "dist/",
        "",
        "# Coverage",
        ".coverage",
        "htmlcov/",
        "",
        "# Environment",
        ".env",
        ".venv/",
        "venv/",
        "",
        "# Other",
        "node_modules/",
        ".DS_Store",
    ]

    if not gitignore_path.exists():
        gitignore_path.write_text("\n".join(noise_patterns) + "\n")


def _ensure_git_repo(repo_root: Path) -> None:
    """Initialise a git repo at *repo_root* if it is not already one.

    Always stamps the newly-created repo's local git config with the HELIX
    identity (name="HELIX", email="helix@noreply").  This config is scoped
    to the fixture's own .git directory and never touches the user's global
    git config.
    """
    if _is_git_repo(repo_root):
        return
    _run(["git", "init"], cwd=repo_root, operation="git init")
    # Pin the HELIX identity in the local config of this fresh repo.
    # These settings live inside <repo_root>/.git/config — they are isolated
    # to this directory and have zero effect on the user's global git config.
    _run(
        ["git", "config", "user.name", "HELIX"],
        cwd=repo_root,
        operation="set local user.name",
    )
    _run(
        ["git", "config", "user.email", "helix@noreply"],
        cwd=repo_root,
        operation="set local user.email",
    )
    # Create .gitignore before git add to filter noise files
    _create_initial_gitignore(repo_root)
    _run(["git", "add", "-A"], cwd=repo_root, operation="git add -A (initial)")
    _run(
        ["git", "commit", "-m", "helix: initial seed"],
        cwd=repo_root,
        operation="git commit (initial seed)",
        env={**os.environ, **helix_git_env()},
    )


def _warn_uncommitted_changes(repo_root: Path) -> None:
    """Print a Rich warning if there are uncommitted changes in *repo_root*.

    HELIX snapshots the current working tree into the seed worktree so the
    user can evolve their local edits without creating a temporary commit.
    """
    result = _run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        operation="check uncommitted changes",
    )
    if result.returncode == 0 and result.stdout.strip():
        # Local import avoids any circular-import risk at module load time.
        from helix.display import console  # noqa: PLC0415
        console.print(
            f"[yellow]⚠️  You have uncommitted changes in {repo_root}. "
            f"HELIX will snapshot the current working tree into the seed "
            f"worktree and preserve your original checkout unchanged.[/yellow]"
        )


def _copy_path(src: Path, dst: Path) -> None:
    """Copy a file or symlink into the seed snapshot."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if src.is_symlink():
        os.symlink(os.readlink(src), dst)
    else:
        shutil.copy2(src, dst)


def _snapshot_dirty_working_tree(repo_root: Path, worktree_path: Path) -> bool:
    """Materialize tracked+untracked local changes into *worktree_path*.

    Returns True when any dirty changes were snapshotted, False when the repo
    was already clean relative to HEAD.
    """
    status = _run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        check=False,
        operation="check dirty working tree for seed snapshot",
    )
    if status.returncode != 0 or not status.stdout.strip():
        return False

    diff_proc = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    if diff_proc.stdout:
        subprocess.run(
            ["git", "apply", "--allow-empty", "--whitespace=nowarn", "-"],
            cwd=worktree_path,
            check=True,
            input=diff_proc.stdout,
        )

    untracked_proc = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    for raw_path in untracked_proc.stdout.decode("utf-8", errors="surrogateescape").split("\x00"):
        if not raw_path:
            continue
        rel_path = Path(raw_path)
        if rel_path.parts and rel_path.parts[0] in {".git", ".helix"}:
            continue
        src = repo_root / rel_path
        if not src.exists() and not src.is_symlink():
            continue
        _copy_path(src, worktree_path / rel_path)

    return True


def _check_no_stale_helix_branches(repo_root: Path) -> None:
    """Error out if stale ``helix/*`` branches exist from a previous run.

    Raises :class:`~helix.exceptions.HelixError` listing the stale branches
    and instructing the user to run ``helix clean`` or ``helix resume``.
    """
    result = _run(
        ["git", "branch", "--list", "helix/*"],
        cwd=repo_root,
        check=False,
        operation="check stale helix branches",
    )
    if result.returncode != 0:
        return  # git not available or not a repo yet — skip
    branches = [b.strip().lstrip("* ") for b in result.stdout.splitlines() if b.strip()]
    if not branches:
        return
    n = len(branches)
    branch_list = ", ".join(branches[:5])
    if n > 5:
        branch_list += f", … ({n - 5} more)"
    raise HelixError(
        f"Found {n} existing helix/* branch(es) from a previous run: {branch_list}. "
        f"Run `helix clean` first, or `helix resume` to continue it.",
        operation="pre-flight helix branch check",
        suggestion=(
            "Run `helix clean` to remove all previous HELIX state and start fresh, "
            "or `helix resume` to continue the previous run."
        ),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_empty_seed_worktree(repo_root: Path, base_dir: Path) -> Candidate:
    """Bootstrap an empty worktree for seedless mode.

    Creates a fresh git repository at *repo_root* (if it does not already
    exist) with a single empty initial commit, then adds a detached-HEAD
    worktree at ``<base_dir>/g0-s0``.  The worktree starts with **no
    project files** — Claude Code will write the first candidate from
    scratch based on the objective prompt.

    Parameters
    ----------
    repo_root:
        Directory that will become (or already is) the git repository root.
    base_dir:
        Directory under which all HELIX worktrees are stored.

    Returns
    -------
    Candidate
        The seed candidate (id="g0-s0", generation=0, parent=None).
    """
    # Initialise a bare git repo with an empty initial commit.
    if not _is_git_repo(repo_root):
        repo_root.mkdir(parents=True, exist_ok=True)
        _run(["git", "init"], cwd=repo_root, operation="git init (seedless)")
        # Pin the HELIX identity in the local config of this fresh repo.
        _run(
            ["git", "config", "user.name", "HELIX"],
            cwd=repo_root,
            operation="set local user.name (seedless)",
        )
        _run(
            ["git", "config", "user.email", "helix@noreply"],
            cwd=repo_root,
            operation="set local user.email (seedless)",
        )
        # Create an empty initial commit so HEAD is valid.
        _run(
            ["git", "commit", "--allow-empty", "-m", "helix: empty seed (seedless mode)"],
            cwd=repo_root,
            operation="git commit --allow-empty (seedless)",
            env={**os.environ, **helix_git_env()},
        )

    # Prune stale worktree registrations before creating new ones.
    subprocess.run(["git", "worktree", "prune"], cwd=repo_root, check=False)

    seed_id = "g0-s0"
    worktree_path = base_dir / seed_id
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run(
            ["git", "worktree", "add", str(worktree_path), "--detach", "HEAD"],
            cwd=repo_root,
            operation=f"create empty seed worktree at {worktree_path}",
        )
    except GitError as exc:
        exc.suggestion = (
            f"Could not create empty seed worktree at {worktree_path}. "
            f"Try: git worktree prune && rm -rf {worktree_path}, then retry."
        )
        print_helix_error(exc)
        raise

    return Candidate(
        id=seed_id,
        worktree_path=str(worktree_path),
        branch_name=f"helix/{seed_id}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def create_seed_worktree(repo_root: Path, base_dir: Path) -> Candidate:
    """Bootstrap the very first evolutionary candidate.

    If *repo_root* is not yet a git repository it will be initialised and an
    initial commit created.  A detached-HEAD worktree is then added at
    ``<base_dir>/g0-s0``.

    Parameters
    ----------
    repo_root:
        Root of the project whose code will be evolved.
    base_dir:
        Directory under which all HELIX worktrees are stored.

    Returns
    -------
    Candidate
        The seed candidate (generation 0, slot 0).
    """
    _ensure_git_repo(repo_root)

    # Warn when local edits exist so the user knows HELIX will snapshot them.
    _warn_uncommitted_changes(repo_root)

    # Fix 6: Abort early if stale helix/* branches exist from a previous run.
    _check_no_stale_helix_branches(repo_root)

    # Prune stale worktree registrations before creating new ones
    subprocess.run(["git", "worktree", "prune"], cwd=repo_root, check=False)

    seed_id = "g0-s0"
    worktree_path = base_dir / seed_id
    base_dir.mkdir(parents=True, exist_ok=True)

    try:
        _run(
            ["git", "worktree", "add", str(worktree_path), "--detach", "HEAD"],
            cwd=repo_root,
            operation=f"create seed worktree at {worktree_path}",
        )
    except GitError as exc:
        exc.suggestion = (
            f"Could not create seed worktree at {worktree_path}. "
            f"Try: git worktree prune && rm -rf {worktree_path}, then retry. "
            f"If the path already exists as a worktree, also try: "
            f"git worktree remove --force {worktree_path}"
        )
        print_helix_error(exc)
        raise

    candidate = Candidate(
        id=seed_id,
        worktree_path=str(worktree_path),
        branch_name=f"helix/{seed_id}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )
    if _snapshot_dirty_working_tree(repo_root, worktree_path):
        snapshot_candidate(candidate, "helix: snapshot dirty seed")
    return candidate


def clone_candidate(parent: Candidate, new_id: str, base_dir: Path) -> Candidate:
    """Create a new worktree branched from *parent*.

    A new branch ``helix/<new_id>`` is created at the tip of the parent's
    branch and a new worktree is checked out there.

    Parameters
    ----------
    parent:
        The candidate to clone from.
    new_id:
        Identifier for the new candidate (e.g. ``"g1-s0"``).
    base_dir:
        Root directory for HELIX worktrees.

    Returns
    -------
    Candidate
        The cloned candidate with its own isolated worktree.
    """
    new_branch = f"helix/{new_id}"
    new_worktree_path = base_dir / new_id
    base_dir.mkdir(parents=True, exist_ok=True)

    # Determine the repo root from the existing worktree
    result = _run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=Path(parent.worktree_path),
        operation=f"resolve repo root from {parent.worktree_path}",
    )
    # --git-common-dir returns the .git directory of the main worktree
    git_common = result.stdout.strip()
    if git_common.endswith("/.git"):
        repo_root = Path(git_common[: -len("/.git")])
    elif git_common == ".git":
        repo_root = Path(parent.worktree_path)
    else:
        # bare or unusual layout — use the parent worktree itself as cwd
        repo_root = Path(parent.worktree_path)

    # Get the commit to branch from (tip of parent branch or HEAD of worktree)
    head_result = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(parent.worktree_path),
        operation=f"resolve HEAD for parent {parent.id}",
    )
    commit_sha = head_result.stdout.strip()

    # Prune stale worktree registrations (safe — only removes metadata for missing paths)
    subprocess.run(["git", "worktree", "prune"], cwd=repo_root, check=False)

    try:
        _run(
            [
                "git", "worktree", "add",
                str(new_worktree_path),
                "-b", new_branch,
                commit_sha,
            ],
            cwd=repo_root,
            operation=f"clone candidate {parent.id} -> {new_id}",
        )
    except GitError as exc:
        exc.suggestion = (
            f"Could not create worktree for {new_id} (branch {new_branch}). "
            f"If a stale branch exists, try: git branch -D {new_branch} && "
            f"git worktree prune. "
            f"If the worktree path already exists, try: "
            f"git worktree remove --force {new_worktree_path} && "
            f"git worktree prune. "
            f"Then retry the evolution."
        )
        print_helix_error(exc)
        raise

    # Parse generation from new_id (format: g<gen>-s<slot>)
    try:
        gen = int(new_id.split("-")[0].lstrip("g"))
    except (IndexError, ValueError):
        gen = 0

    return Candidate(
        id=new_id,
        worktree_path=str(new_worktree_path),
        branch_name=new_branch,
        generation=gen,
        parent_id=parent.id,
        parent_ids=[parent.id],
        operation="clone",
    )


def snapshot_candidate(candidate: Candidate, message: str) -> str:
    """Stage all changes in the candidate's worktree and create a commit.

    Parameters
    ----------
    candidate:
        The candidate whose worktree should be snapshotted.
    message:
        Commit message.

    Returns
    -------
    str
        The full SHA of the newly created commit.
    """
    wt = Path(candidate.worktree_path)
    _run(
        ["git", "add", "-A"],
        cwd=wt,
        operation=f"snapshot {candidate.id}: git add",
    )
    # Check if there is anything to commit — skip if worktree is clean
    diff_check = _run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=wt,
        check=False,
        operation=f"snapshot {candidate.id}: check staged diff",
    )
    if diff_check.returncode == 0:
        # Nothing staged — return current HEAD SHA without creating an empty commit
        result = _run(
            ["git", "rev-parse", "HEAD"],
            cwd=wt,
            operation=f"snapshot {candidate.id}: resolve HEAD (no changes)",
        )
        return result.stdout.strip()
    _run(
        ["git", "commit", "-m", message],
        cwd=wt,
        operation=f"snapshot {candidate.id}: git commit",
        env={**os.environ, **helix_git_env()},
    )
    result = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=wt,
        operation=f"snapshot {candidate.id}: resolve HEAD",
    )
    return result.stdout.strip()


def remove_worktree(candidate: Candidate) -> None:
    """Remove a candidate's worktree and its tracking branch.

    Parameters
    ----------
    candidate:
        The candidate to remove.
    """
    wt = Path(candidate.worktree_path)

    # Find the repo root via git-common-dir so we can run worktree commands
    result = _run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=wt,
        check=False,
        operation=f"remove worktree {candidate.id}: resolve repo root",
    )
    if result.returncode == 0:
        git_common = result.stdout.strip()
        if git_common.endswith("/.git"):
            repo_root = Path(git_common[: -len("/.git")])
        elif git_common == ".git":
            repo_root = wt
        else:
            repo_root = wt
    else:
        repo_root = wt

    _run(
        ["git", "worktree", "remove", "--force", str(wt)],
        cwd=repo_root,
        operation=f"remove worktree {candidate.id}",
    )

    # Delete the tracking branch; ignore errors (branch may not exist)
    branch = f"helix/{candidate.id}"
    _run(
        ["git", "branch", "-D", branch],
        cwd=repo_root,
        check=False,
        operation=f"delete branch {branch}",
    )


def get_diff(candidate_a: Candidate, candidate_b: Candidate) -> str:
    """Return the unified diff between two candidates' branch tips.

    Parameters
    ----------
    candidate_a:
        Base candidate (left side of diff).
    candidate_b:
        Target candidate (right side of diff).

    Returns
    -------
    str
        Output of ``git diff <branch_a> <branch_b>``.
    """
    wt_a = Path(candidate_a.worktree_path)
    result = _run(
        ["git", "diff", candidate_a.branch_name, candidate_b.branch_name],
        cwd=wt_a,
        check=False,
        operation=f"diff {candidate_a.id} vs {candidate_b.id}",
    )
    return result.stdout
