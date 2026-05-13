"""Protected-evaluator-file SHA-256 manifest for tamper detection.

This module owns two closely-related responsibilities:

1. **Protected-file refresh** — copy the project's evaluator script(s) and
   explicitly listed protected files into candidate worktrees so mutations
   cannot alter them (functions ``_collect_protected_evaluator_paths``,
   ``_copy_protected_path``, ``_refresh_protected_evaluator_files``,
   ``_refresh_and_snapshot_protected_evaluator_files``).

2. **Integrity manifest** — build a ``{repo_relative_path: sha256}`` map from
   the seed worktree and detect changes in mutation/merge candidates
   (functions ``_sha256_file`` … ``_detect_evaluator_tamper``).

These helpers have no dependency on ``EvolutionState``, ``ParetoFrontier``, or
the evolution loop.  They operate only on ``Path``, ``HelixConfig``, and
``Candidate.worktree_path``, so they can be unit-tested in isolation without
constructing any evolution-loop state.

Used for tamper detection during sandboxed mutation: before any mutation or
merge candidate is evaluated, :func:`_detect_evaluator_tamper` compares each
file in the manifest (seeded from the baseline worktree) against the
candidate's worktree to catch modifications, deletions, or additions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import shlex
from pathlib import Path

from helix.config import HelixConfig
from helix.exceptions import HelixError
from helix.population import Candidate
from helix.worktree import snapshot_candidate

logger = logging.getLogger(__name__)

# NOTE: ``shutil.ignore_patterns`` uses ``fnmatch`` *basename* matching, not
# gitignore semantics.  Add basename globs here only -- patterns like
# ``**/build/`` or ``cache/**`` will not match.  If we ever need gitignore
# semantics, switch to ``pathspec`` rather than extending this tuple.
_PROTECTED_DIRECTORY_IGNORE_PATTERNS = (
    ".git",
    "__pycache__",
    "*.pyc",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
)
# Reusable matcher; the patterns are constant so we build it once.
_PROTECTED_DIRECTORY_IGNORE_MATCHER = shutil.ignore_patterns(
    *_PROTECTED_DIRECTORY_IGNORE_PATTERNS
)

_NO_SCRIPT_COMMANDS = {"make", "pytest"}
_INTERPRETERS = {"python", "python3", "uv", "poetry", "node", "bash", "sh"}
_SKIP_TOKENS = {"run"}
_SCRIPT_SUFFIXES = (".py", ".sh", ".js", ".ts")
# Shell wrappers whose command body (after -c/-lc/...) is opaque to path-level
# validation — e.g. `bash -lc "cd /x && python evaluate.py"`.
# Note: only the adjacent `{wrapper} {flag}` prefix is exempted. Forms like
# `bash --login -c "..."` (separate tokens) fall through to the normal
# script-path checks by design — extend _SHELL_COMMAND_FLAGS if that becomes
# a real-world need.
_SHELL_WRAPPERS = {"bash", "sh", "zsh", "fish", "dash"}
_SHELL_COMMAND_FLAGS = {"-c", "-lc", "-ic", "-ilc", "-lic"}
_EVALUATOR_MANIFEST_FILENAME = "evaluator_manifest.json"


def _extract_script_token(tokens: list[str]) -> str | None:
    """Return the most likely script token from a tokenized command."""
    skip_next = False
    for token in tokens:
        if skip_next:
            skip_next = False
            continue
        if token == "-m":
            # `python -m module` has no script path token.
            skip_next = True
            continue
        if token in _INTERPRETERS or token in _SKIP_TOKENS:
            continue
        if token.startswith("-"):
            continue
        return token
    return None


def _looks_like_script_file(path_token: str) -> bool:
    """Heuristic for script-like file paths used by evaluator commands."""
    if path_token.endswith("/"):
        return False
    return any(path_token.endswith(ext) for ext in _SCRIPT_SUFFIXES)


def _to_repo_relative(path_token: str, project_root: Path) -> str | None:
    """Normalize a path token to a repo-relative POSIX path when possible."""
    project_root_resolved = project_root.resolve()
    token_path = Path(path_token)
    abs_path = (
        token_path.resolve()
        if token_path.is_absolute()
        else (project_root_resolved / token_path).resolve()
    )
    try:
        return abs_path.relative_to(project_root_resolved).as_posix()
    except ValueError:
        return None


def _collect_protected_evaluator_paths(
    config: HelixConfig, project_root: Path
) -> list[str]:
    """Collect repo-relative files that should stay immutable during evolution."""
    protected: set[str] = set()

    for cmd in [config.evaluator.command, *config.evaluator.extra_commands]:
        try:
            tokens = shlex.split(cmd)
        except ValueError:
            continue
        if not tokens or tokens[0] in _NO_SCRIPT_COMMANDS:
            continue
        # Shell wrappers like `bash -c "..."` hide the real command inside an
        # opaque body string; path-level validation cannot reason about it.
        if (
            tokens[0] in _SHELL_WRAPPERS
            and len(tokens) >= 2
            and tokens[1] in _SHELL_COMMAND_FLAGS
        ):
            continue
        script_token = _extract_script_token(tokens)
        if script_token is None or not _looks_like_script_file(script_token):
            continue
        rel = _to_repo_relative(script_token, project_root)
        if rel is not None:
            protected.add(rel)

    for path_str in config.evaluator.protected_files:
        rel = _to_repo_relative(path_str, project_root)
        if rel is None:
            raise HelixError(
                f"evaluator.protected_files path is outside project root: {path_str}",
                operation="resolve protected evaluator files",
                suggestion="Use repo-relative paths under the project root.",
            )
        protected.add(rel)

    return sorted(protected)


def _copy_protected_path(source: Path, destination: Path) -> None:
    """Refresh one protected file/directory in a candidate worktree."""
    if source.resolve(strict=False) == destination.resolve(strict=False):
        return

    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir() and not source.is_symlink():
        shutil.copytree(
            source,
            destination,
            ignore=_PROTECTED_DIRECTORY_IGNORE_MATCHER,
        )
    elif source.is_symlink():
        os.symlink(os.readlink(source), destination)
    else:
        shutil.copy2(source, destination)


def _refresh_protected_evaluator_files(
    candidate: Candidate,
    config: HelixConfig,
    project_root: Path,
) -> None:
    """Copy current root protected evaluator/runtime files into a worktree."""
    worktree_root = Path(candidate.worktree_path)
    for rel_path in _collect_protected_evaluator_paths(config, project_root):
        source = project_root / rel_path
        if not source.exists() and not source.is_symlink():
            continue
        _copy_protected_path(source, worktree_root / rel_path)


def _refresh_and_snapshot_protected_evaluator_files(
    candidate: Candidate,
    config: HelixConfig,
    project_root: Path,
) -> None:
    """Normalize HELIX-owned protected-file refresh before backend mutation."""
    _refresh_protected_evaluator_files(candidate, config, project_root)
    snapshot_candidate(candidate, "helix: refresh protected evaluator files")


def _sha256_file(path: Path) -> str:
    """Streamed SHA-256 of ``path``.

    Uses ``hashlib.file_digest`` (Python 3.11+) so large protected files
    (e.g., evaluator dataset ``.jsonl`` files) are not buffered in memory.
    """
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def _iter_protected_manifest_files(
    source_path: Path,
    rel_path: str,
) -> list[tuple[str, Path]]:
    """Return manifest entries under one protected file/directory.

    ``source_path`` is expected to already be a real path (callers resolve
    symlinks before calling).  ``os.walk`` is used with ``followlinks=False``
    (the default) so symlinks inside the directory are not traversed.
    """
    if source_path.is_file():
        return [(rel_path, source_path)]
    if not source_path.is_dir():
        return []

    entries: list[tuple[str, Path]] = []
    for dirpath_str, dirnames, filenames in os.walk(source_path):
        ignored = _PROTECTED_DIRECTORY_IGNORE_MATCHER(
            dirpath_str, [*dirnames, *filenames]
        )
        dirnames[:] = sorted(name for name in dirnames if name not in ignored)
        for filename in sorted(name for name in filenames if name not in ignored):
            file_path = Path(dirpath_str) / filename
            # Defensive against TOCTOU: ``os.walk`` already separated files
            # from directories, but a concurrent unlink/rename could leave a
            # stale name behind.
            if not file_path.is_file():
                continue
            entry_path = Path(rel_path) / file_path.relative_to(source_path)
            entries.append((entry_path.as_posix(), file_path))
    return entries


def _evaluator_manifest_path(base_dir: Path) -> Path:
    return base_dir / _EVALUATOR_MANIFEST_FILENAME


def _build_evaluator_integrity_manifest(
    config: HelixConfig,
    baseline_root: Path,
    project_root: Path,
) -> dict[str, str]:
    """Build {repo_relative_path: sha256} for protected evaluator files.

    Fails closed: an unreadable protected file raises ``HelixError`` rather
    than silently being omitted from the manifest -- otherwise tamper
    detection would not flag changes to that file on resume.
    """
    manifest: dict[str, str] = {}
    for rel_path in _collect_protected_evaluator_paths(config, project_root):
        source_path = (baseline_root / rel_path).resolve()
        if not source_path.exists():
            logger.warning(
                "Skipping protected evaluator path %s: missing from baseline %s",
                rel_path,
                baseline_root,
            )
            continue
        entries = _iter_protected_manifest_files(source_path, rel_path)
        if not entries and source_path.is_dir():
            # Surface likely misconfigurations (e.g., a protected directory
            # whose only contents match the ignore list).
            logger.warning(
                "Protected evaluator directory %s contributed 0 manifest "
                "entries (every file matched the ignore list).",
                rel_path,
            )
        for manifest_rel_path, file_path in entries:
            try:
                manifest[manifest_rel_path] = _sha256_file(file_path)
            except OSError as exc:
                raise HelixError(
                    f"Failed hashing protected evaluator file: {file_path}",
                    operation="build evaluator integrity manifest",
                    suggestion=(
                        "Ensure the file is readable and not held open by "
                        "another process before retrying."
                    ),
                ) from exc
    return manifest


def _write_evaluator_integrity_manifest(
    base_dir: Path, manifest: dict[str, str]
) -> None:
    """Persist immutable evaluator manifest for resume."""
    path = _evaluator_manifest_path(base_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 1, "files": manifest}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _load_evaluator_integrity_manifest(base_dir: Path) -> dict[str, str] | None:
    """Load persisted immutable evaluator manifest, if available."""
    path = _evaluator_manifest_path(base_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        logger.exception("Failed to read evaluator integrity manifest: %s", path)
        return None
    files = payload.get("files")
    if not isinstance(files, dict):
        return None
    manifest: dict[str, str] = {}
    for key, value in files.items():
        if isinstance(key, str) and isinstance(value, str):
            manifest[key] = value
    return manifest


def _detect_evaluator_tamper(
    candidate: Candidate,
    manifest: dict[str, str],
    config: HelixConfig | None = None,
    project_root: Path | None = None,
) -> list[str]:
    """Return protected paths that diverge from the frozen evaluator manifest.

    Detects three classes of tamper:

    1. Modification of a baseline-listed file (hash mismatch).
    2. Deletion of a baseline-listed file.
    3. *Addition* of a previously-unknown file inside a protected directory,
       when ``config`` and ``project_root`` are supplied.  Without these
       arguments only the first two classes are reported, preserving the
       legacy contract for callers that don't have config in scope.
    """
    if not manifest:
        return []
    violations: set[str] = set()
    worktree_root = Path(candidate.worktree_path)
    for rel_path, expected_hash in manifest.items():
        candidate_path = worktree_root / rel_path
        if not candidate_path.exists() or not candidate_path.is_file():
            violations.add(rel_path)
            continue
        try:
            if _sha256_file(candidate_path) != expected_hash:
                violations.add(rel_path)
        except OSError:
            violations.add(rel_path)

    if config is not None and project_root is not None:
        known = set(manifest)
        for protected_rel in _collect_protected_evaluator_paths(
            config, project_root
        ):
            candidate_path = worktree_root / protected_rel
            # Only directories can hold *new* files; single-file protected
            # paths are fully covered by the manifest loop above.
            if not candidate_path.is_dir() or candidate_path.is_symlink():
                continue
            for entry_rel, _ in _iter_protected_manifest_files(
                candidate_path, protected_rel
            ):
                if entry_rel not in known:
                    violations.add(entry_rel)
    return sorted(violations)
