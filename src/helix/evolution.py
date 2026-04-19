"""HELIX main evolution loop."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import random as _random
import shlex
import threading
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from helix.batch_sampler import BatchSampler, EpochShuffledBatchSampler, StratifiedBatchSampler
from helix.config import HelixConfig, load_dataset_examples
from helix.eval_cache import EvaluationCache as MinibatchEvalCache
from helix.eval_policy import (
    FullEvaluationPolicy,
    ImprovementOrEqualAcceptance,
    StrictImprovementAcceptance,
)
from helix.display import (
    HelixPhase,
    console,
    print_error,
    print_info,
    print_success,
    print_warning,
    render_budget,
    render_generation,
    set_phase,
)
from helix.exceptions import HelixError, RateLimitError, print_helix_error
from helix.executor import run_evaluator
from helix.lineage import LineageEntry, find_merge_triplet, load_lineage, record_entry
from helix.merger import merge, select_eval_subsample_for_merged_program
from helix.mutator import mutate, build_seed_generation_prompt, generate_seed
from helix.population import Candidate, EvalResult, ParetoFrontier
from helix.state import (
    BudgetState,
    EvaluationCache,
    EvolutionState,
    load_eval_cache,
    load_state,
    save_eval_cache,
    save_state,
)
from helix.trace import TRACE, EventType
from helix.worktree import create_seed_worktree, create_empty_seed_worktree, remove_worktree, snapshot_candidate


# ---------------------------------------------------------------------------
# HelixProgress — Rich progress bar for the generation loop
# ---------------------------------------------------------------------------


class _BestScoreColumn(ProgressColumn):
    """Renders the current best Pareto-frontier aggregate score."""

    def render(self, task: Task) -> Text:
        score_obj = task.fields.get("best_score", None)
        if score_obj is None or not isinstance(score_obj, (int, float)):
            return Text("best=n/a", style="dim")
        score = float(score_obj)
        if math.isnan(score):
            return Text("best=n/a", style="dim")
        return Text(f"best={score:.4f}", style="bold green")


class HelixProgress:
    """Rich progress bar context manager for the HELIX evolution loop.

    Shows per-generation progress with elapsed time, ETA, and the current
    best aggregate score on the Pareto frontier.

    Usage::

        with HelixProgress(max_generations=cfg.evolution.max_generations) as prog:
            for gen in range(start_gen, max_generations + 1):
                ...
                prog.update(gen, best_score=frontier_best_score)

    The bar is automatically disabled when the ``HELIX_NO_PROGRESS``
    environment variable is set to any non-empty string (useful in tests and
    non-interactive CI pipelines).
    """

    def __init__(self, max_generations: int) -> None:
        self._max = max_generations
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        # Disable when HELIX_NO_PROGRESS is set to any non-empty string.
        self._enabled: bool = not bool(os.environ.get("HELIX_NO_PROGRESS", ""))

    # ------------------------------------------------------------------
    # Context-manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "HelixProgress":
        if self._enabled:
            self._progress = Progress(
                TextColumn("[bold blue]Generation[/bold blue]"),
                MofNCompleteColumn(),
                BarColumn(bar_width=None),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                _BestScoreColumn(),
                console=console,
                transient=False,
                expand=True,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                "evolving",
                total=self._max,
                completed=0,
                best_score=float("nan"),
            )
        return self

    def __exit__(self, *args: object) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None
            self._task_id = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True when the Rich live display is running."""
        return self._progress is not None

    def update(self, generation: int, best_score: float) -> None:
        """Advance the bar to *generation* and display *best_score*."""
        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=generation,
                best_score=best_score,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def budget_exhausted(state: EvolutionState, config: HelixConfig) -> bool:
    """Return True if evaluation budget is exhausted.

    GEPA parity (C1): metric_calls was dead code (never incremented but
    checked).  Budget now uses only the ``evaluations`` counter, matching
    GEPA's ``total_num_evals`` budget.
    """
    return state.budget.evaluations >= config.evolution.max_metric_calls


def degrades(new_result: EvalResult, baseline: EvalResult, threshold: float) -> bool:
    """Return True if the new result regresses below baseline by more than threshold."""
    return new_result.sum_score() < baseline.sum_score() - threshold


def _config_hash(config: HelixConfig) -> str:
    data = config.model_dump_json()
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def init_base_dir(base_dir: Path, config: HelixConfig) -> None:
    """Create HELIX working directories and snapshot config to base_dir/config.toml."""
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "worktrees").mkdir(parents=True, exist_ok=True)
    (base_dir / "evaluations").mkdir(parents=True, exist_ok=True)

    config_path = base_dir / "config.toml"
    if not config_path.exists():
        lines = [
            "# HELIX config snapshot (auto-generated)\n",
            f'objective = {json.dumps(config.objective)}\n',
            "\n[evaluator]\n",
            f'command = {json.dumps(config.evaluator.command)}\n',
        ]
        config_path.write_text("".join(lines))


def _save_evaluation(base_dir: Path, result: EvalResult) -> None:
    """Persist an EvalResult to evaluations/<candidate_id>.json."""
    eval_dir = base_dir / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "candidate_id": result.candidate_id,
        "scores": result.scores,
        "instance_scores": result.instance_scores,
        "asi": result.asi,
    }
    (eval_dir / f"{result.candidate_id}.json").write_text(json.dumps(data, indent=2))


def _load_evaluation(base_dir: Path, candidate_id: str) -> EvalResult | None:
    """Load a saved EvalResult, or None if the file is absent."""
    path = base_dir / "evaluations" / f"{candidate_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return EvalResult(
        candidate_id=data["candidate_id"],
        scores=data["scores"],
        instance_scores=data["instance_scores"],
        asi=data.get("asi", {}),
    )


def _gen_from_id(candidate_id: str) -> int:
    """Parse the generation number from a candidate id like 'g3-s1'."""
    try:
        return int(candidate_id.split("-")[0].lstrip("g"))
    except (IndexError, ValueError):
        return 0


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluator_manifest_path(base_dir: Path) -> Path:
    return base_dir / _EVALUATOR_MANIFEST_FILENAME


def _build_evaluator_integrity_manifest(
    config: HelixConfig,
    baseline_root: Path,
    project_root: Path,
) -> dict[str, str]:
    """Build {repo_relative_path: sha256} for protected evaluator files."""
    manifest: dict[str, str] = {}
    for rel_path in _collect_protected_evaluator_paths(config, project_root):
        source_path = (baseline_root / rel_path).resolve()
        if not source_path.exists() or not source_path.is_file():
            logger.warning(
                "Skipping protected evaluator file %s: missing from baseline %s",
                rel_path,
                baseline_root,
            )
            continue
        try:
            manifest[rel_path] = _sha256_file(source_path)
        except OSError:
            logger.exception("Failed hashing protected evaluator file: %s", source_path)
    return manifest


def _write_evaluator_integrity_manifest(base_dir: Path, manifest: dict[str, str]) -> None:
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
) -> list[str]:
    """Return protected paths that diverge from the frozen evaluator manifest."""
    if not manifest:
        return []
    violations: list[str] = []
    worktree_root = Path(candidate.worktree_path)
    for rel_path, expected_hash in manifest.items():
        candidate_path = worktree_root / rel_path
        if not candidate_path.exists() or not candidate_path.is_file():
            violations.append(rel_path)
            continue
        try:
            if _sha256_file(candidate_path) != expected_hash:
                violations.append(rel_path)
        except OSError:
            violations.append(rel_path)
    return sorted(violations)


def _check_evaluator_script_exists(
    evaluator_command: str, project_root: Path
) -> None:
    """Validate that the evaluator script exists before starting evolution.

    Args:
        evaluator_command: The evaluator command from config.
        project_root: Root of the project being evolved.

    Raises:
        SystemExit: If the evaluator script is not found.
    """
    try:
        tokens = shlex.split(evaluator_command)
    except ValueError as e:
        print_error(
            f"Error: Failed to parse evaluator command: {evaluator_command}\n"
            f"Parse error: {e}\n"
            "Check the evaluator.command in helix.toml."
        )
        raise SystemExit(1)

    if not tokens:
        print_error(
            "Error: Evaluator command is empty.\n"
            "Check the evaluator.command in helix.toml."
        )
        raise SystemExit(1)

    # A shell wrapper like `bash -c "..."` hides the real command inside an
    # opaque body string; path-level validation cannot reason about it.
    if (
        tokens[0] in _SHELL_WRAPPERS
        and len(tokens) >= 2
        and tokens[1] in _SHELL_COMMAND_FLAGS
    ):
        return

    # If the first token is a command that doesn't need a script, allow it
    if tokens[0] in _NO_SCRIPT_COMMANDS:
        return

    script_path = _extract_script_token(tokens)

    if script_path is None:
        # No script found - allow it (might be a command-only invocation)
        return

    # If the token looks like a directory or doesn't look like a script, skip validation
    # (e.g., "pytest tests/" where tests/ is a directory)
    if not _looks_like_script_file(script_path):
        # Not a script file pattern - allow it
        return

    # Resolve the script path relative to project_root
    script_file = project_root / script_path
    if not script_file.exists():
        print_error(
            f"Error: Evaluator script not found: {script_path}\n"
            f"Resolved path: {script_file}\n"
            "Check the evaluator.command in helix.toml.\n"
            "The script path should be relative to the project root."
        )
        raise SystemExit(1)

    if not script_file.is_file():
        print_error(
            f"Error: Evaluator script path exists but is not a file: {script_path}\n"
            f"Resolved path: {script_file}\n"
            "Check the evaluator.command in helix.toml."
        )
        raise SystemExit(1)


# ---------------------------------------------------------------------------
# HelixDataLoader — minimal in-memory loader exposing example ids
# ---------------------------------------------------------------------------


class HelixDataLoader:
    """Minimal dataset loader for HELIX minibatch sampling.

    Wraps a dataset file/directory (same format as
    :func:`helix.config.load_dataset_examples`) and exposes the stable
    list of example ids used by :class:`EpochShuffledBatchSampler`.

    Layouts supported:
      * **JSON array**: ids are the stringified indices ``"0"``, ``"1"``, …
      * **JSONL**:      ids are the stringified indices ``"0"``, ``"1"``, …
      * **Directory** of ``*.json``: ids are the file stems (sorted).
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._ids: list[str] = _load_dataset_ids(path)

    def all_ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


def _load_dataset_ids(path: Path) -> list[str]:
    """Return the stable list of example ids for *path*.

    Raises ValueError if *path* cannot be parsed.  Empty datasets yield
    an empty list (callers must handle this — the minibatch sampler
    rejects empty loaders).
    """
    if not path.exists():
        raise ValueError(f"dataset path does not exist: {path}")

    if path.is_dir():
        return [p.stem for p in sorted(path.glob("*.json"))]

    raw = path.read_text().strip()
    if not raw:
        return []
    if raw.startswith("["):
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(
                f"dataset JSON file does not contain a top-level array: {path}"
            )
        return [str(i) for i in range(len(data))]
    # JSONL
    count = 0
    for line in raw.splitlines():
        if line.strip():
            count += 1
    return [str(i) for i in range(count)]


def _make_data_loader(path: Path | None) -> HelixDataLoader | None:
    """Construct a HelixDataLoader or return None if *path* is None.

    Also returns None when the resulting loader would be empty — the
    caller then falls back to the single-task / no-minibatch code path.
    """
    if path is None:
        return None
    loader = HelixDataLoader(path)
    if len(loader) == 0:
        return None
    return loader


class _RangeDataLoader:
    """Synthetic loader for Architecture A positional-index handoff.

    Exposes ids ``["0", "1", ..., str(size-1)]`` — no underlying
    payload is loaded.  The evaluator (running in the worktree) is
    responsible for loading its own dataset and filtering by the
    integer indices written to ``helix_batch.json``.
    """

    def __init__(self, size: int) -> None:
        if size < 0:
            raise ValueError(f"_RangeDataLoader size must be >= 0 (got {size})")
        self._size = size
        self._ids: list[str] = [str(i) for i in range(size)]

    def all_ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return self._size


def _write_helix_batch(worktree_path: str | Path, indices: list[int]) -> None:
    """Write positional indices to ``{worktree}/helix_batch.json``.

    Side-channel handoff to the evaluator (Architecture A).  The
    evaluator, when run with cwd=worktree_path, reads this file and
    filters its dataset by the supplied indices.
    """
    path = Path(worktree_path) / "helix_batch.json"
    try:
        path.write_text(json.dumps([int(i) for i in indices]))
    except FileNotFoundError:
        # Worktree directory does not exist (e.g. under unit-test mocks
        # that fabricate fake paths).  Silently skip — production paths
        # always have the worktree on disk before eval.
        logger.debug("worktree %s missing; skipping helix_batch.json write", worktree_path)


# Per-worktree lock registry.  Used to serialize ``_write_helix_batch`` +
# ``run_evaluator`` calls that share the same worktree path.  GEPA calls
# ``adapter.evaluate`` in-process so has no file-handoff race (see GEPA
# core/engine.py:381-452); HELIX's Architecture A writes per-batch indices
# to ``{worktree}/helix_batch.json`` before subprocess launch, so concurrent
# parallel parent-evals on the same worktree would clobber each other's
# batch file.  Different worktrees may evaluate concurrently.
_WORKTREE_LOCKS: dict[str, threading.Lock] = {}
_WORKTREE_LOCKS_MUTEX = threading.Lock()


def _worktree_lock(worktree_path: str | Path) -> threading.Lock:
    key = str(worktree_path)
    with _WORKTREE_LOCKS_MUTEX:
        lock = _WORKTREE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _WORKTREE_LOCKS[key] = lock
        return lock


def _cached_eval(
    candidate: Candidate,
    config: HelixConfig,
    split: str,
    cache: EvaluationCache,
) -> tuple[EvalResult, bool]:
    """Run evaluator with cache.  Returns (result, was_cached).

    GEPA parity (Fix 11): avoid re-evaluating identical (candidate_id, split)
    pairs.  Only genuinely new evaluations should consume budget.
    """
    cached = cache.get(candidate.id, split)
    if cached is not None:
        return EvalResult.from_dict(cached), True
    result = run_evaluator(candidate, config, split=split)
    cache.put(candidate.id, split, result.to_dict())
    return result, False


def _cached_evaluate_batch(
    candidate: Candidate,
    example_ids: list[str],
    cache: "MinibatchEvalCache[object, str] | None",
    config: HelixConfig,
    split: str,
) -> tuple[EvalResult, int]:
    """Evaluate ``candidate`` on ``example_ids`` with per-example caching.

    GEPA parity: line-for-line mirror of GEPA's ``cached_evaluate_full``
    (``gepa/core/state.py:618-633`` → ``EvaluationCache.evaluate_with_cache_full``
    at ``gepa/core/state.py:94-130``).  Flow:

      1. ``cache.get_batch(candidate, example_ids)`` partitions into
         ``(cached, uncached_ids)``.
      2. If ``uncached_ids`` is non-empty, write a reduced
         ``helix_batch.json`` containing ONLY those indices, run the
         evaluator subprocess, ``cache.put_batch`` the fresh scores.
      3. Merge cached + fresh into a single ``EvalResult`` whose
         ``instance_scores`` covers every requested id.

    Returns ``(merged EvalResult, num_actual_evals)``.  ``num_actual_evals``
    is the number of examples that were actually sent to the evaluator
    subprocess (0 if all were cached) — mirrors GEPA's
    ``len(uncached_ids)`` return value.
    """
    # Cache keys must remain stable across re-evals of the same candidate
    # identity, but train/val batches must not alias when they share
    # positional ids like "0", "1", ... .
    cand_dict: dict[str, str] = {"id": candidate.id, "split": split}

    # Non-cached branch — mirrors GEPA state.py:628-633 verbatim.
    if cache is None:
        # Per-worktree lock (see ``_worktree_lock`` docstring): serializes
        # concurrent ``write_helix_batch`` + ``run_evaluator`` on the same
        # worktree when parent-minibatch evals run in parallel.
        with _worktree_lock(candidate.worktree_path):
            _write_helix_batch(
                candidate.worktree_path, [int(s) for s in example_ids]
            )
            result = run_evaluator(
                candidate, config, split=split, instance_ids=example_ids,
            )
        return result, len(example_ids)

    # Cached branch — delegate to the GEPA-parity helper on the cache
    # itself (helix.eval_cache.EvaluationCache.evaluate_with_cache_full,
    # which is a line-for-line port of GEPA state.py:94-130).

    def _fetcher(ids: list[str]) -> list[str]:
        # HELIX evaluators read batches off disk via helix_batch.json;
        # the "batch" handed to the evaluator callable is just the list
        # of ids to run.  GEPA's fetcher signature is preserved for
        # semantic parity even though we only use the id list itself.
        return list(ids)

    def _evaluator(
        batch: list[str], _candidate: dict[str, str],
    ) -> tuple[list[object], list[float], list[dict[str, float]] | None]:
        # Write a REDUCED helix_batch.json containing only the uncached
        # indices, then invoke the evaluator subprocess. Evaluators using
        # positional-index handoff read that file from cwd and filter their
        # own dataset to exactly these indices; run_evaluator additionally
        # post-filters instance_scores to ``batch`` in executor.py:245.
        # Per-worktree lock: see ``_worktree_lock`` — parent-minibatch
        # parallelism (audit-mutation §C4) requires serialising the
        # ``write_helix_batch`` + ``run_evaluator`` pair on a given worktree.
        with _worktree_lock(candidate.worktree_path):
            _write_helix_batch(
                candidate.worktree_path, [int(s) for s in batch]
            )
            fresh = run_evaluator(
                candidate, config, split=split, instance_ids=batch,
            )
        # HELIX does not track rollout outputs per-example; store ``None``
        # per slot (the cache's ``RolloutOutput`` type parameter is
        # ``object`` precisely for this reason — see evolution.py:536).
        outputs: list[object] = [None] * len(batch)
        # GEPA parity (adapter.py:154 — ``len(outputs) == len(scores) ==
        # len(batch)``): a missing instance id is an evaluator bug, not a
        # benign zero.  Mirrors the minibatch-acceptance and merge-gate
        # asserts (evolution.py:1394-1411, :1838-1862).
        missing = set(batch) - set(fresh.instance_scores)
        assert not missing, (
            f"Evaluator did not return scores for requested ids: "
            f"{sorted(missing)}"
        )
        scores = [float(fresh.instance_scores[eid]) for eid in batch]
        return outputs, scores, None

    _, scores_by_id, _, num_actual_evals = cache.evaluate_with_cache_full(
        cand_dict, example_ids, _fetcher, _evaluator,
    )

    # Merge hits + fresh into a single EvalResult.  ``scores_by_id``
    # covers every requested id (GEPA state.py:108-127 guarantees this).
    # ``scores`` (aggregate dict) and ``asi`` (metadata) are not carried
    # on cached paths: the minibatch gate and frontier update logic only
    # read ``instance_scores``.
    merged = EvalResult(
        candidate_id=candidate.id,
        scores={},
        asi={},
        instance_scores={eid: scores_by_id[eid] for eid in example_ids},
    )
    return merged, num_actual_evals


def _full_val_example_ids(config: HelixConfig) -> list[str]:
    """Return deterministic full validation ids for positional-index evaluators."""
    val_size = config.dataset.val_size
    if val_size is None or val_size <= 0:
        return []
    return [str(i) for i in range(val_size)]


def _stage_val_example_ids(config: HelixConfig, full_example_ids: list[str]) -> list[str]:
    """Return the deterministic first-N validation ids for the stage gate."""
    stage_size = config.evolution.val_stage_size
    if stage_size is None or stage_size <= 0:
        return []
    return full_example_ids[: min(stage_size, len(full_example_ids))]


def _scores_for_example_ids(result: EvalResult, example_ids: list[str]) -> list[float]:
    """Return per-id scores in a stable order for acceptance comparisons."""
    return [float(result.instance_scores.get(eid, 0.0)) for eid in example_ids]


def _has_example_scores(result: EvalResult | None, example_ids: list[str]) -> bool:
    """Return whether a result contains every requested per-example score."""
    if result is None:
        return False
    return all(eid in result.instance_scores for eid in example_ids)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_evolution(
    config: HelixConfig,
    project_root: Path,
    base_dir: Path,
) -> Candidate:
    """Run the HELIX evolutionary loop.

    Parameters
    ----------
    config:
        Full HELIX configuration.
    project_root:
        Root of the project being evolved (used for git operations and state
        persistence -- state is stored at ``project_root/.helix/state.json``).
    base_dir:
        HELIX working directory (typically ``project_root / ".helix"``).

    Returns
    -------
    Candidate
        The best candidate from the final frontier.
    """
    TRACE.emit(EventType.OPT_START)
    init_base_dir(base_dir, config)

    # Pre-flight check: validate evaluator script exists
    _check_evaluator_script_exists(config.evaluator.command, project_root)

    worktrees_dir = base_dir / "worktrees"
    lineage_path = base_dir / "lineage.json"

    # In-memory candidate registry and frontier
    candidates: dict[str, Candidate] = {}
    rng = _random.Random(config.rng_seed)
    frontier = ParetoFrontier(rng=rng)

    # GEPA parity (Fix 11): evaluation cache — skip re-evaluation of
    # identical (candidate_id, split) pairs.
    eval_cache = EvaluationCache()

    # -- Phase 3 integration: minibatch sampling + GEPA-style acceptance.
    # Construct train/val loaders.  Missing train_path → single-task
    # mode (circle_packing parity); the existing non-minibatch path is
    # used end-to-end.
    # Prompt-grounding / legacy payload paths live on SeedlessConfig.
    train_loader: HelixDataLoader | _RangeDataLoader | None = _make_data_loader(
        config.seedless.train_path
    )
    val_loader: HelixDataLoader | _RangeDataLoader | None = (
        _make_data_loader(config.seedless.val_path)
        if config.seedless.val_path is not None
        else train_loader
    )
    # Architecture A (positional-index handoff): a dataset.train_size
    # synthesises a _RangeDataLoader that yields the ids "0"…"N-1".
    # HELIX writes these indices to {worktree}/helix_batch.json and the
    # evaluator filters its own dataset accordingly.  dataset.val_size
    # handles the full-valset evaluation identically.
    if (
        train_loader is None
        and config.dataset.train_size is not None
        and config.dataset.train_size > 0
    ):
        train_loader = _RangeDataLoader(config.dataset.train_size)
    if (
        val_loader is None
        and config.dataset.val_size is not None
        and config.dataset.val_size > 0
    ):
        val_loader = _RangeDataLoader(config.dataset.val_size)
    use_minibatch_gate = train_loader is not None

    # Batch sampler (GEPA §2) — only wired in when a train loader exists.
    #
    # GEPA parity (harness-detected): GEPA shares a single ``random.Random``
    # across ``candidate_selector`` AND ``EpochShuffledBatchSampler``
    # (gepa/optimize_anything.py:1417 + 1423-1491).  Each
    # ``candidate_selector.select()`` consumes one draw from the shared rng
    # *before* the sampler's first shuffle.  Passing a fresh
    # ``random.Random(seed)`` here would leave HELIX's sampler rng
    # untouched at first shuffle while GEPA's has already advanced — the
    # result is that minibatches diverge from GEPA starting with the very
    # first iteration on identical seeds.  Detected by the GEPA
    # differential testing harness (tests/unit/test_gepa_diff_harness.py).
    batch_sampler: BatchSampler[str] | None = None
    if use_minibatch_gate:
        if config.evolution.batch_sampler == "stratified":
            # Derive group key from instance id by splitting on the
            # configured separator and taking the first part.  E.g.
            # 'cube_stack__s3' -> 'cube_stack' with separator='__'.
            sep = config.evolution.group_key_separator

            def _group_fn(example_id: str) -> str:
                return example_id.split(sep, 1)[0]

            batch_sampler = StratifiedBatchSampler[str](
                minibatch_size=config.evolution.minibatch_size,
                group_fn=_group_fn,
                rng=rng,
            )
        else:
            batch_sampler = EpochShuffledBatchSampler[str](
                minibatch_size=config.evolution.minibatch_size,
                rng=rng,
            )

    # GEPA-parity per-(candidate_hash, example_id) eval cache.  Kept
    # distinct from the legacy ``eval_cache`` above, which is keyed by
    # (candidate_id, split) and used for merge / non-minibatch paths.
    # Use ``object`` for the output type parameter: HELIX only stores
    # per-(candidate, example) scores here, not rollout outputs.
    #
    # GEPA parity (audit-rng-state-persist C1): on resume, restore the
    # cache contents from .helix/eval_cache.pkl when caching is enabled.
    # Mirrors GEPA's behaviour at gepa/core/state.py:683-687
    # (initialize_gepa_state) — when ``cache_evaluation`` is off we drop any
    # persisted cache, when it is on we merge the on-disk dict into the
    # fresh cache instance.  The actual persistence happens via the
    # ``_save_state`` helper defined below, called at every existing
    # ``save_state`` site so the cache survives crash/resume the same way
    # GEPA's pickled state does.
    minibatch_cache: MinibatchEvalCache[object, str] | None = (
        MinibatchEvalCache[object, str]()
        if config.evolution.cache_evaluation
        else None
    )
    if minibatch_cache is not None:
        _persisted_cache = load_eval_cache(project_root)
        if _persisted_cache is not None:
            minibatch_cache._cache.update(_persisted_cache)

    # Acceptance criterion (GEPA §5.1).
    acceptance = (
        StrictImprovementAcceptance()
        if config.evolution.acceptance_criterion == "strict_improvement"
        else ImprovementOrEqualAcceptance()
    )

    # Full-eval policy (GEPA §4.2) — kept for parity and future policy-based
    # val scheduling refactors.
    _full_eval_policy = FullEvaluationPolicy()
    full_val_example_ids = _full_val_example_ids(config)
    stage_val_example_ids = _stage_val_example_ids(config, full_val_example_ids)

    # ------------------------------------------------------------------
    # State: load (resume) or initialise (fresh run)
    # ------------------------------------------------------------------
    state = load_state(project_root)
    cfg_hash = _config_hash(config)
    evaluator_manifest = _load_evaluator_integrity_manifest(base_dir)

    # GEPA parity (audit-rng-state-persist C1): bundle eval-cache persistence
    # with state.json writes.  GEPA's single ``GEPAState.save`` call pickles
    # the cache atomically alongside everything else (state.py:306-340); HELIX
    # routes the (candidate_id, example_id)-keyed companion pickle through
    # this helper so every save site stays consistent without rewriting them.
    def _save_state(s: EvolutionState) -> None:
        save_state(s, project_root)
        if minibatch_cache is not None:
            save_eval_cache(minibatch_cache._cache, project_root)

    if state is None:
        state = EvolutionState(
            generation=0,
            frontier=[],
            instance_scores={},
            budget=BudgetState(),
            config_hash=cfg_hash,
        )
        needs_seed = True
    else:
        needs_seed = False
        if state.config_hash != cfg_hash:
            print_warning(
                "Config hash differs from the saved state; resuming with the current "
                "config while keeping the existing frontier and history."
            )
        # Reconstruct in-memory frontier from persisted evaluations
        for cid in state.frontier:
            result = _load_evaluation(base_dir, cid)
            wt_path = worktrees_dir / cid
            if wt_path.exists() and result is not None:
                cand = Candidate(
                    id=cid,
                    worktree_path=str(wt_path),
                    branch_name=f"helix/{cid}",
                    generation=_gen_from_id(cid),
                    parent_id=None,
                    parent_ids=[],
                    operation="restored",
                )
                candidates[cid] = cand
                frontier.add(cand, result)

    # ------------------------------------------------------------------
    # Seed evaluation
    # ------------------------------------------------------------------
    if needs_seed:
        if config.seedless.enabled:
            # Seedless mode (GEPA parity): generate the initial candidate
            # from scratch via a single LLM invocation — no retry loop.
            set_phase(HelixPhase.SEED_GENERATION)
            print_info("Seedless mode: creating empty worktree for seed generation...")
            seed = create_empty_seed_worktree(project_root, worktrees_dir)
            candidates[seed.id] = seed

            print_info("Generating seed candidate via Claude Code (single attempt)...")
            # GEPA parity: if a training dataset is provided, read up to 3
            # representative examples and include them in the seed prompt
            # (multi-task / generalization mode).  No train_path → single-task
            # mode (dataset_examples=None skips the ## Sample Inputs section).
            _dataset_examples: list[str] | None = None
            if config.seedless.train_path is not None:
                _dataset_examples = load_dataset_examples(config.seedless.train_path)
            seed_prompt = build_seed_generation_prompt(
                objective=config.objective,
                background=config.claude.background,
                evaluator_cmd=config.evaluator.command,
                dataset_examples=_dataset_examples,
            )
            # Single attempt, fail-fast — matches GEPA's _generate_seed_candidate.
            # Clean up the empty worktree if generation raises so we don't
            # leave orphaned worktrees behind on failure.
            try:
                generate_seed(seed.worktree_path, seed_prompt, config)
            except Exception:
                remove_worktree(seed)
                raise
            print_success("Seed generation complete.")
        else:
            print_info("Creating seed worktree...")
            seed = create_seed_worktree(project_root, worktrees_dir)
            candidates[seed.id] = seed

        # Freeze evaluator-related files from the seed baseline so mutation/
        # merge candidates cannot tamper with evaluator logic.
        evaluator_manifest = _build_evaluator_integrity_manifest(
            config=config,
            baseline_root=Path(seed.worktree_path),
            project_root=project_root,
        )
        _write_evaluator_integrity_manifest(base_dir, evaluator_manifest)

        set_phase(HelixPhase.SEED_EVAL)

        print_info("Evaluating seed...")
        # Architecture A: write full val indices to helix_batch.json so the
        # evaluator scores the seed on the complete val split.
        # GEPA parity: when val_size is set we route through
        # ``_cached_evaluate_batch`` — the per-example cache consumer mirrors
        # ``cached_evaluate_full`` at gepa/core/state.py:618.  When val_size
        # is None (single-task mode, e.g. circle_packing) we cannot key the
        # cache by example id, so we fall back to the legacy single-shot
        # evaluator call.
        if full_val_example_ids:
            _seed_example_ids = list(full_val_example_ids)
            seed_result, _seed_num_actual = _cached_evaluate_batch(
                seed, _seed_example_ids, minibatch_cache, config, "val",
            )
            seed_result.candidate_id = seed.id
            state.budget.evaluations += _seed_num_actual
        else:
            seed_result = run_evaluator(seed, config)
            seed_result.candidate_id = seed.id
            # GEPA parity: charge the actual number of per-instance evals,
            # never clamp to 1.  When the evaluator returns an empty
            # ``instance_scores`` dict GEPA charges 0 (engine.py:167 via
            # ``state.increment_evals(num_actual_evals)``).
            state.budget.evaluations += len(seed_result.instance_scores)

        _save_evaluation(base_dir, seed_result)
        frontier.add(seed, seed_result)
        state.frontier = list(frontier._candidates.keys())
        state.instance_scores[seed.id] = seed_result.instance_scores
        # GEPA parity (audit-rng-state-persist C/§3): record per-program
        # discovery budget at the moment the program enters the frontier.
        # Mirrors GEPA core/state.py:537 (``num_metric_calls_by_discovery
        # .append(num_metric_calls_by_discovery_of_new_program)`` inside
        # ``update_state_with_new_program``).
        state.num_metric_calls_by_discovery[seed.id] = state.budget.evaluations
        _save_state(state)

        record_entry(
            lineage_path,
            LineageEntry(
                id=seed.id,
                parent=None,
                parents=[],
                operation="seed",
                generation=0,
                files_changed=[],
            ),
        )
        print_success(f"Seed evaluated: {seed_result.aggregate_score():.4f}")
        render_generation(0, config.evolution.max_generations, frontier, seed_result)
    elif evaluator_manifest is None:
        # Resume compatibility: older runs may not have a persisted manifest.
        # Rebuild one from seed worktree when available, else from project root.
        baseline_root = worktrees_dir / "g0-s0"
        if not baseline_root.exists():
            baseline_root = project_root
        evaluator_manifest = _build_evaluator_integrity_manifest(
            config=config,
            baseline_root=baseline_root,
            project_root=project_root,
        )
        _write_evaluator_integrity_manifest(base_dir, evaluator_manifest)

    # ------------------------------------------------------------------
    # Generation loop
    # ------------------------------------------------------------------
    start_gen = state.generation + 1
    # GEPA parity: discovery-based merge trigger.  merges_due increments
    # when a new candidate is accepted to the frontier.
    merges_due = 0
    # GEPA parity (M1): merge only fires when the *previous* iteration
    # found (accepted) a new program.  This prevents consecutive merge-only
    # generations after a rejected merge.  Mirrors GEPA engine.py:666.
    last_iter_found_new_program = False
    # Mutation counters for display
    mutations_attempted = 0
    mutations_accepted = 0

    def _frontier_best_score() -> float:
        """Return the best aggregate score from the frontier, or 0.0 if empty."""
        if not frontier._candidates:
            return 0.0
        try:
            best_cand = frontier.best()
            r = frontier._results.get(best_cand.id)
            return r.aggregate_score() if r is not None else 0.0
        except (ValueError, KeyError):
            return 0.0

    # Start the Rich progress bar (no-op when HELIX_NO_PROGRESS is set).
    # __enter__ returns self, so _hprog and _hprog_ctx are the same object.
    # We call __exit__ at the end of the loop (and on exception via try/finally
    # wrapped around the loop entry point below).
    _hprog = HelixProgress(config.evolution.max_generations).__enter__()

    for gen in range(start_gen, config.evolution.max_generations + 1):
        state.generation = gen
        # GEPA parity (engine.py:649): bump ``state.i`` unconditionally at
        # the top of every iteration so the proposal counter — used by the
        # batch sampler and as a tiebreaker elsewhere — advances regardless
        # of which path (merge, mutation, early-exit) the iteration takes.
        state.i += 1
        TRACE.emit(EventType.ITER_START, decision=str(gen))

        if budget_exhausted(state, config):
            print_warning("Budget exhausted -- stopping early.")
            break

        # =============================================================
        # GEPA parity (Fix 6/7): Merge OR mutate per iteration.
        # Merge fires FIRST at the start of the iteration (deferred from
        # previous acceptance).  If merge fires, skip mutation entirely
        # (``continue``).  This matches GEPA core/engine.py:664-737.
        # =============================================================
        # GEPA parity (M2 fallthrough — audit-init-engine.md B3):
        # merge_attempted tracks whether an actual merge eval happened this
        # iteration.  GEPA engine.py:664-741 only ``continue``s past the
        # reflective mutation block when a merge is accepted (line 719) or
        # rejected (line 737) — i.e. after the merged candidate has been
        # evaluated.  All earlier fail-fast paths (<2 non-dominated, no
        # triplet, pair already attempted, missing/insufficient val overlap,
        # merge operator failure, evaluator-tamper pre-eval reject) fall
        # through to reflective mutation.  HELIX previously ``continue``d
        # on every merge-gate entry regardless of attempt outcome, cutting
        # the effective mutation count by the merge-gate failure rate.
        merge_attempted = False
        if (
            config.evolution.merge_enabled
            and merges_due > 0
            and last_iter_found_new_program
            and state.total_merge_invocations < config.evolution.max_merge_invocations
            and len(frontier) >= 2
        ):
            set_phase(HelixPhase.MERGE)
            # GEPA parity (M1): clear the flag so consecutive merge-only
            # generations cannot fire (mirrors engine.py:668,740).
            last_iter_found_new_program = False

            lineage = load_lineage(lineage_path)
            score_map: dict[str, float] = {}
            for cid, res in frontier._results.items():
                score_map[cid] = res.aggregate_score()
            for cid, inst_scores in state.instance_scores.items():
                if cid not in score_map and inst_scores:
                    score_map[cid] = (
                        sum(inst_scores.values()) / len(inst_scores)
                    )

            # GEPA parity (M2): merge candidates must be non-dominated.
            # GEPA merge.py:299-304 uses find_dominator_programs() to filter.
            non_dominated = frontier.get_non_dominated()
            merge_candidate_ids = [cid for cid in frontier._candidates if cid in non_dominated]

            # GEPA parity (L3): ``find_merge_triplet`` returns None when
            # ``len(frontier_ids) < 2`` (lineage.py:145) without consuming
            # rng, so the "< 2 non-dominated" fail-fast reduces to
            # ``triplet is None`` — both paths now fall through to
            # reflective mutation (GEPA engine.py:741-742).
            #
            # GEPA parity (merge-pairing audit D1, /tmp/audit_audit-merge-pairing.md:49-50):
            # mirror GEPA ``merge.py:130-131`` — you need two siblings plus
            # one ancestor, so fewer than 3 total candidates can never
            # yield a valid triplet.  Kept as an explicit guard for
            # clarity; functionally equivalent to ``find_merge_triplet``
            # returning ``None`` in that regime.  Fall-through style
            # (audit B3): when the gate fails we leave ``merge_attempted``
            # False and drop into reflective mutation below.
            triplet: tuple[str, str, str] | None
            if len(lineage) < 3:
                triplet = None
            else:
                # GEPA parity (merge-pairing audit B1/B2,
                # /tmp/audit_audit-merge-pairing.md:10-22): push the
                # "already-attempted pair" and "val-support overlap"
                # filters INTO ``find_merge_triplet``'s retry loop so a
                # blocked sample triggers resampling rather than bailing
                # the iteration.  Mirrors GEPA
                # ``sample_and_attempt_merge_programs_by_common_predictors``
                # (merge.py:118-207) where the same filters are inside the
                # ``for _ in range(max_attempts)`` loop.
                _attempted_pairs: set[tuple[str, str]] = {
                    (p[0], p[1]) for p in state.merge_attempted_pairs if len(p) >= 2
                }

                def _has_val_support_overlap(i: str, j: str) -> bool:
                    era_i = frontier._results.get(i)
                    erb_j = frontier._results.get(j)
                    if era_i is None or erb_j is None:
                        return False
                    common = set(era_i.instance_scores.keys()) & set(erb_j.instance_scores.keys())
                    return len(common) >= config.evolution.merge_val_overlap_floor

                triplet = find_merge_triplet(
                    lineage,
                    merge_candidate_ids,
                    score_map,
                    rng=rng,
                    attempted_pairs=_attempted_pairs,
                    has_val_support_overlap=_has_val_support_overlap,
                )

            if triplet is not None:
                # GEPA parity (merge-pairing audit C3, merge.py:94-95):
                # ``find_merge_triplet`` now returns the canonical
                # ``(i, j)`` (lex-sorted), so ``cid_i <= cid_j`` always —
                # the merge subprocess, attempted-pair ledger and the
                # description-triplet dedup all see the same tuple order.
                cid_i, cid_j, _ancestor_id = triplet
                pair_key = [cid_i, cid_j]

                # Resolve parent val results once; by contract the
                # ``_has_val_support_overlap`` closure passed to
                # ``find_merge_triplet`` guarantees era/erb are non-None
                # and their common-id set meets the overlap floor, but we
                # narrow for mypy and downstream asserts.
                era = frontier._results.get(cid_i)
                erb = frontier._results.get(cid_j)
                assert era is not None and erb is not None, (
                    "find_merge_triplet returned a pair that failed "
                    "_has_val_support_overlap -- invariant violation"
                )

                state.merge_attempted_pairs.append(pair_key)

                a = frontier._candidates[cid_i]
                b = frontier._candidates[cid_j]

                state.merge_counter += 1
                merge_id = f"g{gen}-m{state.merge_counter}"

                merged = merge(
                    candidate_a=a,
                    candidate_b=b,
                    new_id=merge_id,
                    config=config,
                    base_dir=worktrees_dir,
                    background=config.claude.background,
                    eval_result_a=era,
                    eval_result_b=erb,
                )

                if merged is None:
                    # GEPA parity (M2/B3): merge operator failed before
                    # any eval; no attempt, fall through to mutation.
                    print_error(
                        f"Merge {merge_id} failed "
                        f"(candidates: {a.id} + {b.id}, gen {gen}). "
                        f"Claude Code returned no output or the merge subprocess errored. "
                        f"Check the HELIX ERROR panel above for full diagnostics."
                    )
                else:
                    merge_tamper = _detect_evaluator_tamper(
                        merged, evaluator_manifest
                    )
                    if merge_tamper:
                        # Evaluator-tamper reject happens PRE-eval — no
                        # merge was attempted in the GEPA sense
                        # (audit-init-engine.md B3).  Fall through.
                        print_warning(
                            f"Merge {merge_id} touched protected evaluator files "
                            f"({', '.join(merge_tamper)}) -- rejecting."
                        )
                        try:
                            remove_worktree(merged)
                        except Exception as _rm_exc:
                            print_warning(
                                f"Could not remove worktree for rejected merge {merge_id}: {_rm_exc}"
                            )
                    else:
                        candidates[merged.id] = merged
                        record_entry(
                            lineage_path,
                            LineageEntry(
                                id=merged.id,
                                parent=a.id,
                                parents=[a.id, b.id],
                                operation="merge",
                                generation=gen,
                                files_changed=[],
                            ),
                        )
                        # Save state BEFORE snapshot so that if the commit
                        # crashes (e.g. empty-commit), state is already
                        # persisted and resume can skip re-doing this merge.
                        _save_state(state)
                        # GEPA parity (merge-pairing audit C1,
                        # /tmp/audit_audit-merge-pairing.md:28-31): the
                        # HEAD SHA of the snapshotted worktree is HELIX's
                        # port of GEPA's ``new_prog_desc`` (merge.py:195-203);
                        # content-addressed so two different triplets that
                        # land on the same merged output hash once and skip
                        # the eval on the duplicate, while the same pair
                        # with a differently-merged result is still allowed
                        # to retry.
                        merged_sha = snapshot_candidate(
                            merged,
                            f"helix: merge {merge_id} ({cid_i}+{cid_j})",
                        )
                        _desc_triplet = [cid_i, cid_j, merged_sha]
                        if _desc_triplet in state.merge_description_triplets:
                            print_warning(
                                f"Merge {merge_id} produced a previously-seen "
                                f"output (desc {merged_sha[:8]}) -- skipping."
                            )
                            try:
                                remove_worktree(merged)
                            except Exception as _rm_exc:
                                print_warning(
                                    f"Could not remove worktree for duplicate-desc merge {merge_id}: {_rm_exc}"
                                )
                            if merged.id in candidates:
                                del candidates[merged.id]
                            _save_state(state)
                            if not _hprog.is_active:
                                render_budget(state.budget, config.evolution)
                            _hprog.update(gen, _frontier_best_score())
                            continue
                        state.merge_description_triplets.append(_desc_triplet)
                        # GEPA parity (M5): merge acceptance evaluates merged on a
                        # size-bounded stratified subsample of ids both parents have
                        # val-scored. Subsample selection ported from GEPA
                        # merge.py:258-288 (select_eval_subsample_for_merged_program);
                        # default size 5 matches GEPA's hardcoded constant, overridable
                        # via evolution.merge_subsample_size. Required score is
                        # max(parent subsample sums); mirrors GEPA merge.py:344-345, 394-395.
                        merge_subsample_ids = sorted(
                            select_eval_subsample_for_merged_program(
                                era.instance_scores,
                                erb.instance_scores,
                                rng,
                                num_subsample_ids=config.evolution.merge_subsample_size,
                            )
                        )
                        # GEPA parity (M2/B3): from here on, the merged
                        # candidate is evaluated, so this iteration is
                        # consumed (GEPA engine.py:719 on accept,
                        # engine.py:737 on reject).  merge_attempted=True
                        # causes the end-of-branch guard below to
                        # ``continue`` past reflective mutation.
                        merge_attempted = True
                        merge_result, _merge_evals = _cached_evaluate_batch(
                            merged, merge_subsample_ids, minibatch_cache, config, "val",
                        )
                        merge_result.candidate_id = merged.id
                        state.budget.evaluations += _merge_evals
                        _save_evaluation(base_dir, merge_result)

                        # GEPA parity (Fix 13): mid-generation budget check.
                        if budget_exhausted(state, config):
                            print_warning("Budget exhausted mid-generation -- stopping.")
                            _save_state(state)
                            break

                        # Merged subsample sum must be >= max of parent
                        # subsample sums (GEPA merge.py:344-345, 394-395).
                        # merge_subsample_ids is sorted(select_eval_subsample_for_merged_program(
                        #   era.instance_scores, erb.instance_scores, ...))
                        # — every sampled id is drawn from the intersection
                        # of era.instance_scores and erb.instance_scores
                        # (common_val_ids above).  The asserts keep the
                        # invariant loud (GEPA merge.py:342-343).
                        assert set(merge_subsample_ids).issubset(
                            era.instance_scores
                        ), (
                            "merge_subsample_ids must be a subset of "
                            "era.instance_scores"
                        )
                        a_score = sum(
                            era.instance_scores[k] for k in merge_subsample_ids
                        )
                        assert set(merge_subsample_ids).issubset(
                            erb.instance_scores
                        ), (
                            "merge_subsample_ids must be a subset of "
                            "erb.instance_scores"
                        )
                        b_score = sum(
                            erb.instance_scores[k] for k in merge_subsample_ids
                        )
                        required_score = max(a_score, b_score)

                        # GEPA parity: iterate the subsample list (not the dict) so the
                        # rng.choices fallback path (duplicate ids when |common| < size)
                        # counts duplicates equally on both sides.  Intentional divergence
                        # from HELIX's usual dict-based aggregation; flagged for a future
                        # ablation study (unique-count vs duplicate-count would be an
                        # interesting knob to vary once we have an evolution baseline).
                        assert set(merge_subsample_ids).issubset(merge_result.instance_scores), (
                            "merge_subsample_ids must be a subset of merge_result.instance_scores"
                        )
                        merge_score = sum(
                            merge_result.instance_scores[k] for k in merge_subsample_ids
                        )

                        if merge_score >= required_score:
                            # GEPA parity (merge-gate audit M3,
                            # /tmp/audit_audit-merge-gate.md:10-32): after
                            # the subsample gate passes, run a FULL-valset
                            # eval on the merged candidate and pass THAT
                            # result (not the 5-id subsample) to
                            # ``frontier.add`` / ``state.instance_scores``.
                            # Mirrors GEPA ``engine.py:688-696`` →
                            # ``_run_full_eval_and_add`` (engine.py:175-197)
                            # → ``_evaluate_on_valset`` (engine.py:154-173).
                            # Without this, the merged entry carries only
                            # subsample coverage and Pareto dominance /
                            # ``sum_score`` comparisons skew against the
                            # merged candidate once it is picked as a parent.
                            # Budget accounting is via ``_cached_evaluate_batch``'s
                            # ``num_actual_evals`` (uncached-only), mirroring
                            # GEPA ``state.increment_evals(num_actual_evals)``
                            # at ``engine.py:167``.
                            if full_val_example_ids:
                                _full_val_ids = list(full_val_example_ids)
                                full_val_result, _full_n = _cached_evaluate_batch(
                                    merged, _full_val_ids, minibatch_cache, config, "val",
                                )
                                full_val_result.candidate_id = merged.id
                                state.budget.evaluations += _full_n
                            else:
                                full_val_result, _full_val_cached = _cached_eval(
                                    merged, config, "val", eval_cache,
                                )
                                full_val_result.candidate_id = merged.id
                                if not _full_val_cached:
                                    # GEPA parity: charge the actual count
                                    # of per-instance evals (never clamp to 1).
                                    state.budget.evaluations += len(
                                        full_val_result.instance_scores
                                    )
                            _save_evaluation(base_dir, full_val_result)

                            if budget_exhausted(state, config):
                                print_warning(
                                    "Budget exhausted during merge full-val eval -- stopping."
                                )
                                _save_state(state)
                                break

                            merges_due -= 1
                            state.total_merge_invocations += 1
                            frontier.add(merged, full_val_result)
                            state.frontier = list(frontier._candidates.keys())
                            state.instance_scores[merged.id] = full_val_result.instance_scores
                            # GEPA parity (audit-rng-state-persist C/§3):
                            # record per-program discovery budget at the
                            # moment the merged program enters the
                            # frontier.  GEPA core/state.py:537.
                            state.num_metric_calls_by_discovery[merged.id] = (
                                state.budget.evaluations
                            )
                        else:
                            print_warning(
                                f"Merge {merge_id} score {merge_score:.4f} < "
                                f"max parent {required_score:.4f} -- rejecting."
                            )
                            try:
                                remove_worktree(merged)
                            except Exception as _rm_exc:
                                print_warning(
                                    f"Could not remove worktree for rejected merge {merge_id}: {_rm_exc}"
                                )
                            if merged.id in candidates:
                                del candidates[merged.id]

            # GEPA parity (M2/B3): only consume this iteration when a merge
            # was actually evaluated (engine.py:719,737).  On any fall-through
            # (triplet None, pair already attempted, overlap fail, merge op
            # failure, tamper reject) we drop into reflective mutation below.
            if merge_attempted:
                _save_state(state)
                if not _hprog.is_active:
                    render_budget(state.budget, config.evolution)
                _hprog.update(gen, _frontier_best_score())
                continue

        elif config.evolution.merge_enabled:
            # GEPA parity (C1): unconditionally clear flag when merge is enabled
            # but gate conditions not met (merges_due==0 or last_iter_found=False).
            # GEPA engine.py:739-740 always clears before reflective mutation.
            last_iter_found_new_program = False

        # =============================================================
        # Phase 2: Mutation (only when merge did not fire above)
        #
        # GEPA parity (M6): when num_parallel_proposals > 1, run the
        # GEPA 3-step parallel pipeline (engine.py:381-452):
        #   1. Sample N parent contexts sequentially
        #   2. Execute N mutations in ThreadPoolExecutor
        #   3. Process acceptances SEQUENTIALLY
        # When num_parallel_proposals == 1, behaviour is identical to
        # the original single-mutation path.
        # =============================================================

        n_proposals = config.evolution.num_parallel_proposals

        # ---- Step 1a: Build pre-sample contexts (SEQUENTIAL) ----
        # GEPA core/engine.py:381-452 has three stages:
        #   1. ``prepare_proposal`` — sequential (parent select + minibatch sample)
        #   2. ``execute_proposal`` — PARALLEL (parent eval + reflect + child eval)
        #   3. ``apply_proposal_output`` — sequential (budget + cache write)
        # HELIX §1a mirrors GEPA's prepare_proposal: candidate selection,
        # ``state.i`` bump, and minibatch sampling live here.  The parent
        # minibatch EVAL is moved out to §1b (see below) to parallelise it,
        # matching GEPA reflective_mutation.py:268 (``eval_curr = adapter.evaluate(...)``)
        # running inside the thread pool submitted at engine.py:422-426.
        #
        # Entries are tuples ``(parent, parent_frontier_result, subsample_ids, new_id)``.
        presample_contexts: list[tuple[Candidate, EvalResult | None, list[str] | None, str]] = []
        _budget_break = False

        for _p_idx in range(n_proposals):
            if budget_exhausted(state, config):
                _budget_break = True
                break

            # GEPA parity (engine.py:405-410): the FIRST proposal reuses
            # the iteration slot already created at the top of the outer
            # loop (state.i bumped at evolution.py loop head).  For each
            # ADDITIONAL parallel proposal, GEPA bumps state.i again so
            # the per-proposal counter advances independently of the
            # outer loop.  The bump is unconditional (no minibatch gate).
            if _p_idx > 0:
                state.i += 1

            parent = frontier.select_parent()
            parent_frontier_result = frontier._results.get(parent.id)

            # --- Minibatch gate pre-sampling (GEPA §5.1) --------------
            # Sample subsample ids.  ``state.i`` is now advanced at the
            # top of the run loop (GEPA engine.py:649) — and again per
            # extra parallel proposal above (engine.py:408) — so the
            # sampler always sees the right counter.  The parent-on-
            # minibatch eval is deferred to §1b so that N parent evals
            # overlap under ``num_parallel_proposals > 1``
            # (audit-mutation §C4 MODERATE E).
            subsample_ids: list[str] | None = None
            if use_minibatch_gate and train_loader is not None and batch_sampler is not None:
                subsample_ids = batch_sampler.next_minibatch_ids(train_loader, state)
                TRACE.emit(
                    EventType.SAMPLE_MINIBATCH,
                    candidate_id=parent.id,
                    example_ids=list(subsample_ids),
                    split="train",
                )

            state.mutation_counter += 1
            new_id = f"g{gen}-s{state.mutation_counter}"
            presample_contexts.append((parent, parent_frontier_result, subsample_ids, new_id))

        # ---- Step 1b: Parent minibatch eval (PARALLEL) ----
        # GEPA parity (audit-mutation §C4 MODERATE E): GEPA runs
        # ``eval_curr = self.adapter.evaluate(ctx.minibatch, ctx.curr_prog, ...)``
        # (reflective_mutation.py:268) inside ``execute_proposal``, which
        # engine.py:422-426 submits to a ThreadPoolExecutor so N parent evals
        # overlap.  HELIX used to run this sequentially in the pre-sample loop
        # (evolution.py:1406-1444 pre-fix).  Legacy no-minibatch path still
        # runs ``_cached_eval(parent, "train")`` here because there is no
        # per-proposal subsample to use.
        def _eval_parent(
            pre_ctx: tuple[Candidate, EvalResult | None, list[str] | None, str],
        ) -> tuple[EvalResult | None, int]:
            _parent, _pfr, _sub_ids, _new_id = pre_ctx
            if _sub_ids is not None:
                _mb, _n_uncached = _cached_evaluate_batch(
                    _parent, list(_sub_ids), minibatch_cache, config, "train",
                )
                _mb.candidate_id = _parent.id
                return _mb, _n_uncached
            return None, 0

        parent_eval_results: list[tuple[EvalResult | None, int]]
        if n_proposals > 1 and len(presample_contexts) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
            parent_eval_results = [(None, 0)] * len(presample_contexts)
            with ThreadPoolExecutor(max_workers=len(presample_contexts)) as _pe_pool:
                _pe_future_to_idx = {
                    _pe_pool.submit(_eval_parent, _pre_ctx): _idx
                    for _idx, _pre_ctx in enumerate(presample_contexts)
                }
                for _pe_future in _as_completed(_pe_future_to_idx):
                    _pe_idx = _pe_future_to_idx[_pe_future]
                    parent_eval_results[_pe_idx] = _pe_future.result()
        else:
            parent_eval_results = [_eval_parent(_pre_ctx) for _pre_ctx in presample_contexts]

        # ---- Step 1c: Charge budget + skip-perfect (SEQUENTIAL) ----
        # GEPA core/engine.py:361 applies deferred updates sequentially via
        # ``apply_proposal_output``.  Here we (a) charge the minibatch size to
        # the budget unconditionally per audit-budget-caching §C1 MODERATE H,
        # mirroring reflective_mutation.py:269 (``total_evals +=
        # eval_curr.num_metric_calls if not None else len(ctx.subsample_ids)``),
        # and (b) apply the skip-perfect gate (reflective_mutation.py:308-312).
        proposal_contexts: list[tuple[Candidate, EvalResult | None, EvalResult, str]] = []
        proposal_subsamples: list[list[str] | None] = []
        proposal_parent_mb_results: list[EvalResult | None] = []

        for _p_idx, (_pre_ctx, (_mb_result, _n_uncached)) in enumerate(
            zip(presample_contexts, parent_eval_results)
        ):
            parent, parent_frontier_result, subsample_ids, new_id = _pre_ctx
            parent_mb_result: EvalResult | None = _mb_result

            if subsample_ids is not None:
                # MODERATE H (audit-budget-caching §C1): GEPA charges the
                # full minibatch size regardless of cache hit status
                # (reflective_mutation.py:269 charges
                # ``eval_curr.num_metric_calls if not None else
                # len(ctx.subsample_ids)`` — the adapter.evaluate call at
                # :268 bypasses the cache, so cache hits never reduce the
                # charge).  HELIX formerly charged ``len(uncached_ids)``
                # which let overlapping minibatches burn budget more slowly
                # than GEPA; now charge the full minibatch width.
                state.budget.evaluations += len(subsample_ids)

            if budget_exhausted(state, config):
                _budget_break = True
                break

            # GEPA parity: the eval_result passed to the mutator is the
            # parent's minibatch eval (reflective_mutation.py:268,341), not
            # a full-train re-eval. HELIX previously ran a redundant
            # _cached_eval(parent, 'train') here, which forced evaluator-owned
            # datasets to rescore the entire training split just to provide
            # prompt context. Legacy (no-minibatch) single-task mode keeps a
            # train eval because there is no minibatch to use.
            eval_for_mutate: EvalResult
            if parent_mb_result is not None:
                eval_for_mutate = parent_mb_result
            else:
                set_phase(HelixPhase.TRAIN_EVALUATION)
                eval_for_mutate, _ = _cached_eval(parent, config, "train", eval_cache)
                eval_for_mutate.candidate_id = parent.id

            # Skip-if-perfect (GEPA reflective_mutation.py:308-327, audit
            # finding M1 in audit-init-engine.md B1/B2):
            #   * GEPA returns ProposalOutput(proposal=None) and the outer
            #     iteration loop CONTINUES with a new parent — it does NOT
            #     terminate the whole run.  Mirror that by ``continue``-ing
            #     the per-proposal sampling loop (HELIX evolution.py).
            #   * GEPA's condition is ``all(s >= perfect_score for s in
            #     eval_curr.scores)`` (reflective_mutation.py:311) — every
            #     per-example score must clear the bar.  HELIX previously
            #     used the mean ``aggregate_score()``, which fires sooner
            #     when a minibatch is [0.5, 1.0, 1.0] @ threshold=0.8.
            if (
                config.evolution.perfect_score_threshold is not None
                and all(
                    s >= config.evolution.perfect_score_threshold
                    for s in eval_for_mutate.instance_scores.values()
                )
            ):
                print_info(
                    f"Iteration {gen}: all subsample scores perfect for "
                    f"parent {parent.id}; skipping proposal "
                    f"(GEPA reflective_mutation.py:308-327)."
                )
                continue

            proposal_contexts.append((parent, parent_frontier_result, eval_for_mutate, new_id))
            proposal_subsamples.append(subsample_ids)
            proposal_parent_mb_results.append(parent_mb_result)

        if _budget_break and not proposal_contexts:
            print_warning("Budget exhausted mid-generation -- stopping.")
            _save_state(state)
            break

        # ---- Step 2: Execute mutations (parallel if N > 1) ----
        set_phase(HelixPhase.MUTATION)

        def _do_mutate(ctx: tuple[Candidate, EvalResult | None, EvalResult, str]) -> Candidate | None:
            _parent, _parent_frontier_result, _eval_for_mutate, _new_id = ctx
            # GEPA parity: mutator receives the parent's minibatch eval
            # (see make_reflective_dataset(curr_prog, eval_curr, ...) in
            # reflective_mutation.py:341), not a full-train re-eval.
            return mutate(
                parent=_parent,
                eval_result=_eval_for_mutate,
                new_id=_new_id,
                config=config,
                base_dir=worktrees_dir,
                background=config.claude.background,
            )

        mutation_results: list[Candidate | None]
        if n_proposals > 1 and len(proposal_contexts) > 1:
            from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed
            mutation_results = [None] * len(proposal_contexts)
            with ThreadPoolExecutor(max_workers=len(proposal_contexts)) as pool:
                future_to_idx = {
                    pool.submit(_do_mutate, ctx): idx
                    for idx, ctx in enumerate(proposal_contexts)
                }
                for future in _as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        mutation_results[idx] = future.result()
                    except Exception as exc:
                        _ctx = proposal_contexts[idx]
                        _fail_id = _ctx[3]
                        _fail_parent = _ctx[0]
                        if isinstance(exc, HelixError):
                            exc.operation = exc.operation or f"parallel mutate {_fail_id}"
                            print_helix_error(exc)
                            if isinstance(exc, RateLimitError):
                                # Loud ERROR log so the user clearly sees which
                                # proposal slot was dropped and why (Fix 2).
                                logger.error(
                                    "Mutation %s (parent: %s, gen %d) failed after "
                                    "all retries: %s: %s — proposal slot skipped.",
                                    _fail_id,
                                    _fail_parent.id,
                                    gen,
                                    type(exc).__name__,
                                    exc,
                                )
                                print_error(
                                    f"Mutation [bold]{_fail_id}[/bold] hit rate limit "
                                    f"after all retries — proposal slot skipped. "
                                    f"Run [cyan]helix resume[/cyan] when rate limits clear."
                                )
                        else:
                            print_error(
                                f"Parallel mutation {_fail_id} "
                                f"(parent: {_fail_parent.id}, gen {gen}) "
                                f"failed with exception:\n"
                                f"{traceback.format_exc()}"
                            )
        else:
            mutation_results = [_do_mutate(ctx) for ctx in proposal_contexts]

        # ---- Step 3: Process acceptances SEQUENTIALLY ----
        # Each accepted mutation's full val eval updates state before
        # the next is processed (GEPA engine.py:444-452).
        any_accepted_this_gen = False
        _last_eval_result: EvalResult | None = None

        for _p_idx, (ctx, child) in enumerate(zip(proposal_contexts, mutation_results)):
            _parent, _parent_frontier_result, _eval_for_mutate, _new_id = ctx
            _subsample_ids = proposal_subsamples[_p_idx]
            _parent_mb = proposal_parent_mb_results[_p_idx]

            if child is None:
                print_warning(f"Mutation {_new_id} failed -- skipping.")
                continue

            mutations_attempted += 1
            tampered_paths = _detect_evaluator_tamper(child, evaluator_manifest)
            if tampered_paths:
                print_warning(
                    f"Mutation {child.id} touched protected evaluator files "
                    f"({', '.join(tampered_paths)}) -- rejecting."
                )
                try:
                    remove_worktree(child)
                except Exception as _rm_exc:
                    print_warning(
                        f"Could not remove worktree for rejected candidate {child.id}: {_rm_exc}"
                    )
                continue

            candidates[child.id] = child
            record_entry(
                lineage_path,
                LineageEntry(
                    id=child.id,
                    parent=_parent.id,
                    parents=[_parent.id],
                    operation="mutate",
                    generation=gen,
                    files_changed=[],
                ),
            )
            # Save state BEFORE snapshot so that if the commit crashes
            # (e.g. empty-commit), state is already persisted and resume
            # can skip re-doing this mutation.
            _save_state(state)
            snapshot_candidate(child, f"helix: mutate {child.id}")

            # --- Gating evaluation ----------------------------------------
            # Two paths here — both gate on GEPA's strict-sum acceptance
            # (``acceptance.should_accept``), matching GEPA engine.py:303:
            #  (a) Minibatch gate (GEPA §5.1): when train_loader exists,
            #      run the child on the SAME subsample as the parent and
            #      accept on sum improvement over the parent's minibatch
            #      scores.  No separate train re-eval.
            #  (b) Legacy train-gating: no train_loader — run the child
            #      on the full train split and apply the same acceptance
            #      criterion (GEPA parity MODERATE D — audit-mutation.md C3).
            if use_minibatch_gate and _subsample_ids is not None and _parent_mb is not None:
                set_phase(HelixPhase.MUTATION_GATING)
                # GEPA parity: route child-on-minibatch through the
                # per-example cache consumer.  For a freshly mutated
                # child the cache never hits (the candidate hash is new),
                # but the fresh scores are put_batch'd so that any later
                # re-eval of the same child on overlapping indices is free.
                gating_result, _n = _cached_evaluate_batch(
                    child,
                    list(_subsample_ids),
                    minibatch_cache,
                    config,
                    "train",
                )
                gating_result.candidate_id = child.id
                _last_eval_result = gating_result
                state.budget.evaluations += _n

                if budget_exhausted(state, config):
                    print_warning("Budget exhausted mid-generation -- stopping.")
                    _save_state(state)
                    _budget_break = True
                    break

                # Apply the configured acceptance criterion on the
                # per-instance score vectors (GEPA §5.1).
                #
                # GEPA parity (adapter.py:154 — ``len(outputs) == len(scores)
                # == len(batch)``): a missing instance id in the parent or
                # child minibatch result is an evaluator bug, not a benign
                # zero.  Both vectors must cover every id in
                # ``_subsample_ids`` so the acceptance criterion compares
                # like-for-like.  The merge path enforces the same invariant
                # at evolution.py:1394-1411.
                from types import SimpleNamespace as _SN
                assert set(_subsample_ids).issubset(_parent_mb.instance_scores), (
                    f"Parent minibatch eval missing ids: "
                    f"{set(_subsample_ids) - set(_parent_mb.instance_scores)}"
                )
                _before = [
                    float(_parent_mb.instance_scores[str(eid)])
                    for eid in _subsample_ids
                ]
                assert set(_subsample_ids).issubset(gating_result.instance_scores), (
                    f"Child minibatch eval missing ids: "
                    f"{set(_subsample_ids) - set(gating_result.instance_scores)}"
                )
                _after = [
                    float(gating_result.instance_scores[str(eid)])
                    for eid in _subsample_ids
                ]
                _proposal = _SN(
                    subsample_scores_before=_before,
                    subsample_scores_after=_after,
                )
                if not acceptance.should_accept(_proposal):
                    TRACE.emit(
                        EventType.ACCEPT_DECISION,
                        candidate_id=child.id,
                        decision="reject",
                        example_ids=list(_subsample_ids),
                        score=float(sum(_after)),
                    )
                    print_warning(
                        f"Minibatch gate: {child.id} rejected "
                        f"(sum {sum(_after):.4f} vs parent {sum(_before):.4f}) -- removing."
                    )
                    try:
                        remove_worktree(child)
                    except Exception as _rm_exc:
                        print_warning(
                            f"Could not remove worktree for rejected candidate {child.id}: {_rm_exc}"
                        )
                    del candidates[child.id]
                    continue
                else:
                    TRACE.emit(
                        EventType.ACCEPT_DECISION,
                        candidate_id=child.id,
                        decision="accept",
                        example_ids=list(_subsample_ids),
                        score=float(sum(_after)),
                    )
            else:
                set_phase(HelixPhase.MUTATION_GATING)
                gating_result, _gating_cached = _cached_eval(child, config, "train", eval_cache)
                gating_result.candidate_id = child.id
                _last_eval_result = gating_result
                # Legacy single-task mode still gates on train, but the parent baseline
                # must come from the same train eval that was passed into the mutator,
                # not the parent's stored val frontier result.
                parent_acceptance_result = _eval_for_mutate
                if not _gating_cached:
                    # GEPA parity: charge actual count, never clamp to 1.
                    state.budget.evaluations += len(gating_result.instance_scores)

                if budget_exhausted(state, config):
                    print_warning("Budget exhausted mid-generation -- stopping.")
                    _save_state(state)
                    _budget_break = True
                    break

                # GEPA parity (MODERATE D — audit-mutation.md C3):
                # GEPA has a single acceptance path — sum-score strict
                # improvement on the same minibatch (engine.py:287-303,
                # reflective_mutation.py:420).  HELIX previously ran
                # ``degrades()`` as a pre-check in this legacy (no-minibatch)
                # path, applying a tolerance that GEPA does not have.  The
                # pre-check is removed so both paths now gate on the SAME
                # criterion (``acceptance.should_accept`` = strict sum by
                # default).  Routing through the acceptance criterion
                # instead of the inline ``child_sum <= parent_sum`` check
                # mirrors GEPA engine.py:303 and lets callers swap in
                # ``ImprovementOrEqualAcceptance`` uniformly across both
                # paths.
                from types import SimpleNamespace as _SN
                _legacy_before = list(
                    parent_acceptance_result.instance_scores.values()
                )
                _legacy_after = list(gating_result.instance_scores.values())
                _legacy_proposal = _SN(
                    subsample_scores_before=_legacy_before,
                    subsample_scores_after=_legacy_after,
                )
                if not acceptance.should_accept(_legacy_proposal):
                    parent_sum = sum(_legacy_before)
                    child_sum = sum(_legacy_after)
                    print_warning(
                        f"Acceptance: {child.id} does not improve "
                        f"(child_sum={child_sum:.4f}, parent_sum={parent_sum:.4f}) -- removing."
                    )
                    try:
                        remove_worktree(child)
                    except Exception as _rm_exc:
                        print_warning(
                            f"Could not remove worktree for rejected candidate {child.id}: {_rm_exc}"
                        )
                    del candidates[child.id]
                    continue

            # --- Staged val gate ------------------------------------------
            use_val_stage_gate = _has_example_scores(
                _parent_frontier_result, stage_val_example_ids
            )
            if stage_val_example_ids and use_val_stage_gate:
                set_phase(HelixPhase.VAL_EVALUATION)
                stage_result, _n = _cached_evaluate_batch(
                    child,
                    list(stage_val_example_ids),
                    minibatch_cache,
                    config,
                    "val",
                )
                stage_result.candidate_id = child.id
                _last_eval_result = stage_result
                state.budget.evaluations += _n

                if budget_exhausted(state, config):
                    print_warning("Budget exhausted mid-generation -- stopping.")
                    _save_state(state)
                    _budget_break = True
                    break

                from types import SimpleNamespace as _SN

                _stage_before = _scores_for_example_ids(
                    _parent_frontier_result or EvalResult(candidate_id="", scores={}, asi={}, instance_scores={}),
                    stage_val_example_ids,
                )
                _stage_after = _scores_for_example_ids(
                    stage_result, stage_val_example_ids
                )
                _proposal = _SN(
                    subsample_scores_before=_stage_before,
                    subsample_scores_after=_stage_after,
                )
                if not acceptance.should_accept(_proposal):
                    TRACE.emit(
                        EventType.ACCEPT_DECISION,
                        candidate_id=child.id,
                        decision="reject_stage",
                        example_ids=list(stage_val_example_ids),
                        score=float(sum(_stage_after)),
                    )
                    print_warning(
                        f"Val stage: {child.id} rejected on first "
                        f"{len(stage_val_example_ids)} val ids "
                        f"(sum {sum(_stage_after):.4f} vs parent "
                        f"{sum(_stage_before):.4f}) -- removing."
                    )
                    try:
                        remove_worktree(child)
                    except Exception as _rm_exc:
                        print_warning(
                            f"Could not remove worktree for stage-rejected candidate "
                            f"{child.id}: {_rm_exc}"
                        )
                    del candidates[child.id]
                    continue

                TRACE.emit(
                    EventType.ACCEPT_DECISION,
                    candidate_id=child.id,
                    decision="accept_stage",
                    example_ids=list(stage_val_example_ids),
                    score=float(sum(_stage_after)),
                )
                print_info(
                    f"Val stage: {child.id} passed on first {len(stage_val_example_ids)} "
                    f"val ids (sum {sum(_stage_after):.4f} vs parent "
                    f"{sum(_stage_before):.4f}); promoting to full val."
                )

            # --- Val evaluation -------------------------------------------
            set_phase(HelixPhase.VAL_EVALUATION)
            # GEPA parity: when val_size is set, route the full-val eval
            # through the per-example cache consumer so that overlapping
            # val indices across candidates (or re-evals of the same
            # candidate) are not recomputed.  When val_size is unset we
            # fall back to the legacy ``_cached_eval`` path keyed by
            # ``(candidate_id, split)`` — single-task mode has no example
            # ids to key on.
            if full_val_example_ids:
                _val_example_ids = list(full_val_example_ids)
                val_result, _n = _cached_evaluate_batch(
                    child, _val_example_ids, minibatch_cache, config, "val",
                )
                val_result.candidate_id = child.id
                _last_eval_result = val_result
                state.budget.evaluations += _n
            else:
                val_result, _val_cached = _cached_eval(child, config, "val", eval_cache)
                val_result.candidate_id = child.id
                _last_eval_result = val_result
                if not _val_cached:
                    # GEPA parity: charge actual count, never clamp to 1.
                    state.budget.evaluations += len(val_result.instance_scores)

            if budget_exhausted(state, config):
                print_warning("Budget exhausted mid-generation -- stopping.")
                _save_state(state)
                _budget_break = True
                break

            # --- Update frontier ------------------------------------------
            set_phase(HelixPhase.PARETO_UPDATE)
            _save_evaluation(base_dir, val_result)
            frontier.add(child, val_result)
            state.frontier = list(frontier._candidates.keys())
            state.instance_scores[child.id] = val_result.instance_scores
            # GEPA parity (audit-rng-state-persist C/§3): record per-program
            # discovery budget at the moment the child enters the frontier.
            # GEPA core/state.py:537.
            state.num_metric_calls_by_discovery[child.id] = state.budget.evaluations
            TRACE.emit(
                EventType.FRONTIER_UPDATE,
                candidate_id=child.id,
                score=val_result.aggregate_score(),
            )
            # GEPA parity (Fix 7): accepting a new program increments merges_due.
            if (
                config.evolution.merge_enabled
                and state.total_merge_invocations < config.evolution.max_merge_invocations
            ):
                merges_due += 1
            # GEPA parity (M1): flag that this iteration found a new program.
            last_iter_found_new_program = True
            any_accepted_this_gen = True

            mutations_accepted += 1

        # If budget was exhausted during sequential acceptance, break outer loop.
        if _budget_break:
            break

        # Render at end of generation using the last result seen.
        if _last_eval_result is not None:
            render_generation(
                gen, config.evolution.max_generations, frontier, _last_eval_result,
                mutations_attempted=mutations_attempted,
                mutations_accepted=mutations_accepted,
            )

        # GEPA parity (C2): removed parallel re-eval of all frontier
        # candidates.  GEPA does NOT re-evaluate the entire frontier after
        # every acceptance — the old block bypassed _cached_eval() and burned
        # budget quadratically with growing frontier size.

        _save_state(state)
        if not _hprog.is_active:
            render_budget(state.budget, config.evolution)
        _hprog.update(gen, _frontier_best_score())
        TRACE.emit(EventType.ITER_END, decision=str(gen))

    # Stop the Rich progress bar (idempotent if already stopped or disabled).
    _hprog.__exit__(None, None, None)

    # ------------------------------------------------------------------
    # Return best
    # ------------------------------------------------------------------
    best = frontier.best()
    print_success(f"Evolution complete.  Best candidate: {best.id}")
    TRACE.emit(EventType.OPT_END, candidate_id=best.id)
    return best
