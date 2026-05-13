"""HELIX main evolution loop."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random as _random
import shutil
import shlex
import subprocess
import tempfile
import threading
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any


from helix.batch_sampler import (
    BatchSampler,
    EpochShuffledBatchSampler,
    StratifiedBatchSampler,
)
from helix import budget as budget_api
from helix.config import HelixConfig, load_dataset_examples
from helix.eval_cache import EvaluationCache as MinibatchEvalCache
from helix.eval_policy import (
    FullEvaluationPolicy,
    ImprovementOrEqualAcceptance,
    StrictImprovementAcceptance,
)
from helix.display import (
    HelixLiveDisplay,
    HelixPhase,
    UsageStats,
    print_error,
    print_info,
    print_success,
    print_warning,
    render_budget,
    render_frontier_table,
    render_generation,
    set_phase,
)

from helix.exceptions import (
    HelixError,
    PromptArtifactCollisionError,
    RateLimitError,
    ResumeIncompatibleError,
    print_helix_error,
)
from helix.executor import run_evaluator
from helix.lineage import LineageEntry, find_merge_triplet, load_lineage, record_entry
from helix.merger import merge, select_eval_subsample_for_merged_program
from helix.mutator import mutate, build_seed_generation_prompt, generate_seed
from helix.population import (
    Candidate,
    CandidateSummary,
    EvalResult,
    HelixResult,
    ParetoFrontier,
)
from helix.sandbox import start_evaluator_sidecar
from helix.state import (
    BudgetState,
    clear_eval_cache,
    EvaluationCache,
    EvolutionState,
    load_eval_cache,
    load_state,
    save_eval_cache,
    save_state,
)
from helix.trace import TRACE, EventType
from helix.worktree import (
    create_seed_worktree,
    create_empty_seed_worktree,
    remove_worktree,
    snapshot_candidate,
)
from helix.evaluator_manifest import (  # noqa: E402  (after helix.worktree intentionally)
    # 14 moved symbols — re-exported for backward compatibility and internal use
    _collect_protected_evaluator_paths,
    _copy_protected_path,
    _detect_evaluator_tamper,
    _evaluator_manifest_path,
    _extract_script_token,
    _iter_protected_manifest_files,
    _load_evaluator_integrity_manifest,
    _looks_like_script_file,
    _build_evaluator_integrity_manifest,
    _refresh_and_snapshot_protected_evaluator_files,
    _refresh_protected_evaluator_files,
    _sha256_file,
    _to_repo_relative,
    _write_evaluator_integrity_manifest,
    # Constants needed by _check_evaluator_script_exists (which stays in this module)
    _NO_SCRIPT_COMMANDS,
    _SHELL_COMMAND_FLAGS,
    _SHELL_WRAPPERS,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def degrades(new_result: EvalResult, baseline: EvalResult, threshold: float) -> bool:
    """Return True if the new result regresses below baseline by more than threshold."""
    return new_result.sum_score() < baseline.sum_score() - threshold


def _config_hash(config: HelixConfig) -> str:
    data = config.model_dump_json()
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def _resume_semantics(config: HelixConfig) -> dict[str, Any]:
    """Return config fields that define resume-compatible optimization semantics.

    The returned shape is the authoritative resume-compatibility contract.
    Any field present here is hard-rejected on resume if it changes; fields
    deliberately omitted (notably ``evolution.cache_evaluation``) may be
    toggled freely between runs.

    Top-level keys:
      * ``objective``: changes the optimization target outright.
      * ``rng_seed``: pinned for full determinism parity with the prior run.
      * ``evaluator``: command, parser, stdout/stderr capture flags, extra
        commands, protected files, sidecar configuration — all influence
        what saved scores actually mean (GEPA Optimize Anything parity).
      * ``dataset``: full pydantic dump; train/val splits and grouping
        affect which examples saved scores were computed against.
      * ``seedless``: enabled flag and external train/val paths.
      * ``evolution``: frontier dimensionality, acceptance, minibatch /
        sampler shape, val-stage size, and merge policy.

    Deliberately omitted (safe to toggle between runs):
      * ``evolution.cache_evaluation``: cache state is handled
        separately via ``eval_cache.pkl`` persistence.
      * ``evolution.max_generations`` / ``max_evaluations`` / parallelism:
        budgets and concurrency, not score interpretation.
      * ``agent.*``: changing the mutation backend / model / effort
        affects only future proposals, not the meaning of saved scores.
        (GEPA likewise treats the LM client as out-of-band of the
        persisted state.)
      * ``sandbox`` / ``worktree`` / top-level ``seed`` / ``passthrough_env``
        / ``env``: runtime / filesystem layout.  The separate
        ``evaluator_integrity_manifest`` covers evaluator-environment
        drift.
    """
    # GEPA Optimize Anything parity: ``EvaluatorConfig.include_stdout`` /
    # ``include_stderr`` directly change which bytes the score parser sees,
    # so toggling them mid-run can silently reinterpret saved scores.
    # ``sidecar`` describes a completely different evaluator runtime
    # (image, runner image, endpoint); resuming under a different sidecar
    # would compare scores produced by different binaries.  Both are
    # hard-rejected on resume to match the strict-equality stance GEPA
    # gets implicitly by pickling its full config alongside state.
    return {
        "objective": config.objective,
        "rng_seed": config.rng_seed,
        "evaluator": {
            "command": config.evaluator.command,
            "score_parser": config.evaluator.score_parser,
            "include_stdout": config.evaluator.include_stdout,
            "include_stderr": config.evaluator.include_stderr,
            "extra_commands": list(config.evaluator.extra_commands),
            "protected_files": list(config.evaluator.protected_files),
            "sidecar": (
                config.evaluator.sidecar.model_dump(mode="json")
                if config.evaluator.sidecar is not None
                else None
            ),
        },
        "dataset": config.dataset.model_dump(mode="json"),
        "seedless": {
            "enabled": config.seedless.enabled,
            "train_path": (
                str(config.seedless.train_path)
                if config.seedless.train_path is not None
                else None
            ),
            "val_path": (
                str(config.seedless.val_path)
                if config.seedless.val_path is not None
                else None
            ),
        },
        "evolution": {
            "frontier_type": config.evolution.frontier_type,
            "acceptance_criterion": config.evolution.acceptance_criterion,
            "minibatch_size": config.evolution.minibatch_size,
            "batch_sampler": config.evolution.batch_sampler,
            "group_key_separator": config.evolution.group_key_separator,
            "val_stage_size": config.evolution.val_stage_size,
            "merge_enabled": config.evolution.merge_enabled,
            "max_merge_invocations": config.evolution.max_merge_invocations,
            "merge_val_overlap_floor": config.evolution.merge_val_overlap_floor,
            "merge_subsample_size": config.evolution.merge_subsample_size,
        },
    }


_DIFF_VALUE_PREVIEW_CHARS = 80


def _format_diff_value(value: Any) -> str:
    """Render a single value in a diff message, truncating long reprs.

    List-shaped fields (e.g. ``evaluator.extra_commands``,
    ``evaluator.protected_files``) can otherwise overwhelm the rejection
    message with full file/path dumps; clamp to a fixed character budget
    with an explicit ellipsis so the structural shape stays visible but
    long bodies don't drown the user.
    """
    rendered = repr(value)
    if len(rendered) > _DIFF_VALUE_PREVIEW_CHARS:
        return rendered[: _DIFF_VALUE_PREVIEW_CHARS - 3] + "..."
    return rendered


def _semantic_diffs(
    saved: dict[str, Any],
    current: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    diffs: list[str] = []
    for key in sorted(set(saved) | set(current)):
        path = f"{prefix}.{key}" if prefix else key
        if key not in saved:
            diffs.append(
                f"{path}: missing in saved state, current={_format_diff_value(current[key])}"
            )
            continue
        if key not in current:
            diffs.append(
                f"{path}: saved={_format_diff_value(saved[key])}, missing in current config"
            )
            continue
        saved_value = saved[key]
        current_value = current[key]
        if isinstance(saved_value, dict) and isinstance(current_value, dict):
            diffs.extend(_semantic_diffs(saved_value, current_value, path))
        elif saved_value != current_value:
            diffs.append(
                f"{path}: saved={_format_diff_value(saved_value)}, "
                f"current={_format_diff_value(current_value)}"
            )
    return diffs


_RESUME_REMEDIATION_SUGGESTION = (
    "Restore the original config to keep resuming, or run `helix clean` and "
    "start a fresh run to use the new config."
)


def _validate_resume_semantics(state: EvolutionState, config: HelixConfig) -> None:
    """Reject resume when current config would reinterpret persisted state.

    Raises :class:`ResumeIncompatibleError` (a :class:`HelixError`) so the
    CLI can route through ``print_helix_error`` instead of letting the
    user see a raw Python traceback.
    """
    current = _resume_semantics(config)
    if state.frontier_type != config.evolution.frontier_type:
        raise ResumeIncompatibleError(
            "Cannot resume with a different evolution.frontier_type: "
            f"saved={state.frontier_type!r}, current={config.evolution.frontier_type!r}.",
            operation="resume",
            phase="validate_resume_semantics",
            suggestion=_RESUME_REMEDIATION_SUGGESTION,
        )

    if not state.resume_semantics:
        return

    diffs = _semantic_diffs(state.resume_semantics, current)
    if diffs:
        preview = "; ".join(diffs[:5])
        if len(diffs) > 5:
            preview += f"; ... {len(diffs) - 5} more"
        raise ResumeIncompatibleError(
            "Cannot resume because the current config changes optimization "
            f"semantics: {preview}.",
            operation="resume",
            phase="validate_resume_semantics",
            suggestion=_RESUME_REMEDIATION_SUGGESTION,
        )


def init_base_dir(base_dir: Path, config: HelixConfig) -> None:
    """Create HELIX working directories and snapshot config to base_dir/config.toml."""
    base_dir.mkdir(parents=True, exist_ok=True)
    (base_dir / "worktrees").mkdir(parents=True, exist_ok=True)
    (base_dir / "evaluations").mkdir(parents=True, exist_ok=True)

    config_path = base_dir / "config.toml"
    if not config_path.exists():
        lines = [
            "# HELIX config snapshot (auto-generated)\n",
            f"objective = {json.dumps(config.objective)}\n",
            "\n[evaluator]\n",
            f"command = {json.dumps(config.evaluator.command)}\n",
        ]
        config_path.write_text("".join(lines))


def _atomic_write_json(path: Path, data: Any) -> None:
    """Write *data* as JSON to *path* atomically using a sibling temp file.

    Uses ``tempfile.mkstemp`` + ``Path.replace()`` (atomic on POSIX) to avoid
    partial-write corruption when a concurrent reader or a crash mid-write
    could otherwise observe a truncated file.  The temp file is created in the
    same directory as *path* so that ``replace()`` is an intra-filesystem
    rename rather than a cross-device copy.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        Path(tmp).replace(path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_evaluation(base_dir: Path, result: EvalResult) -> None:
    """Persist an EvalResult to evaluations/<candidate_id>.json.

    Uses ``EvalResult.to_dict()`` so every field — including the newer
    ``side_info`` / ``per_example_side_info`` / ``objective_scores``
    optionals — round-trips through the on-disk JSON.  Optional fields
    are omitted by ``to_dict`` when ``None`` so evaluations from
    single-task / non-helix_result paths stay byte-identical to their
    pre-multi-axis shape.
    """
    eval_dir = base_dir / "evaluations"
    _atomic_write_json(eval_dir / f"{result.candidate_id}.json", result.to_dict())


def _save_attempt_result(
    base_dir: Path,
    result: EvalResult,
    *,
    status: str,
    reason: str,
    parent_id: str | None,
    generation: int,
    stage: str,
    example_ids: list[str] | None = None,
) -> None:
    """Persist a non-frontier attempt result without polluting evaluations/.

    ``evaluations/`` is the accepted/full-evaluation surface used for frontier
    reporting.  Rejected mutations can still consume a generation and useful
    evaluator work, so keep their gating/stage results under ``attempts/`` for
    resume/debugging instead of leaving lineage entries with no result artifact.
    """
    attempts_dir = base_dir / "attempts"
    payload = result.to_dict()
    payload["attempt"] = {
        "status": status,
        "reason": reason,
        "parent_id": parent_id,
        "generation": generation,
        "stage": stage,
        "example_ids": example_ids,
    }
    _atomic_write_json(attempts_dir / f"{result.candidate_id}.json", payload)


def _save_skip_record(
    base_dir: Path,
    *,
    generation: int,
    records: list[dict[str, Any]],
) -> None:
    """Persist all per-proposal skip records for a generation as a JSON list.

    All parents that triggered the perfect-subsample gate within the same
    generation are written as a single list so that ``n_proposals > 1`` with
    multiple perfect parents does not silently overwrite earlier records.
    Skip records are intentionally separate from candidate attempts: no child
    program exists, but the run should still explain why a generation has no
    candidate result.
    """
    skips_dir = base_dir / "skips"
    _atomic_write_json(skips_dir / f"g{generation}.json", records)


def _load_evaluation(base_dir: Path, candidate_id: str) -> EvalResult | None:
    """Load a saved EvalResult, or None if the file is absent."""
    path = base_dir / "evaluations" / f"{candidate_id}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return EvalResult.from_dict(data)


def _gen_from_id(candidate_id: str) -> int:
    """Parse the generation number from a candidate id like 'g3-s1'."""
    try:
        return int(candidate_id.split("-")[0].lstrip("g"))
    except (IndexError, ValueError):
        return 0


def _proposal_counter_from_id(candidate_id: str) -> int:
    """Parse the proposal counter from a candidate id like 'g3-s4'.

    Returns 0 on parse failure (e.g., legacy or seed-only IDs that do not
    match the ``gN-sN`` format).  Note that counter=0 is ambiguous with the
    first valid proposal slot ``g0-s0``; callers that filter by
    ``counter > 0`` intentionally exclude both parse failures and the zeroth
    slot for that reason.
    """
    try:
        return int(candidate_id.split("-")[1].lstrip("s"))
    except (IndexError, ValueError):
        return 0


def _remove_lineage_records(lineage_path: Path, candidate_ids: set[str]) -> None:
    """Remove candidate ids from the raw lineage JSON array."""
    if not lineage_path.exists() or not candidate_ids:
        return
    records = json.loads(lineage_path.read_text())
    kept = [record for record in records if record.get("id") not in candidate_ids]
    tmp_path = lineage_path.with_suffix(f"{lineage_path.suffix}.tmp")
    tmp_path.write_text(json.dumps(kept, indent=2))
    tmp_path.replace(lineage_path)


def _reconcile_incomplete_attempts_on_resume(
    *,
    state: EvolutionState,
    base_dir: Path,
    worktrees_dir: Path,
    lineage_path: Path,
) -> bool:
    """Discard interrupted live attempts so resume retries their visible slot.

    A completed proposal has either an evaluation artifact, an attempt artifact,
    or frontier membership.  If a lineage entry still has a worktree but none of
    those completion markers, HELIX likely stopped between candidate creation
    and final result persistence.  Remove only those live incomplete worktrees;
    historical lineage-only entries without worktrees are left untouched because
    they may come from older completed runs that predate attempt artifacts.
    """
    if not lineage_path.exists():
        return False

    lineage = load_lineage(lineage_path)
    completed_ids = set(state.frontier)
    for artifact_dir in (base_dir / "evaluations", base_dir / "attempts"):
        if artifact_dir.exists():
            completed_ids.update(path.stem for path in artifact_dir.glob("*.json"))

    incomplete_ids: set[str] = set()
    for candidate_id, entry in lineage.items():
        if candidate_id in completed_ids:
            continue
        if entry.operation not in {"mutate", "mutation", "merge"}:
            continue
        if (worktrees_dir / candidate_id).exists():
            incomplete_ids.add(candidate_id)

    for wt_path in worktrees_dir.glob("g*-s*"):
        candidate_id = wt_path.name
        if candidate_id in completed_ids or candidate_id in lineage:
            continue
        if wt_path.is_dir():
            incomplete_ids.add(candidate_id)

    if not incomplete_ids:
        return False

    first_generation = min(_gen_from_id(candidate_id) for candidate_id in incomplete_ids)
    first_counter = min(_proposal_counter_from_id(candidate_id) for candidate_id in incomplete_ids)

    for candidate_id in sorted(incomplete_ids):
        wt_path = worktrees_dir / candidate_id
        _safe_remove_worktree(
            Candidate(
                id=candidate_id,
                worktree_path=str(wt_path),
                branch_name=f"helix/{candidate_id}",
                generation=_gen_from_id(candidate_id),
                parent_id=None,
                parent_ids=[],
                operation="interrupted",
            ),
            label="orphan worktree from prior run",
        )

    _remove_lineage_records(lineage_path, incomplete_ids)
    for candidate_id in incomplete_ids:
        state.instance_scores.pop(candidate_id, None)
    state.frontier = [cid for cid in state.frontier if cid not in incomplete_ids]
    state.active_frontier = {
        objective: [cid for cid in ids if cid not in incomplete_ids]
        for objective, ids in state.active_frontier.items()
    }

    completed_counters = [
        _proposal_counter_from_id(candidate_id)
        for candidate_id in completed_ids
        if _proposal_counter_from_id(candidate_id) > 0
    ]
    max_completed_counter = max(completed_counters, default=0)
    state.mutation_counter = max(max_completed_counter, first_counter - 1)
    budget_api.set_generation(state, max(0, first_generation - 1))
    print_warning(
        "Removed incomplete attempt(s) from prior interruption: "
        f"{', '.join(sorted(incomplete_ids))}. The next resume will retry "
        f"generation {first_generation}."
    )
    return True


def _safe_remove_worktree(candidate: Candidate, *, label: str) -> None:
    """Remove ``candidate``'s worktree, warning on failure rather than raising.

    The nine rejection / cleanup paths in :func:`_run_evolution_impl` and
    :func:`_reconcile_incomplete_attempts_on_resume` all need the same
    "best-effort remove + warn" behaviour.  Extracting this avoids 9× copies
    of the same try/except.
    """
    try:
        remove_worktree(candidate)
    except Exception as exc:
        print_warning(
            f"Could not remove worktree for {label} {candidate.id}: {exc}"
        )


def _check_evaluator_script_exists(evaluator_command: str, project_root: Path) -> None:
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

    # Sidecar runner scripts often live inside the sandbox image rather than
    # the candidate repo, e.g. /runner/evaluate_client.py.
    if Path(script_path).is_absolute():
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
    """Synthetic loader for Architecture A example-id handoff.

    Exposes ids ``["0", "1", ..., str(size-1)]`` — no underlying
    payload is loaded.  The evaluator (running in the worktree) is
    responsible for loading its own dataset and filtering by the
    ids written to ``helix_batch.json`` (casting back to ``int``
    for positional indexing if that matches its dataset layout).
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


def _write_helix_batch(worktree_path: str | Path, example_ids: list[str]) -> None:
    """Write example ids to ``{worktree}/helix_batch.json``.

    Side-channel handoff to the evaluator (Architecture A).  The
    evaluator, when run with cwd=worktree_path, reads this file and
    filters its dataset by the supplied ids.

    Ids are written verbatim as JSON strings; the evaluator is
    responsible for whatever interpretation its dataset requires
    (stringified integer indices, composite ``group__N`` task ids, etc.).
    Historically this function coerced every id to ``int`` which made
    structured ids like ``"group_alpha__case_3"`` — required by
    :class:`helix.batch_sampler.StratifiedBatchSampler` — raise
    ``ValueError`` at the serialization boundary.
    """
    path = Path(worktree_path) / "helix_batch.json"
    try:
        path.write_text(json.dumps(example_ids))
    except FileNotFoundError:
        # Worktree directory does not exist (e.g. under unit-test mocks
        # that fabricate fake paths).  Silently skip — production paths
        # always have the worktree on disk before eval.
        logger.debug(
            "worktree %s missing; skipping helix_batch.json write", worktree_path
        )


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


def _candidate_content_key(candidate: Candidate) -> str:
    """Return a stable content key for cache reuse across equivalent candidates.

    Contract: the caller is responsible for ensuring the candidate's worktree
    has its evaluation-relevant content committed (clean working tree, no
    untracked files). The key is derived from ``HEAD^{tree}``, which is
    content-addressable over the *committed* tracked tree only — it does not
    reflect uncommitted modifications, the staged index, or untracked files.

    Defensive behavior: if the worktree is detected to be dirty (modified
    tracked files or untracked, non-ignored files present), we fall back to
    ``candidate.id`` so that two candidates with identical HEAD trees but
    differing dirty state cannot collide on the same cache key. Submodule
    contents are summarized as gitlinks inside the parent tree, so changing
    the submodule pointer changes the key but in-place edits inside an
    uncommitted submodule do not — keep submodule state committed for
    reliable reuse.
    """
    try:
        tree_proc = subprocess.run(
            ["git", "rev-parse", "HEAD^{tree}"],
            cwd=candidate.worktree_path,
            check=True,
            capture_output=True,
            text=True,
        )
        sha = tree_proc.stdout.strip()
        if not sha:
            return candidate.id

        # Reject the tree SHA if the worktree is dirty or has untracked
        # (non-ignored) files; otherwise we'd risk a false cache hit.
        status_proc = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            cwd=candidate.worktree_path,
            check=True,
            capture_output=True,
            text=True,
        )
        if status_proc.stdout.strip():
            logger.warning(
                "Candidate %s worktree is not clean; falling back to id cache "
                "key to avoid stale-content cache hits.",
                candidate.id,
            )
            return candidate.id
        return sha
    except Exception as exc:
        logger.warning(
            "Could not resolve git tree for candidate %s (%s); falling back "
            "to id cache key.",
            candidate.id,
            exc,
        )
    return candidate.id


def _cached_eval(
    candidate: Candidate,
    config: HelixConfig,
    split: str,
    cache: EvaluationCache | None,
) -> tuple[EvalResult, bool]:
    """Run evaluator with cache.  Returns (result, was_cached).

    GEPA parity: cache only when enabled, and key by candidate content rather
    than HELIX's lineage id so equivalent candidates can reuse results.
    """
    if cache is None:
        return run_evaluator(candidate, config, split=split), False
    candidate_key = _candidate_content_key(candidate)
    cached = cache.get(candidate_key, split)
    if cached is not None:
        return EvalResult.from_dict(cached), True
    result = run_evaluator(candidate, config, split=split)
    cache.put(candidate_key, split, result.to_dict())
    return result, False


def _cached_evaluate_batch(
    candidate: Candidate,
    example_ids: list[str],
    cache: "MinibatchEvalCache[object, str] | None",
    config: HelixConfig,
    split: str,
    project_root: Path,
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
    # Non-cached branch — mirrors GEPA state.py:628-633 verbatim.
    if cache is None:
        # Per-worktree lock (see ``_worktree_lock`` docstring): serializes
        # concurrent ``write_helix_batch`` + ``run_evaluator`` on the same
        # worktree when parent-minibatch evals run in parallel.
        with _worktree_lock(candidate.worktree_path):
            _refresh_protected_evaluator_files(candidate, config, project_root)
            _write_helix_batch(candidate.worktree_path, example_ids)
            result = run_evaluator(
                candidate,
                config,
                split=split,
                instance_ids=example_ids,
            )
        return result, len(example_ids)

    # Cached branch — delegate to the GEPA-parity helper on the cache
    # itself (helix.eval_cache.EvaluationCache.evaluate_with_cache_full,
    # which is a line-for-line port of GEPA state.py:94-130).
    # Cache keys must remain stable across equivalent candidate content, but
    # train/val batches must not alias when they share positional ids like
    # "0", "1", ... .
    cand_dict: dict[str, str] = {
        "content_key": _candidate_content_key(candidate),
        "split": split,
    }

    # Closure side-channel: the cache stores ``(output, score, objective_scores)``
    # per id but has no slot for freeform ``side_info``.  Collect fresh
    # per-example side_info in a closure dict so we can attach it to the
    # merged ``EvalResult`` below.  Cache-hit ids (not in this dict) get
    # ``{}`` as their per_example_side_info slot — their prior side_info
    # was consumed by the reflection prompt at their original eval time
    # and is not round-tripped through the cache.
    fresh_side_info_by_id: dict[str, dict[str, Any]] = {}

    def _fetcher(ids: list[str]) -> list[str]:
        # HELIX evaluators read batches off disk via helix_batch.json;
        # the "batch" handed to the evaluator callable is just the list
        # of ids to run.  GEPA's fetcher signature is preserved for
        # semantic parity even though we only use the id list itself.
        return list(ids)

    def _evaluator(
        batch: list[str],
        _candidate: dict[str, str],
    ) -> tuple[list[object], list[float], list[dict[str, float]] | None]:
        # Write a REDUCED helix_batch.json containing only the uncached
        # example ids, then invoke the evaluator subprocess.  Evaluators
        # read that file from cwd and filter their own dataset to exactly
        # these ids; run_evaluator additionally post-filters
        # instance_scores to ``batch`` in executor.py:245.
        # Per-worktree lock: see ``_worktree_lock`` — parent-minibatch
        # parallelism (audit-mutation §C4) requires serialising the
        # ``write_helix_batch`` + ``run_evaluator`` pair on a given worktree.
        with _worktree_lock(candidate.worktree_path):
            _refresh_protected_evaluator_files(candidate, config, project_root)
            _write_helix_batch(candidate.worktree_path, batch)
            fresh = run_evaluator(
                candidate,
                config,
                split=split,
                instance_ids=batch,
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
            f"Evaluator did not return scores for requested ids: {sorted(missing)}"
        )
        scores = [float(fresh.instance_scores[eid]) for eid in batch]
        # Thread per-example objective_scores through the cache: GEPA
        # ``EvaluationBatch.objective_scores`` parity
        # (``src/gepa/core/adapter.py:26``).  Feeds the multi-axis
        # Pareto frontier when ``evolution.frontier_type`` is
        # ``"objective"``, ``"hybrid"``, or ``"cartesian"``.  The
        # underlying ``EvaluationCache`` already has a slot for this
        # (``put_batch(..., objective_scores_list=...)``); previously
        # ``_evaluator`` returned ``None`` here and the multi-axis data
        # was dropped on the cached path.
        obj_list: list[dict[str, float]] | None = None
        if fresh.objective_scores is not None and len(fresh.objective_scores) == len(
            batch
        ):
            obj_list = [fresh.objective_scores[i] for i in range(len(batch))]
        # Capture per-example side_info for the outer merge.  Only
        # freshly-evaluated ids populate this; cache hits remain ``{}``.
        if fresh.per_example_side_info is not None and len(
            fresh.per_example_side_info
        ) == len(batch):
            for i, eid in enumerate(batch):
                fresh_side_info_by_id[eid] = fresh.per_example_side_info[i]
        return outputs, scores, obj_list

    _, scores_by_id, objective_by_id, num_actual_evals = cache.evaluate_with_cache_full(
        cand_dict,
        example_ids,
        _fetcher,
        _evaluator,
    )

    # Merge hits + fresh into a single EvalResult.  ``scores_by_id``
    # covers every requested id (GEPA state.py:108-127 guarantees this).
    # ``scores`` (aggregate dict) and ``asi`` (metadata) are not carried
    # on cached paths: the minibatch gate and frontier update logic only
    # read ``instance_scores``.
    #
    # Per-example ``objective_scores`` and ``per_example_side_info`` ARE
    # attached when any data was produced (freshly-evaluated) or cached
    # for any id in this batch — the multi-axis frontier and reflection
    # paths both depend on them.  Missing entries get ``{}`` placeholders
    # so the list length always equals ``len(example_ids)`` (positional
    # alignment is the whole point of the per-example contract).
    objective_scores_list: list[dict[str, float]] | None = None
    if objective_by_id is not None:
        objective_scores_list = [objective_by_id.get(eid, {}) for eid in example_ids]
    per_example_side_info_list: list[dict[str, Any]] | None = None
    if fresh_side_info_by_id:
        per_example_side_info_list = [
            fresh_side_info_by_id.get(eid, {}) for eid in example_ids
        ]

    merged = EvalResult(
        candidate_id=candidate.id,
        scores={},
        asi={},
        instance_scores={eid: scores_by_id[eid] for eid in example_ids},
        per_example_side_info=per_example_side_info_list,
        objective_scores=objective_scores_list,
    )
    return merged, num_actual_evals


def _inject_top_k_best_example_history(
    eval_result: EvalResult,
    frontier: ParetoFrontier,
    *,
    k: int = 3,
) -> EvalResult:
    """Attach per-example best-history context for reflection prompts.

    Optimize Anything exposes top-performing example histories to reflection.
    HELIX's compatible surface is a per-example ``side_info`` field named
    ``top_k_best_example_history``; it is rendered by the existing diagnostics
    prompt path without changing candidate/worktree semantics.
    """
    if k <= 0 or not eval_result.instance_scores or not frontier.has_results():
        return eval_result

    example_ids = list(eval_result.instance_scores.keys())
    history_by_id: dict[str, list[dict[str, Any]]] = {}
    for eid in example_ids:
        rows: list[dict[str, Any]] = []
        for cid, result in frontier.iter_results():
            if eid in result.instance_scores:
                rows.append(
                    {
                        "candidate_id": cid,
                        "score": float(result.instance_scores[eid]),
                    }
                )
        rows.sort(key=lambda row: row["score"], reverse=True)
        if rows:
            history_by_id[eid] = rows[:k]

    if not history_by_id:
        return eval_result

    if eval_result.per_example_side_info is None:
        per_example: list[dict[str, Any]] = [{} for _ in example_ids]
    else:
        per_example = [dict(item) for item in eval_result.per_example_side_info]
        if len(per_example) != len(example_ids):
            # Length skew between ``instance_scores`` and ``per_example_side_info``
            # almost certainly indicates an upstream bug (the two are positional
            # over the same example id list).  Don't silently throw away the
            # caller's diagnostics — warn loudly and rebuild empty so reflection
            # still gets the top-K history rather than crashing.
            logger.warning(
                "per_example_side_info length (%d) does not match "
                "instance_scores length (%d) for candidate %s; rebuilding "
                "side_info from empty before injecting top_k history.",
                len(per_example),
                len(example_ids),
                eval_result.candidate_id,
            )
            per_example = [{} for _ in example_ids]

    for idx, eid in enumerate(example_ids):
        history_rows = history_by_id.get(eid)
        if history_rows:
            per_example[idx]["top_k_best_example_history"] = history_rows
    eval_result.per_example_side_info = per_example
    return eval_result


def _run_full_val_eval(
    candidate: Candidate,
    state: EvolutionState,
    *,
    full_val_example_ids: list[str] | tuple[str, ...],
    minibatch_cache: "MinibatchEvalCache[object, str] | None",
    eval_cache: EvaluationCache | None,
    config: HelixConfig,
    project_root: Path,
    source_batch: str,
    source_single: str,
) -> EvalResult:
    """Full-val eval on whichever path (batch via cache OR single-task) the
    config selects; charges the budget with the appropriate ``source`` tag.

    The batch path calls :func:`_cached_evaluate_batch` (per-example cache,
    GEPA parity).  The single-task path calls :func:`_cached_eval` (split-level
    cache).  Charges via :func:`budget_api.charge_evaluation` with
    ``source=source_batch`` on the batch path and ``source=source_single`` on
    the single-task path.

    The three :func:`_run_evolution_impl` / seed-eval full-val blocks are
    structurally identical except for the candidate variable and ``source``
    strings (§3 D3 of the scope report).  This helper eliminates that
    duplication.  Callers are responsible for any pre-eval side effects (e.g.
    the seed-eval path calls ``_refresh_protected_evaluator_files`` before
    invoking this helper on the single-task branch).
    """
    if full_val_example_ids:
        result, n_uncached = _cached_evaluate_batch(
            candidate,
            list(full_val_example_ids),
            minibatch_cache,
            config,
            "val",
            project_root,
        )
        result.candidate_id = candidate.id
        budget_api.charge_evaluation(
            state,
            num_actual_examples=n_uncached,
            candidate_id=candidate.id,
            split="val",
            source=source_batch,
        )
    else:
        result, cached = _cached_eval(candidate, config, "val", eval_cache)
        result.candidate_id = candidate.id
        budget_api.charge_evaluation(
            state,
            was_cached=cached,
            candidate_id=candidate.id,
            split="val",
            source=source_single,
        )
    return result


def _full_val_example_ids(
    config: HelixConfig,
    val_loader: "HelixDataLoader | _RangeDataLoader | None" = None,
) -> list[str]:
    """Return deterministic full validation ids.

    Priority (first non-empty wins):

    1. ``dataset.val_size`` — stringified range ids ``"0".."N-1"``
       (Architecture A cardinality-only path).
    2. ``val_loader.all_ids()`` — ids from the configured
       :class:`HelixDataLoader` over ``seedless.val_path`` (or
       ``seedless.train_path`` when val_path is unset).  File-stem
       ids for a directory layout, stringified indices otherwise.
       This is what lets the seed-eval and full-val paths write
       ``helix_batch.json`` for evaluators that use ``helix_result``
       and need the positional id handoff — the
       strict ``helix_result`` parser requires the batch file to be
       present on every evaluator invocation.
    3. Empty list — single-task/no-example path (no example-id handoff).
    """
    val_size = config.dataset.val_size
    if val_size is not None and val_size > 0:
        return [str(i) for i in range(val_size)]
    if val_loader is not None and len(val_loader) > 0:
        return list(val_loader.all_ids())
    return []


def _stage_val_example_ids(
    config: HelixConfig, full_example_ids: list[str]
) -> list[str]:
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


def _build_helix_result(
    *,
    best: Candidate,
    frontier: ParetoFrontier,
    state: EvolutionState,
    base_dir: Path,
    config: HelixConfig,
    lineage_path: Path,
) -> HelixResult:
    """Build the structured programmatic result for a completed run.

    Reads the lineage JSON once at end-of-run.  ``record_entry`` flushes
    every accept-time mutation/merge to disk synchronously, so the file
    is the authoritative source by the time we get here.  Falls back to
    each candidate's in-memory ``parent_ids`` / ``operation`` /
    ``generation`` when no lineage entry is present (defensive — every
    real accept site writes one).

    Per-candidate data is materialized once on each
    :class:`CandidateSummary`; :class:`HelixResult` exposes
    ``aggregate_scores`` / ``sum_scores`` / ``instance_scores`` /
    ``objective_scores`` / ``parents`` as cached views over that list,
    so there is a single source of truth.
    """
    lineage = load_lineage(lineage_path)
    non_dominated_ids = sorted(frontier.get_non_dominated())
    non_dominated = set(non_dominated_ids)
    summaries: list[CandidateSummary] = []

    for cid, candidate in frontier.candidates.items():
        entry = lineage.get(cid)
        # Parent resolution priority:
        #   1. ``LineageEntry.parents`` — multi-parent (merge) source of truth.
        #   2. ``Candidate.parent_ids`` — populated at construction for both
        #      mutations and merges, used when lineage has nothing.
        #   3. ``LineageEntry.parent`` — legacy single-parent fallback.
        candidate_parents = (
            list(entry.parents)
            if entry is not None and entry.parents
            else list(candidate.parent_ids)
        )
        if not candidate_parents and entry is not None and entry.parent is not None:
            candidate_parents = [entry.parent]

        result = frontier.get_result(cid)
        aggregate_score = result.aggregate_score() if result is not None else 0.0
        sum_score = result.sum_score() if result is not None else 0.0
        summaries.append(
            CandidateSummary(
                candidate=candidate,
                aggregate_score=aggregate_score,
                sum_score=sum_score,
                scores=dict(result.scores) if result is not None else {},
                instance_scores=(
                    dict(result.instance_scores) if result is not None else {}
                ),
                objective_scores=(
                    list(result.objective_scores)
                    if result is not None and result.objective_scores is not None
                    else None
                ),
                parents=candidate_parents,
                operation=entry.operation if entry is not None else candidate.operation,
                generation=entry.generation if entry is not None else candidate.generation,
                discovered_at_evaluation=state.num_metric_calls_by_discovery.get(cid),
                is_non_dominated=cid in non_dominated,
            )
        )

    summaries.sort(key=lambda s: (s.generation, s.id))
    return HelixResult(
        best_candidate=best,
        best_result=frontier.get_result(best.id),
        candidates=list(frontier.candidates.values()),
        candidate_summaries=summaries,
        # Insertion-order copy of state.frontier — preserves accept order
        # so callers can replay ``frontier_ids`` in chronological order.
        # ``non_dominated_ids`` is sorted (above) for stable diffing.
        frontier_ids=list(state.frontier),
        non_dominated_ids=non_dominated_ids,
        frontier_type=state.frontier_type,
        # Defensive shallow copy — ``state.budget`` is mutated by the
        # budget-accounting code throughout the run; the snapshot must
        # not alias it (GEPA parity: ``GEPAResult.from_state`` shallow-
        # copies every collection coming off the live state).
        budget=replace(state.budget),
        discovery_counts=dict(state.num_metric_calls_by_discovery),
        run_dir=str(base_dir),
        seed=config.rng_seed,
        config_hash=state.config_hash,
    )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_evolution(
    config: HelixConfig,
    project_root: Path,
    base_dir: Path,
) -> HelixResult:
    """Run the HELIX evolutionary loop."""
    # Mirror GEPA api.py:262-265: at least one stopping condition is required.
    # Without this, perfect-skip + always-perfect data + no effective bound =
    # a run that terminates only by the OS.  In HELIX, max_generations (loop
    # bound) and max_evaluations (budget cap, <= 0 disables) are the two
    # supported stopping conditions.  max_generations defaults to 10 and is
    # always a positive int; this guard fires only if a caller explicitly sets
    # it to <= 0 while leaving max_evaluations disabled.
    if (
        config.evolution.max_generations <= 0
        and config.evolution.max_evaluations <= 0
    ):
        raise ValueError(
            "At least one stopping condition is required: set "
            "config.evolution.max_generations to a positive integer, or "
            "config.evolution.max_evaluations > 0. "
            "See GEPA api.py:262-265 for the upstream equivalent check."
        )
    if config.sandbox.enabled and config.sandbox.evaluator:
        if config.evaluator.sidecar is None:
            raise HelixError(
                "Docker sandboxing requires [evaluator.sidecar].",
                operation="start evaluator sidecar",
                suggestion="Configure evaluator.sidecar.image, command, and endpoint.",
            )
        with start_evaluator_sidecar(
            config.evaluator.sidecar,
            passthrough_env=config.passthrough_env,
            fixed_env=config.env,
            extra_hosts=config.sandbox.extra_hosts,
        ):
            return _run_evolution_impl(config, project_root, base_dir)
    return _run_evolution_impl(config, project_root, base_dir)


def _run_evolution_impl(
    config: HelixConfig,
    project_root: Path,
    base_dir: Path,
) -> HelixResult:
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
    HelixResult
        Structured result for the completed run, including the best candidate.
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
    frontier = ParetoFrontier(rng=rng, frontier_type=config.evolution.frontier_type)

    # No-example/single-task cache.  This is deliberately gated by
    # cache_evaluation, the same flag as the per-example cache below.
    eval_cache: EvaluationCache | None = (
        EvaluationCache() if config.evolution.cache_evaluation else None
    )

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
    # Architecture A (example-id handoff): a dataset.train_size
    # synthesises a _RangeDataLoader that yields the ids "0"…"N-1".
    # HELIX writes these ids to {worktree}/helix_batch.json as opaque
    # strings and the evaluator filters its own dataset accordingly.
    # dataset.val_size handles the full-valset evaluation identically.
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
            # 'group_alpha__case_3' -> 'group_alpha' with separator='__'.
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
        MinibatchEvalCache[object, str]() if config.evolution.cache_evaluation else None
    )
    if minibatch_cache is not None:
        _persisted_cache = load_eval_cache(project_root)
        if _persisted_cache is not None:
            minibatch_cache._cache.update(_persisted_cache)
    # NOTE: when cache_evaluation is disabled we DO NOT eagerly delete the
    # persisted ``eval_cache.pkl`` here.  Doing so before ``load_state`` +
    # ``_validate_resume_semantics`` could destroy the cache on a resume
    # that we are about to reject (e.g. wrong frontier_type), leaving the
    # user with no way to fall back to the original config.  The first
    # ``_save_state`` call below performs the delete once we have decided
    # the run is actually proceeding under this config.

    # Acceptance criterion (GEPA §5.1).
    acceptance = (
        StrictImprovementAcceptance()
        if config.evolution.acceptance_criterion == "strict_improvement"
        else ImprovementOrEqualAcceptance()
    )

    # Full-eval policy (GEPA §4.2) — kept for parity and future policy-based
    # val scheduling refactors.
    _full_eval_policy = FullEvaluationPolicy()
    full_val_example_ids = _full_val_example_ids(config, val_loader)
    stage_val_example_ids = _stage_val_example_ids(config, full_val_example_ids)

    # ------------------------------------------------------------------
    # State: load (resume) or initialise (fresh run)
    # ------------------------------------------------------------------
    state = load_state(project_root)
    cfg_hash = _config_hash(config)
    evaluator_manifest = _load_evaluator_integrity_manifest(base_dir)
    current_root_manifest = _build_evaluator_integrity_manifest(
        config=config,
        baseline_root=project_root,
        project_root=project_root,
    )
    if current_root_manifest and current_root_manifest != evaluator_manifest:
        if evaluator_manifest is not None:
            print_warning(
                "Protected evaluator manifest differs from current project root; "
                "refreshing manifest so resumed candidates use the current "
                "evaluator/runtime/split contract."
            )
        evaluator_manifest = current_root_manifest
        _write_evaluator_integrity_manifest(base_dir, evaluator_manifest)

    # GEPA parity (audit-rng-state-persist C1): bundle eval-cache persistence
    # with state.json writes.  GEPA's single ``GEPAState.save`` call pickles
    # the cache atomically alongside everything else (state.py:306-340); HELIX
    # routes the (candidate_id, example_id)-keyed companion pickle through
    # this helper so every save site stays consistent without rewriting them.
    # One-shot guard for the disabled-cache cleanup: ``clear_eval_cache``
    # is idempotent but on every save invocation we'd otherwise issue a
    # fresh ``unlink()`` that always raises FileNotFoundError after the
    # first successful delete.  A flag avoids that noise while still
    # guaranteeing the stale on-disk pickle is cleared at least once per
    # run.
    _eval_cache_cleared = [False]

    def _save_state(s: EvolutionState) -> None:
        save_state(s, project_root)
        if minibatch_cache is not None:
            save_eval_cache(minibatch_cache._cache, project_root)
        elif not _eval_cache_cleared[0]:
            clear_eval_cache(project_root)
            _eval_cache_cleared[0] = True

    def _sync_frontier_state() -> None:
        assert state is not None
        state.frontier = frontier.candidate_ids()
        state.active_frontier = frontier.active_frontier_snapshot()

    if state is None:
        state = EvolutionState(
            generation=0,
            frontier=[],
            instance_scores={},
            budget=BudgetState(),
            config_hash=cfg_hash,
            # Pin the frontier dimensionality to whatever the evolve
            # run uses so ``helix frontier`` / ``helix best`` display
            # with the SAME axis later — even if ``helix.toml``'s
            # ``evolution.frontier_type`` is edited between runs.
            frontier_type=config.evolution.frontier_type,
            resume_semantics=_resume_semantics(config),
        )
        needs_seed = True
    else:
        needs_seed = False
        _validate_resume_semantics(state, config)
        if not state.resume_semantics:
            # Legacy state predating the resume_semantics guard: validation
            # short-circuits, so surface that the guard is dormant for THIS
            # resume but will be active on the next save and onward.
            print_info(
                "No persisted resume_semantics found on this state (legacy run); "
                "the current config will be pinned for future resumes."
            )
            state.resume_semantics = _resume_semantics(config)
        # Keep ``state.frontier_type`` pinned to whatever was stored;
        # it already matches the frontier that was built.  On a legacy
        # state with no persisted field (defaulted to "instance" by
        # ``load_state``) a config change to e.g. "hybrid" does NOT
        # retroactively rebuild the frontier — display stays on
        # "instance" for legacy runs.  A fresh run picks up the new
        # type via the branch above.
        if state.config_hash != cfg_hash:
            print_warning(
                "Config hash differs from the saved state; resuming with the current "
                "config while keeping the existing frontier and history."
            )
        if _reconcile_incomplete_attempts_on_resume(
            state=state,
            base_dir=base_dir,
            worktrees_dir=worktrees_dir,
            lineage_path=lineage_path,
        ):
            _save_state(state)
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
        _sync_frontier_state()

    # ------------------------------------------------------------------
    # Seed evaluation
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Main Loop
    # ------------------------------------------------------------------

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
                background=config.agent.background,
                evaluator_cmd=config.evaluator.command,
                dataset_examples=_dataset_examples,
            )
            # Single attempt, fail-fast — matches GEPA's _generate_seed_candidate.
            # Clean up the empty worktree if generation raises so we don't
            # leave orphaned worktrees behind on failure.
            try:
                usage = generate_seed(seed.worktree_path, seed_prompt, config)
                seed.usage = usage
                budget_api.charge_llm_usage(
                    state,
                    usage,
                    candidate_id=seed.id,
                    source="seed_generation",
                )
            except Exception:
                _safe_remove_worktree(seed, label="failed seed generation")
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
        # is None (single-task/no-example mode, e.g. circle_packing) we
        # cannot key the cache by example id, so we fall back to one evaluator
        # call (counted as one uncached metric call).
        # Pre-refresh on the single-task branch (no example ids): the protected
        # evaluator files must be current before _cached_eval reads them.  The
        # batch branch goes through _cached_evaluate_batch which rewrites the
        # worktree each call, so no pre-refresh is needed there.
        if not full_val_example_ids:
            _refresh_protected_evaluator_files(seed, config, project_root)
        seed_result = _run_full_val_eval(
            seed,
            state,
            full_val_example_ids=full_val_example_ids,
            minibatch_cache=minibatch_cache,
            eval_cache=eval_cache,
            config=config,
            project_root=project_root,
            source_batch="seed_val_batch",
            source_single="seed_val",
        )

        _save_evaluation(base_dir, seed_result)
        frontier.add(seed, seed_result)
        _sync_frontier_state()
        state.instance_scores[seed.id] = seed_result.instance_scores
        # GEPA parity (audit-rng-state-persist C/§3): record per-program
        # discovery budget at the moment the program enters the frontier.
        # Mirrors GEPA core/state.py:537 (``num_metric_calls_by_discovery
        # .append(num_metric_calls_by_discovery_of_new_program)`` inside
        # ``update_state_with_new_program``).
        budget_api.record_discovery_budget(state, seed.id)
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
    with HelixLiveDisplay(
        gen=state.generation,
        total=config.evolution.max_generations,
        cumulative_budget=state.budget,
        frontier=frontier,
        config_evolution=config.evolution,
    ) as live:
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

        gen = start_gen - 1
        while gen < config.evolution.max_generations:
            # GEPA parity (engine.py:649): increment generation UNCONDITIONALLY
            # at the top of the loop body, before any proposal work.  GEPA
            # advances ``state.i`` at line 649, before any proposal logic.
            # Moving the increment here eliminates the helix-original rollback
            # construct that caused the NB-2 infinite-retry scenario
            # (always-perfect data + cached parent eval + no budget cap →
            # gen never advances).  After this change there is no code path
            # that skips or rewinds ``gen``; budget_api.advance_proposal_counter
            # (which increments state.i) continues to run here unconditionally.
            gen += 1
            live.gen = gen
            live.current_usage = UsageStats()
            live.update(phase=f"Starting Generation {gen}")

            budget_api.set_generation(state, gen)
            # GEPA parity (engine.py:649): bump ``state.i`` unconditionally at
            # the top of every iteration so the proposal counter — used by the
            # batch sampler and as a tiebreaker elsewhere — advances regardless
            # of whether we sampled from the frontier or skipped.
            budget_api.advance_proposal_counter(state, source="iteration")
            TRACE.emit(EventType.ITER_START, decision=str(gen))

            if budget_api.budget_exhausted(state, config):
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
            # GEPA-parity note: the *merge operator itself* (helix.merger.merge,
            # invoked below) deliberately diverges from GEPA's deterministic
            # text-component splicing (gepa/proposer/merge.py:155-203).  GEPA
            # candidates are dict[str, str], so a syntactic per-component swap
            # is well-defined; helix candidates are full git worktrees, where
            # LLM-mediated file editing is the only viable approach.  Every
            # surrounding piece — the trigger condition, parent selection,
            # subsample selection, acceptance criterion, full-val post-eval, and
            # rejection control flow — mirrors GEPA verbatim.
            merge_attempted = False
            if (
                config.evolution.merge_enabled
                and merges_due > 0
                and last_iter_found_new_program
                and state.total_merge_invocations
                < config.evolution.max_merge_invocations
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
                        score_map[cid] = sum(inst_scores.values()) / len(inst_scores)

                # GEPA parity (M2): merge candidates must be non-dominated.
                # GEPA merge.py:299-304 uses find_dominator_programs() to filter.
                non_dominated = frontier.get_non_dominated()
                merge_candidate_ids = [
                    cid for cid in frontier._candidates if cid in non_dominated
                ]

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
                        common = set(era_i.instance_scores.keys()) & set(
                            erb_j.instance_scores.keys()
                        )
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

                    merge_id = budget_api.next_merge_id(state, gen)

                    merged = merge(
                        candidate_a=a,
                        candidate_b=b,
                        new_id=merge_id,
                        config=config,
                        base_dir=worktrees_dir,
                        background=config.agent.background,
                        eval_result_a=era,
                        eval_result_b=erb,
                        prepare_worktree=lambda cand: (
                            _refresh_and_snapshot_protected_evaluator_files(
                                cand, config, project_root
                            )
                        ),
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
                        if merged.usage:
                            live.update(usage=merged.usage)
                            budget_api.charge_llm_usage(
                                state,
                                merged.usage,
                                candidate_id=merged.id,
                                source="merge",
                            )

                        merge_tamper = _detect_evaluator_tamper(
                            merged,
                            evaluator_manifest,
                            config,
                            project_root,
                        )
                        if merge_tamper:
                            # Evaluator-tamper reject happens PRE-eval — no
                            # merge was attempted in the GEPA sense
                            # (audit-init-engine.md B3).  Fall through.
                            print_warning(
                                f"Merge {merge_id} touched protected evaluator files "
                                f"({', '.join(merge_tamper)}) -- rejecting."
                            )
                            _safe_remove_worktree(merged, label="tamper-rejected merge candidate")
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
                                _safe_remove_worktree(merged, label="duplicate-desc merge candidate")
                                if merged.id in candidates:
                                    del candidates[merged.id]
                                _save_state(state)
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
                                merged,
                                merge_subsample_ids,
                                minibatch_cache,
                                config,
                                "val",
                                project_root,
                            )
                            merge_result.candidate_id = merged.id
                            budget_api.charge_evaluation(
                                state,
                                num_actual_examples=_merge_evals,
                                candidate_id=merged.id,
                                split="val",
                                source="merge_subsample",
                            )
                            _save_evaluation(base_dir, merge_result)

                            # GEPA parity (Fix 13): mid-generation budget check.
                            if budget_api.budget_exhausted(state, config):
                                print_warning(
                                    "Budget exhausted mid-generation -- stopping."
                                )
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
                            assert set(merge_subsample_ids).issubset(
                                merge_result.instance_scores
                            ), (
                                "merge_subsample_ids must be a subset of merge_result.instance_scores"
                            )
                            merge_score = sum(
                                merge_result.instance_scores[k]
                                for k in merge_subsample_ids
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
                                # Budget accounting charges the uncached
                                # full-val example count; single-task/no-example
                                # evals still charge 0/1 metric calls via _cached_eval.
                                full_val_result = _run_full_val_eval(
                                    merged,
                                    state,
                                    full_val_example_ids=full_val_example_ids,
                                    minibatch_cache=minibatch_cache,
                                    eval_cache=eval_cache,
                                    config=config,
                                    project_root=project_root,
                                    source_batch="merge_full_val_batch",
                                    source_single="merge_full_val",
                                )
                                _save_evaluation(base_dir, full_val_result)

                                if budget_api.budget_exhausted(state, config):
                                    print_warning(
                                        "Budget exhausted during merge full-val eval -- stopping."
                                    )
                                    _save_state(state)
                                    break

                                merges_due -= 1
                                budget_api.record_merge_invocation(state)
                                frontier.add(merged, full_val_result)
                                _sync_frontier_state()
                                state.instance_scores[merged.id] = (
                                    full_val_result.instance_scores
                                )
                                # GEPA parity (audit-rng-state-persist C/§3):
                                # record per-program discovery budget at the
                                # moment the merged program enters the
                                # frontier.  GEPA core/state.py:537.
                                budget_api.record_discovery_budget(state, merged.id)
                            else:
                                print_warning(
                                    f"Merge {merge_id} score {merge_score:.4f} < "
                                    f"max parent {required_score:.4f} -- rejecting."
                                )
                                _safe_remove_worktree(merged, label="score-rejected merge candidate")
                                if merged.id in candidates:
                                    del candidates[merged.id]

                # GEPA parity (M2/B3): only consume this iteration when a merge
                # was actually evaluated (engine.py:719,737).  On any fall-through
                # (triplet None, pair already attempted, overlap fail, merge op
                # failure, tamper reject) we drop into reflective mutation below.
                if merge_attempted:
                    _save_state(state)
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

            # GEPA parity: ``num_parallel_proposals="auto"`` is resolved to an int
            # in ``EvolutionConfig.model_post_init`` (mirrors GEPA
            # optimize_anything._resolve_num_parallel_proposals).  The assert both
            # documents that invariant and narrows the type for mypy --strict.
            _np_raw = config.evolution.num_parallel_proposals
            assert isinstance(_np_raw, int)
            n_proposals = _np_raw

            # -- Architecture D: _ProposalResult (atomic worker result type) --
            from dataclasses import dataclass as _dc  # noqa: PLC0415

            @_dc
            class _ProposalResult:
                """Sealed-union result from one atomic proposal worker (Architecture D).

                Each worker thread runs: parent_eval → skip-perfect → LLM → tamper → child_eval
                atomically (GEPA reflective_mutation.py:268,308,369,420 shape).
                Budget charging is deferred to the sequential acceptance loop
                (GEPA apply_proposal_output parity, reflective_mutation.py:472).

                kind discriminator:
                  "success"    — all steps completed; child_eval_result may be None
                  "skipped"    — skip-perfect fired; child is None
                  "tampered"   — LLM call succeeded but child modified protected files
                  "llm_failed" — mutate() returned None or raised; child is None
                """
                kind: str  # "success" | "skipped" | "tampered" | "llm_failed"
                presample_ctx: object  # (parent, parent_frontier_result, subsample_ids, new_id)
                parent_eval_result: object  # EvalResult | None
                child: object = None  # Candidate | None
                child_eval_result: object = None  # EvalResult | None (success+minibatch only)
                tampered_paths: object = None  # list[str] | None
                parent_n_uncached: int = 0  # for budget charging
                child_n_uncached: int = 0   # for budget charging (success+minibatch only)
                child_usage: object = None  # UsageStats | None


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
            presample_contexts: list[
                tuple[Candidate, EvalResult | None, list[str] | None, str]
            ] = []
            _budget_break = False

            for _p_idx in range(n_proposals):
                if budget_api.budget_exhausted(state, config):
                    _budget_break = True
                    break

                # GEPA parity (engine.py:405-410): the FIRST proposal reuses
                # the iteration slot already created at the top of the outer
                # loop (state.i bumped at evolution.py loop head).  For each
                # ADDITIONAL parallel proposal, GEPA bumps state.i again so
                # the per-proposal counter advances independently of the
                # outer loop.  The bump is unconditional (no minibatch gate).
                if _p_idx > 0:
                    budget_api.advance_proposal_counter(
                        state, source="parallel_proposal"
                    )

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
                if (
                    use_minibatch_gate
                    and train_loader is not None
                    and batch_sampler is not None
                ):
                    subsample_ids = batch_sampler.next_minibatch_ids(
                        train_loader, state
                    )
                    TRACE.emit(
                        EventType.SAMPLE_MINIBATCH,
                        candidate_id=parent.id,
                        example_ids=list(subsample_ids),
                        split="train",
                    )

                # g (gen) and s (mutation_counter) advance together under n_proposals=1
                # / merge_disabled defaults. They diverge when n_proposals > 1 (multiple
                # s-slots per gen) or when merge fires (gen advances without incrementing s).
                new_id = budget_api.next_mutation_id(state, gen)
                presample_contexts.append(
                    (parent, parent_frontier_result, subsample_ids, new_id)
                )


            # -- _run_proposal_worker closure (Architecture D) --
            # Replaces the old _eval_parent + _do_mutate split closures with a
            # single atomic worker that runs the full GEPA execute_proposal shape:
            #   parent_eval (reflective_mutation.py:268) →
            #   skip-perfect (reflective_mutation.py:308) →
            #   LLM mutate (reflective_mutation.py:369) →
            #   tamper check → child_eval (reflective_mutation.py:420).
            # Budget charging is deferred to the sequential acceptance loop
            # (apply_proposal_output parity).
            def _run_proposal_worker(
                pre_ctx: tuple,
            ) -> "_ProposalResult":
                """Atomic proposal worker — GEPA execute_proposal shape.

                Runs inside a ThreadPoolExecutor. All parameters except pre_ctx
                are captured from the enclosing scope (config, project_root,
                frontier, minibatch_cache, eval_cache, worktrees_dir,
                evaluator_manifest, use_minibatch_gate, gen).

                Thread-safety:
                - Reads frontier (read-only during worker phase) ✓
                - Reads config, project_root (immutable) ✓
                - Writes to per-candidate worktree only (no shared mutable state) ✓
                - Budget mutations DEFERRED to sequential acceptance loop ✓
                """
                _parent, _pfr, _sub_ids, _new_id = pre_ctx

                # ---- Step W1: Parent eval ----
                # Minibatch path: run parent eval fresh (bypass cache, GEPA parity).
                # No-minibatch path: run train eval (single-task mode, n=1 in practice).
                _parent_eval: "EvalResult | None" = None
                _parent_n_uncached: int = 0
                if _sub_ids is not None:
                    try:
                        # Bypass minibatch_cache — mirrors GEPA reflective_mutation.py:268
                        _mb, _n = _cached_evaluate_batch(
                            _parent,
                            list(_sub_ids),
                            None,  # bypass minibatch_cache
                            config,
                            "train",
                            project_root,
                        )
                        _mb.candidate_id = _parent.id
                        _parent_eval = _mb
                        _parent_n_uncached = _n
                    except Exception as _pe_exc:
                        print_warning(
                            f"Parent eval for proposal {_new_id} "
                            f"(parent: {_parent.id}, gen {gen}) failed: "
                            f"{type(_pe_exc).__name__}: {_pe_exc} — proposal slot skipped."
                        )
                        return _ProposalResult(
                            kind="llm_failed",
                            presample_ctx=pre_ctx,
                            parent_eval_result=None,
                            parent_n_uncached=0,
                        )
                else:
                    # No-minibatch path: run train eval inside worker.
                    # eval_cache is a shared dict; safe for n=1 (single-task mode).
                    set_phase(HelixPhase.TRAIN_EVALUATION)
                    _train_result, _train_cached = _cached_eval(
                        _parent, config, "train", eval_cache
                    )
                    _train_result.candidate_id = _parent.id
                    _parent_eval = _train_result
                    # Encode was_cached as n_uncached (0 = was_cached, 1 = not cached)
                    _parent_n_uncached = 0 if _train_cached else 1

                # ---- Step W2: Build eval_for_mutate ----
                assert _parent_eval is not None
                _eval_for_mutate = _parent_eval

                # Inject top-k history (read-only from frontier — safe in parallel)
                _eval_for_mutate = _inject_top_k_best_example_history(
                    _eval_for_mutate, frontier
                )

                # ---- Step W3: Skip-perfect check (minibatch path only) ----
                # GEPA parity (reflective_mutation.py:308-327): only gate when
                # subsample_ids is not None (no-minibatch never fires this gate).
                if (
                    config.evolution.perfect_score_threshold is not None
                    and _sub_ids is not None
                    and _parent_eval is not None
                    and all(
                        s >= config.evolution.perfect_score_threshold
                        for s in _parent_eval.instance_scores.values()
                    )
                ):
                    return _ProposalResult(
                        kind="skipped",
                        presample_ctx=pre_ctx,
                        parent_eval_result=_parent_eval,
                        parent_n_uncached=_parent_n_uncached,
                    )

                # ---- Step W4: LLM mutation ----
                try:
                    _child = mutate(
                        parent=_parent,
                        eval_result=_eval_for_mutate,
                        new_id=_new_id,
                        config=config,
                        base_dir=worktrees_dir,
                        background=config.agent.background,
                        prepare_worktree=lambda cand: (
                            _refresh_and_snapshot_protected_evaluator_files(
                                cand, config, project_root
                            )
                        ),
                    )
                except Exception as _mu_exc:
                    # Re-raise PromptArtifactCollisionError (fatal for the whole run)
                    if isinstance(_mu_exc, PromptArtifactCollisionError):
                        raise
                    # For HelixError / RateLimitError and other errors, log and return llm_failed
                    _is_helix_err = isinstance(_mu_exc, HelixError)
                    if _is_helix_err:
                        _mu_exc.operation = (
                            _mu_exc.operation or f"parallel mutate {_new_id}"
                        )
                        print_helix_error(_mu_exc)
                        if isinstance(_mu_exc, RateLimitError):
                            logger.error(
                                "Mutation %s (parent: %s, gen %d) failed after all retries: "
                                "%s: %s — proposal slot skipped.",
                                _new_id, _parent.id, gen,
                                type(_mu_exc).__name__, _mu_exc,
                            )
                            print_error(
                                f"Mutation [bold]{_new_id}[/bold] hit rate limit after all "
                                f"retries — proposal slot skipped. "
                                f"Run [cyan]helix resume[/cyan] when rate limits clear."
                            )
                    else:
                        print_error(
                            f"Parallel mutation {_new_id} (parent: {_parent.id}, gen {gen}) "
                            f"failed with exception:\n{traceback.format_exc()}"
                        )
                    return _ProposalResult(
                        kind="llm_failed",
                        presample_ctx=pre_ctx,
                        parent_eval_result=_parent_eval,
                        parent_n_uncached=_parent_n_uncached,
                    )

                if _child is None:
                    return _ProposalResult(
                        kind="llm_failed",
                        presample_ctx=pre_ctx,
                        parent_eval_result=_parent_eval,
                        parent_n_uncached=_parent_n_uncached,
                    )

                # ---- Step W5: Tamper check ----
                _tampered = _detect_evaluator_tamper(
                    _child, evaluator_manifest, config, project_root
                )
                if _tampered:
                    return _ProposalResult(
                        kind="tampered",
                        presample_ctx=pre_ctx,
                        parent_eval_result=_parent_eval,
                        child=_child,
                        tampered_paths=_tampered,
                        parent_n_uncached=_parent_n_uncached,
                        child_usage=_child.usage if _child.usage else None,
                    )

                # ---- Step W6: Child minibatch eval ----
                _child_eval: "EvalResult | None" = None
                _child_n_uncached: int = 0
                if _sub_ids is not None:
                    _ce, _cn = _cached_evaluate_batch(
                        _child,
                        list(_sub_ids),
                        minibatch_cache,  # use cache for child (unlike parent bypass)
                        config,
                        "train",
                        project_root,
                    )
                    _ce.candidate_id = _child.id
                    _child_eval = _ce
                    _child_n_uncached = _cn

                return _ProposalResult(
                    kind="success",
                    presample_ctx=pre_ctx,
                    parent_eval_result=_parent_eval,
                    child=_child,
                    child_eval_result=_child_eval,
                    parent_n_uncached=_parent_n_uncached,
                    child_n_uncached=_child_n_uncached,
                    child_usage=_child.usage if _child.usage else None,
                )

            # ---- Step 2: Atomic workers (GEPA execute_proposal parity) ----
            # One worker per proposal slot; all run in a single ThreadPoolExecutor.
            # Each worker executes: parent_eval → skip-perfect → LLM → tamper → child_eval.
            # Budget charging is deferred to the sequential acceptance loop (Step 3).
            # GEPA parity: engine.py:422-443, reflective_mutation.py:268,308,369,420.

            if _budget_break and not presample_contexts:
                print_warning("Budget exhausted mid-generation -- stopping.")
                _save_state(state)
                break

            set_phase(HelixPhase.MUTATION)

            worker_results: "list[_ProposalResult | None]" = [None] * len(presample_contexts)

            from concurrent.futures import (  # noqa: PLC0415
                ThreadPoolExecutor as _TPE,
                as_completed as _ac,
            )

            _w_max = min(len(presample_contexts), config.evolution.max_workers)
            if len(presample_contexts) > 1:
                with _TPE(max_workers=_w_max) as _wpool:
                    _wfutures = {
                        _wpool.submit(_run_proposal_worker, _pctx): _widx
                        for _widx, _pctx in enumerate(presample_contexts)
                    }
                    for _wf in _ac(_wfutures):
                        _widx = _wfutures[_wf]
                        try:
                            worker_results[_widx] = _wf.result()
                        except PromptArtifactCollisionError:
                            raise
                        except Exception as _wexc:
                            _wpctx = presample_contexts[_widx]
                            _wid = _wpctx[3]
                            _wparent = _wpctx[0]
                            print_warning(
                                f"Worker for proposal {_wid} "
                                f"(parent: {_wparent.id}, gen {gen}) "
                                f"raised an unexpected exception: "
                                f"{type(_wexc).__name__}: {_wexc} — proposal slot dropped."
                            )
                            worker_results[_widx] = _ProposalResult(
                                kind="llm_failed",
                                presample_ctx=_wpctx,
                                parent_eval_result=None,
                                parent_n_uncached=0,
                            )
            else:
                # n=1 sequential path (or single surviving slot after budget break)
                for _idx, _pctx in enumerate(presample_contexts):
                    worker_results[_idx] = _run_proposal_worker(_pctx)

            # ---- Step 3: Sequential acceptance (sampling order, GEPA engine.py:446-452) ----
            # Read worker results in pre-sampling order.  Budget mutations, lineage
            # writes, and frontier updates are all sequential here
            # (GEPA apply_proposal_output parity, engine.py:449-452).

            _last_eval_result: "EvalResult | None" = None
            _gen_skip_records: "list[dict[str, Any]]" = []
            semantic_skip_count = 0
            retryable_semantic_skip_count = 0

            for _p_idx, wr in enumerate(worker_results):
                if wr is None:
                    continue

                _parent, _parent_frontier_result, _subsample_ids, _new_id = wr.presample_ctx

                # --- Handle non-success kinds ---

                if wr.kind == "llm_failed":
                    # Charge parent eval budget if the parent eval ran successfully
                    # before the LLM call failed.
                    if _subsample_ids is not None and wr.parent_eval_result is not None:
                        budget_api.charge_evaluation(
                            state,
                            num_actual_examples=wr.parent_n_uncached,
                            candidate_id=_parent.id,
                            split="train",
                            source="parent_minibatch",
                        )
                    print_warning(f"Mutation {_new_id} failed -- skipping.")
                    continue

                if wr.kind == "skipped":
                    # Charge parent eval budget
                    if _subsample_ids is not None and wr.parent_eval_result is not None:
                        budget_api.charge_evaluation(
                            state,
                            num_actual_examples=wr.parent_n_uncached,
                            candidate_id=_parent.id,
                            split="train",
                            source="parent_minibatch",
                        )
                    _gen_skip_records.append({
                        "generation": gen,
                        "parent_id": _parent.id,
                        "reason": "perfect_subsample",
                        "parent_eval": wr.parent_eval_result.to_dict(),
                    })
                    print_info(
                        f"Iteration {gen}: all subsample scores perfect for parent "
                        f"{_parent.id}; skipping proposal "
                        f"(GEPA reflective_mutation.py:308-327)."
                    )
                    semantic_skip_count += 1
                    if _subsample_ids is not None:
                        retryable_semantic_skip_count += 1
                    continue

                if wr.kind == "tampered":
                    # Charge parent eval budget
                    if _subsample_ids is not None and wr.parent_eval_result is not None:
                        budget_api.charge_evaluation(
                            state,
                            num_actual_examples=wr.parent_n_uncached,
                            candidate_id=_parent.id,
                            split="train",
                            source="parent_minibatch",
                        )
                    # Charge LLM usage (mutation happened, even if rejected)
                    if wr.child_usage:
                        live.update(usage=wr.child_usage)
                        budget_api.charge_llm_usage(
                            state, wr.child_usage,
                            candidate_id=wr.child.id, source="mutation",
                        )
                    print_warning(
                        f"Mutation {wr.child.id} touched protected evaluator files "
                        f"({', '.join(wr.tampered_paths)}) -- rejecting."
                    )
                    _safe_remove_worktree(wr.child, label="tamper-rejected mutation candidate")
                    continue

                # kind == "success"
                child = wr.child

                # Charge parent eval budget
                if _subsample_ids is not None and wr.parent_eval_result is not None:
                    # Minibatch path
                    budget_api.charge_evaluation(
                        state,
                        num_actual_examples=wr.parent_n_uncached,
                        candidate_id=_parent.id,
                        split="train",
                        source="parent_minibatch",
                    )
                else:
                    # No-minibatch path: parent_n_uncached encodes was_cached
                    # (0 = was cached, 1 = not cached)
                    budget_api.charge_evaluation(
                        state,
                        was_cached=(wr.parent_n_uncached == 0),
                        candidate_id=_parent.id,
                        split="train",
                        source="parent_train_no_minibatch",
                    )

                if budget_api.budget_exhausted(state, config):
                    print_warning("Budget exhausted mid-generation -- stopping.")
                    _save_state(state)
                    _budget_break = True
                    break

                # Charge LLM usage
                if wr.child_usage:
                    live.update(usage=wr.child_usage)
                    budget_api.charge_llm_usage(
                        state, wr.child_usage, candidate_id=child.id, source="mutation",
                    )

                mutations_attempted += 1
                live.update(mutations_attempted=mutations_attempted)

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
                # Two paths — both gate on GEPA's strict-sum acceptance criterion:
                # (a) Minibatch gate: child eval already done in worker → use directly.
                # (b) Legacy no-minibatch: run child eval sequentially here (old behavior).
                _eval_for_mutate = wr.parent_eval_result  # parent baseline

                if (
                    use_minibatch_gate
                    and _subsample_ids is not None
                    and wr.parent_eval_result is not None
                    and wr.child_eval_result is not None
                ):
                    # (a) Minibatch path: child eval result already in worker result
                    gating_result = wr.child_eval_result
                    _last_eval_result = gating_result
                    budget_api.charge_evaluation(
                        state,
                        num_actual_examples=wr.child_n_uncached,
                        candidate_id=child.id,
                        split="train",
                        source="mutation_minibatch_gate",
                    )

                    if budget_api.budget_exhausted(state, config):
                        print_warning("Budget exhausted mid-generation -- stopping.")
                        _save_state(state)
                        _budget_break = True
                        break

                    # Apply the configured acceptance criterion on the
                    # per-instance score vectors (GEPA §5.1).
                    #
                    # GEPA parity (adapter.py:154): a missing instance id in the
                    # parent or child minibatch result is an evaluator bug, not a
                    # benign zero.  Both vectors must cover every id in
                    # ``_subsample_ids`` so the acceptance criterion compares like-for-like.
                    from types import SimpleNamespace as _SN  # noqa: PLC0415

                    assert set(_subsample_ids).issubset(
                        wr.parent_eval_result.instance_scores
                    ), (
                        f"Parent minibatch eval missing ids: "
                        f"{set(_subsample_ids) - set(wr.parent_eval_result.instance_scores)}"
                    )
                    _before = [
                        float(wr.parent_eval_result.instance_scores[str(eid)])
                        for eid in _subsample_ids
                    ]
                    assert set(_subsample_ids).issubset(
                        gating_result.instance_scores
                    ), (
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
                        _save_attempt_result(
                            base_dir, gating_result,
                            status="rejected", reason="minibatch_gate",
                            parent_id=_parent.id, generation=gen,
                            stage="train_minibatch", example_ids=list(_subsample_ids),
                        )
                        TRACE.emit(
                            EventType.ACCEPT_DECISION, candidate_id=child.id,
                            decision="reject", example_ids=list(_subsample_ids),
                            score=float(sum(_after)),
                        )
                        print_warning(
                            f"Minibatch gate: {child.id} rejected "
                            f"(sum {sum(_after):.4f} vs parent {sum(_before):.4f}) -- removing."
                        )
                        _safe_remove_worktree(
                            child, label="minibatch-gate-rejected candidate"
                        )
                        del candidates[child.id]
                        continue
                    else:
                        TRACE.emit(
                            EventType.ACCEPT_DECISION, candidate_id=child.id,
                            decision="accept", example_ids=list(_subsample_ids),
                            score=float(sum(_after)),
                        )

                else:
                    # (b) Legacy no-minibatch path: run child eval sequentially
                    # Single-task/no-example mode still gates on train eval.
                    # The parent baseline comes from the train eval passed to the mutator.
                    set_phase(HelixPhase.MUTATION_GATING)
                    parent_acceptance_result = _eval_for_mutate
                    gating_result, _gating_cached = _cached_eval(
                        child, config, "train", eval_cache
                    )
                    gating_result.candidate_id = child.id
                    _last_eval_result = gating_result
                    # Single-task/no-example mode still charges on train eval.
                    # GEPA parity (MODERATE D — audit-mutation.md C3):
                    # same strict-sum acceptance criterion as minibatch path.
                    budget_api.charge_evaluation(
                        state, was_cached=_gating_cached, candidate_id=child.id,
                        split="train", source="mutation_train_gate",
                    )
                    if budget_api.budget_exhausted(state, config):
                        print_warning("Budget exhausted mid-generation -- stopping.")
                        _save_state(state)
                        _budget_break = True
                        break

                    from types import SimpleNamespace as _SN  # noqa: PLC0415

                    _legacy_before = list(parent_acceptance_result.instance_scores.values())
                    _legacy_after = list(gating_result.instance_scores.values())
                    _legacy_proposal = _SN(
                        subsample_scores_before=_legacy_before,
                        subsample_scores_after=_legacy_after,
                    )
                    if not acceptance.should_accept(_legacy_proposal):
                        parent_sum = sum(_legacy_before)
                        child_sum = sum(_legacy_after)
                        _save_attempt_result(
                            base_dir, gating_result,
                            status="rejected", reason="train_gate",
                            parent_id=_parent.id, generation=gen,
                            stage="train", example_ids=None,
                        )
                        print_warning(
                            f"Acceptance: {child.id} does not improve "
                            f"(child_sum={child_sum:.4f}, parent_sum={parent_sum:.4f}) -- removing."
                        )
                        _safe_remove_worktree(child, label="train-gate-rejected candidate")
                        del candidates[child.id]
                        continue

                # --- Staged val gate ------------------------------------------
                # UNCHANGED from previous architecture — runs sequentially after gating.
                use_val_stage_gate = _has_example_scores(
                    _parent_frontier_result, stage_val_example_ids
                )
                if stage_val_example_ids and use_val_stage_gate:
                    set_phase(HelixPhase.VAL_EVALUATION)
                    stage_result, _n = _cached_evaluate_batch(
                        child, list(stage_val_example_ids), minibatch_cache,
                        config, "val", project_root,
                    )
                    stage_result.candidate_id = child.id
                    _last_eval_result = stage_result
                    budget_api.charge_evaluation(
                        state, num_actual_examples=_n, candidate_id=child.id,
                        split="val", source="mutation_val_stage",
                    )
                    if budget_api.budget_exhausted(state, config):
                        print_warning("Budget exhausted mid-generation -- stopping.")
                        _save_state(state)
                        _budget_break = True
                        break

                    from types import SimpleNamespace as _SN  # noqa: PLC0415

                    _stage_before = _scores_for_example_ids(
                        _parent_frontier_result
                        or EvalResult(
                            candidate_id="", scores={}, asi={}, instance_scores={}
                        ),
                        stage_val_example_ids,
                    )
                    _stage_after = _scores_for_example_ids(stage_result, stage_val_example_ids)
                    _proposal = _SN(
                        subsample_scores_before=_stage_before,
                        subsample_scores_after=_stage_after,
                    )
                    if not acceptance.should_accept(_proposal):
                        _save_attempt_result(
                            base_dir, stage_result,
                            status="rejected", reason="val_stage",
                            parent_id=_parent.id, generation=gen,
                            stage="val_stage", example_ids=list(stage_val_example_ids),
                        )
                        TRACE.emit(
                            EventType.ACCEPT_DECISION, candidate_id=child.id,
                            decision="reject_stage", example_ids=list(stage_val_example_ids),
                            score=float(sum(_stage_after)),
                        )
                        print_warning(
                            f"Val stage: {child.id} rejected on first "
                            f"{len(stage_val_example_ids)} val ids "
                            f"(sum {sum(_stage_after):.4f} vs parent "
                            f"{sum(_stage_before):.4f}) -- removing."
                        )
                        _safe_remove_worktree(child, label="val-stage-rejected candidate")
                        del candidates[child.id]
                        continue

                    TRACE.emit(
                        EventType.ACCEPT_DECISION, candidate_id=child.id,
                        decision="accept_stage", example_ids=list(stage_val_example_ids),
                        score=float(sum(_stage_after)),
                    )
                    print_info(
                        f"Val stage: {child.id} passed on first {len(stage_val_example_ids)} "
                        f"val ids (sum {sum(_stage_after):.4f} vs parent "
                        f"{sum(_stage_before):.4f}); promoting to full val."
                    )

                # --- Val evaluation -------------------------------------------
                # UNCHANGED: run full val eval sequentially after gating.
                set_phase(HelixPhase.VAL_EVALUATION)
                val_result = _run_full_val_eval(
                    child,
                    state,
                    full_val_example_ids=full_val_example_ids,
                    minibatch_cache=minibatch_cache,
                    eval_cache=eval_cache,
                    config=config,
                    project_root=project_root,
                    source_batch="mutation_full_val_batch",
                    source_single="mutation_full_val",
                )
                _last_eval_result = val_result

                if budget_api.budget_exhausted(state, config):
                    print_warning("Budget exhausted mid-generation -- stopping.")
                    _save_state(state)
                    _budget_break = True
                    break

                # --- Update frontier ------------------------------------------
                set_phase(HelixPhase.PARETO_UPDATE)
                _save_evaluation(base_dir, val_result)
                frontier.add(child, val_result)
                _sync_frontier_state()
                state.instance_scores[child.id] = val_result.instance_scores
                # GEPA parity (audit-rng-state-persist C/§3): record per-program
                # discovery budget at the moment the child enters the frontier.
                # GEPA core/state.py:537.
                budget_api.record_discovery_budget(state, child.id)
                TRACE.emit(
                    EventType.FRONTIER_UPDATE,
                    candidate_id=child.id,
                    score=val_result.aggregate_score(),
                )
                # GEPA parity (Fix 7): accepting a new program increments merges_due.
                if (
                    config.evolution.merge_enabled
                    and state.total_merge_invocations
                    < config.evolution.max_merge_invocations
                ):
                    merges_due += 1
                last_iter_found_new_program = True

                mutations_accepted += 1
                live.update(
                    mutations_attempted=mutations_attempted,
                    mutations_accepted=mutations_accepted,
                )

            # Write skip records for this generation (GEPA parity — single JSON list)
            if _gen_skip_records:
                _save_skip_record(base_dir, generation=gen, records=_gen_skip_records)

            # Check semantic skip: all proposals were perfect on the minibatch.
            # GEPA parity (engine.py:649, reflective_mutation.py:308-327): gen was
            # already incremented unconditionally at the top of the loop, so on
            # resume the next iteration will be gen+1 — no rollback, no infinite loop.
            if (
                not any(
                    wr and wr.kind in ("success", "tampered") for wr in worker_results
                )
                and semantic_skip_count
                and retryable_semantic_skip_count == semantic_skip_count
            ):
                _save_state(state)
                TRACE.emit(EventType.ITER_END, decision=f"{gen}:skip")
                continue
            # If budget was exhausted during sequential acceptance, break outer loop.
            if _budget_break:
                break

            # Render at end of generation using the last result seen.
            # Post-UX upgrade: we update the live display but do NOT
            # print a scrolling panel every generation.

            _save_state(state)
            TRACE.emit(EventType.ITER_END, decision=str(gen))

    # ------------------------------------------------------------------
    # Return best
    # ------------------------------------------------------------------
    # Permanent summary after the live display disappears
    render_budget(state.budget, config.evolution)
    render_frontier_table(frontier, frontier._results)

    best = frontier.best()

    print_success(f"Evolution complete.  Best candidate: {best.id}")
    TRACE.emit(EventType.OPT_END, candidate_id=best.id)
    return _build_helix_result(
        best=best,
        frontier=frontier,
        state=state,
        base_dir=base_dir,
        config=config,
        lineage_path=lineage_path,
    )
