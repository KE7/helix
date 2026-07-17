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
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generic, TypeVar, cast


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
from helix.executor import EvalBatchItem, run_evaluator, run_evaluator_batch
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
from helix.proposals import (
    AllImprovements,
    BestImprovement,
    EvaluatedProposal,
    FailedProposal,
    ProposalTask,
    ProposalAcceptanceCriterion,
    ProposalSelectionStrategy,
    PxNSampling,
    SelectedProposal,
    SkippedProposal,
    TamperedProposal,
    TerminalProposalOutcome,
    TopKImprovements,
)
from helix.sandbox import start_evaluator_sidecar
from helix.state import (
    BudgetState,
    ProposalBatchRecord,
    ProposalCleanupResult,
    ProposalSelectionResult,
    ProposalTaskRecord,
    ProposalTaskStatus,
    build_scheduler_checkpoint,
    checkpoint_batch_after_apply,
    checkpoint_batch_before_dispatch,
    checkpoint_batch_task,
    checkpoint_scheduler_state,
    clear_eval_cache,
    decode_rng_state,
    EvaluationCache,
    EvolutionState,
    load_eval_cache,
    load_state,
    reconcile_interrupted_batches,
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

_T = TypeVar("_T")
_R = TypeVar("_R")


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
            "num_sampled_groups": config.evolution.num_sampled_groups,
            "num_examples_per_group": config.evolution.num_examples_per_group,
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


@dataclass(frozen=True, slots=True)
class _OrderedCallResult(Generic[_R]):
    """One position in a drained bounded call batch.

    Worker exceptions are values here, rather than control flow, so callers can
    account and clean every completed slot in sampled order before deciding
    whether a fatal exception must terminate the run.
    """

    value: _R | None = None
    error: BaseException | None = None


def _run_bounded_ordered(
    items: Sequence[_T],
    worker: Callable[[_T], _R],
    *,
    max_workers: int,
) -> list[_OrderedCallResult[_R]]:
    """Run ``items`` with a worker bound and restore exact input order.

    The pool is always drained.  In particular, this helper never raises a
    worker's exception; fatal-vs-slot classification belongs
    to the phase that owns the corresponding resources and can therefore clean
    siblings before re-raising.  Run-fatal ``BaseException`` subclasses are
    retained only until the pool has drained, then immediately re-raised.
    """

    if not items:
        return []

    ordered: list[_OrderedCallResult[_R] | None] = [None] * len(items)

    def _capture(item: _T) -> _OrderedCallResult[_R]:
        try:
            return _OrderedCallResult(value=worker(item))
        except BaseException as exc:
            return _OrderedCallResult(error=exc)

    # Preserve the historical 1x1 path: it does not pay for a thread-pool hop.
    if len(items) == 1:
        return [_capture(items[0])]

    worker_count = min(len(items), max_workers)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        futures: dict[Future[_OrderedCallResult[_R]], int] = {
            pool.submit(_capture, item): index for index, item in enumerate(items)
        }
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()

    # Every submitted future completed before the executor context exited.
    assert all(result is not None for result in ordered)
    return [result for result in ordered if result is not None]


def _is_fatal_proposal_exception(exc: BaseException) -> bool:
    """Return whether a proposal-stage error invalidates the whole run.

    Evaluator, mutation, rate-limit, container-runtime, and worktree-operation
    failures are isolated to their slot.  Prompt-artifact collisions are a
    security boundary, while ``ValueError``/``TypeError`` represent invalid
    runtime configuration or a broken scheduler contract; those must not be
    converted into a smaller proposal batch.
    """

    return not isinstance(exc, Exception) or isinstance(
        exc,
        (
            PromptArtifactCollisionError,
            ResumeIncompatibleError,
            AssertionError,
            ValueError,
            TypeError,
        ),
    )


def _plan_pxn_tasks(
    *,
    p: int,
    n: int,
    frontier: ParetoFrontier,
    batch_sampler: BatchSampler[str] | None,
    train_loader: "HelixDataLoader | _RangeDataLoader | None",
    state: EvolutionState,
    generation: int,
) -> list[ProposalTask]:
    """Select P parents and reserve their N child slots in parent-major order.

    ``state.i`` advances once per task.  At a parent-group boundary it advances
    immediately before parent selection, which preserves the existing P=K,
    N=1 RNG/event order; within a group it advances before each additional
    sibling minibatch.  Parent selection remains with replacement.
    """

    selected_groups = 0
    sampled_tasks = 0
    advanced_for_current_task = False
    selected_parent: Candidate | None = None

    def _select_parent() -> Candidate:
        nonlocal selected_groups, advanced_for_current_task, selected_parent
        if selected_groups > 0:
            budget_api.advance_proposal_counter(
                state, source="parallel_proposal"
            )
            advanced_for_current_task = True
        selected_parent = frontier.select_parent()
        selected_groups += 1
        return selected_parent

    def _sample_minibatch() -> Sequence[str] | None:
        nonlocal sampled_tasks, advanced_for_current_task
        if sampled_tasks > 0 and not advanced_for_current_task:
            budget_api.advance_proposal_counter(
                state, source="parallel_proposal"
            )
        advanced_for_current_task = False
        sampled_tasks += 1
        if batch_sampler is None or train_loader is None:
            return None
        ids = batch_sampler.next_minibatch_ids(train_loader, state)
        assert selected_parent is not None
        TRACE.emit(
            EventType.SAMPLE_MINIBATCH,
            candidate_id=selected_parent.id,
            example_ids=list(ids),
            split="train",
        )
        return ids

    def _reserve_child_id() -> str:
        return budget_api.next_mutation_id(state, generation)

    tasks = PxNSampling(p, n).sample_tasks(
        select_parent=_select_parent,
        sample_minibatch=_sample_minibatch,
        reserve_child_id=_reserve_child_id,
    )
    assert selected_groups == p
    assert sampled_tasks == p * n
    assert len(tasks) == p * n
    return tasks


def _scheduler_checkpoint(
    rng: _random.Random,
    batch_sampler: BatchSampler[str] | None,
) -> dict[str, Any]:
    """Snapshot the shared frontier/sampler RNG and sampler schedule."""

    sampler_rng_state = rng.getstate()
    sampler_epoch = -1
    shuffled_ids: Sequence[str] = ()
    last_trainset_size = 0
    id_frequencies: Mapping[str, int] | None = None
    fallback: dict[str, Any] | None = None

    if isinstance(batch_sampler, EpochShuffledBatchSampler):
        sampler_rng_state = batch_sampler.rng.getstate()
        sampler_epoch = batch_sampler.epoch
        shuffled_ids = batch_sampler.shuffled_ids
        last_trainset_size = batch_sampler.last_trainset_size
        id_frequencies = {
            str(key): int(value) for key, value in batch_sampler.id_freqs.items()
        }
    elif isinstance(batch_sampler, StratifiedBatchSampler):
        sampler_rng_state = batch_sampler.rng.getstate()
        sampler_epoch = batch_sampler.epoch
        shuffled_ids = batch_sampler.shuffled_ids
        last_trainset_size = batch_sampler.last_trainset_size
        inner = batch_sampler._fallback
        if inner is not None:
            fallback = {
                "rng_state": list(inner.rng.getstate()),
                "epoch": inner.epoch,
                "shuffled_ids": list(inner.shuffled_ids),
                "last_trainset_size": inner.last_trainset_size,
                "id_frequencies": {
                    str(key): int(value) for key, value in inner.id_freqs.items()
                },
            }

    return build_scheduler_checkpoint(
        frontier_rng_state=rng.getstate(),
        sampler_rng_state=sampler_rng_state,
        sampler_epoch=sampler_epoch,
        sampler_shuffled_ids=shuffled_ids,
        sampler_last_trainset_size=last_trainset_size,
        sampler_id_frequencies=id_frequencies,
        sampler_fallback=fallback,
    )


def _restore_scheduler_checkpoint(
    checkpoint: Mapping[str, Any],
    rng: _random.Random,
    batch_sampler: BatchSampler[str] | None,
) -> None:
    """Restore the exact shared RNG/minibatch position from persisted state."""

    raw_frontier_state = checkpoint.get("frontier_rng_state")
    if isinstance(raw_frontier_state, list):
        rng.setstate(decode_rng_state(raw_frontier_state))

    raw_sampler = checkpoint.get("sampler")
    if not isinstance(raw_sampler, Mapping) or batch_sampler is None:
        return
    if not isinstance(
        batch_sampler,
        (EpochShuffledBatchSampler, StratifiedBatchSampler),
    ):
        return
    raw_sampler_rng = raw_sampler.get("rng_state")
    if isinstance(raw_sampler_rng, list):
        batch_sampler.rng.setstate(decode_rng_state(raw_sampler_rng))
    raw_ids = raw_sampler.get("shuffled_ids", [])
    if isinstance(raw_ids, list):
        batch_sampler.shuffled_ids = [str(value) for value in raw_ids]
    batch_sampler.epoch = int(raw_sampler.get("epoch", -1))
    batch_sampler.last_trainset_size = int(
        raw_sampler.get("last_trainset_size", 0)
    )

    raw_frequencies = raw_sampler.get("id_frequencies", {})
    if (
        isinstance(batch_sampler, EpochShuffledBatchSampler)
        and isinstance(raw_frequencies, Mapping)
    ):
        batch_sampler.id_freqs = Counter(
            {str(key): int(value) for key, value in raw_frequencies.items()}
        )

    if not isinstance(batch_sampler, StratifiedBatchSampler):
        return
    raw_fallback = raw_sampler.get("fallback")
    if not isinstance(raw_fallback, Mapping) or not raw_fallback:
        batch_sampler._fallback = None
        return
    inner = EpochShuffledBatchSampler[str](
        minibatch_size=batch_sampler.effective_minibatch_size,
        rng=batch_sampler.rng,
    )
    inner.epoch = int(raw_fallback.get("epoch", -1))
    inner.last_trainset_size = int(raw_fallback.get("last_trainset_size", 0))
    raw_inner_ids = raw_fallback.get("shuffled_ids", [])
    if isinstance(raw_inner_ids, list):
        inner.shuffled_ids = [str(value) for value in raw_inner_ids]
    raw_inner_freqs = raw_fallback.get("id_frequencies", {})
    if isinstance(raw_inner_freqs, Mapping):
        inner.id_freqs = Counter(
            {str(key): int(value) for key, value in raw_inner_freqs.items()}
        )
    raw_inner_rng = raw_fallback.get("rng_state")
    if isinstance(raw_inner_rng, list):
        inner.rng.setstate(decode_rng_state(raw_inner_rng))
    batch_sampler._fallback = inner


def _proposal_terminal_record(
    task: ProposalTask,
    *,
    status: str,
    reason: str,
    child_id: str | None = None,
) -> dict[str, Any]:
    """Return the stable on-disk terminal record for one proposal slot."""

    return {
        "batch_index": task.batch_index,
        "parent_group_index": task.parent_group_index,
        "mutation_index": task.mutation_index,
        "parent_id": task.parent_candidate.id,
        "child_id": child_id or task.reserved_child_id,
        "minibatch_ids": (
            list(task.minibatch_ids) if task.minibatch_ids is not None else None
        ),
        "status": status,
        "reason": reason,
    }


def _save_proposal_terminal_records(
    base_dir: Path,
    *,
    generation: int,
    tasks: Sequence[ProposalTask],
    records: dict[int, dict[str, Any]],
) -> None:
    """Persist currently terminal proposal slots in planned task order."""

    ordered = [records[task.batch_index] for task in tasks if task.batch_index in records]
    batch_dir = base_dir / "proposal_batches"
    _atomic_write_json(batch_dir / f"g{generation}.json", ordered)


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
        missing = set(example_ids) - set(result.instance_scores)
        assert not missing, f"Evaluator result missing ids: {sorted(missing)}"
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

    def _fetcher(ids: list[str]) -> list[str]:
        # HELIX evaluators read batches off disk via helix_batch.json;
        # the "batch" handed to the evaluator callable is just the list
        # of ids to run.  GEPA's fetcher signature is preserved for
        # semantic parity even though we only use the id list itself.
        return list(ids)

    def _evaluator(
        batch: list[str],
        _candidate: dict[str, str],
    ) -> tuple[
        list[object],
        list[float],
        list[dict[str, float]] | None,
        list[dict[str, Any]] | None,
    ]:
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
        # Capture per-example side_info for the cache.  This is keyed by the
        # same (candidate hash, example id) pair as scores/objectives, so a
        # later cache hit can render exactly the diagnostics for the selected
        # examples instead of dropping them or reusing unrelated batch logs.
        side_info_list: list[dict[str, Any]] | None = None
        if fresh.per_example_side_info is not None and len(
            fresh.per_example_side_info
        ) == len(batch):
            side_info_list = [fresh.per_example_side_info[i] for i in range(len(batch))]
        return outputs, scores, obj_list, side_info_list

    (
        _,
        scores_by_id,
        objective_by_id,
        side_info_by_id,
        num_actual_evals,
    ) = cache.evaluate_with_cache_full(
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
    if side_info_by_id is not None:
        per_example_side_info_list = [
            side_info_by_id.get(eid, {}) for eid in example_ids
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
                num_sampled_groups=config.evolution.num_sampled_groups,
                num_examples_per_group=config.evolution.num_examples_per_group,
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

    def _cleanup_interrupted_batch_worktree(
        candidate_id: str,
        worktree_path: Path,
    ) -> bool:
        candidate = Candidate(
            id=candidate_id,
            worktree_path=str(worktree_path),
            branch_name=f"helix/{candidate_id}",
            generation=_gen_from_id(candidate_id),
            parent_id=None,
            parent_ids=[],
            operation="interrupted",
        )
        try:
            remove_worktree(candidate)
        except Exception as exc:
            print_warning(
                f"Could not remove interrupted batch worktree "
                f"{candidate_id}: {exc}"
            )
            return False
        return not worktree_path.exists()

    def _recover_selected_evaluated_tasks() -> None:
        """Finish the crash window between selected-eval and frontier apply."""

        assert state is not None
        for batch in state.proposal_batches:
            for task_record in batch.tasks:
                if not (
                    task_record.status == "evaluated"
                    and task_record.selection == "selected"
                    and not task_record.applied
                ):
                    continue
                result = _load_evaluation(base_dir, task_record.child_id)
                worktree_path = worktrees_dir / task_record.child_id
                if result is None or not worktree_path.exists():
                    # Generic interrupted-batch reconciliation will
                    # terminalize and clean an unrecoverable slot.
                    continue
                if task_record.child_id not in state.frontier:
                    state.frontier.append(task_record.child_id)
                    state.instance_scores[task_record.child_id] = (
                        result.instance_scores
                    )
                if (
                    task_record.child_id
                    not in state.num_metric_calls_by_discovery
                ):
                    budget_api.record_discovery_budget(
                        state,
                        task_record.child_id,
                    )
                checkpoint_batch_task(
                    state,
                    project_root,
                    batch_id=batch.batch_id,
                    task_index=task_record.task_index,
                    status="applied",
                    selection="selected",
                    cleanup="not_required",
                    applied=True,
                    detail="resume recovered selected evaluated task",
                    saver=_save_state,
                )

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
        if state.scheduler_state:
            _restore_scheduler_checkpoint(state.scheduler_state, rng, batch_sampler)
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
        _recover_selected_evaluated_tasks()
        reconcile_interrupted_batches(
            state,
            project_root,
            worktrees_dir=worktrees_dir,
            cleanup_worktree=_cleanup_interrupted_batch_worktree,
            saver=_save_state,
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
                    # GEPA parity (merge.py:94-95): ``find_merge_triplet``
                    # now returns the canonical ``(i, j)`` (lex-sorted),
                    # so ``cid_i <= cid_j`` always — the merge subprocess,
                    # attempted-pair ledger and the description-triplet
                    # dedup all see the same tuple order.
                    cid_i, cid_j, ancestor_id = triplet
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
                    # Resolve the common ancestor for the two-diff merge
                    # prompt (GEPA parity at the file-hunk level: feed the
                    # agent the same three-way structure GEPA's algorithm
                    # uses to attribute changes —
                    # ``gepa/proposer/merge.py:163-191``).  The ancestor
                    # came from ``find_merge_triplet``; resolve it through
                    # the frontier's append-only candidate map.  ``None``
                    # is tolerated downstream — ``merge()`` falls back to
                    # the single A↔B diff when the ancestor isn't
                    # resolvable (defensive: lineage / frontier drift).
                    ancestor_candidate = frontier.candidates.get(ancestor_id)
                    merge_id = budget_api.next_merge_id(state, gen)
                    if ancestor_candidate is None:
                        print_warning(
                            f"Merge {merge_id} ({cid_i} + {cid_j}): common "
                            f"ancestor {ancestor_id} not found in frontier "
                            f"candidate map; falling back to single A↔B "
                            f"diff form for this merge."
                        )

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
                        ancestor=ancestor_candidate,
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
            # Phase 2: Unified P*N mutation scheduler
            #
            # One iteration plans P parent groups with N sibling mutations per
            # group.  Each dispatched phase is drained completely; budget
            # exhaustion only prevents the next phase/batch.
            # =============================================================

            raw_p = config.evolution.num_parallel_proposals
            assert isinstance(raw_p, int)
            p = raw_p
            n = config.evolution.mutations_per_parent
            tasks = _plan_pxn_tasks(
                p=p,
                n=n,
                frontier=frontier,
                batch_sampler=batch_sampler,
                train_loader=train_loader,
                state=state,
                generation=gen,
            )
            batch_id = f"g{gen}-proposals"
            max_in_flight_evaluations = sum(
                2
                * (
                    len(task.minibatch_ids)
                    if task.minibatch_ids is not None
                    else 1
                )
                + len(stage_val_example_ids)
                + (len(full_val_example_ids) if full_val_example_ids else 1)
                for task in tasks
            )
            checkpoint_scheduler_state(
                state,
                project_root,
                _scheduler_checkpoint(rng, batch_sampler),
                saver=_save_state,
            )
            proposal_batch = ProposalBatchRecord(
                batch_id=batch_id,
                generation=gen,
                p=p,
                n=n,
                tasks=[
                    ProposalTaskRecord(
                        batch_id=batch_id,
                        p=p,
                        n=n,
                        task_index=task.batch_index,
                        parent_group=task.parent_group_index,
                        mutation_index=task.mutation_index,
                        parent_id=task.parent_candidate.id,
                        child_id=task.reserved_child_id,
                    )
                    for task in tasks
                ],
            )
            checkpoint_batch_before_dispatch(
                state,
                project_root,
                proposal_batch,
                max_evaluations=config.evolution.max_evaluations,
                max_in_flight_evaluations=max_in_flight_evaluations,
                saver=_save_state,
            )
            for task in tasks:
                checkpoint_batch_task(
                    state,
                    project_root,
                    batch_id=batch_id,
                    task_index=task.batch_index,
                    status="running",
                    saver=_save_state,
                )
            parent_frontier_results = {
                task.batch_index: frontier.get_result(task.parent_candidate.id)
                for task in tasks
            }
            terminal_records: dict[int, dict[str, Any]] = {}
            proposal_outcomes: dict[int, TerminalProposalOutcome] = {}
            cleaned_child_ids: set[str] = set()
            cleanup_results: dict[str, ProposalCleanupResult] = {}
            task_budget_charges = {
                task.batch_index: BudgetState() for task in tasks
            }
            parent_ready: dict[int, tuple[EvalResult, int]] = {}
            child_ready: dict[
                int, tuple[ProposalTask, EvalResult, int, Candidate]
            ] = {}
            evaluated_proposals: list[EvaluatedProposal] = []
            selected_proposals: list[SelectedProposal] = []
            _gen_skip_records: list[dict[str, Any]] = []
            semantic_skip_count = 0
            retryable_semantic_skip_count = 0
            _last_eval_result: EvalResult | None = None
            _budget_break = False

            def _discard_child(
                child: Candidate,
                *,
                label: str,
            ) -> ProposalCleanupResult:
                if child.id in cleaned_child_ids:
                    return cleanup_results[child.id]
                cleaned_child_ids.add(child.id)
                candidates.pop(child.id, None)
                worktree_path = Path(child.worktree_path)
                existed = worktree_path.exists()
                try:
                    remove_worktree(child)
                except Exception as exc:
                    print_warning(
                        f"Could not remove worktree for {label} {child.id}: {exc}"
                    )
                    cleanup: ProposalCleanupResult = "failed"
                else:
                    if existed and worktree_path.exists():
                        cleanup = "failed"
                    elif existed:
                        cleanup = "removed"
                    else:
                        cleanup = "missing"
                cleanup_results[child.id] = cleanup
                return cleanup

            def _record_terminal(
                task: ProposalTask,
                *,
                status: str,
                reason: str,
                child: Candidate | None = None,
            ) -> None:
                if task.batch_index in terminal_records:
                    return
                if status == "accepted":
                    persisted_status: ProposalTaskStatus = "applied"
                elif status == "skipped":
                    persisted_status = "skipped"
                elif status == "tampered":
                    persisted_status = "tampered"
                elif status in {"not_selected", "rejected", "not_applied"}:
                    persisted_status = "rejected"
                else:
                    # Internal labels such as fatal, discarded, and
                    # not_dispatched are detail strings, not persisted enum
                    # values.  They terminalize as failed.
                    persisted_status = "failed"

                if persisted_status == "applied":
                    selection: ProposalSelectionResult = "selected"
                elif status == "not_selected":
                    selection = "not_selected"
                elif status in {"rejected", "not_applied"} or reason in {
                    "val_stage",
                    "full_validation",
                }:
                    selection = "selected"
                else:
                    selection = "not_applicable"

                cleanup: ProposalCleanupResult
                if child is not None and persisted_status != "applied":
                    cleanup = _discard_child(
                        child,
                        label=f"{reason} proposal candidate",
                    )
                else:
                    cleanup = "not_required"

                outcome = proposal_outcomes.get(task.batch_index)
                score_delta = (
                    outcome.improvement
                    if isinstance(outcome, EvaluatedProposal)
                    else None
                )
                persisted_charge = _task_budget_snapshot(task)
                checkpoint_batch_task(
                    state,
                    project_root,
                    batch_id=batch_id,
                    task_index=task.batch_index,
                    status=persisted_status,
                    score_delta=score_delta,
                    selection=selection,
                    cleanup=cleanup,
                    budget_charge=persisted_charge,
                    budget_accounted=True,
                    applied=(persisted_status == "applied"),
                    detail=f"{status}: {reason}",
                    saver=_save_state,
                )
                terminal_records[task.batch_index] = _proposal_terminal_record(
                    task,
                    status=persisted_status,
                    reason=reason,
                    child_id=child.id if child is not None else None,
                )

            def _add_evaluation_charge(task: ProposalTask, units: int) -> None:
                task_budget_charges[task.batch_index].evaluations += units

            def _add_usage_charge(
                task: ProposalTask,
                usage: UsageStats,
            ) -> None:
                charge = task_budget_charges[task.batch_index]
                charge.input_tokens += usage.input_tokens
                charge.output_tokens += usage.output_tokens
                charge.cached_input_tokens += usage.cached_input_tokens
                charge.cache_creation_input_tokens += (
                    usage.cache_creation_input_tokens
                )
                charge.cache_read_input_tokens += usage.cache_read_input_tokens
                charge.reasoning_tokens += usage.reasoning_tokens
                charge.cost_usd += usage.cost_usd

            def _task_budget_snapshot(task: ProposalTask) -> BudgetState:
                charge = task_budget_charges[task.batch_index]
                return BudgetState(
                    evaluations=charge.evaluations,
                    input_tokens=charge.input_tokens,
                    output_tokens=charge.output_tokens,
                    cached_input_tokens=charge.cached_input_tokens,
                    cache_creation_input_tokens=charge.cache_creation_input_tokens,
                    cache_read_input_tokens=charge.cache_read_input_tokens,
                    reasoning_tokens=charge.reasoning_tokens,
                    cost_usd=charge.cost_usd,
                )

            # ---- Parent scoring batch ------------------------------------
            # The same parent can appear in more than one group, and siblings
            # intentionally carry independently sampled minibatches.
            parent_batch_items = [
                EvalBatchItem(
                    candidate=task.parent_candidate,
                    content_key=_candidate_content_key(task.parent_candidate),
                    split="train",
                    instance_ids=task.minibatch_ids,
                )
                for task in tasks
            ]

            def _evaluate_parent(item: EvalBatchItem) -> tuple[EvalResult, int]:
                if item.instance_ids is not None:
                    result, n_uncached = _cached_evaluate_batch(
                        item.candidate,
                        list(item.instance_ids),
                        None,  # parent reflection evals are always fresh
                        config,
                        item.split,
                        project_root,
                    )
                else:
                    result, was_cached = _cached_eval(
                        item.candidate,
                        config,
                        item.split,
                        eval_cache,
                    )
                    n_uncached = 0 if was_cached else 1
                result.candidate_id = item.candidate.id
                return result, n_uncached

            set_phase(HelixPhase.TRAIN_EVALUATION)
            try:
                parent_calls = run_evaluator_batch(
                    parent_batch_items,
                    _evaluate_parent,
                    max_workers=config.evolution.max_workers,
                    config=config,
                )
            except BaseException:
                for task in tasks:
                    _record_terminal(
                        task,
                        status="fatal",
                        reason="fatal_parent_eval_batch",
                    )
                _save_proposal_terminal_records(
                    base_dir,
                    generation=gen,
                    tasks=tasks,
                    records=terminal_records,
                )
                raise
            fatal_parent_error: BaseException | None = None
            proposal_outcome: TerminalProposalOutcome
            for task, parent_call in zip(tasks, parent_calls):
                if parent_call.error is not None:
                    proposal_outcome = FailedProposal(
                        task=task,
                        stage="parent_evaluation",
                        message=(
                            f"{type(parent_call.error).__name__}: "
                            f"{parent_call.error}"
                        ),
                    )
                    proposal_outcomes[task.batch_index] = proposal_outcome
                    _record_terminal(
                        task,
                        status=(
                            "fatal"
                            if _is_fatal_proposal_exception(parent_call.error)
                            else "failed"
                        ),
                        reason="parent_eval",
                    )
                    if _is_fatal_proposal_exception(parent_call.error):
                        fatal_parent_error = fatal_parent_error or parent_call.error
                    else:
                        print_warning(
                            f"Parent eval for proposal {task.reserved_child_id} "
                            f"(parent: {task.parent_candidate.id}, gen {gen}) "
                            f"failed: {type(parent_call.error).__name__}: "
                            f"{parent_call.error} "
                            "— proposal slot skipped."
                        )
                    continue

                assert parent_call.result is not None
                parent_result = parent_call.result
                parent_result.candidate_id = task.parent_candidate.id
                parent_n_uncached = parent_call.num_actual_evaluations
                if task.minibatch_ids is not None:
                    charged = budget_api.charge_evaluation(
                        state,
                        num_actual_examples=parent_n_uncached,
                        candidate_id=task.parent_candidate.id,
                        split="train",
                        source="parent_minibatch",
                    )
                else:
                    charged = budget_api.charge_evaluation(
                        state,
                        was_cached=(parent_n_uncached == 0),
                        candidate_id=task.parent_candidate.id,
                        split="train",
                        source="parent_train_no_minibatch",
                    )
                _add_evaluation_charge(task, charged)

                if (
                    config.evolution.perfect_score_threshold is not None
                    and all(
                        score >= config.evolution.perfect_score_threshold
                        for score in parent_result.instance_scores.values()
                    )
                ):
                    proposal_outcome = SkippedProposal(
                        task=task,
                        parent_eval_result=parent_result,
                        reason="perfect_subsample",
                        parent_n_uncached=parent_n_uncached,
                    )
                    proposal_outcomes[task.batch_index] = proposal_outcome
                    _record_terminal(
                        task,
                        status="skipped",
                        reason="perfect_subsample",
                    )
                    _gen_skip_records.append(
                        {
                            "generation": gen,
                            "parent_id": task.parent_candidate.id,
                            "reason": "perfect_subsample",
                            "parent_eval": parent_result.to_dict(),
                            "batch_index": task.batch_index,
                            "parent_group_index": task.parent_group_index,
                            "mutation_index": task.mutation_index,
                            "child_id": task.reserved_child_id,
                        }
                    )
                    print_info(
                        f"Iteration {gen}: all subsample scores perfect for parent "
                        f"{task.parent_candidate.id}; skipping proposal "
                        f"{task.reserved_child_id}."
                    )
                    semantic_skip_count += 1
                    if task.minibatch_ids is not None:
                        retryable_semantic_skip_count += 1
                    continue

                parent_ready[task.batch_index] = (
                    parent_result,
                    parent_n_uncached,
                )

            _save_proposal_terminal_records(
                base_dir,
                generation=gen,
                tasks=tasks,
                records=terminal_records,
            )
            if fatal_parent_error is not None:
                for task in tasks:
                    if task.batch_index in parent_ready:
                        _record_terminal(
                            task,
                            status="discarded",
                            reason="fatal_sibling_parent_eval",
                        )
                _save_proposal_terminal_records(
                    base_dir,
                    generation=gen,
                    tasks=tasks,
                    records=terminal_records,
                )
                raise fatal_parent_error

            # Parent work was fully drained and charged.  If it crossed the
            # cap, do not start mutation workers; mark every otherwise-ready
            # slot terminal and stop after persistence.
            if budget_api.budget_exhausted(state, config):
                _budget_break = True
                for task in tasks:
                    if task.batch_index not in parent_ready:
                        continue
                    proposal_outcome = FailedProposal(
                        task=task,
                        stage="mutation",
                        message="evaluation budget exhausted before mutation dispatch",
                        parent_eval_result=parent_ready[task.batch_index][0],
                        parent_n_uncached=parent_ready[task.batch_index][1],
                    )
                    proposal_outcomes[task.batch_index] = proposal_outcome
                    _record_terminal(
                        task,
                        status="not_dispatched",
                        reason="budget_exhausted_before_mutation",
                    )

            # ---- Isolated mutation batch ---------------------------------
            mutation_inputs = [
                (task, parent_ready[task.batch_index])
                for task in tasks
                if task.batch_index in parent_ready and not _budget_break
            ]

            def _mutate_one(
                item: tuple[ProposalTask, tuple[EvalResult, int]],
            ) -> tuple[Candidate | None, tuple[str, ...]]:
                task, (parent_result, _) = item
                child = mutate(
                    parent=task.parent_candidate,
                    eval_result=parent_result,
                    new_id=task.reserved_child_id,
                    config=config,
                    base_dir=worktrees_dir,
                    background=config.agent.background,
                    prepare_worktree=lambda candidate: (
                        _refresh_and_snapshot_protected_evaluator_files(
                            candidate, config, project_root
                        )
                    ),
                )
                if child is None:
                    return None, ()
                if child.id != task.reserved_child_id:
                    raise ValueError(
                        "Mutation returned candidate id "
                        f"{child.id!r}; reserved id was "
                        f"{task.reserved_child_id!r}."
                    )
                tampered = tuple(
                    _detect_evaluator_tamper(
                        child,
                        evaluator_manifest,
                        config,
                        project_root,
                    )
                )
                return child, tampered

            set_phase(HelixPhase.MUTATION)
            mutation_calls = _run_bounded_ordered(
                mutation_inputs,
                _mutate_one,
                max_workers=config.evolution.max_workers,
            )
            fatal_mutation_error: BaseException | None = None
            mutation_children: list[Candidate] = []
            for mutation_item, mutation_call in zip(
                mutation_inputs, mutation_calls
            ):
                task, (parent_result, parent_n_uncached) = mutation_item
                if mutation_call.error is not None:
                    proposal_outcome = FailedProposal(
                        task=task,
                        stage="mutation",
                        message=(
                            f"{type(mutation_call.error).__name__}: "
                            f"{mutation_call.error}"
                        ),
                        parent_eval_result=parent_result,
                        parent_n_uncached=parent_n_uncached,
                    )
                    proposal_outcomes[task.batch_index] = proposal_outcome
                    _record_terminal(
                        task,
                        status=(
                            "fatal"
                            if _is_fatal_proposal_exception(mutation_call.error)
                            else "failed"
                        ),
                        reason="mutation",
                    )
                    if _is_fatal_proposal_exception(mutation_call.error):
                        fatal_mutation_error = (
                            fatal_mutation_error or mutation_call.error
                        )
                    elif isinstance(mutation_call.error, HelixError):
                        mutation_call.error.operation = (
                            mutation_call.error.operation
                            or f"parallel mutate {task.reserved_child_id}"
                        )
                        print_helix_error(mutation_call.error)
                    else:
                        print_error(
                            f"Parallel mutation {task.reserved_child_id} "
                            f"(parent: {task.parent_candidate.id}, gen {gen}) "
                            f"failed: {type(mutation_call.error).__name__}: "
                            f"{mutation_call.error}"
                        )
                    continue

                assert mutation_call.value is not None
                child, tampered_paths = mutation_call.value
                if child is None:
                    proposal_outcome = FailedProposal(
                        task=task,
                        stage="mutation",
                        message="mutation returned no candidate",
                        parent_eval_result=parent_result,
                        parent_n_uncached=parent_n_uncached,
                    )
                    proposal_outcomes[task.batch_index] = proposal_outcome
                    _record_terminal(
                        task,
                        status="failed",
                        reason="mutation_returned_none",
                    )
                    continue

                mutation_children.append(child)
                if child.usage:
                    live.update(usage=child.usage)
                    budget_api.charge_llm_usage(
                        state,
                        child.usage,
                        candidate_id=child.id,
                        source="mutation",
                    )
                    _add_usage_charge(task, child.usage)

                if tampered_paths:
                    proposal_outcome = TamperedProposal(
                        task=task,
                        parent_eval_result=parent_result,
                        child_candidate=child,
                        tampered_paths=tampered_paths,
                        parent_n_uncached=parent_n_uncached,
                    )
                    proposal_outcomes[task.batch_index] = proposal_outcome
                    _record_terminal(
                        task,
                        status="tampered",
                        reason="protected_evaluator_files",
                        child=child,
                    )
                    print_warning(
                        f"Mutation {child.id} touched protected evaluator files "
                        f"({', '.join(tampered_paths)}) — rejecting."
                    )
                    _discard_child(
                        child,
                        label="tamper-rejected mutation candidate",
                    )
                    continue

                child_ready[task.batch_index] = (
                    task,
                    parent_result,
                    parent_n_uncached,
                    child,
                )

            # A fatal mutation/setup error ends the run only after every worker
            # has completed and all sibling worktrees have been removed.
            if fatal_mutation_error is not None:
                for child in mutation_children:
                    _discard_child(
                        child,
                        label="sibling of fatal mutation",
                    )
                for task, _, _, child in child_ready.values():
                    _record_terminal(
                        task,
                        status="discarded",
                        reason="fatal_sibling_mutation",
                        child=child,
                    )
                _save_proposal_terminal_records(
                    base_dir,
                    generation=gen,
                    tasks=tasks,
                    records=terminal_records,
                )
                raise fatal_mutation_error

            # Record and snapshot every viable mutation in planned order before
            # cross-candidate scoring.  This makes content hashes stable for
            # deduplication and keeps lineage independent of worker completion.
            for task in tasks:
                prepared = child_ready.get(task.batch_index)
                if prepared is None:
                    continue
                _, _, _, child = prepared
                mutations_attempted += 1
                live.update(mutations_attempted=mutations_attempted)
                candidates[child.id] = child
                record_entry(
                    lineage_path,
                    LineageEntry(
                        id=child.id,
                        parent=task.parent_candidate.id,
                        parents=[task.parent_candidate.id],
                        operation="mutate",
                        generation=gen,
                        files_changed=[],
                    ),
                )
                _save_state(state)
                try:
                    snapshot_candidate(child, f"helix: mutate {child.id}")
                except Exception as exc:
                    proposal_outcome = FailedProposal(
                        task=task,
                        stage="mutation",
                        message=f"{type(exc).__name__}: {exc}",
                        parent_eval_result=prepared[1],
                        child_candidate=child,
                        parent_n_uncached=prepared[2],
                    )
                    proposal_outcomes[task.batch_index] = proposal_outcome
                    _record_terminal(
                        task,
                        status="failed",
                        reason="snapshot",
                        child=child,
                    )
                    _discard_child(child, label="snapshot-failed candidate")
                    del child_ready[task.batch_index]

            # ---- Child scoring batch -------------------------------------
            child_inputs = [
                child_ready[task.batch_index]
                for task in tasks
                if task.batch_index in child_ready and not _budget_break
            ]

            child_batch_items = [
                EvalBatchItem(
                    candidate=child,
                    content_key=_candidate_content_key(child),
                    split="train",
                    instance_ids=task.minibatch_ids,
                )
                for task, _, _, child in child_inputs
            ]

            def _evaluate_child(item: EvalBatchItem) -> tuple[EvalResult, int]:
                if item.instance_ids is not None:
                    result, n_uncached = _cached_evaluate_batch(
                        item.candidate,
                        list(item.instance_ids),
                        minibatch_cache,
                        config,
                        item.split,
                        project_root,
                    )
                else:
                    result, was_cached = _cached_eval(
                        item.candidate,
                        config,
                        item.split,
                        eval_cache,
                    )
                    n_uncached = 0 if was_cached else 1
                result.candidate_id = item.candidate.id
                return result, n_uncached

            set_phase(HelixPhase.MUTATION_GATING)
            try:
                child_calls = run_evaluator_batch(
                    child_batch_items,
                    _evaluate_child,
                    max_workers=config.evolution.max_workers,
                    config=config,
                )
            except BaseException:
                for task, _, _, child in child_inputs:
                    _record_terminal(
                        task,
                        status="discarded",
                        reason="fatal_child_eval_batch",
                        child=child,
                    )
                    _discard_child(
                        child,
                        label="fatal child-evaluation batch",
                    )
                _save_proposal_terminal_records(
                    base_dir,
                    generation=gen,
                    tasks=tasks,
                    records=terminal_records,
                )
                raise
            fatal_child_error: BaseException | None = None
            for child_item, child_call in zip(child_inputs, child_calls):
                task, parent_result, parent_n_uncached, child = child_item
                if child_call.error is not None:
                    proposal_outcome = FailedProposal(
                        task=task,
                        stage="child_evaluation",
                        message=(
                            f"{type(child_call.error).__name__}: "
                            f"{child_call.error}"
                        ),
                        parent_eval_result=parent_result,
                        child_candidate=child,
                        parent_n_uncached=parent_n_uncached,
                    )
                    proposal_outcomes[task.batch_index] = proposal_outcome
                    _record_terminal(
                        task,
                        status=(
                            "fatal"
                            if _is_fatal_proposal_exception(child_call.error)
                            else "failed"
                        ),
                        reason="child_eval",
                        child=child,
                    )
                    if _is_fatal_proposal_exception(child_call.error):
                        fatal_child_error = fatal_child_error or child_call.error
                    else:
                        _discard_child(
                            child,
                            label="child-evaluation-failed candidate",
                        )
                    continue

                assert child_call.result is not None
                child_result = child_call.result
                child_result.candidate_id = child.id
                child_n_uncached = child_call.num_actual_evaluations
                _last_eval_result = child_result
                if task.minibatch_ids is not None:
                    charged = budget_api.charge_evaluation(
                        state,
                        num_actual_examples=child_n_uncached,
                        candidate_id=child.id,
                        split="train",
                        source="mutation_minibatch_gate",
                    )
                else:
                    charged = budget_api.charge_evaluation(
                        state,
                        was_cached=(child_n_uncached == 0),
                        candidate_id=child.id,
                        split="train",
                        source="mutation_train_gate",
                    )
                _add_evaluation_charge(task, charged)
                proposal_outcome = EvaluatedProposal(
                    task=task,
                    parent_eval_result=parent_result,
                    child_candidate=child,
                    child_eval_result=child_result,
                    parent_n_uncached=parent_n_uncached,
                    child_n_uncached=child_n_uncached,
                )
                proposal_outcomes[task.batch_index] = proposal_outcome
                assert isinstance(proposal_outcome, EvaluatedProposal)
                evaluated_proposals.append(proposal_outcome)

            if fatal_child_error is not None:
                for _, _, _, child in child_inputs:
                    _discard_child(
                        child,
                        label="sibling of fatal child evaluation",
                    )
                for proposal in evaluated_proposals:
                    _record_terminal(
                        proposal.task,
                        status="discarded",
                        reason="fatal_sibling_child_eval",
                        child=proposal.child_candidate,
                    )
                _save_proposal_terminal_records(
                    base_dir,
                    generation=gen,
                    tasks=tasks,
                    records=terminal_records,
                )
                raise fatal_child_error

            # Run the configured selection exactly once over all successfully
            # evaluated children.  Acceptance is owned by the selector, so a
            # stateful criterion cannot be accidentally called twice.
            selector: ProposalSelectionStrategy
            if config.evolution.proposal_selection == "all_improvements":
                selector = AllImprovements()
            elif config.evolution.proposal_selection == "best_improvement":
                selector = BestImprovement()
            else:
                assert config.evolution.proposal_top_k is not None
                selector = TopKImprovements(config.evolution.proposal_top_k)
            selected_proposals = selector.select(
                evaluated_proposals,
                cast(ProposalAcceptanceCriterion, acceptance),
            )
            selected_ids = {
                selected.proposal.child_candidate.id
                for selected in selected_proposals
            }

            # Persist and clean every evaluated child not selected.  The
            # default all-improvements mode retains historical gate reasons.
            for proposal in evaluated_proposals:
                child = proposal.child_candidate
                if child.id in selected_ids:
                    continue
                reason = (
                    "minibatch_gate"
                    if proposal.task.minibatch_ids is not None
                    else "train_gate"
                )
                if config.evolution.proposal_selection != "all_improvements":
                    reason = "proposal_selection"
                _save_attempt_result(
                    base_dir,
                    proposal.child_eval_result,
                    status="rejected",
                    reason=reason,
                    parent_id=proposal.task.parent_candidate.id,
                    generation=gen,
                    stage=(
                        "train_minibatch"
                        if proposal.task.minibatch_ids is not None
                        else "train"
                    ),
                    example_ids=(
                        list(proposal.task.minibatch_ids)
                        if proposal.task.minibatch_ids is not None
                        else None
                    ),
                )
                _record_terminal(
                    proposal.task,
                    status="not_selected",
                    reason=reason,
                    child=child,
                )
                TRACE.emit(
                    EventType.ACCEPT_DECISION,
                    candidate_id=child.id,
                    decision="reject",
                    example_ids=(
                        list(proposal.task.minibatch_ids)
                        if proposal.task.minibatch_ids is not None
                        else None
                    ),
                    score=proposal.child_eval_result.sum_score(),
                )
                _discard_child(
                    child,
                    label="proposal-selection-rejected candidate",
                )

            # A dispatched child batch is always fully accounted before this
            # check.  Crossing the cap withholds staged/full validation and
            # cleans selected-but-unapplied children.
            if budget_api.budget_exhausted(state, config):
                _budget_break = True
                for selected in selected_proposals:
                    proposal = selected.proposal
                    _save_attempt_result(
                        base_dir,
                        proposal.child_eval_result,
                        status="rejected",
                        reason="budget_exhausted_after_child_batch",
                        parent_id=proposal.task.parent_candidate.id,
                        generation=gen,
                        stage="train",
                        example_ids=(
                            list(proposal.task.minibatch_ids)
                            if proposal.task.minibatch_ids is not None
                            else None
                        ),
                    )
                    _record_terminal(
                        proposal.task,
                        status="not_applied",
                        reason="budget_exhausted_after_child_batch",
                        child=proposal.child_candidate,
                    )
                    _discard_child(
                        proposal.child_candidate,
                        label="budget-withheld selected candidate",
                    )
                selected_proposals = []

            # ---- Optional staged validation batch ------------------------
            stage_inputs = [
                selected
                for selected in selected_proposals
                if stage_val_example_ids
                and _has_example_scores(
                    parent_frontier_results[selected.proposal.task.batch_index],
                    stage_val_example_ids,
                )
            ]
            stage_input_ids = {
                selected.proposal.child_candidate.id for selected in stage_inputs
            }
            stage_pass_ids: set[str] = set()

            stage_batch_items = [
                EvalBatchItem(
                    candidate=selected.proposal.child_candidate,
                    content_key=_candidate_content_key(
                        selected.proposal.child_candidate
                    ),
                    split="val",
                    instance_ids=tuple(stage_val_example_ids),
                )
                for selected in stage_inputs
            ]

            def _evaluate_stage(item: EvalBatchItem) -> tuple[EvalResult, int]:
                assert item.instance_ids is not None
                result, n_uncached = _cached_evaluate_batch(
                    item.candidate,
                    list(item.instance_ids),
                    minibatch_cache,
                    config,
                    item.split,
                    project_root,
                )
                result.candidate_id = item.candidate.id
                return result, n_uncached

            set_phase(HelixPhase.VAL_EVALUATION)
            try:
                stage_calls = run_evaluator_batch(
                    stage_batch_items,
                    _evaluate_stage,
                    max_workers=config.evolution.max_workers,
                    config=config,
                )
            except BaseException:
                for selected in selected_proposals:
                    proposal = selected.proposal
                    _record_terminal(
                        proposal.task,
                        status="discarded",
                        reason="fatal_val_stage_batch",
                        child=proposal.child_candidate,
                    )
                    _discard_child(
                        proposal.child_candidate,
                        label="fatal staged-validation batch",
                    )
                _save_proposal_terminal_records(
                    base_dir,
                    generation=gen,
                    tasks=tasks,
                    records=terminal_records,
                )
                raise
            fatal_stage_error: BaseException | None = None
            for selected, stage_call in zip(stage_inputs, stage_calls):
                proposal = selected.proposal
                task = proposal.task
                child = proposal.child_candidate
                if stage_call.error is not None:
                    _record_terminal(
                        task,
                        status=(
                            "fatal"
                            if _is_fatal_proposal_exception(stage_call.error)
                            else "failed"
                        ),
                        reason="val_stage",
                        child=child,
                    )
                    if _is_fatal_proposal_exception(stage_call.error):
                        fatal_stage_error = fatal_stage_error or stage_call.error
                    else:
                        _discard_child(
                            child,
                            label="val-stage-evaluation-failed candidate",
                        )
                    continue

                assert stage_call.result is not None
                stage_result = stage_call.result
                stage_result.candidate_id = child.id
                stage_n_uncached = stage_call.num_actual_evaluations
                _last_eval_result = stage_result
                charged = budget_api.charge_evaluation(
                    state,
                    num_actual_examples=stage_n_uncached,
                    candidate_id=child.id,
                    split="val",
                    source="mutation_val_stage",
                )
                _add_evaluation_charge(task, charged)
                parent_frontier_result = parent_frontier_results[task.batch_index]
                assert parent_frontier_result is not None
                before = _scores_for_example_ids(
                    parent_frontier_result,
                    stage_val_example_ids,
                )
                after = _scores_for_example_ids(
                    stage_result,
                    stage_val_example_ids,
                )
                stage_proposal = SimpleNamespace(
                    subsample_scores_before=before,
                    subsample_scores_after=after,
                )
                if not acceptance.should_accept(stage_proposal):
                    _save_attempt_result(
                        base_dir,
                        stage_result,
                        status="rejected",
                        reason="val_stage",
                        parent_id=task.parent_candidate.id,
                        generation=gen,
                        stage="val_stage",
                        example_ids=list(stage_val_example_ids),
                    )
                    _record_terminal(
                        task,
                        status="rejected",
                        reason="val_stage",
                        child=child,
                    )
                    TRACE.emit(
                        EventType.ACCEPT_DECISION,
                        candidate_id=child.id,
                        decision="reject_stage",
                        example_ids=list(stage_val_example_ids),
                        score=float(sum(after)),
                    )
                    _discard_child(
                        child,
                        label="val-stage-rejected candidate",
                    )
                    continue
                stage_pass_ids.add(child.id)
                TRACE.emit(
                    EventType.ACCEPT_DECISION,
                    candidate_id=child.id,
                    decision="accept_stage",
                    example_ids=list(stage_val_example_ids),
                    score=float(sum(after)),
                )

            if fatal_stage_error is not None:
                for selected in selected_proposals:
                    proposal = selected.proposal
                    _discard_child(
                        proposal.child_candidate,
                        label="sibling of fatal staged validation",
                    )
                    if proposal.task.batch_index not in terminal_records:
                        _record_terminal(
                            proposal.task,
                            status="discarded",
                            reason="fatal_sibling_val_stage",
                            child=proposal.child_candidate,
                        )
                _save_proposal_terminal_records(
                    base_dir,
                    generation=gen,
                    tasks=tasks,
                    records=terminal_records,
                )
                raise fatal_stage_error

            # Preserve selector order even when only a subset required the
            # staged gate and those evaluations completed out of order.
            validation_ready = [
                selected
                for selected in selected_proposals
                if (
                    selected.proposal.child_candidate.id not in stage_input_ids
                    or selected.proposal.child_candidate.id in stage_pass_ids
                )
            ]

            if budget_api.budget_exhausted(state, config) and validation_ready:
                _budget_break = True
                for selected in validation_ready:
                    proposal = selected.proposal
                    _record_terminal(
                        proposal.task,
                        status="not_applied",
                        reason="budget_exhausted_after_val_stage",
                        child=proposal.child_candidate,
                    )
                    _discard_child(
                        proposal.child_candidate,
                        label="budget-withheld validation candidate",
                    )
                validation_ready = []

            # ---- Full-validation batch -----------------------------------
            full_batch_items = [
                EvalBatchItem(
                    candidate=selected.proposal.child_candidate,
                    content_key=_candidate_content_key(
                        selected.proposal.child_candidate
                    ),
                    split="val",
                    instance_ids=(
                        tuple(full_val_example_ids)
                        if full_val_example_ids
                        else None
                    ),
                )
                for selected in validation_ready
            ]

            def _evaluate_full(item: EvalBatchItem) -> tuple[EvalResult, int]:
                if item.instance_ids is not None:
                    result, n_uncached = _cached_evaluate_batch(
                        item.candidate,
                        list(item.instance_ids),
                        minibatch_cache,
                        config,
                        item.split,
                        project_root,
                    )
                else:
                    result, was_cached = _cached_eval(
                        item.candidate,
                        config,
                        item.split,
                        eval_cache,
                    )
                    n_uncached = 0 if was_cached else 1
                result.candidate_id = item.candidate.id
                return result, n_uncached

            try:
                full_calls = run_evaluator_batch(
                    full_batch_items,
                    _evaluate_full,
                    max_workers=config.evolution.max_workers,
                    config=config,
                )
            except BaseException:
                for selected in validation_ready:
                    proposal = selected.proposal
                    _record_terminal(
                        proposal.task,
                        status="discarded",
                        reason="fatal_full_validation_batch",
                        child=proposal.child_candidate,
                    )
                    _discard_child(
                        proposal.child_candidate,
                        label="fatal full-validation batch",
                    )
                _save_proposal_terminal_records(
                    base_dir,
                    generation=gen,
                    tasks=tasks,
                    records=terminal_records,
                )
                raise
            fatal_full_error: BaseException | None = None
            full_results: list[tuple[SelectedProposal, EvalResult]] = []
            for selected, full_call in zip(validation_ready, full_calls):
                proposal = selected.proposal
                task = proposal.task
                child = proposal.child_candidate
                if full_call.error is not None:
                    _record_terminal(
                        task,
                        status=(
                            "fatal"
                            if _is_fatal_proposal_exception(full_call.error)
                            else "failed"
                        ),
                        reason="full_validation",
                        child=child,
                    )
                    if _is_fatal_proposal_exception(full_call.error):
                        fatal_full_error = fatal_full_error or full_call.error
                    else:
                        _discard_child(
                            child,
                            label="full-validation-failed candidate",
                        )
                    continue

                assert full_call.result is not None
                val_result = full_call.result
                val_result.candidate_id = child.id
                val_n_uncached = full_call.num_actual_evaluations
                _last_eval_result = val_result
                if full_val_example_ids:
                    charged = budget_api.charge_evaluation(
                        state,
                        num_actual_examples=val_n_uncached,
                        candidate_id=child.id,
                        split="val",
                        source="mutation_full_val_batch",
                    )
                else:
                    charged = budget_api.charge_evaluation(
                        state,
                        was_cached=(val_n_uncached == 0),
                        candidate_id=child.id,
                        split="val",
                        source="mutation_full_val",
                    )
                _add_evaluation_charge(task, charged)
                full_results.append((selected, val_result))

            if fatal_full_error is not None:
                for selected in validation_ready:
                    proposal = selected.proposal
                    _discard_child(
                        proposal.child_candidate,
                        label="sibling of fatal full validation",
                    )
                    if proposal.task.batch_index not in terminal_records:
                        _record_terminal(
                            proposal.task,
                            status="discarded",
                            reason="fatal_sibling_full_validation",
                            child=proposal.child_candidate,
                        )
                _save_proposal_terminal_records(
                    base_dir,
                    generation=gen,
                    tasks=tasks,
                    records=terminal_records,
                )
                raise fatal_full_error

            # Journal every successful full validation in selector order
            # before inserting any selected child into the frontier.  This
            # closes the sibling crash window: once application starts, every
            # selected child has a durable evaluation artifact, full budget
            # charge, and resume-recoverable selected/evaluated task record.
            set_phase(HelixPhase.PARETO_UPDATE)
            for selected, val_result in full_results:
                proposal = selected.proposal
                _save_evaluation(base_dir, val_result)
                checkpoint_batch_task(
                    state,
                    project_root,
                    batch_id=batch_id,
                    task_index=proposal.task.batch_index,
                    status="evaluated",
                    score_delta=selected.improvement,
                    selection="selected",
                    cleanup="not_required",
                    budget_charge=_task_budget_snapshot(proposal.task),
                    budget_accounted=True,
                    applied=False,
                    saver=_save_state,
                )

            # Apply the already-journaled children deterministically.  No
            # budget check may break this loop: every dispatched result was
            # completed, charged, and made recoverable above.
            for selected, val_result in full_results:
                proposal = selected.proposal
                child = proposal.child_candidate
                frontier.add(child, val_result)
                _sync_frontier_state()
                state.instance_scores[child.id] = val_result.instance_scores
                budget_api.record_discovery_budget(state, child.id)
                TRACE.emit(
                    EventType.FRONTIER_UPDATE,
                    candidate_id=child.id,
                    score=val_result.aggregate_score(),
                )
                _record_terminal(
                    proposal.task,
                    status="accepted",
                    reason="selected",
                    child=child,
                )
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

            if budget_api.budget_exhausted(state, config):
                _budget_break = True

            # Every planned task now has a terminal status.  Parent/mutation
            # failures, skipped tasks, unselected children, and accepted
            # children are serialized in parent-major task order.
            for task in tasks:
                if task.batch_index in terminal_records:
                    continue
                _record_terminal(
                    task,
                    status="failed",
                    reason="no_terminal_outcome",
                )
            _save_proposal_terminal_records(
                base_dir,
                generation=gen,
                tasks=tasks,
                records=terminal_records,
            )
            completed_batch = checkpoint_batch_after_apply(
                state,
                project_root,
                batch_id=batch_id,
                saver=_save_state,
            )
            TRACE.emit_proposal_batch_terminal(completed_batch)
            if _gen_skip_records:
                _save_skip_record(
                    base_dir,
                    generation=gen,
                    records=_gen_skip_records,
                )

            if (
                not any(
                    isinstance(
                        outcome,
                        (EvaluatedProposal, TamperedProposal),
                    )
                    for outcome in proposal_outcomes.values()
                )
                and semantic_skip_count
                and retryable_semantic_skip_count == semantic_skip_count
                and not _budget_break
            ):
                _save_state(state)
                TRACE.emit(EventType.ITER_END, decision=f"{gen}:skip")
                continue

            # A cap crossed by any drained phase stops the next batch/iteration,
            # never the accounting or cleanup of work already dispatched.
            if _budget_break:
                _save_state(state)
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
