"""HELIX executor: run evaluators in subprocess and collect results."""

from __future__ import annotations

import copy
import logging
import os
import shlex
import subprocess
import threading
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

from helix.asi import (
    HELIX_ASI_LOG_ENV,
    clear as clear_asi_log,
    read as read_asi_log,
    read_text as read_asi_log_text,
)
from helix.eval_cache import EvaluationBatchKey
from helix.population import Candidate, EvalResult
from helix.config import HelixConfig
from helix.exceptions import EvaluatorError, format_error_context
from helix.parsers import get_parser
from helix.sandbox import (
    current_evaluator_sidecar_runtime,
    run_sandboxed_commands,
)
from helix.trace import TRACE, EventType

logger = logging.getLogger(__name__)

# Differential-testing hook: when set to a callable, ``run_evaluator`` bypasses
# the subprocess/env-scrub path entirely and delegates to the override with
# ``(candidate, split, instance_ids) -> EvalResult``.  The production path is
# untouched when this is ``None`` (default).
_EVALUATOR_OVERRIDE = None


def _validate_and_split_command(cmd: str) -> list[str]:
    """Tokenize a command string for subprocess.run with shell=False.

    On the happy path, returns ``shlex.split(command)``.

    The real safety boundary is ``shell=False``: shell metacharacters in the
    command string are treated as literal arguments, so injection via
    ``helix.toml`` is not possible regardless of the first token.  A
    ``helix.toml`` author is already trusted to run arbitrary code (they
    can simply write ``python -c "..."``), so we do not gate commands by
    an allow-list.
    """
    try:
        tokens = shlex.split(cmd)
    except ValueError as e:
        raise EvaluatorError(
            f"Failed to parse evaluator command: {e}",
            operation="validate_command",
            command=cmd,
        ) from e
    if not tokens:
        raise EvaluatorError(
            "Empty command string",
            operation="validate_command",
            command=cmd,
        )
    return tokens


def _scrub_environment(
    split: str | None = None,
    instance_ids: list[str] | None = None,
    passthrough_env: list[str] | None = None,
    fixed_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create a scrubbed environment with only allowed variables.

    This is the single source of truth for env-scrubbing across HELIX.
    Both the evaluator subprocess (via :func:`run_evaluator`) and the
    Claude Code subprocess (via :func:`~helix.mutator.invoke_claude_code`)
    call this function.

    Args:
        split: Dataset split name to pass as HELIX_SPLIT.  When *None*
            (the default), HELIX_SPLIT is not added — useful for
            non-evaluator subprocesses like Claude Code.
        instance_ids: Optional list of example IDs to evaluate on. Passed
            to the evaluator as HELIX_INSTANCE_IDS (comma-separated).
            Evaluators that honor it restrict to these; others ignore it
            and HELIX post-filters the returned instance_scores.
        passthrough_env: Optional list of extra env var names to preserve
            from the parent process (e.g. CUDA_VISIBLE_DEVICES, HF_HOME).
        fixed_env: Optional mapping of explicit env var values to inject after
            passthrough values. Useful for run-local endpoints captured in
            helix.toml.

    Returns:
        Dict containing only PATH, HOME, HELIX_* variables,
        and any explicitly listed passthrough_env keys.
    """
    scrubbed: dict[str, str] = {}

    # Always include PATH and HOME if available
    if "PATH" in os.environ:
        scrubbed["PATH"] = os.environ["PATH"]
    if "HOME" in os.environ:
        scrubbed["HOME"] = os.environ["HOME"]

    # Add HELIX_SPLIT when running evaluators.
    if split is not None:
        scrubbed["HELIX_SPLIT"] = split

    # Add HELIX_INSTANCE_IDS when a minibatch subset is requested.
    if instance_ids is not None:
        scrubbed["HELIX_INSTANCE_IDS"] = ",".join(str(i) for i in instance_ids)

    # Include any existing HELIX_* variables
    for key, value in os.environ.items():
        if (
            key.startswith("HELIX_")
            and key != "HELIX_SPLIT"
            and key != "HELIX_INSTANCE_IDS"
        ):
            scrubbed[key] = value

    # Include user-specified passthrough variables
    for key in passthrough_env or []:
        if key in os.environ:
            scrubbed[key] = os.environ[key]

    for key, value in (fixed_env or {}).items():
        scrubbed[str(key)] = str(value)

    return scrubbed


def _collect_asi(
    stdout: str,
    stderr: str,
    extra_outputs: list[tuple[str, str]],
    config: HelixConfig,
    *,
    log: str = "",
    returncode: int | None = None,
) -> dict[str, str]:
    """Collect arbitrary string info from evaluator outputs and helix.log notes.

    Args:
        stdout: Main command stdout.
        stderr: Main command stderr.
        extra_outputs: List of (stdout, stderr) tuples from extra_commands.
        config: HelixConfig controlling what to include.
        log: Rendered ``helix.log()`` notes captured from the evaluator's
            ``HELIX_ASI_LOG`` file.  Stored under the ``"log"`` key when
            non-empty so the mutation prompt can render it as
            ``## Evaluator Notes``.
        returncode: Evaluator subprocess exit code.  Stored as a
            stringified ``"_returncode"`` key so downstream consumers
            (notably the mutation prompt builder) can distinguish
            success from failure when deciding whether stdout/stderr
            should be surfaced as fallback debug context.  Kept inside
            ``asi`` rather than promoted to a typed field to preserve
            the GEPA O.A. EvaluationBatch interface.

    Returns:
        Dict with keys "stdout", "stderr", "log", "_returncode",
        "extra_0", "extra_1", etc.  All values are the FULL output —
        never truncated.  Reserved keys consumed by HELIX itself
        (``log``, ``_returncode``) are filtered out before the catch-all
        "extra" rendering in the mutation prompt.
    """
    asi: dict[str, str] = {}

    if returncode is not None:
        asi["_returncode"] = str(returncode)
    if log:
        asi["log"] = log
    if config.evaluator.include_stdout:
        asi["stdout"] = stdout
    if config.evaluator.include_stderr:
        asi["stderr"] = stderr

    for i, (extra_stdout, extra_stderr) in enumerate(extra_outputs):
        asi[f"extra_{i}"] = extra_stdout

    return asi


def run_evaluator(
    candidate: Candidate,
    config: HelixConfig,
    split: str = "val",
    instance_ids: list[str] | None = None,
) -> EvalResult:
    """Run the evaluator for a single candidate.

    Args:
        candidate: The candidate to evaluate.
        config: HelixConfig with evaluator settings.
        split: Dataset split to use (default "val").
        instance_ids: Optional list of example ids to restrict the
            evaluation to (GEPA §5.1 minibatch gate).  Exposed to
            the evaluator via ``HELIX_INSTANCE_IDS`` and applied as
            a post-filter to ``instance_scores`` for evaluators that
            do not honour it.  None → evaluate the whole split.

    Returns:
        EvalResult with scores and instance_scores.

    Note:
        On subprocess failure, full diagnostics are logged
        (never truncated) including the command, full stdout/stderr,
        exit code, and the candidate being evaluated.
    """
    TRACE.emit(
        EventType.EVAL_START,
        candidate_id=candidate.id,
        split=split,
        example_ids=list(instance_ids) if instance_ids is not None else None,
    )

    # Differential-testing short-circuit: when an override callable is
    # installed (see ``_EVALUATOR_OVERRIDE`` at module top), skip subprocess
    # and run the override directly.  Production code paths never set this.
    if _EVALUATOR_OVERRIDE is not None:
        _override_result = _EVALUATOR_OVERRIDE(candidate, split, instance_ids)
        TRACE.emit(
            EventType.EVAL_END,
            candidate_id=candidate.id,
            split=split,
            example_ids=list(instance_ids) if instance_ids is not None else None,
            score=_override_result.aggregate_score()
            if hasattr(_override_result, "aggregate_score")
            else None,
        )
        return _override_result

    evaluator = config.evaluator
    sandbox_image = None
    if config.sandbox.enabled and config.sandbox.evaluator:
        if evaluator.sidecar is None:
            raise ValueError("Sandboxed evaluation requires [evaluator.sidecar].")
        sandbox_image = evaluator.sidecar.resolved_runner_image

    # Run main evaluation command
    env = _scrub_environment(
        split,
        instance_ids=instance_ids,
        passthrough_env=config.passthrough_env,
        fixed_env=config.env,
    )
    helix_log_name = f".helix_asi_log_{uuid.uuid4().hex}.jsonl"
    helix_log_path = Path(candidate.worktree_path) / helix_log_name
    # Absolute path inside the sidecar — the evaluator command may run
    # with a different cwd than the one we mount the worktree at, so the
    # ``HELIX_ASI_LOG`` env var and the trailing capture command both
    # use the absolute ``/workspace/...`` form to remove that
    # assumption.
    helix_log_sandbox_path = f"/workspace/{helix_log_name}"
    cmd_tokens = _validate_and_split_command(evaluator.command)
    if config.sandbox.enabled and config.sandbox.evaluator:
        env[HELIX_ASI_LOG_ENV] = helix_log_sandbox_path
        if current_evaluator_sidecar_runtime() is None:
            raise ValueError(
                "Sandboxed sidecar evaluation requires an active evaluator sidecar. "
                "Run evaluations through helix.evolution.run_evolution."
            )
        # The trailing ``cat`` command captures the JSONL log file from
        # inside the sidecar.  ``2>/dev/null || true`` keeps a missing
        # file (no ``helix.log()`` calls) from looking like a sandbox
        # error.  Cleanup is deliberately asymmetric vs. the host path
        # below: the worktree is disposed by the caller after
        # evaluation, so we do not need to ``rm`` the log inside the
        # sandbox.  If worktrees become reusable, reintroduce an
        # explicit ``rm`` step here to mirror the host ``finally``.
        command_results = run_sandboxed_commands(
            [
                cmd_tokens,
                *[_validate_and_split_command(cmd) for cmd in evaluator.extra_commands],
                [
                    "sh",
                    "-c",
                    f"cat {shlex.quote(helix_log_sandbox_path)} 2>/dev/null || true",
                ],
            ],
            cwd=candidate.worktree_path,
            env=env,
            sandbox=config.sandbox,
            scope="evaluator",
            sync_back=False,
            image=sandbox_image,
            agent_backend=config.agent.backend,
        )
        result = command_results[0]
        helix_log_text = read_asi_log_text(command_results[-1].stdout)
        extra_outputs = [(item.stdout, item.stderr) for item in command_results[1:-1]]
    else:
        # Host path: the same env var also propagates to ``extra_commands``
        # below — those subprocesses will append to the same JSONL file
        # if they call ``helix.log()``.  Notes from extras are merged in
        # invocation order with the main command's notes.
        env[HELIX_ASI_LOG_ENV] = str(helix_log_path)
        try:
            result = subprocess.run(
                cmd_tokens,
                shell=False,
                cwd=candidate.worktree_path,
                capture_output=True,
                text=True,
                env=env,
            )
            extra_outputs = []
            for extra_cmd in evaluator.extra_commands:
                extra_cmd_tokens = _validate_and_split_command(extra_cmd)
                extra_result = subprocess.run(
                    extra_cmd_tokens,
                    shell=False,
                    cwd=candidate.worktree_path,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                extra_outputs.append((extra_result.stdout, extra_result.stderr))
            helix_log_text = read_asi_log(helix_log_path)
        finally:
            clear_asi_log(helix_log_path)

    stdout = result.stdout
    stderr = result.stderr
    returncode = result.returncode

    # Log non-zero exit for diagnostics (full output, never truncated)
    if returncode != 0:
        error_ctx = format_error_context(
            operation=f"evaluate {candidate.id} (split={split})",
            phase="evaluator subprocess (non-zero exit)",
            command=evaluator.command,
            cwd=str(candidate.worktree_path),
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
        )
        logger.info(
            "Evaluator exited with code %d for candidate %s (split=%s):\n%s",
            returncode,
            candidate.id,
            split,
            error_ctx,
        )

    # Collect ASI
    asi = _collect_asi(
        stdout,
        stderr,
        extra_outputs,
        config,
        log=helix_log_text,
        returncode=returncode,
    )

    # Guard: at most one HELIX_RESULT= line is expected.  Multiple is an
    # evaluator-contract violation (race or accidental double-emit).
    # Surface as ``EvaluatorError`` so upstream HelixError handlers in
    # ``evolution.py`` can route it uniformly with the rest of the
    # evaluator-contract failures the parser raises (length mismatch,
    # missing batch file, etc.).  The ``helix_result`` parser does its
    # own reverse-scan; this pre-check fires across all parser paths
    # before any parser runs.  Payload shape is parser-specific —
    # ``helix_result`` takes a list of per-example [score, side_info]
    # pairs; other parsers ignore this line entirely.
    result_line_count = 0
    for line in reversed(stdout.splitlines()):
        if line.startswith("HELIX_RESULT="):
            result_line_count += 1
            if result_line_count > 1:
                raise EvaluatorError(
                    "Multiple HELIX_RESULT= lines found in evaluator output. "
                    "Expected exactly one.",
                    operation="run_evaluator",
                    command=evaluator.command,
                    cwd=str(candidate.worktree_path),
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=returncode,
                )

    # Parse scores.  ``helix_result`` returns a 4-tuple with per-example
    # side_info (GEPA O.A. evaluator contract: one ``(score, side_info)``
    # pair per example) and the per-example ``objective_scores`` harvest
    # from ``side_info["scores"]``.  All other parsers return the
    # 2-tuple ``(scores, instance_scores)``.
    parser = get_parser(evaluator.score_parser)
    per_example_side_info: list[dict[str, Any]] | None = None
    objective_scores: list[dict[str, float]] | None = None

    if evaluator.score_parser == "pytest":
        scores, instance_scores = parser(stdout, stderr)
    elif evaluator.score_parser == "helix_result":
        # helix_result reads ``{worktree}/helix_batch.json`` to recover
        # the id list HELIX wrote pre-invocation and zips it with the
        # per-example ``[score, side_info]`` payload on stdout.
        (
            scores,
            instance_scores,
            per_example_side_info,
            objective_scores,
        ) = parser(returncode, stdout, stderr, candidate.worktree_path)
    else:
        # exitcode, json_accuracy, and other parsers take (returncode, stdout, stderr)
        scores, instance_scores = parser(returncode, stdout, stderr)

    # Post-filter instance_scores when a subset was requested: evaluators
    # that ignore HELIX_INSTANCE_IDS will still have returned the whole
    # split, but the minibatch gate only looks at the requested subset.
    if instance_ids is not None:
        # ``exitcode`` is a global pass/fail parser: it returns
        # ``{"success": score}`` as instance_scores rather than per-example
        # ids.  Broadcasting the global result to all requested ids is
        # correct here — a process that exits 0 passed for all examples; one
        # that exits non-zero failed for all examples.  Without this
        # broadcast every requested id lands in ``missing`` and gets filled
        # with 0.0, causing spurious rejections when the evaluator exited 0.
        if evaluator.score_parser == "exitcode":
            global_score = instance_scores.get("success", 0.0)
            instance_scores = {str(eid): global_score for eid in instance_ids}
        else:
            filtered: dict[str, float] = {}
            missing: list[str] = []
            for eid in instance_ids:
                eid_s = str(eid)
                if eid_s in instance_scores:
                    filtered[eid_s] = instance_scores[eid_s]
                else:
                    # Evaluator produced no result for this id → 0.0
                    filtered[eid_s] = 0.0
                    missing.append(eid_s)
            if missing:
                # Diagnostic: the silent zero-fill above used to hide evaluator
                # bugs — most infamously an ``instance_scores`` dict keyed by
                # aggregate metric names (``task__metric``) instead of the
                # per-example ids HELIX writes to ``helix_batch.json``
                # (``task__trialN``).  That mismatch made strict-improvement
                # acceptance compare ``0.0 vs 0.0`` for 113 straight generations
                # in one real run.  The per-example ``helix_result`` contract
                # removes that class of bug at the parser level, but this
                # warning is still useful defense in depth: e.g. when a user
                # picks ``score_parser="exitcode"`` and then asks for a
                # minibatch subset, every requested id lands here.
                sample = missing[:5]
                logger.warning(
                    "evaluator returned %d/%d missing instance_scores for "
                    "requested ids (sample: %r%s); these were filled with 0.0. "
                    "If you need per-id scores for the minibatch gate, use "
                    "score_parser='helix_result' (per-example contract — "
                    "HELIX reads helix_batch.json and zips it with your list "
                    "of [score, side_info] pairs).",
                    len(missing),
                    len(instance_ids),
                    sample,
                    ""
                    if len(missing) <= len(sample)
                    else f" ... +{len(missing) - len(sample)} more",
                )
            instance_scores = filtered

    _result = EvalResult(
        candidate_id=candidate.id,
        scores=scores,
        asi=asi,
        instance_scores=instance_scores,
        # ``side_info`` (legacy batch-level dict) is no longer populated
        # by the executor.  The per-example list in
        # ``per_example_side_info`` replaces it for the reflection path.
        per_example_side_info=per_example_side_info,
        # ``objective_scores`` — per-example ``side_info["scores"]``
        # harvest.  Feeds the multi-axis Pareto frontier
        # (``ParetoFrontier._update_objective`` /
        # ``_update_cartesian``) when
        # ``config.evolution.frontier_type`` is ``"objective"``,
        # ``"hybrid"``, or ``"cartesian"``.
        objective_scores=objective_scores,
    )
    TRACE.emit(
        EventType.EVAL_END,
        candidate_id=candidate.id,
        split=split,
        example_ids=list(instance_ids) if instance_ids is not None else None,
        score=_result.aggregate_score(),
    )
    return _result


# ---------------------------------------------------------------------------
# Step 3: ordered cross-candidate evaluator batching
# ---------------------------------------------------------------------------
#
# ``run_evaluator_batch`` runs a list of evaluator requests concurrently under
# a bounded worker pool, returns one result per input in the SAME order, and
# collapses cross-candidate duplicates so identical evaluator work runs once.
#
# Two requests are duplicates iff their :class:`EvaluationBatchKey` matches —
# ``(content_key, split, instance_ids)``.  Candidate *lineage* ids are absent
# from the key on purpose: two distinct candidates whose committed content,
# split, and exact ordered minibatch are identical evaluate to the same scores,
# so only the first (the "leader") is dispatched.  Every later request sharing
# that key (a "follower") reuses the leader's outcome and is charged
# ``num_actual_evaluations == 0`` so evaluation-budget accounting never
# double-counts shared work, including when the shared outcome is a failure.
# ``instance_ids`` is carried and hashed as an ordered tuple because evaluator
# side information is positional to ``helix_batch.json`` — reordering the
# minibatch is a *different* request.


@dataclass(frozen=True)
class EvalBatchItem:
    """One evaluator request submitted to :func:`run_evaluator_batch`.

    ``content_key`` is the caller-computed cache identity of the candidate's
    committed content (see ``evolution._candidate_content_key``); it — not
    ``candidate.id`` — participates in cross-candidate deduplication.
    ``instance_ids`` is an ordered tuple (``None`` → evaluate the whole split);
    its order is significant and preserved through hashing and dispatch.
    """

    candidate: Candidate
    content_key: str
    split: str
    instance_ids: tuple[str, ...] | None

    @property
    def dedup_key(self) -> EvaluationBatchKey:
        """Cross-candidate dedup identity: ``(content_key, split, instance_ids)``."""
        return EvaluationBatchKey(
            content_key=self.content_key,
            split=self.split,
            instance_ids=self.instance_ids,
        )


@dataclass(frozen=True)
class EvalBatchResult:
    """One result slot from :func:`run_evaluator_batch`, positional to input.

    Exactly one of ``result``/``error`` is populated:

    * ``result`` set, ``error`` None  → the evaluator succeeded.
    * ``result`` None, ``error`` set  → a per-item failure or fatal contract /
      integrity error; the rest of the drained batch remains positional so its
      successful work can be accounted before the owner re-raises.

    ``num_actual_evaluations`` is the evaluation-budget charge for this slot.
    A successful leader carries the runner's returned count; a failed leader
    carries the progress reported with :func:`record_evaluator_units` before
    the exception; and every deduplicated follower is charged ``0``.  Thus a
    failure never erases work that already ran and shared work is never charged
    twice.  ``deduplicated_from`` is ``None`` for a leader (the request that
    actually ran) or the input index of the leader this follower reused.
    """

    item: EvalBatchItem
    result: EvalResult | None
    error: BaseException | None
    num_actual_evaluations: int
    deduplicated_from: int | None


# The runner seam: ``runner(item) -> (EvalResult, num_actual_evaluations)``.
# Production wires this to a closure over the per-example cache (see
# ``evolution._cached_evaluate_batch``) so ``num_actual_evaluations`` reflects
# cache hits; tests inject fakes to exercise ordering/dedup/failure/fatal paths
# without a subprocess.
BatchEvaluatorRunner: TypeAlias = Callable[[EvalBatchItem], tuple[EvalResult, int]]

# A runner reports actual work at the dispatch boundary, before an evaluator
# attempt can raise and erase its normal ``(result, count)`` return value.  The
# context is installed independently in each leader worker, so concurrent
# leaders cannot leak units into one another.  Outside ``run_evaluator_batch``
# the helper is intentionally a no-op: the ordinary sequential call sites use
# their existing direct budget charge.
_EVALUATION_UNIT_REPORTER: ContextVar[Callable[[int], None] | None] = ContextVar(
    "helix_evaluation_unit_reporter",
    default=None,
)


def record_evaluator_units(units: int) -> None:
    """Record newly dispatched evaluator units for the current batch leader.

    Runner implementations that can fail after dispatch must call this once at
    the point each group of uncached units becomes actual.  Calls accumulate;
    cache hits report nothing.  The successful runner return remains the
    authoritative count for compatibility with existing one-argument runners.
    """
    if units < 0:
        raise ValueError(f"evaluator units must be >= 0, got {units}")
    reporter = _EVALUATION_UNIT_REPORTER.get()
    if reporter is not None and units:
        reporter(units)


# Per-worktree serialization: two leaders that share a worktree path must not
# run concurrently, because the evaluator handoff writes ``helix_batch.json``
# into that worktree and parallel runs would clobber each other's batch file.
# Distinct worktrees evaluate concurrently.  Mirrors ``evolution._worktree_lock``
# but is owned here so the batch runner is self-contained.
_BATCH_WORKTREE_LOCKS: dict[str, threading.Lock] = {}
_BATCH_WORKTREE_LOCKS_MUTEX = threading.Lock()


def _batch_worktree_lock(worktree_path: str) -> threading.Lock:
    key = str(worktree_path)
    with _BATCH_WORKTREE_LOCKS_MUTEX:
        lock = _BATCH_WORKTREE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _BATCH_WORKTREE_LOCKS[key] = lock
        return lock


def _clone_result_for_follower(result: EvalResult, candidate_id: str) -> EvalResult:
    """Return an independent ``EvalResult`` for a deduplicated follower.

    The leader's :class:`~helix.population.EvalResult` is deep-copied so a
    follower slot never aliases the leader's mutable ``scores`` /
    ``instance_scores`` / ``asi`` / side-info containers — HELIX relabels
    ``candidate_id`` and mutates these dicts per candidate downstream (see the
    many ``result.candidate_id = ...`` sites in ``evolution.py``), so a shared
    object would silently corrupt every peer slot.  ``candidate_id`` is set to
    the follower's own candidate to preserve positional candidate identity.
    """
    cloned = copy.deepcopy(result)
    cloned.candidate_id = candidate_id
    return cloned


def make_default_batch_runner(config: HelixConfig) -> BatchEvaluatorRunner:
    """Default runner seam: evaluate each item via :func:`run_evaluator`.

    ``num_actual_evaluations`` is the number of examples actually sent to the
    evaluator: ``len(instance_ids)`` for a minibatch, else the number of
    instance scores the whole-split run produced.  Production overrides this
    with a cache-aware runner so hits are charged ``0``.
    """

    def _run(item: EvalBatchItem) -> tuple[EvalResult, int]:
        instance_ids = (
            list(item.instance_ids) if item.instance_ids is not None else None
        )
        # The default runner has no cache.  Record before invoking the
        # evaluator so an exception after dispatch retains the attempted work.
        # Whole-split calls consume one HELIX evaluation-budget unit.
        record_evaluator_units(
            len(item.instance_ids) if item.instance_ids is not None else 1
        )
        result = run_evaluator(
            item.candidate,
            config,
            split=item.split,
            instance_ids=instance_ids,
        )
        count = (
            len(item.instance_ids)
            if item.instance_ids is not None
            else len(result.instance_scores)
        )
        return result, count

    return _run


def run_evaluator_batch(
    items: Sequence[EvalBatchItem],
    runner: BatchEvaluatorRunner,
    *,
    max_workers: int,
    config: HelixConfig | None = None,
) -> list[EvalBatchResult]:
    """Run evaluator requests concurrently, deduplicated, in input order.

    Args:
        items: Evaluator requests.  The returned list is positional to this
            sequence — ``result[i]`` corresponds to ``items[i]``.
        runner: The runner seam, ``runner(item) -> (EvalResult, count)``.  Only
            *leaders* (the first item bearing a given
            :attr:`EvalBatchItem.dedup_key`) are handed to the runner; followers
            reuse the leader's outcome.  A runner must call
            :func:`record_evaluator_units` when uncached work is dispatched if
            later operations can fail before it returns its count.
        max_workers: Upper bound on concurrent runner invocations.  The pool is
            additionally capped at the number of leaders.  Must be ``>= 1``.
        config: When provided, the evaluator command is validated ONCE before
            any dispatch; a malformed command raises :class:`EvaluatorError`
            here rather than as N identical per-item failures.

    Returns:
        One :class:`EvalBatchResult` per input item, in order.

    Raises:
        The pre-dispatch config error (e.g. :class:`EvaluatorError` from command
        validation, or ``ValueError`` for ``max_workers < 1``) before any work
        starts.  Once dispatch begins, every runner exception -- including
        ``BaseException`` subclasses and run-fatal integrity failures -- is
        returned in its positional :attr:`EvalBatchResult.error` slot after the
        executor drains.  The owning evolution phase can then account every
        reported attempt, account successful siblings, and clean resources
        before re-raising the fatal error.
    """
    # --- Pre-dispatch config validation (propagates; never per-item) --------
    if max_workers < 1:
        raise ValueError(f"max_workers must be >= 1, got {max_workers}")
    if config is not None:
        # Fail the whole batch once on a malformed command rather than letting
        # every leader raise the identical EvaluatorError inside a worker.
        _validate_and_split_command(config.evaluator.command)

    results: list[EvalBatchResult | None] = [None] * len(items)
    if not items:
        return []

    # --- Deduplicate: first occurrence of a key is the leader --------------
    leader_index_by_key: dict[EvaluationBatchKey, int] = {}
    leader_of: list[int] = []  # leader_of[i] = input index of i's leader
    leader_indices: list[int] = []
    for i, item in enumerate(items):
        key = item.dedup_key
        leader = leader_index_by_key.get(key)
        if leader is None:
            leader_index_by_key[key] = i
            leader_of.append(i)
            leader_indices.append(i)
        else:
            leader_of.append(leader)

    # --- Dispatch leaders concurrently, serialized per worktree ------------
    # leader input index -> (result, count) on success, or a captured error.
    leader_outcome: dict[int, tuple[EvalResult, int]] = {}
    leader_error: dict[int, BaseException] = {}
    leader_reported_units: dict[int, int] = {}

    def _run_leader(index: int) -> None:
        item = items[index]
        reported_units = 0

        def _record(units: int) -> None:
            nonlocal reported_units
            reported_units += units

        token = _EVALUATION_UNIT_REPORTER.set(_record)
        try:
            with _batch_worktree_lock(item.candidate.worktree_path):
                leader_outcome[index] = runner(item)
        finally:
            _EVALUATION_UNIT_REPORTER.reset(token)
            leader_reported_units[index] = reported_units

    pool_workers = min(max_workers, len(leader_indices))
    with ThreadPoolExecutor(max_workers=pool_workers) as pool:
        futures = {pool.submit(_run_leader, idx): idx for idx in leader_indices}
        # ``future.result()`` re-raises whatever the worker raised.  Capture all
        # post-dispatch failures as positional outcomes; returning partial
        # successes is what lets the caller conserve accounting before it
        # decides whether a fatal slot must terminate the run.
        for future, idx in futures.items():
            try:
                future.result()
            except BaseException as exc:
                leader_error[idx] = exc

    # --- Validate runner-reported counts before assembly -------------------
    # A negative evaluation charge is a fatal runner-contract violation, but it
    # is still a *positional* post-dispatch outcome.  Convert only that leader
    # to an error so successful siblings survive assembly and the owning phase
    # can account them before re-raising the ValueError.
    for idx, (_res, count) in list(leader_outcome.items()):
        if count < 0:
            leader_error[idx] = ValueError(
                "runner reported negative num_actual_evaluations "
                f"({count}) for item index {idx}"
            )
            del leader_outcome[idx]

    # --- Assemble results in input order ------------------------------------
    for i, item in enumerate(items):
        leader = leader_of[i]
        is_leader = leader == i
        if leader in leader_error:
            # A failed leader retains units reported before the exception.
            # Followers reuse the same error but remain zero-charge: the
            # leader alone owns the shared evaluator attempt.
            results[i] = EvalBatchResult(
                item=item,
                result=None,
                error=leader_error[leader],
                num_actual_evaluations=(
                    leader_reported_units.get(leader, 0) if is_leader else 0
                ),
                deduplicated_from=None if is_leader else leader,
            )
        elif is_leader:
            leader_result, count = leader_outcome[leader]
            results[i] = EvalBatchResult(
                item=item,
                result=leader_result,
                error=None,
                num_actual_evaluations=count,
                deduplicated_from=None,
            )
        else:
            # Deduplicated follower: hand back an INDEPENDENT clone relabeled to
            # this slot's candidate, charged zero.  Never share the leader's
            # mutable EvalResult (see ``_clone_result_for_follower``).
            leader_result, _count = leader_outcome[leader]
            results[i] = EvalBatchResult(
                item=item,
                result=_clone_result_for_follower(leader_result, item.candidate.id),
                error=None,
                num_actual_evaluations=0,
                deduplicated_from=leader,
            )

    # Cardinality guard: exactly one result per input, all populated.
    assert all(r is not None for r in results) and len(results) == len(items)
    return [r for r in results if r is not None]
