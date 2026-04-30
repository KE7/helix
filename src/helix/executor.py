"""HELIX executor: run evaluators in subprocess and collect results."""

from __future__ import annotations

import logging
import os
import shlex
import subprocess
from typing import Any

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
) -> dict[str, str]:
    """Collect arbitrary string info from stdout, stderr, and extra command outputs.

    Args:
        stdout: Main command stdout.
        stderr: Main command stderr.
        extra_outputs: List of (stdout, stderr) tuples from extra_commands.
        config: HelixConfig controlling what to include.

    Returns:
        Dict with keys "stdout", "stderr", "extra_0", "extra_1", etc.
        All values are the FULL output — never truncated.
    """
    asi: dict[str, str] = {}

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
    cmd_tokens = _validate_and_split_command(evaluator.command)
    if config.sandbox.enabled and config.sandbox.evaluator:
        if current_evaluator_sidecar_runtime() is None:
            raise ValueError(
                "Sandboxed sidecar evaluation requires an active evaluator sidecar. "
                "Run evaluations through helix.evolution.run_evolution."
            )
        command_results = run_sandboxed_commands(
            [cmd_tokens, *[_validate_and_split_command(cmd) for cmd in evaluator.extra_commands]],
            cwd=candidate.worktree_path,
            env=env,
            sandbox=config.sandbox,
            scope="evaluator",
            sync_back=False,
            image=sandbox_image,
            agent_backend=config.agent.backend,
        )
        result = command_results[0]
        extra_outputs = [(item.stdout, item.stderr) for item in command_results[1:]]
    else:
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
            returncode, candidate.id, split, error_ctx,
        )

    # Collect ASI
    asi = _collect_asi(stdout, stderr, extra_outputs, config)

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
                "" if len(missing) <= len(sample) else f" ... +{len(missing) - len(sample)} more",
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
