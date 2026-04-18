"""HELIX executor: run evaluators in subprocess and collect results."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from helix.population import Candidate, EvalResult
from helix.config import HelixConfig, EvaluatorConfig
from helix.exceptions import EvaluatorError, print_helix_error, format_error_context
from helix.parsers import get_parser
from helix.trace import TRACE, EventType

logger = logging.getLogger(__name__)

# Differential-testing hook: when set to a callable, ``run_evaluator`` bypasses
# the subprocess/env-scrub path entirely and delegates to the override with
# ``(candidate, split, instance_ids) -> EvalResult``.  The production path is
# untouched when this is ``None`` (default).
_EVALUATOR_OVERRIDE = None


def _validate_and_split_command(cmd: str) -> list[str]:
    """Tokenize a command string for subprocess.run with shell=False.

    The real safety boundary is ``shell=False``: shell metacharacters in the
    command string are treated as literal arguments, so injection via
    ``helix.toml`` is not possible regardless of the first token.  A
    ``helix.toml`` author is already trusted to run arbitrary code (they
    can simply write ``python -c "..."``), so we do not gate commands by
    an allow-list.
    """
    tokens = shlex.split(cmd)
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

    # Run main evaluation command
    env = _scrub_environment(split, instance_ids=instance_ids, passthrough_env=config.passthrough_env)
    cmd_tokens = _validate_and_split_command(evaluator.command)
    result = subprocess.run(
        cmd_tokens,
        shell=False,
        cwd=candidate.worktree_path,
        capture_output=True,
        text=True,
        env=env,
    )

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

    # Run extra commands and collect output
    extra_outputs: list[tuple[str, str]] = []
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

    # Collect ASI
    asi = _collect_asi(stdout, stderr, extra_outputs, config)

    # Check for HELIX_RESULT= line (GEPA OA contract: [score, side_info_dict])
    # Exactly zero or one HELIX_RESULT= line is expected; multiple is an evaluator bug.
    helix_result_score = None
    side_info = None
    result_line = None
    for line in reversed(stdout.splitlines()):
        if line.startswith("HELIX_RESULT="):
            if result_line is not None:
                raise RuntimeError(
                    "Multiple HELIX_RESULT= lines found in evaluator output. "
                    "Expected exactly one."
                )
            result_line = line
    if result_line is not None:
        line = result_line
        try:
            payload = json.loads(line[len("HELIX_RESULT="):])
            if isinstance(payload, list) and len(payload) == 2:
                helix_result_score = float(payload[0])
                if isinstance(payload[1], dict):
                    side_info = payload[1]
        except (json.JSONDecodeError, ValueError, TypeError):
            logger.warning("Failed to parse HELIX_RESULT= line: %s", line)

    # Parse scores (fallback path, always runs for instance_scores)
    parser = get_parser(evaluator.score_parser)
    if evaluator.score_parser == "pytest":
        scores, instance_scores = parser(stdout, stderr)
    else:
        # exitcode, json_accuracy, and other parsers take (returncode, stdout, stderr)
        scores, instance_scores = parser(returncode, stdout, stderr)

    # If HELIX_RESULT= provided a score, override the parser-derived aggregate
    if helix_result_score is not None:
        scores["success"] = helix_result_score

    # Post-filter instance_scores when a subset was requested: evaluators
    # that ignore HELIX_INSTANCE_IDS will still have returned the whole
    # split, but the minibatch gate only looks at the requested subset.
    if instance_ids is not None:
        filtered: dict[str, float] = {}
        for eid in instance_ids:
            eid_s = str(eid)
            if eid_s in instance_scores:
                filtered[eid_s] = instance_scores[eid_s]
            else:
                # Evaluator produced no result for this id → 0.0
                filtered[eid_s] = 0.0
        instance_scores = filtered

    _result = EvalResult(
        candidate_id=candidate.id,
        scores=scores,
        asi=asi,
        instance_scores=instance_scores,
        side_info=side_info,
    )
    TRACE.emit(
        EventType.EVAL_END,
        candidate_id=candidate.id,
        split=split,
        example_ids=list(instance_ids) if instance_ids is not None else None,
        score=_result.aggregate_score(),
    )
    return _result


def run_evaluators_parallel(
    candidates: list[Candidate],
    config: HelixConfig,
    split: str = "val",
) -> list[EvalResult]:
    """Run evaluators for multiple candidates in parallel.

    Args:
        candidates: List of candidates to evaluate.
        config: HelixConfig with evaluator settings.
        split: Dataset split to use (default "val").

    Returns:
        List of EvalResult in the same order as input candidates.
    """
    if not candidates:
        return []

    max_workers = min(len(candidates), os.cpu_count() or 1)
    results: dict[int, EvalResult] = {}

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(run_evaluator, candidate, config, split): i
            for i, candidate in enumerate(candidates)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            candidate = candidates[idx]
            try:
                results[idx] = future.result()
            except Exception as exc:
                error_ctx = format_error_context(
                    operation=f"parallel evaluate {candidate.id} (split={split})",
                    phase="ProcessPoolExecutor future",
                    cwd=str(candidate.worktree_path),
                    suggestion="Check that the evaluator command is valid and the worktree exists.",
                )
                logger.exception(
                    "Evaluator failed for candidate %s:\n%s\nFull exception chain:",
                    candidate.id, error_ctx,
                )
                results[idx] = EvalResult(
                    candidate_id=candidate.id,
                    scores={"success": 0.0},
                    asi={"error": f"evaluator_exception: {exc}"},
                    instance_scores={},
                )

    return [results[i] for i in range(len(candidates))]
