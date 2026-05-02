"""HELIX merger: combines two evolutionary candidates via Claude Code."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Callable, Mapping

from helix.population import Candidate, EvalResult
from helix.config import HelixConfig
from helix.worktree import clone_candidate, snapshot_candidate, remove_worktree, get_diff  # noqa: F401
from helix.exceptions import MutationError, RateLimitError, print_helix_error
from helix.mutator import invoke_claude_code, AUTONOMOUS_SYSTEM_PROMPT, _turn_budget_section

# ---------------------------------------------------------------------------
# Merge-acceptance subsample selection (GEPA parity)
# ---------------------------------------------------------------------------


def select_eval_subsample_for_merged_program(
    scores1: Mapping[str, float],
    scores2: Mapping[str, float],
    rng: random.Random,
    num_subsample_ids: int = 5,
) -> list[str]:
    """Port of GEPA's MergeProposer.select_eval_subsample_for_merged_program
    (gepa/src/gepa/proposer/merge.py:258-288). Stratifies the subsample
    across ids where one parent wins, the other parent wins, or they tie,
    then tops up from unused common ids (with-replacement fallback when
    common ids are exhausted).
    """
    common_ids = list(set(scores1.keys()) & set(scores2.keys()))

    p1: list[str] = [idx for idx in common_ids if scores1[idx] > scores2[idx]]
    p2: list[str] = [idx for idx in common_ids if scores2[idx] > scores1[idx]]
    p3: list[str] = [idx for idx in common_ids if idx not in p1 and idx not in p2]

    n_each = max(1, math.ceil(num_subsample_ids / 3))
    selected: list[str] = []
    for bucket in (p1, p2, p3):
        if len(selected) >= num_subsample_ids:
            break
        available = [idx for idx in bucket if idx not in selected]
        take = min(len(available), n_each, num_subsample_ids - len(selected))
        if take > 0:
            selected += rng.sample(available, k=take)

    remaining = num_subsample_ids - len(selected)
    if remaining > 0:
        unused = [idx for idx in common_ids if idx not in selected]
        if len(unused) >= remaining:
            selected += rng.sample(unused, k=remaining)
        elif common_ids:
            selected += rng.choices(common_ids, k=remaining)

    return selected[:num_subsample_ids]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

MERGE_PROMPT_TEMPLATE = """\
{system_prompt}

## Objective
{objective}

## Candidate A Strengths
{strengths_a}

## Candidate B Strengths
{strengths_b}

## Diff (B relative to A)
```diff
{diff}
```

## Background / Context
{background}

## Your Task
You are merging the best aspects of Candidate A and Candidate B to create a superior
combined solution that better achieves the objective.

Candidate A is already checked out in your working directory.  Apply the changes from
Candidate B that are beneficial, and discard or adapt those that conflict or regress.
You may read, edit, create, or delete files as needed.

When you have finished making all your changes, output the exact text:
[MERGE COMPLETE]
{turn_budget}"""

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_eval_strengths(eval_result: EvalResult | None, label: str) -> str:
    """Return a human-readable summary of a candidate's eval result."""
    if eval_result is None:
        return f"  {label}: (no evaluation data)"
    lines = [f"  Aggregate score: {eval_result.aggregate_score():.4f}"]
    for k, v in sorted(eval_result.scores.items()):
        lines.append(f"  {k}: {v}")
    if eval_result.instance_scores:
        top = sorted(
            eval_result.instance_scores.items(), key=lambda x: x[1], reverse=True
        )[:5]
        lines.append("  Top instances:")
        for inst, score in top:
            lines.append(f"    {inst}: {score:.4f}")
    return "\n".join(lines)


def build_merge_prompt(
    objective: str,
    eval_result_a: EvalResult | None,
    eval_result_b: EvalResult | None,
    diff: str,
    background: str | None = None,
    max_turns: int | None = None,
) -> str:
    """Construct the merge prompt for Claude Code."""
    strengths_a = _format_eval_strengths(eval_result_a, "Candidate A")
    strengths_b = _format_eval_strengths(eval_result_b, "Candidate B")
    bg = background or "(no additional background provided)"
    diff_text = diff.strip() if diff.strip() else "(no diff — candidates are identical)"

    return MERGE_PROMPT_TEMPLATE.format(
        system_prompt=AUTONOMOUS_SYSTEM_PROMPT,
        objective=objective,
        strengths_a=strengths_a,
        strengths_b=strengths_b,
        diff=diff_text,
        background=bg,
        turn_budget=_turn_budget_section(max_turns),
    )


# ---------------------------------------------------------------------------
# High-level merge entry point
# ---------------------------------------------------------------------------


def merge(
    candidate_a: Candidate,
    candidate_b: Candidate,
    new_id: str,
    config: HelixConfig,
    base_dir: Path,
    background: str | None = None,
    eval_result_a: EvalResult | None = None,
    eval_result_b: EvalResult | None = None,
    prepare_worktree: Callable[[Candidate], None] | None = None,
) -> Candidate | None:
    """Merge *candidate_a* and *candidate_b* using Claude Code.

    Clones *candidate_a*, computes the diff to *candidate_b*, builds a merge
    prompt, and invokes Claude Code.  Snapshots on success; removes the
    worktree and returns ``None`` on failure.

    Parameters
    ----------
    candidate_a:
        Base candidate (checked out in the new worktree).
    candidate_b:
        The candidate whose changes will be considered for merging in.
    new_id:
        Identifier for the merged candidate.
    config:
        Full HELIX config.
    base_dir:
        Base directory for worktrees.
    background:
        Optional background/context text.
    eval_result_a:
        Evaluation result for candidate A (optional, for richer prompt).
    eval_result_b:
        Evaluation result for candidate B (optional, for richer prompt).

    Returns
    -------
    Candidate | None
        The merged candidate on success, or ``None`` on failure.
    """
    child = clone_candidate(candidate_a, new_id, base_dir)
    child.operation = "merge"
    child.parent_ids = [candidate_a.id, candidate_b.id]
    if prepare_worktree is not None:
        prepare_worktree(child)

    diff = get_diff(candidate_a, candidate_b)

    prompt = build_merge_prompt(
        config.objective,
        eval_result_a,
        eval_result_b,
        diff,
        background,
        config.agent.max_turns,
    )

    try:
        _, usage = invoke_claude_code(
            child.worktree_path,
            prompt,
            config.agent,
            passthrough_env=config.passthrough_env,
            fixed_env=config.env,
            sandbox=config.sandbox,
        )
        child.usage = usage
    except MutationError as exc:
        exc.operation = f"merge {new_id} ({candidate_a.id} + {candidate_b.id})"
        print_helix_error(exc)
        try:
            remove_worktree(child)
        except Exception:
            pass
        return None
    except RateLimitError:
        # Rate limit — clean up orphaned worktree, then re-raise.
        try:
            remove_worktree(child)
        except Exception:
            pass
        raise

    # NOTE: snapshot_candidate() is intentionally NOT called here.
    # The caller (evolution.py) is responsible for calling save_state()
    # BEFORE snapshot_candidate() so that state is persisted even if
    # the commit step crashes (e.g. empty-commit error).
    return child
