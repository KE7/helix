"""HELIX merger: combines two evolutionary candidates via Claude Code."""

from __future__ import annotations

from pathlib import Path

from helix.population import Candidate, EvalResult
from helix.config import HelixConfig
from helix.worktree import clone_candidate, snapshot_candidate, remove_worktree, get_diff
from helix.exceptions import MutationError, RateLimitError, print_helix_error
from helix.mutator import invoke_claude_code, AUTONOMOUS_SYSTEM_PROMPT, _turn_budget_section

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

    diff = get_diff(candidate_a, candidate_b)

    prompt = build_merge_prompt(
        config.objective,
        eval_result_a,
        eval_result_b,
        diff,
        background,
        config.claude.max_turns,
    )

    try:
        invoke_claude_code(child.worktree_path, prompt, config.claude, passthrough_env=config.passthrough_env)
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
