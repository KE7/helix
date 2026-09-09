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
    (gepa/src/gepa/proposer/merge.py::MergeProposer.select_eval_subsample_for_merged_program).
    Stratifies the subsample
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

MERGE_TASK_INSTRUCTIONS_SINGLE_DIFF = """\
## Your Task
You are merging the best aspects of Candidate A and Candidate B to create a superior
combined solution that better achieves the objective.

Candidate A is already checked out in your working directory.  Apply the changes from
Candidate B that are beneficial, and discard or adapt those that conflict or regress.
You may read, edit, create, or delete files as needed."""

MERGE_TASK_INSTRUCTIONS_TWO_DIFF = """\
## Your Task
You are merging the best aspects of Candidate A and Candidate B to create a superior
combined solution that better achieves the objective.

Candidate A's worktree is already checked out — A's contribution (the hunks shown in
"Diff: Candidate A relative to common ancestor") is already in place, so you do not
need to re-apply it.  Your job is to bring in Candidate B's contribution (the hunks
shown in "Diff: Candidate B relative to common ancestor") wherever it is beneficial.

For each hunk in B's diff:
- If the file is untouched by A's diff, the change is independent — apply it.
- If the file is also touched by A's diff (overlapping region), the two parents
  diverged from the ancestor on the same region: reconcile the changes, picking
  whichever side better serves the objective, or synthesize a combined version.

You may read, edit, create, or delete files as needed."""

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_eval_strengths(eval_result: EvalResult) -> str:
    """Return a human-readable summary of a candidate's eval result.

    Returns the section body only (no header); the caller is responsible
    for the ``## Candidate {A,B} Strengths`` heading.  Empty input is
    handled by skipping the section entirely in :func:`build_merge_prompt`
    rather than emitting a ``"(no evaluation data)"`` placeholder.
    """
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
    *,
    ancestor_id: str | None = None,
    diff_a_from_ancestor: str | None = None,
    diff_b_from_ancestor: str | None = None,
) -> str:
    """Construct the merge prompt for the configured agent backend.

    Sections are emitted only when they have content, mirroring GEPA O.A.'s
    ``_build_reflection_prompt_template`` accumulator pattern
    in ``gepa_launcher.py``.

    **Diff section format** — two-diff (ancestor-relative) vs single (A↔B):

    GEPA's merge operator
    (``gepa/proposer/merge.py::sample_and_attempt_merge_programs_by_common_predictors``)
    reasons over
    *three* program states — common ancestor, candidate A, candidate B —
    to attribute each component change to whichever parent diverged from
    the ancestor.  When ``ancestor_id`` + both ``diff_a_from_ancestor``
    and ``diff_b_from_ancestor`` are supplied, this prompt mirrors that
    structure at the file-hunk level: each parent's diff against the
    common ancestor is rendered as its own labelled section, so the
    agent can read off "A's contribution" and "B's contribution"
    directly instead of inferring three-way info from a single A↔B diff.

    When the ancestor-relative pair is not supplied (e.g. legacy callers,
    tests that don't have an ancestor handy), the prompt falls back to
    the single ``## Diff (B relative to A)`` section driven by ``diff``.

    Absent eval results, absent diff(s), and absent ``background`` all
    skip their respective sections entirely instead of emitting
    placeholder strings.
    """
    sections: list[str] = [AUTONOMOUS_SYSTEM_PROMPT.rstrip()]

    if objective:
        sections.append(f"## Objective\n{objective}")

    if eval_result_a is not None:
        sections.append(
            "## Candidate A Strengths\n" + _format_eval_strengths(eval_result_a)
        )

    if eval_result_b is not None:
        sections.append(
            "## Candidate B Strengths\n" + _format_eval_strengths(eval_result_b)
        )

    # Diff section — prefer the two-diff (ancestor-relative) form when
    # both diffs are available, fall back to single A↔B otherwise.  Each
    # branch independently honors the "omit when empty" invariant.
    use_two_diff = (
        ancestor_id is not None
        and diff_a_from_ancestor is not None
        and diff_b_from_ancestor is not None
    )
    if use_two_diff:
        diff_a_stripped = (diff_a_from_ancestor or "").strip()
        diff_b_stripped = (diff_b_from_ancestor or "").strip()
        if diff_a_stripped:
            sections.append(
                f"## Diff: Candidate A relative to common ancestor {ancestor_id}\n"
                f"```diff\n{diff_a_stripped}\n```"
            )
        if diff_b_stripped:
            sections.append(
                f"## Diff: Candidate B relative to common ancestor {ancestor_id}\n"
                f"```diff\n{diff_b_stripped}\n```"
            )
    else:
        diff_stripped = diff.strip()
        if diff_stripped:
            sections.append(
                f"## Diff (B relative to A)\n```diff\n{diff_stripped}\n```"
            )

    if background:
        sections.append(f"## Background / Context\n{background}")

    # Task instructions vary by diff form.  Two-diff form gets explicit
    # guidance on what A's contribution vs B's contribution means and how
    # to reason about overlapping vs disjoint hunks; single-diff form
    # keeps the legacy "apply B's changes" framing.
    sections.append(
        MERGE_TASK_INSTRUCTIONS_TWO_DIFF if use_two_diff
        else MERGE_TASK_INSTRUCTIONS_SINGLE_DIFF
    )

    turn_budget = _turn_budget_section(max_turns)
    if turn_budget:
        sections.append(turn_budget.strip())

    return "\n\n".join(sections) + "\n"


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
    ancestor: Candidate | None = None,
) -> Candidate | None:
    """Merge *candidate_a* and *candidate_b* using Claude Code.

    Clones *candidate_a*, computes the relevant diffs, builds a merge
    prompt, and invokes Claude Code.  Snapshots on success; removes the
    worktree and returns ``None`` on failure.

    Two diff-rendering modes, controlled by the optional ``ancestor``
    argument:

    * **Two-diff (ancestor-relative)** — when ``ancestor`` is provided,
      computes ``get_diff(ancestor, candidate_a)`` and
      ``get_diff(ancestor, candidate_b)`` and renders both as separately
      labelled sections in the prompt.  The agent can then attribute
      each hunk to whichever parent diverged from the common ancestor —
      file-hunk-level analogue of GEPA's component-wise attribution
      (``gepa/proposer/merge.py::sample_and_attempt_merge_programs_by_common_predictors``:
      ``if pred_anc == pred_id1 …``
      → take id2's version; ``elif pred_anc != pred_id1 and pred_anc !=
      pred_id2`` → tiebreak by score).
    * **Single (A↔B)** — fallback when no ancestor is provided.
      Computes ``get_diff(candidate_a, candidate_b)`` and renders a
      single ``## Diff (B relative to A)`` section.  The agent has to
      infer three-way info from a two-way comparison.

    GEPA-parity note: this is the correct domain adaptation of GEPA's
    text-component merge
    (``gepa/proposer/merge.py::sample_and_attempt_merge_programs_by_common_predictors``)
    for HELIX's
    full-codebase setting.  GEPA can splice ``dict[str, str]`` programs
    deterministically by swapping components from each parent; HELIX
    candidates are full git worktrees, where syntactic per-component swap
    is undefined, so an LLM-mediated edit is the only viable approach —
    but feeding the agent the three-way diff structure GEPA's algorithm
    uses (two ancestor-relative diffs instead of one A↔B diff) gives it
    the same shape of attribution information.

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
    prepare_worktree:
        Optional callback to refresh protected files in the new worktree
        before the agent runs.
    ancestor:
        Optional most-recent common ancestor of A and B (typically
        ``frontier.candidates[ancestor_id]`` where ``ancestor_id`` came
        from :func:`helix.lineage.find_merge_triplet`).  When supplied,
        the prompt uses the two-diff (ancestor-relative) form; when
        ``None``, falls back to the single A↔B diff.

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

    # Diff-rendering mode selection.  ``ancestor`` available → compute
    # the two ancestor-relative diffs that drive the GEPA-style
    # attribution prompt.  Otherwise compute the single A↔B fallback.
    if ancestor is not None:
        diff_a_from_ancestor: str | None = get_diff(ancestor, candidate_a)
        diff_b_from_ancestor: str | None = get_diff(ancestor, candidate_b)
        # ``diff`` (single A↔B) is computed lazily only for the fallback
        # path; with both ancestor-relative diffs in hand, the prompt
        # builder ignores the legacy parameter, so pass an empty string.
        legacy_diff = ""
    else:
        diff_a_from_ancestor = None
        diff_b_from_ancestor = None
        legacy_diff = get_diff(candidate_a, candidate_b)

    prompt = build_merge_prompt(
        config.objective,
        eval_result_a,
        eval_result_b,
        legacy_diff,
        background,
        config.agent.max_turns,
        ancestor_id=ancestor.id if ancestor is not None else None,
        diff_a_from_ancestor=diff_a_from_ancestor,
        diff_b_from_ancestor=diff_b_from_ancestor,
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
