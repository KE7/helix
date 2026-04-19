"""HELIX mutator: applies code mutations to evolutionary candidates via Claude Code."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helix.population import Candidate, EvalResult
from helix.config import ClaudeConfig, HelixConfig
from helix.exceptions import MutationError, RateLimitError, print_helix_error
from helix.executor import _scrub_environment
from helix.worktree import clone_candidate, snapshot_candidate, remove_worktree

logger = logging.getLogger(__name__)

# Differential-testing hook: when set to a callable, ``invoke_claude_code``
# bypasses the subprocess invocation and delegates to the override with
# ``(worktree_path, prompt, config) -> dict[str, Any]``.  None (default) =
# unchanged production behavior.
_MUTATOR_OVERRIDE = None

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

AUTONOMOUS_SYSTEM_PROMPT = """\
You are an autonomous code-improvement agent operating inside an automated evolutionary loop.

CRITICAL RULES — you MUST follow these at all times:
- NEVER ask for human input, clarification, or confirmation of any kind.
- NEVER use tools that request human interaction or pause for approval.
- NEVER pause, wait, or yield control back to a human.
- If you are uncertain about something, make your best judgment and proceed.
- If you are blocked by one approach, try an alternative — do not stop.
- You are running fully unattended; there is no human monitoring this session.
"""

SEEDLESS_INIT_PROMPT_TEMPLATE = """\
You are an expert assistant. Your task is to generate an initial candidate \
that will be iteratively refined by an optimization system.

## Goal

{objective}
{background_section}{dataset_section}## Output Format

Generate a strong initial candidate based on the goal above.
Create all necessary files directly in the current working directory.
Make your implementation complete and ready to be evaluated immediately.

When you have finished creating all files, output the exact text:
[SEED GENERATION COMPLETE]
{turn_budget}"""

MUTATION_PROMPT_TEMPLATE = """\
{system_prompt}

## Objective
{objective}

## Current Evaluation Scores
{scores}

## Evaluator Output

### stdout
{asi_stdout}

### stderr
{asi_stderr}

{extra_asi_section}{diagnostics_section}## Background / Context
{background}

## Your Task
Analyse the evaluation results above and improve the code to better achieve the objective.
Make targeted, meaningful changes. You may read, edit, create, or delete files as needed.

When you have finished making all your changes, output the exact text:
[MUTATION COMPLETE]
{turn_budget}"""

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _turn_budget_section(max_turns: int | None) -> str:
    """Return the turn budget prompt section, or empty string if unbounded."""
    if max_turns is None:
        return ""
    return (
        f"\n## Turn Budget\n"
        f"You have a {max_turns}-turn limit for this task, where turns refer to "
        f"how many tool calls or interactions you can make. Plan your work "
        f"accordingly — prioritize the highest-impact changes first and be "
        f"efficient with your tool usage.\n"
    )


def build_seed_generation_prompt(
    objective: str,
    background: str | None = None,
    evaluator_cmd: str | None = None,
    dataset_examples: list[str] | None = None,
) -> str:
    """Construct the seed generation prompt for Claude Code.

    Mirrors GEPA's ``_build_seed_generation_prompt`` — a single structured
    prompt that gives the LLM everything it needs to write a first candidate
    from scratch.

    Parameters
    ----------
    objective:
        Natural-language description of what to optimise.
    background:
        Optional domain context / constraints.
    evaluator_cmd:
        Optional evaluator command shown to the LLM for context.
    dataset_examples:
        Optional list of representative dataset inputs used to ground the LLM,
        matching GEPA's ``_build_seed_generation_prompt`` ``dataset`` argument.
        At most the first 3 items are included.

    Returns
    -------
    str
        The formatted seed-generation prompt.
    """
    background_lines: list[str] = []
    if background:
        background_lines.append(f"\n## Domain Context & Constraints\n\n{background}")
    if evaluator_cmd:
        background_lines.append(
            f"\n## Evaluator\n\nYour candidate will be evaluated by running:\n\n"
            f"    {evaluator_cmd}\n\n"
            f"Make sure your implementation is compatible with this command."
        )
    background_section = "\n".join(background_lines) + ("\n" if background_lines else "")

    # Mirror GEPA's dataset grounding: include up to 3 representative examples
    # so the LLM understands the input format before writing the first candidate.
    dataset_section = ""
    if dataset_examples:
        examples = dataset_examples[:3]
        example_lines = [f"- Example {i}: {ex}" for i, ex in enumerate(examples, 1)]
        dataset_section = (
            "\n## Sample Inputs\n\n"
            "The candidate will be evaluated on inputs like these:\n\n"
            + "\n".join(example_lines)
            + "\n\n"
        )

    return SEEDLESS_INIT_PROMPT_TEMPLATE.format(
        objective=objective,
        background_section=background_section,
        dataset_section=dataset_section,
        turn_budget=_turn_budget_section(None),
    )


def generate_seed(
    worktree_path: str,
    prompt: str,
    config: "HelixConfig",
) -> None:
    """Generate an initial seed candidate by invoking Claude Code once.

    Matches GEPA's ``_generate_seed_candidate`` pattern exactly:
    - Single LLM attempt, no retry loop.
    - If ``invoke_claude_code`` raises, the error propagates immediately
      (fail-fast).

    Parameters
    ----------
    worktree_path:
        Path to the (empty) seed worktree where Claude will write files.
    prompt:
        The seed-generation prompt built by :func:`build_seed_generation_prompt`.
    config:
        Full HELIX config (``config.claude`` is used for the LLM invocation).

    Raises
    ------
    MutationError
        Propagated directly from :func:`invoke_claude_code` on failure.
    """
    invoke_claude_code(worktree_path, prompt, config.claude, passthrough_env=config.passthrough_env)


_MAX_MARKDOWN_HEADER_LEVEL = 6


def _render_side_info_value(value: Any, level: int) -> str:
    """Render a single side_info value as markdown.

    Line-for-line port of GEPA's ``render_value`` closure inside
    ``format_samples`` at
    ``src/gepa/strategies/instruction_proposal.py:63-85``:

      * ``dict`` → ``{'#' * level} {key}`` for each item, recursing
        at ``level + 1`` (capped at ``#_MAX_MARKDOWN_HEADER_LEVEL``
        to stay inside valid markdown depth).
      * ``list`` / ``tuple`` → ``{'#' * level} Item N`` headers,
        recursing at ``level + 1``.
      * primitive → ``str(value).strip() + "\n\n"``.

    Empty containers still emit a trailing blank line so surrounding
    headers don't collapse against the next block.
    """
    if isinstance(value, dict):
        parts: list[str] = []
        for k, v in value.items():
            parts.append(f"{'#' * level} {k}")
            parts.append(
                _render_side_info_value(
                    v, min(level + 1, _MAX_MARKDOWN_HEADER_LEVEL),
                )
            )
        if not value:
            parts.append("")
        return "\n".join(parts)
    if isinstance(value, (list, tuple)):
        parts = []
        for i, item in enumerate(value):
            parts.append(f"{'#' * level} Item {i + 1}")
            parts.append(
                _render_side_info_value(
                    item, min(level + 1, _MAX_MARKDOWN_HEADER_LEVEL),
                )
            )
        if not value:
            parts.append("")
        return "\n".join(parts)
    # Primitive — GEPA renders with ``str(...).strip() + "\n\n"``.
    return str(value).strip() + "\n\n"


def _render_per_example_diagnostics(
    example_ids: list[str],
    per_example_side_info: list[dict[str, Any]],
    example_header_level: int = 1,
    key_header_level: int = 2,
) -> str:
    """Render per-example side_info as the mutation-prompt Diagnostics section.

    Mirrors GEPA's ``OptimizeAnythingAdapter.make_reflective_dataset`` +
    ``format_samples`` at
    ``src/gepa/adapters/optimize_anything_adapter/optimize_anything_adapter.py:524-553``
    and ``src/gepa/strategies/instruction_proposal.py:54-95``:

      * each example gets an ``{'#' * example_header_level} Example <id>``
        header (id recovered from ``helix_batch.json`` via
        ``eval_result.instance_scores.keys()``);
      * the reserved ``scores`` key renames to
        ``Scores (Higher is Better)`` at ``key_header_level``;
      * any other side_info key renders as a ``{'#' * key_header_level}``
        header with a recursive :func:`_render_side_info_value` body
        (nested dicts bump to ``key_header_level + 1``, lists become
        ``### Item N`` sub-headers, primitives render as plain text).

    ``example_header_level`` / ``key_header_level`` are parameterised
    so the surrounding Diagnostics section's own level (``## Diagnostics``)
    can drive a monotonic hierarchy from the outside.

    Length mismatch between ``example_ids`` and ``per_example_side_info``
    is tolerated by iterating over ``zip`` — the parser enforces
    equality on the helix_result path; other paths should never hit
    this function.

    Empty per-example side_info (every slot is ``{}``) still produces
    the section header + per-example headers, so the mutator can see
    that the evaluator had no reflection data rather than silently
    dropping the section.
    """
    if not per_example_side_info:
        return ""

    example_hashes = "#" * example_header_level
    key_hashes = "#" * key_header_level
    nested_level = min(key_header_level + 1, _MAX_MARKDOWN_HEADER_LEVEL)

    lines: list[str] = ["## Diagnostics"]
    for eid, side_info in zip(example_ids, per_example_side_info):
        lines.append("")
        lines.append(f"{example_hashes} Example {eid}")
        if not side_info:
            lines.append("(no per-example side_info)")
            continue
        for key, value in sorted(side_info.items()):
            if key == "scores":
                # GEPA parity: the reserved ``scores`` sub-dict renames
                # to "Scores (Higher is Better)" and still renders
                # recursively underneath.
                lines.append(f"{key_hashes} Scores (Higher is Better)")
            else:
                lines.append(f"{key_hashes} {key}")
            body = _render_side_info_value(value, nested_level).rstrip("\n")
            if body:
                lines.append(body)
    lines.append("")  # trailing blank line before the next section
    return "\n".join(lines) + "\n"


def build_mutation_prompt(
    objective: str,
    eval_result: EvalResult,
    background: str | None = None,
    max_turns: int | None = None,
) -> str:
    """Construct the mutation prompt for Claude Code."""
    scores_text = "\n".join(
        f"  {k}: {v}" for k, v in sorted(eval_result.scores.items())
    )
    if not scores_text:
        scores_text = "  (no scores recorded)"

    asi_stdout = eval_result.asi.get("stdout", "(no stdout)")
    asi_stderr = eval_result.asi.get("stderr", "(no stderr)")

    # Collect any extra_N entries from ASI
    extra_entries = {
        k: v for k, v in sorted(eval_result.asi.items())
        if k not in ("stdout", "stderr", "error")
    }
    if extra_entries:
        extra_lines = "\n".join(
            f"### {k}\n{v}" for k, v in extra_entries.items()
        )
        extra_asi_section = f"### Extra Evaluator Info\n{extra_lines}\n\n"
    else:
        extra_asi_section = ""

    # Render side_info diagnostics.  Precedence:
    #   1. ``eval_result.per_example_side_info`` (new per-example GEPA
    #      O.A. contract — list of dicts positional to instance_scores
    #      ids) when populated; mirrors GEPA's
    #      ``OptimizeAnythingAdapter.make_reflective_dataset`` at
    #      ``optimize_anything_adapter.py:524-553`` combined with
    #      ``format_samples`` at
    #      ``gepa/strategies/instruction_proposal.py:54-95``.
    #   2. ``eval_result.side_info`` (legacy batch-level dict) when
    #      ``per_example_side_info`` is absent — unchanged rendering
    #      for non-``helix_result`` paths that still populate the
    #      legacy field.
    #   3. No diagnostics section otherwise.
    diagnostics_section = ""
    if eval_result.per_example_side_info is not None:
        # Monotonic markdown hierarchy under the surrounding
        # ``## Diagnostics`` (h2): each example is ``### Example <id>``
        # (h3), each side_info key is ``#### {key}`` (h4), nested
        # values bump further.  Before this the Example header was
        # ``#`` (h1), which inverted the hierarchy and confused
        # markdown-aware tooling / LLM markdown parsers.
        diagnostics_section = _render_per_example_diagnostics(
            example_ids=list(eval_result.instance_scores.keys()),
            per_example_side_info=eval_result.per_example_side_info,
            example_header_level=3,
            key_header_level=4,
        )
    elif eval_result.side_info is not None:
        diag_lines = "\n".join(
            f"  {k}: {v}" for k, v in sorted(eval_result.side_info.items())
        )
        diagnostics_section = f"## Diagnostics\n{diag_lines}\n\n"

    bg = background or "(no additional background provided)"

    return MUTATION_PROMPT_TEMPLATE.format(
        system_prompt=AUTONOMOUS_SYSTEM_PROMPT,
        objective=objective,
        scores=scores_text,
        asi_stdout=asi_stdout,
        asi_stderr=asi_stderr,
        extra_asi_section=extra_asi_section,
        diagnostics_section=diagnostics_section,
        background=bg,
        turn_budget=_turn_budget_section(max_turns),
    )


# ---------------------------------------------------------------------------
# Mutation summary parsing
# ---------------------------------------------------------------------------


def parse_mutation_summary(output: str) -> dict[str, str]:
    """Parse a ``[SUMMARY]...[END SUMMARY]`` block from Claude Code output.

    Extracts structured key-value pairs written by the mutation/merge agent
    after ``[MUTATION COMPLETE]`` or ``[MERGE COMPLETE]``.  The block format
    is::

        [SUMMARY]
        files_changed: src/foo.py, src/bar.py
        root_cause: ...
        changes_made: ...
        [END SUMMARY]

    Returns
    -------
    dict[str, str]
        Parsed key-value pairs.  Returns an empty dict if no block is found
        or the block contains no valid ``key: value`` lines.  Never raises.
    """
    result: dict[str, str] = {}
    in_block = False
    for line in output.splitlines():
        if "[SUMMARY]" in line:
            in_block = True
            continue
        if "[END SUMMARY]" in line:
            break
        if in_block and ":" in line:
            key, _, value = line.partition(":")
            key = key.strip()
            value = value.strip()
            if key:
                result[key] = value
    return result


# ---------------------------------------------------------------------------
# Rate-limit detection
# ---------------------------------------------------------------------------

_RATE_LIMIT_KEYWORDS = [
    "rate limit",
    "overloaded",
    "529",
    "usage limit",
    "extra usage",
]


def _looks_like_rate_limit(text: str) -> bool:
    """Return True if *text* contains a rate-limit / overload keyword."""
    lower = text.lower()
    return any(kw in lower for kw in _RATE_LIMIT_KEYWORDS)


# ---------------------------------------------------------------------------
# Rendered-mutation-prompt artifact
# ---------------------------------------------------------------------------


#: Filename of the post-hoc mutation-prompt artifact persisted in each
#: candidate's worktree root alongside ``helix_batch.json``.  The leading
#: dot + per-worktree ``.gitignore`` entry keep it out of the candidate
#: git tree.
MUTATION_PROMPT_ARTIFACT_NAME = ".helix_mutation_prompt.md"


def _ignore_helix_artifacts(worktree_path: Path) -> None:
    """Append HELIX artifact names to ``<worktree>/.gitignore``.

    The candidate worktree is a real git tree; anything committed there
    bakes into the candidate's evolutionary history.  HELIX writes
    a handful of per-invocation metadata files (``helix_batch.json``,
    ``.helix_mutation_prompt.md``) that must NOT flow into those
    diffs — otherwise the next generation's mutator sees the prior
    artifact as part of the codebase and the lineage grows a
    meaningless file-rename trail.

    Idempotent: only appends patterns that aren't already present.
    Creates the ``.gitignore`` file if missing.
    """
    gitignore = worktree_path / ".gitignore"
    patterns = [
        "# HELIX per-invocation artifacts (never commit to candidate tree)",
        MUTATION_PROMPT_ARTIFACT_NAME,
        "helix_batch.json",
    ]
    existing = gitignore.read_text() if gitignore.exists() else ""
    to_append = [p for p in patterns if p not in existing]
    if not to_append:
        return
    sep = "" if existing.endswith("\n") or not existing else "\n"
    gitignore.write_text(existing + sep + "\n".join(to_append) + "\n")


def _write_mutation_prompt_artifact(worktree_path: str, prompt: str) -> None:
    """Persist the rendered mutation prompt to the worktree for post-hoc inspection.

    Writes to ``<worktree>/.helix_mutation_prompt.md`` and ensures the
    per-worktree ``.gitignore`` excludes the file.  Best-effort: any I/O
    exception is logged at DEBUG and swallowed — a missing artifact is
    not worth failing a mutation over.
    """
    try:
        wt = Path(worktree_path)
        _ignore_helix_artifacts(wt)
        (wt / MUTATION_PROMPT_ARTIFACT_NAME).write_text(prompt)
    except OSError as e:
        logger.debug(
            "failed to write mutation-prompt artifact to %s: %s",
            worktree_path, e,
        )


# ---------------------------------------------------------------------------
# Claude Code invocation
# ---------------------------------------------------------------------------


def invoke_claude_code(
    worktree_path: str,
    prompt: str,
    config: ClaudeConfig,
    passthrough_env: list[str] | None = None,
) -> dict[str, Any]:
    """Invoke Claude Code CLI in *worktree_path* and return the parsed JSON output.

    Parameters
    ----------
    worktree_path:
        Working directory for the Claude Code subprocess.
    prompt:
        The prompt to pass via ``-p``.
    config:
        Claude configuration (model, effort, allowed_tools).
    passthrough_env:
        Optional list of extra env var names to preserve from the parent
        process through the env scrub (e.g. CUDA_VISIBLE_DEVICES).

    Returns
    -------
    dict
        Parsed JSON from Claude Code's stdout.

    Raises
    ------
    MutationError
        On non-zero return code or JSON decode failure.
        All errors include the full command, full stdout, full stderr
        (never truncated), exit code, and working directory.
    """
    if _MUTATOR_OVERRIDE is not None:
        return _MUTATOR_OVERRIDE(worktree_path, prompt, config)

    tools_str = ",".join(config.allowed_tools)
    args = [
        "claude",
        "--dangerously-skip-permissions",
        "--print",
        "--output-format", "json",
        "--allowedTools", tools_str,
    ]
    if config.model:
        args.extend(["--model", config.model])
    if config.effort:
        args.extend(["--effort", config.effort])
    if config.max_turns is not None:
        args.extend(["--max-turns", str(config.max_turns)])
    args.extend(["-p", prompt])

    cmd_str = " ".join(args)

    # Reuse the shared scrub helper (no split/instance_ids for CC sessions).
    cc_env = _scrub_environment(passthrough_env=passthrough_env)

    result = subprocess.run(
        args,
        cwd=worktree_path,
        capture_output=True,
        text=True,
        env=cc_env,
    )

    if result.returncode != 0:
        # Check for rate-limit / overload before raising generic MutationError
        if _looks_like_rate_limit(result.stderr) or _looks_like_rate_limit(result.stdout):
            logger.error(
                "Rate limit detected in subprocess exit (code %d): %s",
                result.returncode,
                (result.stderr or result.stdout)[:200],
            )
            raise RateLimitError(
                f"Claude Code hit a rate/usage limit (exit code {result.returncode})",
                operation="Claude Code invocation",
                phase="subprocess exit",
                command=cmd_str,
                cwd=str(worktree_path),
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                suggestion=(
                    "Claude Code reported a rate limit. "
                    "Retry after backoff or check your API quota."
                ),
            )

        # error_max_turns: Claude hit the turn limit but may have made useful changes.
        # Parse stdout as JSON and return it as a partial success rather than discarding.
        try:
            parsed = json.loads(result.stdout)
            if isinstance(parsed, dict) and parsed.get("subtype") == "error_max_turns":
                logger.warning(
                    "Claude Code reached max_turns limit (%s turns) — treating as partial success.",
                    parsed.get("num_turns", "?"),
                )
                return parsed
        except json.JSONDecodeError:
            pass

        raise MutationError(
            f"Claude Code exited with code {result.returncode}",
            operation="Claude Code invocation",
            phase="subprocess exit",
            command=cmd_str,
            cwd=str(worktree_path),
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            suggestion="Check stderr for rate limits, permission errors, or model availability.",
        )

    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise MutationError(
            f"Failed to parse Claude Code JSON output: {exc}",
            operation="Claude Code invocation",
            phase="JSON parsing",
            command=cmd_str,
            cwd=str(worktree_path),
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            suggestion="Claude Code returned non-JSON output. Check stdout above for details.",
        ) from exc

    # Check for rate-limit indicators in successful JSON responses
    if not isinstance(parsed, dict):
        raise MutationError(
            f"Claude Code returned non-object JSON (got {type(parsed).__name__})",
            operation="Claude Code invocation",
            phase="JSON parsing",
            command=cmd_str,
            cwd=str(worktree_path),
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            suggestion="Claude Code returned a non-object JSON value. Expected a JSON object.",
        )
    error_text = str(parsed.get("error", ""))
    if _looks_like_rate_limit(error_text):
        logger.error(
            "Rate limit detected in JSON response: %s",
            error_text[:200],
        )
        raise RateLimitError(
            "Claude Code returned a rate/usage limit error in JSON response",
            operation="Claude Code invocation",
            phase="JSON parsing",
            command=cmd_str,
            cwd=str(worktree_path),
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            suggestion=(
                "Claude Code reported a rate limit. "
                "Retry after backoff or check your API quota."
            ),
        )

    return parsed


# ---------------------------------------------------------------------------
# High-level mutate entry point
# ---------------------------------------------------------------------------


def mutate(
    parent: Candidate,
    eval_result: EvalResult,
    new_id: str,
    config: HelixConfig,
    base_dir: Path,
    background: str | None = None,
) -> Candidate | None:
    """Mutate *parent* using Claude Code and return the new candidate.

    Clones the parent worktree, builds a mutation prompt, invokes Claude Code,
    then snapshots on success.  Returns ``None`` on any :class:`MutationError`.

    Parameters
    ----------
    parent:
        The candidate to mutate.
    eval_result:
        Most recent evaluation result for *parent*.
    new_id:
        Identifier for the mutated candidate.
    config:
        Full HELIX config (``config.claude`` and ``config.objective`` are used).
    base_dir:
        Base directory for worktrees.
    background:
        Optional background/context text injected into the prompt.

    Returns
    -------
    Candidate | None
        The new candidate on success, or ``None`` if mutation failed.
    """
    child = clone_candidate(parent, new_id, base_dir)
    child.operation = "mutate"

    prompt = build_mutation_prompt(
        config.objective, eval_result, background, config.claude.max_turns,
    )

    # Persist the rendered prompt to the worktree for post-hoc inspection:
    # what did the mutator actually see on this generation?  Sits next to
    # ``helix_batch.json`` in the worktree root.  The leading dot and the
    # per-worktree ``.gitignore`` entry (see ``_ignore_helix_artifacts``)
    # keep it out of the candidate git tree — otherwise it'd leak into
    # every subsequent mutation's diff and the mutator would see its own
    # prior prompt file as part of the codebase.
    _write_mutation_prompt_artifact(child.worktree_path, prompt)

    try:
        invoke_claude_code(child.worktree_path, prompt, config.claude, passthrough_env=config.passthrough_env)
    except MutationError as exc:
        exc.operation = f"mutate {new_id} (parent: {parent.id})"
        print_helix_error(exc)
        try:
            remove_worktree(child)
        except Exception:
            pass
        return None
    except RateLimitError:
        # Rate limit — clean up orphaned worktree, then re-raise so the parallel
        # futures handler in evolution.py can log it and continue with a smaller
        # proposal set.
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
