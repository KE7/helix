"""HELIX mutator: applies code mutations via agentic coding backends (claude, codex, cursor, gemini, opencode)."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

from helix.backends import BACKEND_AUTH_ENV, backend_display_name
from helix.display import UsageStats
from helix.population import Candidate, EvalResult
from helix.config import AgentConfig, HelixConfig, SandboxConfig
from helix.exceptions import (
    MutationError,
    PromptArtifactCollisionError,
    RateLimitError,
    print_helix_error,
)
from helix.executor import _scrub_environment
from helix.sandbox import resolve_sandbox_image, run_sandboxed_command
from helix.worktree import clone_candidate, snapshot_candidate, remove_worktree  # noqa: F401

logger = logging.getLogger(__name__)

# Differential-testing hook: when set to a callable, ``invoke_claude_code``
# bypasses the subprocess invocation and delegates to the override with
# ``(worktree_path, prompt, config) -> tuple[dict[str, Any], dict[str, Any]]``.
# None (default) = unchanged production behavior.
_MUTATOR_OVERRIDE: (
    Callable[[str, str, AgentConfig], tuple[dict[str, Any], UsageStats]] | None
) = None

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

AUTONOMOUS_SYSTEM_PROMPT = """\
Task instructions:
- Work directly on the requested code changes using the workspace files.
- Do not request confirmation or clarification; choose a reasonable approach and continue.
- If one approach fails, try an alternative and keep progressing.
- Use available tools to inspect, edit, and validate changes.
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
{turn_budget}"""

MUTATION_TASK_INSTRUCTIONS = """\
## Your Task
Analyse the evaluation results above and improve the code to better achieve the objective.
Make targeted, meaningful changes. You may read, edit, create, or delete files as needed."""

# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _indefinite_article(n: int) -> str:
    """Pick ``"a"`` or ``"an"`` to match the spoken pronunciation of ``n``.

    Within HELIX's realistic max-turns range (1 ≤ n ≤ ~1000), the
    vowel-leading numbers are 8, 11, 18, and the 80s / 800s.  Everything
    else takes ``"a"``.  Handles the visible ``"You have a 8-turn limit"``
    article-agreement glitch without imposing a full English-number
    pronunciation rule on the codebase.
    """
    s = str(abs(n))
    if s in {"11", "18"}:
        return "an"
    if s.startswith("8"):  # 8, 80-89, 800-899, ...
        return "an"
    return "a"


def _turn_budget_section(max_turns: int | None) -> str:
    """Return the turn budget prompt section, or empty string if unbounded.

    Enforcement semantics differ by backend:

    * ``claude`` — hard cap.  HELIX passes ``--max-turns N`` to the Claude
      Code CLI (``_build_cli_args``), the runtime kills the session at the
      limit, and the resulting ``subtype="error_max_turns"`` response is
      detected at :func:`invoke_claude_code` and treated as partial
      success.
    * ``codex`` / ``cursor`` / ``gemini`` / ``opencode`` — soft hint only.
      None of these CLIs expose an equivalent flag (verified against
      ``--help`` for the installed binaries), so the in-prompt request is
      the only signal the agent receives.  Whether the agent self-honors
      the limit depends entirely on its own behaviour.

    The section is therefore emitted for every backend regardless — it
    still has some value as a soft hint — but callers depending on hard
    enforcement should set the budget low enough to also be enforced via
    subprocess-level mechanisms (wall-clock timeout, sandbox limits) or
    use the Claude backend.
    """
    if max_turns is None:
        return ""
    article = _indefinite_article(max_turns)
    return (
        f"\n## Turn Budget\n"
        f"You have {article} {max_turns}-turn limit for this task, where turns refer to "
        f"how many tool calls or interactions you can make. Plan your work "
        f"accordingly — prioritize the highest-impact changes first and be "
        f"efficient with your tool usage.\n"
    )


def _strip_machine_protocol_from_evaluator_stream(text: str) -> str:
    """Remove evaluator machine-contract lines before showing stdout/stderr.

    The raw evaluator stdout/stderr remains stored in ASI artifacts for
    debugging and parsing.  This helper only affects mutation prompts.
    ``HELIX_RESULT=`` is a machine protocol consumed by HELIX, never human
    reflection context for the mutator.
    """
    if not text:
        return ""

    kept: list[str] = []
    for line in text.splitlines():
        if line.strip().startswith("HELIX_RESULT="):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _has_structured_diagnostics(eval_result: EvalResult) -> bool:
    return (
        eval_result.per_example_side_info is not None
        or eval_result.side_info is not None
    )


def _evaluator_failed(eval_result: EvalResult) -> bool:
    """Return True iff the evaluator subprocess exited non-zero.

    ``_returncode`` rides inside ``asi`` (rather than as a typed field
    on :class:`EvalResult`) to preserve the GEPA O.A. EvaluationBatch
    interface — every HELIX-specific signal that downstream tooling
    might want to inspect lives in the same string-keyed bag.
    ``executor._collect_asi`` writes ``str(returncode)``; absent or
    unparseable values are treated as success so we never spuriously
    promote stdout/stderr into the prompt for archived records.
    """
    return eval_result.asi.get("_returncode") not in (None, "0", 0)


# All ``_render_*`` helpers below return either the empty string (the
# section is suppressed) or a fully-formed Markdown section.  Empty
# returns let ``build_mutation_prompt`` skip the section entirely —
# mirrors GEPA's ``_build_reflection_prompt_template`` accumulator
# pattern (``gepa/optimize_anything.py:501-596``).  Non-empty returns
# may carry trailing whitespace; ``build_mutation_prompt`` rstrips
# before joining sections with a uniform blank-line separator.


def _render_evaluator_notes(eval_result: EvalResult) -> str:
    notes = eval_result.asi.get("log", "").strip()
    if not notes:
        return ""
    return f"## Evaluator Notes\n{notes}\n\n"


def _render_evaluator_output_fallback(eval_result: EvalResult) -> str:
    """Render ``## Evaluator Output`` from stdout/stderr, or ``""``.

    Two distinct cases:

    * **Evaluator subprocess failed** (non-zero exit) — always emit the
      section.  Empty streams render with ``(no stdout)`` / ``(no stderr)``
      placeholders here intentionally: the agent needs to know the
      evaluator failed but produced no output to inspect (a meaningful
      diagnostic on its own).

    * **Evaluator succeeded** — only emit the section when at least one
      stream has content *and* no richer diagnostic surface
      (``log`` notes or structured side_info) exists.  Empty streams are
      omitted entirely; partial coverage (only ``stdout`` non-empty, or
      only ``stderr``) renders just the present sub-section instead of
      padding the other with a ``(no X)`` placeholder.
    """
    stdout = _strip_machine_protocol_from_evaluator_stream(
        eval_result.asi.get("stdout", "")
    )
    stderr = _strip_machine_protocol_from_evaluator_stream(
        eval_result.asi.get("stderr", "")
    )

    if _evaluator_failed(eval_result):
        stdout_text = stdout or "(no stdout)"
        stderr_text = stderr or "(no stderr)"
        return (
            f"## Evaluator Output\n\n"
            f"### stdout\n{stdout_text}\n\n"
            f"### stderr\n{stderr_text}"
        )

    # Evaluator succeeded — defer to richer surfaces when they exist,
    # otherwise surface only the streams that actually have content.
    has_notes = bool(eval_result.asi.get("log", "").strip())
    if _has_structured_diagnostics(eval_result) or has_notes:
        return ""

    parts: list[str] = []
    if stdout:
        parts.append(f"### stdout\n{stdout}")
    if stderr:
        parts.append(f"### stderr\n{stderr}")
    if not parts:
        return ""
    return "## Evaluator Output\n\n" + "\n\n".join(parts)


def build_seed_generation_prompt(
    objective: str,
    background: str | None = None,
    evaluator_cmd: str | None = None,
    dataset_examples: list[str] | None = None,
) -> str:
    """Construct the seed generation prompt for the configured backend.

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
    background_section = "\n".join(background_lines) + (
        "\n" if background_lines else ""
    )

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
) -> UsageStats:
    """Generate an initial seed candidate by invoking Claude Code once.

    Matches GEPA's ``_generate_seed_candidate`` pattern exactly:
    - Single LLM attempt, no retry loop.
    - If ``invoke_claude_code`` raises, the error propagates immediately
      (fail-fast).

    Parameters
    ----------
    worktree_path:
        Path to the (empty) seed worktree where the backend will write files.
    prompt:
        The seed-generation prompt built by :func:`build_seed_generation_prompt`.
    config:
        Full HELIX config (``config.agent`` is used for the backend invocation).

    Returns
    -------
    dict[str, Any]
        Normalized usage stats from the backend invocation.

    Raises
    ------
    MutationError
        Propagated directly from :func:`invoke_claude_code` on failure.
    """
    _, usage = invoke_claude_code(
        worktree_path,
        prompt,
        config.agent,
        passthrough_env=config.passthrough_env,
        fixed_env=config.env,
        sandbox=config.sandbox,
    )
    return usage


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
                    v,
                    min(level + 1, _MAX_MARKDOWN_HEADER_LEVEL),
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
                    item,
                    min(level + 1, _MAX_MARKDOWN_HEADER_LEVEL),
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


def _render_scores_section(eval_result: EvalResult) -> str:
    """Render ``## Current Evaluation Scores`` or ``""`` when no scores exist.

    Mirrors GEPA O.A.'s "only emit a section when there is content for it"
    pattern (``gepa/optimize_anything.py:501-596``).  Previously HELIX
    emitted the section with a ``"(no scores recorded)"`` placeholder; now
    the section header is omitted entirely so the agent never sees a stub.
    """
    lines = [f"  {k}: {v}" for k, v in sorted(eval_result.scores.items())]
    if not lines:
        return ""
    return "## Current Evaluation Scores\n" + "\n".join(lines)


def _render_extra_asi(eval_result: EvalResult) -> str:
    """Render any free-form ``extra_*`` ASI keys, or ``""`` when none exist.

    Reserved keys (``stdout``, ``stderr``, ``error``, ``log``, ``_returncode``)
    are filtered out — they're surfaced through dedicated sections
    (``## Evaluator Notes``, ``## Evaluator Output``) or the
    ``_returncode`` legacy sentinel — and must never leak into this
    catch-all rendering.
    """
    entries = {
        k: v
        for k, v in sorted(eval_result.asi.items())
        if k not in ("stdout", "stderr", "error", "log", "_returncode")
    }
    if not entries:
        return ""
    body = "\n".join(f"### {k}\n{v}" for k, v in entries.items())
    return f"### Extra Evaluator Info\n{body}"


def _render_diagnostics(eval_result: EvalResult) -> str:
    """Render the ``## Diagnostics`` section, or ``""`` when no side_info.

    Precedence:
      1. ``eval_result.per_example_side_info`` (per-example GEPA O.A.
         contract — list of dicts positional to instance_scores ids) when
         populated; mirrors GEPA's
         ``OptimizeAnythingAdapter.make_reflective_dataset`` combined
         with ``format_samples`` at
         ``gepa/strategies/instruction_proposal.py:54-95``.
      2. ``eval_result.side_info`` (legacy batch-level dict) when
         per-example data is absent.
      3. Empty string when neither is present.
    """
    if eval_result.per_example_side_info is not None:
        # Monotonic markdown hierarchy under the surrounding
        # ``## Diagnostics`` (h2): each example is ``### Example <id>``
        # (h3), each side_info key is ``#### {key}`` (h4), nested
        # values bump further.
        return _render_per_example_diagnostics(
            example_ids=list(eval_result.instance_scores.keys()),
            per_example_side_info=eval_result.per_example_side_info,
            example_header_level=3,
            key_header_level=4,
        )
    if eval_result.side_info is not None:
        diag_lines = "\n".join(
            f"  {k}: {v}" for k, v in sorted(eval_result.side_info.items())
        )
        return f"## Diagnostics\n{diag_lines}"
    return ""


def build_mutation_prompt(
    objective: str,
    eval_result: EvalResult,
    background: str | None = None,
    max_turns: int | None = None,
) -> str:
    """Construct the mutation prompt for the configured agent backend.

    Sections are emitted only when they have content, mirroring GEPA O.A.'s
    ``_build_reflection_prompt_template`` accumulator pattern
    (``gepa/optimize_anything.py:501-596``).  Empty ``objective``, empty
    ``eval_result.scores``, absent diagnostics, absent evaluator notes,
    absent stdout/stderr fallback, absent extra ASI, and absent
    ``background`` all skip their respective sections entirely instead of
    rendering placeholder strings like ``"(no additional background
    provided)"`` or ``"(no scores recorded)"`` that taught nothing.

    ``## Your Task`` and the system prompt are always emitted; they are
    the only sections that don't depend on caller-provided content.
    """
    sections: list[str] = [AUTONOMOUS_SYSTEM_PROMPT.rstrip()]

    if objective:
        sections.append(f"## Objective\n{objective}")

    scores = _render_scores_section(eval_result)
    if scores:
        sections.append(scores)

    diagnostics = _render_diagnostics(eval_result)
    if diagnostics:
        sections.append(diagnostics.rstrip())

    notes = _render_evaluator_notes(eval_result)
    if notes:
        sections.append(notes.rstrip())

    output_fallback = _render_evaluator_output_fallback(eval_result)
    if output_fallback:
        sections.append(output_fallback.rstrip())

    extra_asi = _render_extra_asi(eval_result)
    if extra_asi:
        sections.append(extra_asi)

    if background:
        sections.append(f"## Background / Context\n{background}")

    sections.append(MUTATION_TASK_INSTRUCTIONS)

    turn_budget = _turn_budget_section(max_turns)
    if turn_budget:
        sections.append(turn_budget.strip())

    return "\n\n".join(sections) + "\n"


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
MUTATION_PROMPT_ARTIFACT_NAME = ".agent_task_prompt.md"
MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME = ".agent_internal/task_prompt.md"
BACKEND_RESULT_ARTIFACT_NAME = ".helix_backend_result.json"
BACKEND_STDOUT_ARTIFACT_NAME = ".helix_backend_stdout.txt"
BACKEND_STDERR_ARTIFACT_NAME = ".helix_backend_stderr.txt"
BACKEND_TRANSCRIPT_ARTIFACT_DIR = ".helix_artifacts/backend_transcripts"


def _prompt_file_instruction(prompt_artifact_name: str) -> str:
    return (
        f"Read {prompt_artifact_name} in the current workspace and follow "
        "those instructions exactly."
    )


def _ignore_helix_artifacts(worktree_path: Path) -> None:
    """Append HELIX artifact names to ``<worktree>/.gitignore``.

    The candidate worktree is a real git tree; anything committed there
    bakes into the candidate's evolutionary history.  HELIX writes
    a handful of per-invocation metadata files (``helix_batch.json``,
    ``.agent_task_prompt.md``) that must NOT flow into those
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
        ".agent_internal/",
        BACKEND_RESULT_ARTIFACT_NAME,
        BACKEND_STDOUT_ARTIFACT_NAME,
        BACKEND_STDERR_ARTIFACT_NAME,
        ".helix_artifacts/",
        "helix_batch.json",
        # Per-candidate OpenCode SQLite state (XDG_DATA_HOME isolation).
        # Each parallel opencode worker gets a fresh database here; keeps
        # the candidate git tree free of opencode's session/session files.
        ".helix_opencode_state/",
    ]
    existing = gitignore.read_text() if gitignore.exists() else ""
    to_append = [p for p in patterns if p not in existing]
    if not to_append:
        return
    sep = "" if existing.endswith("\n") or not existing else "\n"
    gitignore.write_text(existing + sep + "\n".join(to_append) + "\n")


def _write_mutation_prompt_artifact(worktree_path: str, prompt: str) -> str:
    """Persist the rendered mutation prompt to the worktree for post-hoc inspection.

    Writes to a reserved stable path (``<worktree>/.agent_task_prompt.md``)
    with deterministic fallback (``<worktree>/.agent_internal/task_prompt.md``)
    and ensures the per-worktree ``.gitignore`` excludes the file.  Existing
    files are treated as user-owned collisions and are never overwritten.
    Returns the artifact filename chosen for this invocation.
    """

    def _try_write(path: Path) -> bool:
        """Write *prompt* to *path* without clobbering a user-owned file.

        Returns True on success. If the file already exists with identical
        content (e.g. a resumed run rewriting the same artifact), succeeds
        idempotently. Returns False if the file exists with different content
        so the caller can try the fallback path.
        """
        try:
            with path.open("x") as f:
                f.write(prompt)
            return True
        except FileExistsError:
            try:
                if path.read_text() == prompt:
                    # Same artifact already written by a prior attempt; fine
                    # to reuse without modifying the file.
                    return True
            except OSError:
                pass
            return False

    try:
        wt = Path(worktree_path)
        _ignore_helix_artifacts(wt)
        primary_path = wt / MUTATION_PROMPT_ARTIFACT_NAME
        fallback_path = wt / MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME
        if _try_write(primary_path):
            return MUTATION_PROMPT_ARTIFACT_NAME
        fallback_path.parent.mkdir(parents=True, exist_ok=True)
        if _try_write(fallback_path):
            return MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME
        # Both paths exist with content that differs from the new prompt;
        # treat as a user collision so we never silently overwrite.
        raise FileExistsError(
            f"both {primary_path.name} and {fallback_path} already exist with "
            "different content"
        )
    except OSError as e:
        raise PromptArtifactCollisionError(
            "Failed to create prompt artifact without overwriting an existing file",
            operation="write mutation prompt artifact",
            cwd=worktree_path,
            suggestion=(
                f"Remove or rename {MUTATION_PROMPT_ARTIFACT_NAME} and "
                f"{MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME}, or choose a different "
                "reserved prompt artifact path."
            ),
        ) from e


# ---------------------------------------------------------------------------
# Backend invocation
# ---------------------------------------------------------------------------


def _add_backend_auth_env(env: dict[str, str], backend: str) -> None:
    """Pass official headless auth env vars without requiring TOML config."""
    for key in BACKEND_AUTH_ENV.get(backend, ()):
        if key in os.environ and key not in env:
            env[key] = os.environ[key]


def _build_backend_args(
    worktree_path: str,
    config: AgentConfig,
    prompt_artifact_name: str,
) -> list[str]:
    backend = config.backend
    if backend == "claude":
        args = [
            "claude",
            "--dangerously-skip-permissions",
            "--print",
            "--output-format",
            "json",
        ]
        if config.model:
            args.extend(["--model", config.model])
        # ``effort`` is the user-facing knob for reasoning-level / thinking
        # budget; the registry of backends that honor it lives in
        # ``helix.backends.EFFORT_AWARE_BACKENDS`` and must stay in sync
        # with this branch (and the ``codex`` / ``opencode`` branches below).
        if config.effort:
            args.extend(["--effort", config.effort])
        if config.max_turns is not None:
            args.extend(["--max-turns", str(config.max_turns)])
        args.append(_prompt_file_instruction(prompt_artifact_name))
        return args

    if backend == "codex":
        args = [
            "codex",
            "exec",
            "--json",
            "--dangerously-bypass-approvals-and-sandbox",
        ]
        if config.model:
            args.extend(["--model", config.model])
        if config.effort:
            # Codex CLI accepts runtime config overrides via ``-c key=value``
            # where the value must be a valid TOML literal.  ``json.dumps``
            # produces a JSON string (double-quoted, with ``\"`` escapes) which
            # is valid TOML basic-string syntax for all printable ASCII — safe
            # for any realistic effort value.  See ``codex exec --help`` for
            # the full ``-c`` interface.
            args.extend(
                ["-c", f"model_reasoning_effort={json.dumps(config.effort)}"]
            )
        args.append(_prompt_file_instruction(prompt_artifact_name))
        return args

    if backend == "cursor":
        args = [
            "cursor",
            "agent",
            "--print",
            "--output-format",
            "stream-json",
            "--yolo",
            "--approve-mcps",
            "--trust",
            "--workspace",
            worktree_path,
        ]
        if config.model:
            args.extend(["--model", config.model])
        args.append(_prompt_file_instruction(prompt_artifact_name))
        return args

    if backend == "gemini":
        args = [
            "gemini",
            "--yolo",
            "--output-format",
            "stream-json",
        ]
        if config.model:
            args.extend(["--model", config.model])
        args.append(_prompt_file_instruction(prompt_artifact_name))
        return args

    if backend == "opencode":
        args = [
            "opencode",
            "run",
            "--format",
            "json",
            "--dangerously-skip-permissions",
        ]
        if config.model:
            args.extend(["--model", config.model])
        # opencode reuses ``agent.effort`` as a model-variant selector; see
        # ``helix.backends.EFFORT_AWARE_BACKENDS`` / ``EFFORT_VALID_VALUES``
        # for the source of truth on which backends propagate the field.
        if config.effort:
            args.extend(["--variant", config.effort])
        args.append(_prompt_file_instruction(prompt_artifact_name))
        return args

    raise ValueError(f"Unsupported backend: {backend}")


def _parse_json_object_output(
    stdout: str,
    *,
    backend: str,
    cmd_str: str,
    worktree_path: str,
    stderr: str,
    exit_code: int,
) -> dict[str, Any]:
    if not stdout.strip():
        return {}
    try:
        parsed = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise MutationError(
            f"Failed to parse {backend_display_name(backend)} JSON output: {exc}",
            operation=f"{backend_display_name(backend)} invocation",
            phase="JSON parsing",
            command=cmd_str,
            cwd=str(worktree_path),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            suggestion=(
                f"{backend_display_name(backend)} returned non-JSON output. "
                "Check stdout above for details."
            ),
        ) from exc
    if not isinstance(parsed, dict):
        raise MutationError(
            f"{backend_display_name(backend)} returned non-object JSON "
            f"(got {type(parsed).__name__})",
            operation=f"{backend_display_name(backend)} invocation",
            phase="JSON parsing",
            command=cmd_str,
            cwd=str(worktree_path),
            stdout=stdout,
            stderr=stderr,
            exit_code=exit_code,
            suggestion=(
                f"{backend_display_name(backend)} returned a non-object JSON "
                "value. Expected a JSON object."
            ),
        )
    return parsed


def _parse_jsonl_output(
    stdout: str,
    *,
    backend: str,
    cmd_str: str,
    worktree_path: str,
    stderr: str,
    exit_code: int,
    strict: bool,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    unparsable: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            if strict:
                # Gemini CLI may prepend advisory text such as MCP-health
                # warnings before the JSON stream even when
                # `--output-format stream-json` is requested.
                if backend == "gemini":
                    unparsable.append(line)
                    continue
                raise MutationError(
                    f"Failed to parse {backend_display_name(backend)} JSONL output line",
                    operation=f"{backend_display_name(backend)} invocation",
                    phase="JSON parsing",
                    command=cmd_str,
                    cwd=str(worktree_path),
                    stdout=stdout,
                    stderr=stderr,
                    exit_code=exit_code,
                    suggestion=(
                        f"{backend_display_name(backend)} emitted a non-JSON line "
                        "in structured output mode. Check stdout above for details."
                    ),
                )
            unparsable.append(line)
            continue
        if isinstance(parsed, dict):
            events.append(parsed)
    return {
        "events": events,
        "unparsable_lines": unparsable,
    }


def _parse_backend_output(
    backend: str,
    result: subprocess.CompletedProcess[str],
    *,
    cmd_str: str,
    worktree_path: str,
) -> dict[str, Any]:
    if backend == "claude":
        return _parse_json_object_output(
            result.stdout,
            backend=backend,
            cmd_str=cmd_str,
            worktree_path=worktree_path,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
    if backend in {"codex", "cursor", "gemini", "opencode"}:
        return _parse_jsonl_output(
            result.stdout,
            backend=backend,
            cmd_str=cmd_str,
            worktree_path=worktree_path,
            stderr=result.stderr,
            exit_code=result.returncode,
            strict=result.returncode == 0,
        )
    raise ValueError(f"Unsupported backend: {backend}")


def _walk_json(obj: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        found.append(obj)
        for value in obj.values():
            found.extend(_walk_json(value))
    elif isinstance(obj, list):
        for item in obj:
            found.extend(_walk_json(item))
    return found


def _coerce_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _normalise_usage_stats(parsed: dict[str, Any]) -> UsageStats:
    # Collect raw values into a plain dict first; construct UsageStats at the end.
    _d: dict[str, Any] = {}
    tool_event_count = 0
    tool_names: list[str] = []
    num_turns = 0

    # Claude Code returns num_turns and tool_use_count in the top-level object.
    if "num_turns" in parsed:
        _d["num_turns"] = _coerce_number(parsed["num_turns"])
    if "tool_use_count" in parsed:
        _d["tool_event_count"] = _coerce_number(parsed["tool_use_count"])

    # Recognise the exact node types that backends emit for tool/function
    # invocations. Substring matching on "call" is too loose -- it also fires
    # for unrelated event types like "recall", "callback", "rollback", etc.
    _TOOL_NODE_TYPES = {
        "tool_use",
        "tool_call",
        "tool.call",
        "tool-call",
        "tool_result",
        "function_call",
        "function-call",
    }
    for node in _walk_json(parsed):
        node_type = str(node.get("type", "")).lower()

        # Tool usage detection
        if node_type in _TOOL_NODE_TYPES:
            if "name" in node:
                tool_names.append(str(node["name"]))
            # ``tool_result`` is the backend's reply event; only count the
            # invocation events to avoid double-counting.
            if node_type != "tool_result":
                tool_event_count += 1

        # Turn detection (backend-specific heuristics)
        # Avoid double-counting against top-level num_turns if already present
        if "num_turns" not in parsed:
            if node_type in ("turn", "exchange") or node_type == "step_start":
                num_turns += 1

        for key, aliases in (
            (
                "input_tokens",
                (
                    "input_tokens",
                    "inputTokens",
                    "prompt_tokens",
                    "promptTokens",
                    "input",
                ),
            ),
            (
                "output_tokens",
                (
                    "output_tokens",
                    "outputTokens",
                    "completion_tokens",
                    "completionTokens",
                    "output",
                ),
            ),
            (
                "cached_input_tokens",
                (
                    "cached_input_tokens",
                    "cachedTokens",
                    "cacheReadInputTokens",
                    "cacheRead",
                    "cached",
                ),
            ),
            (
                "cache_creation_input_tokens",
                (
                    "cache_creation_input_tokens",
                    "cacheCreationInputTokens",
                    "cacheCreation",
                    "cache_write_input_tokens",
                    "cacheWriteTokens",
                ),
            ),
            (
                "cache_read_input_tokens",
                (
                    "cache_read_input_tokens",
                    "cacheReadInputTokens",
                    "cacheReadTokens",
                    "cacheRead",
                ),
            ),
            (
                "reasoning_tokens",
                (
                    "reasoning_tokens",
                    "reasoningTokens",
                    "reasoning_output_tokens",
                    "thoughts",
                    "reasoning",
                ),
            ),
            (
                "cost_usd",
                ("cost_usd", "costUsd", "total_cost_usd", "totalCostUsd", "total"),
            ),
        ):
            if key in _d:
                continue
            for alias in aliases:
                if alias in node:
                    value = _coerce_number(node[alias])
                    if value is not None:
                        _d[key] = value
                        break
        if "session_id" not in _d:
            for alias in (
                "session_id",
                "sessionId",
                "chat_id",
                "chatId",
                "thread_id",
                "threadId",
            ):
                value = node.get(alias)
                if isinstance(value, str) and value:
                    _d["session_id"] = value
                    break
            if "session_id" not in _d:
                value = node.get("sessionID")
                if isinstance(value, str) and value:
                    _d["session_id"] = value
        if "cost_usd" not in _d:
            value = node.get("cost")
            coerced = _coerce_number(value)
            if coerced is not None:
                _d["cost_usd"] = coerced

        # OpenCode emits cache accounting as ``tokens.cache.{read,write}``.
        # Keep this contextual rather than treating generic ``read``/``write``
        # fields elsewhere in a transcript as token counts.
        cache = node.get("cache")
        if isinstance(cache, dict):
            for key, field in (
                ("cache_creation_input_tokens", "write"),
                ("cache_read_input_tokens", "read"),
            ):
                if key in _d:
                    continue
                value = _coerce_number(cache.get(field))
                if value is not None:
                    _d[key] = value

    if tool_event_count and "tool_event_count" not in _d:
        _d["tool_event_count"] = tool_event_count
    if tool_names:
        _d["tool_names"] = tool_names
    if num_turns and "num_turns" not in _d:
        _d["num_turns"] = num_turns

    return UsageStats(
        input_tokens=int(_d.get("input_tokens", 0)),
        output_tokens=int(_d.get("output_tokens", 0)),
        cached_input_tokens=int(_d.get("cached_input_tokens", 0)),
        cache_creation_input_tokens=int(_d.get("cache_creation_input_tokens", 0)),
        cache_read_input_tokens=int(_d.get("cache_read_input_tokens", 0)),
        reasoning_tokens=int(_d.get("reasoning_tokens", 0)),
        num_turns=int(_d.get("num_turns", 0)),
        tool_event_count=int(_d.get("tool_event_count", 0)),
        tool_names=_d.get("tool_names", []),
        cost_usd=float(_d.get("cost_usd", 0.0)),
        session_id=_d.get("session_id"),
    )


# ---------------------------------------------------------------------------
# Per-backend transcript tool-event counters
# ---------------------------------------------------------------------------
# Claude's ``--output-format json`` summary omits per-turn tool invocations;
# the other backends either emit tool events in their JSONL stream with
# types/fields that ``_normalise_usage_stats`` doesn't fully parse (wrong
# key names, double-counted started+completed pairs, etc.).  Each function
# below parses the backend's native format and returns an accurate
# ``(count, names)`` pair.  ``_count_transcript_tool_events`` dispatches to
# the right function and is called from ``_write_backend_artifacts`` /
# ``_collect_backend_transcript_artifacts`` to patch ``usage`` after the fact.


def _count_claude_transcript_tool_events(path: Path) -> tuple[int, list[str]]:
    """Count tool invocations from a Claude JSONL transcript file.

    Claude stores tool calls as ``type='assistant'`` events whose
    ``message.content`` list contains items with ``type='tool_use'``.
    The top-level ``type`` is never ``'tool_use'`` — it only appears
    inside the nested ``message.content`` array.

    Returns ``(0, [])`` when the file is missing or a line is malformed.
    """
    count = 0
    names: list[str] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue  # skip malformed lines
            if not isinstance(event, dict) or event.get("type") != "assistant":
                continue
            message = event.get("message")
            if not isinstance(message, dict):
                continue
            for item in message.get("content") or []:
                if isinstance(item, dict) and item.get("type") == "tool_use":
                    count += 1
                    name = item.get("name")
                    if isinstance(name, str) and name:
                        names.append(name)
    except OSError:
        return 0, []
    return count, names


def _count_codex_stdout_tool_events(path: Path) -> tuple[int, list[str]]:
    """Count tool invocations from a Codex ``exec --json`` stdout artifact.

    Codex emits ``item.completed`` events whose ``item.type`` is one of
    ``command_execution`` (shell commands) or ``file_change`` (edits).
    These types are not in ``_normalise_usage_stats``'s ``_TOOL_NODE_TYPES``
    set, so the initial count is 0; this function provides the correct value.
    """
    count = 0
    names: list[str] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(event, dict) or event.get("type") != "item.completed":
                continue
            item = event.get("item", {})
            if not isinstance(item, dict):
                continue
            itype = item.get("type", "")
            if itype == "command_execution":
                count += 1
                names.append("exec_command")
            elif itype == "file_change":
                count += 1
                names.append("apply_patch")
    except OSError:
        return 0, []
    return count, names


def _count_cursor_stdout_tool_events(path: Path) -> tuple[int, list[str]]:
    """Count tool invocations from a Cursor stream-json stdout artifact.

    Cursor emits ``tool_call`` events with ``subtype`` values ``started``
    and ``completed``.  Counting both would double the true count, so only
    ``started`` events are counted.  The tool name is the first key of the
    nested ``tool_call`` object (e.g. ``readToolCall``, ``editToolCall``).
    """
    count = 0
    names: list[str] = []
    _NAME_MAP: dict[str, str] = {
        "readToolCall": "read",
        "editToolCall": "edit",
        "shellToolCall": "shell",
        "writeToolCall": "write",
        "searchToolCall": "search",
        "deleteToolCall": "delete",
        "listToolCall": "list",
        "grepToolCall": "grep",
    }
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(event, dict):
                continue
            if event.get("type") != "tool_call" or event.get("subtype") != "started":
                continue
            count += 1
            tool_call = event.get("tool_call", {})
            if isinstance(tool_call, dict):
                for key in tool_call:
                    key_str = str(key)
                    names.append(_NAME_MAP.get(key_str, key_str))
                    break
            else:
                names.append("unknown")
    except OSError:
        return 0, []
    return count, names


def _count_gemini_stdout_tool_events(path: Path) -> tuple[int, list[str]]:
    """Count tool invocations from a Gemini stream-json stdout artifact.

    Gemini emits ``tool_use`` events (correctly counted by
    ``_normalise_usage_stats``) but stores the tool name in ``tool_name``
    rather than ``name``, so ``tool_names`` remains empty after the initial
    parse.  This function provides both the count and the names.
    """
    count = 0
    names: list[str] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(event, dict) or event.get("type") != "tool_use":
                continue
            count += 1
            name = event.get("tool_name")
            if isinstance(name, str) and name:
                names.append(name)
    except OSError:
        return 0, []
    return count, names


def _count_opencode_stdout_tool_events(path: Path) -> tuple[int, list[str]]:
    """Count tool invocations from an OpenCode ``--format json`` stdout artifact.

    OpenCode emits ``tool_use`` events with the tool name in ``part.tool``
    rather than a top-level ``name`` field.  This function extracts both
    the accurate count and the tool names.
    """
    count = 0
    names: list[str] = []
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            raw = raw.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(event, dict) or event.get("type") != "tool_use":
                continue
            count += 1
            part = event.get("part", {})
            if isinstance(part, dict):
                name = part.get("tool")
                if isinstance(name, str) and name:
                    names.append(name)
    except OSError:
        return 0, []
    return count, names


# Dispatcher: maps backend name → per-backend counter function.
_TRANSCRIPT_TOOL_COUNTERS: dict[str, Callable[[Path], tuple[int, list[str]]]] = {
    "claude": _count_claude_transcript_tool_events,
    "codex": _count_codex_stdout_tool_events,
    "cursor": _count_cursor_stdout_tool_events,
    "gemini": _count_gemini_stdout_tool_events,
    "opencode": _count_opencode_stdout_tool_events,
}


def _count_transcript_tool_events(path: Path, backend: str) -> tuple[int, list[str]]:
    """Dispatch to the per-backend transcript tool-event counter.

    Returns ``(0, [])`` for unknown backends or when the file is missing /
    corrupt.  Never raises.
    """
    counter = _TRANSCRIPT_TOOL_COUNTERS.get(backend)
    if counter is None or not path.is_file():
        return 0, []
    try:
        return counter(path)
    except Exception:  # noqa: BLE001
        return 0, []


def _copy_local_claude_transcript(
    worktree_path: str,
    *,
    session_id: str | None,
    artifact_dir: str = BACKEND_TRANSCRIPT_ARTIFACT_DIR,
    transcript_root: str | None = None,
) -> dict[str, Any] | None:
    if not session_id:
        return None
    wt = Path(worktree_path)
    rel_path = Path(artifact_dir) / "claude" / f"{session_id}.jsonl"
    dst = wt / rel_path
    if dst.exists():
        return {
            "backend": "claude",
            "session_id": session_id,
            "path": str(rel_path),
            "source": "sandbox_auth_volume",
            "available": True,
        }
    if transcript_root == "sandbox_auth_volume":
        return {
            "backend": "claude",
            "session_id": session_id,
            "path": str(rel_path),
            "source": "sandbox_auth_volume",
            "available": False,
            "reason": "transcript_not_found",
        }
    root = (
        Path(transcript_root)
        if transcript_root is not None
        else Path(
            os.environ.get(
                "HELIX_CLAUDE_TRANSCRIPT_ROOT",
                Path.home() / ".claude/projects/-workspace",
            )
        )
    )
    src = root / f"{session_id}.jsonl"
    if not src.is_file():
        return {
            "backend": "claude",
            "session_id": session_id,
            "path": str(rel_path),
            "source": str(src),
            "available": False,
            "reason": "transcript_not_found",
        }
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        import shutil

        shutil.copy2(src, dst)
    except OSError as exc:
        return {
            "backend": "claude",
            "session_id": session_id,
            "path": str(rel_path),
            "source": str(src),
            "available": False,
            "reason": f"copy_failed: {exc}",
        }
    return {
        "backend": "claude",
        "session_id": session_id,
        "path": str(rel_path),
        "source": str(src),
        "available": True,
    }


def _collect_backend_transcript_artifacts(
    worktree_path: str,
    *,
    backend: str,
    usage: UsageStats,
    sandbox: SandboxConfig | None,
) -> list[dict[str, Any]]:
    if backend != "claude":
        return []
    if sandbox is not None and not sandbox.preserve_backend_transcripts:
        return []
    session_id = usage.session_id
    if not isinstance(session_id, str) or not session_id:
        return []
    artifact_dir = (
        sandbox.transcript_artifact_dir
        if sandbox is not None
        else BACKEND_TRANSCRIPT_ARTIFACT_DIR
    )
    transcript_root = (
        "sandbox_auth_volume"
        if sandbox is not None and sandbox.enabled
        else sandbox.claude_transcript_root
        if sandbox is not None
        else None
    )
    artifact = _copy_local_claude_transcript(
        worktree_path,
        session_id=session_id,
        artifact_dir=artifact_dir,
        transcript_root=transcript_root,
    )
    if artifact is not None and artifact.get("available"):
        # Claude's ``--output-format json`` summary omits per-turn tool
        # invocations.  Now that the transcript is on disk, read it and
        # patch the ``usage`` object with the accurate counts.  Only
        # applied when the summary gave 0 tool events to avoid overriding
        # a non-zero value that a future Claude version might expose.
        dst = Path(worktree_path) / Path(artifact["path"])
        tc, tn = _count_transcript_tool_events(dst, "claude")
        if tc > 0 and usage.tool_event_count == 0:
            usage.tool_event_count = tc
            usage.tool_names = list(tn)
    return [artifact] if artifact is not None else []


def _write_backend_artifacts(
    worktree_path: str,
    *,
    backend: str,
    command: str,
    result: subprocess.CompletedProcess[str],
    parsed: dict[str, Any] | None,
    sandbox: SandboxConfig | None = None,
) -> None:
    try:
        wt = Path(worktree_path)
        _ignore_helix_artifacts(wt)
        (wt / BACKEND_STDOUT_ARTIFACT_NAME).write_text(result.stdout or "")
        (wt / BACKEND_STDERR_ARTIFACT_NAME).write_text(result.stderr or "")
        usage = _normalise_usage_stats(parsed or {})
        # For non-Claude backends the stdout JSONL IS the transcript; patch
        # ``usage`` with backend-specific tool-event counts now that the
        # stdout artifact is on disk.  Claude is handled separately inside
        # ``_collect_backend_transcript_artifacts`` where the external
        # transcript file is copied first.
        if backend != "claude":
            stdout_path = wt / BACKEND_STDOUT_ARTIFACT_NAME
            tc, tn = _count_transcript_tool_events(stdout_path, backend)
            if tc > 0:
                usage.tool_event_count = tc
                usage.tool_names = list(tn)
        transcript_artifacts = _collect_backend_transcript_artifacts(
            worktree_path,
            backend=backend,
            usage=usage,
            sandbox=sandbox,
        )
        payload = {
            "backend": backend,
            "backend_display_name": backend_display_name(backend),
            "command": command,
            "cwd": worktree_path,
            "returncode": result.returncode,
            "stdout_artifact": BACKEND_STDOUT_ARTIFACT_NAME,
            "stderr_artifact": BACKEND_STDERR_ARTIFACT_NAME,
            "usage": usage.to_dict(),
            "transcript_artifacts": transcript_artifacts,
            "parsed": parsed,
        }
        (wt / BACKEND_RESULT_ARTIFACT_NAME).write_text(json.dumps(payload, indent=2))
    except OSError as e:
        logger.debug(
            "failed to write backend artifacts to %s: %s",
            worktree_path,
            e,
        )


def invoke_claude_code(
    worktree_path: str,
    prompt: str,
    config: AgentConfig,
    passthrough_env: list[str] | None = None,
    fixed_env: dict[str, str] | None = None,
    sandbox: SandboxConfig | None = None,
    prompt_artifact_name: str = MUTATION_PROMPT_ARTIFACT_NAME,
) -> tuple[dict[str, Any], UsageStats]:
    """Invoke the configured backend CLI in *worktree_path*.

    Parameters
    ----------
    worktree_path:
        Working directory for the backend subprocess.
    prompt:
        Prompt / task instructions for the backend.
    config:
        Backend configuration (backend selector, model, effort, tool policy).
    passthrough_env:
        Optional list of extra env var names to preserve from the parent
        process through the env scrub (e.g. CUDA_VISIBLE_DEVICES).
    fixed_env:
        Optional mapping of explicit env var values to inject after
        passthrough values.

    Returns
    -------
    dict
        Parsed structured output from the backend.

    Raises
    ------
    MutationError
        On non-zero return code or JSON decode failure.
        All errors include the full command, full stdout, full stderr
        (never truncated), exit code, and working directory.
    """
    if _MUTATOR_OVERRIDE is not None:
        return _MUTATOR_OVERRIDE(worktree_path, prompt, config)
    backend = config.backend
    backend_name = backend_display_name(backend)
    backend_worktree_path = (
        "/workspace" if sandbox is not None and sandbox.enabled else worktree_path
    )
    args = _build_backend_args(
        backend_worktree_path,
        config,
        prompt_artifact_name,
    )
    cmd_str = shlex.join(args)
    backend_env = _scrub_environment(
        passthrough_env=passthrough_env, fixed_env=fixed_env
    )
    _add_backend_auth_env(backend_env, backend)
    if backend == "gemini":
        backend_env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
    if backend == "opencode" and (sandbox is None or not sandbox.enabled):
        # Per-candidate SQLite isolation for concurrent opencode subprocesses.
        #
        # OpenCode stores its session database at:
        #   macOS: ~/Library/Application Support/opencode/opencode.db
        #   Linux: $XDG_DATA_HOME/opencode/opencode.db  (default ~/.local/share/opencode/)
        #
        # When multiple proposals run in parallel (num_parallel_proposals > 1),
        # every worker spawns a fresh `opencode run` subprocess that issues
        # `PRAGMA journal_mode = WAL` against this shared database at startup.
        # Concurrent WAL-mode requests on the same file produce:
        #   "Failed to run the query 'PRAGMA journal_mode = WAL'"
        # (observed in PR #34 E2E re-verify: g1-s1 lost to this error while g1-s2 succeeded).
        #
        # Fix: set XDG_DATA_HOME to a per-candidate directory.  OpenCode respects
        # XDG_DATA_HOME and will create an isolated database at:
        #   <worktree>/.helix_opencode_state/opencode/opencode.db
        # Each parallel worker gets its own fresh database; no contention.
        #
        # The sandbox branch is excluded: container isolation already provides
        # per-candidate filesystem separation, so XDG_DATA_HOME would be redundant.
        opencode_state_dir = Path(worktree_path) / ".helix_opencode_state"
        opencode_state_dir.mkdir(parents=True, exist_ok=True)
        backend_env["XDG_DATA_HOME"] = str(opencode_state_dir)
    if sandbox is not None and sandbox.enabled:
        sandbox_image = resolve_sandbox_image(sandbox, backend)
        result = run_sandboxed_command(
            args,
            cwd=worktree_path,
            env=backend_env,
            sandbox=sandbox,
            scope="agent",
            sync_back=True,
            image=sandbox_image,
            agent_backend=backend,
        )
    else:
        result = subprocess.run(
            args,
            cwd=worktree_path,
            capture_output=True,
            text=True,
            env=backend_env,
        )

    parsed: dict[str, Any] | None = None
    try:
        if result.returncode == 0:
            parsed = _parse_backend_output(
                backend,
                result,
                cmd_str=cmd_str,
                worktree_path=worktree_path,
            )
            usage = _normalise_usage_stats(parsed)
            if backend == "claude":
                error_text = str(parsed.get("error", ""))
                if _looks_like_rate_limit(error_text):
                    logger.error(
                        "Rate limit detected in JSON response: %s", error_text[:200]
                    )
                    raise RateLimitError(
                        f"{backend_name} returned a rate/usage limit error in JSON response",
                        operation=f"{backend_name} invocation",
                        phase="JSON parsing",
                        command=cmd_str,
                        cwd=str(worktree_path),
                        stdout=result.stdout,
                        stderr=result.stderr,
                        exit_code=result.returncode,
                        suggestion=(
                            f"{backend_name} reported a rate limit. "
                            "Retry after backoff or check your API quota."
                        ),
                    )
            return parsed, usage

        rate_limit_source = result.stderr or result.stdout
        if _looks_like_rate_limit(rate_limit_source):
            logger.error(
                "Rate limit detected in subprocess exit for %s (code %d): %s",
                backend_name,
                result.returncode,
                rate_limit_source[:200],
            )
            raise RateLimitError(
                f"{backend_name} hit a rate/usage limit (exit code {result.returncode})",
                operation=f"{backend_name} invocation",
                phase="subprocess exit",
                command=cmd_str,
                cwd=str(worktree_path),
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
                suggestion=(
                    f"{backend_name} reported a rate limit. "
                    "Retry after backoff or check your quota."
                ),
            )

        # Claude's max-turns exhaustion is intentionally treated as partial
        # success because the subprocess may have already produced useful edits.
        if backend == "claude":
            try:
                parsed = _parse_backend_output(
                    backend,
                    result,
                    cmd_str=cmd_str,
                    worktree_path=worktree_path,
                )
                usage = _normalise_usage_stats(parsed)
                if parsed.get("subtype") == "error_max_turns":
                    logger.warning(
                        "Claude Code reached max_turns limit (%s turns) — treating as partial success.",
                        parsed.get("num_turns", "?"),
                    )
                    return parsed, usage
            except MutationError:
                parsed = None

        parsed = _parse_backend_output(
            backend,
            result,
            cmd_str=cmd_str,
            worktree_path=worktree_path,
        )
        usage = _normalise_usage_stats(parsed)

        raise MutationError(
            f"{backend_name} exited with code {result.returncode}",
            operation=f"{backend_name} invocation",
            phase="subprocess exit",
            command=cmd_str,
            cwd=str(worktree_path),
            stdout=result.stdout,
            stderr=result.stderr,
            exit_code=result.returncode,
            suggestion="Check stderr for rate limits, permission errors, or model availability.",
        )
    finally:
        _write_backend_artifacts(
            worktree_path,
            backend=backend,
            command=cmd_str,
            result=result,
            parsed=parsed,
            sandbox=sandbox,
        )


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
    prepare_worktree: Callable[[Candidate], None] | None = None,
) -> Candidate | None:
    """Mutate *parent* using the configured backend and return the new candidate.

    Clones the parent worktree, builds a mutation prompt, invokes the backend,
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
        Full HELIX config (``config.agent`` and ``config.objective`` are used).
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
    if prepare_worktree is not None:
        prepare_worktree(child)

    prompt = build_mutation_prompt(
        config.objective,
        eval_result,
        background,
        config.agent.max_turns,
    )

    # Persist the rendered prompt to the worktree for post-hoc inspection:
    # what did the mutator actually see on this generation?  Sits next to
    # ``helix_batch.json`` in the worktree root.  The leading dot and the
    # per-worktree ``.gitignore`` entry (see ``_ignore_helix_artifacts``)
    # keep it out of the candidate git tree — otherwise it'd leak into
    # every subsequent mutation's diff and the mutator would see its own
    # prior prompt file as part of the codebase.
    prompt_artifact_name = _write_mutation_prompt_artifact(child.worktree_path, prompt)

    try:
        _, usage = invoke_claude_code(
            child.worktree_path,
            prompt,
            config.agent,
            passthrough_env=config.passthrough_env,
            fixed_env=config.env,
            sandbox=config.sandbox,
            prompt_artifact_name=prompt_artifact_name,
        )
        child.usage = usage
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
