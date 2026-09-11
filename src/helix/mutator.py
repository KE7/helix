"""HELIX mutator: applies code mutations via agentic coding backends (agy, claude, codex, cursor, opencode)."""

from __future__ import annotations

import json
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable

from helix.backends import (
    BACKEND_AUTH_ENV,
    BACKEND_FRESH_SESSION_ENV,
    BACKEND_TRANSCRIPT_SOURCES,
    TRANSCRIPT_SESSION_ID_RE,
    backend_display_name,
)
from helix.display import UsageStats
from helix.population import Candidate, EvalResult
from helix.config import AgentConfig, HelixConfig, SandboxConfig
from helix.exceptions import (
    CredentialRefreshError,
    HelixError,
    MutationError,
    PromptArtifactCollisionError,
    RateLimitError,
    print_helix_error,
)
from helix.executor import _scrub_environment
from helix.lines import split_lf_lines
from helix.sandbox import (
    OPENCODE_STATE_DIR_NAME,
    resolve_sandbox_image,
    run_sandboxed_command,
)
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
    * ``agy`` / ``codex`` / ``cursor`` / ``opencode`` — soft hint only.
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
    for line in split_lf_lines(text):
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
# pattern in ``gepa_launcher.py``.  Non-empty returns
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
    ``src/gepa/strategies/instruction_proposal.py::format_samples.render_value``:

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

    Mirrors GEPA's ``OptimizeAnythingAdapter.make_reflective_dataset`` in
    ``adapters/optimize_anything_adapter/optimize_anything_adapter.py`` and
    the ``format_samples`` closure inside
    ``InstructionProposalSignature.prompt_renderer`` in
    ``src/gepa/strategies/instruction_proposal.py``:

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
    pattern in ``_build_reflection_prompt_template`` (``gepa_launcher.py``).
    Previously HELIX emitted the section with a ``"(no scores recorded)"``
    placeholder; now the section header is omitted entirely so the agent
    never sees a stub.
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
         ``gepa/strategies/instruction_proposal.py::format_samples``.
      2. ``eval_result.side_info`` (legacy batch-level dict) when
         per-example data is absent. The reserved ``scores`` key is
         relabelled ``Scores (Higher is Better)`` here too.
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
            f"  {'Scores (Higher is Better)' if k == 'scores' else k}: {v}"
            for k, v in sorted(eval_result.side_info.items())
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
    in ``gepa_launcher.py``.  Empty ``objective``, empty
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
# Credential / refresh-exhaustion detection
# ---------------------------------------------------------------------------
#
# Distinct from the rate-limit detection above.  A rate limit means "come back
# later on the same credential"; these markers mean "the credential itself can
# no longer be used", which is what a lost refresh race looks like from inside
# a candidate.  Telling them apart matters because only the second one is
# unrecoverable without operator action, and neither is a code failure.
#
# Every marker below is a phrase read out of the shipped CLI it belongs to, not
# a guess at the wording.  They are deliberately whole distinctive sentences or
# clauses: matching a bare number or a single common word ("401", "token",
# "auth") is the false-positive trap this repo has already paid for once --
# a candidate whose own diff or test output mentions tokens must never be
# reported to the operator as a broken login.
#
# Which backends can actually lose a refresh race.  Every candidate container
# mounts the same login volume read-write, so when a credential goes stale the
# candidates in flight may each decide a refresh is due at the same moment.
# Whether that is a hazard depends on how the CLI performs the refresh:
#
#   codex     YES.  Single-use refresh token, and no cross-process
#             serialisation.  Measured against codex-cli 0.130.0 with a
#             synthetic credential and a local single-use token endpoint: five
#             simultaneous candidates produced five token exchanges, one
#             granted and four rejected with "your refresh token was already
#             used" -- and all five processes exited 0 with empty stderr.
#   opencode  YES, and unmitigated.  It refreshes an ``oauth`` credential only
#             from inside the fetch wrapper that issues a model request
#             (opencode-ai 1.14.24), writing the new credential back unlocked.
#             No free command takes the refresh path, so there is nothing that
#             could serialise it short of a model call.  (``api``-type
#             credentials never refresh and are not at risk.)
#   claude    NO.  Claude Code takes a real cross-process lock file, retries
#             while another process holds it, and re-reads the credential
#             afterwards, so two candidates cannot spend the same grant.
#   cursor    NO.  It re-exchanges an API key rather than spending a stored
#             refresh token; there is no single-use grant to race for.
#   agy       UNMEASURED.  No agy credential was available to test against,
#             so this is unknown rather than safe.
#
# Nothing pre-empts the race: what recovers a lost one is the caller-level
# one-shot retry (``invoke_with_refresh_race_retry``), which re-runs the
# invocation from a fresh worktree against the credential the winner stored.
#
# Markers whose failure is *transient*: the credential is not broken, this
# invocation merely lost a refresh race.  Checked before the general markers so
# the more specific wording is what gets reported, and so the caller can retry
# once against the credential the winner has just stored.
_TRANSIENT_CREDENTIAL_FAILURE_MARKERS: tuple[str, ...] = (
    # Codex CLI (codex-cli 0.130.0): the suffix it appends when another
    # process spent the single-use refresh token first.  The full sentence is
    # "Your access token could not be refreshed because your refresh token
    # was already used. Please log out and sign in again." -- the "log out"
    # advice is the CLI's, and is wrong for this case: the shared auth.json
    # already holds the refreshed credential.
    "because your refresh token was already used",
)

# KNOWN RISK: every marker is pinned to the prose of a specific CLI release
# (versions noted per entry), while the sandbox images install the unpinned
# CLI package and are rebuilt on a schedule.  A wording change upstream turns
# a credential failure back into an unclassified one -- it never produces a
# false positive.  No test reads real CLI output; when a marker stops matching,
# the symptom is a generic MutationError whose stderr carries the new wording.
_CREDENTIAL_FAILURE_MARKERS: tuple[str, ...] = (
    # Codex CLI (codex-cli 0.130.0).  One prefix covers every suffix the CLI
    # appends: "... has expired." / "... was revoked." / "... because you have
    # since logged out or signed in to another account." / the bare
    # "Your access token could not be refreshed. Please log out and sign in
    # again."  (The already-used suffix is matched first, above, as transient.)
    "your access token could not be refreshed",
    "failed to refresh token while getting account",
    "chatgpt account id not available, please re-run `codex login`",
    # OpenCode (opencode-ai 1.14.24) -- thrown by the provider fetch wrapper
    # when the OAuth token exchange is rejected, e.g. "Token refresh failed:
    # 400".  The colon is kept so the phrase cannot match narrative prose.
    "token refresh failed:",
    # Claude Code (2.1.138).
    "user oauth refresh failed",
    "api error: 401 invalid api key",
)


def credential_failure_marker(text: str) -> str | None:
    """Return the credential-failure marker *text* contains, or ``None``.

    Returns the matched marker rather than a bool so callers can name the
    evidence in the operator-facing message instead of asserting a verdict
    with nothing behind it.
    """
    if not text:
        return None
    lower = text.lower()
    for marker in _TRANSIENT_CREDENTIAL_FAILURE_MARKERS + _CREDENTIAL_FAILURE_MARKERS:
        if marker in lower:
            return marker
    return None


def credential_failure_is_transient(marker: str) -> bool:
    """True when *marker* names a lost refresh race rather than a dead login."""
    return marker in _TRANSIENT_CREDENTIAL_FAILURE_MARKERS


#: Event types under which the JSONL backends report a failure of the
#: invocation itself, as opposed to a tool call the agent made.  Read from the
#: shipped CLIs' own event definitions:
#:
#:   codex     ``{"type": "error", "message": ...}`` and
#:             ``{"type": "turn.failed", "error": {"message": ...}}``
#:             (codex-rs ``exec/src/exec_events.rs``: ``ThreadErrorEvent``,
#:             ``TurnFailedEvent``).  ``codex exec --json`` never emits an
#:             ``is_error`` key.
#:   opencode  ``{"type": "error", ..., "error": {"name": ..., "data":
#:             {"message": ...}}}`` -- the ``session.error`` event that
#:             ``opencode run --format json`` re-emits (``cli/cmd/run.ts``).
#:   cursor    no documented failure event; a ``type: "error"`` line carrying
#:             ``message`` / ``error`` is read the same way.
_STRUCTURED_ERROR_EVENT_TYPES: frozenset[str] = frozenset(
    {"error", "turn.failed", "session.error"}
)


def _error_field_texts(value: Any) -> list[str]:
    """Flatten an ``error``-shaped field into the message strings it carries.

    Accepts the shapes the backends use: a bare string, a list of strings
    (Claude Code's ``errors``), or an object with ``message`` / ``error`` /
    ``data.message`` (opencode's ``{name, data: {message}}``).  Anything else
    contributes nothing -- an unknown shape is not evidence.
    """
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, list):
        texts: list[str] = []
        for item in value:
            texts.extend(_error_field_texts(item))
        return texts
    if isinstance(value, dict):
        texts = []
        for key in ("message", "error"):
            texts.extend(_error_field_texts(value.get(key)))
        data = value.get("data")
        if isinstance(data, dict):
            texts.extend(_error_field_texts(data.get("message")))
        return texts
    return []


def _structured_error_texts(
    backend: str, parsed: dict[str, Any] | None, stdout: str
) -> list[str]:
    """Return every failure the backend reported through a *structured* field.

    This is the only stdout-derived text the credential classifier is ever
    asked about.  It is deliberately narrow: the fields read here are the ones
    the CLI itself uses to say "this invocation failed", never the agent's
    prose and never a tool's captured output.  A candidate whose test suite
    prints ``Token refresh failed: 400`` into a ``command_execution`` item, or
    whose final message quotes a 401, must not be able to talk HELIX into
    declaring the operator's login dead.

    JSONL backends (codex, cursor, opencode): the ``message`` / ``error`` of
    events whose type is in :data:`_STRUCTURED_ERROR_EVENT_TYPES`.  When the
    strict parse has not happened yet (non-zero exit), the stream is read
    leniently so a malformed line elsewhere does not hide the error event.

    Envelope backends (claude, agy): the envelope's ``errors`` list and
    ``error`` field always, and ``result`` only when the envelope flags
    ``is_error`` -- on a successful turn ``result`` is the assistant's prose.
    """
    if backend in {"codex", "cursor", "opencode"}:
        events: Any = parsed.get("events") if parsed is not None else None
        if not isinstance(events, list):
            events = _parse_jsonl_output(
                stdout,
                backend=backend,
                cmd_str="",
                worktree_path="",
                stderr="",
                exit_code=0,
                strict=False,
            )["events"]
        texts: list[str] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            if event.get("type") not in _STRUCTURED_ERROR_EVENT_TYPES:
                continue
            texts.extend(_error_field_texts(event.get("message")))
            texts.extend(_error_field_texts(event.get("error")))
        return texts

    envelope: dict[str, Any] | None = parsed
    if envelope is None:
        try:
            loaded = json.loads(stdout) if stdout.strip() else None
        except (json.JSONDecodeError, ValueError, RecursionError):
            loaded = None
        if isinstance(loaded, dict):
            envelope = loaded
    if envelope is None:
        return []
    texts = []
    texts.extend(_error_field_texts(envelope.get("errors")))
    texts.extend(_error_field_texts(envelope.get("error")))
    if envelope.get("is_error") is True:
        texts.extend(_error_field_texts(envelope.get("result")))
    return texts


def _credential_failure_evidence(
    backend: str,
    parsed: dict[str, Any] | None,
    result: subprocess.CompletedProcess[str],
) -> tuple[str, str] | None:
    """Return ``(marker, where)`` when this invocation is a credential failure.

    Structured error fields (see :func:`_structured_error_texts`) are read on
    every exit code: they are how a backend that swallows its own failure and
    exits 0 -- codex on a rejected refresh, measured on codex-cli 0.130.0 --
    still says what went wrong.

    On a **non-zero** exit stderr is read as well: CLIs routinely report an
    unusable credential there without emitting a structured event.  The raw
    stdout stream is never scanned on any exit code; for the JSONL backends it
    is the whole transcript, tool output and agent prose included.
    """
    for text in _structured_error_texts(backend, parsed, result.stdout or ""):
        marker = credential_failure_marker(text)
        if marker is not None:
            return marker, "structured error event"
    if result.returncode == 0:
        return None
    marker = credential_failure_marker(result.stderr or "")
    if marker is not None:
        return marker, "stderr"
    return None


def _credential_refresh_error(
    *,
    backend: str,
    backend_name: str,
    marker: str,
    where: str,
    cmd_str: str,
    worktree_path: str,
    result: subprocess.CompletedProcess[str],
    retried: bool = False,
) -> CredentialRefreshError:
    transient = credential_failure_is_transient(marker)
    if transient:
        # A lost refresh race.  The winner has stored a fresh credential, so
        # sending the operator to re-login would discard a working one.
        message = (
            f"{backend_name} lost a refresh race on the shared credential "
            f"(matched {marker!r} in {where}"
            f"{'; failed again on retry' if retried else ''})"
        )
        suggestion = (
            f"This is a credential failure, not a failed mutation: another "
            f"candidate refreshed the shared {backend_name} login first and "
            "the token this invocation held was already spent. The refreshed "
            "credential is stored and should be usable"
            + (
                ", but a retry with it also failed. Run `helix resume`; the "
                "stored login most likely does not need to be redone, so only "
                "re-authenticate if this keeps recurring."
                if retried
                else "."
            )
        )
    else:
        message = (
            f"{backend_name} could not use its stored credential "
            f"(matched {marker!r} in {where})"
        )
        suggestion = (
            f"This is a credential failure, not a failed mutation: {backend_name} "
            "reported that its stored login could not be used or refreshed. "
            f"Re-authenticate with `helix sandbox login {backend}`, then resume "
            "the run; nothing is wrong with the candidate's code."
        )
    error = CredentialRefreshError(
        message,
        operation=f"{backend_name} invocation",
        phase="credential check",
        command=cmd_str,
        cwd=str(worktree_path),
        stdout=result.stdout,
        stderr=result.stderr,
        exit_code=result.returncode,
        suggestion=suggestion,
    )
    error.transient = transient
    return error


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

#: Where a sandboxed opencode candidate keeps its SQLite database: under the
#: per-candidate workspace copy, which the container sees at ``/workspace``.
OPENCODE_SANDBOX_DB_PATH = f"/workspace/{OPENCODE_STATE_DIR_NAME}/opencode/opencode.db"


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
        # The first attempt's copies after a lost refresh race (see
        # ``invoke_with_refresh_race_retry``).
        attempt_artifact_name(BACKEND_RESULT_ARTIFACT_NAME, 1),
        attempt_artifact_name(BACKEND_STDOUT_ARTIFACT_NAME, 1),
        attempt_artifact_name(BACKEND_STDERR_ARTIFACT_NAME, 1),
        ".helix_artifacts/",
        "helix_batch.json",
        # Per-candidate OpenCode SQLite state (OPENCODE_DB isolation).
        # Each parallel opencode worker gets a fresh database here; keeps
        # the candidate git tree free of opencode's session transcripts.
        f"{OPENCODE_STATE_DIR_NAME}/",
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


# agy's ``--print-timeout`` is a Go ``time.Duration`` flag (default ``5m0s``)
# that aborts a headless run when it expires; per agy's changelog a mid-turn
# expiry returns *partial* output with only a stderr warning, so a cut-off
# mutation could pass for a finished one.  There is no disable value: ``0``
# and negative durations fail immediately with ``timeout waiting for
# response``, and no env var or settings key overrides the flag.  So we pass
# the largest duration Go can represent (``math.MaxInt64`` ns, ~292 years) --
# ``2562048h`` is rejected as out of range, so this is the ceiling of the
# flag's type.  Verified against agy 1.1.27.  This keeps agy on the same
# no-timeout policy every other backend already gets (their subprocesses run
# with no ``timeout`` at all; see ``test_no_timeout_in_subprocess``).
_AGY_PRINT_TIMEOUT = "2562047h47m16.854775807s"


def _build_backend_args(
    worktree_path: str,
    config: AgentConfig,
    prompt_artifact_name: str,
) -> list[str]:
    backend = config.backend
    if backend == "agy":
        args = [
            "agy",
            "--dangerously-skip-permissions",
            "--output-format",
            "json",
            "--print-timeout",
            _AGY_PRINT_TIMEOUT,
        ]
        if config.model:
            args.extend(["--model", config.model])
        if config.effort:
            args.extend(["--effort", config.effort])
        # Unlike ``claude``, agy's ``-p``/``--print`` (alias ``--prompt``) is
        # not a boolean flag: it takes the prompt as its VALUE, and a trailing
        # positional is ignored.  ``agy --print --output-format json <prompt>``
        # exits 2 with ``--print took "--output-format" as its prompt``
        # (verified against agy 1.1.27).  So every other flag goes first and
        # ``--print <prompt>`` is the final argv pair; Go's flag parser accepts
        # the space-separated form, and the instruction text never starts
        # with ``-``.  ``test_agy_cli_args_include_required_flags`` pins this.
        args.extend(["--print", _prompt_file_instruction(prompt_artifact_name)])
        return args

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
            # Pin codex's cross-session memory off (default off, but a flag)
            # so no candidate reads a prior candidate's session; see
            # ``helix.backends.BACKEND_FRESH_SESSION_ENV``.
            "-c",
            "features.memories=false",
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

    if backend == "opencode":
        args = [
            "opencode",
            "run",
            "--format",
            "json",
            "--auto",
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
    for raw_line in split_lf_lines(stdout):
        line = raw_line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            if strict:
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
    if backend in {"agy", "claude"}:
        return _parse_json_object_output(
            result.stdout,
            backend=backend,
            cmd_str=cmd_str,
            worktree_path=worktree_path,
            stderr=result.stderr,
            exit_code=result.returncode,
        )
    if backend in {"codex", "cursor", "opencode"}:
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


def _salvage_backend_usage(
    backend: str, result: subprocess.CompletedProcess[str]
) -> UsageStats:
    """Recover token usage from raw backend output without ever raising.

    Token usage is a fact about work the backend has already done.  It must
    therefore be recoverable independently of whether the output *also*
    yields a usable candidate — the strict parsers above raise
    :class:`MutationError` on a single malformed line, and until this
    existed that error discarded a whole invocation's accounting along with
    the candidate.

    The recovery is deliberately lenient and total:

    * JSONL backends reuse :func:`_parse_jsonl_output` with ``strict=False``,
      which collects the records that *did* decode and sets the rest aside.
      The usage record is one line of its own, so a malformed line elsewhere
      in the stream does not hide it.
    * Claude's single-object mode tries the object first, then falls back to
      the same lenient line scan for a stream that was truncated mid-object.

    Returns a zero-token :class:`UsageStats` when nothing is recoverable,
    which is the honest reading of "the backend reported no usage".
    """
    stdout = result.stdout or ""
    if not stdout.strip():
        return _normalise_usage_stats({})

    parsed: dict[str, Any] | None = None
    if backend == "claude":
        try:
            loaded = json.loads(stdout)
        except (json.JSONDecodeError, ValueError, RecursionError):
            loaded = None
        if isinstance(loaded, dict):
            parsed = loaded

    if parsed is None:
        try:
            parsed = _parse_jsonl_output(
                stdout,
                backend=backend,
                cmd_str="",
                worktree_path="",
                stderr="",
                exit_code=0,
                strict=False,
            )
        except (ValueError, RecursionError):  # pragma: no cover - defensive
            return _normalise_usage_stats({})

    try:
        return _normalise_usage_stats(parsed)
    except (ValueError, TypeError, RecursionError):  # pragma: no cover
        return _normalise_usage_stats({})


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
                    # agy: ``usage.cache_read_tokens``
                    "cache_read_tokens",
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
                    # agy: ``usage.thinking_tokens``
                    "thinking_tokens",
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
                # agy: top-level ``conversation_id``
                "conversation_id",
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
        for raw in split_lf_lines(path.read_text(encoding="utf-8")):
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
        for raw in split_lf_lines(path.read_text(encoding="utf-8")):
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
        for raw in split_lf_lines(path.read_text(encoding="utf-8")):
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


def _count_opencode_stdout_tool_events(path: Path) -> tuple[int, list[str]]:
    """Count tool invocations from an OpenCode ``--format json`` stdout artifact.

    OpenCode emits ``tool_use`` events with the tool name in ``part.tool``
    rather than a top-level ``name`` field.  This function extracts both
    the accurate count and the tool names.
    """
    count = 0
    names: list[str] = []
    try:
        for raw in split_lf_lines(path.read_text(encoding="utf-8")):
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


def _count_agy_transcript_tool_events(path: Path) -> tuple[int, list[str]]:
    """Count tool invocations from an Antigravity ``transcript.jsonl`` file.

    The native transcript (``brain/<conversation_id>/.system_generated/logs/
    transcript.jsonl``, observed against agy 1.1.28) is one JSON object per
    step; ``tool_calls: [{"name", "args"}]`` was observed on
    ``PLANNER_RESPONSE`` steps.  Every ``tool_calls`` entry with a string
    ``name`` counts as one event, whatever the step type, so a future step
    type carrying the same field is counted rather than silently dropped.
    """
    count = 0
    names: list[str] = []
    try:
        for raw_line in split_lf_lines(
            path.read_text(encoding="utf-8", errors="replace")
        ):
            line = raw_line.strip()
            if not line:
                continue
            try:
                step = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(step, dict):
                continue
            calls = step.get("tool_calls")
            if not isinstance(calls, list):
                continue
            for call in calls:
                if isinstance(call, dict) and isinstance(call.get("name"), str):
                    count += 1
                    names.append(call["name"])
    except OSError:
        return 0, []
    return count, names


# Dispatcher: maps backend name → per-backend counter function.  These read
# the file HELIX writes for the backend: the stdout artifact for the JSONL
# backends (codex/cursor/opencode) and the copied native transcript for
# claude and agy (``_collect_backend_transcript_artifacts``).
#
# ``agy``'s ``--output-format json`` stdout (observed against agy 1.1.27) is a
# single envelope -- ``conversation_id``, ``status``, ``response``, ``error``
# (on failure), ``duration_seconds``, ``num_turns`` and a ``usage`` block --
# with no per-tool event list, so its counter is fed the native transcript
# and is not applied to stdout.
_TRANSCRIPT_TOOL_COUNTERS: dict[str, Callable[[Path], tuple[int, list[str]]]] = {
    "claude": _count_claude_transcript_tool_events,
    "agy": _count_agy_transcript_tool_events,
    "codex": _count_codex_stdout_tool_events,
    "cursor": _count_cursor_stdout_tool_events,
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


def _export_sqlite_session_rows(
    db_path: Path,
    *,
    session_id: str,
    tables: tuple[tuple[str, str], ...],
    dst: Path,
) -> int:
    """Write this session's rows from ``tables`` in ``db_path`` to ``dst`` as JSONL.

    One line per row, ``{"table": <name>, "row": {<column>: <value>, ...}}``,
    ordered as the tables are listed.  ``db_path`` must be a private copy:
    sqlite may replay the sibling ``-wal`` into it.  Returns the row count.
    """
    import sqlite3

    written = 0
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        with dst.open("w", encoding="utf-8") as out:
            for table, column in tables:
                rows = conn.execute(
                    f'SELECT * FROM "{table}" WHERE "{column}" = ? ORDER BY rowid',
                    (session_id,),
                )
                for row in rows:
                    record = {
                        key: (
                            value.decode("utf-8", "replace")
                            if isinstance(value, bytes)
                            else value
                        )
                        for key, value in dict(row).items()
                    }
                    out.write(json.dumps({"table": table, "row": record}) + "\n")
                    written += 1
    return written


def _copy_local_backend_transcripts(
    worktree_path: str,
    *,
    backend: str,
    session_id: str,
    artifact_dir: str = BACKEND_TRANSCRIPT_ARTIFACT_DIR,
    home: Path | None,
    miss_level: int = logging.DEBUG,
) -> list[dict[str, Any]]:
    """Collect ``backend``'s native transcript for ``session_id`` into the worktree.

    Driven by ``BACKEND_TRANSCRIPT_SOURCES``.  With ``home`` set (unsandboxed)
    the files are globbed under that directory and copied; with ``home`` of
    ``None`` (sandboxed) the sandbox helper has already staged them under the
    artifact dir and only their presence is recorded.  An sqlite-backed store
    is then reduced to this session's rows and the staged copy removed.  A
    missing source yields an ``available: False`` entry with a reason, mirroring
    the claude path, plus a debug log saying why nothing was preserved.
    """
    source_spec = BACKEND_TRANSCRIPT_SOURCES.get(backend)
    if source_spec is None or not source_spec.files:
        logger.debug(
            "%s transcript not preserved: no known native transcript location",
            backend,
        )
        return []
    if not TRANSCRIPT_SESSION_ID_RE.match(session_id):
        logger.debug(
            "%s transcript not preserved: unexpected session id %r",
            backend,
            session_id,
        )
        return []
    rel_dir = Path(artifact_dir) / backend
    out_dir = Path(worktree_path) / rel_dir
    artifacts: list[dict[str, Any]] = []
    staged: list[Path] = []
    try:
        for pattern, dest_name in source_spec.files:
            rel_path = rel_dir / dest_name.format(session_id=session_id)
            dst = Path(worktree_path) / rel_path
            entry: dict[str, Any] = {
                "backend": backend,
                "session_id": session_id,
                "path": str(rel_path),
                "source": "sandbox_auth_volume",
                "available": dst.is_file(),
            }
            if home is not None and not dst.is_file():
                glob_pattern = pattern.format(session_id=session_id)
                matches = sorted(p for p in home.glob(glob_pattern) if p.is_file())
                entry["source"] = str(matches[0]) if matches else str(home / glob_pattern)
                if matches:
                    try:
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        import shutil

                        shutil.copy2(matches[0], dst)
                        entry["available"] = True
                    except OSError as exc:
                        entry["reason"] = f"copy_failed: {exc}"
            if not entry["available"]:
                entry.setdefault("reason", "transcript_not_found")
            if dest_name.startswith("."):
                staged.append(dst)
            else:
                artifacts.append(entry)
        if source_spec.sqlite_export is not None:
            raw_name, tables = source_spec.sqlite_export
            raw_db = out_dir / raw_name.format(session_id=session_id)
            rel_path = rel_dir / f"{session_id}.jsonl"
            entry = {
                "backend": backend,
                "session_id": session_id,
                "path": str(rel_path),
                "source": str(raw_db)
                if home is None
                else str(home / source_spec.files[0][0]),
                "available": False,
            }
            if raw_db.is_file():
                try:
                    rows = _export_sqlite_session_rows(
                        raw_db,
                        session_id=session_id,
                        tables=tables,
                        dst=Path(worktree_path) / rel_path,
                    )
                    entry["available"] = rows > 0
                    if rows == 0:
                        entry["reason"] = "session_rows_not_found"
                except Exception as exc:  # noqa: BLE001 - sqlite errors are not fatal
                    entry["reason"] = f"export_failed: {exc}"
            else:
                entry["reason"] = "transcript_not_found"
            artifacts.append(entry)
    finally:
        # ``staged`` holds raw CLI state copied out verbatim -- for opencode
        # the whole ``opencode.db``, whose ``account`` table carries the OAuth
        # tokens and whose rows cover every candidate, not just this one.  It
        # is removed here rather than after the export so that an interrupt
        # (Ctrl-C, SIGTERM) between the copy and the export cannot leave a
        # credential-bearing file behind in the candidate worktree.
        for path in staged:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
    for entry in artifacts:
        if not entry["available"]:
            logger.log(
                miss_level,
                "%s transcript for candidate %s not preserved: %s (expected %s from %s)",
                backend,
                Path(worktree_path).name,
                entry.get("reason"),
                entry["path"],
                entry["source"],
            )
    return artifacts


# Backends whose tool events are counted from the copied native transcript
# rather than from the stdout artifact.
_NATIVE_TRANSCRIPT_COUNTED_BACKENDS = frozenset({"claude", "agy"})


def _collect_backend_transcript_artifacts(
    worktree_path: str,
    *,
    backend: str,
    usage: UsageStats,
    sandbox: SandboxConfig | None,
) -> list[dict[str, Any]]:
    """Copy the backend's native transcript for this invocation into the worktree.

    Claude keeps its pre-existing path (``claude_transcript_root`` /
    ``HELIX_CLAUDE_TRANSCRIPT_ROOT``); every other backend is located through
    ``BACKEND_TRANSCRIPT_SOURCES``.  In sandbox mode the files were copied out
    of the auth volume by ``helix.sandbox`` before sync-back; unsandboxed they
    are read from the operator's ``$HOME``.

    The copy is always made and kept: it is the run's only durable record
    once the sandbox resets the backend's state at the next generation, and
    it feeds tool-event counting for the backends whose stdout carries none
    (claude, agy).
    """
    session_id = usage.session_id
    if not isinstance(session_id, str) or not session_id:
        logger.debug(
            "%s transcript not preserved: no session id in backend output", backend
        )
        return []
    artifact_dir = (
        sandbox.transcript_artifact_dir
        if sandbox is not None
        else BACKEND_TRANSCRIPT_ARTIFACT_DIR
    )
    sandboxed = sandbox is not None and sandbox.enabled
    # The sandbox wipes the backend's state at the next generation start, so
    # a miss there is the last chance to notice; unsandboxed, the operator's
    # $HOME keeps the source and debug is enough.
    miss_level = logging.WARNING if sandboxed else logging.DEBUG
    if backend != "claude":
        artifacts = _copy_local_backend_transcripts(
            worktree_path,
            backend=backend,
            session_id=session_id,
            artifact_dir=artifact_dir,
            home=None if sandboxed else Path.home(),
            miss_level=miss_level,
        )
    else:
        transcript_root = (
            "sandbox_auth_volume"
            if sandboxed
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
        artifacts = [artifact] if artifact is not None else []
        if artifact is not None and not artifact.get("available"):
            logger.log(
                miss_level,
                "claude transcript for candidate %s not preserved: %s (expected %s from %s)",
                Path(worktree_path).name,
                artifact.get("reason"),
                artifact["path"],
                artifact["source"],
            )
    if backend in _NATIVE_TRANSCRIPT_COUNTED_BACKENDS:
        # The structured stdout of these backends omits per-turn tool
        # invocations.  Now that the native transcript is on disk, read it
        # and patch the ``usage`` object with the accurate counts.  Only
        # applied when the summary gave 0 tool events to avoid overriding a
        # non-zero value that a future CLI version might expose.  The first
        # artifact is the primary transcript (``<session_id>.jsonl``).
        primary = next((a for a in artifacts if a.get("available")), None)
        if primary is not None:
            dst = Path(worktree_path) / Path(primary["path"])
            tc, tn = _count_transcript_tool_events(dst, backend)
            if tc > 0 and usage.tool_event_count == 0:
                usage.tool_event_count = tc
                usage.tool_names = list(tn)
    return artifacts


def _write_backend_artifacts(
    worktree_path: str,
    *,
    backend: str,
    command: str,
    result: subprocess.CompletedProcess[str],
    parsed: dict[str, Any] | None,
    sandbox: SandboxConfig | None = None,
    fallback_usage: UsageStats | None = None,
) -> None:
    try:
        wt = Path(worktree_path)
        _ignore_helix_artifacts(wt)
        (wt / BACKEND_STDOUT_ARTIFACT_NAME).write_text(result.stdout or "")
        (wt / BACKEND_STDERR_ARTIFACT_NAME).write_text(result.stderr or "")
        # ``parsed is None`` means the strict parse failed.  Record the
        # leniently recovered usage rather than zeros, so the on-disk
        # artifact stays a faithful account of what the invocation spent.
        if parsed is not None:
            usage = _normalise_usage_stats(parsed)
        elif fallback_usage is not None:
            usage = fallback_usage
        else:
            usage = _normalise_usage_stats({})
        # For the JSONL backends the stdout stream IS the transcript; patch
        # ``usage`` with backend-specific tool-event counts now that the
        # stdout artifact is on disk.  Claude and agy are handled inside
        # ``_collect_backend_transcript_artifacts`` where the native
        # transcript file is copied first.
        if backend not in _NATIVE_TRANSCRIPT_COUNTED_BACKENDS:
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


def _combine_usage(
    first: UsageStats | None, second: UsageStats | None
) -> UsageStats | None:
    """Sum two per-attempt usage records into one, tolerating ``None``.

    Used when an invocation is retried after a lost refresh race: both
    attempts spent tokens, and the caller sees only one ``UsageStats``, so the
    first attempt's spend has to ride along with the second's.  Neither input
    is mutated.  ``session_id`` is taken from the later attempt, which is the
    one whose output the caller receives; the earlier one is used only when
    the later attempt reported none.  Returns ``None`` only when both inputs
    are ``None``, which keeps the "no backend ran" reading of a missing
    record intact.
    """
    if first is None:
        return second
    total = UsageStats.from_dict(first.to_dict())
    if second is not None:
        total.add(second)
        if second.session_id is not None:
            total.session_id = second.session_id
    return total



# ---------------------------------------------------------------------------
# One-shot retry after a lost refresh race
# ---------------------------------------------------------------------------


def attempt_artifact_name(name: str, attempt: int) -> str:
    """Name under which an earlier attempt's artifact is kept.

    ``.helix_backend_result.json`` -> ``.helix_backend_result.attempt1.json``;
    a name without an extension gets the suffix appended.
    """
    stem, dot, ext = name.rpartition(".")
    if not stem:
        return f"{name}.attempt{attempt}"
    return f"{stem}.attempt{attempt}{dot}{ext}"


_ATTEMPT_ARTIFACT_NAMES: tuple[str, ...] = (
    BACKEND_RESULT_ARTIFACT_NAME,
    BACKEND_STDOUT_ARTIFACT_NAME,
    BACKEND_STDERR_ARTIFACT_NAME,
    BACKEND_TRANSCRIPT_ARTIFACT_DIR,
)


def _stash_attempt_artifacts(worktree_path: str, stash_dir: Path) -> None:
    """Copy the invocation artifacts of *worktree_path* into *stash_dir*."""
    wt = Path(worktree_path)
    for name in _ATTEMPT_ARTIFACT_NAMES:
        src = wt / name
        dst = stash_dir / name
        try:
            if src.is_dir():
                shutil.copytree(src, dst)
            elif src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        except OSError as exc:
            logger.debug("could not stash %s from %s: %s", name, worktree_path, exc)


def _restore_attempt_artifacts(
    worktree_path: str, stash_dir: Path, *, attempt: int
) -> None:
    """Write stashed artifacts into *worktree_path* under their attempt names."""
    wt = Path(worktree_path)
    for name in _ATTEMPT_ARTIFACT_NAMES:
        src = stash_dir / name
        dst = wt / attempt_artifact_name(name, attempt)
        try:
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            elif src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
        except OSError as exc:
            logger.debug("could not restore %s into %s: %s", name, worktree_path, exc)


def _note_retry_in_result_artifact(
    worktree_path: str,
    *,
    first_usage: UsageStats | None,
    combined_usage: UsageStats,
) -> None:
    """Make the final attempt's result artifact account for both attempts.

    The artifact on disk describes the retry only; the caller is charged the
    sum.  Record the sum and point at the first attempt's artifact so the
    file stays a faithful account of what the invocation spent.
    """
    path = Path(worktree_path) / BACKEND_RESULT_ARTIFACT_NAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    if not isinstance(payload, dict):
        return
    payload["attempts"] = 2
    payload["retry_of"] = attempt_artifact_name(BACKEND_RESULT_ARTIFACT_NAME, 1)
    payload["retry_reason"] = "lost refresh race on the shared credential"
    payload["usage_first_attempt"] = (
        first_usage.to_dict() if first_usage is not None else None
    )
    payload["usage_combined"] = combined_usage.to_dict()
    try:
        path.write_text(json.dumps(payload, indent=2))
    except OSError as exc:
        logger.debug("could not annotate %s: %s", path, exc)


def invoke_with_refresh_race_retry(
    child: Candidate,
    *,
    backend_name: str,
    invoke: Callable[[Candidate, bool], tuple[dict[str, Any], UsageStats]],
    fresh_child: Callable[[], Candidate],
    remove_child: Callable[[Candidate], None],
    on_child_replaced: Callable[[Candidate], None],
    record_usage: Callable[[UsageStats], None] | None = None,
    on_refresh_race_recovered: Callable[[str], None] | None = None,
) -> tuple[Candidate, UsageStats]:
    """Run *invoke* on *child*, retrying once on a lost refresh race.

    A transient :class:`CredentialRefreshError` means another candidate
    refreshed the shared login first and the token this invocation held was
    already spent; the winner has stored a working credential, so a second
    invocation against it is the right response.  The retry must not run on
    the tree the first attempt was editing: with the sandbox, ``sync_back``
    copies the first attempt's partial edits into the worktree regardless of
    exit code, and the retry's workspace copy would commit them as its
    synthetic baseline, hiding them from ``git diff``.  So the retry starts
    from a *fresh* worktree -- ``remove_child`` then ``fresh_child`` -- and
    the first attempt's artifacts are kept beside the retry's under
    ``attempt1`` names (:func:`attempt_artifact_name`).

    ``on_child_replaced`` is called with the fresh worktree the moment it
    exists, so the caller's cleanup handlers always address the live one.

    Usage accounting: the first attempt's spend rides along.  On a successful
    retry the returned usage is the sum and the result artifact is annotated;
    when the retry raises a :class:`HelixError` the sum is attached to it;
    when it raises anything else (a sandbox ``TimeoutExpired``, an
    ``OSError``) the first attempt's spend is handed to ``record_usage``
    directly before the exception propagates, since nothing else will carry it.

    A recovered race is reported through ``on_refresh_race_recovered`` so it
    reaches the operator (the end-of-run summary), not only the log file.
    A second loss in a row is raised as-is; there is no loop.
    """
    try:
        _, usage = invoke(child, False)
        return child, usage
    except CredentialRefreshError as exc:
        if not exc.transient:
            raise
        first_error = exc
        first_usage = exc.usage

    logger.warning(
        "%s lost a refresh race on the shared credential; retrying once from "
        "a fresh worktree against the refreshed credential (%s).",
        backend_name,
        first_error,
    )
    with tempfile.TemporaryDirectory(prefix="helix-attempt1-") as stash:
        _stash_attempt_artifacts(child.worktree_path, Path(stash))
        try:
            remove_child(child)
        except Exception as exc:  # noqa: BLE001 - re-clone below reports the real problem
            logger.debug("could not remove %s before retry: %s", child.worktree_path, exc)
        child = fresh_child()
        on_child_replaced(child)
        _restore_attempt_artifacts(child.worktree_path, Path(stash), attempt=1)

    try:
        _, usage = invoke(child, True)
    except HelixError as retry_exc:
        retry_exc.usage = _combine_usage(first_usage, retry_exc.usage)
        raise
    except Exception:
        if record_usage is not None and first_usage is not None:
            record_usage(first_usage)
        raise

    combined = _combine_usage(first_usage, usage)
    assert combined is not None  # ``usage`` is never None
    _note_retry_in_result_artifact(
        child.worktree_path, first_usage=first_usage, combined_usage=combined
    )
    message = (
        f"{child.id} recovered after a lost refresh race on the shared "
        f"{backend_name} credential: the first attempt reported "
        f"{first_error}; the retry from a fresh worktree succeeded. Both "
        "attempts' tokens are charged and the first attempt's output is kept "
        f"as {attempt_artifact_name(BACKEND_RESULT_ARTIFACT_NAME, 1)}."
    )
    logger.warning("%s", message)
    if on_refresh_race_recovered is not None:
        on_refresh_race_recovered(message)
    return child, combined


def invoke_claude_code(
    worktree_path: str,
    prompt: str,
    config: AgentConfig,
    passthrough_env: list[str] | None = None,
    fixed_env: dict[str, str] | None = None,
    sandbox: SandboxConfig | None = None,
    prompt_artifact_name: str = MUTATION_PROMPT_ARTIFACT_NAME,
    *,
    retried: bool = False,
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
    retried:
        True when this is the one-shot retry after a lost refresh race (see
        :func:`invoke_with_refresh_race_retry`).  Only changes what a second
        credential failure says to the operator; this function itself never
        retries, because a retry has to start from a fresh worktree and only
        the caller owns the worktree.

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
    CredentialRefreshError
        When the backend reported, in its own structured error fields or on
        stderr, that its stored credential could not be used or refreshed.
        ``transient`` is set when it merely lost a refresh race.
    """
    if _MUTATOR_OVERRIDE is not None:
        return _MUTATOR_OVERRIDE(worktree_path, prompt, config)
    backend = config.backend
    backend_name = backend_display_name(backend)
    sandboxed = sandbox is not None and sandbox.enabled
    backend_worktree_path = "/workspace" if sandboxed else worktree_path
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
    # The fresh-session switch applies to sandboxed and unsandboxed runs
    # alike.  An operator who names the same key in ``[env]`` has made a
    # deliberate choice, so their value wins.
    for key, value in BACKEND_FRESH_SESSION_ENV.get(backend, {}).items():
        backend_env.setdefault(key, value)
    if backend == "opencode":
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
        # Fix: point OPENCODE_DB at a per-candidate database.  OPENCODE_DB
        # relocates opencode.db and its -wal/-shm companions and nothing else.
        # XDG_DATA_HOME would also solve the locking problem but moves
        # auth.json along with the database, which makes an existing opencode
        # login invisible -- verified against the real CLI, where
        # `opencode auth list` then reports 0 credentials.  The on-disk layout
        # is unchanged from the earlier XDG_DATA_HOME approach:
        #   <worktree>/.helix_opencode_state/opencode/opencode.db
        # Each parallel worker gets its own fresh database; no contention.
        #
        # The sandbox needs the same knob, not less: every sandboxed agent
        # container mounts the one ``helix-auth-opencode`` volume at
        # ``/home/node`` read-write with ``HOME=/home/node`` forced, so
        # without it every concurrent container opens the same
        # ``~/.local/share/opencode/opencode.db``.  Container isolation
        # separates ``/workspace``, which is why the database goes there:
        # the container's cwd is the per-candidate workspace copy, the
        # sandbox creates the directory in that copy before the run (see
        # ``helix.sandbox.run_sandboxed_commands``), and ``.helix*`` paths
        # are excluded from sync-back and gitignored like the unsandboxed
        # layout.
        if sandboxed:
            backend_env["OPENCODE_DB"] = OPENCODE_SANDBOX_DB_PATH
        else:
            opencode_state_dir = (
                Path(worktree_path) / OPENCODE_STATE_DIR_NAME / "opencode"
            )
            opencode_state_dir.mkdir(parents=True, exist_ok=True)
            backend_env["OPENCODE_DB"] = str(opencode_state_dir / "opencode.db")

    if sandboxed:
        assert sandbox is not None
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

    # Recover the usage record from the raw stream FIRST, with a parse
    # that cannot raise.  Everything below this line can fail -- and
    # when it does, the tokens have still been spent.  Charging must
    # not be conditional on the candidate being usable, so every error
    # raised from here carries this on ``HelixError.usage`` for the
    # caller to charge.
    spent_usage = _salvage_backend_usage(backend, result)

    def _raise_if_credential_failure(parsed: dict[str, Any] | None) -> None:
        evidence = _credential_failure_evidence(backend, parsed, result)
        if evidence is None:
            return
        marker, where = evidence
        logger.error(
            "Credential failure detected for %s (exit %d) in %s: matched %r",
            backend_name,
            result.returncode,
            where,
            marker,
        )
        raise _credential_refresh_error(
            backend=backend,
            backend_name=backend_name,
            marker=marker,
            where=where,
            cmd_str=cmd_str,
            worktree_path=worktree_path,
            result=result,
            retried=retried,
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
            # A backend can report an unusable credential and still exit 0 --
            # measured on codex-cli 0.130.0, whose refresh failure is swallowed
            # entirely (exit 0, empty stderr, even at RUST_LOG=info).  Its
            # own structured error event is the only signal left on this
            # path, so read it here rather than letting the failure pass as
            # a successful-but-useless mutation.
            _raise_if_credential_failure(parsed)
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

        # Non-zero exit.  Claude's max-turns exhaustion is partial success
        # and is decided FIRST: the subprocess may have already produced
        # useful edits, and the envelope's ``result`` is the assistant's
        # own prose, which no later classifier may read as evidence.
        if backend == "claude":
            try:
                parsed = _parse_backend_output(
                    backend,
                    result,
                    cmd_str=cmd_str,
                    worktree_path=worktree_path,
                )
            except MutationError:
                parsed = None
            else:
                if parsed.get("subtype") == "error_max_turns":
                    logger.warning(
                        "Claude Code reached max_turns limit (%s turns) — treating as partial success.",
                        parsed.get("num_turns", "?"),
                    )
                    return parsed, _normalise_usage_stats(parsed)

        # Then a credential failure, from structured error fields and stderr
        # only.  The markers are disjoint from the rate-limit keywords, and
        # "the login is unusable" is a strictly more actionable verdict than
        # "the backend exited non-zero".
        _raise_if_credential_failure(parsed)

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

        parsed = _parse_backend_output(
            backend,
            result,
            cmd_str=cmd_str,
            worktree_path=worktree_path,
        )

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
    except HelixError as exc:
        # Attach only when the raiser did not already supply a more
        # precise record; never overwrite one.
        if exc.usage is None:
            exc.usage = spent_usage
        raise
    finally:
        _write_backend_artifacts(
            worktree_path,
            backend=backend,
            command=cmd_str,
            result=result,
            parsed=parsed,
            sandbox=sandbox,
            fallback_usage=spent_usage,
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
    record_usage: Callable[[UsageStats], None] | None = None,
    on_refresh_race_recovered: Callable[[str], None] | None = None,
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
    record_usage:
        Optional sink called exactly once with the backend's token usage,
        whether or not the mutation produced a usable candidate.  The tokens
        are spent either way, so this is how a caller charges the budget for
        an attempt that ends in ``None``.  Not called when no backend
        invocation happened (e.g. the worktree clone raised).
    on_refresh_race_recovered:
        Optional sink told, in operator-facing words, when the invocation lost
        a refresh race on the shared credential and succeeded on its one retry
        from a fresh worktree (see :func:`invoke_with_refresh_race_retry`).

    Returns
    -------
    Candidate | None
        The new candidate on success, or ``None`` if mutation failed.
    """

    def _fresh_child() -> Candidate:
        fresh = clone_candidate(parent, new_id, base_dir)
        fresh.operation = "mutate"
        if prepare_worktree is not None:
            prepare_worktree(fresh)
        return fresh

    child = _fresh_child()

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
    # prior prompt file as part of the codebase.  Written here, before the
    # first invocation, so a collision is raised as-is; the retry path
    # rewrites it into its fresh worktree (the write is idempotent).
    _write_mutation_prompt_artifact(child.worktree_path, prompt)

    def _invoke(
        target: Candidate, retried: bool
    ) -> tuple[dict[str, Any], UsageStats]:
        return invoke_claude_code(
            target.worktree_path,
            prompt,
            config.agent,
            passthrough_env=config.passthrough_env,
            fixed_env=config.env,
            sandbox=config.sandbox,
            prompt_artifact_name=_write_mutation_prompt_artifact(
                target.worktree_path, prompt
            ),
            retried=retried,
        )

    def _replace(fresh: Candidate) -> None:
        nonlocal child
        child = fresh

    def _discard_child() -> None:
        try:
            remove_worktree(child)
        except Exception:
            pass

    try:
        child, usage = invoke_with_refresh_race_retry(
            child,
            backend_name=backend_display_name(config.agent.backend),
            invoke=_invoke,
            fresh_child=_fresh_child,
            remove_child=remove_worktree,
            on_child_replaced=_replace,
            record_usage=record_usage,
            on_refresh_race_recovered=on_refresh_race_recovered,
        )
        child.usage = usage
        if record_usage is not None:
            record_usage(usage)
    except MutationError as exc:
        # The worktree is about to be removed and the candidate dropped, but
        # the tokens were spent.  Hand them to the caller before both go.
        if record_usage is not None and exc.usage is not None:
            record_usage(exc.usage)
        exc.operation = f"mutate {new_id} (parent: {parent.id})"
        print_helix_error(exc)
        _discard_child()
        return None
    except RateLimitError as exc:
        # Rate limit — clean up orphaned worktree, then re-raise so the parallel
        # futures handler in evolution.py can log it and continue with a smaller
        # proposal set.  A rate-limited invocation can still have burned tokens
        # before the limit hit, so the same handoff applies.
        if record_usage is not None and exc.usage is not None:
            record_usage(exc.usage)
        _discard_child()
        raise
    except CredentialRefreshError as exc:
        # The stored login, not this candidate, is what failed.  The tokens
        # spent before the credential gave out are still spent, so hand them
        # to the sink like the other paths do; then clean up the orphaned
        # worktree and re-raise so evolution.py can count it and name it as a
        # credential failure instead of filing it under "the agent wrote bad
        # code".
        if record_usage is not None and exc.usage is not None:
            record_usage(exc.usage)
        _discard_child()
        raise
    except Exception:
        # Anything else (a sandbox ``TimeoutExpired``, an ``OSError`` from
        # the launch) carries no usage of its own; whatever an earlier
        # attempt spent has already been handed to the sink.  Do not leak
        # the worktree on the way out.
        _discard_child()
        raise

    # NOTE: snapshot_candidate() is intentionally NOT called here.
    # The caller (evolution.py) is responsible for calling save_state()
    # BEFORE snapshot_candidate() so that state is persisted even if
    # the commit step crashes (e.g. empty-commit error).
    return child
