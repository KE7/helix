"""HELIX mutator: applies code mutations to evolutionary candidates via Claude Code."""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable

from helix.backends import BACKEND_AUTH_ENV, backend_display_name
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
# ``(worktree_path, prompt, config) -> dict[str, Any]``.  None (default) =
# unchanged production behavior.
_MUTATOR_OVERRIDE = None

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

AUTONOMOUS_SYSTEM_PROMPT = """\
Task instructions:
- Work directly on the requested code changes using the workspace files.
- Do not request confirmation or clarification; choose a reasonable approach and continue.
- If one approach fails, try an alternative and keep progressing.
- Use available tools to inspect, edit, and validate changes.
- When finished, print exactly: [MUTATION COMPLETE]
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
        Path to the (empty) seed worktree where the backend will write files.
    prompt:
        The seed-generation prompt built by :func:`build_seed_generation_prompt`.
    config:
        Full HELIX config (``config.agent`` is used for the backend invocation).

    Raises
    ------
    MutationError
        Propagated directly from :func:`invoke_claude_code` on failure.
    """
    invoke_claude_code(
        worktree_path,
        prompt,
        config.agent,
        passthrough_env=config.passthrough_env,
        fixed_env=config.env,
        sandbox=config.sandbox,
    )


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
    try:
        wt = Path(worktree_path)
        _ignore_helix_artifacts(wt)
        primary_path = wt / MUTATION_PROMPT_ARTIFACT_NAME
        fallback_path = wt / MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME
        try:
            with primary_path.open("x") as f:
                f.write(prompt)
            return MUTATION_PROMPT_ARTIFACT_NAME
        except FileExistsError:
            fallback_path.parent.mkdir(parents=True, exist_ok=True)
            with fallback_path.open("x") as f:
                f.write(prompt)
            return MUTATION_PROMPT_ARTIFACT_FALLBACK_NAME
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
        args.append(_prompt_file_instruction(prompt_artifact_name))
        return args

    if backend == "cursor":
        args = [
            "cursor",
            "agent",
            "--print",
            "--output-format", "stream-json",
            "--yolo",
            "--approve-mcps",
            "--trust",
            "--workspace", worktree_path,
        ]
        if config.model:
            args.extend(["--model", config.model])
        args.append(_prompt_file_instruction(prompt_artifact_name))
        return args

    if backend == "gemini":
        args = [
            "gemini",
            "--yolo",
            "--output-format", "stream-json",
        ]
        if config.model:
            args.extend(["--model", config.model])
        args.extend(["--prompt", _prompt_file_instruction(prompt_artifact_name)])
        return args

    if backend == "opencode":
        args = [
            "opencode",
            "run",
            "--format", "json",
            "--dangerously-skip-permissions",
        ]
        if config.model:
            args.extend(["--model", config.model])
        if config.effort:
            args.extend(["--variant", config.effort])
        args.extend([
            "--file",
            prompt_artifact_name,
            _prompt_file_instruction(prompt_artifact_name),
        ])
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


def _normalise_usage_stats(parsed: dict[str, Any]) -> dict[str, Any]:
    usage: dict[str, Any] = {}
    tool_event_count = 0
    for node in _walk_json(parsed):
        node_type = str(node.get("type", "")).lower()
        if "tool" in node_type:
            tool_event_count += 1
        for key, aliases in (
            ("input_tokens", ("input_tokens", "inputTokens", "prompt_tokens", "promptTokens", "input")),
            ("output_tokens", ("output_tokens", "outputTokens", "completion_tokens", "completionTokens", "output")),
            ("cached_input_tokens", ("cached_input_tokens", "cachedTokens", "cacheReadInputTokens", "cacheRead", "cached")),
            ("reasoning_tokens", ("reasoning_tokens", "reasoningTokens", "thoughts")),
            ("cost_usd", ("cost_usd", "costUsd", "total_cost_usd", "totalCostUsd", "total")),
        ):
            if key in usage:
                continue
            for alias in aliases:
                if alias in node:
                    value = _coerce_number(node[alias])
                    if value is not None:
                        usage[key] = value
                        break
        if "session_id" not in usage:
            for alias in ("session_id", "sessionId", "chat_id", "chatId", "thread_id", "threadId"):
                value = node.get(alias)
                if isinstance(value, str) and value:
                    usage["session_id"] = value
                    break
            if "session_id" not in usage:
                value = node.get("sessionID")
                if isinstance(value, str) and value:
                    usage["session_id"] = value
        if "cost_usd" not in usage:
            value = node.get("cost")
            coerced = _coerce_number(value)
            if coerced is not None:
                usage["cost_usd"] = coerced
    if tool_event_count:
        usage["tool_event_count"] = tool_event_count
    return usage


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
        else Path(os.environ.get("HELIX_CLAUDE_TRANSCRIPT_ROOT", Path.home() / ".claude/projects/-workspace"))
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
    usage: dict[str, Any],
    sandbox: SandboxConfig | None,
) -> list[dict[str, Any]]:
    if backend != "claude":
        return []
    if sandbox is not None and not sandbox.preserve_backend_transcripts:
        return []
    session_id = usage.get("session_id")
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
            "usage": usage,
            "transcript_artifacts": transcript_artifacts,
            "parsed": parsed,
        }
        (wt / BACKEND_RESULT_ARTIFACT_NAME).write_text(json.dumps(payload, indent=2))
    except OSError as e:
        logger.debug(
            "failed to write backend artifacts to %s: %s",
            worktree_path, e,
        )


def invoke_claude_code(
    worktree_path: str,
    prompt: str,
    config: AgentConfig,
    passthrough_env: list[str] | None = None,
    fixed_env: dict[str, str] | None = None,
    sandbox: SandboxConfig | None = None,
    prompt_artifact_name: str = MUTATION_PROMPT_ARTIFACT_NAME,
) -> dict[str, Any]:
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
    backend_worktree_path = "/workspace" if sandbox is not None and sandbox.enabled else worktree_path
    args = _build_backend_args(
        backend_worktree_path,
        config,
        prompt_artifact_name,
    )
    cmd_str = shlex.join(args)
    backend_env = _scrub_environment(passthrough_env=passthrough_env, fixed_env=fixed_env)
    _add_backend_auth_env(backend_env, backend)
    if backend == "gemini":
        backend_env["GEMINI_CLI_TRUST_WORKSPACE"] = "true"
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
            if backend == "claude":
                error_text = str(parsed.get("error", ""))
                if _looks_like_rate_limit(error_text):
                    logger.error("Rate limit detected in JSON response: %s", error_text[:200])
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
            return parsed

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
                if parsed.get("subtype") == "error_max_turns":
                    logger.warning(
                        "Claude Code reached max_turns limit (%s turns) — treating as partial success.",
                        parsed.get("num_turns", "?"),
                    )
                    return parsed
            except MutationError:
                parsed = None

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
        config.objective, eval_result, background, config.agent.max_turns,
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
        invoke_claude_code(
            child.worktree_path,
            prompt,
            config.agent,
            passthrough_env=config.passthrough_env,
            fixed_env=config.env,
            sandbox=config.sandbox,
            prompt_artifact_name=prompt_artifact_name,
        )
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
