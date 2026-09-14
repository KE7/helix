"""Per-parent memory of the children already proposed from it.

A parent can be sampled many times.  Without a record of what its earlier
children tried, each mutation starts blind and the agent re-proposes ideas that
have already been evaluated -- including ones that were already rejected.

After every child is evaluated, one entry is appended under its parent: the
child's score, whether it was kept, and a short description of what it changed.
When that parent is sampled again, those entries are rendered into the next
mutation prompt.

The description is produced by a dedicated call made after the mutation
finishes, working from the diff the mutation left behind -- not by asking the
mutating agent to write it.  That agent is bounded by ``agent.max_turns`` and
routinely spends every turn on the optimization itself, so a "before you
finish" instruction is the first thing dropped.  Working from the diff also
means this does not depend on anything backend-specific.

The record is informational, never a filter.  A rejected approach is not
forbidden -- a child that lost on one minibatch can win on another -- so the
history is shown to the agent and nothing in the engine consults it to exclude
a choice.

Two properties the rest of the engine relies on:

* **Capture is early.**  The description is produced and read inside
  ``mutate``, while the child worktree is guaranteed to exist.  A rejected
  child's worktree is deleted by default, so a later read would find nothing.
* **Every child is kept.**  There is no cap on how many children a parent
  remembers.  A parent sampled many times is precisely the case this exists
  for, so truncating its record would discard the attempts most worth knowing
  about.  ``evolution.sibling_history`` is a plain on/off switch.

The self-report is untrusted input: an agent chooses its content and could
choose its path, so ``capture_summary`` refuses anything that is not a regular
file living directly in the worktree, and bounds what it returns.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The agent writes this file; HELIX reads it once and never commits it.  The
# ``.agent_`` prefix marks artifacts that belong to the agent side of the
# worktree, matching ``.agent_task_prompt.md``; ``.helix_*`` is the engine's own
# namespace and is what an operator hides from agents via
# ``sandbox.omit_from_agent``.
SUMMARY_ARTIFACT_NAME = ".agent_change_summary.md"

#: Default for ``evolution.sibling_summary_max_words``, which is the authority
#: at runtime.  Defined here so the prompt's request and the read-back
#: truncation share one source and cannot drift apart.
MAX_SUMMARY_WORDS = 150

# Stops a runaway artifact being read into memory at all.  The word cap is the
# real bound on what is stored; this only prevents a 50 MB file reaching it.
_MAX_READ_BYTES = 256 * 1024

_TRUNCATION_MARKER = " [... truncated]"

# Tab and newline survive so paragraphs and indentation render as written.
_ALLOWED_CONTROL_CHARS = frozenset("\t\n")

# Absent on platforms without symlinks; 0 makes the flag a no-op there.
_O_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)


def summary_prompt(
    *,
    objective: str = "",
    diff: str = "",
    max_words: int = MAX_SUMMARY_WORDS,
) -> str:
    """Build the whole prompt for the dedicated change-summary call.

    This is a standalone task, not an addendum to the mutation prompt: the
    call that runs it has no memory of the mutation and sees only what is
    written here, so the diff has to carry the content.

    Parameters
    ----------
    objective:
        What the run is optimizing, for context on *why* a change was made.
    diff:
        The mutation's changes against its parent.  Empty when git produced
        nothing usable -- the prompt then says so explicitly rather than
        leaving the model to invent an approach it cannot see.
    max_words:
        Word budget.  The same number bounds what :func:`capture_summary`
        keeps, so the request and the enforcement cannot drift apart.
    """
    sections = [
        "You are documenting one attempt in an automated code-evolution run.",
        "Another agent has just modified this workspace. Your only job is to describe what "
        "it did, so that the next attempt starting from the same parent can tell whether it "
        "would be repeating this one. Do not change any code.",
    ]
    if objective:
        sections.append(f"## Objective being optimized\n{objective}")
    if diff:
        sections.append(
            f"## The change that was made, as a diff against the parent\n```diff\n{diff}\n```"
        )
    else:
        sections.append(
            "## The change that was made\n"
            "No diff was available. Say that plainly in the file; do not guess at an "
            "approach you cannot see."
        )
    sections.append(
        "## Your task\n"
        f"Write `{SUMMARY_ARTIFACT_NAME}` in the workspace root: at most {max_words} words "
        "stating the approach taken and how it changed the parent. Name the technique, not "
        "just the files touched. Write that one file and nothing else.\n\n"
        "Everything you need is above, so do not explore the repository, read other files, "
        "or run anything: this should take one or two tool calls. Only the Claude backend "
        "enforces a turn limit, so on every other backend this sentence is the only bound "
        "there is."
    )
    return "\n\n".join(sections) + "\n"


def history_instruction() -> str:
    """Return the task-instruction bullet telling the agent to use the history.

    This is a directive about how to work, so it belongs with the other task
    instructions rather than inside the rendered data section.  Callers emit it
    only alongside a non-empty section, so it can speak in the indicative
    rather than hedging about whether anything is there.
    """
    return (
        "- Previous attempts from this parent are listed below. Read them before "
        "deciding what to change: they record what has already been tried here and "
        "what scored well or badly. Choose an approach that differs from those "
        "already tried, building on what scored well rather than repeating it."
    )


def _clean(text: str) -> str:
    """Drop control characters that would corrupt a prompt, keeping tab/newline."""
    return "".join(
        ch
        for ch in text
        if ch in _ALLOWED_CONTROL_CHARS or (ch >= " " and ch != "\x7f")
    )


def _truncate_words(text: str, max_words: int) -> str:
    """Cut *text* after *max_words* words, disclosing the cut inline.

    Cuts on the original string rather than re-joining a split, so blank lines,
    bullets and indentation before the cut survive as the agent wrote them.
    """
    end = -1
    for count, match in enumerate(re.finditer(r"\S+", text), start=1):
        if count > max_words:
            break
        end = match.end()
    if end < 0:
        return ""
    remainder = text[end:]
    if remainder.strip():
        return text[:end].rstrip() + _TRUNCATION_MARKER
    return text[:end].rstrip()


def capture_summary(
    worktree_path: str | Path, *, max_words: int = MAX_SUMMARY_WORDS
) -> str | None:
    """Read the agent's self-report out of *worktree_path*, or ``None``.

    Returns ``None`` whenever the artifact is missing, empty, unreadable, or
    fails a safety check -- all ordinary outcomes that leave the caller to fall
    back to a derived summary.

    The artifact is attacker-controlled, so it is refused unless it is a regular
    file resolving directly inside the worktree.  Without that, an agent could
    leave a symlink to any readable path and have its contents copied into
    persisted state and into a later prompt.

    Parameters
    ----------
    worktree_path:
        Root of the candidate worktree the agent just ran in.
    max_words:
        Word budget; longer reports are truncated with the cut disclosed.

    Returns
    -------
    str | None
        The bounded report, or ``None`` when there is nothing usable.
    """
    root = Path(worktree_path).resolve()
    path = root / SUMMARY_ARTIFACT_NAME
    try:
        # O_NOFOLLOW closes the check-then-open race a separate is_symlink()
        # test would leave open.
        fd = os.open(path, os.O_RDONLY | _O_NOFOLLOW)
    except OSError as exc:
        logger.debug(
            "no usable change summary in %s (%s)", root, exc.__class__.__name__
        )
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            logger.debug("change summary in %s refused: not a regular file", root)
            return None
        if path.resolve().parent != root:
            logger.debug(
                "change summary in %s refused: resolves outside the worktree", root
            )
            return None
        raw = os.read(fd, _MAX_READ_BYTES).decode("utf-8", errors="replace")
    except OSError as exc:
        logger.debug(
            "change summary in %s unreadable (%s)", root, exc.__class__.__name__
        )
        return None
    finally:
        os.close(fd)

    summary = _truncate_words(_clean(raw).strip(), max_words)
    return summary or None


def append_child(
    history: dict[str, list[dict[str, Any]]],
    parent_id: str,
    *,
    child_id: str,
    score: float | None,
    accepted: bool,
    summary: str | None,
) -> None:
    """Record one evaluated child under its parent, in place.

    Called once per child whatever the outcome: a rejected attempt is the half
    of the history that stops an approach being retried, and an accepted one is
    the half worth building on.

    Every child is retained; nothing is ever evicted.
    """
    history.setdefault(parent_id, []).append(
        {
            "child_id": child_id,
            "score": None if score is None else float(score),
            "accepted": bool(accepted),
            "summary": summary,
        }
    )


def _indent(text: str) -> str:
    """Indent every line so a multi-line report stays inside its bullet."""
    return "\n".join(
        f"    {line}" if line.strip() else "" for line in text.splitlines()
    )


def render_history(history: dict[str, list[dict[str, Any]]], parent_id: str) -> str:
    """Render every child of *parent_id* as a mutation-prompt section.

    Returns ``""`` only when the parent has no recorded children, so the caller
    can skip the section rather than emit an empty heading.  Whether the
    feature is on at all is the caller's decision, not this function's: an off
    run passes ``None`` to ``mutate`` and never gets here.

    Parameters
    ----------
    history:
        The full per-parent record.
    parent_id:
        Parent about to be mutated.
    """
    entries = history.get(parent_id) or []
    if not entries:
        return ""

    lines = [
        "## Previous Attempts From This Parent",
        # Descriptive only: what to *do* with this is stated once, up in the
        # task instructions (see ``history_instruction``).
        "Children already proposed from this same parent, oldest first. They were scored "
        "by the same evaluator you are being judged by.",
        "",
    ]
    for entry in entries:
        verdict = "KEPT" if entry.get("accepted") else "REJECTED"
        raw_score = entry.get("score")
        score = (
            f"{float(raw_score):+.4f}" if isinstance(raw_score, (int, float)) else "n/a"
        )
        lines.append(f"- [{verdict}] score {score}")
        summary = entry.get("summary")
        if isinstance(summary, str) and summary.strip():
            lines.append(_indent(summary.strip()))
        else:
            lines.append("    (this attempt left no summary)")
    return "\n".join(lines)


def normalize_history(value: object) -> dict[str, list[dict[str, Any]]]:
    """Coerce persisted history back into shape, dropping anything malformed.

    ``state.json`` is on-disk data that may predate this field or have been
    hand-edited, so nothing about its shape is assumed: anything that is not a
    list of dicts under a string key is dropped rather than trusted.
    """
    normalized: dict[str, list[dict[str, Any]]] = {}
    if not isinstance(value, dict):
        return normalized
    for parent_id, entries in value.items():
        if not isinstance(parent_id, str) or not isinstance(entries, list):
            continue
        kept = [entry for entry in entries if isinstance(entry, dict)]
        if kept:
            normalized[parent_id] = kept
    return normalized
