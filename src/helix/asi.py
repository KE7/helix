"""Evaluator-facing arbitrary side information channel."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


HELIX_ASI_LOG_ENV = "HELIX_ASI_LOG"


def log(*values: object, sep: str = " ", **fields: Any) -> None:
    """Record evaluator notes for HELIX mutation prompts.

    Evaluators can call ``from helix import log`` and then ``log(...)`` during
    a HELIX-managed evaluation.  HELIX captures the notes through a per-
    invocation file path in ``HELIX_ASI_LOG`` rather than through ordinary
    stdout, keeping stdout free for machine protocols such as ``HELIX_RESULT``.

    Outside a HELIX evaluator invocation this is a no-op, which lets evaluator
    code keep the same imports in local debug runs.
    """
    path = os.environ.get(HELIX_ASI_LOG_ENV)
    if not path:
        return

    record: dict[str, Any] = {}
    if values:
        record["message"] = sep.join(str(value) for value in values)
    record.update(fields)
    if not record:
        return

    # Single ``write`` call so the JSON record + trailing newline land
    # together: POSIX guarantees ``write(2)`` of <= ``PIPE_BUF`` bytes
    # to a file opened ``O_APPEND`` is atomic across processes /
    # threads, so concurrent ``helix.log()`` calls from a multi-worker
    # evaluator can interleave whole records but never split a single
    # record across two appends.
    line = json.dumps(record, sort_keys=True, default=str) + "\n"
    try:
        with Path(path).open("a", encoding="utf-8") as fh:
            fh.write(line)
    except OSError:
        # Logging is best-effort: if the worktree filesystem is
        # read-only or full we silently drop the note rather than
        # turning an evaluator-side observability call into a hard
        # crash.  Tested in ``test_asi.py``.
        return


def _render_field_value(value: Any) -> str:
    """Render a single ``helix.log(**fields)`` value for prompt context.

    Scalars round-trip via ``str(...)`` so a logged ``score=0.7`` shows
    up as ``0.7`` (not ``"0.7"``).  Nested dicts / lists / tuples
    serialize via ``json.dumps`` with ``default=str`` so an evaluator
    that logs structured data sees a readable JSON literal in the
    mutation prompt instead of Python's repr-style ``{'k': 'v'}``.
    """
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, sort_keys=True, default=str)
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def read_text(raw: str) -> str:
    """Render raw HELIX ASI log text."""
    lines: list[str] = []
    for raw_line in raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            lines.append(line)
            continue
        if not isinstance(payload, dict):
            lines.append(str(payload))
            continue
        message = payload.pop("message", None)
        if message is not None:
            lines.append(str(message))
        for key, value in sorted(payload.items()):
            lines.append(f"{key}: {_render_field_value(value)}")
    return "\n".join(lines)


def read(path: str | Path) -> str:
    """Read and render a HELIX ASI log file."""
    log_path = Path(path)
    try:
        raw = log_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""
    return read_text(raw)


def clear(path: str | Path) -> None:
    """Remove a HELIX ASI log file if it exists."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass
