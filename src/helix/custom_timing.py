"""Evaluator-facing timing channel: ``from helix import timed``.

Code that HELIX runs -- the evaluator, its ``extra_commands``, and scripts the
mutating agent runs -- can time its own operations::

    with timed("sam3", example=example_id):
        masks = sam3(frame)

    @timed("molmo")                 # sync or async functions
    def point(...): ...

    timed.record("owl-vit", 0.41)   # a duration measured elsewhere

Each call appends one JSON line, ``{"name", "seconds", "example", "error"}``,
to the per-invocation file named by ``HELIX_TIMING_LOG``; any language can
write the same line.  HELIX totals the file per ``(name, example)`` after the
call returns and records the totals in ``.helix/timing.jsonl``.  Outside a
HELIX invocation this is a no-op, so the same code runs unchanged locally.

Time where the result becomes real: a GPU call returns before its kernels
finish, so time up to ``.cpu()`` / ``synchronize()`` or around the request to
an inference server.
"""

from __future__ import annotations

import functools
import inspect
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Callable, TypeVar

from helix.asi import append_line

HELIX_TIMING_LOG_ENV = "HELIX_TIMING_LOG"

#: A longer line is dropped, so one ``write`` stays atomic (POSIX ``PIPE_BUF``)
#: across the threads and processes appending to the same file.
_MAX_LINE_BYTES = 4096
#: HELIX reads at most this much of one invocation's file.
MAX_LOG_BYTES = 64 * 1024 * 1024

F = TypeVar("F", bound=Callable[..., Any])


def _record(
    name: str, seconds: float, *, example: object = None, error: bool = False
) -> None:
    """Append one timing line; best-effort, never raises."""
    if not os.environ.get(HELIX_TIMING_LOG_ENV):
        return
    entry: dict[str, Any] = {"name": str(name), "seconds": float(seconds)}
    if example is not None:
        entry["example"] = str(example)
    if error:
        entry["error"] = True
    line = json.dumps(entry) + "\n"
    if len(line.encode()) <= _MAX_LINE_BYTES:
        append_line(HELIX_TIMING_LOG_ENV, line)


class timed:
    """Time a block (``with``) or every call of a function (decorator)."""

    record = staticmethod(_record)

    def __init__(self, name: str, *, example: object = None) -> None:
        self.name = name
        self.example = example
        self._started = 0.0

    def __enter__(self) -> timed:
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        _record(
            self.name,
            time.perf_counter() - self._started,
            example=self.example,
            error=exc_type is not None,
        )

    def __call__(self, fn: F) -> F:
        # A fresh instance per call, so concurrent calls never share a start.
        name, example = self.name, self.example
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                with timed(name, example=example):
                    return await fn(*args, **kwargs)

            return awrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timed(name, example=example):
                return fn(*args, **kwargs)

        return wrapper  # type: ignore[return-value]


def log_name() -> str:
    """A fresh per-invocation log file name (git-ignored in worktrees)."""
    return f".helix_timing_{uuid.uuid4().hex}.jsonl"


def totals(raw: str) -> list[dict[str, Any]]:
    """Total a timing log per ``(name, example)``; malformed lines are skipped."""
    by_key: dict[tuple[str, str | None], dict[str, Any]] = {}
    for line in raw.splitlines():
        try:
            entry = json.loads(line)
            name, seconds = str(entry["name"]), float(entry["seconds"])
        except (ValueError, TypeError, KeyError):
            continue
        example = entry.get("example")
        key = (name, None if example is None else str(example))
        total = by_key.setdefault(
            key,
            {"name": key[0], "example": key[1], "count": 0, "seconds": 0.0,
             "max_seconds": 0.0, "errors": 0},
        )
        total["count"] += 1
        total["seconds"] += seconds
        total["max_seconds"] = max(total["max_seconds"], seconds)
        total["errors"] += bool(entry.get("error"))
    return list(by_key.values())


def read(path: Path) -> str:
    """The first :data:`MAX_LOG_BYTES` of *path*, or ``""`` if it is missing."""
    try:
        with path.open("rb") as fh:
            return fh.read(MAX_LOG_BYTES).decode("utf-8", "replace")
    except OSError:
        return ""
