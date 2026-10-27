"""Split one agent call into model, per-tool and other time from its transcript.

Each parser pairs every tool call's start with its end and finds the
transcript's first and last timestamps.  Everything else follows the same way
for every backend:

* ``tool:<name>`` -- the paired tool intervals;
* ``model`` -- the transcript window not covered by any tool: the backend was
  waiting on the model (thinking, generating, compacting, retrying);
* ``other`` -- the call's wall time outside the window: CLI and MCP startup
  and shutdown, and in sandboxed runs container start plus HELIX copying the
  worktree into the sandbox and back.

Backends whose transcripts cannot support this get one ``unavailable`` total:
cursor records no timestamps, and agy records only when each step starts, to
the second, with the model's writing of a tool's arguments inside the step
that follows the call.  A transcript a parser cannot read is ``unavailable``
too, never a guess.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

Interval = tuple[float, float]
#: ``(window_start, window_end, {tool name: [intervals]})``, or ``None``.
Parsed = tuple[float, float, dict[str, list[Interval]]] | None

_UNAVAILABLE = {
    "cursor": "cursor transcripts carry no timestamps",
    "agy": "agy timestamps are per step and to the second",
}


def _iso(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def _lines(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _claude(path: Path) -> Parsed:
    """``tool_use`` (assistant) to ``tool_result`` (user), paired by id.

    Subagent (sidechain) entries run inside their ``Task`` call's interval
    and are skipped so they are not counted twice.
    """
    times: list[float] = []
    open_calls: dict[str, tuple[str, float]] = {}
    tools: dict[str, list[Interval]] = {}
    for row in _lines(path):
        if row.get("isSidechain") or not isinstance(row.get("timestamp"), str):
            continue
        at = _iso(row["timestamp"])
        times.append(at)
        content = (row.get("message") or {}).get("content")
        for item in content if isinstance(content, list) else []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "tool_use":
                open_calls[str(item.get("id"))] = (str(item.get("name")), at)
            elif item.get("type") == "tool_result":
                call = open_calls.pop(str(item.get("tool_use_id")), None)
                if call is not None:
                    tools.setdefault(call[0], []).append((call[1], at))
    return (min(times), max(times), tools) if times else None


def _codex(path: Path) -> Parsed:
    """``function_call`` / ``custom_tool_call`` to its ``*_output``, by ``call_id``."""
    times: list[float] = []
    open_calls: dict[str, tuple[str, float]] = {}
    tools: dict[str, list[Interval]] = {}
    for row in _lines(path):
        if not isinstance(row.get("timestamp"), str):
            continue
        at = _iso(row["timestamp"])
        times.append(at)
        payload = row.get("payload")
        if row.get("type") != "response_item" or not isinstance(payload, dict):
            continue
        kind, call_id = payload.get("type"), str(payload.get("call_id"))
        if kind in ("function_call", "custom_tool_call"):
            open_calls[call_id] = (str(payload.get("name") or kind), at)
        elif kind in ("function_call_output", "custom_tool_call_output"):
            call = open_calls.pop(call_id, None)
            if call is not None:
                tools.setdefault(call[0], []).append((call[1], at))
    return (min(times), max(times), tools) if times else None


def _opencode(path: Path) -> Parsed:
    """Tool parts carry ``state.time.{start,end}``; messages bound the window."""
    times: list[float] = []
    tools: dict[str, list[Interval]] = {}
    for row in _lines(path):
        data = (row.get("row") or {}).get("data")
        if not isinstance(data, str):
            continue
        data = json.loads(data)
        when = data.get("time") or {}
        times += [when[k] / 1000 for k in ("created", "completed", "start", "end") if isinstance(when.get(k), (int, float))]
        span = (data.get("state") or {}).get("time") or {}
        if data.get("type") == "tool" and isinstance(span.get("start"), (int, float)) and isinstance(span.get("end"), (int, float)):
            interval = (span["start"] / 1000, span["end"] / 1000)
            tools.setdefault(str(data.get("tool")), []).append(interval)
            times += list(interval)
    return (min(times), max(times), tools) if times else None


_PARSERS: dict[str, Callable[[Path], Parsed]] = {
    "claude": _claude,
    "codex": _codex,
    "opencode": _opencode,
}


def _union(intervals: list[Interval]) -> float:
    covered, reach = 0.0, float("-inf")
    for start, end in sorted(intervals):
        if end > reach:
            covered += end - max(start, reach)
            reach = end
    return covered


def _total(name: str, seconds: float, count: int = 1, longest: float | None = None, **extra: Any) -> dict[str, Any]:
    return {
        "name": name,
        "example": None,
        "count": count,
        "seconds": seconds,
        "max_seconds": seconds if longest is None else longest,
        **extra,
    }


def breakdown(backend: str, transcript: Path | None, agent_seconds: float) -> list[dict[str, Any]]:
    """Totals for one agent call of *agent_seconds* wall time."""
    parser = _PARSERS.get(backend)
    reason = _UNAVAILABLE.get(backend, f"no transcript parser for {backend}")
    parsed: Parsed = None
    if parser is not None:
        reason = "transcript missing or unreadable"
        if transcript is not None and transcript.is_file():
            try:
                parsed = parser(transcript)
            except Exception:  # noqa: BLE001 - a format change must not fail the run
                parsed = None
    if parsed is None:
        return [_total("unavailable", agent_seconds, reason=reason)]

    start, end, tools = parsed
    window = min(max(end - start, 0.0), agent_seconds)
    all_tools = [interval for intervals in tools.values() for interval in intervals]
    totals = [
        _total("model", max(window - _union(all_tools), 0.0)),
        _total("other", agent_seconds - window),
    ]
    for name, intervals in tools.items():
        durations = [max(e - s, 0.0) for s, e in intervals]
        totals.append(_total(f"tool:{name}", sum(durations), len(durations), max(durations)))
    return totals
