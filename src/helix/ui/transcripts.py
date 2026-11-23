"""``GET /api/transcript/{id}``: what a candidate's agent did, live or after the call.

Finished calls are read from the copy HELIX keeps in the worktree
(``<worktree>/<sandbox.transcript_artifact_dir>/<backend>/<session>.jsonl``,
mutator.py:2160-2171).  A call still running is read from the backend's own
session file: under ``$HOME`` when unsandboxed, and from the sandbox auth
volume through one read-only ``docker run`` otherwise, the way HELIX's own
copy helper reads it (sandbox.py:258-326).  Reads are incremental: the cursor
is a byte offset into the session file.  Only codex rollouts and claude JSONL
are parsed; credential files are never listed or opened, and every text is
passed through :func:`helix.ui.redact.redact`.
"""

from __future__ import annotations

import json
import math
import re
import shlex
import subprocess
import threading
import time
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from helix.config import HelixConfig
from helix.sandbox import resolve_sandbox_image, sandbox_auth_volume_name
from helix.ui.candidate import CANDIDATE_ID_RE
from helix.ui.redact import add_env_names, redact

if TYPE_CHECKING:
    from helix.ui.run_state import RunReader

_BACKENDS = ("codex", "claude")
_MAX_READ = 1 << 20  # bytes per read; a first read starts this far before the end
_DOCKER_INTERVAL = 4.0
_DOCKER_TIMEOUT = 10.0
_CLOCK_SLACK = 2.0  # seconds a session file may appear to start before its agent span
_MARKER_WAIT = 20.0  # seconds an unmarked session may still be waiting for its launch line
_LIMITS = {"message": 2000, "command": 1000, "output": 400, "tool": 300, "diff": 2000, "tokens": 200}

# Sandboxed session files, as the auth volume holds them (datamap transcripts
# section 4; config.py:857 for claude's ``-workspace`` folder).  Only these
# two roots are searched, never the rest of the volume, which holds
# credentials (backends.py:331-401 "kept:" lists).
_VOLUME_FIND = {
    "codex": "find /home/node/.codex/sessions -type f -name 'rollout-*.jsonl'",
    "claude": "find /home/node/.claude/projects/-workspace -maxdepth 1 -type f -name '*.jsonl'",
}
#: The one session file of a known session id (``helix.timing.session_recorder``).
_VOLUME_FIND_ID = {
    "codex": "find /home/node/.codex/sessions -type f -name {}",
    "claude": "find /home/node/.claude/projects/-workspace -maxdepth 1 -type f -name {}",
}
_VOLUME_NAME_OF_ID = {"codex": "rollout-*-{}.jsonl", "claude": "{}.jsonl"}
_SESSION_ID_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z-]{7,63}$")
_VOLUME_PATH_RE = {
    "codex": re.compile(r"^/home/node/\.codex/sessions/\d{4}/\d{2}/\d{2}/rollout-[A-Za-z0-9._:-]+\.jsonl$"),
    "claude": re.compile(r"^/home/node/\.claude/projects/-workspace/[A-Za-z0-9._-]+\.jsonl$"),
}
_FILE_MARK = b"\n@@HELIX_UI_FILE@@\t"
_END_MARK = b"\n@@HELIX_UI_END@@"
_TIMESTAMP_RE = re.compile(r'"timestamp"\s*:\s*"([^"]+)"')
_UUID_TAIL_RE = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})$")


class TranscriptReader:
    """Serves one candidate's agent transcript as redacted events, incrementally by byte cursor."""

    def __init__(self, project_root: Path, config: HelixConfig, reader: RunReader) -> None:
        """Remember the project; nothing is read until :meth:`read`."""
        self._root = project_root
        self._config = config
        self._reader = reader
        self._backend = config.agent.backend
        self._lock = threading.Lock()
        # cid -> (agent span_id, session file, session id) once a live session is matched.
        self._sessions: dict[str, tuple[str, str, str, bool]] = {}
        self._last_docker: dict[str, float] = {}
        add_env_names([*config.passthrough_env, *config.env])

    def read(self, cid: str, cursor: int = 0) -> dict[str, Any]:
        """Events after byte *cursor* of *cid*'s transcript: live while its agent runs, else finished."""
        cursor = max(int(cursor), 0)
        if not CANDIDATE_ID_RE.match(cid):
            return self._reply("none", cursor, note="unknown candidate")
        records = self._reader.timing_records()
        # The backend the agent ran with: `helix evolve --backend` can differ from helix.toml.
        backend = _agent_backend(records, cid) or self._backend
        if backend not in _BACKENDS:
            reply = self._reply("none", cursor, note=f"{backend} transcripts are not shown")
        else:
            span = _open_agent_span(records, cid)
            if span is None:
                reply = self._finished(cid, cursor, records, backend)
            elif self._config.sandbox.enabled:
                reply = self._live_sandboxed(cid, cursor, span, records, backend)
            else:
                reply = self._live_host(cid, cursor, span, backend)
        return {**reply, "backend": backend}

    def _reply(
        self,
        source: str,
        cursor: int,
        *,
        session: str | None = None,
        events: list[dict[str, Any]] | None = None,
        note: str | None = None,
    ) -> dict[str, Any]:
        return {
            "source": source,
            "backend": self._backend,
            "session": session,
            "events": events or [],
            "cursor": cursor,
            "note": note,
        }

    def _worktree(self, cid: str) -> Path:
        return self._root / ".helix" / "worktrees" / cid

    # -- finished ---------------------------------------------------------

    def _finished(self, cid: str, cursor: int, records: list[dict[str, Any]], backend: str) -> dict[str, Any]:
        worktree = self._worktree(cid)
        if not worktree.is_dir():
            # A proposal waiting for its slot has no worktree yet; one whose
            # agent ran and lost it was rejected.  A run without timing
            # records cannot tell, and only finished runs lack them.
            started = not records or any(
                rec.get("event") == "start" and rec.get("span") == "agent" and rec.get("candidate") == cid
                for rec in records
            )
            note = "worktree removed, and its transcript with it" if started else "agent not started yet"
            return self._reply("none", cursor, note=note)
        path = _finished_transcript(worktree, self._config.sandbox.transcript_artifact_dir, backend)
        if path is None:
            return self._reply("none", cursor, note="no transcript recorded")
        try:
            data, start, size = _read_local(path, cursor)
        except OSError:
            return self._reply("none", cursor, note="transcript unreadable")
        events, new_cursor = _consume(
            backend, data, start, complete=start + len(data) - int(start > 0) >= size
        )
        return self._reply(
            "finished", new_cursor, session=_session_id(path.name), events=events, note=_tail_note(cursor, start)
        )

    # -- live, unsandboxed ------------------------------------------------

    def _live_host(self, cid: str, cursor: int, span: dict[str, Any], backend: str) -> dict[str, Any]:
        worktree = self._worktree(cid)
        start = float(span["wall_time"])
        if backend == "codex":
            path = _host_codex_session(Path.home(), worktree, start)
        else:
            path = _host_claude_session(Path.home(), worktree, start)
        if path is None:
            return self._reply("live", cursor, note="no session file found yet")
        try:
            data, offset, _ = _read_local(path, cursor)
        except OSError:
            return self._reply("live", cursor, note="session file unreadable")
        events, new_cursor = _consume(backend, data, offset, complete=False)
        return self._reply(
            "live", new_cursor, session=_session_id(path.name), events=events, note=_tail_note(cursor, offset)
        )

    # -- live, sandboxed --------------------------------------------------

    def _live_sandboxed(
        self, cid: str, cursor: int, span: dict[str, Any], records: list[dict[str, Any]], backend: str
    ) -> dict[str, Any]:
        heuristic = "session matched to this agent by start time"
        session_id = _span_session(records, span)
        with self._lock:
            known = self._sessions.get(cid)
            # A start-order guess gives way once the agent's session id is recorded.
            if known is not None and (known[0] != span["span_id"] or (session_id is not None and not known[3])):
                known = None
                self._sessions.pop(cid, None)
            now = time.monotonic()
            if now - self._last_docker.get(cid, -math.inf) < _DOCKER_INTERVAL:
                note = "no session file found yet" if known is None else None if known[3] else heuristic
                return self._reply("live", cursor, session=known[2] if known else None, note=note)
            self._last_docker[cid] = now
        try:
            image = resolve_sandbox_image(self._config.sandbox, backend)
            volume = sandbox_auth_volume_name(backend, self._config.sandbox)
        except ValueError as exc:
            return self._reply("none", cursor, note=str(exc))
        if known is not None:
            listing = f"printf '%s\\n' {shlex.quote(known[1])}"
        elif session_id is not None:
            name = _VOLUME_NAME_OF_ID[backend].format(session_id)
            listing = f"{_VOLUME_FIND_ID[backend].format(shlex.quote(name))} 2>/dev/null"
        else:
            minutes = math.ceil(max(time.time() - float(span["wall_time"]), 0.0) / 60) + 1
            listing = f"{_VOLUME_FIND[backend]} -mmin -{minutes} 2>/dev/null"
        try:
            result = subprocess.run(
                docker_argv(image, volume, _volume_script(listing, cursor)),
                capture_output=True,
                timeout=_DOCKER_TIMEOUT,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return self._reply("live", cursor, note=f"could not read the auth volume: {type(exc).__name__}")
        if result.returncode != 0:
            return self._reply("live", cursor, note=f"could not read the auth volume: docker exit {result.returncode}")
        files = {
            path: (offset, stamp, marker, content)
            for path, offset, stamp, marker, content in _split_volume_output(result.stdout)
            if _VOLUME_PATH_RE[backend].match(path)
        }
        path: str | None
        if known is not None:
            path, exact = known[1], known[3]
        elif session_id is not None:
            path = next((p for p in files if _session_id(Path(p).name) == session_id), None)
            exact = True
        elif _records_sessions(records):
            # This run records session ids; this agent's is not announced yet.
            path, exact = None, False
        else:
            path, exact = _match_session(span, records, files, cid, time.time())
        if path is None or path not in files:
            with self._lock:
                self._sessions.pop(cid, None)
            return self._reply("live", cursor, note="no session file found yet")
        session = _session_id(Path(path).name)
        with self._lock:
            self._sessions[cid] = (str(span["span_id"]), path, session, exact)
        offset, _, _, content = files[path]
        events, new_cursor = _consume(backend, content, offset, complete=False)
        note = _tail_note(cursor, offset) or (None if exact else heuristic)
        return self._reply("live", new_cursor, session=session, events=events, note=note)


# -- locating files ----------------------------------------------------------


def _latest_run(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Records of the latest ``run_id``: the one whose ``run`` span started last."""
    run_id = None
    for rec in records:
        if rec.get("event") == "start" and rec.get("span") == "run":
            run_id = rec.get("run_id")
    return [rec for rec in records if rec.get("run_id") == run_id] if run_id else []


def _open_agent_span(records: list[dict[str, Any]], cid: str) -> dict[str, Any] | None:
    """The latest ``agent`` span of *cid* that has a ``start`` but no ``end`` in the latest run."""
    run = _latest_run(records)
    ended = {rec.get("span_id") for rec in run if rec.get("event") == "end"}
    spans = [
        rec
        for rec in run
        if rec.get("event") == "start"
        and rec.get("span") == "agent"
        and rec.get("candidate") == cid
        and rec.get("span_id") not in ended
        and isinstance(rec.get("wall_time"), (int, float))
    ]
    return max(spans, key=lambda rec: rec["wall_time"]) if spans else None


def _agent_backend(records: list[dict[str, Any]], cid: str) -> str | None:
    """The backend of *cid*'s latest agent call, else of the latest agent call recorded at all."""
    own: str | None = None
    latest: str | None = None
    for rec in records:
        if rec.get("event") != "start" or rec.get("span") != "agent":
            continue
        backend = (rec.get("attrs") or {}).get("backend")
        if not isinstance(backend, str):
            continue
        if rec.get("candidate") == cid:
            own = backend
        latest = backend
    return own or latest


def _span_session(records: list[dict[str, Any]], span: dict[str, Any]) -> str | None:
    """The session id HELIX recorded for the agent call *span* (``helix.timing.session_recorder``)."""
    for rec in records:
        if rec.get("event") == "session" and rec.get("parent_span_id") == span.get("span_id"):
            sid = rec.get("session_id")
            if isinstance(sid, str) and _SESSION_ID_RE.match(sid):
                return sid
    return None


def _records_sessions(records: list[dict[str, Any]]) -> bool:
    """Whether the latest run records each agent call's session id."""
    return any(rec.get("event") == "session" for rec in _latest_run(records))


def _match_session(
    span: dict[str, Any],
    records: list[dict[str, Any]],
    files: dict[str, tuple[int, str, str, bytes]],
    cid: str,
    now: float,
) -> tuple[str | None, bool]:
    """The session file of the agent call *span* of *cid* in a run that records no session ids.

    Returns the file and whether the match is exact.  Runs since 0.4.0 record
    each agent call's session id, and are matched by it instead.

    Runs between 2026-11-16 and 0.4.0 launched each agent with a line naming
    its candidate ("This is HELIX candidate g3-s12"), the first user message
    of its session file; later runs no longer tell the agent its candidate.
    The earliest session carrying *cid*'s marker that began at or after the
    span's start is exact.

    Sessions from runs before that line carry no marker.  Parallel agents of
    one backend share one auth volume and every session records ``cwd``
    ``/workspace``, so for those, each ``agent`` span since the latest
    ``sandbox_reset`` (which empties the volume's sessions every generation,
    sandbox.py:363) without a marked session, taken in start order, gets the
    earliest unmarked session whose first timestamp is at or after its start.
    Codex writes ``session_meta`` a few seconds before the launch line, so a
    session counts as unmarked only once it is ``_MARKER_WAIT`` seconds old; a
    match, once made, is kept for the rest of the agent call.
    """
    stamped = [(at, path, marker) for path, (_, stamp, marker, _) in files.items() if (at := _epoch(stamp)) is not None]
    start = float(span["wall_time"]) - _CLOCK_SLACK
    mine = sorted((at, path) for at, path, marker in stamped if marker == cid and at >= start)
    if mine:
        return mine[0][1], True
    marked = {marker for _, _, marker in stamped if marker}
    if marked or any(at > now - _MARKER_WAIT for at, _, _ in stamped):
        # A session without a marker yet may be one still starting up, not an old-style one.
        return None, False
    run = _latest_run(records)
    resets = [rec["wall_time"] for rec in run if rec.get("span") == "sandbox_reset" and rec.get("event") == "start"]
    since = max(resets, default=-math.inf)
    spans = sorted(
        (rec for rec in run if rec.get("event") == "start" and rec.get("span") == "agent"
         and rec["wall_time"] >= since and rec.get("candidate") not in marked),
        key=lambda rec: rec["wall_time"],
    )
    sessions = sorted((at, path) for at, path, marker in stamped if not marker)
    claimed: set[str] = set()
    for rec in spans:
        pick = next(
            (path for at, path in sessions if path not in claimed and at >= rec["wall_time"] - _CLOCK_SLACK), None
        )
        if pick is None:
            continue
        claimed.add(pick)
        if rec.get("span_id") == span.get("span_id"):
            return pick, False
    return None, False


def _host_codex_session(home: Path, worktree: Path, start: float) -> Path | None:
    """Newest codex rollout under *home* begun since *start* whose ``session_meta`` cwd is *worktree*."""
    cwds = {str(worktree), str(worktree.resolve())}
    recent = []
    for path in (home / ".codex" / "sessions").glob("*/*/*/rollout-*.jsonl"):
        try:
            if not path.is_symlink() and path.stat().st_mtime >= start - _CLOCK_SLACK:
                recent.append(path)
        except OSError:
            continue
    for path in sorted(recent, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with path.open("rb") as f:
                meta = json.loads(f.readline(_MAX_READ))
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or meta.get("type") != "session_meta":
            continue
        payload = meta.get("payload")
        if isinstance(payload, dict) and payload.get("cwd") in cwds:
            return path
    return None


def _host_claude_session(home: Path, worktree: Path, start: float) -> Path | None:
    """Newest session in the worktree's claude project folder, modified since *start*.

    Claude names the folder after the cwd with every non-alphanumeric
    character turned into ``-`` (``/a/.helix/w`` -> ``-a--helix-w``); HELIX's
    comment at mutator.py:1808-1821 describes the ``/`` case.
    """
    best: tuple[float, Path] | None = None
    for cwd in {str(worktree), str(worktree.resolve())}:
        folder = home / ".claude" / "projects" / re.sub(r"[^A-Za-z0-9]", "-", cwd)
        for path in folder.glob("*.jsonl"):
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            if not path.is_symlink() and mtime >= start - _CLOCK_SLACK and (best is None or mtime > best[0]):
                best = (mtime, path)
    return best[1] if best else None


def _finished_transcript(worktree: Path, artifact_dir: str, backend: str) -> Path | None:
    """The mutation call's transcript copy, not the change-summary call's.

    ``.helix_backend_result.json`` names the call's own transcript; without it,
    the oldest copy not named by ``.helix_backend_summary_result.json`` (the
    summary call runs after the mutation, mutator.py:659-670).
    """
    folder = worktree / artifact_dir / backend
    try:
        root = folder.resolve(strict=True)
    except OSError:
        return None

    def inside(rel: str) -> Path | None:
        path = worktree / rel
        try:
            ok = path.is_file() and not path.is_symlink() and path.resolve().parent == root
        except OSError:
            ok = False
        return path if ok else None

    named = _artifacts(worktree / ".helix_backend_result.json")
    for rel in named:
        if (path := inside(rel)) is not None:
            return path
    summary = {Path(rel).name for rel in _artifacts(worktree / ".helix_backend_summary_result.json")}
    copies = [
        path
        for path in folder.glob("*.jsonl")
        if path.name not in summary and "summary" not in path.name and inside(str(path.relative_to(worktree)))
    ]
    return min(copies, key=lambda p: p.stat().st_mtime) if copies else None


def _artifacts(result_file: Path) -> list[str]:
    """Worktree-relative transcript paths an HELIX backend-result file lists."""
    if not result_file.is_file() or result_file.is_symlink():
        return []
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    entries = data.get("transcript_artifacts") if isinstance(data, dict) else None
    return [
        str(e["path"])
        for e in entries or []
        if isinstance(e, dict) and e.get("available") and isinstance(e.get("path"), str)
    ]


def _session_id(name: str) -> str:
    stem = name.removesuffix(".jsonl")
    match = _UUID_TAIL_RE.search(stem)
    return match.group(1) if match else stem


# -- reading bytes -----------------------------------------------------------


def _read_local(path: Path, cursor: int) -> tuple[bytes, int, int]:
    """A bounded chunk at the cursor or tail, with one lookbehind byte when available."""
    size = path.stat().st_size
    start = cursor if 0 < cursor <= size else max(size - _MAX_READ, 0)
    lookbehind = int(start > 0)
    with path.open("rb") as f:
        f.seek(start - lookbehind)
        return f.read(_MAX_READ + lookbehind), start, size


def _tail_note(cursor: int, start: int) -> str | None:
    return f"showing the last {_MAX_READ >> 20} MiB" if cursor == 0 and start > 0 else None


def docker_argv(image: str, volume: str, script: str) -> list[str]:
    """The read-only, network-less container that reads session files from the auth volume."""
    return [
        "docker", "run", "--rm", "--pull", "never",
        "--network", "none",
        "--user", "node",
        "--security-opt", "no-new-privileges",
        "-v", f"{volume}:/home/node:ro",
        image, "sh", "-c", script,
    ]  # fmt: skip


def _volume_script(listing: str, cursor: int) -> str:
    """Shell that, for each file *listing* prints, emits a header, then its bytes from the cursor.

    Include the byte before a nonzero start, as in :func:`_read_local`.
    The header carries the start offset, the path, the first ``timestamp``
    in the first five lines (codex ``session_meta``; claude's first timed entry,
    since current Claude Code opens with an untimed ``session-owner`` line), and
    the candidate id the launch line names in the first 40 lines, if any (runs
    between 2026-11-16 and 0.4.0).
    Symlinks are skipped, so nothing outside the session folders is ever opened.
    """
    return (
        f"{listing} | while IFS= read -r f; do "
        '[ -f "$f" ] && [ ! -L "$f" ] || continue; '
        's=$(( $(wc -c < "$f") )); '
        f"if [ {cursor} -gt 0 ] && [ {cursor} -le \"$s\" ]; then st={cursor}; "
        f"elif [ \"$s\" -gt {_MAX_READ} ]; then st=$((s - {_MAX_READ})); else st=0; fi; "
        "ts=$(head -n 5 \"$f\" | head -c 1048576 | grep -o '\"timestamp\": *\"[^\"]*\"' | head -n 1); "
        "mk=$(head -n 40 \"$f\" | head -c 4194304 | grep -o 'HELIX candidate g[0-9][0-9]*-[sm][0-9][0-9]*' | head -n 1); "
        "printf '\\n@@HELIX_UI_FILE@@\\t%s\\t%s\\t%s\\t%s\\n' \"$st\" \"$f\" \"$ts\" \"${mk##* }\"; "
        'lookbehind=$((st > 0)); '
        f'tail -c +$((st + 1 - lookbehind)) "$f" | head -c $(({_MAX_READ} + lookbehind)); '
        "printf '\\n@@HELIX_UI_END@@\\n'; "
        "done"
    )


#: The candidate id in the launch line of runs between 2026-11-16 and 0.4.0.
_MARKER_RE = re.compile(r"^g\d+-[sm]\d+$")


def _split_volume_output(out: bytes) -> Iterable[tuple[str, int, str, str, bytes]]:
    """``(path, start offset, first timestamp, candidate marker, bytes)`` per file block of :func:`_volume_script`."""
    for block in out.split(_FILE_MARK)[1:]:
        header, _, rest = block.partition(b"\n")
        fields = header.decode("utf-8", "replace").split("\t")
        if len(fields) != 4 or not fields[0].isdigit() or _END_MARK not in rest:
            continue
        match = _TIMESTAMP_RE.search(fields[2])
        marker = fields[3] if _MARKER_RE.match(fields[3]) else ""
        yield fields[1], int(fields[0]), match.group(1) if match else "", marker, rest.rpartition(_END_MARK)[0]


def _consume(backend: str, data: bytes, start: int, *, complete: bool) -> tuple[list[dict[str, Any]], int]:
    """Consume whole JSONL records, advancing over fragments of oversized records."""
    if start:
        preceding, data = data[:1], data[1:]
        if preceding != b"\n":
            # The cursor is inside a record after a tail read or a bounded skip.
            end = data.find(b"\n") + 1
            if not end:
                return [], start + len(data)
            start, data = start + end, data[end:]
    end = len(data) if complete else data.rfind(b"\n") + 1
    if not end and len(data) == _MAX_READ:
        # This record cannot fit in one read. The next chunk discards its suffix.
        return [], start + len(data)
    rows = []
    for line in data[:end].splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue  # a line cut by a tail read, or torn by a writer
        if isinstance(row, dict):
            rows.append(row)
    return parse_events(backend, rows), start + end


# -- parsing -----------------------------------------------------------------


def parse_events(backend: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Transcript rows of *backend* (``codex`` rollout or ``claude`` JSONL) as redacted events."""
    return _codex(rows) if backend == "codex" else _claude(rows)


def _epoch(value: Any) -> float | None:
    """Epoch seconds of an ISO timestamp, as ``helix.agent_timing._iso`` reads it."""
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def _event(t: float | None, kind: str, text: str, exit_code: int | None = None) -> dict[str, Any]:
    limit = _LIMITS[kind]
    # Redact before cutting, so a cut cannot split a secret past its pattern.
    clipped = redact(text[: limit * 2 + 200])
    if len(clipped) > limit:
        clipped = clipped[: limit - 1] + "…"
    return {"t": t, "kind": kind, "text": clipped, "exit_code": exit_code}


def _shell_text(command: Any) -> str:
    """``["/bin/bash", "-lc", "ls"]`` -> ``ls``; other lists are shell-joined."""
    if isinstance(command, str):
        return command
    if isinstance(command, list) and all(isinstance(c, str) for c in command):
        if len(command) == 3 and command[0].endswith("sh") and command[1] in ("-c", "-lc"):
            return str(command[2])
        return shlex.join(command)
    return ""


_EXEC_CMD_RE = re.compile(r'exec_command\(\s*\{\s*"?cmd"?\s*:\s*("(?:[^"\\]|\\.)*")')
_JS_TOOL_RE = re.compile(r"tools\.(\w+)\(")


def _exec_commands(source: str) -> list[str]:
    """The ``cmd`` strings of ``tools.exec_command({cmd: "..."})`` calls in a codex ``exec`` script."""
    found = []
    for literal in _EXEC_CMD_RE.findall(source):
        try:
            found.append(str(json.loads(literal)))
        except ValueError:
            found.append(literal[1:-1])
    return found


def _codex_output(output: Any) -> tuple[str, int | None]:
    """Text and exit code of a codex ``*_output``: the ``output`` of its JSON chunks, else its plain text."""
    items = [output] if isinstance(output, str) else [o.get("text") for o in output or [] if isinstance(o, dict)]
    chunks: list[str] | None = None  # None until a JSON chunk is seen
    plain: list[str] = []
    code: int | None = None
    for text in items:
        if not isinstance(text, str):
            continue
        try:
            data = json.loads(text)
        except ValueError:
            data = None
        if isinstance(data, dict) and "output" in data:
            chunks = chunks or []
            if data["output"]:
                chunks.append(str(data["output"]))
            meta = data.get("metadata")
            exit_code = (meta if isinstance(meta, dict) else data).get("exit_code")
            if isinstance(exit_code, int):
                code = exit_code
        else:
            plain.append(text)
    return "\n".join(plain if chunks is None else chunks), code


def _codex(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Codex rollout events; mirrors ``helix.agent_timing._codex`` on which rows are tool calls.

    Reasoning, developer and user messages and ``session_meta`` are skipped,
    so ``base_instructions``, ``encrypted_content``, ``creator_*`` ids and
    ``rate_limits`` never leave this function.
    """
    events: list[dict[str, Any]] = []
    for row in rows:
        payload = row.get("payload")
        if not isinstance(payload, dict):
            continue
        t = _epoch(row.get("timestamp"))
        kind = payload.get("type")
        if row.get("type") == "response_item":
            if kind == "message" and payload.get("role") == "assistant":
                content = payload.get("content")
                text = "".join(
                    str(c.get("text", "")) for c in content or [] if isinstance(c, dict) and "text" in c
                )
                if text.strip():
                    events.append(_event(t, "message", text))
            elif kind in ("custom_tool_call", "function_call"):
                events += _codex_call(t, payload)
            elif kind in ("custom_tool_call_output", "function_call_output"):
                text, code = _codex_output(payload.get("output"))
                events.append(_event(t, "output", text, code))
        elif row.get("type") == "event_msg" and kind == "item_completed":
            item = payload.get("item")
            if isinstance(item, dict) and item.get("type") == "CommandExecution":
                code = item.get("exit_code") if isinstance(item.get("exit_code"), int) else None
                command = _event(t, "command", _shell_text(item.get("command")), code)
                same = next(
                    (
                        e
                        for e in reversed(events)
                        if e["kind"] == "command" and e["exit_code"] is None and e["text"] == command["text"]
                    ),
                    None,
                )
                if same is not None:
                    same["exit_code"] = code
                else:
                    events.append(command)
        elif row.get("type") == "event_msg" and kind == "token_count":
            info = payload.get("info")
            usage = info.get("total_token_usage") if isinstance(info, dict) else None
            if isinstance(usage, dict):
                text = (
                    f"{int(usage.get('input_tokens') or 0):,} in · "
                    f"{int(usage.get('cached_input_tokens') or 0):,} cached · "
                    f"{int(usage.get('output_tokens') or 0):,} out"
                )
                events.append(_event(t, "tokens", text))
    return events


def _codex_call(t: float | None, payload: dict[str, Any]) -> list[dict[str, Any]]:
    name = str(payload.get("name") or "")
    if payload.get("type") == "custom_tool_call":
        source = str(payload.get("input") or "")
        if name == "apply_patch":
            return [_event(t, "diff", source)]
        commands = _exec_commands(source)
        if commands:
            return [_event(t, "command", cmd) for cmd in commands]
        tools = list(dict.fromkeys(_JS_TOOL_RE.findall(source)))
        return [_event(t, "tool", ", ".join(tools) or name)]
    raw = payload.get("arguments")
    try:
        args = json.loads(raw) if isinstance(raw, str) else raw
    except ValueError:
        args = None
    args = args if isinstance(args, dict) else {}
    if name == "apply_patch":
        return [_event(t, "diff", str(args.get("input") or raw or ""))]
    command = _shell_text(args.get("cmd") or args.get("command"))
    if command:
        return [_event(t, "command", command)]
    return [_event(t, "tool", f"{name} {json.dumps(args, separators=(',', ':'))}")]


_CLAUDE_EDITS = ("Edit", "MultiEdit", "Write", "NotebookEdit")
_CLAUDE_SUMMARY_KEYS = ("file_path", "path", "pattern", "url", "query", "description", "prompt")


def _claude(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Claude JSONL events; sidechain rows are skipped as ``helix.agent_timing._claude`` does."""
    events: list[dict[str, Any]] = []
    for row in rows:
        if row.get("isSidechain") or row.get("type") not in ("assistant", "user"):
            continue
        t = _epoch(row.get("timestamp"))
        message = row.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue  # a plain-string user turn is the prompt
        for item in content:
            if not isinstance(item, dict):
                continue
            kind = item.get("type")
            if row["type"] == "assistant" and kind == "text" and str(item.get("text", "")).strip():
                events.append(_event(t, "message", str(item["text"])))
            elif row["type"] == "assistant" and kind == "tool_use":
                events.append(_claude_tool(t, str(item.get("name")), item.get("input")))
            elif row["type"] == "user" and kind == "tool_result":
                result = item.get("content")
                if isinstance(result, list):
                    result = "\n".join(
                        str(r.get("text", "")) for r in result if isinstance(r, dict) and r.get("type") == "text"
                    )
                events.append(_event(t, "output", str(result or "")))
    return events


def _claude_tool(t: float | None, name: str, args: Any) -> dict[str, Any]:
    args = args if isinstance(args, dict) else {}
    if name == "Bash":
        return _event(t, "command", str(args.get("command", "")))
    if name in _CLAUDE_EDITS:
        return _event(t, "diff", f"{name} {args.get('file_path') or args.get('notebook_path') or ''}".strip())
    summary = next((str(args[k]) for k in _CLAUDE_SUMMARY_KEYS if args.get(k)), None)
    return _event(t, "tool", f"{name} {summary or json.dumps(args, separators=(',', ':'), default=str)}")
