"""`GET /api/transcript/{id}`: finished and live agent transcripts, parsed and redacted."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any

import pytest

from helix.backends import DEFAULT_BACKEND_IMAGES
from helix.config import AgentConfig, EvaluatorConfig, HelixConfig, SandboxConfig
from helix.ui import transcripts as tmod
from helix.ui.redact import MASK
from helix.ui.transcripts import TranscriptReader, parse_events

T0 = 1_794_787_200.0  # 2026-11-15T00:00:00Z


def _iso(t: float) -> str:
    from datetime import UTC, datetime

    return datetime.fromtimestamp(t, UTC).isoformat().replace("+00:00", "Z")


def _codex_rows(t: float = T0) -> list[dict[str, Any]]:
    """A short rollout with the same row shapes codex 0.162 writes."""
    return [
        {"timestamp": _iso(t), "ordinal": 0, "type": "session_meta",
         "payload": {"id": "sess", "cwd": "/workspace", "creator_user_id": "user-SECRETID",
                     "creator_account_id": "acct-SECRETID", "base_instructions": {"text": "BASE PROMPT"}}},
        {"timestamp": _iso(t + 1), "ordinal": 1, "type": "response_item",
         "payload": {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "DEV"}]}},
        {"timestamp": _iso(t + 1), "ordinal": 2, "type": "response_item",
         "payload": {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "TASK PROMPT"}]}},
        {"timestamp": _iso(t + 2), "ordinal": 3, "type": "response_item",
         "payload": {"type": "reasoning", "summary": [], "encrypted_content": "gAAAAENCRYPTED"}},
        {"timestamp": _iso(t + 3), "ordinal": 4, "type": "response_item",
         "payload": {"type": "message", "role": "assistant", "phase": "commentary",
                     "content": [{"type": "output_text", "text": "Checking the grasp script."}]}},
        {"timestamp": _iso(t + 4), "ordinal": 5, "type": "response_item",
         "payload": {"type": "custom_tool_call", "call_id": "c1", "name": "exec", "status": "completed",
                     "input": '// @exec: {"yield_time_ms": 1000}\n'
                              'text(await tools.exec_command({cmd:"ls -la \\"policy dir\\"","max_output_tokens":2000}));\n'}},
        {"timestamp": _iso(t + 5), "ordinal": 6, "type": "event_msg",
         "payload": {"type": "item_completed", "item": {
             "type": "CommandExecution", "command": ["/bin/bash", "-lc", 'ls -la "policy dir"'],
             "aggregated_output": "total 0\n", "exit_code": 2}}},
        {"timestamp": _iso(t + 5), "ordinal": 7, "type": "response_item",
         "payload": {"type": "custom_tool_call_output", "call_id": "c1", "output": [
             {"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"},
             {"type": "input_text", "text": json.dumps(
                 {"chunk_id": "a", "exit_code": 2, "output": "OPENAI_API_KEY=hunter2 " + "y" * 600})},
         ]}},
        {"timestamp": _iso(t + 6), "ordinal": 8, "type": "response_item",
         "payload": {"type": "custom_tool_call", "call_id": "c2", "name": "exec",
                     "input": "text(await tools.write_stdin({session_id:5,chars:\"\"}));"}},
        {"timestamp": _iso(t + 7), "ordinal": 9, "type": "response_item",
         "payload": {"type": "function_call", "call_id": "c3", "name": "shell",
                     "arguments": json.dumps({"command": ["bash", "-lc", "pytest -q"]})}},
        {"timestamp": _iso(t + 8), "ordinal": 10, "type": "response_item",
         "payload": {"type": "function_call_output", "call_id": "c3",
                     "output": json.dumps({"output": "1 passed", "metadata": {"exit_code": 0}})}},
        {"timestamp": _iso(t + 9), "ordinal": 11, "type": "response_item",
         "payload": {"type": "custom_tool_call", "call_id": "c4", "name": "apply_patch",
                     "input": "*** Begin Patch\n*** Update File: policy.py\n-x = 1\n+x = 2\n*** End Patch"}},
        {"timestamp": _iso(t + 10), "ordinal": 12, "type": "event_msg",
         "payload": {"type": "token_count",
                     "info": {"total_token_usage": {"input_tokens": 6113091, "cached_input_tokens": 5977344,
                                                    "output_tokens": 25435}},
                     "rate_limits": {"plan_type": "pro", "credits": {"balance": "SECRETBALANCE"}}}},
        {"timestamp": _iso(t + 11), "ordinal": 13, "type": "event_msg",
         "payload": {"type": "item_completed", "item": {"type": "Reasoning", "summary_text": ["thinking"]}}},
    ]


def _claude_rows(t: float = T0) -> list[dict[str, Any]]:
    return [
        {"type": "user", "isSidechain": False, "timestamp": _iso(t),
         "message": {"role": "user", "content": "TASK PROMPT"}},
        {"type": "assistant", "isSidechain": False, "timestamp": _iso(t + 1), "message": {"role": "assistant",
         "content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "Reading the policy."},
                     {"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "cat policy.py"}}]}},
        {"type": "user", "isSidechain": False, "timestamp": _iso(t + 2), "message": {"role": "user",
         "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "x = 1 Bearer abc.def"}]}},
        {"type": "assistant", "isSidechain": False, "timestamp": _iso(t + 3), "message": {"role": "assistant",
         "content": [{"type": "tool_use", "id": "t2", "name": "Edit",
                      "input": {"file_path": "/workspace/policy.py", "old_string": "1", "new_string": "2"}},
                     {"type": "tool_use", "id": "t3", "name": "Grep", "input": {"pattern": "grasp"}}]}},
        {"type": "user", "isSidechain": False, "timestamp": _iso(t + 4), "message": {"role": "user",
         "content": [{"type": "tool_result", "tool_use_id": "t2",
                      "content": [{"type": "text", "text": "edited"}]}]}},
        {"type": "assistant", "isSidechain": True, "timestamp": _iso(t + 5), "message": {"role": "assistant",
         "content": [{"type": "text", "text": "SIDECHAIN"}]}},
    ]


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(json.dumps(r).encode() + b"\n" for r in rows)


def test_codex_rollout_events() -> None:
    events = parse_events("codex", _codex_rows())
    assert [(e["kind"], e["exit_code"]) for e in events] == [
        ("message", None), ("command", 2), ("output", 2), ("tool", None),
        ("command", None), ("output", 0), ("diff", None), ("tokens", None),
    ]
    assert events[0]["text"] == "Checking the grasp script."
    assert events[1]["text"] == 'ls -la "policy dir"'
    assert events[1]["t"] == pytest.approx(T0 + 4)
    assert events[2]["text"].startswith(f"OPENAI_API_KEY={MASK} ") and len(events[2]["text"]) == 400
    assert events[3]["text"] == "write_stdin"
    assert events[4]["text"] == "pytest -q" and events[5]["text"] == "1 passed"
    assert "+x = 2" in events[6]["text"]
    assert events[7]["text"] == "6,113,091 in · 5,977,344 cached · 25,435 out"
    dumped = json.dumps(events)
    for private in ("SECRETID", "BASE PROMPT", "DEV", "TASK PROMPT", "ENCRYPTED", "SECRETBALANCE", "hunter2",
                    "thinking"):
        assert private not in dumped


def test_claude_jsonl_events() -> None:
    events = parse_events("claude", _claude_rows())
    assert [(e["kind"], e["text"]) for e in events] == [
        ("message", "Reading the policy."),
        ("command", "cat policy.py"),
        ("output", f"x = 1 Bearer {MASK}"),
        ("diff", "Edit /workspace/policy.py"),
        ("tool", "Grep grasp"),
        ("output", "edited"),
    ]


# -- the reader ----------------------------------------------------------------


class FakeReader:
    def __init__(self, records: list[dict[str, Any]] | None = None) -> None:
        self.records = records or []

    def snapshot(self, now: float | None = None) -> dict[str, Any] | None:
        return None

    def timing_records(self) -> list[dict[str, Any]]:
        return self.records

    def candidate_ids(self) -> set[str]:
        return set()


def _config(backend: str = "codex", sandbox: bool = False) -> HelixConfig:
    return HelixConfig(
        objective="grasp",
        evaluator=EvaluatorConfig(command="python eval.py"),
        agent=AgentConfig(backend=backend),  # type: ignore[arg-type]
        sandbox=SandboxConfig(enabled=sandbox),
    )


def _start(span: str, span_id: str, t: float, cid: str | None = None) -> dict[str, Any]:
    return {"event": "start", "span": span, "span_id": span_id, "run_id": "r1", "candidate": cid,
            "wall_time": t, "attrs": {}}


def _end(rec: dict[str, Any]) -> dict[str, Any]:
    return {**rec, "event": "end", "duration_seconds": 1.0}


def _worktree(root: Path, cid: str) -> Path:
    wt = root / ".helix" / "worktrees" / cid
    wt.mkdir(parents=True)
    return wt


def test_finished_transcript_from_worktree(tmp_path: Path) -> None:
    wt = _worktree(tmp_path, "g1-s1")
    folder = wt / ".helix_artifacts" / "backend_transcripts" / "codex"
    folder.mkdir(parents=True)
    data = _jsonl(_codex_rows())
    (folder / "sess-main.jsonl").write_bytes(data)
    (folder / "sess-sum.jsonl").write_bytes(_jsonl(_codex_rows()[:5]))
    (wt / ".helix_backend_result.json").write_text(json.dumps({"transcript_artifacts": [
        {"backend": "codex", "session_id": "sess-main", "available": True,
         "path": ".helix_artifacts/backend_transcripts/codex/sess-main.jsonl"}]}))
    reader = TranscriptReader(tmp_path, _config(), FakeReader())

    out = reader.read("g1-s1")
    assert out["source"] == "finished" and out["session"] == "sess-main" and out["backend"] == "codex"
    assert out["cursor"] == len(data) and len(out["events"]) == 8 and out["note"] is None
    again = reader.read("g1-s1", out["cursor"])
    assert again["events"] == [] and again["cursor"] == len(data)


def test_finished_without_result_file_skips_the_summary_call(tmp_path: Path) -> None:
    wt = _worktree(tmp_path, "g1-s1")
    folder = wt / ".helix_artifacts" / "backend_transcripts" / "claude"
    folder.mkdir(parents=True)
    (folder / "first.jsonl").write_bytes(_jsonl(_claude_rows()))
    (folder / "second.jsonl").write_bytes(b"")
    (wt / ".helix_backend_summary_result.json").write_text(json.dumps({"transcript_artifacts": [
        {"available": True, "path": ".helix_artifacts/backend_transcripts/claude/first.jsonl"}]}))
    out = TranscriptReader(tmp_path, _config("claude"), FakeReader()).read("g1-s1")
    assert out["session"] == "second" and out["events"] == []


def test_symlinked_transcript_is_not_followed(tmp_path: Path) -> None:
    wt = _worktree(tmp_path, "g1-s1")
    folder = wt / ".helix_artifacts" / "backend_transcripts" / "codex"
    folder.mkdir(parents=True)
    secret = tmp_path / "auth.json"
    secret.write_text('{"timestamp": "x", "type": "response_item"}\n')
    (folder / "s.jsonl").symlink_to(secret)
    out = TranscriptReader(tmp_path, _config(), FakeReader()).read("g1-s1")
    assert out["source"] == "none" and out["note"] == "no transcript recorded"


def test_missing_worktree_and_bad_ids(tmp_path: Path) -> None:
    reader = TranscriptReader(tmp_path, _config(), FakeReader())
    assert reader.read("g1-s1")["note"] == "worktree removed, and its transcript with it"
    for cid in ("../x", "g1-s1/..", "x"):
        assert reader.read(cid) == {"source": "none", "backend": "codex", "session": None, "events": [],
                                    "cursor": 0, "note": "unknown candidate"}
    other = TranscriptReader(tmp_path, _config("cursor"), FakeReader())
    assert other.read("g1-s1")["note"] == "cursor transcripts are not shown"


def test_a_proposal_whose_agent_has_not_started_is_not_called_removed(tmp_path: Path) -> None:
    # g1-s2 waits for a slot while g1-s1's agent ran and its worktree is gone.
    started = _start("agent", "a1", 100.0, "g1-s1")
    records = [started, _end(started)]
    reader = TranscriptReader(tmp_path, _config(), FakeReader(records))
    assert reader.read("g1-s2")["note"] == "agent not started yet"
    assert reader.read("g1-s1")["note"] == "worktree removed, and its transcript with it"


def test_first_read_of_a_large_file_starts_near_the_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tmod, "_MAX_READ", 300)
    wt = _worktree(tmp_path, "g1-s1")
    folder = wt / ".helix_artifacts" / "backend_transcripts" / "claude"
    folder.mkdir(parents=True)
    data = _jsonl(_claude_rows())
    (folder / "s.jsonl").write_bytes(data)
    out = TranscriptReader(tmp_path, _config("claude"), FakeReader()).read("g1-s1")
    assert out["cursor"] == len(data) and out["note"] == "showing the last 0 MiB"
    assert len(out["events"]) < 6


def test_live_unsandboxed_codex_matches_worktree_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "proj"
    wt = _worktree(root, "g1-s1")
    day = home / ".codex" / "sessions" / "2026" / "11" / "15"
    day.mkdir(parents=True)
    (home / ".codex" / "auth.json").write_text("CREDENTIAL")
    mine = _codex_rows()
    mine[0]["payload"]["cwd"] = str(wt)
    (day / "rollout-2026-11-15T00-00-00-11111111-2222-3333-4444-555555555555.jsonl").write_bytes(_jsonl(mine))
    (day / "rollout-2026-11-15T00-00-01-other.jsonl").write_bytes(_jsonl(_codex_rows()))
    (day / "rollout-link.jsonl").symlink_to(home / ".codex" / "auth.json")
    span = _start("agent", "a1", 0.0, "g1-s1")
    records = [_start("run", "run", 0.0), span]
    reader = TranscriptReader(root, _config(), FakeReader(records))

    out = reader.read("g1-s1")
    assert out["source"] == "live" and out["session"] == "11111111-2222-3333-4444-555555555555"
    assert [e["kind"] for e in out["events"]][:2] == ["message", "command"]
    assert "CREDENTIAL" not in json.dumps(out)

    reader._reader = FakeReader([*records, _end(span)])  # type: ignore[assignment]
    assert reader.read("g1-s1")["source"] == "none"  # finished, but no copy in the worktree


def test_live_unsandboxed_claude_project_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    root = tmp_path / "proj"
    wt = _worktree(root, "g1-s1")
    folder = home / ".claude" / "projects" / "".join(c if c.isalnum() else "-" for c in str(wt))
    folder.mkdir(parents=True)
    (folder / "abc.jsonl").write_bytes(_jsonl(_claude_rows()))
    records = [_start("run", "run", 0.0), _start("agent", "a1", 0.0, "g1-s1")]
    out = TranscriptReader(root, _config("claude"), FakeReader(records)).read("g1-s1")
    assert out["source"] == "live" and out["session"] == "abc" and len(out["events"]) == 6


def _volume_block(path: str, offset: int, first_ts: str, content: bytes, marker: str = "") -> bytes:
    header = f"\n@@HELIX_UI_FILE@@\t{offset}\t{path}\t\"timestamp\":\"{first_ts}\"\t{marker}\n".encode()
    return header + content + b"\n@@HELIX_UI_END@@\n"


def test_live_sandboxed_reads_the_auth_volume_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _worktree(tmp_path, "g1-s2")
    sessions = "/home/node/.codex/sessions/2026/11/15/"
    first = sessions + "rollout-2026-11-15T00-00-05-aaaaaaaa-0000-0000-0000-000000000001.jsonl"
    second = sessions + "rollout-2026-11-15T00-00-07-aaaaaaaa-0000-0000-0000-000000000002.jsonl"
    data = _jsonl(_codex_rows(T0 + 7))
    calls: list[list[str]] = []

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append(argv)
        assert kwargs["timeout"] == 10.0
        out = (
            _volume_block(first, 0, _iso(T0 + 5), b'{"type":"session_meta"}\n')
            + _volume_block(second, 0, _iso(T0 + 7), data + b'{"partial')
            + _volume_block("/home/node/.codex/auth.json", 0, "", b"CREDENTIAL\n")
        )
        return subprocess.CompletedProcess(argv, 0, out, b"")

    monkeypatch.setattr(tmod.subprocess, "run", fake_run)
    clock = [100.0]
    monkeypatch.setattr(tmod.time, "monotonic", lambda: clock[0])
    records = [
        _start("run", "run", T0 - 600),
        _end(_start("sandbox_reset", "r0", T0 - 500)),  # an earlier generation's agent
        _start("agent", "old", T0 - 400, "g0-s9"),
        _start("sandbox_reset", "r1", T0 - 10),
        _start("agent", "a1", T0, "g1-s1"),
        _start("agent", "a2", T0 + 1, "g1-s2"),
    ]
    reader = TranscriptReader(tmp_path, _config(sandbox=True), FakeReader(records))

    out = reader.read("g1-s2")
    assert len(calls) == 1
    argv = calls[0]
    assert argv[:-1] == [
        "docker", "run", "--rm", "--pull", "never", "--network", "none", "--user", "node",
        "--security-opt", "no-new-privileges", "-v", "helix-auth-codex:/home/node:ro",
        DEFAULT_BACKEND_IMAGES["codex"], "sh", "-c",
    ]
    script = argv[-1]
    assert "find /home/node/.codex/sessions -type f -name 'rollout-*.jsonl'" in script
    assert "/home/node/" not in script.replace("/home/node/.codex/sessions", "")
    assert "auth" not in script and "credentials" not in script
    assert [a for a in argv if a.startswith("-v") or ":/" in a] == ["-v", "helix-auth-codex:/home/node:ro"]

    # g1-s1 started first and takes the first session; g1-s2 gets the second.
    assert out["source"] == "live" and out["session"] == "aaaaaaaa-0000-0000-0000-000000000002"
    assert out["cursor"] == len(data) and len(out["events"]) == 8
    assert out["note"] == "session matched to this agent by start time"
    assert "CREDENTIAL" not in json.dumps(out)

    # Within 4 s: no docker call, nothing new.
    clock[0] += 1.0
    assert reader.read("g1-s2", out["cursor"])["events"] == []
    assert len(calls) == 1

    # After 4 s: one call that tails only the matched file from the cursor.
    clock[0] += 4.0
    reader.read("g1-s2", out["cursor"])
    assert len(calls) == 2
    follow = calls[1][-1]
    assert follow.startswith(f"printf '%s\\n' {shlex.quote(second)} |")
    assert f"st={len(data)}" in follow and "find" not in follow


def test_live_sandboxed_docker_timeout_is_a_note(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def slow(argv: list[str], **kwargs: Any) -> Any:
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    monkeypatch.setattr(tmod.subprocess, "run", slow)
    records = [_start("run", "run", T0), _start("agent", "a1", T0, "g1-s1")]
    out = TranscriptReader(tmp_path, _config(sandbox=True), FakeReader(records)).read("g1-s1")
    assert out["source"] == "live" and out["events"] == []
    assert out["note"] == "could not read the auth volume: TimeoutExpired"


def test_volume_script_runs_under_sh(tmp_path: Path) -> None:
    """The script's framing, offsets and first-line timestamp, run on real files with a local listing."""
    path = tmp_path / "s.jsonl"
    data = _jsonl(_codex_rows())
    path.write_bytes(data)
    listing = f"printf '%s\\n' {shlex.quote(str(path))}"

    out = subprocess.run(["sh", "-c", tmod._volume_script(listing, 0)], capture_output=True, check=True).stdout
    [(name, offset, stamp, marker, content)] = list(tmod._split_volume_output(out))
    assert (name, offset, marker, content) == (str(path), 0, "", data)
    assert tmod._epoch(stamp) == pytest.approx(T0)

    out = subprocess.run(["sh", "-c", tmod._volume_script(listing, 10)], capture_output=True, check=True).stdout
    [(_, offset, _, _, content)] = list(tmod._split_volume_output(out))
    assert offset == 10 and content == data[10:]


def test_volume_script_finds_claude_start_after_an_untimed_first_line(tmp_path: Path) -> None:
    """Current Claude Code opens a session with an untimed ``session-owner`` line."""
    path = tmp_path / "c.jsonl"
    owner = b'{"type":"session-owner","sessionId":"abc"}\n'
    path.write_bytes(owner + b'{"type":"queue-operation","timestamp":"2026-11-16T07:31:33.062Z"}\n')
    listing = f"printf '%s\\n' {shlex.quote(str(path))}"
    out = subprocess.run(["sh", "-c", tmod._volume_script(listing, 0)], capture_output=True, check=True).stdout
    [(_, _, stamp, _, _)] = list(tmod._split_volume_output(out))
    assert tmod._epoch(stamp) == pytest.approx(1794814293.062)


def test_volume_script_reads_the_candidate_from_the_launch_line(tmp_path: Path) -> None:
    """HELIX names the candidate in the line it launches the agent with, the session's first user message."""
    path = tmp_path / "s.jsonl"
    launch = {"type": "response_item", "payload": {"role": "user", "content": [
        {"type": "input_text", "text": "Read .agent_task_prompt.md in the current workspace and follow "
                                       "those instructions exactly. This is HELIX candidate g12-s345."}]}}
    path.write_bytes(_jsonl(_codex_rows()[:1]) + _jsonl([launch]))
    listing = f"printf '%s\\n' {shlex.quote(str(path))}"
    out = subprocess.run(["sh", "-c", tmod._volume_script(listing, 0)], capture_output=True, check=True).stdout
    [(_, _, _, marker, _)] = list(tmod._split_volume_output(out))
    assert marker == "g12-s345"


def test_marked_sessions_match_exactly_whatever_their_start_order(tmp_path: Path) -> None:
    """Parallel agents' sessions can start out of order; the launch-line marker decides."""
    s = "/home/node/.codex/sessions/2026/11/15/rollout-2026-11-15T00-00-0{}-aaaaaaaa-0000-0000-0000-00000000000{}.jsonl"
    files = {
        s.format(5, 1): (0, _iso(T0 + 5), "g1-s2", b""),   # g1-s2's session began first
        s.format(6, 2): (0, _iso(T0 + 6), "g1-s1", b""),
        s.format(7, 3): (0, _iso(T0 + 7), "", b""),         # an older-style session with no marker
    }
    records = [_start("run", "run", T0 - 60), _start("sandbox_reset", "r", T0 - 10),
               _start("agent", "a1", T0, "g1-s1"), _start("agent", "a2", T0 + 1, "g1-s2"),
               _start("agent", "a3", T0 + 2, "g1-s3")]
    spans = {r["candidate"]: r for r in records if r.get("span") == "agent"}
    assert tmod._match_session(spans["g1-s1"], records, files, "g1-s1") == (s.format(6, 2), True)
    assert tmod._match_session(spans["g1-s2"], records, files, "g1-s2") == (s.format(5, 1), True)
    # g1-s3 has no marked session: start order among the unmarked ones.
    assert tmod._match_session(spans["g1-s3"], records, files, "g1-s3") == (s.format(7, 3), False)
