"""What the agent is told about its candidate (nothing), and what ties its session and a rejected proposal back to it."""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from helix import timing
from helix.config import AgentConfig, SandboxConfig
from helix.mutator import _build_backend_args, _prompt_file_instruction, invoke_claude_code
from helix.sandbox import _run_streaming
from helix.timing import recording, span

BACKENDS = ["agy", "claude", "codex", "cursor", "opencode", "pi"]
LAUNCH_LINE = "Read .agent_task_prompt.md in the current workspace and follow those instructions exactly."


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_launch_line_is_the_same_for_every_candidate(backend: str) -> None:
    args = _build_backend_args("/workspace", AgentConfig(backend=backend), ".agent_task_prompt.md")
    assert [a for a in args if ".agent_task_prompt.md" in a] == [LAUNCH_LINE]
    assert not any("HELIX" in a for a in args)
    assert _prompt_file_instruction(".agent_task_prompt.md") == LAUNCH_LINE


def _invoke_sandboxed(tmp_path: Path, mocker: Any, backend: str, stdout_lines: list[str]) -> MagicMock:
    """Invoke the backend for candidate g3-s12 in the sandbox, its output streamed line by line."""
    worktree = tmp_path / ".helix" / "worktrees" / "g3-s12"
    worktree.mkdir(parents=True)

    def fake_run(args: list[str], **kwargs: Any) -> MagicMock:
        for line in stdout_lines:
            if kwargs.get("on_stdout_line") is not None:
                kwargs["on_stdout_line"](line)
        return MagicMock(stdout="".join(stdout_lines), stderr="", returncode=0)

    run = mocker.patch("helix.mutator.run_sandboxed_command", side_effect=fake_run)
    mocker.patch("helix.mutator.resolve_sandbox_image", return_value="img")
    mocker.patch("helix.mutator._parse_backend_output", return_value={})
    invoke_claude_code(str(worktree), "the prompt", AgentConfig(backend=backend), sandbox=SandboxConfig(enabled=True))
    return run


@pytest.mark.parametrize("backend", BACKENDS)
def test_nothing_the_agent_is_handed_names_its_candidate(tmp_path: Path, mocker: Any, backend: str) -> None:
    run = _invoke_sandboxed(tmp_path, mocker, backend, [])
    args, env = run.call_args[0][0], run.call_args[1]["env"]
    handed = [*args, *env.values()]
    assert not any("g3-s12" in a or "HELIX candidate" in a for a in handed)
    assert LAUNCH_LINE in args


def test_claude_gets_a_session_id_that_helix_records(tmp_path: Path, mocker: Any) -> None:
    path = tmp_path / "timing.jsonl"
    with recording(path), span("generation", generation=3), span("proposal", candidate_id="g3-s12"):
        run = _invoke_sandboxed(tmp_path, mocker, "claude", [])
    args = run.call_args[0][0]
    session_id = args[args.index("--session-id") + 1]
    assert str(uuid.UUID(session_id)) == session_id
    records = timing.read_records(path)
    agent = next(r for r in records if r.get("span") == "agent" and r["event"] == "start")
    [rec] = [r for r in records if r.get("event") == "session"]
    assert rec["session_id"] == session_id
    assert rec["parent_span_id"] == agent["span_id"] and rec["candidate"] == "g3-s12" and rec["generation"] == 3


def test_codex_session_id_is_read_from_its_first_event_as_it_streams(tmp_path: Path, mocker: Any) -> None:
    path = tmp_path / "timing.jsonl"
    lines = [
        '{"type":"thread.started","thread_id":"019a7c3e-1111-7222-8333-944445555666"}\n',
        '{"type":"turn.started"}\n',
        '{"type":"thread.started","thread_id":"not-this-one"}\n',
    ]
    with recording(path), span("proposal", candidate_id="g3-s12"):
        run = _invoke_sandboxed(tmp_path, mocker, "codex", lines)
    assert "--session-id" not in run.call_args[0][0]
    sessions = [r for r in timing.read_records(path) if r.get("event") == "session"]
    assert [r["session_id"] for r in sessions] == ["019a7c3e-1111-7222-8333-944445555666"]


def test_streaming_run_hands_each_line_over_and_keeps_the_output() -> None:
    seen: list[str] = []
    script = "import sys; print('one'); print('two'); print('err', file=sys.stderr); sys.exit(3)"
    result = _run_streaming([sys.executable, "-c", script], cwd=None, timeout=30, on_stdout_line=seen.append)
    assert seen == ["one\n", "two\n"]
    assert (result.returncode, result.stdout, result.stderr) == (3, "one\ntwo\n", "err\n")


def test_streaming_run_times_out_with_the_output_so_far() -> None:
    script = "import sys, time; print('first', flush=True); time.sleep(30)"
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        _run_streaming([sys.executable, "-c", script], cwd=None, timeout=1, on_stdout_line=lambda line: None)
    assert caught.value.output == "first\n"


def test_streaming_run_deadline_covers_output_left_open_by_a_child(monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that exits while a child it started keeps the pipe open still times out, as subprocess.run does."""
    import helix.sandbox as sandbox

    monkeypatch.setattr(sandbox, "_DRAIN_AFTER_KILL", 0.1)
    script = ("import subprocess, sys; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(4)']); "
              "print('parent done', flush=True)")
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        _run_streaming([sys.executable, "-c", script], cwd=None, timeout=1, on_stdout_line=lambda line: None)
    assert caught.value.output == "parent done\n"


def test_streaming_run_replaces_undecodable_bytes() -> None:
    script = "import sys; sys.stdout.buffer.write(b'ok \\xff\\n'); sys.stderr.buffer.write(b'\\xe2\\x82')"
    result = _run_streaming([sys.executable, "-c", script], cwd=None, timeout=30, on_stdout_line=lambda line: None)
    assert result.stdout == "ok \ufffd\n" and result.stderr == "\ufffd"


def test_a_failing_line_callback_does_not_stop_the_capture() -> None:
    def boom(line: str) -> None:
        raise ValueError(line)

    result = _run_streaming([sys.executable, "-c", "print('a'); print('b')"], cwd=None, timeout=30, on_stdout_line=boom)
    assert result.stdout == "a\nb\n" and result.returncode == 0


def test_a_session_record_outside_a_recording_is_a_no_op() -> None:
    timing.session_recorder()("abc")  # no sink: nothing to write, nothing raised


def test_session_records_are_valid_for_the_dashboard(tmp_path: Path) -> None:
    from helix.ui.run_state import _valid_record

    path = tmp_path / "timing.jsonl"
    with recording(path), span("agent"):
        timing.session_recorder()("0b5c3a0e-0000-4000-8000-000000000001")
    [rec] = [r for r in timing.read_records(path) if r.get("event") == "session"]
    assert _valid_record(json.loads(json.dumps(rec)))


def test_a_decision_records_paths_only_when_given(tmp_path: Path) -> None:
    path = tmp_path / "timing.jsonl"
    with recording(path), span("generation", generation=2):
        timing.decision("g2-s5", accepted=False, reason="tamper-rejected mutation candidate",
                        paths=["val/b.json", "evaluate.py"])
        timing.decision("g2-s6", accepted=False, reason="minibatch-gate-rejected candidate")
    decisions = [r for r in timing.read_records(path) if r.get("event") == "decision"]
    assert decisions[0]["paths"] == ["evaluate.py", "val/b.json"]
    assert "paths" not in decisions[1]


@pytest.mark.parametrize("retain", [False, True])
def test_a_tamper_reject_records_its_paths_whether_kept_or_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retain: bool
) -> None:
    from helix import evolution
    from helix.population import Candidate

    monkeypatch.setattr(evolution, "remove_worktree", lambda c: None)
    monkeypatch.setattr(evolution, "snapshot_candidate", lambda c, m: "sha")
    child = Candidate(id="g2-s5", worktree_path=str(tmp_path / "g2-s5"), branch_name="helix/g2-s5",
                      generation=2, parent_id="g1-s1", parent_ids=["g1-s1"], operation="mutate")
    path = tmp_path / "timing.jsonl"
    with recording(path), span("generation", generation=2):
        evolution._discard_rejected(child, label="tamper-rejected mutation candidate", retain=retain,
                                    commit=True, paths=["val/b.json", "evaluate.py"])
    [rec] = [r for r in timing.read_records(path) if r.get("event") == "decision"]
    assert rec["paths"] == ["evaluate.py", "val/b.json"] and rec["accepted"] is False
