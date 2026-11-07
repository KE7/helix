"""The ``pi`` backend: argv, stream parsing, usage, failures, transcripts, sandbox.

The event shapes below are trimmed from a real ``pi --mode json --print`` run
(pi 1.5.0) with two model calls: a ``bash`` tool call, then a final reply.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from helix import agent_timing
from helix.backends import (
    BACKEND_AUTH_COMMANDS,
    BACKEND_STATE_PATHS,
    transcript_session_id_keys,
)
from helix.config import AgentConfig, HelixConfig, EvaluatorConfig, SandboxConfig
from helix.exceptions import MutationError
from helix.mutator import (
    BACKEND_RESULT_ARTIFACT_NAME,
    _build_backend_args,
    _count_pi_stdout_tool_events,
    invoke_claude_code,
)
from helix import sandbox as sandbox_module

SID = "01a1b3ac-a1fd-7361-ae3a-22ebbac9c00f"


def _usage(inp: int, out: int, cache_read: int, cost: float, **extra: int) -> dict[str, Any]:
    return {
        "input": inp,
        "output": out,
        "cacheRead": cache_read,
        "cacheWrite": 0,
        **extra,
        "totalTokens": inp + out + cache_read,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": cost},
    }


def _assistant(content: list[dict[str, Any]], usage: dict[str, Any], stop: str, **extra: Any) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "api": "openai-completions",
        "provider": "mock",
        "model": "mock-1",
        "usage": usage,
        "stopReason": stop,
        **extra,
    }


def _stream(final_stop: str = "stop", **final_extra: Any) -> str:
    call = {"type": "toolCall", "id": "call_1", "name": "bash", "arguments": {"command": "echo hi > a.txt"}}
    first = _assistant([call], _usage(200, 20, 0, 0.00024, reasoning=4), "toolUse")
    last = _assistant([{"type": "text", "text": "done"}], _usage(200, 5, 100, 0.00022), final_stop, **final_extra)
    pending = _assistant([], _usage(0, 0, 0, 0.0), "pending")
    events: list[dict[str, Any]] = [
        {"type": "session", "version": 3, "id": SID, "timestamp": "2026-11-07T01:43:17.758Z", "cwd": "/workspace"},
        {"type": "agent_start"},
        {"type": "turn_start"},
        {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "Read it"}]}},
        {"type": "message_start", "message": pending},
        # Streaming updates carry running usage; they must not be summed.
        {"type": "message_update", "usage": _usage(200, 20, 0, 0.00024),
         "assistantMessageEvent": {"type": "toolcall_end", "contentIndex": 0, "toolCall": call}},
        {"type": "message_end", "message": first},
        {"type": "tool_execution_start", "toolCallId": "call_1", "toolName": "bash", "args": call["arguments"]},
        {"type": "tool_execution_end", "toolCallId": "call_1", "toolName": "bash",
         "result": {"content": [{"type": "text", "text": "(no output)"}]}, "isError": False, "durationMs": 168},
        {"type": "message_end", "message": {"role": "toolResult", "toolCallId": "call_1", "toolName": "bash",
                                            "content": [{"type": "text", "text": "(no output)"}], "isError": False}},
        {"type": "turn_end", "message": first, "toolResults": []},
        {"type": "turn_start"},
        {"type": "message_end", "message": last},
        {"type": "turn_end", "message": last, "toolResults": []},
        {"type": "agent_end", "messages": [first, last], "willRetry": False},
        {"type": "agent_settled", "aborted": False},
    ]
    return "".join(json.dumps(e) + "\n" for e in events)


def _run(mocker: Any, stdout: str, returncode: int = 0) -> MagicMock:
    run = mocker.patch("helix.mutator.subprocess.run")
    run.return_value = subprocess.CompletedProcess(["pi"], returncode, stdout=stdout, stderr="")
    return run


def test_pi_cli_args() -> None:
    args = _build_backend_args(
        "/wt", AgentConfig(backend="pi", model="anthropic/claude-sonnet-4-5", effort="high"), ".agent_task_prompt.md"
    )
    assert args[:4] == ["pi", "--mode", "json", "--print"]
    assert args[4:8] == ["--model", "anthropic/claude-sonnet-4-5", "--thinking", "high"]
    assert ".agent_task_prompt.md" in args[-1]
    assert not args[-1].startswith(("-", "@"))  # pi reads ``@x`` as a file


def test_pi_cli_args_without_model_or_effort() -> None:
    args = _build_backend_args("/wt", AgentConfig(backend="pi"), ".agent_task_prompt.md")
    assert args[:4] == ["pi", "--mode", "json", "--print"]
    assert len(args) == 5


def test_pi_host_run_gets_closed_stdin(mocker: Any, tmp_path: Path) -> None:
    run = _run(mocker, _stream())
    invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="pi"))
    assert run.call_args.kwargs["stdin"] is subprocess.DEVNULL


def test_pi_usage_sums_every_model_call(mocker: Any, tmp_path: Path) -> None:
    _run(mocker, _stream())

    parsed, usage = invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="pi"))

    assert parsed["session_id"] == SID
    assert usage.input_tokens == 400
    assert usage.output_tokens == 25
    assert usage.cache_read_input_tokens == 100
    assert usage.cache_creation_input_tokens == 0
    assert usage.cached_input_tokens == 0  # pi's input already excludes cache reads
    assert usage.reasoning_tokens == 4
    assert usage.cost_usd == pytest.approx(0.00046)
    assert usage.num_turns == 2
    assert usage.tool_event_count == 1
    assert usage.session_id == SID

    payload = json.loads((tmp_path / BACKEND_RESULT_ARTIFACT_NAME).read_text())
    assert payload["usage"]["input_tokens"] == 400
    assert payload["usage"]["tool_event_count"] == 1
    assert payload["usage"]["tool_names"] == ["bash"]


@pytest.mark.parametrize("stop", ["error", "aborted"])
def test_pi_failed_model_call_is_a_mutation_error(mocker: Any, tmp_path: Path, stop: str) -> None:
    # pi exits 0 when the provider rejects the call; only the stream says so.
    _run(mocker, _stream(stop, errorMessage="Your credit balance is too low"))

    with pytest.raises(MutationError, match="stopReason='" + stop) as exc_info:
        invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="pi"))

    assert "credit balance" in str(exc_info.value)
    # The calls that did run are still charged.
    assert exc_info.value.usage is not None
    assert exc_info.value.usage.input_tokens == 400


def test_pi_malformed_line_still_charges_usage(mocker: Any, tmp_path: Path) -> None:
    _run(mocker, "{truncated\n" + _stream())

    with pytest.raises(MutationError) as exc_info:
        invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="pi"))

    assert exc_info.value.usage is not None
    assert exc_info.value.usage.output_tokens == 25


def test_pi_tool_counter_reads_tool_execution_start(tmp_path: Path) -> None:
    path = tmp_path / "stdout.jsonl"
    path.write_text(_stream())
    assert _count_pi_stdout_tool_events(path) == (1, ["bash"])
    assert _count_pi_stdout_tool_events(tmp_path / "missing") == (0, [])


def test_pi_effort_values_validate(recwarn: Any) -> None:
    base = {"objective": "o", "evaluator": EvaluatorConfig(command="true")}
    for value in ("off", "minimal", "low", "medium", "high", "xhigh", "max"):
        HelixConfig(**base, agent=AgentConfig(backend="pi", effort=value))
    assert [w for w in recwarn.list if issubclass(w.category, UserWarning)] == []
    HelixConfig(**base, agent=AgentConfig(backend="pi", effort="extreme"))
    warnings = [str(w.message) for w in recwarn.list if w.category is UserWarning]
    assert any("not a recognized value" in w and "extreme" in w for w in warnings), warnings


def test_pi_web_access_false_is_accepted_and_adds_nothing() -> None:
    on = _build_backend_args("/wt", AgentConfig(backend="pi"), "p.md")
    off = _build_backend_args("/wt", AgentConfig(backend="pi", web_access=False), "p.md")
    assert on == off


# ---------------------------------------------------------------------------
# Transcripts and timing
# ---------------------------------------------------------------------------


def _session_file(path: Path) -> Path:
    """A pi session file: every entry has an ISO timestamp."""
    call = {"type": "toolCall", "id": "call_1", "name": "bash", "arguments": {"command": "x"}}
    rows = [
        {"type": "session", "version": 3, "id": SID, "timestamp": "2026-11-07T01:43:10.000Z", "cwd": "/workspace"},
        {"type": "message", "id": "a", "parentId": None, "timestamp": "2026-11-07T01:43:11.000Z",
         "message": {"role": "user", "content": [{"type": "text", "text": "go"}]}},
        {"type": "message", "id": "b", "parentId": "a", "timestamp": "2026-11-07T01:43:13.000Z",
         "message": _assistant([call], _usage(1, 1, 0, 0.0), "toolUse")},
        {"type": "message", "id": "c", "parentId": "b", "timestamp": "2026-11-07T01:43:16.000Z",
         "message": {"role": "toolResult", "toolCallId": "call_1", "toolName": "bash", "content": []}},
        {"type": "message", "id": "d", "parentId": "c", "timestamp": "2026-11-07T01:43:18.000Z",
         "message": _assistant([{"type": "text", "text": "done"}], _usage(1, 1, 0, 0.0), "stop")},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def test_pi_timing_pairs_tool_call_with_its_result(tmp_path: Path) -> None:
    totals = {t["name"]: t for t in agent_timing.breakdown("pi", _session_file(tmp_path / "s.jsonl"), 10.0)}
    assert totals["tool:bash"]["seconds"] == pytest.approx(3.0)
    assert totals["model"]["seconds"] == pytest.approx(5.0)
    assert totals["other"]["seconds"] == pytest.approx(2.0)


def test_pi_host_run_preserves_session_transcript(mocker: Any, tmp_path: Path, monkeypatch: Any) -> None:
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    _session_file(home / ".pi/agent/sessions/--wt--" / f"2026-11-07T01-43-17-758Z_{SID}.jsonl")
    wt = tmp_path / "wt"
    wt.mkdir()
    _run(mocker, _stream())

    invoke_claude_code(str(wt), "prompt", AgentConfig(backend="pi"))

    payload = json.loads((wt / BACKEND_RESULT_ARTIFACT_NAME).read_text())
    (artifact,) = payload["transcript_artifacts"]
    assert artifact["available"] is True
    assert artifact["path"] == f".helix_artifacts/backend_transcripts/pi/{SID}.jsonl"
    assert (wt / artifact["path"]).read_text().startswith('{"type": "session"')


# ---------------------------------------------------------------------------
# Sandbox
# ---------------------------------------------------------------------------


def test_pi_sandbox_session_id_and_copy_script() -> None:
    stdout = _stream()
    assert sandbox_module._extract_session_id_from_json_output(stdout, transcript_session_id_keys("pi")) == SID
    script = sandbox_module._backend_transcript_copy_script(
        agent_backend="pi", sandbox=SandboxConfig(enabled=True), session_id=SID
    )
    assert script is not None
    assert f'"$HOME"/.pi/agent/sessions/*/*_{SID}.jsonl' in script
    assert f"/workspace/.helix_artifacts/backend_transcripts/pi/{SID}.jsonl" in script


def test_pi_sandbox_uses_pi_runner_image(mocker: Any, tmp_path: Path) -> None:
    run = mocker.patch("helix.mutator.run_sandboxed_command")
    run.return_value = MagicMock(stdout=_stream(), stderr="", returncode=0)
    invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="pi"), sandbox=SandboxConfig(enabled=True))
    assert run.call_args.kwargs["image"] == "ghcr.io/ke7/helix-evo-runner-pi:latest"
    assert run.call_args.kwargs["agent_backend"] == "pi"


def test_pi_sandbox_auth_commands() -> None:
    login = sandbox_module.sandbox_auth_docker_args("pi", image="img", action="login", interactive=True)
    assert "helix-auth-pi:/home/node:rw" in login
    assert login[-1:] == ["pi"]
    status = BACKEND_AUTH_COMMANDS["pi"]["status"]
    assert status[:2] == ["sh", "-lc"] and ".pi/agent/auth.json" in status[2]
    logout = BACKEND_AUTH_COMMANDS["pi"]["logout"][-1]
    assert 'rm -f "${HOME:-/home/node}/.pi/agent/auth.json"' in logout
    # Session state is reset per generation; credentials never are.
    assert BACKEND_STATE_PATHS["pi"] == (".pi/agent/sessions",)
