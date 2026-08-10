"""Backwards-compatibility tests for the opencode backend.

The opencode CLI changed its non-interactive permission flag and its
structured-output shape between releases. HELIX must work against both the
older and the newer builds rather than pinning either one.
"""

from __future__ import annotations

import json
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import helix.mutator as mutator
from helix.config import HelixConfig
from helix.exceptions import MutationError, RateLimitError
from helix.population import Candidate, EvalResult


@pytest.fixture(autouse=True)
def _clear_flag_cache():
    mutator._opencode_permission_flag_cache = None
    yield
    mutator._opencode_permission_flag_cache = None


def _help(text: str):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=text, stderr="")


class TestPermissionFlagProbe:
    def test_new_cli_gets_auto(self, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_help("  --auto  auto-approve permissions (dangerous!)"),
        )
        assert mutator.opencode_permission_flag() == "--auto"

    def test_old_cli_keeps_legacy_flag(self, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_help("  --dangerously-skip-permissions  skip prompts"),
        )
        assert mutator.opencode_permission_flag() == "--dangerously-skip-permissions"

    def test_probe_is_cached_across_calls(self, mocker):
        run = mocker.patch("helix.mutator.subprocess.run", return_value=_help("--auto"))
        for _ in range(5):
            mutator.opencode_permission_flag()
        assert run.call_count == 1, "a P×N batch must not re-probe per mutation"

    def test_probe_is_cached_across_parallel_workers(self, mocker):
        """Concurrent first mutations still pay for exactly one probe."""
        probe_started = threading.Event()
        release_probe = threading.Event()

        def delayed_help(*_args, **_kwargs):
            probe_started.set()
            assert release_probe.wait(timeout=2)
            return _help("--auto")

        run = mocker.patch("helix.mutator.subprocess.run", side_effect=delayed_help)
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(mutator.opencode_permission_flag)
            assert probe_started.wait(timeout=2)
            second = executor.submit(mutator.opencode_permission_flag)
            release_probe.set()
            assert first.result(timeout=2) == "--auto"
            assert second.result(timeout=2) == "--auto"
        assert run.call_count == 1

    def test_lookalike_flag_does_not_mask_legacy_flag(self, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_help("--auto-continue\n--dangerously-skip-permissions"),
        )
        assert mutator.opencode_permission_flag() == "--dangerously-skip-permissions"

    def test_probe_failure_falls_back(self, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run", side_effect=FileNotFoundError("opencode")
        )
        assert (
            mutator.opencode_permission_flag()
            == mutator.OPENCODE_PERMISSION_FLAG_FALLBACK
        )

    def test_built_args_use_probed_flag(self, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_help("  --dangerously-skip-permissions  skip prompts"),
        )
        from helix.config import AgentConfig

        args = mutator._build_backend_args(
            "/tmp/wt", AgentConfig(backend="opencode"), "prompt.md"
        )
        assert "--dangerously-skip-permissions" in args
        assert "--auto" not in args


class TestOpenCodeOutputParsing:
    def _parse(self, stdout: str, *, strict: bool = True):
        return mutator._parse_jsonl_output(
            stdout,
            backend="opencode",
            cmd_str="opencode run",
            worktree_path="/tmp/wt",
            stderr="",
            exit_code=0,
            strict=strict,
        )

    def test_jsonl_stream(self):
        stdout = (
            '{"type":"step_start"}\n'
            '{"type":"tool_use","part":{"tool":"edit"}}\n'
            '{"type":"step_finish"}\n'
        )
        result = self._parse(stdout)
        assert [e["type"] for e in result["events"]] == [
            "step_start",
            "tool_use",
            "step_finish",
        ]
        assert result["unparsable_lines"] == []

    def test_blank_lines_are_skipped(self):
        result = self._parse('\n{"type":"a"}\n\n\n{"type":"b"}\n\n')
        assert len(result["events"]) == 2

    def test_single_line_object_still_parses(self):
        result = self._parse('{"type":"result","sessionID":"ses_1"}\n')
        assert result["events"] == [{"type": "result", "sessionID": "ses_1"}]

    def test_pretty_printed_single_object_is_accepted(self):
        """Older builds emitted one indented object spanning many lines."""
        stdout = json.dumps({"type": "result", "sessionID": "ses_1"}, indent=2)
        assert "\n" in stdout
        result = self._parse(stdout)
        assert result["events"] == [{"type": "result", "sessionID": "ses_1"}]
        assert result["unparsable_lines"] == []

    def test_empty_stdout_is_not_an_error(self):
        result = self._parse("")
        assert result["events"] == []
        assert result["unparsable_lines"] == []

    def test_unparseable_stream_still_fails_loudly(self):
        with pytest.raises(MutationError):
            self._parse('{"type":"a"}\nthis is not json at all\n')

    def test_non_strict_records_unparsable_instead_of_raising(self):
        result = self._parse('{"type":"a"}\nnot json\n', strict=False)
        assert len(result["events"]) == 1
        assert result["unparsable_lines"] == ["not json"]


class TestRateLimitClassification:
    @pytest.mark.parametrize(
        "text",
        [
            "HTTP 429 Too Many Requests",
            "429",
            "You have exceeded your rate limit",
            "server overloaded",
            "usage limit reached",
            "quota exceeded for this model",
        ],
    )
    def test_real_rate_limit_signals_are_detected(self, text):
        assert mutator._looks_like_rate_limit(text) is True

    @pytest.mark.parametrize(
        "text",
        [
            "UnknownError: Unexpected server error. Check server logs for details.",
            "ENOENT: no such file or directory",
            "model not found",
            "Configuration error: quota field is missing",
            "Configuration error: rate limit must be a positive integer",
            "The quota identifier is invalid",
            "b883f164-a429-4646-91a6-5a606e5292a9",
            "",
        ],
    )
    def test_unrelated_errors_are_not_called_rate_limits(self, text):
        """An unclassified failure must not be reported as a quota problem.

        Sending an operator to a quota dashboard for a non-quota failure is
        strictly worse than saying the error is unrecognised.
        """
        assert mutator._looks_like_rate_limit(text) is False

    def test_http_429_raises_rate_limit_error(self, tmp_path, mocker):
        run = mocker.patch(
            "helix.mutator.subprocess.run",
            side_effect=[
                _help("--auto"),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=1,
                    stdout="",
                    stderr="HTTP 429 Too Many Requests",
                ),
            ],
        )
        from helix.config import AgentConfig

        with pytest.raises(mutator.RateLimitError):
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="opencode")
            )
        assert run.call_count == 2

    def test_structured_api_status_429_is_rate_limit_with_stderr_present(
        self, tmp_path, mocker
    ):
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[],
                returncode=1,
                stdout=json.dumps(
                    {"api_error_status": 429, "error": "request rejected"}
                ),
                stderr="unrelated warning emitted on stderr",
            ),
        )
        from helix.config import AgentConfig

        with pytest.raises(RateLimitError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="claude")
            )
        assert exc_info.value.exit_code == 1
        assert exc_info.value.phase == "structured backend result"
        assert exc_info.value.stdout
        assert exc_info.value.stderr == "unrelated warning emitted on stderr"

    def test_stdout_429_is_checked_when_stderr_is_non_empty(self, tmp_path, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            side_effect=[
                _help("--auto"),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=1,
                    stdout="HTTP 429 Too Many Requests",
                    stderr="non-empty unrelated warning",
                ),
            ],
        )
        from helix.config import AgentConfig

        with pytest.raises(RateLimitError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="opencode")
            )
        assert exc_info.value.stdout == "HTTP 429 Too Many Requests"
        assert exc_info.value.stderr == "non-empty unrelated warning"

    def test_max_turns_session_id_collision_remains_partial_success(
        self, tmp_path, mocker
    ):
        payload = {
            "subtype": "error_max_turns",
            "session_id": "b883f164-a429-4646-91a6-5a606e5292a9",
            "num_turns": 60,
            "error": "Reached maximum number of turns (60)",
        }
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stdout=json.dumps(payload), stderr=""
            ),
        )
        from helix.config import AgentConfig

        parsed, usage = mutator.invoke_claude_code(
            str(tmp_path), "prompt", AgentConfig(backend="claude")
        )
        assert parsed["subtype"] == "error_max_turns"
        assert usage.num_turns == 60

    def test_max_turns_collision_returns_candidate_to_evolution_gate(
        self, tmp_path, mocker
    ):
        """A partial result must leave the child available for normal gating."""
        parent = Candidate(
            id="g0-s0",
            worktree_path=str(tmp_path),
            branch_name="main",
            generation=0,
            parent_id=None,
            parent_ids=[],
            operation="seed",
        )
        child_path = tmp_path / "g1-s1"
        child_path.mkdir()
        child = Candidate(
            id="g1-s1",
            worktree_path=str(child_path),
            branch_name="helix/g1-s1",
            generation=1,
            parent_id="g0-s0",
            parent_ids=["g0-s0"],
            operation="mutate",
        )
        payload = {
            "subtype": "error_max_turns",
            "session_id": "b883f164-a429-4646-91a6-5a606e5292a9",
            "num_turns": 60,
        }
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=1, stdout=json.dumps(payload), stderr=""
            ),
        )
        config = HelixConfig(
            objective="objective",
            evaluator={"command": "true"},
            agent={"backend": "claude"},
        )

        returned = mutator.mutate(
            parent=parent,
            eval_result=EvalResult(candidate_id="g0-s0", scores={}, asi={}, instance_scores={}),
            new_id="g1-s1",
            config=config,
            base_dir=tmp_path,
        )

        assert returned is child, "partial max-turns work must reach the evolution gate"

    def test_unknown_failure_preserves_underlying_message(self, tmp_path, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            side_effect=[
                _help("--auto"),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=1,
                    stdout="",
                    stderr="backend exploded while preparing request",
                ),
            ],
        )
        from helix.config import AgentConfig

        with pytest.raises(MutationError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="opencode")
            )
        assert not isinstance(exc_info.value, RateLimitError)
        assert exc_info.value.stderr == "backend exploded while preparing request"

    def test_quota_configuration_error_is_not_a_rate_limit(self, tmp_path, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            side_effect=[
                _help("--auto"),
                subprocess.CompletedProcess(
                    args=[],
                    returncode=1,
                    stdout="",
                    stderr="Configuration error: quota field is missing",
                ),
            ],
        )
        from helix.config import AgentConfig

        with pytest.raises(MutationError):
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="opencode")
            )
