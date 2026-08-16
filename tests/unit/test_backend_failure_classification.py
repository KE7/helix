"""Backend failure classification.

A non-zero backend exit must be classified from the structured result
envelope first, and only from raw output when the envelope says nothing.
Getting the order wrong discards completed work: a session id containing a
"429"/"529" hex fragment used to be read as a rate limit and thrown away a
proposal that had already run to its turn limit.

This is backend-agnostic; the claude backend is used because it is the one
that emits a structured envelope alongside a non-zero exit.
"""

from __future__ import annotations

import json
import subprocess

import pytest

import helix.mutator as mutator
from helix.config import AgentConfig, HelixConfig
from helix.exceptions import MutationError, RateLimitError
from helix.population import Candidate, EvalResult

_MAX_TURNS_SESSION_ID = "00000000-0429-4000-8000-000000000529"


def _completed(stdout: str = "", stderr: str = "", returncode: int = 1):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestRateLimitSignals:
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
            _MAX_TURNS_SESSION_ID,
            "",
        ],
    )
    def test_unrelated_errors_are_not_called_rate_limits(self, text):
        """An unclassified failure must not be reported as a quota problem.

        Sending an operator to a quota dashboard for a non-quota failure is
        strictly worse than saying the error is unrecognised.
        """
        assert mutator._looks_like_rate_limit(text) is False


class TestNonZeroExitClassification:
    def test_structured_api_status_429_is_a_rate_limit(self, tmp_path, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps({"api_error_status": 429, "error": "request rejected"}),
                stderr="unrelated warning emitted on stderr",
            ),
        )

        with pytest.raises(RateLimitError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="claude")
            )

        assert exc_info.value.phase == "structured backend result"
        assert exc_info.value.exit_code == 1
        assert exc_info.value.stderr == "unrelated warning emitted on stderr"

    def test_stdout_rate_limit_is_seen_when_stderr_is_non_empty(self, tmp_path, mocker):
        """An unstructured status on stdout must not be hidden by stderr."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout="HTTP 429 Too Many Requests",
                stderr="non-empty unrelated warning",
            ),
        )

        with pytest.raises(RateLimitError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="opencode")
            )

        assert exc_info.value.stdout == "HTTP 429 Too Many Requests"
        assert exc_info.value.stderr == "non-empty unrelated warning"

    def test_max_turns_session_id_collision_remains_partial_success(
        self, tmp_path, mocker
    ):
        """The session id carries "429"; the envelope must still win."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(
                    {
                        "subtype": "error_max_turns",
                        "session_id": _MAX_TURNS_SESSION_ID,
                        "num_turns": 60,
                        "error": "Reached maximum number of turns (60)",
                    }
                )
            ),
        )

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
        mocker.patch("helix.mutator.clone_candidate", return_value=child)
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(
                    {
                        "subtype": "error_max_turns",
                        "session_id": _MAX_TURNS_SESSION_ID,
                        "num_turns": 60,
                    }
                )
            ),
        )

        returned = mutator.mutate(
            parent=parent,
            eval_result=EvalResult(
                candidate_id="g0-s0", scores={}, asi={}, instance_scores={}
            ),
            new_id="g1-s1",
            config=HelixConfig(
                objective="objective",
                evaluator={"command": "true"},
                agent={"backend": "claude"},
            ),
            base_dir=tmp_path,
        )

        assert returned is child, "partial max-turns work must reach the evolution gate"

    def test_unknown_failure_preserves_underlying_message(self, tmp_path, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(stderr="backend exploded while preparing request"),
        )

        with pytest.raises(MutationError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="opencode")
            )

        assert not isinstance(exc_info.value, RateLimitError)
        assert exc_info.value.stderr == "backend exploded while preparing request"
