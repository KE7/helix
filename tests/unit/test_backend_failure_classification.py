"""Backend failure classification.

A non-zero backend exit must be classified from the structured result
envelope first, and only from raw output when the envelope says nothing.
Getting the order wrong discards completed work: a session id containing a
"429"/"529" hex fragment used to be read as a rate limit and thrown away a
proposal that had already run to its turn limit.

Claude fixtures follow the real Claude Code envelope: ``subtype`` is always
present ("success" or an ``error_*`` value); success envelopes carry
``result: <str>``, error subtypes carry ``errors: [<str>, ...]``, and there
is no top-level ``error`` field.  Newer CLIs add ``api_error_status`` to
success envelopes; older CLIs and error subtypes do not.
"""

from __future__ import annotations

import json
import subprocess

import pytest

import helix.mutator as mutator
from helix.config import AgentConfig, HelixConfig
from helix.exceptions import MutationError, RateLimitError
from helix.population import Candidate, EvalResult

# Synthetic. Shaped like a session id and carrying "429"/"529" as hex
# fragments, which is the collision the whole-token match must survive.
_MAX_TURNS_SESSION_ID = "00000000-0429-4000-8000-000000000529"


def _completed(stdout: str = "", stderr: str = "", returncode: int = 1):
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _success_envelope(**fields):
    """A Claude Code ``subtype: "success"`` result envelope."""
    envelope = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "result": "Done.",
        "session_id": _MAX_TURNS_SESSION_ID,
        "num_turns": 3,
    }
    envelope.update(fields)
    return envelope


def _error_envelope(subtype: str, *errors: str):
    """A Claude Code ``error_*`` result envelope (``errors`` list, no result)."""
    return {
        "type": "result",
        "subtype": subtype,
        "is_error": True,
        "errors": list(errors),
        "session_id": _MAX_TURNS_SESSION_ID,
        "num_turns": 3,
    }


def _jsonl(*events):
    return "\n".join(json.dumps(event) for event in events) + "\n"


class TestRateLimitSignals:
    @pytest.mark.parametrize(
        "text",
        [
            "HTTP 429 Too Many Requests",
            "HTTP 529",
            "status code: 429",
            "You have exceeded your rate limit",
            "rate-limit hit",
            "server overloaded",
            "usage limit reached",
            "extra usage consumed",
            "too many requests",
            "quota exceeded for this model",
            "You exceeded your quota",
            # The backend's own type tokens: ``\b`` treats "_" as a word
            # character, so these need letter/digit lookarounds to match.
            "overloaded_error",
            "rate_limit_error",
            'API Error: 529 {"type":"overloaded_error","message":"Overloaded"}',
            'API Error: 429 {"type":"rate_limit_error","message":"..."}',
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
            # A bare 429/529 with no accompanying backend-error wording is
            # ordinary output an agent backend can print for any reason —
            # it must not be read as a rate limit.
            "429",
            "529",
            "Line 429: unexpected token",
            "conversation used 429 tokens so far",
            "record 529 created successfully",
            "529 ms elapsed",
            "@@ -12,6 +429,9 @@ def run():",
            "prorate limits apply",
        ],
    )
    def test_unrelated_errors_are_not_called_rate_limits(self, text):
        """An unclassified failure must not be reported as a quota problem.

        Sending an operator to a quota dashboard for a non-quota failure is
        strictly worse than saying the error is unrecognised. This also
        covers the bare-number case: a "429"/"529" with no HTTP/status/code
        wording nearby is ordinary backend chatter, not a rate-limit signal.
        """
        assert mutator._looks_like_rate_limit(text) is False


class TestNonZeroExitClassification:
    def test_structured_api_status_429_is_a_rate_limit(self, tmp_path, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(_success_envelope(api_error_status=429, result="")),
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

    def test_structured_api_status_529_is_an_overload_rate_limit(self, tmp_path, mocker):
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(_success_envelope(api_error_status=529, result="")),
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

    def test_jsonl_error_event_rate_limit_is_seen_when_stderr_is_non_empty(
        self, tmp_path, mocker
    ):
        """A structured error event on stdout must not be hidden by stderr."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=_jsonl(
                    {"type": "step_start"},
                    {
                        "type": "error",
                        "error": {
                            "name": "APIError",
                            "data": {"message": "HTTP 429 Too Many Requests"},
                        },
                    },
                ),
                stderr="non-empty unrelated warning",
            ),
        )

        with pytest.raises(RateLimitError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="opencode")
            )

        assert exc_info.value.phase == "structured backend result"
        assert exc_info.value.stderr == "non-empty unrelated warning"

    def test_jsonl_transcript_text_is_never_scanned_for_rate_limits(
        self, tmp_path, mocker
    ):
        """Tool output echoed in the transcript is not a backend failure.

        A repository whose source mentions "rate limit" (this one does) is
        read by the agent and echoed on stdout; an unrelated non-zero exit
        must stay a generic failure whether stderr is empty or not.
        """
        transcript = _jsonl(
            {"type": "step_start"},
            {
                "type": "tool_result",
                "name": "read",
                "output": "def _looks_like_rate_limit(text):  # HTTP 429 handling",
            },
            {"type": "text", "text": "The rate limit helper looks fine; status code 429."},
        )
        for stderr in ("", "unrelated warning"):
            mocker.patch(
                "helix.mutator.subprocess.run",
                return_value=_completed(stdout=transcript, stderr=stderr),
            )

            with pytest.raises(MutationError) as exc_info:
                mutator.invoke_claude_code(
                    str(tmp_path), "prompt", AgentConfig(backend="opencode")
                )

            assert not isinstance(exc_info.value, RateLimitError)

    def test_jsonl_non_json_stdout_line_is_scanned_when_stderr_is_empty(
        self, tmp_path, mocker
    ):
        """A plain-text error printed on stdout (not a transcript event)
        is still consulted once stderr has nothing to say."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout="Error: HTTP 429 Too Many Requests\n", stderr=""
            ),
        )

        with pytest.raises(RateLimitError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="opencode")
            )

        assert exc_info.value.phase == "subprocess exit fallback"

    def test_stderr_takes_precedence_over_unparsable_stdout(self, tmp_path, mocker):
        """Base behaviour: with a non-empty stderr, raw stdout is not scanned."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout="see docs on rate limit handling\n",
                stderr="ENOENT: no such file or directory",
            ),
        )

        with pytest.raises(MutationError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="opencode")
            )

        assert not isinstance(exc_info.value, RateLimitError)

    def test_success_envelope_flagged_is_error_with_429_result_is_a_rate_limit(
        self, tmp_path, mocker
    ):
        """Older CLIs omit ``api_error_status``; the message lives in ``result``."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(
                    _success_envelope(
                        is_error=True, result="API Error: 429 rate limit exceeded"
                    )
                ),
            ),
        )

        with pytest.raises(RateLimitError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="claude")
            )

        assert exc_info.value.phase == "structured backend result"
        assert exc_info.value.exit_code == 1

    def test_error_subtype_with_overloaded_error_token_is_a_rate_limit(
        self, tmp_path, mocker
    ):
        """Error subtypes carry ``errors: [...]`` and no ``api_error_status``."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(
                    _error_envelope(
                        "error_during_execution",
                        'API Error: 529 {"type":"overloaded_error","message":"Overloaded"}',
                    )
                ),
            ),
        )

        with pytest.raises(RateLimitError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="claude")
            )

        assert exc_info.value.phase == "structured backend result"

    def test_overloaded_envelope_with_subtype_success_is_still_a_rate_limit(
        self, tmp_path, mocker
    ):
        """Regression: ``subtype`` is always present and must not, on its own,
        be read as "already classified" and turn an overload into a success.
        """
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(
                    _success_envelope(is_error=True, result="Claude is overloaded")
                ),
                returncode=0,
            ),
        )

        with pytest.raises(RateLimitError):
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="claude")
            )

    def test_successful_turn_mentioning_rate_limits_is_returned(
        self, tmp_path, mocker
    ):
        """The agent's own prose is not a backend failure report."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(
                    _success_envelope(
                        result="Added retry-on-rate-limit handling for HTTP 429 responses."
                    )
                ),
                returncode=0,
            ),
        )

        parsed, _usage = mutator.invoke_claude_code(
            str(tmp_path), "prompt", AgentConfig(backend="claude")
        )

        assert parsed["subtype"] == "success"

    def test_max_turns_session_id_collision_remains_partial_success(
        self, tmp_path, mocker
    ):
        """The session id carries "429"; the envelope must still win."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(
                    {
                        "type": "result",
                        "subtype": "error_max_turns",
                        "session_id": _MAX_TURNS_SESSION_ID,
                        "num_turns": 60,
                        "errors": ["Reached maximum number of turns (60)"],
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
                        "type": "result",
                        "subtype": "error_max_turns",
                        "session_id": _MAX_TURNS_SESSION_ID,
                        "num_turns": 60,
                        "errors": ["Reached maximum number of turns (60)"],
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

    def test_error_subtype_without_rate_limit_wording_stays_generic(
        self, tmp_path, mocker
    ):
        """An ``error_*`` envelope is classified from its own ``errors``
        strings.  A bare number there is not a status code, so this stays a
        generic failure with the backend's message preserved, and the
        envelope's session id (which carries "429"/"529") is never scanned.
        """
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(
                    _error_envelope(
                        "error_during_execution",
                        "worker restart id 429 logged",
                    )
                ),
            ),
        )

        with pytest.raises(MutationError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="claude")
            )

        assert not isinstance(exc_info.value, RateLimitError)
        assert "worker restart id 429 logged" in exc_info.value.stdout

    def test_claude_envelope_stdout_is_not_rescanned_as_raw_text(
        self, tmp_path, mocker
    ):
        """Once the envelope parsed, stdout is structured output already
        judged; a "rate limit" mention in the agent's ``result`` prose on an
        unrelated non-zero exit must not become a rate-limit verdict."""
        mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(
                stdout=json.dumps(
                    _success_envelope(
                        result="Implemented the rate limit middleware; tests pass."
                    )
                ),
                stderr="",
            ),
        )

        with pytest.raises(MutationError) as exc_info:
            mutator.invoke_claude_code(
                str(tmp_path), "prompt", AgentConfig(backend="claude")
            )

        assert not isinstance(exc_info.value, RateLimitError)
