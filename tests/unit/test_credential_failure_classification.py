"""A dead credential must not read as a candidate that wrote bad code.

Before this, a run whose shared login stopped working looked exactly like a run
whose agent kept producing broken diffs: the mutation was abandoned, the slot
was dropped, and nothing said why.  These tests pin the two halves of the fix
-- that the real wording the shipped CLIs emit is recognised, and that ordinary
agent output is not.

Every positive string below is a phrase read out of the shipped binary of the
CLI it is attributed to, not invented for the test.  The negatives are the
false-positive trap: candidate work that merely *mentions* tokens, auth, or a
401 must never be reported to an operator as a broken login.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from helix.config import AgentConfig, EvaluatorConfig, HelixConfig
from helix.display import UsageStats
from helix.exceptions import (
    CredentialRefreshError,
    HelixError,
    MutationError,
    RateLimitError,
)
from helix.mutator import (
    credential_failure_is_transient,
    credential_failure_marker,
    invoke_claude_code,
    mutate,
)
from helix.population import Candidate
from tests.unit.test_mutator import make_eval_result  # type: ignore[import-untyped]


# Codex CLI 0.130.0 -- the four suffixes it appends to one prefix, plus the
# bare form.  The "already used" variant is what losing a refresh race looks
# like from inside a candidate.
CODEX_ALREADY_USED = (
    "Your access token could not be refreshed because your refresh token was "
    "already used. Please log out and sign in again."
)
CODEX_EXPIRED = (
    "Your access token could not be refreshed because your refresh token has "
    "expired. Please log out and sign in again."
)
CODEX_REVOKED = (
    "Your access token could not be refreshed because your refresh token was "
    "revoked. Please log out and sign in again."
)
CODEX_OTHER_ACCOUNT = (
    "Your access token could not be refreshed because you have since logged "
    "out or signed in to another account. Please sign in again."
)
CODEX_BARE = (
    "Your access token could not be refreshed. Please log out and sign in again."
)
CODEX_GET_ACCOUNT = "failed to refresh token while getting account: 401"
CODEX_NO_ACCOUNT = (
    "ChatGPT account ID not available, please re-run `codex login`"
)
# OpenCode 1.14.24 -- thrown by the provider fetch wrapper.
OPENCODE_REFRESH_FAILED = "Token refresh failed: 400"
# Claude Code 2.1.138.
CLAUDE_OAUTH_REFRESH = "User OAuth refresh failed (HTTP 401): invalid_grant"
CLAUDE_INVALID_KEY = "API Error: 401 Invalid API key · Please run /login"

CREDENTIAL_STRINGS = [
    CODEX_ALREADY_USED,
    CODEX_EXPIRED,
    CODEX_REVOKED,
    CODEX_OTHER_ACCOUNT,
    CODEX_BARE,
    CODEX_GET_ACCOUNT,
    CODEX_NO_ACCOUNT,
    OPENCODE_REFRESH_FAILED,
    CLAUDE_OAUTH_REFRESH,
    CLAUDE_INVALID_KEY,
]

# Things a candidate legitimately produces while working on code.  None of them
# is a statement about HELIX's own login.
INNOCENT_STRINGS = [
    "",
    "401",
    "token",
    "auth",
    "refresh",
    "TypeError: 'NoneType' object is not subscriptable",
    "FAILED tests/test_auth.py::test_refresh_token_rotation - assert 0 == 1",
    "+    def refresh_token(self) -> str:\n+        raise NotImplementedError",
    "The test suite returned 401 for 3 requests; see auth.py line 88.",
    "Added a refresh token cache so the client stops re-authenticating.",
    "Error: 529 overloaded, please retry",
    "You have exceeded your usage limit",
]


class TestMarkerRecognition:
    @pytest.mark.parametrize("text", CREDENTIAL_STRINGS)
    def test_real_cli_wording_is_recognised(self, text: str) -> None:
        assert credential_failure_marker(text) is not None

    @pytest.mark.parametrize("text", CREDENTIAL_STRINGS)
    def test_recognition_is_case_insensitive(self, text: str) -> None:
        assert credential_failure_marker(text.upper()) is not None

    @pytest.mark.parametrize("text", INNOCENT_STRINGS)
    def test_ordinary_candidate_output_is_not_a_credential_failure(
        self, text: str
    ) -> None:
        assert credential_failure_marker(text) is None

    def test_marker_is_returned_as_evidence(self) -> None:
        """The caller names what matched instead of asserting a bare verdict."""
        marker = credential_failure_marker(CODEX_EXPIRED)
        assert marker == "your access token could not be refreshed"

    def test_already_used_is_the_transient_marker(self) -> None:
        """Losing a refresh race is named as such, not as a dead login.

        The already-used suffix is the exact outcome a lost race produces; it
        must be matched ahead of the generic prefix so it can be retried and
        so the operator is not sent to re-login over a working credential.
        """
        marker = credential_failure_marker(CODEX_ALREADY_USED)
        assert marker == "because your refresh token was already used"
        assert credential_failure_is_transient(marker)

    @pytest.mark.parametrize(
        "text",
        [CODEX_EXPIRED, CODEX_REVOKED, CODEX_OTHER_ACCOUNT, CODEX_BARE,
         CODEX_GET_ACCOUNT, CODEX_NO_ACCOUNT, OPENCODE_REFRESH_FAILED,
         CLAUDE_OAUTH_REFRESH, CLAUDE_INVALID_KEY],
    )
    def test_every_other_wording_is_not_transient(self, text: str) -> None:
        marker = credential_failure_marker(text)
        assert marker is not None
        assert not credential_failure_is_transient(marker)

    def test_embedded_in_a_larger_stream_is_still_found(self) -> None:
        stream = "\n".join(
            ["running 3 tests", CODEX_ALREADY_USED, "process exited"]
        )
        assert credential_failure_marker(stream) is not None


class TestExceptionTaxonomy:
    def test_credential_error_is_a_helix_error(self) -> None:
        assert isinstance(CredentialRefreshError("x"), HelixError)

    def test_credential_error_is_not_a_mutation_error(self) -> None:
        """``except MutationError`` in mutate() must not swallow it as a
        failed mutation -- that is the exact conflation being removed."""
        assert not isinstance(CredentialRefreshError("x"), MutationError)

    def test_credential_error_is_not_a_rate_limit(self) -> None:
        """A rate limit clears on its own; an unusable credential does not."""
        assert not isinstance(CredentialRefreshError("x"), RateLimitError)


def _patch_backend(
    mocker: Any, *, returncode: int, stdout: str = "", stderr: str = ""
) -> None:
    result = subprocess.CompletedProcess(
        args=["backend"], returncode=returncode, stdout=stdout, stderr=stderr
    )
    mocker.patch("helix.mutator.subprocess.run", return_value=result)


def _jsonl(*events: dict[str, Any]) -> str:
    return "\n".join(json.dumps(event) for event in events)


# The real shapes, read from the CLIs' own event definitions:
#   codex     ``ThreadErrorEvent`` / ``TurnFailedEvent`` in
#             codex-rs/exec/src/exec_events.rs -- no ``is_error`` key exists
#             anywhere in ``codex exec --json`` output.
#   opencode  ``emit("error", { error: props.error })`` in cli/cmd/run.ts,
#             where ``props.error`` is ``{ name, data: { message } }``.
def _codex_error_event(message: str) -> dict[str, Any]:
    return {"type": "error", "message": message}


def _codex_turn_failed(message: str) -> dict[str, Any]:
    return {"type": "turn.failed", "error": {"message": message}}


def _opencode_error_event(message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "timestamp": 1757500000000,
        "sessionID": "ses_x",
        "error": {"name": "UnknownError", "data": {"message": message}},
    }


def _codex_command_output(output: str) -> dict[str, Any]:
    """A completed ``command_execution`` item: the candidate's own tool output."""
    return {
        "type": "item.completed",
        "item": {
            "id": "item_1",
            "type": "command_execution",
            "command": "pytest -q",
            "aggregated_output": output,
            "exit_code": 1,
            "status": "completed",
        },
    }


def _codex_agent_message(text: str) -> dict[str, Any]:
    return {
        "type": "item.completed",
        "item": {"id": "item_2", "type": "agent_message", "text": text},
    }


class TestInvocationClassification:
    def test_non_zero_exit_with_cli_wording_on_stderr(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        _patch_backend(mocker, returncode=1, stderr=CODEX_EXPIRED)
        with pytest.raises(CredentialRefreshError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )
        err = exc.value
        assert err.exit_code == 1
        assert err.stderr == CODEX_EXPIRED
        assert err.transient is False
        assert "credential" in err.suggestion.lower()
        assert "helix sandbox login codex" in err.suggestion

    def test_zero_exit_codex_error_event_is_read(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Codex swallows its own refresh failure: exit 0, empty stderr.

        Measured on codex-cli 0.130.0 against a synthetic credential whose
        refresh was rejected -- the process exits 0 and prints nothing on
        stderr, even at RUST_LOG=info.  What it does emit is its own
        ``{"type": "error", "message": ...}`` event, so that is what has to
        be read.
        """
        stream = _jsonl(
            {"type": "thread.started", "thread_id": "t1"},
            _codex_error_event(CODEX_ALREADY_USED),
        )
        _patch_backend(mocker, returncode=0, stdout=stream)
        with pytest.raises(CredentialRefreshError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )
        assert exc.value.exit_code == 0
        assert exc.value.transient is True
        assert "structured error event" in str(exc.value)

    def test_zero_exit_codex_turn_failed_is_read(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        stream = _jsonl(
            {"type": "thread.started", "thread_id": "t1"},
            {"type": "turn.started"},
            _codex_turn_failed(CODEX_EXPIRED),
        )
        _patch_backend(mocker, returncode=0, stdout=stream)
        with pytest.raises(CredentialRefreshError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )
        assert exc.value.transient is False

    def test_zero_exit_opencode_error_event_is_read(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        stream = _jsonl(
            {"type": "step_start", "sessionID": "ses_x"},
            _opencode_error_event(OPENCODE_REFRESH_FAILED),
        )
        _patch_backend(mocker, returncode=0, stdout=stream)
        with pytest.raises(CredentialRefreshError):
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="opencode")
            )

    def test_claude_errors_list_is_read(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Claude Code's error envelope carries an ``errors`` list, not a string."""
        envelope = json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "errors": [CLAUDE_OAUTH_REFRESH],
            }
        )
        _patch_backend(mocker, returncode=0, stdout=envelope)
        with pytest.raises(CredentialRefreshError):
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="claude")
            )

    def test_claude_errored_result_is_read(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        envelope = json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "result": CLAUDE_INVALID_KEY,
            }
        )
        _patch_backend(mocker, returncode=0, stdout=envelope)
        with pytest.raises(CredentialRefreshError):
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="claude")
            )

    def test_claude_successful_result_prose_is_not_read(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """On a successful turn ``result`` is the assistant's own words."""
        envelope = json.dumps(
            {
                "type": "result",
                "subtype": "success",
                "is_error": False,
                "result": f"I fixed the handler that raised '{CLAUDE_INVALID_KEY}'.",
            }
        )
        _patch_backend(mocker, returncode=0, stdout=envelope)
        parsed, _usage = invoke_claude_code(
            str(tmp_path), "p", AgentConfig(backend="claude")
        )
        assert parsed["subtype"] == "success"

    def test_tool_result_is_not_evidence(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """A failing tool call is the agent's own failing command -- an
        ordinary code failure -- whatever flag the backend puts on it."""
        stream = _jsonl(
            {
                "type": "tool_result",
                "is_error": True,
                "content": f"pytest exited 1: {CODEX_ALREADY_USED}",
            },
            {"type": "turn.completed"},
        )
        _patch_backend(mocker, returncode=0, stdout=stream)
        parsed, _usage = invoke_claude_code(
            str(tmp_path), "p", AgentConfig(backend="codex")
        )
        assert parsed["events"]

    def test_agent_prose_with_marker_on_zero_exit_is_not_classified(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """The agent quoting the CLI's own wording is still prose."""
        stream = _jsonl(
            {"type": "thread.started", "thread_id": "t1"},
            _codex_agent_message(
                f"The old client printed '{CODEX_ALREADY_USED}'; I removed it."
            ),
            {"type": "turn.completed", "usage": {"input_tokens": 1}},
        )
        _patch_backend(mocker, returncode=0, stdout=stream)
        parsed, _usage = invoke_claude_code(
            str(tmp_path), "p", AgentConfig(backend="codex")
        )
        assert parsed["events"]

    def test_candidate_editing_oauth_code_is_not_classified(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """A candidate whose own work is about refresh tokens must not be able
        to talk HELIX into declaring the operator's login broken."""
        stream = _jsonl(
            _codex_command_output(
                "FAILED tests/test_oauth.py::test_reuse - "
                "expected the refresh token to be rejected"
            ),
            {"type": "turn.completed"},
        )
        _patch_backend(mocker, returncode=0, stdout=stream)
        parsed, _usage = invoke_claude_code(
            str(tmp_path), "p", AgentConfig(backend="codex")
        )
        assert parsed["events"]

    def test_non_zero_exit_never_scans_tool_output_in_the_transcript(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Exit 1 on a 502 after the candidate's tests printed opencode's
        refresh wording into a command's output is a generic failure."""
        stream = _jsonl(
            {"type": "thread.started", "thread_id": "t1"},
            _codex_command_output(f"E   RuntimeError: {OPENCODE_REFRESH_FAILED}"),
        )
        _patch_backend(
            mocker, returncode=1, stdout=stream, stderr="error: 502 Bad Gateway"
        )
        with pytest.raises(MutationError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )
        assert not isinstance(exc.value, CredentialRefreshError)

    def test_non_zero_exit_never_scans_raw_stdout_prose(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        _patch_backend(
            mocker,
            returncode=1,
            stdout=f"note: {CODEX_EXPIRED}",
            stderr="Traceback (most recent call last): SyntaxError",
        )
        with pytest.raises(MutationError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )
        assert not isinstance(exc.value, CredentialRefreshError)

    @pytest.mark.parametrize("is_error", [False, True])
    def test_claude_max_turns_with_auth_wording_in_prose_stays_partial_success(
        self, mocker: Any, tmp_path: Path, is_error: bool
    ) -> None:
        """Pre-existing contract: max-turns exhaustion is partial success
        because the edits may be useful.  It is decided before any credential
        check, so the assistant's prose in ``result`` is never read."""
        envelope = json.dumps(
            {
                "type": "result",
                "subtype": "error_max_turns",
                "is_error": is_error,
                "num_turns": 30,
                "result": f"I hit '{CLAUDE_INVALID_KEY}' in the old test fixture.",
            }
        )
        _patch_backend(mocker, returncode=1, stdout=envelope)
        parsed, _usage = invoke_claude_code(
            str(tmp_path), "p", AgentConfig(backend="claude")
        )
        assert parsed["subtype"] == "error_max_turns"

    def test_ordinary_non_zero_exit_stays_a_mutation_error(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        _patch_backend(
            mocker,
            returncode=1,
            stderr="Traceback (most recent call last): SyntaxError",
        )
        with pytest.raises(MutationError):
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )

    def test_rate_limit_still_wins_its_own_classification(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        _patch_backend(
            mocker, returncode=1, stderr="Error: 529 overloaded please retry"
        )
        with pytest.raises(RateLimitError):
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )

    def test_invoke_itself_never_retries(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """The retry needs a fresh worktree, which only the caller owns."""
        run = mocker.patch(
            "helix.mutator.subprocess.run",
            return_value=_completed(1, stderr=CODEX_ALREADY_USED),
        )
        with pytest.raises(CredentialRefreshError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )
        assert run.call_count == 1
        assert exc.value.transient is True
        assert "retry" not in str(exc.value).lower()

    def test_retried_flag_changes_the_second_loss_wording(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        _patch_backend(mocker, returncode=1, stderr=CODEX_ALREADY_USED)
        with pytest.raises(CredentialRefreshError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex"), retried=True
            )
        assert "failed again on retry" in str(exc.value)
        assert "helix resume" in exc.value.suggestion
        assert "sandbox login" not in exc.value.suggestion


class TestRateLimitClassification:
    """A rate limit is what the *backend* said, never what the agent read.

    The same conflation the credential markers avoid applied to the rate-limit
    check, which scanned ``stderr or stdout``: for the JSONL backends stdout is
    the whole transcript, so a candidate working on a repository whose source
    contains the words "rate limit" -- this one does -- plus any unrelated
    non-zero exit with an empty stderr was reported to the operator as a quota
    failure and sent to a dashboard.
    """

    def test_transcript_tool_output_mentioning_a_rate_limit_is_not_one(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Exit 1 with an empty stderr, after the candidate read this repo's
        own rate-limit keywords into a command's output, is a code failure."""
        stream = _jsonl(
            {"type": "thread.started", "thread_id": "t1"},
            _codex_command_output(
                "src/helix/mutator.py:599:_RATE_LIMIT_KEYWORDS = [\n"
                '    "rate limit",\n    "overloaded",\n    "529",\n]'
            ),
        )
        _patch_backend(mocker, returncode=1, stdout=stream, stderr="")
        with pytest.raises(MutationError) as exc:
            invoke_claude_code(str(tmp_path), "p", AgentConfig(backend="codex"))
        assert not isinstance(exc.value, RateLimitError)

    def test_a_real_rate_limit_on_stderr_is_unchanged(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """stderr is the CLI's own channel and is always in scope."""
        _patch_backend(
            mocker,
            returncode=1,
            stdout=_jsonl({"type": "thread.started", "thread_id": "t1"}),
            stderr="stream error: rate limit exceeded; retry after 60s",
        )
        with pytest.raises(RateLimitError):
            invoke_claude_code(str(tmp_path), "p", AgentConfig(backend="codex"))

    @pytest.mark.parametrize(
        ("backend", "event"),
        [
            ("codex", _codex_error_event("stream error: rate limit exceeded")),
            ("codex", _codex_turn_failed("rate limit exceeded; retry after 60s")),
            ("opencode", _opencode_error_event("Overloaded: retry later")),
        ],
    )
    def test_structured_error_event_wording_is_a_rate_limit(
        self, mocker: Any, tmp_path: Path, backend: str, event: dict[str, Any]
    ) -> None:
        """The backend's own error event counts even when stderr says something
        unrelated -- which used to hide it, since stdout was read only when
        stderr was empty."""
        _patch_backend(
            mocker,
            returncode=1,
            stdout=_jsonl({"type": "thread.started", "thread_id": "t1"}, event),
            stderr="error: exit status 1",
        )
        with pytest.raises(RateLimitError):
            invoke_claude_code(str(tmp_path), "p", AgentConfig(backend=backend))

    def test_claude_api_error_status_is_a_rate_limit(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Claude reports the HTTP status structurally; 429 appears in no prose."""
        envelope = json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "api_error_status": 429,
                "result": "The request failed.",
            }
        )
        _patch_backend(mocker, returncode=1, stdout=envelope, stderr="")
        with pytest.raises(RateLimitError):
            invoke_claude_code(str(tmp_path), "p", AgentConfig(backend="claude"))

    def test_claude_errored_envelope_wording_is_a_rate_limit(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Regression guard: the envelope's ``errors`` list and, when it flags
        ``is_error``, its ``result`` are structured fields, so they stay in
        scope now that the raw stream is not."""
        envelope = json.dumps(
            {
                "type": "result",
                "subtype": "error_during_execution",
                "is_error": True,
                "errors": ["API Error: 429 rate limit exceeded"],
            }
        )
        _patch_backend(mocker, returncode=1, stdout=envelope, stderr="")
        with pytest.raises(RateLimitError):
            invoke_claude_code(str(tmp_path), "p", AgentConfig(backend="claude"))

    def test_a_dead_stream_still_has_its_stdout_read(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """The one carve-out: a CLI that emitted no parseable output at all and
        nothing on stderr has said it nowhere else."""
        _patch_backend(
            mocker,
            returncode=1,
            stdout="error: rate limit exceeded for this organization",
            stderr="",
        )
        with pytest.raises(RateLimitError):
            invoke_claude_code(str(tmp_path), "p", AgentConfig(backend="codex"))


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["backend"], returncode=returncode, stdout=stdout, stderr=stderr
    )


def _codex_stream(*, session: str, input_tokens: int, output_tokens: int) -> str:
    return _jsonl(
        {"type": "session.started", "session_id": session},
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            },
        },
    )


CODEX_SUCCESS_STREAM = _codex_stream(session="won", input_tokens=12, output_tokens=8)
CODEX_LOST_STREAM = _codex_stream(session="lost", input_tokens=5, output_tokens=2)


def _codex_config() -> HelixConfig:
    return HelixConfig(
        objective="Improve the code",
        evaluator=EvaluatorConfig(command="true"),
        agent=AgentConfig(backend="codex"),
    )


def _candidate(cid: str, path: Path) -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=str(path),
        branch_name=f"helix/{cid}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


class _RetryHarness:
    """``mutate()`` with a clone that hands out a fresh directory per call.

    Each ``clone_candidate`` call creates ``<tmp>/clone<N>`` so the test can
    see which tree each attempt ran in, what it left behind, and which trees
    were removed.
    """

    def __init__(self, tmp_path: Path, mocker: Any) -> None:
        self.tmp_path = tmp_path
        self.clones: list[Candidate] = []
        self.removed: list[Candidate] = []
        self.run_cwds: list[str] = []
        self.spent: list[UsageStats] = []
        self.wasted: list[UsageStats] = []
        self.recovered: list[str] = []
        self.parent = _candidate("g0-s0", tmp_path / "parent")

        def _clone(parent: Candidate, new_id: str, base_dir: Path) -> Candidate:
            path = tmp_path / f"clone{len(self.clones) + 1}"
            path.mkdir()
            child = _candidate(new_id, path)
            self.clones.append(child)
            return child

        mocker.patch("helix.mutator.clone_candidate", side_effect=_clone)
        mocker.patch(
            "helix.mutator.remove_worktree", side_effect=self.removed.append
        )
        mocker.patch("helix.mutator.snapshot_candidate")
        self.mocker = mocker

    def run_backend(self, outcomes: list[Any]) -> Any:
        """Patch ``subprocess.run`` with per-attempt outcomes.

        An outcome is a ``CompletedProcess`` to return or an exception to
        raise.  Every attempt first drops ``partial.py`` into its cwd, the way
        a backend that lost mid-turn leaves half-applied edits behind.
        """
        outcomes = list(outcomes)

        def _run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            cwd = kwargs["cwd"]
            self.run_cwds.append(cwd)
            (Path(cwd) / "partial.py").write_text("# half-applied edit\n")
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        return self.mocker.patch("helix.mutator.subprocess.run", side_effect=_run)

    def mutate(self, *, split_waste: bool = True) -> Candidate | None:
        """Run the mutation.

        ``split_waste`` mirrors what ``evolution.py`` does: it supplies the
        second sink, so an attempt lost to a refresh race is reported apart
        from the work that was actually kept.  Pass ``False`` to exercise the
        fallback a caller without a second sink gets.
        """
        return mutate(
            self.parent,
            make_eval_result("g0-s0"),
            "g1-s0",
            _codex_config(),
            self.tmp_path,
            record_usage=self.spent.append,
            record_wasted_usage=self.wasted.append if split_waste else None,
            on_refresh_race_recovered=self.recovered.append,
        )


class TestLostRefreshRaceIsRetried:
    """The already-used outcome is the race this work exists to handle.

    When it happens, the *winner* has just written a refreshed credential to
    the shared volume, so the right response is to invoke again against it,
    not to drop the slot and tell the operator to redo a login that is fine.
    The retry lives in ``mutate()`` / ``merge()`` rather than in
    ``invoke_claude_code`` because it must start from a fresh worktree.
    """

    def test_retry_runs_in_a_fresh_worktree(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Attempt 1's half-applied edits must not be under the retry."""
        h = _RetryHarness(tmp_path, mocker)
        run = h.run_backend(
            [
                _completed(1, stdout=CODEX_LOST_STREAM, stderr=CODEX_ALREADY_USED),
                _completed(0, stdout=CODEX_SUCCESS_STREAM),
            ]
        )

        child = h.mutate()

        assert run.call_count == 2
        assert len(h.clones) == 2
        assert h.run_cwds == [h.clones[0].worktree_path, h.clones[1].worktree_path]
        assert child is h.clones[1]
        # The first tree was discarded before the retry, and the retry's tree
        # was clean when the backend started in it.
        assert h.removed == [h.clones[0]]
        retry_tree = Path(h.clones[1].worktree_path)
        assert sorted(p.name for p in retry_tree.iterdir() if p.name == "partial.py") == [
            "partial.py"
        ], "only the retry's own edit is present"
        assert not (retry_tree / ".helix_backend_result.attempt1.json").read_text().count(
            "partial"
        )

    def test_both_attempts_artifacts_are_kept(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        h = _RetryHarness(tmp_path, mocker)
        h.run_backend(
            [
                _completed(1, stdout=CODEX_LOST_STREAM, stderr=CODEX_ALREADY_USED),
                _completed(0, stdout=CODEX_SUCCESS_STREAM),
            ]
        )

        child = h.mutate()
        assert child is not None
        tree = Path(child.worktree_path)

        # Attempt 1, under its own names, with the marker that explains why
        # there was a retry.
        assert (tree / ".helix_backend_stderr.attempt1.txt").read_text() == CODEX_ALREADY_USED
        assert (tree / ".helix_backend_stdout.attempt1.txt").read_text() == CODEX_LOST_STREAM
        first = json.loads((tree / ".helix_backend_result.attempt1.json").read_text())
        assert first["returncode"] == 1
        assert first["usage"]["input_tokens"] == 5

        # The final attempt's artifact accounts for both.
        final = json.loads((tree / ".helix_backend_result.json").read_text())
        assert final["returncode"] == 0
        assert final["attempts"] == 2
        assert final["retry_of"] == ".helix_backend_result.attempt1.json"
        assert final["usage_first_attempt"]["input_tokens"] == 5
        assert final["usage_combined"]["input_tokens"] == 5 + 12
        assert final["usage_combined"]["output_tokens"] == 2 + 8

        # And none of it can leak into the candidate's git tree.
        gitignore = (tree / ".gitignore").read_text()
        for name in (
            ".helix_backend_result.attempt1.json",
            ".helix_backend_stdout.attempt1.txt",
            ".helix_backend_stderr.attempt1.txt",
        ):
            assert name in gitignore

    def test_recovered_race_reports_the_lost_attempt_apart_from_the_kept_one(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Both attempts are reported once — but not as the same spend.

        Folding them together made the lost attempt look like part of a
        productive mutation.  The two sinks split it: the retry's tokens are
        the mutation's, the lost attempt's are their own record, and the sum
        is what it always was.
        """
        h = _RetryHarness(tmp_path, mocker)
        h.run_backend(
            [
                _completed(1, stdout=CODEX_LOST_STREAM, stderr=CODEX_ALREADY_USED),
                _completed(0, stdout=CODEX_SUCCESS_STREAM),
            ]
        )

        child = h.mutate()
        assert child is not None

        # The kept work is billed for its own tokens only.
        assert len(h.spent) == 1
        assert h.spent[0].input_tokens == 12
        assert h.spent[0].output_tokens == 8
        assert h.spent[0].session_id == "won"
        assert child.usage.input_tokens == 12
        # The thrown-away attempt is reported once, separately.
        assert len(h.wasted) == 1
        assert h.wasted[0].input_tokens == 5
        assert h.wasted[0].output_tokens == 2
        # And the total is unchanged.
        assert h.spent[0].input_tokens + h.wasted[0].input_tokens == 5 + 12
        assert h.spent[0].output_tokens + h.wasted[0].output_tokens == 2 + 8
        # The race is visible to whoever keeps the end-of-run summary.
        assert len(h.recovered) == 1
        assert "recovered after a lost refresh race" in h.recovered[0]
        assert "g1-s0" in h.recovered[0]

    def test_without_a_waste_sink_the_first_attempt_still_rides_along(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """A caller with nowhere to put the second record must not lose it.

        Separating the two spends is an attribution improvement; it must
        never become a way for tokens to go uncharged.
        """
        h = _RetryHarness(tmp_path, mocker)
        h.run_backend(
            [
                _completed(1, stdout=CODEX_LOST_STREAM, stderr=CODEX_ALREADY_USED),
                _completed(0, stdout=CODEX_SUCCESS_STREAM),
            ]
        )

        child = h.mutate(split_waste=False)
        assert child is not None
        assert h.wasted == []
        assert len(h.spent) == 1
        assert h.spent[0].input_tokens == 5 + 12
        assert h.spent[0].output_tokens == 2 + 8
        assert child.usage.input_tokens == 5 + 12

    def test_zero_exit_codex_error_event_is_retried_too(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Codex swallows the failure on exit 0; the event path retries as well."""
        h = _RetryHarness(tmp_path, mocker)
        run = h.run_backend(
            [
                _completed(
                    0,
                    stdout=_jsonl(
                        {"type": "thread.started", "thread_id": "t1"},
                        _codex_error_event(CODEX_ALREADY_USED),
                    ),
                ),
                _completed(0, stdout=CODEX_SUCCESS_STREAM),
            ]
        )
        assert h.mutate() is h.clones[1]
        assert run.call_count == 2

    def test_second_loss_charges_both_attempts_once_and_no_relogin_instruction(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        h = _RetryHarness(tmp_path, mocker)
        run = h.run_backend(
            [
                _completed(1, stdout=CODEX_LOST_STREAM, stderr=CODEX_ALREADY_USED),
                _completed(
                    1,
                    stdout=_codex_stream(session="lost-2", input_tokens=3, output_tokens=1),
                    stderr=CODEX_ALREADY_USED,
                ),
            ]
        )
        with pytest.raises(CredentialRefreshError) as exc:
            h.mutate()
        err = exc.value
        assert run.call_count == 2  # exactly one retry, no loop
        assert err.transient is True
        assert "failed again on retry" in str(err)
        # The stored credential is the winner's fresh one; do not tell the
        # operator to throw it away.
        assert "sandbox login" not in err.suggestion
        assert "helix resume" in err.suggestion
        # Each attempt's spend is reported exactly once: the retry's on the
        # error it raised, the lost first attempt's on its own sink.  Their
        # sum is what a single combined record used to carry.
        assert err.usage is not None
        assert err.usage.input_tokens == 3
        assert err.usage.output_tokens == 1
        assert err.usage.session_id == "lost-2"
        assert h.spent == [err.usage]
        assert len(h.wasted) == 1
        assert h.wasted[0].input_tokens == 5
        assert h.wasted[0].output_tokens == 2
        assert err.usage.input_tokens + h.wasted[0].input_tokens == 5 + 3
        assert err.usage.output_tokens + h.wasted[0].output_tokens == 2 + 1
        # Neither worktree leaks.
        assert h.removed == h.clones
        assert h.recovered == []

    def test_retry_failing_for_another_reason_still_charges_both_attempts(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """The retry's error class does not matter; neither spend goes missing."""
        h = _RetryHarness(tmp_path, mocker)
        h.run_backend(
            [
                _completed(1, stdout=CODEX_LOST_STREAM, stderr=CODEX_ALREADY_USED),
                _completed(
                    2,
                    stdout=_codex_stream(session="crashed", input_tokens=4, output_tokens=0),
                    stderr="segfault",
                ),
            ]
        )
        assert h.mutate() is None  # MutationError -> None, by contract
        assert len(h.spent) == 1
        assert h.spent[0].input_tokens == 4
        assert h.spent[0].output_tokens == 0
        assert len(h.wasted) == 1
        assert h.wasted[0].input_tokens == 5
        assert h.wasted[0].output_tokens == 2
        assert h.spent[0].input_tokens + h.wasted[0].input_tokens == 5 + 4
        assert h.spent[0].output_tokens + h.wasted[0].output_tokens == 2
        assert h.removed == h.clones

    def test_timeout_on_the_retry_records_the_wasted_spend_and_cleans_up(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """A sandbox ``TimeoutExpired`` is not a HelixError and carries no
        usage: the first attempt's spend must still be reported — as waste,
        since nothing productive came of it — and the retry's worktree must
        not leak."""
        h = _RetryHarness(tmp_path, mocker)
        h.run_backend(
            [
                _completed(1, stdout=CODEX_LOST_STREAM, stderr=CODEX_ALREADY_USED),
                subprocess.TimeoutExpired(cmd=["codex"], timeout=30),
            ]
        )
        with pytest.raises(subprocess.TimeoutExpired):
            h.mutate()
        assert h.spent == []
        assert len(h.wasted) == 1
        assert h.wasted[0].input_tokens == 5
        assert h.wasted[0].output_tokens == 2
        assert h.removed == h.clones

    def test_timeout_on_the_first_attempt_cleans_up(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        h = _RetryHarness(tmp_path, mocker)
        h.run_backend([subprocess.TimeoutExpired(cmd=["codex"], timeout=30)])
        with pytest.raises(subprocess.TimeoutExpired):
            h.mutate()
        assert h.spent == []
        assert h.removed == h.clones == h.clones[:1]

    @pytest.mark.parametrize("text", [CODEX_EXPIRED, CODEX_REVOKED, CODEX_BARE])
    def test_a_dead_login_is_not_retried(
        self, mocker: Any, tmp_path: Path, text: str
    ) -> None:
        """Retrying an expired or revoked credential only burns a turn."""
        h = _RetryHarness(tmp_path, mocker)
        run = h.run_backend([_completed(1, stdout=CODEX_LOST_STREAM, stderr=text)])
        with pytest.raises(CredentialRefreshError) as exc:
            h.mutate()
        assert run.call_count == 1
        assert len(h.clones) == 1
        assert exc.value.transient is False
        assert "helix sandbox login codex" in exc.value.suggestion
        # Item: the credential path records usage like its siblings.
        assert len(h.spent) == 1
        assert h.spent[0].input_tokens == 5
        # No retry happened, so nothing was wasted on a lost race.
        assert h.wasted == []
        assert h.removed == h.clones


class TestRetryOnARealWorktree:
    """The same contract against real git worktrees, not a stubbed clone."""

    def test_retry_starts_from_the_parent_commit(
        self, mocker: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from helix.worktree import create_seed_worktree

        for key, value in {
            "GIT_AUTHOR_NAME": "HELIX Test",
            "GIT_AUTHOR_EMAIL": "helix@test.local",
            "GIT_COMMITTER_NAME": "HELIX Test",
            "GIT_COMMITTER_EMAIL": "helix@test.local",
        }.items():
            monkeypatch.setenv(key, value)
        project = tmp_path / "project"
        project.mkdir()
        (project / "main.py").write_text("print('hello')\n")
        base_dir = tmp_path / "worktrees"
        seed = create_seed_worktree(project, base_dir)

        seen: list[dict[str, Any]] = []
        outcomes = [
            _completed(1, stdout=CODEX_LOST_STREAM, stderr=CODEX_ALREADY_USED),
            _completed(0, stdout=CODEX_SUCCESS_STREAM),
        ]

        real_run = subprocess.run

        def _run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            if args[0] != "codex":
                # git, driven by the real clone_candidate / remove_worktree
                return real_run(args, **kwargs)
            cwd = Path(kwargs["cwd"])
            seen.append(
                {
                    "cwd": str(cwd),
                    "partial_present": (cwd / "partial.py").exists(),
                    "main": (cwd / "main.py").read_text(),
                }
            )
            (cwd / "partial.py").write_text("# half-applied\n")
            (cwd / "main.py").write_text("print('half edited')\n")
            return outcomes.pop(0)

        mocker.patch("helix.mutator.subprocess.run", side_effect=_run)
        recovered: list[str] = []
        child = mutate(
            seed,
            make_eval_result(seed.id),
            "g1-s0",
            _codex_config(),
            base_dir,
            on_refresh_race_recovered=recovered.append,
        )

        assert child is not None
        assert len(seen) == 2
        assert seen[0]["cwd"] == seen[1]["cwd"] == str(base_dir / "g1-s0")
        # The retry saw the parent's tree, not attempt 1's half-applied edits.
        assert seen[1]["partial_present"] is False
        assert seen[1]["main"] == "print('hello')\n"
        tree = Path(child.worktree_path)
        assert (tree / ".helix_backend_result.attempt1.json").is_file()
        assert (tree / ".helix_backend_result.json").is_file()
        assert recovered and "recovered after a lost refresh race" in recovered[0]
