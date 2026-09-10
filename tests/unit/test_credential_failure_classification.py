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

from helix.config import AgentConfig
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
)


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

    def test_zero_exit_is_error_envelope_is_read(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Codex swallows its own refresh failure: exit 0, empty stderr.

        Measured on codex-cli 0.130.0 against a synthetic credential whose
        refresh was rejected -- the process exits 0 and prints nothing, even at
        RUST_LOG=info.  The envelope's ``is_error`` flag is the only signal
        left, so it has to be read.
        """
        stream = "\n".join(
            [
                json.dumps({"type": "thread.started"}),
                json.dumps({"type": "error", "is_error": True,
                            "message": CODEX_ALREADY_USED}),
            ]
        )
        _patch_backend(mocker, returncode=0, stdout=stream)
        with pytest.raises(CredentialRefreshError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )
        assert exc.value.exit_code == 0
        assert "is_error" in str(exc.value)

    def test_claude_top_level_envelope_is_read(
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

    def test_is_error_alone_does_not_classify(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """An ``is_error`` tool result is usually the agent's own failing
        command.  That is an ordinary code failure and must stay one."""
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_result",
                        "is_error": True,
                        "content": "pytest exited 1: 2 failed, 9 passed",
                    }
                ),
                json.dumps({"type": "turn.completed"}),
            ]
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
        stream = "\n".join(
            [
                json.dumps(
                    {
                        "type": "tool_result",
                        "is_error": True,
                        "content": (
                            "FAILED tests/test_oauth.py::test_reuse - "
                            "expected the refresh token to be rejected"
                        ),
                    }
                )
            ]
        )
        _patch_backend(mocker, returncode=0, stdout=stream)
        parsed, _usage = invoke_claude_code(
            str(tmp_path), "p", AgentConfig(backend="codex")
        )
        assert parsed["events"]

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


def _completed(
    returncode: int, stdout: str = "", stderr: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["backend"], returncode=returncode, stdout=stdout, stderr=stderr
    )


CODEX_SUCCESS_STREAM = json.dumps({"type": "turn.completed"})


class TestLostRefreshRaceIsRetried:
    """The already-used outcome is the race this work exists to handle.

    When it happens, the *winner* has just written a refreshed credential to
    the shared volume, so the right response is to invoke again against it,
    not to drop the slot and tell the operator to redo a login that is fine.
    """

    def test_already_used_is_retried_once_and_the_retry_can_succeed(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        run = mocker.patch(
            "helix.mutator.subprocess.run",
            side_effect=[
                _completed(1, stderr=CODEX_ALREADY_USED),
                _completed(0, stdout=CODEX_SUCCESS_STREAM),
            ],
        )
        parsed, _usage = invoke_claude_code(
            str(tmp_path), "p", AgentConfig(backend="codex")
        )
        assert parsed["events"]
        assert run.call_count == 2

    def test_zero_exit_already_used_envelope_is_retried_too(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Codex swallows the failure on exit 0; the envelope path retries as well."""
        stream = "\n".join(
            [
                json.dumps({"type": "thread.started"}),
                json.dumps({"type": "error", "is_error": True,
                            "message": CODEX_ALREADY_USED}),
            ]
        )
        run = mocker.patch(
            "helix.mutator.subprocess.run",
            side_effect=[
                _completed(0, stdout=stream),
                _completed(0, stdout=CODEX_SUCCESS_STREAM),
            ],
        )
        parsed, _usage = invoke_claude_code(
            str(tmp_path), "p", AgentConfig(backend="codex")
        )
        assert parsed["events"]
        assert run.call_count == 2

    def test_second_loss_is_raised_without_a_relogin_instruction(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        run = mocker.patch(
            "helix.mutator.subprocess.run",
            side_effect=[
                _completed(1, stderr=CODEX_ALREADY_USED),
                _completed(1, stderr=CODEX_ALREADY_USED),
            ],
        )
        with pytest.raises(CredentialRefreshError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )
        err = exc.value
        assert run.call_count == 2  # exactly one retry, no loop
        assert err.transient is True
        assert "retry" in str(err).lower()
        # The stored credential is the winner's fresh one; do not tell the
        # operator to throw it away.
        assert "sandbox login" not in err.suggestion
        assert "helix resume" in err.suggestion

    @pytest.mark.parametrize("text", [CODEX_EXPIRED, CODEX_REVOKED, CODEX_BARE])
    def test_a_dead_login_is_not_retried(
        self, mocker: Any, tmp_path: Path, text: str
    ) -> None:
        """Retrying an expired or revoked credential only burns a turn."""
        run = mocker.patch(
            "helix.mutator.subprocess.run",
            side_effect=[_completed(1, stderr=text)],
        )
        with pytest.raises(CredentialRefreshError) as exc:
            invoke_claude_code(
                str(tmp_path), "p", AgentConfig(backend="codex")
            )
        assert run.call_count == 1
        assert exc.value.transient is False
        assert "helix sandbox login codex" in exc.value.suggestion
