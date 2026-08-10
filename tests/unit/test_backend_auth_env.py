"""Anthropic API keys are never auto-forwarded into agent subprocesses.

Anthropic backends authenticate by *login*: ``helix sandbox login <backend>``
writes credentials into the persistent ``helix-auth-<backend>`` Docker volume
that agent containers mount at ``/home/node``.  An ``ANTHROPIC_API_KEY`` in the
agent's environment is the mutually exclusive alternative — when present, the
CLI bills the API key and the login credential is ignored.

HELIX used to forward that variable automatically, which meant a user who had
just run ``helix sandbox login claude`` had their login silently revoked by a
key merely sitting in their shell.  These tests pin the corrected behaviour:
the login wins by default, and the API key travels only on an explicit opt-in.
"""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helix.config import AgentConfig, SandboxConfig
from helix.exceptions import MutationError
from helix.mutator import invoke_claude_code
import helix.mutator as mutator


CLAUDE_STDOUT = '{"type":"system","subtype":"init","session_id":"sess_123"}\n'
OPENCODE_STDOUT = '{"type":"result","sessionID":"ses_abc"}\n'


@pytest.fixture(autouse=True)
def _reset_warn_state():
    """The 'key not forwarded' warning is emitted once per backend per run."""
    mutator._ANTHROPIC_KEY_WARNED.clear()
    yield
    mutator._ANTHROPIC_KEY_WARNED.clear()


@pytest.fixture
def local_run(mocker):
    """Patch the unsandboxed subprocess branch and expose the env it received."""

    def _run(stdout: str = CLAUDE_STDOUT):
        mock = mocker.patch("helix.mutator.subprocess.run")
        mock.return_value = MagicMock(stdout=stdout, stderr="", returncode=0)
        return mock

    return _run


def _env_of(mock) -> dict[str, str]:
    return mock.call_args.kwargs["env"]


# ---------------------------------------------------------------------------
# Default: not forwarded
# ---------------------------------------------------------------------------


class TestAnthropicKeyNotForwardedByDefault:
    def test_claude_login_environment_keeps_identity_and_excludes_api_credentials(
        self, tmp_path: Path, monkeypatch, local_run
    ):
        """The scrubbed Claude environment can resolve a stored login.

        On macOS, Claude Code's Keychain lookup needs ``USER``.  This asserts
        the environment that reaches the real Claude subprocess has the
        complete non-secret login identity (rather than merely checking a
        configuration allowlist), while both API credential overrides remain
        absent.
        """
        mock_run = local_run()
        monkeypatch.setenv("HOME", "/home/login-user")
        monkeypatch.setenv("USER", "login-user")
        monkeypatch.setenv("LOGNAME", "login-logname")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-travel")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "token-should-not-travel")

        invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="claude"))

        env = _env_of(mock_run)
        assert env["HOME"] == "/home/login-user"
        assert env["USER"] == "login-user"
        assert env["LOGNAME"] == "login-logname"
        assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} & set(env)

    def test_default_login_identity_does_not_clobber_explicit_passthrough(
        self, tmp_path: Path, monkeypatch, local_run
    ):
        """Configured passthrough values coexist with the default identity."""
        mock_run = local_run()
        monkeypatch.setenv("USER", "login-user")
        monkeypatch.setenv("HELIX_LOGIN_TEST_SETTING", "preserve-me")

        invoke_claude_code(
            str(tmp_path),
            "prompt",
            AgentConfig(backend="claude"),
            passthrough_env=["HELIX_LOGIN_TEST_SETTING"],
        )

        env = _env_of(mock_run)
        assert env["USER"] == "login-user"
        assert env["HELIX_LOGIN_TEST_SETTING"] == "preserve-me"

    @pytest.mark.parametrize("key", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
    def test_claude_unsandboxed_does_not_forward(
        self, key: str, tmp_path: Path, monkeypatch, local_run
    ):
        mock_run = local_run()
        monkeypatch.setenv(key, "sk-ant-should-not-travel")

        invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="claude"))

        assert key not in _env_of(mock_run), (
            f"{key} was forwarded to the claude agent; Anthropic backends must "
            "authenticate through `helix sandbox login claude` unless the user "
            "opted the key in explicitly"
        )

    def test_opencode_drops_anthropic_key_but_keeps_openai(
        self, tmp_path: Path, monkeypatch, local_run
    ):
        mock_run = local_run(OPENCODE_STDOUT)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-travel")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-authorized")
        monkeypatch.setenv("OPENCODE_API_KEY", "sk-opencode-authorized")

        invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="opencode"))

        env = _env_of(mock_run)
        assert "ANTHROPIC_API_KEY" not in env
        # The OpenAI path is authorized and must keep working.
        assert env["OPENAI_API_KEY"] == "sk-openai-authorized"
        assert env["OPENCODE_API_KEY"] == "sk-opencode-authorized"

    def test_non_anthropic_backends_still_auto_forward(
        self, tmp_path: Path, monkeypatch, local_run
    ):
        """The change is scoped to Anthropic credentials only."""
        mock_run = local_run()
        monkeypatch.setenv("CURSOR_API_KEY", "cursor-key")

        invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="cursor"))

        assert _env_of(mock_run)["CURSOR_API_KEY"] == "cursor-key"


# ---------------------------------------------------------------------------
# The sandboxed branch — the one _add_backend_auth_env never gated
# ---------------------------------------------------------------------------


class TestSandboxedAgentBranch:
    """``_add_backend_auth_env`` runs before the sandbox branch is chosen.

    It has no sandbox gate at all, so the key used to be injected as a
    ``docker run -e`` argument into the very container that mounts the login
    volume.  Both branches must be covered.
    """

    def test_sandboxed_claude_container_gets_no_anthropic_key(
        self, tmp_path: Path, monkeypatch, mocker
    ):
        mock_sandboxed = mocker.patch("helix.mutator.run_sandboxed_command")
        mock_sandboxed.return_value = MagicMock(
            stdout=CLAUDE_STDOUT, stderr="", returncode=0
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-travel")

        invoke_claude_code(
            str(tmp_path),
            "prompt",
            AgentConfig(backend="claude"),
            sandbox=SandboxConfig(enabled=True),
        )

        assert "ANTHROPIC_API_KEY" not in _env_of(mock_sandboxed)

    def test_login_credential_is_not_silently_suppressed(
        self, tmp_path: Path, monkeypatch, mocker
    ):
        """Regression: the contradiction that motivated this change.

        A sandboxed agent run mounts ``helix-auth-claude`` at /home/node — the
        volume ``helix sandbox login claude`` wrote credentials into.  If HELIX
        also injects an ambient ANTHROPIC_API_KEY into that container, the CLI
        prefers the key and the login the user just performed is silently dead.
        The container env must carry no Anthropic API credential, so the
        mounted login credential is the one that gets used.
        """
        mock_sandboxed = mocker.patch("helix.mutator.run_sandboxed_command")
        mock_sandboxed.return_value = MagicMock(
            stdout=CLAUDE_STDOUT, stderr="", returncode=0
        )
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")
        monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "sk-ant-ambient-token")

        invoke_claude_code(
            str(tmp_path),
            "prompt",
            AgentConfig(backend="claude"),
            sandbox=SandboxConfig(enabled=True),
        )

        env = _env_of(mock_sandboxed)
        assert not {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"} & set(env), (
            "an ambient API key reached the container that mounts the login "
            "volume, silently disabling `helix sandbox login claude`"
        )
        # The auth volume is still mounted for the agent scope.
        assert mock_sandboxed.call_args.kwargs["scope"] == "agent"
        assert mock_sandboxed.call_args.kwargs["agent_backend"] == "claude"


# ---------------------------------------------------------------------------
# The escape hatch
# ---------------------------------------------------------------------------


class TestExplicitOptIn:
    def test_fixed_env_opt_in_forwards_the_key(
        self, tmp_path: Path, monkeypatch, local_run
    ):
        """``[env]`` in helix.toml is a deliberate, recorded choice."""
        mock_run = local_run()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        invoke_claude_code(
            str(tmp_path),
            "prompt",
            AgentConfig(backend="claude"),
            fixed_env={"ANTHROPIC_API_KEY": "sk-ant-explicit"},
        )

        assert _env_of(mock_run)["ANTHROPIC_API_KEY"] == "sk-ant-explicit"

    def test_passthrough_env_opt_in_forwards_the_key(
        self, tmp_path: Path, monkeypatch, local_run
    ):
        """``passthrough_env`` names the variable explicitly in helix.toml."""
        mock_run = local_run()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-explicit")

        invoke_claude_code(
            str(tmp_path),
            "prompt",
            AgentConfig(backend="claude"),
            passthrough_env=["ANTHROPIC_API_KEY"],
        )

        assert _env_of(mock_run)["ANTHROPIC_API_KEY"] == "sk-ant-explicit"

    def test_opencode_opt_in_forwards_the_key(
        self, tmp_path: Path, monkeypatch, local_run
    ):
        mock_run = local_run(OPENCODE_STDOUT)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-explicit")

        invoke_claude_code(
            str(tmp_path),
            "prompt",
            AgentConfig(backend="opencode"),
            passthrough_env=["ANTHROPIC_API_KEY"],
        )

        assert _env_of(mock_run)["ANTHROPIC_API_KEY"] == "sk-ant-explicit"


# ---------------------------------------------------------------------------
# Neither outcome is silent
# ---------------------------------------------------------------------------


class TestUserIsTold:
    def test_auth_failure_reports_the_scrubbed_agent_environment(
        self, tmp_path: Path, monkeypatch, mocker
    ):
        """Authentication errors reveal the non-secret env names the agent got."""
        mock_run = mocker.patch("helix.mutator.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="Not logged in · Please run /login",
            returncode=1,
        )
        monkeypatch.setenv("USER", "login-user")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-should-not-travel")

        with pytest.raises(MutationError) as exc_info:
            invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="claude"))

        suggestion = exc_info.value.suggestion
        assert "environment after HELIX scrubbing contained" in suggestion
        assert "USER" in suggestion
        assert "ANTHROPIC_API_KEY" not in suggestion

    def test_dropped_key_is_announced(
        self, tmp_path: Path, monkeypatch, local_run, caplog
    ):
        local_run()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")

        with caplog.at_level(logging.WARNING, logger="helix.mutator"):
            invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="claude"))

        text = caplog.text
        assert "ANTHROPIC_API_KEY" in text
        assert "NOT forwarded" in text
        assert "helix sandbox login claude" in text

    def test_opt_in_precedence_over_login_is_announced(
        self, tmp_path: Path, monkeypatch, local_run, caplog
    ):
        """The both-present case: HELIX says which credential won."""
        local_run()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        with caplog.at_level(logging.WARNING, logger="helix.mutator"):
            invoke_claude_code(
                str(tmp_path),
                "prompt",
                AgentConfig(backend="claude"),
                fixed_env={"ANTHROPIC_API_KEY": "sk-ant-explicit"},
            )

        text = caplog.text
        assert "takes precedence" in text
        assert "helix-auth-claude" in text

    def test_no_warning_without_an_anthropic_credential(
        self, tmp_path: Path, monkeypatch, local_run, caplog
    ):
        local_run()
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)

        with caplog.at_level(logging.WARNING, logger="helix.mutator"):
            invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="claude"))

        assert "ANTHROPIC" not in caplog.text

    def test_warning_is_emitted_once_per_backend(
        self, tmp_path: Path, monkeypatch, local_run, caplog
    ):
        """A run has many candidates; the warning must not spam per mutation."""
        local_run()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-ambient")

        with caplog.at_level(logging.WARNING, logger="helix.mutator"):
            for _ in range(3):
                invoke_claude_code(
                    str(tmp_path), "prompt", AgentConfig(backend="claude")
                )

        assert caplog.text.count("NOT forwarded") == 1


# ---------------------------------------------------------------------------
# Registry-level invariant
# ---------------------------------------------------------------------------


def test_no_anthropic_name_in_the_auto_forward_registry():
    """Guards against a future backend re-adding an Anthropic key to the table."""
    from helix.backends import ANTHROPIC_KEY_ENV, BACKEND_AUTH_ENV

    for backend, names in BACKEND_AUTH_ENV.items():
        leaked = set(names) & set(ANTHROPIC_KEY_ENV)
        assert not leaked, (
            f"BACKEND_AUTH_ENV[{backend!r}] auto-forwards {sorted(leaked)}; "
            "Anthropic credentials require an explicit opt-in"
        )
