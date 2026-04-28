"""Unit tests for HELIX executor security features."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from helix.population import Candidate
from helix.config import EvaluatorSidecarConfig, EvaluatorConfig, HelixConfig, SandboxConfig
from helix.executor import (
    run_evaluator,
    _validate_and_split_command,
    _scrub_environment,
)
from helix.exceptions import EvaluatorError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidate(worktree_path: str = "/tmp/fake-worktree") -> Candidate:
    return Candidate(
        id="cand-001",
        worktree_path=worktree_path,
        branch_name="helix/cand-001",
        generation=1,
        parent_id=None,
        parent_ids=[],
        operation="mutate",
    )


def make_config(
    command: str = "pytest -q",
    score_parser: str = "exitcode",
    extra_commands: list[str] | None = None,
) -> HelixConfig:
    evaluator = EvaluatorConfig(
        command=command,
        score_parser=score_parser,
        include_stdout=True,
        include_stderr=True,
        extra_commands=extra_commands or [],
    )
    return HelixConfig(objective="test objective", evaluator=evaluator)


# ---------------------------------------------------------------------------
# Tests: Command tokenization
# ---------------------------------------------------------------------------


class TestCommandValidation:
    """Tokenization invariants of _validate_and_split_command.

    The real safety boundary is ``shell=False``; the validator only
    tokenizes and rejects an empty command.
    """

    def test_basic_tokenization(self):
        """shlex.split splits a simple command into tokens."""
        assert _validate_and_split_command("python test.py") == ["python", "test.py"]

    def test_quoted_argument_preserved(self):
        """Quoted arguments stay intact through shlex.split."""
        tokens = _validate_and_split_command('python test.py --arg "value with spaces"')
        assert tokens == ["python", "test.py", "--arg", "value with spaces"]

    def test_empty_command_raises_error(self):
        """Empty command string raises EvaluatorError."""
        with pytest.raises(EvaluatorError) as exc_info:
            _validate_and_split_command("")

        assert "Empty command" in str(exc_info.value)

    def test_unclosed_quote_raises_evaluator_error(self):
        """Malformed quoting surfaces as EvaluatorError, not bare ValueError."""
        with pytest.raises(EvaluatorError) as exc_info:
            _validate_and_split_command('python "unterminated')

        assert "Failed to parse evaluator command" in str(exc_info.value)
        assert exc_info.value.command == 'python "unterminated'


# ---------------------------------------------------------------------------
# Tests: Environment scrubbing
# ---------------------------------------------------------------------------


class TestEnvironmentScrubbing:
    """Test that environment variables are properly scrubbed."""

    def test_scrubbed_env_includes_path(self, monkeypatch):
        """PATH is preserved in scrubbed environment."""
        monkeypatch.setenv("PATH", "/usr/bin:/bin")
        env = _scrub_environment("val")
        assert env["PATH"] == "/usr/bin:/bin"

    def test_scrubbed_env_includes_home(self, monkeypatch):
        """HOME is preserved in scrubbed environment."""
        monkeypatch.setenv("HOME", "/home/testuser")
        env = _scrub_environment("val")
        assert env["HOME"] == "/home/testuser"

    def test_scrubbed_env_includes_helix_split(self):
        """HELIX_SPLIT is set from the split parameter."""
        env = _scrub_environment("test")
        assert env["HELIX_SPLIT"] == "test"

    def test_scrubbed_env_includes_helix_vars(self, monkeypatch):
        """HELIX_* variables are preserved."""
        monkeypatch.setenv("HELIX_DEBUG", "1")
        monkeypatch.setenv("HELIX_TIMEOUT", "300")
        env = _scrub_environment("val")
        assert env["HELIX_DEBUG"] == "1"
        assert env["HELIX_TIMEOUT"] == "300"

    def test_scrubbed_env_excludes_sensitive_vars(self, monkeypatch):
        """Sensitive variables are stripped."""
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret123")
        monkeypatch.setenv("DATABASE_PASSWORD", "pass123")
        monkeypatch.setenv("API_KEY", "key123")
        env = _scrub_environment("val")
        assert "AWS_SECRET_ACCESS_KEY" not in env
        assert "DATABASE_PASSWORD" not in env
        assert "API_KEY" not in env

    def test_scrubbed_env_excludes_user_vars(self, monkeypatch):
        """User-defined variables are stripped."""
        monkeypatch.setenv("MY_CUSTOM_VAR", "value")
        monkeypatch.setenv("RANDOM_ENV", "test")
        env = _scrub_environment("val")
        assert "MY_CUSTOM_VAR" not in env
        assert "RANDOM_ENV" not in env

    def test_scrubbed_env_only_allowed_keys(self, monkeypatch):
        """Only PATH, HOME, HELIX_SPLIT, and HELIX_* are in scrubbed env."""
        # Clear and set specific environment
        for key in list(os.environ.keys()):
            if key not in ["PATH", "HOME"]:
                monkeypatch.delenv(key, raising=False)

        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("HOME", "/home/user")
        monkeypatch.setenv("HELIX_VAR", "value")
        monkeypatch.setenv("FORBIDDEN", "bad")

        env = _scrub_environment("val")

        # Should have exactly: PATH, HOME, HELIX_SPLIT, HELIX_VAR
        allowed_keys = {"PATH", "HOME", "HELIX_SPLIT", "HELIX_VAR"}
        assert set(env.keys()) == allowed_keys

    def test_passthrough_env_preserves_listed_vars(self, monkeypatch):
        """passthrough_env includes specified variables in the scrubbed env."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
        monkeypatch.setenv("HF_HOME", "/data/hf")
        monkeypatch.setenv("SECRET_KEY", "should_not_appear")

        env = _scrub_environment("val", passthrough_env=["CUDA_VISIBLE_DEVICES", "HF_HOME"])
        assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
        assert env["HF_HOME"] == "/data/hf"
        assert "SECRET_KEY" not in env

    def test_passthrough_env_missing_var_is_ignored(self, monkeypatch):
        """passthrough_env silently skips vars not present in os.environ."""
        monkeypatch.delenv("NONEXISTENT_VAR", raising=False)
        env = _scrub_environment("val", passthrough_env=["NONEXISTENT_VAR"])
        assert "NONEXISTENT_VAR" not in env

    def test_passthrough_env_empty_list_no_change(self):
        """Empty passthrough_env behaves identically to the default."""
        env_default = _scrub_environment("val")
        env_empty = _scrub_environment("val", passthrough_env=[])
        assert env_default == env_empty

    def test_scrub_without_split_omits_helix_split(self, monkeypatch):
        """When split is None (CC subprocess path), HELIX_SPLIT is not set."""
        monkeypatch.delenv("HELIX_SPLIT", raising=False)
        env = _scrub_environment(passthrough_env=["PATH"])
        assert "HELIX_SPLIT" not in env
        # PATH and HOME should still be present
        assert "PATH" in env

    def test_scrub_without_split_preserves_passthrough(self, monkeypatch):
        """CC subprocess path (split=None) still honours passthrough_env."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        env = _scrub_environment(passthrough_env=["CUDA_VISIBLE_DEVICES"])
        assert env["CUDA_VISIBLE_DEVICES"] == "2"
        assert "HELIX_SPLIT" not in env


# ---------------------------------------------------------------------------
# Tests: Integration - run_evaluator with security
# ---------------------------------------------------------------------------


class TestRunEvaluator:
    """Integration-style tests for run_evaluator: tokenization, env scrub, and the shell=False invariant."""

    def test_run_evaluator_uses_scrubbed_env(self, mocker, monkeypatch):
        """run_evaluator passes only scrubbed environment variables."""
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="output",
            stderr="",
            returncode=0,
        )

        # Set some environment variables
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("HOME", "/home/user")
        monkeypatch.setenv("AWS_SECRET", "secret")
        monkeypatch.setenv("HELIX_DEBUG", "1")

        candidate = make_candidate()
        config = make_config(command="python test.py")

        run_evaluator(candidate, config, split="train")

        # Check that subprocess.run was called with scrubbed env
        call_kwargs = mock_run.call_args[1]
        env = call_kwargs["env"]

        assert "PATH" in env
        assert "HOME" in env
        assert "HELIX_SPLIT" in env
        assert env["HELIX_SPLIT"] == "train"
        assert "HELIX_DEBUG" in env
        assert "AWS_SECRET" not in env

    def test_run_evaluator_uses_shell_false(self, mocker):
        """run_evaluator calls subprocess.run with shell=False."""
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="output",
            stderr="",
            returncode=0,
        )

        candidate = make_candidate()
        config = make_config(command="python test.py")

        run_evaluator(candidate, config)

        # Check that shell=False was used
        call_kwargs = mock_run.call_args[1]
        assert call_kwargs["shell"] is False

    def test_run_evaluator_splits_command_properly(self, mocker):
        """run_evaluator passes split command as list."""
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="output",
            stderr="",
            returncode=0,
        )

        candidate = make_candidate()
        config = make_config(command='python test.py --arg "value with spaces"')

        run_evaluator(candidate, config)

        # Check that command was split into list
        call_args = mock_run.call_args[0]
        assert isinstance(call_args[0], list)
        assert call_args[0][0] == "python"
        assert "test.py" in call_args[0]

    def test_run_evaluator_uses_sandbox_when_enabled(self, mocker):
        mock_run = mocker.patch("helix.executor.run_sandboxed_commands")
        mocker.patch(
            "helix.executor.current_evaluator_sidecar_runtime",
            return_value=MagicMock(),
        )
        mock_run.return_value = [MagicMock(stdout="", stderr="", returncode=0)]

        candidate = make_candidate()
        config = make_config(command="python /runner/evaluate.py")
        config.evaluator.sidecar = EvaluatorSidecarConfig(
            image="eval:latest",
            runner_image="eval-runner:latest",
            command="python -m server",
            endpoint="http://helix-evaluator:8080/evaluate",
        )
        config = config.model_copy(update={"sandbox": SandboxConfig(enabled=True)})

        run_evaluator(candidate, config)

        mock_run.assert_called_once()
        assert mock_run.call_args.kwargs["scope"] == "evaluator"
        assert mock_run.call_args.kwargs["sync_back"] is False
        assert mock_run.call_args.kwargs["image"] == "eval-runner:latest"

    def test_sandboxed_evaluator_runs_extra_commands_in_same_sequence(self, mocker):
        mock_run = mocker.patch("helix.executor.run_sandboxed_commands")
        mocker.patch(
            "helix.executor.current_evaluator_sidecar_runtime",
            return_value=MagicMock(),
        )
        mock_run.return_value = [
            MagicMock(stdout="main", stderr="", returncode=0),
            MagicMock(stdout="extra", stderr="", returncode=0),
        ]

        candidate = make_candidate()
        config = make_config(command="python /runner/evaluate.py")
        config.evaluator.extra_commands = ["python extra.py"]
        config.evaluator.sidecar = EvaluatorSidecarConfig(
            image="eval:latest",
            runner_image="eval-runner:latest",
            command="python -m server",
            endpoint="http://helix-evaluator:8080/evaluate",
        )
        config = config.model_copy(update={"sandbox": SandboxConfig(enabled=True)})

        run_evaluator(candidate, config)

        assert mock_run.call_args.args[0] == [["python", "/runner/evaluate.py"], ["python", "extra.py"]]
