"""Unit tests for HELIX executor security features."""

from __future__ import annotations

import json
import logging
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
from helix.mutator import build_mutation_prompt


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

    def test_fixed_env_injects_values_after_passthrough(self, monkeypatch):
        """fixed_env records run-local values without relying on parent env."""
        monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://wrong")

        env = _scrub_environment(
            passthrough_env=["ANTHROPIC_BASE_URL"],
            fixed_env={
                "ANTHROPIC_BASE_URL": "https://model-service.example.invalid/v1",
                "ANTHROPIC_API_KEY": "dummy",
            },
        )

        assert env["ANTHROPIC_BASE_URL"] == "https://model-service.example.invalid/v1"
        assert env["ANTHROPIC_API_KEY"] == "dummy"

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
        mock_sandbox_run = mocker.patch("helix.executor.run_sandboxed_commands")
        mock_host_run = mocker.patch("helix.executor.subprocess.run")
        mocker.patch(
            "helix.executor.current_evaluator_sidecar_runtime",
            return_value=MagicMock(),
        )
        mock_sandbox_run.return_value = [MagicMock(stdout="", stderr="", returncode=0)]

        candidate = make_candidate()
        config = make_config(command="python /runner/evaluate.py")
        config.evaluator.sidecar = EvaluatorSidecarConfig(
            image="eval:latest",
            runner_image="eval-runner:latest",
            command="python -m server",
            endpoint="http://helix-evaluator:8080/evaluate",
        )
        config = config.model_copy(update={"sandbox": SandboxConfig(enabled=True, evaluator=True)})

        run_evaluator(candidate, config)

        mock_sandbox_run.assert_called_once()
        mock_host_run.assert_not_called()
        assert mock_sandbox_run.call_args.kwargs["scope"] == "evaluator"
        assert mock_sandbox_run.call_args.kwargs["sync_back"] is False
        assert mock_sandbox_run.call_args.kwargs["image"] == "eval-runner:latest"

    def test_sandboxed_evaluator_runs_extra_commands_in_same_sequence(self, mocker):
        mock_run = mocker.patch("helix.executor.run_sandboxed_commands")
        mocker.patch(
            "helix.executor.current_evaluator_sidecar_runtime",
            return_value=MagicMock(),
        )
        mock_run.return_value = [
            MagicMock(stdout="main", stderr="", returncode=0),
            MagicMock(stdout="extra", stderr="", returncode=0),
            MagicMock(stdout="", stderr="", returncode=0),
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
        config = config.model_copy(
            update={"sandbox": SandboxConfig(enabled=True, evaluator=True)}
        )

        run_evaluator(candidate, config)

        commands = mock_run.call_args.args[0]
        assert commands[:2] == [["python", "/runner/evaluate.py"], ["python", "extra.py"]]
        assert commands[2][:2] == ["sh", "-c"]
        assert ".helix_asi_log_" in commands[2][2]
        # Capture command must use the absolute ``/workspace/...`` path
        # so the cwd assumption inside the sidecar does not affect what
        # the ``cat`` resolves.
        assert "/workspace/.helix_asi_log_" in commands[2][2]

    def test_sandboxed_evaluator_parses_helix_log_capture_into_asi(self, mocker):
        """The trailing ``cat`` capture command's stdout is JSONL from
        the evaluator's ``helix.log()`` calls.  The executor must run
        it through :func:`helix.asi.read_text` and surface the
        rendered notes under ``asi["log"]`` — this end-to-end check
        protects against accidentally treating the capture stdout as
        an ``extra_N`` entry or dropping it."""
        mock_run = mocker.patch("helix.executor.run_sandboxed_commands")
        mocker.patch(
            "helix.executor.current_evaluator_sidecar_runtime",
            return_value=MagicMock(),
        )
        # Two records simulating an evaluator that called
        # ``helix.log("starting", phase="warmup")`` then
        # ``helix.log(score=0.7)``.
        capture_stdout = (
            '{"message": "starting", "phase": "warmup"}\n'
            '{"score": 0.7}\n'
        )
        mock_run.return_value = [
            MagicMock(stdout="HELIX_RESULT=[]\n", stderr="", returncode=0),
            MagicMock(stdout=capture_stdout, stderr="", returncode=0),
        ]

        candidate = make_candidate()
        config = make_config(command="python /runner/evaluate.py")
        config.evaluator.sidecar = EvaluatorSidecarConfig(
            image="eval:latest",
            runner_image="eval-runner:latest",
            command="python -m server",
            endpoint="http://helix-evaluator:8080/evaluate",
        )
        config = config.model_copy(
            update={"sandbox": SandboxConfig(enabled=True, evaluator=True)}
        )

        result = run_evaluator(candidate, config)

        # Notes were captured via the ``cat`` command and routed to
        # ``asi["log"]`` (NOT exposed as ``extra_0``, which would
        # mistakenly treat the capture as a user-provided extra
        # command).
        assert "log" in result.asi
        assert "starting" in result.asi["log"]
        assert "phase: warmup" in result.asi["log"]
        assert "score: 0.7" in result.asi["log"]
        assert "extra_0" not in result.asi

    def test_run_evaluator_uses_host_when_sandbox_enabled_but_evaluator_disabled(self, mocker):
        mock_host_run = mocker.patch("helix.executor.subprocess.run")
        mock_sandbox_run = mocker.patch("helix.executor.run_sandboxed_commands")
        mock_host_run.return_value = MagicMock(stdout="", stderr="", returncode=0)

        candidate = make_candidate()
        config = make_config(command="python evaluate.py")
        config = config.model_copy(update={"sandbox": SandboxConfig(enabled=True, evaluator=False)})

        run_evaluator(candidate, config)

        mock_host_run.assert_called_once()
        mock_sandbox_run.assert_not_called()


class TestEvaluatorDiagnosticSecretRedaction:
    """Evaluator data is parsed raw, then redacted before rendering."""

    SECRET = "sk-evaluator-secret-47"

    def _config(self, *, score_parser: str = "exitcode") -> HelixConfig:
        config = make_config(command="python evaluate.py", score_parser=score_parser)
        return config.model_copy(update={"env": {"EVALUATOR_TOKEN": self.SECRET}})

    def _sandbox_config(self, *, score_parser: str = "exitcode") -> HelixConfig:
        config = self._config(score_parser=score_parser)
        config.evaluator.sidecar = EvaluatorSidecarConfig(
            image="eval:latest",
            runner_image="eval-runner:latest",
            command="python -m server",
            endpoint="http://helix-evaluator:8080/evaluate",
        )
        return config.model_copy(
            update={"sandbox": SandboxConfig(enabled=True, evaluator=True)}
        )

    def test_stdout_secret_is_absent_from_rendered_prompt(self, mocker):
        mocker.patch(
            "helix.executor.current_evaluator_sidecar_runtime",
            return_value=MagicMock(),
        )
        mocker.patch(
            "helix.executor.run_sandboxed_commands",
            return_value=[
                MagicMock(
                    stdout=f"EVALUATOR_TOKEN={self.SECRET} emitted by evaluator",
                    stderr="",
                    returncode=0,
                ),
                MagicMock(stdout="", stderr="", returncode=0),
            ],
        )

        result = run_evaluator(make_candidate(), self._sandbox_config())
        prompt = build_mutation_prompt("goal", result)

        assert self.SECRET not in prompt
        assert "EVALUATOR_TOKEN=<redacted>" in prompt

    def test_stderr_and_untruncated_error_context_are_redacted(self, mocker, caplog):
        mocker.patch(
            "helix.executor.current_evaluator_sidecar_runtime",
            return_value=MagicMock(),
        )
        mocker.patch(
            "helix.executor.run_sandboxed_commands",
            return_value=[
                MagicMock(
                    stdout=f"stdout includes {self.SECRET}",
                    stderr=f"stderr includes {self.SECRET}",
                    returncode=1,
                ),
                MagicMock(stdout="", stderr="", returncode=0),
            ],
        )
        caplog.set_level(logging.INFO, logger="helix.executor")

        result = run_evaluator(make_candidate(), self._sandbox_config())
        prompt = build_mutation_prompt("goal", result)

        assert self.SECRET not in prompt
        assert self.SECRET not in caplog.text
        assert "stdout includes <redacted>" in caplog.text
        assert "stderr includes <redacted>" in caplog.text

    def test_parser_receives_raw_stdout_before_rendering_redaction(self, mocker):
        observed: dict[str, str] = {}

        def parser(returncode: int, stdout: str, stderr: str):
            observed["stdout"] = stdout
            return {"success": 1.0}, {"success": 1.0}

        mocker.patch("helix.executor.get_parser", return_value=parser)
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=MagicMock(stdout=self.SECRET, stderr="", returncode=0),
        )

        result = run_evaluator(make_candidate(), self._config())

        assert observed["stdout"] == self.SECRET
        assert self.SECRET not in build_mutation_prompt("goal", result)

    def test_nested_side_info_secret_is_absent_from_diagnostics(self, mocker, tmp_path):
        (tmp_path / "helix_batch.json").write_text(json.dumps(["example-1"]))
        side_info = {"diagnostic": {"credential": self.SECRET}}
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=MagicMock(
                stdout=f"HELIX_RESULT=[[1.0, {json.dumps(side_info)}]]\n",
                stderr="",
                returncode=0,
            ),
        )

        result = run_evaluator(
            make_candidate(str(tmp_path)), self._config(score_parser="helix_result")
        )
        prompt = build_mutation_prompt("goal", result)

        assert self.SECRET not in prompt
        assert "diagnostic" in prompt
        assert "credential" in prompt
        assert "<redacted>" in prompt
