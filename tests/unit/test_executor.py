"""Unit tests for HELIX executor."""

from __future__ import annotations

from unittest.mock import MagicMock


from helix.population import Candidate, EvalResult
from helix.config import HelixConfig, EvaluatorConfig
from helix.executor import run_evaluator


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
    include_stdout: bool = True,
    include_stderr: bool = True,
    extra_commands: list[str] | None = None,
) -> HelixConfig:
    evaluator = EvaluatorConfig(
        command=command,
        score_parser=score_parser,
        include_stdout=include_stdout,
        include_stderr=include_stderr,
        extra_commands=extra_commands or [],
    )
    return HelixConfig(objective="test objective", evaluator=evaluator)


# ---------------------------------------------------------------------------
# Tests: successful evaluation
# ---------------------------------------------------------------------------


class TestRunEvaluatorSuccess:
    def test_returns_eval_result(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="output text",
            stderr="",
            returncode=0,
        )
        candidate = make_candidate()
        config = make_config(score_parser="exitcode")

        result = run_evaluator(candidate, config)

        assert isinstance(result, EvalResult)
        assert result.candidate_id == candidate.id

    def test_success_scores(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="",
            returncode=0,
        )
        candidate = make_candidate()
        config = make_config(score_parser="exitcode")

        result = run_evaluator(candidate, config)

        assert result.scores["success"] == 1.0

    def test_failure_scores(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="error output",
            returncode=1,
        )
        candidate = make_candidate()
        config = make_config(score_parser="exitcode")

        result = run_evaluator(candidate, config)

        assert result.scores["success"] == 0.0

    def test_stdout_included_in_asi(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="hello stdout",
            stderr="",
            returncode=0,
        )
        candidate = make_candidate()
        config = make_config(score_parser="exitcode", include_stdout=True)

        result = run_evaluator(candidate, config)

        assert result.asi["stdout"] == "hello stdout"

    def test_stderr_included_in_asi(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="hello stderr",
            returncode=0,
        )
        candidate = make_candidate()
        config = make_config(score_parser="exitcode", include_stderr=True)

        result = run_evaluator(candidate, config)

        assert result.asi["stderr"] == "hello stderr"

    def test_stdout_excluded_when_disabled(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="hello stdout",
            stderr="",
            returncode=0,
        )
        candidate = make_candidate()
        config = make_config(score_parser="exitcode", include_stdout=False)

        result = run_evaluator(candidate, config)

        assert "stdout" not in result.asi

    def test_stderr_excluded_when_disabled(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(
            stdout="",
            stderr="hello stderr",
            returncode=0,
        )
        candidate = make_candidate()
        config = make_config(score_parser="exitcode", include_stderr=False)

        result = run_evaluator(candidate, config)

        assert "stderr" not in result.asi


# ---------------------------------------------------------------------------
# Tests: extra_commands
# ---------------------------------------------------------------------------


class TestRunEvaluatorExtraCommands:
    def test_extra_commands_run(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")

        # First call: main command; second: extra_command
        mock_run.side_effect = [
            MagicMock(stdout="main out", stderr="", returncode=0),
            MagicMock(stdout="extra out", stderr="", returncode=0),
        ]
        candidate = make_candidate()
        config = make_config(
            score_parser="exitcode",
            extra_commands=["cat coverage.txt"],
        )

        run_evaluator(candidate, config)

    def test_extra_command_output_in_asi(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")

        mock_run.side_effect = [
            MagicMock(stdout="main out", stderr="", returncode=0),
            MagicMock(stdout="extra out 0", stderr="", returncode=0),
        ]
        candidate = make_candidate()
        config = make_config(
            score_parser="exitcode",
            extra_commands=["cat coverage.txt"],
        )

        result = run_evaluator(candidate, config)

        assert "extra_0" in result.asi
        assert result.asi["extra_0"] == "extra out 0"

    def test_multiple_extra_commands_in_asi(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")

        mock_run.side_effect = [
            MagicMock(stdout="main out", stderr="", returncode=0),
            MagicMock(stdout="extra 0", stderr="", returncode=0),
            MagicMock(stdout="extra 1", stderr="", returncode=0),
        ]
        candidate = make_candidate()
        config = make_config(
            score_parser="exitcode",
            extra_commands=["cat file0.txt", "cat file1.txt"],
        )

        result = run_evaluator(candidate, config)

        assert result.asi["extra_0"] == "extra 0"
        assert result.asi["extra_1"] == "extra 1"


# ---------------------------------------------------------------------------
# Tests: pytest parser integration via executor
# ---------------------------------------------------------------------------


class TestRunEvaluatorWithPytestParser:
    def test_pytest_parser_pass_rate(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        pytest_output = (
            "tests/test_foo.py::test_a PASSED\n"
            "FAILED tests/test_foo.py::test_b - AssertionError\n"
            "1 passed, 1 failed in 0.5s\n"
        )
        mock_run.return_value = MagicMock(
            stdout=pytest_output,
            stderr="",
            returncode=1,
        )
        candidate = make_candidate()
        config = make_config(score_parser="pytest")

        result = run_evaluator(candidate, config)

        assert abs(result.scores["pass_rate"] - 0.5) < 1e-6

    def test_pytest_instance_scores(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        pytest_output = (
            "tests/test_foo.py::test_a PASSED\n"
            "FAILED tests/test_foo.py::test_b - AssertionError\n"
            "1 passed, 1 failed in 0.5s\n"
        )
        mock_run.return_value = MagicMock(
            stdout=pytest_output,
            stderr="",
            returncode=1,
        )
        candidate = make_candidate()
        config = make_config(score_parser="pytest")

        result = run_evaluator(candidate, config)

        assert result.instance_scores["tests/test_foo.py::test_a"] == 1.0
        assert result.instance_scores["tests/test_foo.py::test_b"] == 0.0
