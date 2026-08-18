"""Unit tests for the single-parser HELIX executor path."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helix.config import (
    EvaluatorConfig,
    EvaluatorSidecarConfig,
    HelixConfig,
    SandboxConfig,
)
from helix.exceptions import EvaluatorError
from helix.executor import run_evaluator
from helix.population import Candidate, EvalResult


def make_candidate(worktree_path: str) -> Candidate:
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
    command: str = "python eval.py",
    include_stdout: bool = True,
    include_stderr: bool = True,
    extra_commands: list[str] | None = None,
) -> HelixConfig:
    return HelixConfig(
        objective="test objective",
        evaluator=EvaluatorConfig(
            command=command,
            include_stdout=include_stdout,
            include_stderr=include_stderr,
            extra_commands=extra_commands or [],
        ),
    )


def _result_line(scores: list[float], side_info: list[dict] | None = None) -> str:
    infos = side_info or [{} for _ in scores]
    return "HELIX_RESULT=" + json.dumps(
        [[score, info] for score, info in zip(scores, infos)]
    )


def _prepare_batch(path: Path, ids: list[str] | None = None) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "helix_batch.json").write_text(json.dumps(ids or ["example-0"]))


class TestRunEvaluator:
    def test_returns_eval_result_and_per_example_fields(self, mocker, tmp_path: Path):
        _prepare_batch(tmp_path, ["a", "b"])
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=MagicMock(
                stdout=_result_line([1.0, 0.5], [{"note": "a"}, {"note": "b"}]),
                stderr="",
                returncode=0,
            ),
        )

        result = run_evaluator(make_candidate(str(tmp_path)), make_config())

        assert isinstance(result, EvalResult)
        assert result.scores["success"] == 0.75
        assert result.instance_scores == {"a": 1.0, "b": 0.5}
        assert result.per_example_side_info == [{"note": "a"}, {"note": "b"}]

    def test_nonzero_return_code_zeroes_aggregate(self, mocker, tmp_path: Path):
        _prepare_batch(tmp_path)
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=MagicMock(
                stdout=_result_line([1.0]), stderr="error", returncode=1
            ),
        )

        result = run_evaluator(make_candidate(str(tmp_path)), make_config())

        assert result.scores["success"] == 0.0
        assert result.instance_scores == {"example-0": 1.0}

    def test_docker_exit_125_surfaces_docker_diagnostic_when_docker_was_invoked(
        self, tmp_path, mocker
    ):
        """Exit 125 is only a Docker diagnostic when HELIX itself invoked
        Docker for this run (sandbox.enabled + sandbox.evaluator) — the
        branch this test exercises via ``run_sandboxed_commands``.
        """
        mocker.patch(
            "helix.executor.current_evaluator_sidecar_runtime",
            return_value=MagicMock(),
        )
        mocker.patch(
            "helix.executor.run_sandboxed_commands",
            return_value=[
                MagicMock(
                    stdout="",
                    stderr="docker: Error response from daemon: image unavailable",
                    returncode=125,
                ),
                MagicMock(stdout=""),
            ],
        )
        candidate = make_candidate(str(tmp_path))
        config = HelixConfig(
            objective="test objective",
            evaluator=EvaluatorConfig(
                command="python eval.py",
                sidecar=EvaluatorSidecarConfig(
                    image="runner:latest",
                    command="serve",
                    endpoint="http://sidecar",
                ),
            ),
            sandbox=SandboxConfig(enabled=True, evaluator=True),
        )

        with pytest.raises(EvaluatorError) as exc_info:
            run_evaluator(candidate, config)
        assert exc_info.value.phase == "docker invocation"
        assert "image unavailable" in exc_info.value.stderr
        assert "HELIX_RESULT" not in str(exc_info.value)

    def test_non_docker_command_exit_125_with_valid_result_is_not_a_docker_failure(
        self, tmp_path, mocker
    ):
        """Evaluator commands are arbitrary user commands. HELIX did not
        invoke Docker for this run (sandboxing is off), so an evaluator that
        happens to exit 125 while still emitting a valid HELIX_RESULT= line
        must be scored normally, not raised as a Docker invocation failure.
        """
        _prepare_batch(tmp_path)
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=MagicMock(
                stdout=_result_line([1.0]), stderr="", returncode=125
            ),
        )
        candidate = make_candidate(str(tmp_path))
        config = make_config()

        result = run_evaluator(candidate, config)

        assert isinstance(result, EvalResult)
        assert result.instance_scores == {"example-0": 1.0}

    def test_asi_options_and_helix_log(self, tmp_path: Path):
        _prepare_batch(tmp_path)
        command = (
            f"{sys.executable} -c "
            '"from helix import log; log(\'unique evaluator note\', score=0.7); '
            "print('" + _result_line([0.7]) + "')" + '"'
        )
        result = run_evaluator(
            make_candidate(str(tmp_path)),
            make_config(command=command),
        )

        assert "unique evaluator note" in result.asi["log"]
        assert "score: 0.7" in result.asi["log"]
        assert _result_line([0.7]) in result.asi["stdout"]

    def test_extra_commands_are_captured(self, mocker, tmp_path: Path):
        _prepare_batch(tmp_path)
        mocker.patch(
            "helix.executor.subprocess.run",
            side_effect=[
                MagicMock(stdout=_result_line([0.8]), stderr="", returncode=0),
                MagicMock(stdout="extra output", stderr="", returncode=0),
            ],
        )

        result = run_evaluator(
            make_candidate(str(tmp_path)),
            make_config(extra_commands=["echo extra"]),
        )

        assert result.asi["extra_0"] == "extra output"

    def test_requested_ids_follow_batch_order(self, mocker, tmp_path: Path):
        _prepare_batch(tmp_path, ["a", "b"])
        mocker.patch(
            "helix.executor.subprocess.run",
            return_value=MagicMock(
                stdout=_result_line([0.25, 0.75]), stderr="", returncode=0
            ),
        )

        result = run_evaluator(
            make_candidate(str(tmp_path)),
            make_config(),
            split="train",
            instance_ids=["a", "b"],
        )

        assert result.instance_scores == {"a": 0.25, "b": 0.75}
