"""Unit tests for HELIX executor."""

from __future__ import annotations

import dataclasses
import sys
import threading
from unittest.mock import MagicMock

import pytest

from helix.population import Candidate, EvalResult
from helix.config import HelixConfig, EvaluatorConfig
from helix.exceptions import (
    EvaluatorError,
    PromptArtifactCollisionError,
    ResumeIncompatibleError,
)
from helix.executor import (
    EvalBatchItem,
    EvalBatchResult,
    make_default_batch_runner,
    run_evaluator,
    run_evaluator_batch,
)


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

    def test_sets_helix_asi_log_env(self, mocker):
        mock_run = mocker.patch("helix.executor.subprocess.run")
        mock_run.return_value = MagicMock(stdout="", stderr="", returncode=0)
        candidate = make_candidate()
        config = make_config(score_parser="exitcode")

        run_evaluator(candidate, config)

        env = mock_run.call_args.kwargs["env"]
        assert "HELIX_ASI_LOG" in env

    def test_captures_helix_log_in_asi(self, tmp_path):
        command = (
            f"{sys.executable} -c "
            "\"from helix import log; log('unique evaluator note', score=0.7)\""
        )
        candidate = make_candidate(str(tmp_path))
        config = make_config(command=command, score_parser="exitcode")

        result = run_evaluator(candidate, config)

        assert "unique evaluator note" in result.asi["log"]
        assert "score: 0.7" in result.asi["log"]


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


# ---------------------------------------------------------------------------
# Tests: run_evaluator_batch — ordered cross-candidate evaluator batching
# ---------------------------------------------------------------------------


def _batch_candidate(cid: str, worktree: str) -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=worktree,
        branch_name=f"helix/{cid}",
        generation=1,
        parent_id=None,
        parent_ids=[],
        operation="mutate",
    )


def make_item(
    cid: str,
    worktree: str,
    *,
    content_key: str | None = None,
    split: str = "val",
    instance_ids: tuple[str, ...] | None = ("0",),
) -> EvalBatchItem:
    return EvalBatchItem(
        candidate=_batch_candidate(cid, worktree),
        content_key=content_key if content_key is not None else cid,
        split=split,
        instance_ids=instance_ids,
    )


def make_eval_result(
    cid: str, instance_scores: dict[str, float] | None = None
) -> EvalResult:
    return EvalResult(
        candidate_id=cid,
        scores={"success": 1.0},
        asi={},
        instance_scores=instance_scores if instance_scores is not None else {"0": 1.0},
    )


class TestRunEvaluatorBatchOrdering:
    def test_results_in_input_order_under_reverse_completion(self):
        # Force leaders to COMPLETE in reverse of input order; results must
        # still be positional to the input sequence.
        n = 4
        start_barrier = threading.Barrier(n)
        gate = [threading.Event() for _ in range(n)]
        completion_order: list[int] = []
        lock = threading.Lock()

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            idx = int(item.candidate.id)
            start_barrier.wait()  # all n leaders in flight concurrently
            if idx < n - 1:
                gate[idx].wait()  # wait until the higher index finishes first
            with lock:
                completion_order.append(idx)
            if idx > 0:
                gate[idx - 1].set()  # release the next-lower index
            return make_eval_result(item.candidate.id), 1

        items = [
            make_item(str(i), f"/wt/{i}", instance_ids=(str(i),)) for i in range(n)
        ]
        results = run_evaluator_batch(items, runner, max_workers=n)

        # Completion happened newest-first, proving out-of-order execution...
        assert completion_order == [3, 2, 1, 0]
        # ...yet results are returned strictly in input order.
        assert [r.item.candidate.id for r in results] == ["0", "1", "2", "3"]
        assert all(r.result is not None for r in results)

    def test_result_count_and_order_match_input(self):
        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            return make_eval_result(item.candidate.id), 1

        items = [make_item(f"c{i}", f"/w{i}", instance_ids=(str(i),)) for i in range(5)]
        results = run_evaluator_batch(items, runner, max_workers=2)

        assert len(results) == 5
        assert [r.item.candidate.id for r in results] == ["c0", "c1", "c2", "c3", "c4"]

    def test_empty_items_returns_empty(self):
        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            raise AssertionError("runner must not be called for empty input")

        assert run_evaluator_batch([], runner, max_workers=2) == []


class TestRunEvaluatorBatchConcurrency:
    def test_same_worktree_leaders_are_serialized(self):
        # Two leaders that share a worktree must NOT run concurrently: a
        # 2-party barrier can never rendezvous, so both time out (broken).
        barrier = threading.Barrier(2, timeout=0.5)
        overlapped: list[bool] = []
        lock = threading.Lock()

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            try:
                barrier.wait()
                ok = True
            except threading.BrokenBarrierError:
                ok = False
            with lock:
                overlapped.append(ok)
            return make_eval_result(item.candidate.id), 1

        items = [
            make_item("a", "/same-wt", instance_ids=("1",)),
            make_item("b", "/same-wt", instance_ids=("2",)),
        ]
        run_evaluator_batch(items, runner, max_workers=2)

        # Neither leader ever saw the other inside the runner → serialized.
        assert overlapped == [False, False]

    def test_distinct_worktrees_run_concurrently(self):
        # Different worktrees may overlap: a 2-party barrier rendezvous proves
        # both leaders were inside the runner at the same time.
        barrier = threading.Barrier(2, timeout=2.0)
        rendezvous: list[bool] = []
        lock = threading.Lock()

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            try:
                barrier.wait()
                ok = True
            except threading.BrokenBarrierError:
                ok = False
            with lock:
                rendezvous.append(ok)
            return make_eval_result(item.candidate.id), 1

        items = [
            make_item("a", "/wt-a", instance_ids=("1",)),
            make_item("b", "/wt-b", instance_ids=("2",)),
        ]
        run_evaluator_batch(items, runner, max_workers=2)

        assert rendezvous == [True, True]


class TestRunEvaluatorBatchDedup:
    def test_identical_requests_run_once_and_follower_reuses(self):
        calls: list[str] = []
        lock = threading.Lock()

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            with lock:
                calls.append(item.candidate.id)
            return make_eval_result(item.candidate.id), 5

        # Distinct candidate ids, identical (content_key, split, instance_ids).
        items = [
            make_item("a", "/wt-a", content_key="K", instance_ids=("1", "2")),
            make_item("b", "/wt-b", content_key="K", instance_ids=("1", "2")),
        ]
        results = run_evaluator_batch(items, runner, max_workers=2)

        assert calls == ["a"]  # only the leader was dispatched
        # Leader
        assert results[0].deduplicated_from is None
        assert results[0].num_actual_evaluations == 5
        assert results[0].error is None
        assert results[0].result is not None
        assert results[0].result.candidate_id == "a"
        # Follower: independent clone, relabeled to its own candidate, zero charge.
        assert results[1].deduplicated_from == 0
        assert results[1].num_actual_evaluations == 0
        assert results[1].result is not None
        assert results[1].result is not results[0].result  # distinct object
        assert results[1].result.candidate_id == "b"  # positional identity
        assert (
            results[1].result.instance_scores == results[0].result.instance_scores
        )  # equal payload
        # Mutating the follower must not corrupt the leader (no dict aliasing).
        results[1].result.instance_scores["0"] = 999.0
        assert results[0].result.instance_scores["0"] != 999.0

    def test_instance_id_order_is_significant(self):
        calls: list[str] = []

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            calls.append(item.candidate.id)
            return make_eval_result(item.candidate.id), 1

        items = [
            make_item("a", "/wa", content_key="K", instance_ids=("1", "2")),
            make_item("b", "/wb", content_key="K", instance_ids=("2", "1")),
        ]
        results = run_evaluator_batch(items, runner, max_workers=2)

        assert sorted(calls) == ["a", "b"]  # reordered minibatch → not deduped
        assert results[1].deduplicated_from is None

    def test_split_and_content_key_partition_dedup(self):
        calls: list[str] = []

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            calls.append(item.candidate.id)
            return make_eval_result(item.candidate.id), 1

        items = [
            make_item("a", "/wa", content_key="K", split="val", instance_ids=("1",)),
            make_item("b", "/wb", content_key="K", split="train", instance_ids=("1",)),
            make_item("c", "/wc", content_key="J", split="val", instance_ids=("1",)),
            make_item("d", "/wd", content_key="K", split="val", instance_ids=("1",)),
        ]
        results = run_evaluator_batch(items, runner, max_workers=4)

        # a/b differ by split, a/c differ by content_key → all leaders; d==a.
        assert sorted(calls) == ["a", "b", "c"]
        assert results[3].deduplicated_from == 0

    def test_dedup_preserves_budget_accounting(self):
        total_real_evals = 0
        lock = threading.Lock()

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            nonlocal total_real_evals
            assert item.instance_ids is not None
            with lock:
                total_real_evals += len(item.instance_ids)
            return make_eval_result(item.candidate.id), len(item.instance_ids)

        items = [
            make_item("a", "/wa", content_key="K", instance_ids=("1", "2", "3")),
            make_item("b", "/wb", content_key="K", instance_ids=("1", "2", "3")),  # dup
            make_item("c", "/wc", content_key="J", instance_ids=("9",)),
        ]
        results = run_evaluator_batch(items, runner, max_workers=3)

        charged = sum(r.num_actual_evaluations for r in results)
        assert total_real_evals == 4  # leader K (3) + leader J (1); dup ran 0
        assert charged == 4  # 3 (leader) + 0 (follower) + 1 (leader)


class TestRunEvaluatorBatchFailures:
    def test_ordinary_failure_is_isolated(self):
        class Boom(Exception):
            pass

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            if item.candidate.id == "bad":
                raise Boom("nope")
            return make_eval_result(item.candidate.id), 2

        items = [
            make_item("good", "/wa", instance_ids=("1",)),
            make_item("bad", "/wb", instance_ids=("2",)),
            make_item("good2", "/wc", instance_ids=("3",)),
        ]
        results = run_evaluator_batch(items, runner, max_workers=3)

        assert results[0].result is not None and results[0].error is None
        assert results[1].result is None
        assert isinstance(results[1].error, Boom)
        assert results[1].num_actual_evaluations == 0
        assert results[1].deduplicated_from is None
        assert results[2].result is not None and results[2].error is None

    def test_failed_leader_propagates_error_to_followers(self):
        class Boom(Exception):
            pass

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            raise Boom("x")

        items = [
            make_item("a", "/wa", content_key="K", instance_ids=("1",)),
            make_item("b", "/wb", content_key="K", instance_ids=("1",)),
        ]
        results = run_evaluator_batch(items, runner, max_workers=2)

        assert isinstance(results[0].error, Boom)
        assert results[0].deduplicated_from is None
        assert results[1].deduplicated_from == 0
        assert results[1].error is results[0].error
        assert results[1].result is None
        assert results[1].num_actual_evaluations == 0

    @pytest.mark.parametrize(
        "exc",
        [
            PromptArtifactCollisionError("collision"),
            ResumeIncompatibleError("resume"),
            KeyboardInterrupt(),
            SystemExit(1),
        ],
    )
    def test_fatal_exceptions_propagate(self, exc: BaseException):
        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            raise exc

        items = [make_item("a", "/wa", instance_ids=("1",))]
        with pytest.raises(type(exc)):
            run_evaluator_batch(items, runner, max_workers=1)


class TestRunEvaluatorBatchCardinalityAndConfig:
    def test_max_workers_must_be_positive(self):
        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            return make_eval_result(item.candidate.id), 1

        with pytest.raises(ValueError):
            run_evaluator_batch(
                [make_item("a", "/wa", instance_ids=("1",))], runner, max_workers=0
            )

    def test_pre_dispatch_bad_command_propagates_without_dispatch(self):
        config = make_config(command="'unterminated")  # shlex parse failure
        called: list[str] = []

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            called.append(item.candidate.id)
            return make_eval_result(item.candidate.id), 1

        with pytest.raises(EvaluatorError):
            run_evaluator_batch(
                [make_item("a", "/wa", instance_ids=("1",))],
                runner,
                max_workers=1,
                config=config,
            )
        assert called == []  # failed before any leader dispatched

    def test_negative_runner_count_is_rejected(self):
        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            return make_eval_result(item.candidate.id), -1

        with pytest.raises(ValueError, match="negative num_actual_evaluations"):
            run_evaluator_batch(
                [make_item("a", "/wa", instance_ids=("1",))], runner, max_workers=1
            )

    def test_result_positional_to_input_with_dedup_and_failure(self):
        class Boom(Exception):
            pass

        def runner(item: EvalBatchItem) -> tuple[EvalResult, int]:
            if item.candidate.id == "bad":
                raise Boom("x")
            return make_eval_result(item.candidate.id), 1

        items = [
            make_item("a", "/wa", content_key="K", instance_ids=("1",)),
            make_item("bad", "/wb", content_key="B", instance_ids=("1",)),
            make_item("a2", "/wc", content_key="K", instance_ids=("1",)),  # dup of a
        ]
        results = run_evaluator_batch(items, runner, max_workers=3)

        assert len(results) == 3
        assert [r.item.candidate.id for r in results] == ["a", "bad", "a2"]
        assert results[0].error is None and results[0].deduplicated_from is None
        assert isinstance(results[1].error, Boom)
        assert results[2].deduplicated_from == 0
        assert results[2].result is not results[0].result  # independent clone
        assert results[2].result is not None
        assert results[2].result.candidate_id == "a2"


class TestMakeDefaultBatchRunner:
    def test_counts_minibatch_and_forwards_args(self, mocker):
        fake = make_eval_result("a", instance_scores={"1": 1.0, "2": 0.0})
        m = mocker.patch("helix.executor.run_evaluator", return_value=fake)
        config = make_config()
        runner = make_default_batch_runner(config)

        item = make_item("a", "/w", split="train", instance_ids=("1", "2"))
        result, count = runner(item)

        assert result is fake
        assert count == 2
        m.assert_called_once()
        _, kwargs = m.call_args
        assert kwargs["split"] == "train"
        assert kwargs["instance_ids"] == ["1", "2"]

    def test_whole_split_counts_instance_scores(self, mocker):
        fake = make_eval_result("a", instance_scores={"1": 1.0, "2": 0.0, "3": 1.0})
        mocker.patch("helix.executor.run_evaluator", return_value=fake)
        runner = make_default_batch_runner(make_config())

        item = make_item("a", "/w", instance_ids=None)
        _, count = runner(item)

        assert count == 3


class TestEvalBatchDataclasses:
    def test_item_is_frozen_and_exposes_dedup_key(self):
        item = make_item("a", "/w", content_key="K", split="val", instance_ids=("1",))
        assert item.dedup_key.content_key == "K"
        assert item.dedup_key.split == "val"
        assert item.dedup_key.instance_ids == ("1",)
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.split = "train"  # type: ignore[misc]

    def test_result_is_frozen(self):
        item = make_item("a", "/w", instance_ids=("1",))
        res = EvalBatchResult(
            item=item,
            result=make_eval_result("a"),
            error=None,
            num_actual_evaluations=1,
            deduplicated_from=None,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.num_actual_evaluations = 0  # type: ignore[misc]
