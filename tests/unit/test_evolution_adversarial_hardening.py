"""Adversarial regression coverage for evolution durability and fatal drains."""

from __future__ import annotations

import json
import shutil
import threading
from pathlib import Path
from typing import Any, Callable

import pytest

import helix.evolution as evolution
from helix.config import HelixConfig
from helix.display import UsageStats
from helix.exceptions import HelixError, PromptArtifactCollisionError
from helix.population import Candidate, EvalResult
from helix.state import (
    BudgetState,
    EvolutionState,
    ProposalBatchRecord,
    ProposalTaskRecord,
    checkpoint_batch_after_apply,
    checkpoint_batch_before_dispatch,
    checkpoint_batch_task,
    load_eval_cache,
    load_state,
    save_state,
)
from tests.unit.test_parallel_proposals_matrix import (
    _NullDisplay,
    _candidate,
    _eval_result,
    _make_config,
)


class _InjectedCrash(BaseException):
    """Simulated process death that ordinary error isolation must not catch."""


def _patch_runtime(
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutate: Callable[..., Candidate] | None = None,
    evaluator: Callable[..., EvalResult] | None = None,
    remove: Callable[[Candidate], None] | None = None,
    content_key: Callable[[Candidate], str] | None = None,
) -> None:
    """Install deterministic filesystem-only seams around the real loop."""

    def fake_create_seed(_project_root: Path, worktrees_dir: Path) -> Candidate:
        seed_dir = worktrees_dir / "g0-s0"
        seed_dir.mkdir(parents=True, exist_ok=False)
        (seed_dir / "program.txt").write_text("seed\n")
        return _candidate("g0-s0", seed_dir, operation="seed")

    def default_mutate(**kwargs: Any) -> Candidate:
        child_id = str(kwargs["new_id"])
        child_dir = Path(kwargs["base_dir"]) / child_id
        child_dir.mkdir(parents=True, exist_ok=False)
        (child_dir / "program.txt").write_text(f"candidate={child_id}\n")
        return _candidate(child_id, child_dir, parent_id=kwargs["parent"].id)

    def default_evaluator(
        candidate: Candidate,
        _config: HelixConfig,
        split: str = "val",
        instance_ids: list[str] | None = None,
        **_kwargs: Any,
    ) -> EvalResult:
        ids = [str(value) for value in (instance_ids or ["single-task"])]
        score = 0.1 if candidate.id == "g0-s0" else 0.8
        return _eval_result(candidate.id, ids, score)

    def default_remove(candidate: Candidate) -> None:
        shutil.rmtree(candidate.worktree_path, ignore_errors=True)

    def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    actual_evaluator = evaluator or default_evaluator
    actual_mutate = mutate or default_mutate
    actual_remove = remove or default_remove
    monkeypatch.setattr(evolution, "create_seed_worktree", fake_create_seed)
    monkeypatch.setattr(evolution, "mutate", actual_mutate)
    monkeypatch.setattr("helix.mutator.mutate", actual_mutate)
    monkeypatch.setattr(evolution, "run_evaluator", actual_evaluator)
    monkeypatch.setattr("helix.executor.run_evaluator", actual_evaluator)
    monkeypatch.setattr(evolution, "remove_worktree", actual_remove)
    monkeypatch.setattr("helix.worktree.remove_worktree", actual_remove)
    monkeypatch.setattr(
        evolution, "snapshot_candidate", lambda candidate, _message: candidate.id
    )
    monkeypatch.setattr(evolution, "HelixLiveDisplay", _NullDisplay)
    monkeypatch.setattr(evolution, "_check_evaluator_script_exists", noop)
    monkeypatch.setattr(evolution, "_refresh_protected_evaluator_files", noop)
    monkeypatch.setattr(
        evolution, "_refresh_and_snapshot_protected_evaluator_files", noop
    )
    monkeypatch.setattr(
        evolution, "_build_evaluator_integrity_manifest", lambda **_kwargs: {}
    )
    monkeypatch.setattr(evolution, "_write_evaluator_integrity_manifest", noop)
    monkeypatch.setattr(evolution, "_detect_evaluator_tamper", lambda *_args: [])
    monkeypatch.setattr(
        evolution,
        "_candidate_content_key",
        content_key or (lambda candidate: candidate.id),
    )
    for name in (
        "set_phase",
        "print_info",
        "print_success",
        "print_warning",
        "print_error",
        "render_budget",
        "render_generation",
        "render_frontier_table",
    ):
        monkeypatch.setattr(evolution, name, noop)


def _new_project(root: Path) -> tuple[Path, Path]:
    project_root = root / "project"
    project_root.mkdir(parents=True)
    return project_root, project_root / ".helix"


def _batch(
    batch_id: str = "g1-proposals", *, child_count: int = 2
) -> ProposalBatchRecord:
    return ProposalBatchRecord(
        batch_id=batch_id,
        generation=1,
        p=1,
        n=child_count,
        tasks=[
            ProposalTaskRecord(
                batch_id=batch_id,
                p=1,
                n=child_count,
                task_index=index,
                parent_group=0,
                mutation_index=index,
                parent_id="g0-s0",
                child_id=f"g1-s{index + 1}",
            )
            for index in range(child_count)
        ],
    )


def test_crash_between_scheduler_and_plan_save_can_only_observe_both(
    tmp_path: Path,
) -> None:
    """The old first save's crash point is now one combined durable image."""
    state = EvolutionState(
        generation=1,
        frontier=["g0-s0"],
        instance_scores={},
        budget=BudgetState(evaluations=3),
        config_hash="atomic-plan",
    )
    saves = 0

    def persist_then_crash(value: EvolutionState) -> None:
        nonlocal saves
        saves += 1
        save_state(value, tmp_path)
        raise _InjectedCrash()

    with pytest.raises(_InjectedCrash):
        evolution._checkpoint_scheduler_and_batch_plan(
            state,
            tmp_path,
            scheduler_state={"frontier_rng_state": ["advanced"]},
            batch=_batch(),
            max_evaluations=100,
            max_in_flight_evaluations=8,
            saver=persist_then_crash,
        )

    assert saves == 1
    persisted = load_state(tmp_path)
    assert persisted is not None
    assert persisted.scheduler_state == {"frontier_rng_state": ["advanced"]}
    assert [batch.batch_id for batch in persisted.proposal_batches] == ["g1-proposals"]
    assert [task.child_id for task in persisted.proposal_batches[0].tasks] == [
        "g1-s1",
        "g1-s2",
    ]


def test_crash_injection_after_every_proposal_state_barrier_resumes_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Replay a process death after every state barrier reached by one batch."""
    _patch_runtime(monkeypatch)
    config = _make_config(
        p=1,
        n=1,
        train_size=1,
        val_size=1,
        minibatch_size=1,
        max_workers=1,
    )

    control_root, control_base = _new_project(tmp_path / "control")
    barrier_count = 0

    def count_barriers(_name: str, state: EvolutionState) -> None:
        nonlocal barrier_count
        if state.proposal_batches:
            barrier_count += 1

    monkeypatch.setattr(evolution, "_DURABLE_BARRIER_HOOK", count_barriers)
    evolution.run_evolution(config, control_root, control_base)
    assert barrier_count >= 5

    for target in range(1, barrier_count + 1):
        project_root, base_dir = _new_project(tmp_path / f"barrier-{target}")
        observed = 0

        def crash_at_target(_name: str, state: EvolutionState) -> None:
            nonlocal observed
            if not state.proposal_batches:
                return
            observed += 1
            if observed == target:
                raise _InjectedCrash()

        monkeypatch.setattr(evolution, "_DURABLE_BARRIER_HOOK", crash_at_target)
        with pytest.raises(_InjectedCrash):
            evolution.run_evolution(config, project_root, base_dir)

        crashed = load_state(project_root)
        assert crashed is not None
        assert crashed.scheduler_state
        assert len(crashed.proposal_batches) == 1

        monkeypatch.setattr(evolution, "_DURABLE_BARRIER_HOOK", None)
        evolution.run_evolution(config, project_root, base_dir)
        resumed = load_state(project_root)
        assert resumed is not None
        batch = resumed.proposal_batches[0]
        assert batch.phase == "complete"
        assert all(task.budget_accounted for task in batch.tasks)
        assert all(task.cleanup != "failed" for task in batch.tasks)
        assert len({task.child_id for task in batch.tasks}) == len(batch.tasks)
        assert batch.budget_state_before_dispatch is not None
        assert sum(task.budget_charge.evaluations for task in batch.tasks) == (
            resumed.budget.evaluations - batch.budget_state_before_dispatch.evaluations
        )


def test_failed_resume_cleanup_blocks_dispatch_then_retries_idempotently(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _make_config(p=1, n=1, train_size=1, val_size=1, minibatch_size=1)
    project_root, base_dir = _new_project(tmp_path)
    worktrees_dir = base_dir / "worktrees"
    worktrees_dir.mkdir(parents=True)
    seed_dir = worktrees_dir / "g0-s0"
    seed_dir.mkdir()
    child_dir = worktrees_dir / "g1-s1"
    child_dir.mkdir()
    (base_dir / "evaluations").mkdir()
    seed_result = _eval_result("g0-s0", ["0"], 0.1)
    (base_dir / "evaluations" / "g0-s0.json").write_text(
        json.dumps(seed_result.to_dict())
    )
    state = EvolutionState(
        generation=1,
        frontier=["g0-s0"],
        instance_scores={"g0-s0": {"0": 0.1}},
        budget=BudgetState(evaluations=5),
        config_hash=evolution._config_hash(config),
        frontier_type="instance",
        resume_semantics=evolution._resume_semantics(config),
    )
    checkpoint_batch_before_dispatch(
        state,
        project_root,
        _batch(child_count=1),
        max_evaluations=100,
        max_in_flight_evaluations=4,
    )

    cleanup_attempts: list[str] = []

    def flaky_remove(candidate: Candidate) -> None:
        cleanup_attempts.append(candidate.id)
        if len(cleanup_attempts) == 1:
            raise OSError("transient git worktree lock")
        shutil.rmtree(candidate.worktree_path)

    _patch_runtime(monkeypatch, remove=flaky_remove)
    with pytest.raises(HelixError, match="Cannot resume until"):
        evolution.run_evolution(config, project_root, base_dir)

    failed = load_state(project_root)
    assert failed is not None
    failed_task = failed.proposal_batches[0].tasks[0]
    assert failed.proposal_batches[0].phase == "interrupted"
    assert failed_task.cleanup == "failed"
    assert not failed_task.budget_accounted
    assert child_dir.exists()

    result = evolution.run_evolution(config, project_root, base_dir)
    assert result.id == "g0-s0"
    recovered = load_state(project_root)
    assert recovered is not None
    recovered_task = recovered.proposal_batches[0].tasks[0]
    assert recovered.proposal_batches[0].phase == "complete"
    assert recovered_task.cleanup == "removed"
    assert recovered_task.budget_accounted
    assert recovered_task.budget_charge == BudgetState()
    assert recovered.budget == BudgetState(evaluations=5)
    assert cleanup_attempts == ["g1-s1", "g1-s1"]

    evolution.run_evolution(config, project_root, base_dir)
    repeated = load_state(project_root)
    assert repeated == recovered
    assert cleanup_attempts == ["g1-s1", "g1-s1"]


def test_interrupted_earlier_batch_is_bounded_by_next_batch_budget_snapshot(
    tmp_path: Path,
) -> None:
    """Later durable charges cannot leak into an earlier interrupted slot."""
    state = EvolutionState(
        generation=2,
        frontier=["g0-s0"],
        instance_scores={},
        budget=BudgetState(evaluations=5, input_tokens=2),
        config_hash="multi-batch-bound",
    )
    first = ProposalBatchRecord(
        batch_id="g1-b0",
        generation=1,
        p=1,
        n=1,
        tasks=[
            ProposalTaskRecord(
                batch_id="g1-b0",
                p=1,
                n=1,
                task_index=0,
                parent_group=0,
                mutation_index=0,
                parent_id="g0-s0",
                child_id="g1-s1",
            )
        ],
    )
    checkpoint_batch_before_dispatch(state, tmp_path, first)

    # The first batch durably consumed 5 evaluations / 5 tokens without
    # terminalizing its task.  The second batch's pre-dispatch snapshot is the
    # exact upper bound for those missing facts.
    state.budget = BudgetState(evaluations=10, input_tokens=7)
    second = ProposalBatchRecord(
        batch_id="g2-b0",
        generation=2,
        p=1,
        n=1,
        tasks=[
            ProposalTaskRecord(
                batch_id="g2-b0",
                p=1,
                n=1,
                task_index=0,
                parent_group=0,
                mutation_index=0,
                parent_id="g0-s0",
                child_id="g2-s2",
            )
        ],
    )
    checkpoint_batch_before_dispatch(state, tmp_path, second)
    state.budget = BudgetState(evaluations=19, input_tokens=20)
    checkpoint_batch_task(
        state,
        tmp_path,
        batch_id=second.batch_id,
        task_index=0,
        status="failed",
        cleanup="missing",
        budget_charge=BudgetState(evaluations=9, input_tokens=13),
        budget_accounted=True,
    )
    checkpoint_batch_after_apply(state, tmp_path, batch_id=second.batch_id)

    worktrees_dir = tmp_path / ".helix" / "worktrees"
    worktrees_dir.mkdir(parents=True)
    reports = evolution.reconcile_interrupted_batches(
        state,
        tmp_path,
        worktrees_dir=worktrees_dir,
        cleanup_worktree=lambda _candidate_id, _path: True,
    )

    assert len(reports) == 1
    first_task = first.tasks[0]
    assert first_task.budget_accounted
    assert first_task.budget_charge == BudgetState(
        evaluations=5,
        input_tokens=5,
    )
    assert state.budget == BudgetState(evaluations=19, input_tokens=20)
    completed_first = checkpoint_batch_after_apply(
        state, tmp_path, batch_id=first.batch_id
    )
    assert completed_first.budget_after_apply == 10


@pytest.mark.parametrize("validation_failure", ["wrong_id", "tamper_detector"])
def test_successful_mutation_usage_survives_post_validation_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    validation_failure: str,
) -> None:
    usage = UsageStats(
        input_tokens=101,
        output_tokens=29,
        cached_input_tokens=7,
        cache_creation_input_tokens=5,
        cache_read_input_tokens=3,
        reasoning_tokens=11,
        cost_usd=0.42,
    )
    removed: list[str] = []

    def mutate_with_usage(**kwargs: Any) -> Candidate:
        reserved_id = str(kwargs["new_id"])
        child_id = "wrong-id" if validation_failure == "wrong_id" else reserved_id
        child_dir = Path(kwargs["base_dir"]) / child_id
        child_dir.mkdir()
        child = Candidate(
            id=child_id,
            worktree_path=str(child_dir),
            branch_name=f"helix/{child_id}",
            generation=1,
            parent_id=kwargs["parent"].id,
            parent_ids=[kwargs["parent"].id],
            operation="mutation",
        )
        child.usage = usage
        return child

    def remove(candidate: Candidate) -> None:
        removed.append(candidate.id)
        shutil.rmtree(candidate.worktree_path)

    _patch_runtime(monkeypatch, mutate=mutate_with_usage, remove=remove)
    if validation_failure == "tamper_detector":
        monkeypatch.setattr(
            evolution,
            "_detect_evaluator_tamper",
            lambda *_args: (_ for _ in ()).throw(ValueError("detector exploded")),
        )

    project_root, base_dir = _new_project(tmp_path)
    config = _make_config(p=1, n=1, train_size=1, val_size=1, minibatch_size=1)
    with pytest.raises(ValueError):
        evolution.run_evolution(config, project_root, base_dir)

    persisted = load_state(project_root)
    assert persisted is not None
    task = persisted.proposal_batches[0].tasks[0]
    assert persisted.budget.input_tokens == usage.input_tokens
    assert persisted.budget.output_tokens == usage.output_tokens
    assert persisted.budget.cached_input_tokens == usage.cached_input_tokens
    assert persisted.budget.cache_creation_input_tokens == (
        usage.cache_creation_input_tokens
    )
    assert persisted.budget.cache_read_input_tokens == usage.cache_read_input_tokens
    assert persisted.budget.reasoning_tokens == usage.reasoning_tokens
    assert persisted.budget.cost_usd == pytest.approx(usage.cost_usd)
    assert task.budget_charge.input_tokens == usage.input_tokens
    assert task.budget_charge.output_tokens == usage.output_tokens
    assert task.budget_charge.cost_usd == pytest.approx(usage.cost_usd)
    assert task.budget_accounted
    assert task.status == "failed"
    assert task.cleanup == "removed"
    assert removed == ["wrong-id" if validation_failure == "wrong_id" else "g1-s1"]


def test_fatal_full_evaluator_preserves_negative_success_and_dedup_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fatal = PromptArtifactCollisionError("fatal full validation")
    negative_full_finished = threading.Event()
    good_full_finished = threading.Event()
    full_completion_order: list[str] = []
    evaluator_calls: list[tuple[str, str, tuple[str, ...]]] = []
    removed: list[str] = []

    def evaluator(
        candidate: Candidate,
        _config: HelixConfig,
        split: str = "val",
        instance_ids: list[str] | None = None,
        **_kwargs: Any,
    ) -> EvalResult:
        ids = tuple(str(value) for value in (instance_ids or ["single-task"]))
        evaluator_calls.append((candidate.id, split, ids))
        is_full_child = candidate.id.startswith("g1-s") and split == "val"
        if is_full_child and candidate.id == "g1-s1":
            assert good_full_finished.wait(timeout=2)
            full_completion_order.append(candidate.id)
            raise fatal
        if is_full_child and candidate.id == "g1-s2":
            full_completion_order.append(candidate.id)
            negative_full_finished.set()
        if is_full_child and candidate.id == "g1-s3":
            assert negative_full_finished.wait(timeout=2)
            full_completion_order.append(candidate.id)
            good_full_finished.set()
        score = 0.1 if candidate.id == "g0-s0" else 0.8
        return _eval_result(candidate.id, ids, score)

    def mutate_shared(**kwargs: Any) -> Candidate:
        child_id = str(kwargs["new_id"])
        child_dir = Path(kwargs["base_dir"]) / child_id
        child_dir.mkdir()
        (child_dir / "program.txt").write_text("shared\n")
        return _candidate(child_id, child_dir, parent_id=kwargs["parent"].id)

    def content_key(candidate: Candidate) -> str:
        if candidate.id in {"g1-s3", "g1-s4"}:
            return "shared-success"
        return candidate.id

    def remove(candidate: Candidate) -> None:
        removed.append(candidate.id)
        shutil.rmtree(candidate.worktree_path, ignore_errors=True)

    _patch_runtime(
        monkeypatch,
        mutate=mutate_shared,
        evaluator=evaluator,
        remove=remove,
        content_key=content_key,
    )
    original_cached_evaluate_batch = evolution._cached_evaluate_batch

    def negative_count_for_one_full_leader(
        candidate: Candidate,
        example_ids: list[str],
        cache: Any,
        config: HelixConfig,
        split: str,
        project_root: Path,
    ) -> tuple[EvalResult, int]:
        result, count = original_cached_evaluate_batch(
            candidate,
            example_ids,
            cache,
            config,
            split,
            project_root,
        )
        if candidate.id == "g1-s2" and split == "val":
            return result, -7
        return result, count

    monkeypatch.setattr(
        evolution, "_cached_evaluate_batch", negative_count_for_one_full_leader
    )
    project_root, base_dir = _new_project(tmp_path)
    config = _make_config(
        p=1,
        n=4,
        train_size=1,
        val_size=2,
        minibatch_size=1,
        cache=True,
        max_workers=3,
    )

    with pytest.raises(PromptArtifactCollisionError) as raised:
        evolution.run_evolution(config, project_root, base_dir)
    assert raised.value is fatal
    assert full_completion_order == ["g1-s2", "g1-s3", "g1-s1"]

    persisted = load_state(project_root)
    assert persisted is not None
    batch = persisted.proposal_batches[0]
    assert persisted.budget.evaluations == 8
    assert batch.budget_state_before_dispatch is not None
    assert sum(task.budget_charge.evaluations for task in batch.tasks) == (
        persisted.budget.evaluations - batch.budget_state_before_dispatch.evaluations
    )
    assert [task.budget_charge.evaluations for task in batch.tasks] == [2, 1, 3, 0]
    assert all(task.budget_accounted for task in batch.tasks)
    assert all(task.cleanup in {"removed", "missing"} for task in batch.tasks)
    assert sorted(removed) == ["g1-s1", "g1-s2", "g1-s3", "g1-s4"]
    assert not any(
        (base_dir / "worktrees" / candidate_id).exists()
        for candidate_id in ("g1-s1", "g1-s2", "g1-s3", "g1-s4")
    )

    child_full_calls = [
        call
        for call in evaluator_calls
        if call[0].startswith("g1-s") and call[1] == "val"
    ]
    assert sorted(call[0] for call in child_full_calls) == [
        "g1-s1",
        "g1-s2",
        "g1-s3",
    ]
    cache = load_eval_cache(project_root)
    assert cache is not None
    assert len(cache) == 9
