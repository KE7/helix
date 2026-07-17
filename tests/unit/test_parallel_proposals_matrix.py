"""Black-box contract matrix for the P-by-N proposal scheduler.

These tests deliberately exercise :func:`helix.evolution.run_evolution` rather
than reimplementing the scheduler in a test double.  The expensive external
boundaries remain mocked at the seams HELIX already exposes: parent selection,
the coding-agent mutator, evaluator invocation, snapshots, and worktree
removal.  Everything observable above those seams -- planning order, stable
IDs, selection, budget/cache accounting, persistence, and resume cleanup -- is
owned by the production runtime.

The module is collected on the pre-P-by-N baseline too.  It is skipped there
with one explicit reason instead of failing import/collection merely because
``EvolutionConfig.mutations_per_parent`` has not landed yet.  Once that public
configuration field exists, every assertion becomes active.
"""

from __future__ import annotations

import json
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import pytest

from helix.config import DatasetConfig, EvaluatorConfig, EvolutionConfig, HelixConfig
from helix.evolution import run_evolution
from helix.population import Candidate, EvalResult, HelixResult, ParetoFrontier
from helix.state import BudgetState, EvolutionState, load_eval_cache, load_state
from helix.trace import Event, TRACE


_HAS_P_BY_N_CONFIG = "mutations_per_parent" in EvolutionConfig.model_fields

pytestmark = pytest.mark.skipif(
    not _HAS_P_BY_N_CONFIG,
    reason=(
        "expected pre-implementation baseline: "
        "EvolutionConfig.mutations_per_parent is not available"
    ),
)


def _slot(candidate_id: str) -> int:
    """Return the one-based stable proposal slot from ``gN-sM``."""
    return int(candidate_id.rsplit("-s", 1)[1])


def _candidate(
    candidate_id: str,
    worktree: Path,
    *,
    parent_id: str | None = None,
    operation: str = "mutation",
) -> Candidate:
    return Candidate(
        id=candidate_id,
        worktree_path=str(worktree),
        branch_name=f"helix/{candidate_id}",
        generation=int(candidate_id.split("-", 1)[0].lstrip("g")),
        parent_id=parent_id,
        parent_ids=[] if parent_id is None else [parent_id],
        operation=operation,
    )


def _eval_result(candidate_id: str, ids: Sequence[str], score: float) -> EvalResult:
    return EvalResult(
        candidate_id=candidate_id,
        scores={},
        asi={},
        instance_scores={example_id: score for example_id in ids},
    )


class _NullDisplay:
    """Minimal live-display boundary; assertions inspect durable output instead."""

    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.gen = 0
        self.current_usage = None

    def __enter__(self) -> _NullDisplay:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def update(self, **_kwargs: Any) -> None:
        return None


class _CompletionGate:
    """Event-driven completion controller with no timing-based ordering claim."""

    def __init__(self, candidate_ids: Sequence[str]) -> None:
        self._expected = tuple(candidate_ids)
        self._lock = threading.Lock()
        self._started: set[str] = set()
        self.all_started = threading.Event()
        self.release = {
            candidate_id: threading.Event() for candidate_id in candidate_ids
        }
        self.finished = {
            candidate_id: threading.Event() for candidate_id in candidate_ids
        }
        self.completion_order: list[str] = []

    def wait_for_release(self, candidate_id: str) -> None:
        assert candidate_id in self.release, f"unexpected proposal id {candidate_id}"
        with self._lock:
            self._started.add(candidate_id)
            if self._started == set(self._expected):
                self.all_started.set()

        if not self.release[candidate_id].wait(timeout=10):
            raise AssertionError(f"test did not release proposal {candidate_id}")

        with self._lock:
            self.completion_order.append(candidate_id)
        self.finished[candidate_id].set()


@dataclass
class _Scenario:
    result: HelixResult
    project_root: Path
    base_dir: Path
    selector_parent_ids: list[str]
    mutation_parent_by_child: dict[str, str]
    evaluator_calls: list[tuple[str, str, tuple[str, ...]]]
    removed_ids: list[str]
    snapshotted_ids: list[str]
    events: list[Event]
    completion_order: list[str]

    @property
    def child_ids(self) -> list[str]:
        return sorted(self.mutation_parent_by_child, key=_slot)

    @property
    def lineage(self) -> list[dict[str, Any]]:
        path = self.base_dir / "lineage.json"
        if not path.exists():
            return []
        loaded = json.loads(path.read_text())
        assert isinstance(loaded, list)
        return loaded

    @property
    def task_rows(self) -> list[dict[str, Any]]:
        """Find the durable per-slot rows without depending on their container file.

        The work plan fixes the information contract, not whether task rows live
        in ``state.json``, a batch journal, or attempt artifacts.  Search all
        durable JSON and choose the richest row for each reserved child.
        """
        expected = set(self.child_ids)
        richest: dict[str, dict[str, Any]] = {}
        for path in self.base_dir.rglob("*.json"):
            try:
                payload = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            for row in _walk_dicts(payload):
                child_id = _child_id(row)
                if child_id not in expected or _parent_group(row) is None:
                    continue
                if len(row) > len(richest.get(child_id, {})):
                    richest[child_id] = row
        return [
            richest[candidate_id]
            for candidate_id in self.child_ids
            if candidate_id in richest
        ]


def _walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _first(mapping: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _child_id(row: Mapping[str, Any]) -> str | None:
    value = _first(row, ("child_id", "reserved_child_id", "candidate_id"))
    return value if isinstance(value, str) else None


def _parent_group(row: Mapping[str, Any]) -> int | None:
    value = _first(row, ("parent_group_index", "parent_group"))
    return value if isinstance(value, int) else None


def _mutation_index(row: Mapping[str, Any]) -> int | None:
    value = _first(row, ("mutation_index", "mutation_index_within_parent"))
    return value if isinstance(value, int) else None


def _status(row: Mapping[str, Any]) -> str:
    value = _first(row, ("status", "terminal_status", "outcome"))
    return str(value).lower() if value is not None else ""


def _make_config(
    *,
    p: int,
    n: int | None,
    seed: int = 17,
    selection: str = "all_improvements",
    top_k: int | None = None,
    train_size: int = 16,
    val_size: int = 2,
    minibatch_size: int = 2,
    cache: bool = False,
    max_workers: int | None = None,
    max_evaluations: int = 10_000,
) -> HelixConfig:
    evolution: dict[str, Any] = {
        "max_generations": 1,
        "max_evaluations": max_evaluations,
        "perfect_score_threshold": None,
        "num_parallel_proposals": p,
        "proposal_selection": selection,
        "minibatch_size": minibatch_size,
        "cache_evaluation": cache,
        "max_workers": max_workers or max(1, p * (n or 1)),
        "frontier_type": "instance",
    }
    if n is not None:
        evolution["mutations_per_parent"] = n
    if top_k is not None:
        evolution["proposal_top_k"] = top_k

    return HelixConfig(
        objective="parallel proposal black-box matrix",
        evaluator=EvaluatorConfig(command="python evaluate.py"),
        dataset=DatasetConfig(train_size=train_size, val_size=val_size),
        evolution=EvolutionConfig(**evolution),
        rng_seed=seed,
    )


def _run_scenario(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    p: int,
    n: int | None,
    scores: Sequence[float] | None = None,
    selection: str = "all_improvements",
    top_k: int | None = None,
    seed: int = 17,
    completion_order: Sequence[str] | None = None,
    fail_child_eval: frozenset[str] = frozenset(),
    cache: bool = False,
    shared_child_content: bool = False,
    train_size: int = 16,
    val_size: int = 2,
    minibatch_size: int = 2,
    max_workers: int | None = None,
    max_evaluations: int = 10_000,
) -> _Scenario:
    """Run one real scheduler step behind deterministic I/O seams."""
    effective_n = 1 if n is None else n
    task_count = p * effective_n
    child_ids = [f"g1-s{index}" for index in range(1, task_count + 1)]
    score_vector = list(scores or [0.8] * task_count)
    assert len(score_vector) == task_count

    project_root = root / "project"
    project_root.mkdir(parents=True)
    base_dir = project_root / ".helix"
    worktrees_dir = base_dir / "worktrees"

    selector_parent_ids: list[str] = []
    mutation_parent_by_child: dict[str, str] = {}
    evaluator_calls: list[tuple[str, str, tuple[str, ...]]] = []
    removed_ids: list[str] = []
    snapshotted_ids: list[str] = []
    calls_lock = threading.Lock()

    gate = _CompletionGate(child_ids) if completion_order is not None else None

    def fake_create_seed(_project_root: Path, _worktrees_dir: Path) -> Candidate:
        seed_dir = worktrees_dir / "g0-s0"
        seed_dir.mkdir(parents=True, exist_ok=False)
        (seed_dir / "program.txt").write_text("seed\n")
        return _candidate("g0-s0", seed_dir, parent_id=None, operation="seed")

    def fake_mutate(**kwargs: Any) -> Candidate:
        parent = kwargs["parent"]
        new_id = kwargs["new_id"]
        child_dir = Path(kwargs["base_dir"]) / new_id
        child_dir.mkdir(parents=True, exist_ok=False)
        content = "shared\n" if shared_child_content else f"slot={new_id}\n"
        (child_dir / "program.txt").write_text(content)
        child = _candidate(new_id, child_dir, parent_id=parent.id)
        with calls_lock:
            mutation_parent_by_child[new_id] = parent.id
        if gate is not None:
            gate.wait_for_release(new_id)
        return child

    def fake_run_evaluator(
        candidate: Candidate,
        _config: HelixConfig,
        split: str = "val",
        instance_ids: list[str] | None = None,
        **_kwargs: Any,
    ) -> EvalResult:
        ids = tuple(str(value) for value in (instance_ids or ["single-task"]))
        with calls_lock:
            evaluator_calls.append((candidate.id, split, ids))

        if candidate.id in fail_child_eval and split == "train":
            raise RuntimeError(f"simulated child evaluator failure for {candidate.id}")
        if candidate.id == "g0-s0" and split == "val":
            score = 0.1
        elif candidate.id.startswith("g1-s"):
            score = score_vector[_slot(candidate.id) - 1]
        else:
            score = 0.2
        return _eval_result(candidate.id, ids, score)

    def fake_remove(candidate: Candidate) -> None:
        with calls_lock:
            removed_ids.append(candidate.id)
        shutil.rmtree(candidate.worktree_path, ignore_errors=True)

    def fake_snapshot(candidate: Candidate, _message: str) -> str:
        with calls_lock:
            snapshotted_ids.append(candidate.id)
        return f"snapshot-{candidate.id}"

    original_select_parent = ParetoFrontier.select_parent

    def select_parent(frontier: ParetoFrontier) -> Candidate:
        parent = original_select_parent(frontier)
        selector_parent_ids.append(parent.id)
        return parent

    def content_key(candidate: Candidate) -> str:
        if shared_child_content and candidate.id.startswith("g1-s"):
            return "shared-child-content"
        return candidate.id

    def noop(*_args: Any, **_kwargs: Any) -> None:
        return None

    with monkeypatch.context() as patcher:
        patcher.setattr("helix.evolution.create_seed_worktree", fake_create_seed)
        patcher.setattr("helix.evolution.mutate", fake_mutate)
        patcher.setattr("helix.mutator.mutate", fake_mutate)
        patcher.setattr("helix.evolution.run_evaluator", fake_run_evaluator)
        patcher.setattr("helix.executor.run_evaluator", fake_run_evaluator)
        patcher.setattr("helix.evolution.remove_worktree", fake_remove)
        patcher.setattr("helix.worktree.remove_worktree", fake_remove)
        patcher.setattr("helix.evolution.snapshot_candidate", fake_snapshot)
        patcher.setattr("helix.worktree.snapshot_candidate", fake_snapshot)
        patcher.setattr("helix.evolution.HelixLiveDisplay", _NullDisplay)
        patcher.setattr("helix.evolution._check_evaluator_script_exists", noop)
        patcher.setattr("helix.evolution._refresh_protected_evaluator_files", noop)
        patcher.setattr(
            "helix.evolution._refresh_and_snapshot_protected_evaluator_files", noop
        )
        patcher.setattr(
            "helix.evolution._build_evaluator_integrity_manifest",
            lambda **_kwargs: {},
        )
        patcher.setattr("helix.evolution._write_evaluator_integrity_manifest", noop)
        patcher.setattr("helix.evolution._detect_evaluator_tamper", lambda *_a: [])
        patcher.setattr("helix.evolution._candidate_content_key", content_key)
        patcher.setattr(ParetoFrontier, "select_parent", select_parent)
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
            patcher.setattr(f"helix.evolution.{name}", noop)

        config = _make_config(
            p=p,
            n=n,
            seed=seed,
            selection=selection,
            top_k=top_k,
            train_size=train_size,
            val_size=val_size,
            minibatch_size=minibatch_size,
            cache=cache,
            max_workers=max_workers,
            max_evaluations=max_evaluations,
        )

        def invoke() -> tuple[HelixResult, list[Event]]:
            with TRACE.record() as recorded:
                result = run_evolution(config, project_root, base_dir)
                return result, list(recorded)

        if gate is None:
            result, events = invoke()
        else:
            assert completion_order is not None
            with ThreadPoolExecutor(max_workers=1) as driver:
                future = driver.submit(invoke)
                assert gate.all_started.wait(timeout=10), (
                    "all proposal mutations must become active; this also proves "
                    "the worker bound permits overlap without using sleep"
                )
                for candidate_id in completion_order:
                    gate.release[candidate_id].set()
                    assert gate.finished[candidate_id].wait(timeout=10)
                result, events = future.result(timeout=10)

    return _Scenario(
        result=result,
        project_root=project_root,
        base_dir=base_dir,
        selector_parent_ids=selector_parent_ids,
        mutation_parent_by_child=mutation_parent_by_child,
        evaluator_calls=evaluator_calls,
        removed_ids=removed_ids,
        snapshotted_ids=snapshotted_ids,
        events=events,
        completion_order=[] if gate is None else list(gate.completion_order),
    )


def _group_coordinates(scenario: _Scenario) -> list[tuple[int | None, int | None]]:
    assert len(scenario.task_rows) == len(scenario.child_ids), (
        "every planned P*N slot must have one durable task record"
    )
    return [(_parent_group(row), _mutation_index(row)) for row in scenario.task_rows]


def _normalized(scenario: _Scenario) -> dict[str, Any]:
    """Normalize away filesystem roots while retaining semantic state."""
    budget = asdict(scenario.result.budget)
    lineage = [
        {
            "id": row["id"],
            "parent": row.get("parent"),
            "parents": row.get("parents", []),
            "operation": row["operation"],
            "generation": row["generation"],
        }
        for row in scenario.lineage
    ]
    tasks = [
        {
            "child_id": _child_id(row),
            "parent_group": _parent_group(row),
            "mutation_index": _mutation_index(row),
            "status": _status(row),
        }
        for row in scenario.task_rows
    ]
    minibatches = sorted(
        (
            candidate_id,
            split,
            ids,
        )
        for candidate_id, split, ids in scenario.evaluator_calls
    )
    return {
        "frontier_ids": scenario.result.frontier_ids,
        "best_id": scenario.result.id,
        "parents": scenario.result.parents,
        "instance_scores": scenario.result.instance_scores,
        "budget": budget,
        "lineage": lineage,
        "tasks": tasks,
        "minibatches": minibatches,
    }


def test_1x4_and_4x1_have_equal_width_but_different_selection_and_grouping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    one_by_four = _run_scenario(tmp_path / "1x4", monkeypatch, p=1, n=4)
    four_by_one = _run_scenario(tmp_path / "4x1", monkeypatch, p=4, n=1)

    assert len(one_by_four.child_ids) == len(four_by_one.child_ids) == 4
    assert one_by_four.selector_parent_ids == ["g0-s0"]
    assert four_by_one.selector_parent_ids == ["g0-s0"] * 4
    assert _group_coordinates(one_by_four) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    assert _group_coordinates(four_by_one) == [(0, 0), (1, 0), (2, 0), (3, 0)]


def test_duplicate_parent_selections_remain_distinct_parent_groups(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _run_scenario(tmp_path, monkeypatch, p=2, n=2)

    assert scenario.selector_parent_ids == ["g0-s0", "g0-s0"]
    assert [scenario.mutation_parent_by_child[cid] for cid in scenario.child_ids] == [
        "g0-s0",
        "g0-s0",
        "g0-s0",
        "g0-s0",
    ]
    assert _group_coordinates(scenario) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert len(set(scenario.child_ids)) == 4


def test_reverse_completion_preserves_ids_lineage_task_order_and_apply_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reverse = ["g1-s4", "g1-s3", "g1-s2", "g1-s1"]
    scenario = _run_scenario(
        tmp_path,
        monkeypatch,
        p=1,
        n=4,
        scores=[0.4, 0.5, 0.6, 0.7],
        completion_order=reverse,
        max_workers=4,
    )

    assert scenario.completion_order == reverse
    assert scenario.result.frontier_ids == ["g0-s0", "g1-s1", "g1-s2", "g1-s3", "g1-s4"]
    assert [row["id"] for row in scenario.lineage] == [
        "g0-s0",
        "g1-s1",
        "g1-s2",
        "g1-s3",
        "g1-s4",
    ]
    assert [_child_id(row) for row in scenario.task_rows] == scenario.child_ids
    assert scenario.snapshotted_ids == scenario.child_ids


def test_best_improvement_uses_stable_first_tie_and_cleans_other_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reverse = ["g1-s4", "g1-s3", "g1-s2", "g1-s1"]
    scenario = _run_scenario(
        tmp_path,
        monkeypatch,
        p=1,
        n=4,
        scores=[0.8, 0.8, 0.8, 0.8],
        selection="best_improvement",
        completion_order=reverse,
        max_workers=4,
    )

    assert scenario.result.frontier_ids == ["g0-s0", "g1-s1"]
    # HELIX snapshots every viable mutation before child scoring so content
    # hashes and lineage remain stable; selection then retains only the first
    # tied improvement and cleans the other snapshotted worktrees.
    assert scenario.snapshotted_ids == [
        "g1-s1",
        "g1-s2",
        "g1-s3",
        "g1-s4",
    ]
    assert set(scenario.removed_ids) == {"g1-s2", "g1-s3", "g1-s4"}
    for candidate_id in scenario.removed_ids:
        assert not (scenario.base_dir / "worktrees" / candidate_id).exists()


def test_one_child_evaluation_failure_does_not_discard_successful_siblings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _run_scenario(
        tmp_path,
        monkeypatch,
        p=1,
        n=3,
        scores=[0.6, 0.7, 0.8],
        fail_child_eval=frozenset({"g1-s2"}),
        max_workers=3,
    )

    assert scenario.result.frontier_ids == ["g0-s0", "g1-s1", "g1-s3"]
    assert set(scenario.removed_ids) == {"g1-s2"}
    assert not (scenario.base_dir / "worktrees" / "g1-s2").exists()
    failed_row = next(row for row in scenario.task_rows if _child_id(row) == "g1-s2")
    assert any(word in _status(failed_row) for word in ("fail", "error"))


def test_top_k_cleans_every_unselected_worktree_and_records_terminal_slots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _run_scenario(
        tmp_path,
        monkeypatch,
        p=2,
        n=2,
        scores=[0.4, 0.9, 0.8, 0.7],
        selection="top_k",
        top_k=2,
    )

    assert set(scenario.result.frontier_ids[1:]) == {"g1-s2", "g1-s3"}
    assert set(scenario.removed_ids) == {"g1-s1", "g1-s4"}
    assert len(scenario.task_rows) == 4
    assert all(_status(row) for row in scenario.task_rows)
    for candidate_id in scenario.removed_ids:
        assert not (scenario.base_dir / "worktrees" / candidate_id).exists()


def test_exact_budget_and_cache_accounting_for_identical_sibling_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scenario = _run_scenario(
        tmp_path,
        monkeypatch,
        p=1,
        n=2,
        scores=[0.8, 0.8],
        train_size=4,
        val_size=2,
        minibatch_size=2,
        cache=True,
        shared_child_content=True,
    )

    # seed full-val: 2; two distinct parent minibatches: 4; two distinct
    # child minibatches: 4; one deduplicated/cached shared full-val: 2.
    assert scenario.result.budget.evaluations == 12
    assert len(scenario.evaluator_calls) == 6
    child_full_val = [
        call
        for call in scenario.evaluator_calls
        if call[0].startswith("g1-s") and call[1] == "val"
    ]
    assert len(child_full_val) == 1

    cache = load_eval_cache(scenario.project_root)
    assert cache is not None
    assert len(cache) == 8
    persisted = load_state(scenario.project_root)
    assert persisted is not None
    assert persisted.budget.evaluations == 12


def test_same_seed_is_deterministic_across_opposite_completion_schedules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    forward = ["g1-s1", "g1-s2", "g1-s3", "g1-s4"]
    reverse = list(reversed(forward))
    first = _run_scenario(
        tmp_path / "forward",
        monkeypatch,
        p=2,
        n=2,
        seed=2026,
        scores=[0.4, 0.5, 0.6, 0.7],
        completion_order=forward,
        max_workers=4,
    )
    second = _run_scenario(
        tmp_path / "reverse",
        monkeypatch,
        p=2,
        n=2,
        seed=2026,
        scores=[0.4, 0.5, 0.6, 0.7],
        completion_order=reverse,
        max_workers=4,
    )

    assert first.completion_order != second.completion_order
    assert _normalized(first) == _normalized(second)


@pytest.mark.parametrize("p", [1, 2, 4])
def test_omitted_n_matches_explicit_n1_for_existing_k_by_1_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, p: int
) -> None:
    omitted = _run_scenario(tmp_path / f"p{p}-omitted", monkeypatch, p=p, n=None)
    explicit = _run_scenario(tmp_path / f"p{p}-explicit", monkeypatch, p=p, n=1)

    assert _normalized(omitted) == _normalized(explicit)


@pytest.mark.parametrize(
    ("boundary", "expected_cleaned", "expected_missing", "expected_accounted"),
    [
        pytest.param(
            "before_worker_dispatch",
            set(),
            {"g1-s1", "g1-s2"},
            {"g1-s1", "g1-s2"},
            id="before-dispatch",
        ),
        pytest.param(
            "mid_worker",
            {"g1-s1"},
            {"g1-s2"},
            {"g1-s1", "g1-s2"},
            id="mid-worker",
        ),
        pytest.param(
            "after_evaluation",
            {"g1-s1", "g1-s2"},
            set(),
            {"g1-s1", "g1-s2"},
            id="after-evaluation",
        ),
        pytest.param(
            "mid_apply",
            {"g1-s2"},
            set(),
            {"g1-s1", "g1-s2"},
            id="mid-apply",
        ),
    ],
)
def test_resume_reconciles_each_batch_phase_without_double_charge_apply_or_orphans(
    tmp_path: Path,
    boundary: str,
    expected_cleaned: set[str],
    expected_missing: set[str],
    expected_accounted: set[str],
) -> None:
    """Exercise the durable batch ledger at all four crash boundaries.

    This intentionally targets the Step-7 persistence API directly: the
    runtime owns when checkpoints are called, while the state layer owns the
    idempotent facts a resumed process consumes.
    """
    import helix.state as state_api

    task_record_type = getattr(state_api, "ProposalTaskRecord")
    batch_record_type = getattr(state_api, "ProposalBatchRecord")
    checkpoint_before = getattr(state_api, "checkpoint_batch_before_dispatch")
    checkpoint_task = getattr(state_api, "checkpoint_batch_task")
    reconcile = getattr(state_api, "reconcile_interrupted_batches")
    reserved_ids = getattr(state_api, "reserved_candidate_ids")

    project_root = tmp_path / "project"
    project_root.mkdir()
    worktrees_dir = project_root / ".helix" / "worktrees"
    worktrees_dir.mkdir(parents=True)

    state = EvolutionState(
        generation=1,
        frontier=["g0-s0"],
        instance_scores={"g0-s0": {"0": 0.1}},
        budget=BudgetState(),
        config_hash="matrix",
        frontier_type="instance",
    )
    tasks = [
        task_record_type(
            batch_id="g1-b1",
            p=1,
            n=2,
            task_index=index,
            parent_group=0,
            mutation_index=index,
            parent_id="g0-s0",
            child_id=f"g1-s{index + 1}",
        )
        for index in range(2)
    ]
    batch = batch_record_type(
        batch_id="g1-b1",
        generation=1,
        p=1,
        n=2,
        tasks=tasks,
    )
    checkpoint_before(
        state,
        project_root,
        batch,
        max_evaluations=100,
        max_in_flight_evaluations=8,
    )

    if boundary == "mid_worker":
        (worktrees_dir / "g1-s1").mkdir()
        checkpoint_task(
            state,
            project_root,
            batch_id="g1-b1",
            task_index=0,
            status="running",
        )
    elif boundary in {"after_evaluation", "mid_apply"}:
        state.budget.evaluations = 8
        for index in range(2):
            (worktrees_dir / f"g1-s{index + 1}").mkdir()
            checkpoint_task(
                state,
                project_root,
                batch_id="g1-b1",
                task_index=index,
                status="evaluated",
                score_delta=0.5 + index,
                budget_charge=BudgetState(evaluations=4),
                budget_accounted=True,
            )
        if boundary == "mid_apply":
            checkpoint_task(
                state,
                project_root,
                batch_id="g1-b1",
                task_index=0,
                status="applied",
                selection="selected",
                cleanup="not_required",
                applied=True,
            )
            state.frontier.append("g1-s1")

    cleaned_by_callback: list[str] = []

    def cleanup_worktree(candidate_id: str, path: Path) -> bool:
        cleaned_by_callback.append(candidate_id)
        shutil.rmtree(path)
        return True

    budget_before_resume = state.budget.evaluations
    summaries = reconcile(
        state,
        project_root,
        worktrees_dir=worktrees_dir,
        cleanup_worktree=cleanup_worktree,
    )

    assert len(summaries) == 1
    summary = summaries[0]
    assert set(summary.reserved_child_ids) == {"g1-s1", "g1-s2"}
    assert set(summary.cleaned_child_ids) == expected_cleaned
    assert set(summary.missing_child_ids) == expected_missing
    assert set(summary.accounted_child_ids) == expected_accounted
    assert set(cleaned_by_callback) == expected_cleaned
    assert state.budget.evaluations == budget_before_resume
    assert reserved_ids(state) >= {"g0-s0", "g1-s1", "g1-s2"}
    assert state.frontier.count("g1-s1") <= 1

    if boundary == "mid_apply":
        assert set(summary.applied_child_ids) == {"g1-s1"}
        assert (worktrees_dir / "g1-s1").exists()
        assert not (worktrees_dir / "g1-s2").exists()
    else:
        assert set(summary.applied_child_ids) == set()
        assert all(
            not (worktrees_dir / candidate_id).exists()
            for candidate_id in ("g1-s1", "g1-s2")
        )

    loaded = load_state(project_root)
    assert loaded is not None
    persisted_batch = getattr(loaded, "proposal_batches")[-1]
    assert persisted_batch.phase == "interrupted"
    assert [task.child_id for task in persisted_batch.tasks] == ["g1-s1", "g1-s2"]

    # A second resume is a no-op: no double cleanup, charge, or frontier add.
    assert (
        reconcile(
            loaded,
            project_root,
            worktrees_dir=worktrees_dir,
            cleanup_worktree=cleanup_worktree,
        )
        == []
    )
    assert set(cleaned_by_callback) == expected_cleaned
    assert loaded.budget.evaluations == budget_before_resume
    assert loaded.frontier.count("g1-s1") <= 1
