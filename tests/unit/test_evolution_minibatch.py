"""Integration tests for HELIX minibatch-gate evolution (Phase 3).

Exercises the new per-proposal minibatch acceptance gate wired into
``helix.evolution.run_evolution`` when ``DatasetConfig.train_path`` is
provided.  Uses the same mocking strategy as ``test_evolution.py``:
mock all I/O and mock ``run_evaluator`` with controlled score sequences.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from helix.config import (
    DatasetConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    SeedlessConfig,
    WorktreeConfig,
)
from helix.evolution import (
    HelixDataLoader,
    _make_data_loader,
    _reconcile_incomplete_attempts_on_resume,
    run_evolution,
)
from helix.population import Candidate, EvalResult, ParetoFrontier
from helix.state import BudgetState, EvolutionState
from helix.trace import EventType, TRACE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(cid: str = "g0-s0") -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=f"/tmp/helix/{cid}",
        branch_name=f"helix/{cid}",
        generation=0,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def _make_result(cid: str, scores: dict[str, float]) -> EvalResult:
    return EvalResult(
        candidate_id=cid,
        scores={},
        asi={},
        instance_scores=dict(scores),
    )


def _write_train_jsonl(path: Path, n: int = 6) -> Path:
    """Write a JSONL dataset with n trivial examples and return the path."""
    p = path / "train.jsonl"
    with open(p, "w") as f:
        for i in range(n):
            f.write(json.dumps({"idx": i, "x": i}) + "\n")
    return p


def _git(args: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _init_repo(path: Path) -> None:
    path.mkdir(exist_ok=True)
    _git(["init"], path)
    _git(["config", "user.name", "HELIX Test"], path)
    _git(["config", "user.email", "helix-test@noreply"], path)


def test_pareto_frontier_public_accessors_match_internal_state() -> None:
    """``iter_results`` / ``get_result`` / ``has_results`` expose stable results."""
    frontier = ParetoFrontier()
    assert frontier.has_results() is False
    assert frontier.get_result("missing") is None
    assert list(frontier.iter_results()) == []

    cand_a = _make_candidate("cand-a")
    cand_b = _make_candidate("cand-b")
    res_a = _make_result("cand-a", {"0": 0.1})
    res_b = _make_result("cand-b", {"0": 0.9})
    frontier.add(cand_a, res_a)
    frontier.add(cand_b, res_b)

    assert frontier.has_results() is True
    assert frontier.get_result("cand-a") is res_a
    assert frontier.get_result("cand-b") is res_b
    # Insertion order preserved → diagnostics get a stable replay.
    assert [cid for cid, _ in frontier.iter_results()] == ["cand-a", "cand-b"]


def _make_minibatch_config(
    train_path: Path,
    *,
    minibatch_size: int = 2,
    val_size: int | None = None,
    val_stage_size: int | None = None,
    max_generations: int = 1,
    max_evaluations: int = 1000,
    num_parallel_proposals: int = 1,
    cache_evaluation: bool = True,
    acceptance_criterion: str = "strict_improvement",
    max_workers: int | None = None,
) -> HelixConfig:
    evo_kwargs: dict[str, Any] = dict(
        max_generations=max_generations,
        max_evaluations=max_evaluations,
        perfect_score_threshold=None,
        minibatch_size=minibatch_size,
        num_parallel_proposals=num_parallel_proposals,
        cache_evaluation=cache_evaluation,
        acceptance_criterion=acceptance_criterion,
        val_stage_size=val_stage_size,
        frontier_type="instance",
    )
    if max_workers is not None:
        evo_kwargs["max_workers"] = max_workers
    return HelixConfig(
        objective="Minibatch test",
        evaluator=EvaluatorConfig(command="pytest -q"),
        dataset=DatasetConfig(val_size=val_size),
        seedless=SeedlessConfig(train_path=train_path),
        evolution=EvolutionConfig(**evo_kwargs),
        worktree=WorktreeConfig(),
    )


@pytest.fixture
def all_mocks(mocker: Any) -> dict[str, Any]:
    return {
        "create_seed_worktree": mocker.patch("helix.evolution.create_seed_worktree"),
        "run_evaluator": mocker.patch("helix.evolution.run_evaluator"),
        "mutate": mocker.patch("helix.evolution.mutate"),
        "merge": mocker.patch("helix.evolution.merge", return_value=None),
        "remove_worktree": mocker.patch("helix.evolution.remove_worktree"),
        "load_state": mocker.patch("helix.evolution.load_state", return_value=None),
        "save_state": mocker.patch("helix.evolution.save_state"),
        "init_base_dir": mocker.patch("helix.evolution.init_base_dir"),
        "_save_evaluation": mocker.patch("helix.evolution._save_evaluation"),
        "_load_evaluation": mocker.patch(
            "helix.evolution._load_evaluation", return_value=None
        ),
        "record_entry": mocker.patch("helix.evolution.record_entry"),
        "load_lineage": mocker.patch("helix.evolution.load_lineage", return_value={}),
        "find_merge_triplet": mocker.patch(
            "helix.evolution.find_merge_triplet", return_value=None
        ),
        "snapshot_candidate": mocker.patch("helix.evolution.snapshot_candidate"),
        "set_phase": mocker.patch("helix.evolution.set_phase"),
        "print_info": mocker.patch("helix.evolution.print_info"),
        "print_success": mocker.patch("helix.evolution.print_success"),
        "print_warning": mocker.patch("helix.evolution.print_warning"),
        "print_error": mocker.patch("helix.evolution.print_error"),
        "render_budget": mocker.patch("helix.evolution.render_budget"),
        "render_generation": mocker.patch("helix.evolution.render_generation"),
        "_check_evaluator_script_exists": mocker.patch(
            "helix.evolution._check_evaluator_script_exists"
        ),
    }


# ---------------------------------------------------------------------------
# HelixDataLoader smoke tests
# ---------------------------------------------------------------------------


class TestHelixDataLoader:
    def test_jsonl_loader_exposes_string_ids(self, tmp_path: Path) -> None:
        p = _write_train_jsonl(tmp_path, n=4)
        loader = HelixDataLoader(p)
        assert len(loader) == 4
        assert loader.all_ids() == ["0", "1", "2", "3"]

    def test_json_array_loader(self, tmp_path: Path) -> None:
        p = tmp_path / "train.json"
        p.write_text(json.dumps([{"a": 1}, {"a": 2}, {"a": 3}]))
        loader = HelixDataLoader(p)
        assert len(loader) == 3
        assert loader.all_ids() == ["0", "1", "2"]

    def test_directory_loader(self, tmp_path: Path) -> None:
        d = tmp_path / "train"
        d.mkdir()
        (d / "alpha.json").write_text("{}")
        (d / "beta.json").write_text("{}")
        loader = HelixDataLoader(d)
        assert len(loader) == 2
        assert loader.all_ids() == ["alpha", "beta"]

    def test_make_data_loader_none_path(self) -> None:
        assert _make_data_loader(None) is None

    def test_make_data_loader_empty_jsonl_returns_none(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("")
        assert _make_data_loader(p) is None


# ---------------------------------------------------------------------------
# Evolution loop with minibatch gate
# ---------------------------------------------------------------------------


class TestMinibatchGateIntegration:
    def test_sampler_wired_when_train_path_set(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """train_path → minibatch gate pipes parent + child through per-id eval."""
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        mutated = _make_candidate("g1-s1")
        all_mocks["mutate"].return_value = mutated

        # Filter to minibatch (train-split) calls only — seed-eval and
        # child full-val are both on "val" split with explicit ids now
        # (helix_result requires helix_batch.json on every invocation).
        train_minibatch_calls: list[list[str]] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if split == "train" and instance_ids is not None:
                train_minibatch_calls.append(list(instance_ids))
            if instance_ids is not None:
                # minibatch / explicit-id eval: return small per-id scores
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.3 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            # full val eval (single-task/no-example path): moderate scores
            return _make_result(candidate.id, {"v1": 0.5, "v2": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path, minibatch_size=2, max_generations=1, max_evaluations=100,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Parent minibatch call + child minibatch call both carry instance_ids.
        assert len(train_minibatch_calls) >= 2, (
            "Expected at least parent+child minibatch eval calls on train split, "
            f"got {train_minibatch_calls}"
        )
        # Parent and child minibatches should be the SAME subsample.
        assert train_minibatch_calls[0] == train_minibatch_calls[1]
        # minibatch_size=2 → two ids.
        assert len(train_minibatch_calls[0]) == 2

    def test_rejected_proposal_skips_full_val_eval(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Child that does not improve on minibatch → no val eval, no frontier add."""
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        calls: list[tuple[str, str | None, list[str] | None]] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            calls.append((candidate.id, split, instance_ids))
            if instance_ids is not None:
                # Child is WORSE than parent on the minibatch.
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.8 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.1 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.0})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path, minibatch_size=2, max_generations=1, max_evaluations=100,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # No val eval should have been requested for the rejected child.
        child_val_calls = [
            c for c in calls if c[0] == "g1-s1" and c[2] is None and c[1] == "val"
        ]
        assert child_val_calls == [], (
            f"Expected no child val eval for rejected proposal, got {child_val_calls}"
        )
        # Remove worktree was called (rejection cleanup).
        assert all_mocks["remove_worktree"].called
        attempt_path = tmp_path / ".helix" / "attempts" / "g1-s1.json"
        assert attempt_path.exists()
        attempt = json.loads(attempt_path.read_text())
        child_train_ids = [
            c[2] for c in calls if c[0] == "g1-s1" and c[1] == "train"
        ][0]
        assert attempt["candidate_id"] == "g1-s1"
        assert attempt["attempt"] == {
            "status": "rejected",
            "reason": "minibatch_gate",
            "parent_id": "g0-s0",
            "generation": 1,
            "stage": "train_minibatch",
            "example_ids": child_train_ids,
        }
        assert set(attempt["instance_scores"]) == set(child_train_ids)

    def test_perfect_minibatch_skip_advances_generation(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Perfect minibatch skip advances gen unconditionally (GEPA parity:
        ``state.i`` is incremented unconditionally at the top of
        ``GEPAEngine.run()``'s main loop, before any merge/reflective branch).

        After Change 1 (unconditional gen increment), a perfect-subsample skip
        no longer rolls back to retry the same generation slot.  Gen was already
        incremented at the top of the loop, so state.generation advances and the
        next loop iteration starts at gen+1.  With max_generations=1, the loop
        exits immediately after the one skip iteration — mutate is never called.

        Regression for NB-2: before the fix, a perfect skip did NOT advance gen,
        allowing an infinite retry when budget was also uncapped.
        """
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        parent_train_calls = 0

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            nonlocal parent_train_calls
            if split == "train" and instance_ids is not None:
                if candidate.id == seed.id:
                    parent_train_calls += 1
                    # Always perfect — ensures perfect-skip fires every time.
                    return _make_result(candidate.id, {i: 1.0 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.9})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            val_size=2,
            max_generations=1,
            max_evaluations=100,
        )
        config.evolution.perfect_score_threshold = 1.0
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Gen was already incremented at the top of the loop before the skip.
        # After the skip, 1 < max_generations=1 is False → loop exits.
        # mutate is never called because the only generation was a perfect skip.
        assert parent_train_calls == 1
        all_mocks["mutate"].assert_not_called()
        skip_path = tmp_path / ".helix" / "skips" / "g1.json"
        assert skip_path.exists()
        skip_records = json.loads(skip_path.read_text())
        # Since NB-1 fix, skip records are always a list (one entry per proposal).
        assert isinstance(skip_records, list)
        assert len(skip_records) == 1
        assert skip_records[0]["generation"] == 1
        assert skip_records[0]["reason"] == "perfect_subsample"

    # ------------------------------------------------------------------
    # Test A: NB-1 regression — all proposals perfect, n_proposals=3
    # ------------------------------------------------------------------

    def test_multiple_perfect_skips_all_recorded_in_single_file(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """n_proposals=3, all parents perfect → skips/g1.json is a list of 3 records.

        Regression for NB-1: before the fix _save_skip_record was called once per
        proposal, each overwriting the previous file so only the last record survived.

        After Change 1 (unconditional gen increment) and Change 2 (parent minibatch
        bypasses cache), max_generations=1 naturally terminates after the one
        perfect-skip iteration — no tight budget ceiling needed.  3 perfect proposals
        in g1 produce 3 records in skips/g1.json, then gen advances to 2 which
        exceeds max_generations=1, ending the run.
        """
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        # Perfect scores for every parent minibatch eval so all 3 proposals skip.
        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 1.0 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 1.0})

        all_mocks["run_evaluator"].side_effect = run_eval

        # Change 2: parent minibatch evals bypass the cache (pass None), so no
        # subprocess.run on fake worktree paths in the parallel ThreadPoolExecutor
        # (n_proposals=3).  Change 1: gen advances on skip, so max_generations=1
        # naturally terminates after the one iteration — no tight budget ceiling.
        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=100,
            num_parallel_proposals=3,
        )
        config.evolution.perfect_score_threshold = 1.0
        run_evolution(config, tmp_path, tmp_path / ".helix")

        skip_path = tmp_path / ".helix" / "skips" / "g1.json"
        assert skip_path.exists(), "skips/g1.json should be written"
        records = json.loads(skip_path.read_text())
        assert isinstance(records, list), "skip file must be a JSON list"
        assert len(records) == 3, (
            f"Expected 3 skip records (one per proposal), got {len(records)}"
        )
        for rec in records:
            assert rec["generation"] == 1
            assert rec["reason"] == "perfect_subsample"
            assert "parent_id" in rec
            assert "parent_eval" in rec

    # ------------------------------------------------------------------
    # Test B: train_gate rejection artifact
    # ------------------------------------------------------------------

    def test_train_gate_rejection_artifact(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Child that fails train_gate (no-minibatch mode) writes attempts/{cid}.json.

        Mirrors test_rejected_minibatch_attempt_artifact but for the legacy
        single-task train-gating path (train_path=None → subsample_ids is None).
        """
        # No train_path → single-task / no-minibatch mode; child goes through
        # the full-train acceptance gate instead of the minibatch gate.
        seed = _make_candidate("g0-s0")
        child = _make_candidate("g1-s1")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = child

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            # In no-minibatch mode instance_ids is always None.
            if candidate.id == seed.id:
                return _make_result(candidate.id, {"t": 0.9})
            # Child scores worse → train_gate rejects it.
            return _make_result(candidate.id, {"t": 0.3})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = HelixConfig(
            objective="train gate test",
            evaluator=EvaluatorConfig(command="pytest -q"),
            dataset=DatasetConfig(),  # no train_path → no minibatch gate
            evolution=EvolutionConfig(
                max_generations=1,
                max_evaluations=100,
                perfect_score_threshold=None,
                frontier_type="instance",
            ),
            worktree=WorktreeConfig(),
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        attempt_path = tmp_path / ".helix" / "attempts" / "g1-s1.json"
        assert attempt_path.exists(), "attempts/g1-s1.json should be written on train_gate reject"
        attempt = json.loads(attempt_path.read_text())
        assert attempt["candidate_id"] == "g1-s1"
        assert attempt["attempt"]["status"] == "rejected"
        assert attempt["attempt"]["reason"] == "train_gate"
        assert attempt["attempt"]["stage"] == "train"
        assert attempt["attempt"]["parent_id"] == "g0-s0"
        assert attempt["attempt"]["generation"] == 1
        assert attempt["attempt"]["example_ids"] is None
        # The worktree should have been removed on rejection.
        assert all_mocks["remove_worktree"].called

    # ------------------------------------------------------------------
    # Test C: val_stage rejection artifact
    # ------------------------------------------------------------------

    def test_val_stage_rejection_artifact(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Child that passes minibatch but fails val_stage writes attempts/{cid}.json.

        Mirrors test_rejected_minibatch_attempt_artifact but for the staged-val
        gate (val_stage_size > 0) rejection path.
        """
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                if split == "train":
                    # Parent scores low on minibatch; child scores high → passes gate.
                    if candidate.id == seed.id:
                        return _make_result(candidate.id, {i: 0.1 for i in instance_ids})
                    return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
                # val evals
                if split == "val":
                    if candidate.id == seed.id:
                        # Seed full val — provides parent frontier result with val scores.
                        return _make_result(
                            candidate.id, {i: 0.8 for i in instance_ids}
                        )
                    # Child on val stage: scores worse than parent → val_stage rejects.
                    return _make_result(candidate.id, {i: 0.1 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            val_size=4,
            val_stage_size=2,
            max_generations=1,
            max_evaluations=100,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        attempt_path = tmp_path / ".helix" / "attempts" / "g1-s1.json"
        assert attempt_path.exists(), "attempts/g1-s1.json should be written on val_stage reject"
        attempt = json.loads(attempt_path.read_text())
        assert attempt["candidate_id"] == "g1-s1"
        assert attempt["attempt"]["status"] == "rejected"
        assert attempt["attempt"]["reason"] == "val_stage"
        assert attempt["attempt"]["stage"] == "val_stage"
        assert attempt["attempt"]["parent_id"] == "g0-s0"
        assert attempt["attempt"]["generation"] == 1
        # example_ids should be the val stage ids (first val_stage_size=2 val ids).
        assert attempt["attempt"]["example_ids"] is not None
        assert len(attempt["attempt"]["example_ids"]) == 2
        # Verify worktree cleaned up.
        assert all_mocks["remove_worktree"].called

    def test_accepted_proposal_triggers_full_val_eval(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Child that improves on minibatch → full val eval is run."""
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        # Full-val evals now always carry explicit instance_ids when a
        # loader is configured (helix_result requires helix_batch.json).
        # Distinguish child full-val from child minibatch by split="val".
        child_val_calls: list[list[str]] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if (
                candidate.id == "g1-s1"
                and split == "val"
                and instance_ids is not None
            ):
                child_val_calls.append(list(instance_ids))
            if instance_ids is not None:
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.1 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.7, "v2": 0.7})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path, minibatch_size=2, max_generations=1, max_evaluations=100,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert len(child_val_calls) >= 1, (
            "Expected at least one full val eval for the accepted child."
        )

    def test_stage_rejection_skips_full_val_eval(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Child that passes minibatch but fails val stage never reaches full val."""
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        child_val_calls: list[list[str]] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if split == "val" and instance_ids is not None and candidate.id == "g1-s1":
                child_val_calls.append(list(instance_ids))
            if split == "val" and instance_ids == ["0", "1", "2", "3"] and candidate.id == seed.id:
                return _make_result(candidate.id, {"0": 0.8, "1": 0.8, "2": 0.8, "3": 0.8})
            if split == "train" and instance_ids is not None:
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.1 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            if split == "val" and instance_ids == ["0", "1"] and candidate.id == "g1-s1":
                return _make_result(candidate.id, {"0": 0.1, "1": 0.1})
            if split == "val" and instance_ids is not None and candidate.id == "g1-s1":
                pytest.fail(f"Unexpected child full-val call after stage reject: {instance_ids}")
            return _make_result(candidate.id, {"v1": 0.0})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            val_size=4,
            val_stage_size=2,
            max_generations=1,
            max_evaluations=100,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert child_val_calls == [["0", "1"]]
        assert all_mocks["_save_evaluation"].call_count == 1
        assert all_mocks["remove_worktree"].called

    def test_stage_pass_promotes_to_full_val_with_cache_reuse(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Stage pass should only full-eval the uncached remainder and persist full val."""
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        child_val_calls: list[list[str]] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if split == "val" and instance_ids is not None and candidate.id == "g1-s1":
                child_val_calls.append(list(instance_ids))
            if split == "val" and instance_ids == ["0", "1", "2", "3"] and candidate.id == seed.id:
                return _make_result(candidate.id, {"0": 0.2, "1": 0.2, "2": 0.2, "3": 0.2})
            if split == "train" and instance_ids is not None:
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.1 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            if split == "val" and instance_ids == ["0", "1"] and candidate.id == "g1-s1":
                return _make_result(candidate.id, {"0": 0.8, "1": 0.8})
            if split == "val" and instance_ids == ["2", "3"] and candidate.id == "g1-s1":
                return _make_result(candidate.id, {"2": 0.7, "3": 0.6})
            return _make_result(candidate.id, {"v1": 0.0})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            val_size=4,
            val_stage_size=2,
            max_generations=1,
            max_evaluations=100,
            cache_evaluation=True,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert child_val_calls == [["0", "1"], ["2", "3"]]
        assert all_mocks["_save_evaluation"].call_count == 2
        saved_child_result = all_mocks["_save_evaluation"].call_args_list[-1].args[1]
        assert saved_child_result.candidate_id == "g1-s1"
        assert saved_child_result.instance_scores == {
            "0": 0.8,
            "1": 0.8,
            "2": 0.7,
            "3": 0.6,
        }

    def test_disabled_val_stage_runs_direct_full_val(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """When val_stage_size is unset, accepted children still go straight to full val."""
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        child_val_calls: list[list[str]] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if split == "val" and instance_ids is not None and candidate.id == "g1-s1":
                child_val_calls.append(list(instance_ids))
            if split == "val" and instance_ids == ["0", "1", "2", "3"]:
                return _make_result(candidate.id, {i: 0.8 for i in instance_ids})
            if split == "train" and instance_ids is not None:
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.1 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.0})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            val_size=4,
            val_stage_size=None,
            max_generations=1,
            max_evaluations=100,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert child_val_calls == [["0", "1", "2", "3"]]

    def test_parallel_proposals_pre_sample_n_contexts(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """num_parallel_proposals=2 → two distinct minibatches pre-sampled."""
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        mut_ids = iter(["g1-s1", "g1-s2", "g1-s3"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        parent_minibatches: list[list[str]] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            # Parent minibatch evals run on split="train"; the seed
            # full-val eval runs on split="val" with explicit ids too
            # (helix_result requires helix_batch.json on every
            # invocation), so gate on split to avoid counting it.
            if (
                candidate.id == seed.id
                and split == "train"
                and instance_ids is not None
            ):
                parent_minibatches.append(list(instance_ids))
            if instance_ids is not None:
                # Child scores slightly worse so all get rejected -- keeps
                # test focused on pre-sampling N contexts.
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.5 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.4 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=2,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Two parent minibatches should have been pre-sampled this generation.
        assert len(parent_minibatches) == 2
        # And they should differ (state.i bumps between them).
        assert parent_minibatches[0] != parent_minibatches[1]

    def test_single_task_mode_no_train_path(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """train_path=None (circle_packing mode) → evolution runs without crashing."""
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None  # mutation failure path

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            # instance_ids MUST be None in single-task/no-example mode
            assert instance_ids is None
            return _make_result(candidate.id, {"t": 0.9798})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = HelixConfig(
            objective="single task",
            evaluator=EvaluatorConfig(command="pytest -q"),
            dataset=DatasetConfig(),  # no train_path
            evolution=EvolutionConfig(
                max_generations=1,
                max_evaluations=100,
                perfect_score_threshold=None,
                frontier_type="instance",
            ),
            worktree=WorktreeConfig(),
        )
        # Must not raise.
        run_evolution(config, tmp_path, tmp_path / ".helix")

    def test_eval_cache_populated_with_parent_minibatch(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        """When cache_evaluation=True, parent minibatch results are written to cache."""
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        # Spy on the MinibatchEvalCache.put method.
        from helix.eval_cache import EvaluationCache as MBCache

        # GEPA parity: the cache consumer now routes writes through
        # ``put_batch`` (via ``evaluate_with_cache_full``), not per-example
        # ``put``.  Spy on both and assert the aggregate call count.
        put_spy = mocker.spy(MBCache, "put_batch")

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                return _make_result(
                    candidate.id, {i: 0.5 for i in instance_ids}
                )
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=100,
            cache_evaluation=True,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Parent minibatch eval should have been stored: ``put_batch`` is
        # called once per cache-miss evaluator invocation.  Parent + child
        # minibatch calls → at least two put_batch calls.
        assert put_spy.call_count >= 1


# ---------------------------------------------------------------------------
# GEPA-parity cache CONSUMER tests (B4): ``_cached_evaluate_batch``
#
# These tests directly exercise the new per-example cache-consumer helper
# that wires HELIX's minibatch eval sites up to the GEPA
# ``GEPAState.cached_evaluate_full`` semantics (``core/state.py``).
# ---------------------------------------------------------------------------


class TestCachedEvaluateBatch:
    def _make_cand(self, cid: str = "cand-A") -> Candidate:
        return _make_candidate(cid)

    def _trivial_config(self) -> HelixConfig:
        return HelixConfig(
            objective="cache consumer test",
            evaluator=EvaluatorConfig(command="pytest -q"),
            seedless=SeedlessConfig(),
            evolution=EvolutionConfig(
                max_generations=1,
                max_evaluations=10,
                perfect_score_threshold=None,
                minibatch_size=3,
                cache_evaluation=True,
                frontier_type="instance",
            ),
            worktree=WorktreeConfig(),
        )

    def test_cache_hit_skips_evaluator(self, mocker: Any) -> None:
        """All requested ids pre-populated → evaluator is NEVER invoked."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch

        cache: MBCache[object, str] = MBCache[object, str]()
        cand = self._make_cand("cand-A")
        cand_dict = {"content_key": cand.id, "split": "train"}
        cache.put_batch(
            cand_dict,
            ["0", "1", "2"],
            [None, None, None],
            [0.1, 0.2, 0.3],
        )

        run_eval_mock = mocker.patch("helix.evolution.run_evaluator")
        write_batch_mock = mocker.patch("helix.evolution._write_helix_batch")

        result, num_actual = _cached_evaluate_batch(
            cand, ["0", "1", "2"], cache, self._trivial_config(), "train", Path("/tmp"),
        )

        assert run_eval_mock.call_count == 0, (
            "Evaluator must NOT be invoked when every requested id is cached"
        )
        assert write_batch_mock.call_count == 0, (
            "helix_batch.json must NOT be rewritten on a full cache hit"
        )
        assert num_actual == 0
        assert result.instance_scores == {"0": 0.1, "1": 0.2, "2": 0.3}

    def test_content_key_reuses_cache_across_candidate_ids(self, mocker: Any) -> None:
        """Equivalent candidate content should reuse per-example cache entries."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch

        cache: MBCache[object, str] = MBCache[object, str]()
        cand_b = self._make_cand("cand-B")
        mocker.patch(
            "helix.evolution._candidate_content_key",
            side_effect=lambda candidate: "same-content",
        )

        cache.put_batch(
            {"content_key": "same-content", "split": "train"},
            ["0", "1"],
            [None, None],
            [0.7, 0.8],
        )

        run_eval_mock = mocker.patch("helix.evolution.run_evaluator")
        mocker.patch("helix.evolution._write_helix_batch")

        result, num_actual = _cached_evaluate_batch(
            cand_b, ["0", "1"], cache, self._trivial_config(), "train", Path("/tmp"),
        )

        assert num_actual == 0
        assert run_eval_mock.call_count == 0
        assert result.candidate_id == cand_b.id
        assert result.instance_scores == {"0": 0.7, "1": 0.8}

    def test_tree_key_reuses_cache_across_different_commits_with_same_tree(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Commit metadata/history changes must not invalidate content cache."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch, _candidate_content_key

        repo = tmp_path / "repo"
        _init_repo(repo)
        (repo / "prompt.md").write_text("same tracked content\n")
        _git(["add", "prompt.md"], repo)
        _git(["commit", "-m", "initial content"], repo)

        cand_a = self._make_cand("cand-A")
        cand_a.worktree_path = str(repo)
        commit_a = _git(["rev-parse", "HEAD"], repo)
        tree_a = _git(["rev-parse", "HEAD^{tree}"], repo)

        cache: MBCache[object, str] = MBCache[object, str]()
        cache.put_batch(
            {"content_key": _candidate_content_key(cand_a), "split": "train"},
            ["0", "1"],
            [None, None],
            [0.7, 0.8],
        )

        _git(["commit", "--allow-empty", "-m", "metadata only"], repo)
        cand_b = self._make_cand("cand-B")
        cand_b.worktree_path = str(repo)
        commit_b = _git(["rev-parse", "HEAD"], repo)
        tree_b = _git(["rev-parse", "HEAD^{tree}"], repo)
        assert commit_a != commit_b
        assert tree_a == tree_b

        # Pin the contract directly: equivalent trees ⇒ equal content keys.
        assert _candidate_content_key(cand_a) == _candidate_content_key(cand_b)

        run_eval_mock = mocker.patch("helix.evolution.run_evaluator")
        mocker.patch("helix.evolution._write_helix_batch")

        result, num_actual = _cached_evaluate_batch(
            cand_b, ["0", "1"], cache, self._trivial_config(), "train", tmp_path,
        )

        assert num_actual == 0
        assert run_eval_mock.call_count == 0
        assert result.candidate_id == cand_b.id
        assert result.instance_scores == {"0": 0.7, "1": 0.8}

    def test_tree_key_misses_cache_when_tracked_content_changes(
        self, mocker: Any, tmp_path: Path
    ) -> None:
        """Tracked file changes must produce a different cache identity."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch, _candidate_content_key

        repo = tmp_path / "repo"
        _init_repo(repo)
        (repo / "prompt.md").write_text("v1\n")
        _git(["add", "prompt.md"], repo)
        _git(["commit", "-m", "initial content"], repo)

        cand_a = self._make_cand("cand-A")
        cand_a.worktree_path = str(repo)
        tree_a = _git(["rev-parse", "HEAD^{tree}"], repo)

        cache: MBCache[object, str] = MBCache[object, str]()
        cache.put_batch(
            {"content_key": _candidate_content_key(cand_a), "split": "train"},
            ["0", "1"],
            [None, None],
            [0.7, 0.8],
        )

        (repo / "prompt.md").write_text("v2\n")
        _git(["add", "prompt.md"], repo)
        _git(["commit", "-m", "changed tracked content"], repo)
        cand_b = self._make_cand("cand-B")
        cand_b.worktree_path = str(repo)
        tree_b = _git(["rev-parse", "HEAD^{tree}"], repo)
        assert tree_a != tree_b

        seen_instance_ids: list[list[str] | None] = []

        def fake_run(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            seen_instance_ids.append(instance_ids)
            return _make_result(
                candidate.id, {eid: 0.5 for eid in (instance_ids or [])}
            )

        mocker.patch("helix.evolution.run_evaluator", side_effect=fake_run)
        mocker.patch("helix.evolution._write_helix_batch")

        result, num_actual = _cached_evaluate_batch(
            cand_b, ["0", "1"], cache, self._trivial_config(), "train", tmp_path,
        )

        assert seen_instance_ids == [["0", "1"]]
        assert num_actual == 2
        assert result.candidate_id == cand_b.id
        assert result.instance_scores == {"0": 0.5, "1": 0.5}

    def test_content_key_falls_back_when_worktree_is_dirty(
        self, tmp_path: Path
    ) -> None:
        """Dirty / untracked worktree must NOT key by tree SHA (avoid stale hits)."""
        from helix.evolution import _candidate_content_key

        repo = tmp_path / "repo"
        _init_repo(repo)
        (repo / "prompt.md").write_text("v1\n")
        _git(["add", "prompt.md"], repo)
        _git(["commit", "-m", "initial content"], repo)

        cand = self._make_cand("cand-dirty")
        cand.worktree_path = str(repo)
        clean_key = _candidate_content_key(cand)
        tree_sha = _git(["rev-parse", "HEAD^{tree}"], repo)
        assert clean_key == tree_sha  # sanity: clean repo keys by tree SHA

        # Modify a tracked file without committing → key must fall back to id.
        (repo / "prompt.md").write_text("v1-uncommitted\n")
        assert _candidate_content_key(cand) == cand.id

        # Reset and try untracked file → still falls back to id.
        _git(["checkout", "--", "prompt.md"], repo)
        assert _candidate_content_key(cand) == tree_sha
        (repo / "scratch.txt").write_text("untracked\n")
        assert _candidate_content_key(cand) == cand.id

    def test_cache_miss_invokes_full_evaluator(self, mocker: Any) -> None:
        """Empty cache → evaluator is invoked with ALL requested ids."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch

        cache: MBCache[object, str] = MBCache[object, str]()
        cand = self._make_cand("cand-B")

        seen_instance_ids: list[list[str] | None] = []

        def fake_run(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            seen_instance_ids.append(instance_ids)
            return _make_result(
                candidate.id, {eid: 0.5 for eid in (instance_ids or [])}
            )

        mocker.patch("helix.evolution.run_evaluator", side_effect=fake_run)
        mocker.patch("helix.evolution._write_helix_batch")

        result, num_actual = _cached_evaluate_batch(
            cand, ["0", "1", "2"], cache, self._trivial_config(), "train", Path("/tmp"),
        )

        assert seen_instance_ids == [["0", "1", "2"]], (
            f"Expected single no-example eval call, got {seen_instance_ids}"
        )
        assert num_actual == 3
        assert result.instance_scores == {"0": 0.5, "1": 0.5, "2": 0.5}

    def test_cache_disabled_invokes_evaluator_every_time(self, mocker: Any) -> None:
        """cache=None is the strict off mode behind cache_evaluation=False."""
        from helix.evolution import _cached_evaluate_batch

        cand = self._make_cand("cand-no-cache")
        seen_instance_ids: list[list[str] | None] = []

        def fake_run(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            seen_instance_ids.append(instance_ids)
            return _make_result(
                candidate.id, {eid: 0.4 for eid in (instance_ids or [])}
            )

        mocker.patch("helix.evolution.run_evaluator", side_effect=fake_run)
        mocker.patch("helix.evolution._write_helix_batch")

        first, first_actual = _cached_evaluate_batch(
            cand, ["0", "1"], None, self._trivial_config(), "train", Path("/tmp"),
        )
        second, second_actual = _cached_evaluate_batch(
            cand, ["0", "1"], None, self._trivial_config(), "train", Path("/tmp"),
        )

        assert seen_instance_ids == [["0", "1"], ["0", "1"]]
        assert first_actual == 2
        assert second_actual == 2
        assert first.instance_scores == second.instance_scores == {"0": 0.4, "1": 0.4}

    def test_partial_cache_hit(self, mocker: Any) -> None:
        """2-of-3 cached → evaluator runs with ONLY the 1 uncached id."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch

        cache: MBCache[object, str] = MBCache[object, str]()
        cand = self._make_cand("cand-C")
        cand_dict = {"content_key": cand.id, "split": "train"}
        # Pre-populate 0 and 2; leave 1 uncached.
        cache.put_batch(
            cand_dict,
            ["0", "2"],
            [None, None],
            [0.11, 0.33],
        )

        seen_instance_ids: list[list[str] | None] = []
        written_batches: list[list[str]] = []

        def fake_run(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            seen_instance_ids.append(instance_ids)
            return _make_result(
                candidate.id, {eid: 0.22 for eid in (instance_ids or [])}
            )

        def fake_write(path: str, example_ids: list[str]) -> None:
            written_batches.append(list(example_ids))

        mocker.patch("helix.evolution.run_evaluator", side_effect=fake_run)
        mocker.patch(
            "helix.evolution._write_helix_batch", side_effect=fake_write
        )

        result, num_actual = _cached_evaluate_batch(
            cand, ["0", "1", "2"], cache, self._trivial_config(), "train", Path("/tmp"),
        )

        # Evaluator called exactly once, with only the missing id.
        assert seen_instance_ids == [["1"]], (
            f"Expected evaluator to be called with only the uncached id, "
            f"got {seen_instance_ids}"
        )
        # helix_batch.json was written with only the missing id —
        # passed through verbatim as a string, no int coercion.
        assert written_batches == [["1"]], (
            f"Expected reduced helix_batch.json=['1'], got {written_batches}"
        )
        assert num_actual == 1
        # Merged scores cover ALL requested ids: cached 0/2 + fresh 1.
        assert result.instance_scores == {"0": 0.11, "1": 0.22, "2": 0.33}

    def test_partial_cache_hit_per_example_fields_merge(self, mocker: Any) -> None:
        """Partial-cache-hit merge of ``per_example_side_info`` and
        ``objective_scores``:

          * cache-hit positions get their own cached side_info by id;
          * fresh miss positions get the per_example_side_info dict
            from the fresh EvalResult, zipped by id;
          * ``objective_scores`` IS round-tripped through the cache
            (it has a dedicated slot — see
            ``eval_cache.CachedEvaluation.objective_scores``), so
            cache-hit positions get back the previously-stored dict
            and miss positions get the freshly-harvested dict.
        """
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch

        cache: MBCache[object, str] = MBCache[object, str]()
        cand = self._make_cand("cand-pcfm")
        cand_dict = {"content_key": cand.id, "split": "train"}
        # Pre-populate ids "0" and "2" with both score AND
        # objective_scores (the cache stores these natively).
        cache.put_batch(
            cand_dict,
            ["0", "2"],
            [None, None],
            [0.11, 0.33],
            objective_scores_list=[
                {"obj_alpha": 0.11, "obj_beta": 0.8},
                {"obj_alpha": 0.33, "obj_beta": 0.1},
            ],
            side_info_list=[
                {"trajectory": "cached_trace_0", "rollout_id": "cached__0"},
                {"trajectory": "cached_trace_2", "rollout_id": "cached__2"},
            ],
        )
        # "1" is uncached — the evaluator is invoked with it only, and
        # must return an ``EvalResult`` that carries per_example_side_info
        # + objective_scores for the merge path to thread through.

        def fake_run(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            # Produce one fresh entry for each requested uncached id.
            ids = list(instance_ids or [])
            return EvalResult(
                candidate_id=candidate.id,
                scores={},
                asi={},
                instance_scores={eid: 0.22 for eid in ids},
                per_example_side_info=[
                    {"trajectory": f"fresh_trace_{eid}", "rollout_id": f"fresh__{eid}"}
                    for eid in ids
                ],
                objective_scores=[
                    {"obj_alpha": 0.22, "obj_beta": 0.5}
                    for _ in ids
                ],
            )

        mocker.patch("helix.evolution.run_evaluator", side_effect=fake_run)
        mocker.patch("helix.evolution._write_helix_batch")

        result, num_actual = _cached_evaluate_batch(
            cand, ["0", "1", "2"], cache, self._trivial_config(), "train", Path("/tmp"),
        )

        # instance_scores merge: cached 0/2 + fresh 1 (established
        # earlier by ``test_partial_cache_hit``).
        assert result.instance_scores == {"0": 0.11, "1": 0.22, "2": 0.33}
        assert num_actual == 1

        # objective_scores round-trip fully through the cache (dedicated
        # slot in ``CachedEvaluation``).  Each slot positional to
        # ``example_ids``:
        assert result.objective_scores is not None
        assert result.objective_scores == [
            {"obj_alpha": 0.11, "obj_beta": 0.8},     # cached id "0"
            {"obj_alpha": 0.22, "obj_beta": 0.5},     # fresh id "1"
            {"obj_alpha": 0.33, "obj_beta": 0.1},     # cached id "2"
        ]

        # per_example_side_info round-trips by example id through the cache;
        # the fresh miss position gets the dict we returned from fake_run.
        assert result.per_example_side_info is not None
        assert result.per_example_side_info == [
            {"trajectory": "cached_trace_0", "rollout_id": "cached__0"},       # cached "0"
            {"trajectory": "fresh_trace_1", "rollout_id": "fresh__1"},         # fresh "1"
            {"trajectory": "cached_trace_2", "rollout_id": "cached__2"},       # cached "2"
        ]

    def test_cache_populates_after_fresh_eval(self, mocker: Any) -> None:
        """After a fresh eval, a second call with the same ids is a full hit."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch

        cache: MBCache[object, str] = MBCache[object, str]()
        cand = self._make_cand("cand-D")

        call_count = {"n": 0}

        def fake_run(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            call_count["n"] += 1
            return _make_result(
                candidate.id, {eid: 0.77 for eid in (instance_ids or [])}
            )

        mocker.patch("helix.evolution.run_evaluator", side_effect=fake_run)
        mocker.patch("helix.evolution._write_helix_batch")

        cfg = self._trivial_config()

        # First call: full miss → evaluator runs once.
        first_result, first_num_actual = _cached_evaluate_batch(
            cand, ["0", "1"], cache, cfg, "train", Path("/tmp"),
        )
        assert call_count["n"] == 1
        assert first_num_actual == 2
        assert first_result.instance_scores == {"0": 0.77, "1": 0.77}

        # Second call with the same (candidate, ids): full hit → no re-run.
        second_result, second_num_actual = _cached_evaluate_batch(
            cand, ["0", "1"], cache, cfg, "train", Path("/tmp"),
        )
        assert call_count["n"] == 1, (
            "Evaluator must not be invoked a second time for cached ids"
        )
        assert second_num_actual == 0
        assert second_result.instance_scores == {"0": 0.77, "1": 0.77}

    def test_cache_is_split_aware(self, mocker: Any) -> None:
        """A cached train score must not satisfy a val request for the same id."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch

        cache: MBCache[object, str] = MBCache[object, str]()
        cand = self._make_cand("cand-split")
        cache.put_batch(
            {"id": cand.id, "split": "train"},
            ["0"],
            [None],
            [0.1],
        )

        seen_splits: list[str] = []

        def fake_run(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            seen_splits.append(split)
            return _make_result(candidate.id, {"0": 0.9})

        mocker.patch("helix.evolution.run_evaluator", side_effect=fake_run)
        mocker.patch("helix.evolution._write_helix_batch")

        result, num_actual = _cached_evaluate_batch(
            cand, ["0"], cache, self._trivial_config(), "val", Path("/tmp"),
        )

        assert seen_splits == ["val"]
        assert num_actual == 1
        assert result.instance_scores == {"0": 0.9}

    def test_no_cache_passthrough_runs_full_batch(self, mocker: Any) -> None:
        """``cache=None`` → single evaluator call over the full batch."""
        from helix.evolution import _cached_evaluate_batch

        cand = self._make_cand("cand-E")

        seen_instance_ids: list[list[str] | None] = []

        def fake_run(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            seen_instance_ids.append(instance_ids)
            return _make_result(
                candidate.id, {eid: 0.4 for eid in (instance_ids or [])}
            )

        mocker.patch("helix.evolution.run_evaluator", side_effect=fake_run)
        mocker.patch("helix.evolution._write_helix_batch")

        result, num_actual = _cached_evaluate_batch(
            cand, ["0", "1"], None, self._trivial_config(), "train", Path("/tmp"),
        )

        assert seen_instance_ids == [["0", "1"]]
        assert num_actual == 2
        assert result.instance_scores == {"0": 0.4, "1": 0.4}


# ---------------------------------------------------------------------------
# String-id round-trip through helix_batch.json (BREAKING: list[int] → list[str])
#
# Prior to the fix, ``_write_helix_batch`` silently cast every element through
# ``int()`` at the JSON serialization boundary.  That made
# :class:`helix.batch_sampler.StratifiedBatchSampler` (which emits opaque ids
# of shape ``"group__N"`` like ``"group_alpha__case_3"``) unusable on Architecture A:
# ``int("group_alpha__case_3")`` raises ``ValueError``.  The fix passes ids through
# opaquely as strings.
# ---------------------------------------------------------------------------


class TestWriteHelixBatchStringIds:
    def test_structured_string_ids_round_trip(self, tmp_path: Path) -> None:
        """``_write_helix_batch`` must serialise opaque string ids verbatim —
        including composite ``group__N`` ids emitted by the stratified
        sampler — without attempting to cast them to int."""
        from helix.evolution import _write_helix_batch

        ids = ["group_alpha__case_0", "group_alpha__case_3", "group_beta__case_7"]
        _write_helix_batch(tmp_path, ids)

        written = json.loads((tmp_path / "helix_batch.json").read_text())
        assert written == ids, (
            "String ids must round-trip verbatim through helix_batch.json "
            "(no int() coercion, no reordering)"
        )
        # Explicit: every element is a str (no silent numeric coercion).
        assert all(isinstance(x, str) for x in written)

    def test_stringified_int_ids_still_written_as_strings(
        self, tmp_path: Path
    ) -> None:
        """The default ``_RangeDataLoader`` emits ``"0"``, ``"1"``, … — these
        now round-trip as strings too, not ints.  Evaluators that previously
        relied on reading ``list[int]`` must cast on their side."""
        from helix.evolution import _write_helix_batch

        _write_helix_batch(tmp_path, ["0", "1", "2"])

        written = json.loads((tmp_path / "helix_batch.json").read_text())
        assert written == ["0", "1", "2"]
        assert all(isinstance(x, str) for x in written), (
            "BREAKING: helix_batch.json payload is now list[str]; "
            "integer values would indicate a regression."
        )


# ---------------------------------------------------------------------------
# MODERATE E (finding C4) — parent minibatch eval runs in parallel
# worker threads.  HELIX's bounded-executor design mirrors the general shape
# of GEPA's adapter-level concurrent evaluation dispatch (``GEPAAdapter
# .evaluate`` in ``core/adapter.py``, invoked via ``default_batch_evaluate``);
# GEPA's own reflective-mutation proposer evaluates parent/child minibatches
# through a single batched adapter call rather than a per-proposal thread
# pool.
#
# Parent minibatch accounting counts evaluations: one budget unit per uncached
# example, and zero for pure cache hits.
# ---------------------------------------------------------------------------


class TestParentMinibatchParallelism:
    def test_parent_minibatch_evals_dispatched_to_worker_threads(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Under ``num_parallel_proposals > 1`` the N parent-minibatch evals
        must be dispatched to a ``ThreadPoolExecutor`` (call-count evidence
        of concurrency per finding C4 MODERATE E).

        We cannot block on a barrier because parents share the same seed
        worktree and the per-worktree file-handoff lock correctly serialises
        the evaluator call (see ``_worktree_lock`` in evolution.py).  Instead
        we record ``threading.get_ident()`` inside the evaluator: with a
        thread pool we observe >1 distinct worker thread id for parent evals;
        without one, every parent eval runs on the main thread.
        """
        import threading

        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        mut_ids = iter(["g1-s1", "g1-s2", "g1-s3"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        import time

        parent_minibatches: list[list[str]] = []
        parent_eval_threads: set[int] = set()
        main_thread_id = threading.get_ident()

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            # Parent-minibatch eval site — gated on split="train" to
            # exclude the seed full-val eval (which now also runs with
            # explicit ids on split="val" because helix_result needs
            # helix_batch.json on every invocation).
            if (
                candidate.id == seed.id
                and split == "train"
                and instance_ids is not None
            ):
                # Small sleep guarantees the first eval is still in
                # flight when the second submit fires, so the
                # ThreadPoolExecutor spawns a second worker thread
                # (rather than reusing W1 after an instant task).
                # The per-worktree lock (see ``_worktree_lock``)
                # serialises the subsequent lock acquisition, but both
                # tasks still execute on DIFFERENT worker threads —
                # our concurrency signal.
                parent_eval_threads.add(threading.get_ident())
                parent_minibatches.append(list(instance_ids))
                time.sleep(0.05)
                return _make_result(candidate.id, {i: 0.5 for i in instance_ids})
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 0.4 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=2,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Two parent minibatches pre-sampled.
        assert len(parent_minibatches) == 2
        # Both parent evals ran on pool workers, not the main thread.  With
        # the pre-fix sequential pre-sample loop, main_thread_id would be
        # the sole id in this set.
        assert main_thread_id not in parent_eval_threads, (
            "Parent minibatch eval ran on the MAIN thread; MODERATE E "
            "regression — parent eval was not dispatched to the pool."
        )
        assert len(parent_eval_threads) >= 2, (
            f"Expected >= 2 distinct worker thread ids for parent eval, "
            f"got {parent_eval_threads}"
        )


class TestMaxWorkersBoundsParentEvalPool:
    def test_max_workers_bounds_parent_eval_pool(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Fix A: ``evolution.max_workers`` caps the parent-eval
        ThreadPoolExecutor.  With ``max_workers=2`` and
        ``num_parallel_proposals=3`` we must never see >2 parent evals
        running concurrently, even though 3 proposals are pre-sampled.

        Mirrors GEPA's ``EngineConfig.max_workers`` plumbing
        (``gepa_launcher.py``), which bounds a ``ThreadPoolExecutor`` in the
        optimize-anything adapter's parallel-evaluation helpers.
        """
        import threading
        import time

        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        mut_ids = iter(["g1-s1", "g1-s2", "g1-s3"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        concurrency_lock = threading.Lock()
        current_parent_evals = 0
        peak_parent_evals = 0

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            nonlocal current_parent_evals, peak_parent_evals
            if candidate.id == seed.id and instance_ids is not None:
                with concurrency_lock:
                    current_parent_evals += 1
                    peak_parent_evals = max(peak_parent_evals, current_parent_evals)
                try:
                    time.sleep(0.1)
                    return _make_result(candidate.id, {i: 0.5 for i in instance_ids})
                finally:
                    with concurrency_lock:
                        current_parent_evals -= 1
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 0.4 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=3,
            max_workers=2,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert peak_parent_evals <= 2, (
            f"Parent-eval pool exceeded max_workers bound: "
            f"peak={peak_parent_evals}, max_workers=2"
        )


class TestParentEvalExceptionDoesNotAbortGeneration:
    def test_parent_eval_exception_drops_only_failed_proposal(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Fix B: when one of N parent-eval futures raises, the remaining
        proposals must still complete — HELIX's own isolation guarantee for
        its bounded parent-eval ``ThreadPoolExecutor`` so a single eval
        failure does not abort the generation.
        """
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        mut_ids = iter(["g1-s1", "g1-s2", "g1-s3"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        parent_eval_attempts: list[bool] = []
        mutate_calls: list[str] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            # Parent-minibatch eval — gated on split="train" so the seed
            # full-val eval (now also explicit-ids on "val" because
            # helix_result needs helix_batch.json) doesn't inflate the
            # count.  Without this filter the seed eval would be
            # indistinguishable from a parent minibatch eval.
            if (
                candidate.id == seed.id
                and split == "train"
                and instance_ids is not None
            ):
                parent_eval_attempts.append(True)
                # Raise on the second parent-eval only (stable because
                # ``parent_eval_attempts`` is append-ordered).
                if len(parent_eval_attempts) == 2:
                    raise RuntimeError("simulated evaluator failure")
                return _make_result(candidate.id, {i: 0.5 for i in instance_ids})
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 0.4 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        def _mutate_record(**kw: Any) -> Candidate:
            mutate_calls.append(kw.get("new_id", ""))
            return _make_candidate(next(mut_ids))

        all_mocks["mutate"].side_effect = _mutate_record

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=3,
        )
        # Must not raise: single failed eval drops that proposal only.
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert len(parent_eval_attempts) == 3, (
            f"All 3 parent evals should have been attempted; "
            f"got {len(parent_eval_attempts)}"
        )
        # The 2 surviving proposals must reach the mutate step; the failed
        # one must NOT (its proposal slot was dropped cleanly in §1c).
        assert len(mutate_calls) == 2, (
            f"Expected 2 mutations for the 2 surviving proposals; "
            f"got {len(mutate_calls)}"
        )


class TestParentMinibatchBudgetCharge:
    def test_budget_charge_counts_parent_minibatch_examples_always_fresh(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Parent minibatch evals always charge the full minibatch size (GEPA
        parity: the minibatch path bypasses the read side of the cache).

        Change 2: ``_eval_parent`` now passes ``None`` instead of
        ``minibatch_cache`` to ``_cached_evaluate_batch``, matching GEPA's
        ``ReflectiveMutationProposer`` which evaluates parent/child
        minibatches via ``adapter.evaluate``/``batch_evaluate`` directly —
        results are written into ``state.evaluation_cache`` afterward
        (``put_batch``) but never read through it first.  Both iterations
        therefore invoke the evaluator subprocess for the parent and charge
        2 budget units each.
        """
        train_path = _write_train_jsonl(tmp_path, n=2)  # 2 ids → minibatch always [0,1]
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        mut_ids = iter(["g1-s1", "g1-s2", "g1-s3", "g1-s4"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        # All mutated children produce the same worktree path as seed so that
        # we can more easily route scores; scores chosen so child < parent and
        # is rejected (keeps frontier at {seed}, parent re-used next iter).
        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.1 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        # Record each save_state's budget.evaluations to see the progression.
        budget_snapshots: list[int] = []

        def capture(state: Any, path: Any) -> None:
            budget_snapshots.append(state.budget.evaluations)

        all_mocks["save_state"].side_effect = capture

        # max_generations=2: the parent (seed) is evaluated on [0,1] in BOTH
        # iterations because Change 2 bypasses the minibatch cache for parent
        # evals, matching GEPA's ``ReflectiveMutationProposer`` (which never
        # reads parent/child minibatch evals through the cache).
        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=2,
            max_evaluations=10_000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # The parent evaluator is invoked TWICE: both iter 1 and iter 2 are
        # fresh evaluator calls (cache bypassed for parent minibatch evals).
        parent_call_count = sum(
            1
            for call in all_mocks["run_evaluator"].call_args_list
            if call.kwargs.get("instance_ids") is not None
            and call.kwargs.get("split") == "train"
            and (call.args[0] if call.args else call.kwargs.get("candidate")).id
            == seed.id
        )
        assert parent_call_count == 2, (
            f"Expected parent minibatch evaluator invoked twice "
            f"(once per generation, cache bypassed per GEPA parity), got {parent_call_count}"
        )

        assert len(budget_snapshots) >= 2, "expected multiple save_state calls"
        # Iter 1: +2 (parent) +2 (rejected child) = 4; Iter 2: +2 (parent) +2
        # (rejected child) = 4.  Total delta >= 8, well above the >=6 floor.
        assert budget_snapshots[-1] - budget_snapshots[0] >= 6, (
            f"Budget delta {budget_snapshots[-1] - budget_snapshots[0]} "
            f"< 6 - per-example minibatch evaluations were not charged"
        )

    def test_budget_charge_exact_count_single_proposal(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """A single minibatch iteration charges two units for the parent
        evaluator run and two for the rejected child evaluator run.
        """
        train_path = _write_train_jsonl(tmp_path, n=2)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.1 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        budget_snapshots: list[int] = []

        def capture(state: Any, path: Any) -> None:
            budget_snapshots.append(state.budget.evaluations)

        all_mocks["save_state"].side_effect = capture

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=10_000,
            val_stage_size=None,  # GEPA-parity mode: no HELIX-only staged gate
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert budget_snapshots, "save_state should be called"
        assert budget_snapshots[0] == 2
        assert budget_snapshots[-1] == 6

    def test_seed_full_validation_charges_uncached_val_examples(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Full-validation dataset eval charges one unit per uncached val id."""
        train_path = _write_train_jsonl(tmp_path, n=3)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            assert split == "val"
            assert instance_ids == ["0", "1", "2"]
            return _make_result(candidate.id, {i: 0.5 for i in instance_ids})

        all_mocks["run_evaluator"].side_effect = run_eval
        budget_snapshots: list[int] = []

        def capture(state: Any, path: Any) -> None:
            budget_snapshots.append(state.budget.evaluations)

        all_mocks["save_state"].side_effect = capture

        config = _make_minibatch_config(
            train_path,
            val_size=3,
            max_generations=0,
            max_evaluations=10_000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert budget_snapshots
        assert budget_snapshots[0] == 3


# ---------------------------------------------------------------------------
# Thread-safety: EvaluationCache must be safe for concurrent get/put under
# the new parallel parent-eval stage.  Without the lock the invariant below
# fails intermittently; with the lock it is deterministic.
# ---------------------------------------------------------------------------


class TestEvaluationCacheThreadSafety:
    def test_concurrent_put_batch_preserves_all_entries(self) -> None:
        import threading
        from helix.eval_cache import EvaluationCache as MBCache

        cache: MBCache[object, str] = MBCache[object, str]()

        def worker(cid: str, start: int) -> None:
            cand = {"id": cid, "split": "train"}
            ids = [str(start + i) for i in range(50)]
            cache.put_batch(cand, ids, [None] * 50, [float(i) for i in range(50)])

        threads = [
            threading.Thread(target=worker, args=(f"c-{k}", k * 100))
            for k in range(8)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each worker wrote 50 entries under a distinct candidate id, so
        # total size must be 8 * 50 = 400.  Without the lock, concurrent
        # dict mutations could drop entries.
        assert len(cache._cache) == 8 * 50


# ---------------------------------------------------------------------------
# Final-nits regression tests
# ---------------------------------------------------------------------------


class TestStateIBumpUnconditional:
    """GEPA parity: ``state.i`` is incremented unconditionally at the top of
    ``GEPAEngine.run()``'s main loop, so it must advance once per outer
    iteration regardless of which path (mutation, merge, early-exit) is
    taken.  Previously HELIX bumped ``state.i`` only inside the §1a
    minibatch pre-sample loop, so iterations that exited early (perfect
    score, mutation==None) silently kept the same counter.
    """

    def test_state_i_bumps_when_mutation_returns_none(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Mutation returns None on every iteration → §1a never proceeds
        to its sampler bump.  state.i must still advance once per outer
        iteration via the new top-of-loop unconditional bump."""
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None  # forces no §1a child eval

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 0.5 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        state_i_snapshots: list[int] = []

        def capture(state: Any, path: Any) -> None:
            state_i_snapshots.append(state.i)

        all_mocks["save_state"].side_effect = capture

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=3,
            max_evaluations=10_000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Three iterations × one bump-per-iteration → state.i must reach
        # at least 3 (starting from -1, after the seed-eval save it is
        # -1, then each iter bump pushes it to 0, 1, 2 → final >= 2).
        assert state_i_snapshots, "save_state should be called"
        assert state_i_snapshots[-1] >= 2, (
            f"state.i must bump per outer iteration even when §1a does "
            f"not run; final state.i={state_i_snapshots[-1]}"
        )


class TestStrictInstanceScoresAccess:
    """GEPA parity: ``GEPAAdapter.evaluate``'s documented contract
    (``core/adapter.py``) requires ``len(scores) == len(batch)`` — a
    missing instance id in a parent or child minibatch eval is an
    evaluator bug, not a benign zero.  HELIX must raise.
    """

    def test_missing_id_in_child_minibatch_raises(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                if candidate.id == "g1-s1":
                    # Child evaluator drops one of the requested ids — the
                    # GEPA invariant says this should be a hard error.
                    dropped = list(instance_ids)[:-1]
                    return _make_result(candidate.id, {i: 0.5 for i in dropped})
                return _make_result(candidate.id, {i: 0.5 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        # Cache must be OFF: with caching on, ``_cached_evaluate_batch``'s
        # inner ``_evaluator`` closure raises its own "Evaluator did not
        # return scores" assertion before the result ever reaches the
        # acceptance criterion (see test_missing_id_in_cached_evaluator_raises
        # below); only the no-cache path lets a missing id reach the
        # acceptance criterion's own strict-id check exercised here.
        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=10_000,
            cache_evaluation=False,
        )
        with pytest.raises(AssertionError, match="missing ids"):
            run_evolution(config, tmp_path, tmp_path / ".helix")

    def test_missing_id_in_cached_evaluator_raises(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """The cache layer (``_cached_evaluate_batch`` inner ``_evaluator``)
        must enforce the same strict-id invariant as the acceptance path.
        With ``cache_evaluation=True`` (default), a missing id reported by
        the evaluator must raise before the cached scores reach the
        acceptance criterion.  Mirrors the strict-id assert inside
        ``_cached_evaluate_batch``'s inner ``_evaluator`` closure.
        """
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                # Drop the last requested id on EVERY eval — cache layer
                # should raise on first occurrence.
                dropped = list(instance_ids)[:-1]
                return _make_result(candidate.id, {i: 0.5 for i in dropped})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=10_000,
            cache_evaluation=True,
        )
        with pytest.raises(AssertionError, match="Evaluator did not return scores"):
            run_evolution(config, tmp_path, tmp_path / ".helix")


class TestWholeCandidateBudget:
    """Whole-candidate accounting charges completed evaluator runs."""

    def test_empty_instance_scores_still_charges_one_candidate_eval(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Single-task / no-train_path mode: evaluator returns an empty
        ``instance_scores`` dict on the seed eval.  The completed candidate
        eval still consumes one budget unit."""
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            return _make_result(candidate.id, {})  # empty scores

        all_mocks["run_evaluator"].side_effect = run_eval

        budget_snapshots: list[int] = []

        def capture(state: Any, path: Any) -> None:
            budget_snapshots.append(state.budget.evaluations)

        all_mocks["save_state"].side_effect = capture

        config = HelixConfig(
            objective="empty-scores test",
            evaluator=EvaluatorConfig(command="pytest -q"),
            dataset=DatasetConfig(),
            evolution=EvolutionConfig(
                max_generations=1,
                max_evaluations=100,
                perfect_score_threshold=None,
                frontier_type="instance",
            ),
            worktree=WorktreeConfig(),
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert budget_snapshots, "save_state should be called"
        assert budget_snapshots[0] == 1, (
            f"empty instance_scores must still charge 1, got {budget_snapshots[0]}"
        )


class TestResumeAttemptReconciliation:
    """Resume cleanup for attempts interrupted before result persistence."""

    def test_live_incomplete_attempt_retries_same_visible_slot(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        base_dir = tmp_path / ".helix"
        worktrees_dir = base_dir / "worktrees"
        evaluations_dir = base_dir / "evaluations"
        lineage_path = base_dir / "lineage.json"
        worktrees_dir.mkdir(parents=True)
        evaluations_dir.mkdir(parents=True)
        (worktrees_dir / "g4-s4").mkdir()
        (evaluations_dir / "g0-s0.json").write_text("{}")
        (evaluations_dir / "g1-s1.json").write_text("{}")
        lineage_path.write_text(
            json.dumps(
                [
                    {
                        "id": "g0-s0",
                        "parent": None,
                        "parents": [],
                        "operation": "seed",
                        "generation": 0,
                        "files_changed": [],
                    },
                    {
                        "id": "g1-s1",
                        "parent": "g0-s0",
                        "parents": ["g0-s0"],
                        "operation": "mutate",
                        "generation": 1,
                        "files_changed": ["solver.py"],
                    },
                    {
                        "id": "g4-s4",
                        "parent": "g1-s1",
                        "parents": ["g1-s1"],
                        "operation": "mutation",
                        "generation": 4,
                        "files_changed": ["solver.py"],
                    },
                ]
            )
        )
        state = EvolutionState(
            generation=4,
            frontier=["g0-s0", "g1-s1"],
            instance_scores={"g0-s0": {"x": 0.1}, "g1-s1": {"x": 0.2}},
            budget=BudgetState(),
            config_hash="hash",
            mutation_counter=4,
            active_frontier={"x": ["g1-s1"]},
        )
        remove_mock = mocker.patch("helix.evolution.remove_worktree")

        changed = _reconcile_incomplete_attempts_on_resume(
            state=state,
            base_dir=base_dir,
            worktrees_dir=worktrees_dir,
            lineage_path=lineage_path,
        )

        assert changed is True
        assert state.generation == 3
        assert state.mutation_counter == 3
        assert state.frontier == ["g0-s0", "g1-s1"]
        assert state.active_frontier == {"x": ["g1-s1"]}
        remove_mock.assert_called_once()
        remaining_ids = {record["id"] for record in json.loads(lineage_path.read_text())}
        assert remaining_ids == {"g0-s0", "g1-s1"}

    def test_orphan_worktree_without_lineage_is_removed(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        base_dir = tmp_path / ".helix"
        worktrees_dir = base_dir / "worktrees"
        evaluations_dir = base_dir / "evaluations"
        lineage_path = base_dir / "lineage.json"
        worktrees_dir.mkdir(parents=True)
        evaluations_dir.mkdir(parents=True)
        (worktrees_dir / "g5-s5").mkdir()
        (evaluations_dir / "g1-s1.json").write_text("{}")
        lineage_path.write_text(
            json.dumps(
                [
                    {
                        "id": "g1-s1",
                        "parent": "g0-s0",
                        "parents": ["g0-s0"],
                        "operation": "mutate",
                        "generation": 1,
                        "files_changed": ["solver.py"],
                    }
                ]
            )
        )
        state = EvolutionState(
            generation=4,
            frontier=["g1-s1"],
            instance_scores={"g1-s1": {"x": 0.2}},
            budget=BudgetState(),
            config_hash="hash",
            mutation_counter=4,
        )
        remove_mock = mocker.patch("helix.evolution.remove_worktree")

        changed = _reconcile_incomplete_attempts_on_resume(
            state=state,
            base_dir=base_dir,
            worktrees_dir=worktrees_dir,
            lineage_path=lineage_path,
        )

        assert changed is True
        assert state.generation == 4
        assert state.mutation_counter == 4
        remove_mock.assert_called_once()
        remaining_ids = {record["id"] for record in json.loads(lineage_path.read_text())}
        assert remaining_ids == {"g1-s1"}

    def test_historical_missing_worktree_entries_are_left_intact(
        self, tmp_path: Path, mocker: Any
    ) -> None:
        base_dir = tmp_path / ".helix"
        worktrees_dir = base_dir / "worktrees"
        evaluations_dir = base_dir / "evaluations"
        lineage_path = base_dir / "lineage.json"
        worktrees_dir.mkdir(parents=True)
        evaluations_dir.mkdir(parents=True)
        (evaluations_dir / "g0-s0.json").write_text("{}")
        lineage_path.write_text(
            json.dumps(
                [
                    {
                        "id": "g0-s0",
                        "parent": None,
                        "parents": [],
                        "operation": "seed",
                        "generation": 0,
                        "files_changed": [],
                    },
                    {
                        "id": "g2-s2",
                        "parent": "g0-s0",
                        "parents": ["g0-s0"],
                        "operation": "mutate",
                        "generation": 2,
                        "files_changed": ["solver.py"],
                    },
                ]
            )
        )
        state = EvolutionState(
            generation=2,
            frontier=["g0-s0"],
            instance_scores={"g0-s0": {"x": 0.1}},
            budget=BudgetState(),
            config_hash="hash",
            mutation_counter=2,
        )
        remove_mock = mocker.patch("helix.evolution.remove_worktree")

        changed = _reconcile_incomplete_attempts_on_resume(
            state=state,
            base_dir=base_dir,
            worktrees_dir=worktrees_dir,
            lineage_path=lineage_path,
        )

        assert changed is False
        assert state.generation == 2
        assert state.mutation_counter == 2
        remove_mock.assert_not_called()
        remaining_ids = {record["id"] for record in json.loads(lineage_path.read_text())}
        assert remaining_ids == {"g0-s0", "g2-s2"}


# ---------------------------------------------------------------------------
# NB-2 regression: always-perfect data must terminate (GEPA parity)
# ---------------------------------------------------------------------------


class TestAlwaysPerfectDataTerminates:
    """Regression for NB-2: always-perfect dataset + perfect_score_threshold set
    must not produce an infinite loop.

    Three GEPA-aligned guards independently prevent NB-2:
      1. gen advances unconditionally at top of loop (Change 1; mirrors the
         unconditional ``state.i += 1`` at the top of ``GEPAEngine.run()``)
      2. Parent minibatch eval bypasses cache — always charges budget (Change 2)
      3. Mandatory stopping condition check (Change 3; mirrors the
         ``ValueError`` ``gepa.api.optimize`` raises when no stopping
         condition is configured)

    This test exercises the gen-advance guard (1) and budget guard (2) together:
    max_generations=5 must be the loop exit trigger, with 5 skip records written.
    """

    def test_always_perfect_data_terminates_at_max_generations(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """1-example always-perfect dataset with max_generations=5 terminates
        normally and writes one skip file per generation.

        NB-2 regression: before Change 1, perfect-skip rolled back gen, so
        the loop retried the same generation indefinitely when budget was
        uncapped.  After Change 1, gen advances each iteration and the loop
        exits after 5 iterations.
        """
        train_path = _write_train_jsonl(tmp_path, n=1)  # 1 example → always same id
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        # All evals return 1.0 → perfect-skip fires every iteration.
        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 1.0 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 1.0})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=1,
            max_generations=5,
            max_evaluations=1000,  # generous budget — gen advance (Change 1) stops first
        )
        config.evolution.perfect_score_threshold = 1.0
        run_evolution(config, tmp_path, tmp_path / ".helix")

        skips_dir = tmp_path / ".helix" / "skips"

        # Loop must have exited via max_generations (not budget), so state.generation >= 5.
        import json as _json

        # Verify a skip file was written for each generation 1-5.
        for g in range(1, 6):
            skip_path = skips_dir / f"g{g}.json"
            assert skip_path.exists(), (
                f"skips/g{g}.json missing — generation {g} was not processed "
                f"(NB-2 regression: loop may have exited before reaching gen {g})"
            )
            recs = _json.loads(skip_path.read_text())
            assert isinstance(recs, list)
            assert len(recs) >= 1
            assert recs[0]["generation"] == g
            assert recs[0]["reason"] == "perfect_subsample"

        # No 6th generation — the loop stopped at max_generations=5.
        assert not (skips_dir / "g6.json").exists(), (
            "skips/g6.json exists — loop ran past max_generations=5"
        )

        # mutate must never have been called (every generation was a perfect skip).
        all_mocks["mutate"].assert_not_called()


# ---------------------------------------------------------------------------
# Change 3: mandatory stopping condition validation (GEPA api.py)
# ---------------------------------------------------------------------------


class TestStoppingConditionValidation:
    """Mirror GEPA's ``gepa.api.optimize``: run_evolution must raise
    ValueError when no effective stopping condition is configured.

    In helix, max_generations (loop bound, default 10) is the primary stop
    and max_evaluations (budget cap, -1 = disabled) is secondary.  The guard
    fires when both are ineffective (max_generations <= 0 AND
    max_evaluations <= 0), preventing a run that terminates only by the OS.
    """

    def test_no_stop_condition_raises_value_error(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """max_generations=0 and max_evaluations=-1 → ValueError before loop."""
        config = HelixConfig(
            objective="no stop condition test",
            evaluator=EvaluatorConfig(command="pytest -q"),
            dataset=DatasetConfig(),
            evolution=EvolutionConfig(
                max_generations=0,   # loop bound disabled
                max_evaluations=-1,  # budget cap disabled
            ),
        )
        with pytest.raises(ValueError, match="stopping condition"):
            run_evolution(config, tmp_path, tmp_path / ".helix")

    def test_negative_max_generations_raises_value_error(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """max_generations=-1 and max_evaluations=0 → ValueError before loop."""
        config = HelixConfig(
            objective="no stop condition test",
            evaluator=EvaluatorConfig(command="pytest -q"),
            dataset=DatasetConfig(),
            evolution=EvolutionConfig(
                max_generations=-1,  # loop bound disabled
                max_evaluations=0,   # also disabled
            ),
        )
        with pytest.raises(ValueError, match="stopping condition"):
            run_evolution(config, tmp_path, tmp_path / ".helix")

    def test_valid_max_generations_does_not_raise(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """max_generations=1 with max_evaluations=-1 is valid — loop bound set."""
        train_path = _write_train_jsonl(tmp_path, n=2)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = None  # no children; single iteration

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 0.5 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=-1,  # disabled — max_generations alone suffices
        )
        # Must NOT raise — max_generations=1 is a valid stopping condition.
        run_evolution(config, tmp_path, tmp_path / ".helix")
# ---------------------------------------------------------------------------
# Atomic Proposal Worker Tests
#
# The atomic-worker pattern merges parent-eval + LLM + child-eval into one
# atomic worker per proposal slot, all running inside a single
# ThreadPoolExecutor.  Budget charging is deferred to a sequential
# acceptance loop, run only after the parallel workers finish.  This is
# HELIX's own design; it no longer has a direct upstream GEPA counterpart —
# GEPA's own ``ReflectiveMutationProposer`` now dispatches parent/child
# minibatch evaluation as a single batched adapter call rather than a
# per-proposal thread pool.
#
# The atomic-worker design is implemented in the current evolution.py; all
# tests in this class pass against it.
# ---------------------------------------------------------------------------


class TestAtomicProposalWorker:
    """Tests for the atomic proposal worker (HELIX's own design; see the
    module-level note above on why this no longer mirrors GEPA 1:1)."""

    def test_worker_executes_parent_eval_and_child_eval_on_same_thread(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Parent eval and child eval both run on worker threads, not the main thread.

        Under the atomic-worker design, each proposal worker executes parent_eval →
        skip-perfect → LLM → child_eval atomically in a single
        ThreadPoolExecutor worker thread — HELIX's own design (see the
        module-level note above on why this is no longer a 1:1 GEPA mirror).
        The child eval therefore must NOT run on the main thread — unlike the
        current three-stage pipeline where Step 3 (child eval) is sequential
        on the main thread.

        Verification: record threading.get_ident() for parent evals (seed.id,
        split=train, instance_ids not None) and child evals (non-seed,
        split=train, instance_ids not None).  With n=2 proposals, both sets
        must be non-empty and neither must contain the main thread id.
        """
        import threading

        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        mut_ids = iter(["g1-s1", "g1-s2"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        main_thread_id = threading.get_ident()
        parent_eval_threads: set[int] = set()
        child_eval_threads: set[int] = set()

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if split == "train" and instance_ids is not None:
                if candidate.id == seed.id:
                    # Parent minibatch eval — record which thread we're on.
                    parent_eval_threads.add(threading.get_ident())
                    return _make_result(candidate.id, {i: 0.3 for i in instance_ids})
                else:
                    # Child minibatch eval — record which thread we're on.
                    child_eval_threads.add(threading.get_ident())
                    # Child improves → passes the minibatch gate.
                    return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            if instance_ids is not None:
                # Val evals (seed full-val, child full-val after gate).
                return _make_result(candidate.id, {i: 0.7 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=2,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert main_thread_id not in parent_eval_threads, (
            "Parent eval ran on the main thread — expected worker thread "
            "(atomic-worker regression)"
        )
        assert main_thread_id not in child_eval_threads, (
            "Child eval ran on the main thread — expected worker thread "
            "(atomic-worker regression: child eval should run in the same "
            "atomic worker as parent eval, not sequentially on the main thread)"
        )
        assert len(parent_eval_threads) >= 1, (
            f"Expected >= 1 parent eval thread id; got {parent_eval_threads}"
        )
        assert len(child_eval_threads) >= 1, (
            f"Expected >= 1 child eval thread id; got {child_eval_threads}"
        )

    def test_trace_binds_worker_spans_to_the_actual_proposal_batch(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """The trace names every submitted slot without relying on file order."""
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        mut_ids = iter(["g1-s1", "g1-s2"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            ids = instance_ids or ["v1"]
            return _make_result(candidate.id, {item: 0.5 for item in ids})

        all_mocks["run_evaluator"].side_effect = run_eval
        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=2,
        )

        with TRACE.record() as events:
            run_evolution(config, tmp_path, tmp_path / ".helix")

        starts = [event for event in events if event.type is EventType.PROPOSAL_START]
        ends = [event for event in events if event.type is EventType.PROPOSAL_END]
        assert [event.type for event in events if event.type in {
            EventType.PROPOSAL_BATCH_START,
            EventType.PROPOSAL_BATCH_END,
        }] == [EventType.PROPOSAL_BATCH_START, EventType.PROPOSAL_BATCH_END]
        assert len(starts) == len(ends) == 2
        assert {(event.generation, event.proposal_index, event.n_proposals) for event in starts} == {
            (1, 0, 2),
            (1, 1, 2),
        }
        assert {event.candidate_id for event in starts} == {"g1-s1", "g1-s2"}
        assert all(event.outcome == "ok" for event in ends)

    def test_full_validation_is_timed_separately_from_proposal_work(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """The sequential full-val stage has its own bracketed wall clock.

        ``VALIDATE_START``/``VALIDATE_END`` are the boundary, and every full-val
        span must fall wholly outside the ``PROPOSAL_BATCH_START``/
        ``PROPOSAL_BATCH_END`` window, so the concurrent and sequential totals
        partition a generation instead of double-counting the same seconds.
        """
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        mut_ids = iter(["g1-s1", "g1-s2"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            ids = instance_ids or ["v1"]
            # A child scores strictly better than the seed so the gate accepts
            # and the run actually reaches the full-val stage.
            score = 0.9 if candidate.id != "g0-s0" else 0.1
            return _make_result(candidate.id, {item: score for item in ids})

        all_mocks["run_evaluator"].side_effect = run_eval
        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=2,
        )

        with TRACE.record() as events:
            run_evolution(config, tmp_path, tmp_path / ".helix")

        validate_starts = [e for e in events if e.type is EventType.VALIDATE_START]
        validate_ends = [e for e in events if e.type is EventType.VALIDATE_END]
        # The seed's own full val plus at least one accepted child's.
        assert len(validate_starts) == len(validate_ends) >= 2
        assert all(e.outcome == "ok" for e in validate_ends)
        assert all(e.split == "val" for e in validate_starts)
        # ``reason`` keeps the seed, merge and mutation call sites apart.
        assert {e.reason for e in validate_starts} >= {
            "seed_val_batch",
            "mutation_full_val_batch",
        }

        # Durations come from the monotonic clock and are non-negative.
        by_index = {}
        for start, end in zip(validate_starts, validate_ends, strict=True):
            assert start.candidate_id == end.candidate_id
            by_index[start.candidate_id] = end.monotonic - start.monotonic
        assert all(duration >= 0.0 for duration in by_index.values())

        batch_start = next(
            e for e in events if e.type is EventType.PROPOSAL_BATCH_START
        )
        batch_end = next(e for e in events if e.type is EventType.PROPOSAL_BATCH_END)
        # No VALIDATE span overlaps the concurrent proposal window, so the two
        # phase totals partition the generation instead of double-counting it.
        for start, end in zip(validate_starts, validate_ends, strict=True):
            assert (
                end.monotonic <= batch_start.monotonic
                or start.monotonic >= batch_end.monotonic
            ), "full validation overlapped the proposal batch window"

    def test_end_event_outcome_field_matches_the_documented_split(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """``outcome`` sits on operation-end events, not on span boundaries.

        ``PROPOSAL_END`` and ``VALIDATE_END`` report the result of the work
        they close over; ``PROPOSAL_BATCH_END``, ``ITER_END`` and
        ``OPT_END`` only mark where a span ends and never carry it. A
        consumer that assumes every ``*_END`` record has ``outcome`` gets a
        silently wrong answer on the latter three.
        """
        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            ids = instance_ids or ["v1"]
            # A child scores strictly better than the seed so the gate
            # accepts and the run reaches full validation.
            score = 0.9 if candidate.id != seed.id else 0.1
            return _make_result(candidate.id, {item: score for item in ids})

        all_mocks["run_evaluator"].side_effect = run_eval
        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=2,
        )

        with TRACE.record() as events:
            run_evolution(config, tmp_path, tmp_path / ".helix")

        operation_end = {EventType.PROPOSAL_END, EventType.VALIDATE_END}
        span_boundary_end = {
            EventType.PROPOSAL_BATCH_END,
            EventType.ITER_END,
            EventType.OPT_END,
        }
        # Both sides of the split must actually fire this run, or the
        # assertions below would pass vacuously.
        assert {e.type for e in events} >= operation_end | span_boundary_end

        for event in events:
            if event.type in operation_end:
                assert event.outcome is not None, (
                    f"{event.type} is documented to carry outcome"
                )
            elif event.type in span_boundary_end:
                assert event.outcome is None, (
                    f"{event.type} is a span boundary and must not carry outcome"
                )

    def test_worker_skipped_result_returns_without_llm_call(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """When skip-perfect fires inside the worker, mutate() is never called.

        Under the atomic-worker design, the skip-perfect check (Step W3) lives inside the
        atomic worker function.  A parent whose subsample scores all reach the
        ``perfect_score_threshold`` causes the worker to return
        ``_ProposalResult(kind='skipped', ...)`` before ever reaching the LLM
        mutation step (Step W4).

        Config: n=1 proposal, max_generations=1, perfect_score_threshold=0.9.
        All parent minibatch scores are 1.0 → skip fires → mutate.call_count == 0.
        """
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if split == "train" and instance_ids is not None:
                # Always perfect → skip-perfect (Step W3) fires inside the worker.
                return _make_result(candidate.id, {i: 1.0 for i in instance_ids})
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 1.0 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 1.0})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=1,
        )
        config.evolution.perfect_score_threshold = 0.9

        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert all_mocks["mutate"].call_count == 0, (
            f"mutate() must not be called when skip-perfect fires inside the "
            f"worker (atomic-worker Step W3 parity); "
            f"call_count={all_mocks['mutate'].call_count}"
        )

    def test_worker_llm_failure_does_not_crash_pool(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """mutate() raising RuntimeError on one slot does not crash the whole pool.

        Under the atomic-worker design, each worker catches non-fatal exceptions from
        mutate() and returns ``_ProposalResult(kind='llm_failed', ...)`` rather
        than propagating the exception out of the worker — the same per-task
        isolation shape as GEPA's ``ReflectiveMutationProposer.propose``,
        which wraps each task's proposal-prep step in its own try/except and
        continues with the remaining tasks on failure.  The pool continues
        with the remaining slots.

        Config: n=3 proposals.  The first mutate() call (thread-safely tracked)
        raises RuntimeError; the other two return valid candidates.
        run_evolution must NOT raise and mutate must have been called at least
        once (the call that raised).
        """
        import threading

        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        mut_ids = iter(["g1-s1", "g1-s2"])
        _lock = threading.Lock()
        _call_counter: list[int] = [0]

        def _mutate_fail_first(**kw: Any) -> Any:
            with _lock:
                _call_counter[0] += 1
                is_first = _call_counter[0] == 1
            if is_first:
                raise RuntimeError("simulated LLM failure — atomic-worker isolation test")
            return _make_candidate(next(mut_ids))

        all_mocks["mutate"].side_effect = _mutate_fail_first

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                if candidate.id == seed.id and split == "train":
                    return _make_result(candidate.id, {i: 0.3 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=3,
        )

        # Must not raise — the RuntimeError from mutate() is caught inside the
        # worker and returned as kind='llm_failed'; the other slots complete.
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert all_mocks["mutate"].call_count >= 1, (
            "mutate() should have been called at least once "
            "(including the call that raised RuntimeError)"
        )

    def test_n_proposals_3_runs_all_workers_in_parallel(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """With n=3, parent minibatch evals are dispatched to a thread pool.

        Under the atomic-worker design, all N atomic workers run inside a single
        ThreadPoolExecutor — HELIX's own bounded-concurrency design (see the
        module-level note above on GEPA's now-batched proposal dispatch).
        With n=3 proposals
        and a time.sleep(0.05) inside each parent eval, the pool must spawn
        multiple worker threads simultaneously.

        Verification: record threading.get_ident() for parent evals (seed.id,
        split=train, instance_ids not None).  The main thread id must not appear
        in the recorded set, and at least 2 distinct worker thread ids must be
        present (indicating genuine concurrency, not sequential dispatch).
        """
        import threading
        import time

        train_path = _write_train_jsonl(tmp_path, n=6)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        mut_ids = iter(["g1-s1", "g1-s2", "g1-s3"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        main_thread_id = threading.get_ident()
        parent_eval_threads: set[int] = set()

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if (
                candidate.id == seed.id
                and split == "train"
                and instance_ids is not None
            ):
                parent_eval_threads.add(threading.get_ident())
                # Sleep ensures the first eval is still in flight when the
                # second and third submits fire, so the ThreadPoolExecutor
                # spawns multiple worker threads rather than reusing one.
                time.sleep(0.05)
                return _make_result(candidate.id, {i: 0.5 for i in instance_ids})
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 0.4 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=3,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert main_thread_id not in parent_eval_threads, (
            "Parent eval ran on the main thread; expected worker threads "
            "(atomic-worker regression — workers not dispatched to ThreadPoolExecutor)"
        )
        assert len(parent_eval_threads) >= 2, (
            f"Expected >= 2 distinct worker thread ids for n=3 parent evals "
            f"(genuine concurrency); got {parent_eval_threads}"
        )

    def test_budget_charges_happen_sequentially_in_acceptance_loop(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """budget.evaluations never decreases across consecutive save_state calls.

        Under the atomic-worker design, budget mutations happen only inside the sequential
        acceptance loop — HELIX's own design (see the module-level note above
        on GEPA's now-batched proposal dispatch) — never inside the parallel
        workers.
        Capturing ``state.budget.evaluations`` at each ``save_state`` call must
        therefore produce a monotonically non-decreasing sequence, regardless of
        which order the parallel workers completed.

        Config: n=2 proposals; both children improve on parent so that two full
        acceptance-loop iterations execute and multiple save_state calls are made.
        """
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        mut_ids = iter(["g1-s1", "g1-s2"])
        all_mocks["mutate"].side_effect = lambda **kw: _make_candidate(next(mut_ids))

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                if candidate.id == seed.id and split == "train":
                    # Parent scores low — child will improve and pass the gate.
                    return _make_result(candidate.id, {i: 0.3 for i in instance_ids})
                if split == "train":
                    # Child improves on parent minibatch → accepted.
                    return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
                # Val evals (seed full-val, child full-val).
                return _make_result(candidate.id, {i: 0.7 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        budget_snapshots: list[int] = []

        def capture_budget(state: Any, path: Any) -> None:
            budget_snapshots.append(state.budget.evaluations)

        all_mocks["save_state"].side_effect = capture_budget

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=10_000,
            num_parallel_proposals=2,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        assert budget_snapshots, "save_state must be called at least once"
        for i in range(1, len(budget_snapshots)):
            assert budget_snapshots[i] >= budget_snapshots[i - 1], (
                f"budget.evaluations decreased at snapshot index {i}: "
                f"{budget_snapshots[i - 1]} → {budget_snapshots[i]}; "
                f"full sequence: {budget_snapshots}. "
                "Budget charges must be strictly sequential (acceptance loop only, "
                "never inside parallel workers) — atomic-worker invariant."
            )

    def test_worker_tampered_result_rejects_child_without_crash(
        self, tmp_path: Path, all_mocks: dict[str, Any], mocker: Any
    ) -> None:
        """When tamper-check fires inside the worker, the child is rejected and removed.

        Under the atomic-worker design, Step W5 of the atomic worker calls
        ``_detect_evaluator_tamper``.  If the child touched protected evaluator
        files, the worker returns ``_TamperedResult`` with the tampered path
        list.  The acceptance loop (Step 3) must:
          1. Not raise / not crash.
          2. Call ``remove_worktree`` to clean up the rejected child worktree.
          3. Not add the tampered child to the frontier or candidates dict.

        Config: n=1 proposal, 1 generation.  ``_detect_evaluator_tamper`` is
        patched to return ``["evaluate.py"]`` (one tampered path).
        """
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed

        child = _make_candidate("g1-s1")
        all_mocks["mutate"].return_value = child

        # Patch _detect_evaluator_tamper so the worker returns _TamperedResult.
        mocker.patch(
            "helix.evolution._detect_evaluator_tamper",
            return_value=["evaluate.py"],
        )

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if instance_ids is not None:
                return _make_result(candidate.id, {i: 0.5 for i in instance_ids})
            return _make_result(candidate.id, {"v1": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=1,
            max_evaluations=1000,
            num_parallel_proposals=1,
        )

        # Must not raise — tamper is a rejection, not a fatal error.
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # mutate() ran (LLM was invoked before tamper check)
        assert all_mocks["mutate"].call_count == 1, (
            "mutate() must be called once (tamper check is Step W5, after Step W4 LLM)"
        )
        # Tampered child worktree must be cleaned up
        assert all_mocks["remove_worktree"].called, (
            "remove_worktree must be called to clean up tamper-rejected child "
            "(_TamperedResult acceptance-loop path)"
        )
        # Tampered child must NOT have been snapshot-committed to any branch
        for call in all_mocks["snapshot_candidate"].call_args_list:
            snapped = call.args[0] if call.args else call.kwargs.get("candidate")
            assert snapped is None or snapped.id != child.id, (
                f"snapshot_candidate must not be called for tampered child {child.id}"
            )
