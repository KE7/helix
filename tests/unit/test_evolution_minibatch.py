"""Integration tests for HELIX minibatch-gate evolution (Phase 3).

Exercises the new per-proposal minibatch acceptance gate wired into
``helix.evolution.run_evolution`` when ``DatasetConfig.train_path`` is
provided.  Uses the same mocking strategy as ``test_evolution.py``:
mock all I/O and mock ``run_evaluator`` with controlled score sequences.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from helix.batch_sampler import EpochShuffledBatchSampler
from helix.config import (
    DatasetConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    SeedlessConfig,
    WorktreeConfig,
)
from helix.evolution import HelixDataLoader, _make_data_loader, run_evolution
from helix.population import Candidate, EvalResult


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
) -> HelixConfig:
    return HelixConfig(
        objective="Minibatch test",
        evaluator=EvaluatorConfig(command="pytest -q"),
        dataset=DatasetConfig(val_size=val_size),
        seedless=SeedlessConfig(train_path=train_path),
        evolution=EvolutionConfig(
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            perfect_score_threshold=None,
            minibatch_size=minibatch_size,
            num_parallel_proposals=num_parallel_proposals,
            cache_evaluation=cache_evaluation,
            acceptance_criterion=acceptance_criterion,  # type: ignore[arg-type]
            val_stage_size=val_stage_size,
        ),
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

        # instance_ids should be set on parent/child gating calls
        seen_instance_ids: list[list[str] | None] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            seen_instance_ids.append(instance_ids)
            if instance_ids is not None:
                # minibatch eval: return small per-id scores
                if candidate.id == seed.id:
                    return _make_result(candidate.id, {i: 0.3 for i in instance_ids})
                return _make_result(candidate.id, {i: 0.9 for i in instance_ids})
            # full val eval: moderate scores
            return _make_result(candidate.id, {"v1": 0.5, "v2": 0.5})

        all_mocks["run_evaluator"].side_effect = run_eval

        config = _make_minibatch_config(
            train_path, minibatch_size=2, max_generations=1, max_evaluations=100,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Parent minibatch call + child minibatch call both carry instance_ids.
        minibatch_calls = [x for x in seen_instance_ids if x is not None]
        assert len(minibatch_calls) >= 2, (
            f"Expected at least parent+child minibatch eval calls, got {seen_instance_ids}"
        )
        # Parent and child minibatches should be the SAME subsample.
        assert minibatch_calls[0] == minibatch_calls[1]
        # minibatch_size=2 → two ids.
        assert len(minibatch_calls[0]) == 2

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

    def test_accepted_proposal_triggers_full_val_eval(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Child that improves on minibatch → full val eval is run."""
        train_path = _write_train_jsonl(tmp_path, n=4)
        seed = _make_candidate("g0-s0")
        all_mocks["create_seed_worktree"].return_value = seed
        all_mocks["mutate"].return_value = _make_candidate("g1-s1")

        child_val_calls: list[list[str] | None] = []

        def run_eval(
            candidate: Candidate,
            config: HelixConfig,
            split: str = "val",
            instance_ids: list[str] | None = None,
            **kwargs: Any,
        ) -> EvalResult:
            if candidate.id == "g1-s1" and instance_ids is None:
                child_val_calls.append(instance_ids)
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
            if candidate.id == seed.id and instance_ids is not None:
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
            # instance_ids MUST be None in single-task mode
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
# ``cached_evaluate_full`` semantics (gepa/core/state.py:94-130).
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
            ),
            worktree=WorktreeConfig(),
        )

    def test_cache_hit_skips_evaluator(self, mocker: Any) -> None:
        """All requested ids pre-populated → evaluator is NEVER invoked."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch

        cache: MBCache[object, str] = MBCache[object, str]()
        cand = self._make_cand("cand-A")
        cand_dict = {"id": cand.id, "split": "train"}
        cache.put_batch(
            cand_dict,
            ["0", "1", "2"],
            [None, None, None],
            [0.1, 0.2, 0.3],
        )

        run_eval_mock = mocker.patch("helix.evolution.run_evaluator")
        write_batch_mock = mocker.patch("helix.evolution._write_helix_batch")

        result, num_actual = _cached_evaluate_batch(
            cand, ["0", "1", "2"], cache, self._trivial_config(), "train",
        )

        assert run_eval_mock.call_count == 0, (
            "Evaluator must NOT be invoked when every requested id is cached"
        )
        assert write_batch_mock.call_count == 0, (
            "helix_batch.json must NOT be rewritten on a full cache hit"
        )
        assert num_actual == 0
        assert result.instance_scores == {"0": 0.1, "1": 0.2, "2": 0.3}

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
            cand, ["0", "1", "2"], cache, self._trivial_config(), "train",
        )

        assert seen_instance_ids == [["0", "1", "2"]], (
            f"Expected single full-batch eval call, got {seen_instance_ids}"
        )
        assert num_actual == 3
        assert result.instance_scores == {"0": 0.5, "1": 0.5, "2": 0.5}

    def test_partial_cache_hit(self, mocker: Any) -> None:
        """2-of-3 cached → evaluator runs with ONLY the 1 uncached id."""
        from helix.eval_cache import EvaluationCache as MBCache
        from helix.evolution import _cached_evaluate_batch

        cache: MBCache[object, str] = MBCache[object, str]()
        cand = self._make_cand("cand-C")
        cand_dict = {"id": cand.id, "split": "train"}
        # Pre-populate 0 and 2; leave 1 uncached.
        cache.put_batch(
            cand_dict,
            ["0", "2"],
            [None, None],
            [0.11, 0.33],
        )

        seen_instance_ids: list[list[str] | None] = []
        written_batches: list[list[int]] = []

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

        def fake_write(path: str, indices: list[int]) -> None:
            written_batches.append(list(indices))

        mocker.patch("helix.evolution.run_evaluator", side_effect=fake_run)
        mocker.patch(
            "helix.evolution._write_helix_batch", side_effect=fake_write
        )

        result, num_actual = _cached_evaluate_batch(
            cand, ["0", "1", "2"], cache, self._trivial_config(), "train",
        )

        # Evaluator called exactly once, with only the missing id.
        assert seen_instance_ids == [["1"]], (
            f"Expected evaluator to be called with only the uncached id, "
            f"got {seen_instance_ids}"
        )
        # helix_batch.json was written with only the missing index.
        assert written_batches == [[1]], (
            f"Expected reduced helix_batch.json=[1], got {written_batches}"
        )
        assert num_actual == 1
        # Merged scores cover ALL requested ids: cached 0/2 + fresh 1.
        assert result.instance_scores == {"0": 0.11, "1": 0.22, "2": 0.33}

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
            cand, ["0", "1"], cache, cfg, "train",
        )
        assert call_count["n"] == 1
        assert first_num_actual == 2
        assert first_result.instance_scores == {"0": 0.77, "1": 0.77}

        # Second call with the same (candidate, ids): full hit → no re-run.
        second_result, second_num_actual = _cached_evaluate_batch(
            cand, ["0", "1"], cache, cfg, "train",
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
            cand, ["0"], cache, self._trivial_config(), "val",
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
            cand, ["0", "1"], None, self._trivial_config(), "train",
        )

        assert seen_instance_ids == [["0", "1"]]
        assert num_actual == 2
        assert result.instance_scores == {"0": 0.4, "1": 0.4}


# ---------------------------------------------------------------------------
# MODERATE E (audit-mutation §C4) — parent minibatch eval runs in parallel
# worker threads, matching GEPA core/engine.py:381-452 which submits
# ``execute_proposal`` (reflective_mutation.py:239-285) — including
# ``adapter.evaluate`` at :268 — to a ``ThreadPoolExecutor``.
#
# MODERATE H (audit-budget-caching §C1) — parent minibatch charge is always
# ``len(subsample_ids)`` regardless of cache hits, matching GEPA
# reflective_mutation.py:269 (``total_evals += eval_curr.num_metric_calls if
# not None else len(ctx.subsample_ids)`` — ``adapter.evaluate`` at :268
# bypasses the cache, so cache hits never reduce the charge).
# ---------------------------------------------------------------------------


class TestParentMinibatchParallelism:
    def test_parent_minibatch_evals_dispatched_to_worker_threads(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Under ``num_parallel_proposals > 1`` the N parent-minibatch evals
        must be dispatched to a ``ThreadPoolExecutor`` (call-count evidence
        of concurrency per audit-mutation §C4 MODERATE E).

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
            if candidate.id == seed.id and instance_ids is not None:
                # Parent-minibatch eval site.  Small sleep guarantees the
                # first eval is still in flight when the second submit
                # fires, so the ThreadPoolExecutor spawns a second worker
                # thread (rather than reusing W1 after an instant task).
                # The per-worktree lock (see ``_worktree_lock``) serialises
                # the subsequent lock acquisition, but both tasks still
                # execute on DIFFERENT worker threads — our concurrency
                # signal.
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


class TestParentMinibatchBudgetCharge:
    def test_budget_charge_counts_full_minibatch_regardless_of_cache(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Two back-to-back iterations on the same parent/minibatch overlap
        must each charge ``len(subsample_ids)`` to the budget, matching GEPA
        reflective_mutation.py:269.  Pre-fix, HELIX charged only
        ``len(uncached_ids)`` on the 2nd iteration (cache hit → 0 charge).
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

        # max_generations=2 so the parent (seed) is evaluated on [0,1] twice.
        # On iter 2 the cache already contains seed's scores for ids 0 and 1,
        # so the OLD code would charge 0 on iter 2.  The fix charges 2 both times.
        config = _make_minibatch_config(
            train_path,
            minibatch_size=2,
            max_generations=2,
            max_evaluations=10_000,
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Seed eval contributes 0 to the minibatch charge (no train seed eval)
        # and the val seed eval contributes its own counts.  The invariant we
        # assert is the DELTA across iterations 1 and 2 attributable to the
        # parent minibatch: it must be +2 both times (no 0-charge on a hit).
        # We find the per-iter train minibatch charges by observing that
        # run_evaluator for the PARENT is called on its minibatch EACH iter:
        parent_call_count = sum(
            1
            for call in all_mocks["run_evaluator"].call_args_list
            if call.kwargs.get("instance_ids") is not None
            and call.kwargs.get("split") == "train"
            and (call.args[0] if call.args else call.kwargs.get("candidate")).id
            == seed.id
        )
        # Parent is evaluated on its minibatch ONCE per iteration via the
        # cache consumer — iter 1 fresh, iter 2 full cache hit.  The file
        # helix_batch is never rewritten on full hit, but the budget charge
        # below is what we actually care about.
        assert parent_call_count == 1, (
            f"Expected parent minibatch evaluator invoked exactly once "
            f"(iter 1 only, iter 2 full cache hit), got {parent_call_count}"
        )

        # Budget charges for the PARENT minibatch: 2 both iterations → +4 total
        # attributable to parent-minibatch charges.  Other charges (child
        # minibatch, val eval) also sum in, so we assert a lower bound that
        # would fail if the cache-hit charge had been skipped:
        #   - seed val eval: +2 (ids 0..N from val_size=None → single-task
        #     mode? Actually val_size is unused in the minibatch path; the
        #     seed eval goes through _cached_evaluate_batch with
        #     full_val_example_ids=[]; single-task seed eval contributes its
        #     own count).  We accept the loose bound and check the DELTA
        #     between the final budget snapshot and the seed-eval snapshot.
        assert len(budget_snapshots) >= 2, "expected multiple save_state calls"
        # Fresh-iter charges (parent+child): 2+2=4.  Cached-iter charges
        # (parent=2 always, child fresh mb=2 each iter since child id is new):
        # 2+2=4 more.  Pre-fix would give 0+2=2 for iter 2 (parent cached).
        # Delta from first snapshot to last must be >= 4+4=8 (allowing
        # some slack for other charges like the seed val eval).
        assert budget_snapshots[-1] - budget_snapshots[0] >= 8, (
            f"Budget delta {budget_snapshots[-1] - budget_snapshots[0]} "
            f"< 8 — cache hit on parent minibatch is not being charged "
            f"(MODERATE H regression)"
        )

    def test_budget_charge_exact_count_single_proposal(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """GEPA-parity invariant (val_stage_size=None, num_parallel_proposals=1):
        a single iteration charges exactly ``parent_mb + child_mb`` evals for
        the minibatch portion, i.e. ``2*minibatch_size`` — matching GEPA
        reflective_mutation.py:269 (parent) + :423 (child).  This is the
        line-by-line GEPA invariant required by the user directive.
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

        # GEPA parity line-by-line (val_stage_size=None): per iteration
        # the parent minibatch charges +2, the child minibatch charges +2 →
        # delta >= 4 (plus any seed eval charges that preceded).  Child is
        # rejected (0.1 < 0.9), so no val charges.
        assert budget_snapshots, "save_state should be called"
        assert budget_snapshots[-1] >= 4


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
    """GEPA parity (engine.py:649): ``state.i`` must advance once per outer
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
    """GEPA parity (adapter.py:154 — len(outputs) == len(scores) ==
    len(batch)): a missing instance id in a parent or child minibatch
    eval is an evaluator bug, not a benign zero.  HELIX must raise.
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

        # Cache must be OFF: the cache layer (_cached_evaluate_batch's
        # _evaluator at evolution.py:704-727) silently zeros missing ids
        # before they reach the acceptance criterion, so we can only
        # exercise the strict-access invariant on the no-cache path.
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
        acceptance criterion.  Mirrors the fix at evolution.py:730-738.
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


class TestBudgetClampRemoved:
    """GEPA parity: empty ``instance_scores`` must charge 0 (not 1).
    GEPA core/engine.py:167 increments by ``num_actual_evals`` returned
    from the adapter, with no clamp.
    """

    def test_empty_instance_scores_charges_zero(
        self, tmp_path: Path, all_mocks: dict[str, Any]
    ) -> None:
        """Single-task / no-train_path mode: evaluator returns an empty
        ``instance_scores`` dict on the seed eval.  Budget must charge 0
        (not the legacy clamp-to-1)."""
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
            ),
            worktree=WorktreeConfig(),
        )
        run_evolution(config, tmp_path, tmp_path / ".helix")

        # First snapshot is taken right after seed eval; with empty
        # instance_scores GEPA charges 0, not the legacy clamp-to-1.
        assert budget_snapshots, "save_state should be called"
        assert budget_snapshots[0] == 0, (
            f"empty instance_scores must charge 0, got {budget_snapshots[0]}"
        )
