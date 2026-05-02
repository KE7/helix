"""Unit tests for seedless-mode integration in helix.evolution.run_evolution."""

from __future__ import annotations


import pytest

from helix.config import (
    DatasetConfig,
    EvaluatorConfig,
    EvolutionConfig,
    HelixConfig,
    WorktreeConfig,
)
from helix.evolution import run_evolution
from helix.exceptions import MutationError
from helix.population import Candidate, EvalResult
from helix.state import BudgetState, EvolutionState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_candidate(cid: str = "g0-s0", generation: int = 0) -> Candidate:
    return Candidate(
        id=cid,
        worktree_path=f"/tmp/helix/{cid}",
        branch_name=f"helix/{cid}",
        generation=generation,
        parent_id=None,
        parent_ids=[],
        operation="seed",
    )


def make_eval_result(
    candidate_id: str = "g0-s0",
    instance_scores: dict[str, float] | None = None,
) -> EvalResult:
    if instance_scores is None:
        instance_scores = {"i1": 0.5}
    return EvalResult(
        candidate_id=candidate_id,
        scores={"score": 0.5},
        asi={},
        instance_scores=instance_scores,
    )


def make_config(
    seedless: bool = False,
    objective: str = "Optimise the solver",
    max_generations: int = 1,
    max_evaluations: int = 1000,
) -> HelixConfig:
    from helix.config import SeedlessConfig

    return HelixConfig(
        objective=objective,
        seedless=SeedlessConfig(enabled=seedless),
        evaluator=EvaluatorConfig(command="pytest -q"),
        dataset=DatasetConfig(),
        evolution=EvolutionConfig(
            max_generations=max_generations,
            max_evaluations=max_evaluations,
            perfect_score_threshold=None,
        ),
        worktree=WorktreeConfig(),
    )


@pytest.fixture
def seedless_mocks(mocker):
    """Patch all external I/O for seedless run_evolution tests."""
    seed_candidate = make_candidate("g0-s0")
    seed_result = make_eval_result("g0-s0")
    seed_result.candidate_id = "g0-s0"

    mocks = {
        "create_empty_seed_worktree": mocker.patch(
            "helix.evolution.create_empty_seed_worktree",
            return_value=seed_candidate,
        ),
        "create_seed_worktree": mocker.patch(
            "helix.evolution.create_seed_worktree",
            return_value=seed_candidate,
        ),
        "build_seed_generation_prompt": mocker.patch(
            "helix.evolution.build_seed_generation_prompt",
            return_value="<seed prompt>",
        ),
        "generate_seed": mocker.patch("helix.evolution.generate_seed", return_value={}),
        "run_evaluator": mocker.patch(
            "helix.evolution.run_evaluator",
            return_value=seed_result,
        ),
        "mutate": mocker.patch("helix.evolution.mutate", return_value=None),
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
        "render_budget": mocker.patch("helix.evolution.render_budget"),
        "render_generation": mocker.patch("helix.evolution.render_generation"),
    }
    return mocks


# ---------------------------------------------------------------------------
# test_run_evolution_seedless_calls_generate_seed
# ---------------------------------------------------------------------------


class TestRunEvolutionSeedless:
    def test_seedless_calls_create_empty_seed_worktree(self, tmp_path, seedless_mocks):
        """Seedless mode must call create_empty_seed_worktree, NOT create_seed_worktree."""
        config = make_config(seedless=True)

        run_evolution(config, tmp_path, tmp_path / ".helix")

        seedless_mocks["create_empty_seed_worktree"].assert_called_once()
        seedless_mocks["create_seed_worktree"].assert_not_called()

    def test_seedless_calls_build_seed_generation_prompt(self, tmp_path, seedless_mocks):
        """Seedless mode must call build_seed_generation_prompt once."""
        config = make_config(seedless=True)

        run_evolution(config, tmp_path, tmp_path / ".helix")

        seedless_mocks["build_seed_generation_prompt"].assert_called_once()

    def test_seedless_calls_generate_seed_exactly_once(self, tmp_path, seedless_mocks):
        """Seedless mode must call generate_seed exactly once (no retry loop)."""
        config = make_config(seedless=True)

        run_evolution(config, tmp_path, tmp_path / ".helix")

        seedless_mocks["generate_seed"].assert_called_once()

    def test_seedless_generate_seed_receives_prompt(self, tmp_path, seedless_mocks):
        """generate_seed must be called with the prompt returned by build_seed_generation_prompt."""
        config = make_config(seedless=True)
        seedless_mocks["build_seed_generation_prompt"].return_value = "<the seed prompt>"

        run_evolution(config, tmp_path, tmp_path / ".helix")

        args, _ = seedless_mocks["generate_seed"].call_args
        assert args[1] == "<the seed prompt>"

    def test_seedless_generate_seed_receives_worktree_path(self, tmp_path, seedless_mocks):
        """generate_seed must be called with the worktree_path of the seed candidate."""
        config = make_config(seedless=True)
        seed_cand = make_candidate("g0-s0")
        seed_cand_wt = "/tmp/helix/g0-s0"
        seed_cand = Candidate(
            id="g0-s0",
            worktree_path=seed_cand_wt,
            branch_name="helix/g0-s0",
            generation=0,
            parent_id=None,
            parent_ids=[],
            operation="seed",
        )
        seedless_mocks["create_empty_seed_worktree"].return_value = seed_cand

        run_evolution(config, tmp_path, tmp_path / ".helix")

        args, _ = seedless_mocks["generate_seed"].call_args
        assert args[0] == seed_cand_wt


# ---------------------------------------------------------------------------
# test_run_evolution_seedless_raises_immediately_on_failure
# ---------------------------------------------------------------------------


class TestRunEvolutionSeedlessFailFast:
    def test_raises_immediately_when_generate_seed_fails(self, tmp_path, seedless_mocks):
        """If generate_seed raises, run_evolution should propagate the exception immediately."""
        config = make_config(seedless=True)
        exc = MutationError("Claude Code exited with code 1", exit_code=1)
        seedless_mocks["generate_seed"].side_effect = exc

        with pytest.raises(MutationError) as exc_info:
            run_evolution(config, tmp_path, tmp_path / ".helix")

        assert exc_info.value is exc

    def test_generate_seed_called_only_once_on_failure(self, tmp_path, seedless_mocks):
        """generate_seed must only be called once even on failure (no retry)."""
        config = make_config(seedless=True)
        seedless_mocks["generate_seed"].side_effect = MutationError("fail", exit_code=1)

        with pytest.raises(MutationError):
            run_evolution(config, tmp_path, tmp_path / ".helix")

        assert seedless_mocks["generate_seed"].call_count == 1


# ---------------------------------------------------------------------------
# test_run_evolution_non_seedless_uses_seed_path
# ---------------------------------------------------------------------------


class TestRunEvolutionNonSeedless:
    def test_non_seedless_uses_create_seed_worktree(self, tmp_path, seedless_mocks):
        """config.seedless=False must use create_seed_worktree, not create_empty_seed_worktree."""
        config = make_config(seedless=False)

        run_evolution(config, tmp_path, tmp_path / ".helix")

        seedless_mocks["create_seed_worktree"].assert_called_once()
        seedless_mocks["create_empty_seed_worktree"].assert_not_called()

    def test_non_seedless_does_not_call_generate_seed(self, tmp_path, seedless_mocks):
        """config.seedless=False must NOT call generate_seed."""
        config = make_config(seedless=False)

        run_evolution(config, tmp_path, tmp_path / ".helix")

        seedless_mocks["generate_seed"].assert_not_called()

    def test_non_seedless_does_not_call_build_seed_prompt(self, tmp_path, seedless_mocks):
        """config.seedless=False must NOT call build_seed_generation_prompt."""
        config = make_config(seedless=False)

        run_evolution(config, tmp_path, tmp_path / ".helix")

        seedless_mocks["build_seed_generation_prompt"].assert_not_called()


# ---------------------------------------------------------------------------
# test_run_evolution_resume_skips_seedless
# ---------------------------------------------------------------------------


class TestRunEvolutionResume:
    def test_resume_skips_seedless_entirely(self, tmp_path, seedless_mocks):
        """When existing state is loaded (resume), seedless generation must be skipped."""
        config = make_config(seedless=True)

        # Simulate an existing state (resuming a run)
        existing_state = EvolutionState(
            generation=1,
            frontier=["g0-s0"],
            instance_scores={"g0-s0": {"i1": 0.5}},
            budget=BudgetState(evaluations=1),
            config_hash="abc123",
        )
        seedless_mocks["load_state"].return_value = existing_state
        # _load_evaluation must return a result for frontier reconstruction
        seed_result = make_eval_result("g0-s0")
        seedless_mocks["_load_evaluation"].return_value = seed_result

        # Create a fake worktree path for the restored candidate
        worktree_path = tmp_path / ".helix" / "worktrees" / "g0-s0"
        worktree_path.mkdir(parents=True, exist_ok=True)

        run_evolution(config, tmp_path, tmp_path / ".helix")

        # Neither seedless path should be triggered
        seedless_mocks["create_empty_seed_worktree"].assert_not_called()
        seedless_mocks["generate_seed"].assert_not_called()
        seedless_mocks["build_seed_generation_prompt"].assert_not_called()

    def test_resume_non_seedless_also_skips_seed_creation(self, tmp_path, seedless_mocks):
        """Resume with seedless=False also skips create_seed_worktree."""
        config = make_config(seedless=False)

        existing_state = EvolutionState(
            generation=1,
            frontier=["g0-s0"],
            instance_scores={"g0-s0": {"i1": 0.5}},
            budget=BudgetState(evaluations=1),
            config_hash="abc123",
        )
        seedless_mocks["load_state"].return_value = existing_state
        seed_result = make_eval_result("g0-s0")
        seedless_mocks["_load_evaluation"].return_value = seed_result

        worktree_path = tmp_path / ".helix" / "worktrees" / "g0-s0"
        worktree_path.mkdir(parents=True, exist_ok=True)

        run_evolution(config, tmp_path, tmp_path / ".helix")

        seedless_mocks["create_seed_worktree"].assert_not_called()
        seedless_mocks["generate_seed"].assert_not_called()
