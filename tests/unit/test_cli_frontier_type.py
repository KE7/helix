"""Tests for Commit Q: CLI display commands read ``frontier_type``
from persisted state.json rather than defaulting to ``"instance"``.

Motivation: the evolution run that produced ``.helix/state.json`` used
some ``config.evolution.frontier_type`` (hybrid / objective /
cartesian / instance).  ``helix frontier`` / ``helix best`` /
``helix log`` must display the frontier with the SAME dimensionality
the run actually used — regardless of whether the user has edited
``helix.toml`` between the evolve call and the display call.

The fix (persist + read back) keeps display consistent across:
  * ``helix evolve`` with ``frontier_type="hybrid"``
  * human edits ``helix.toml`` to ``frontier_type="instance"``
  * ``helix best --dir <project>`` → still uses ``"hybrid"``
    because the state.json is the source of truth for what the
    existing frontier represents.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from helix.cli import _reconstruct_frontier
from helix.population import Candidate, EvalResult
from helix.state import BudgetState, EvolutionState, load_state, save_state


def _write_evaluation(base_dir: Path, cid: str, result: EvalResult) -> None:
    eval_dir = base_dir / ".helix" / "evaluations"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / f"{cid}.json").write_text(json.dumps(result.to_dict(), indent=2))


def _write_fake_worktree(base_dir: Path, cid: str) -> None:
    wt = base_dir / ".helix" / "worktrees" / cid
    wt.mkdir(parents=True, exist_ok=True)


class TestFrontierTypePersistence:
    def test_state_round_trip_preserves_frontier_type(self, tmp_path: Path) -> None:
        state = EvolutionState(
            generation=3,
            frontier=["g0-s0", "g1-s1"],
            instance_scores={},
            budget=BudgetState(),
            config_hash="abc",
            frontier_type="hybrid",
        )
        save_state(state, tmp_path)
        loaded = load_state(tmp_path)
        assert loaded is not None
        assert loaded.frontier_type == "hybrid"

    @pytest.mark.parametrize(
        "variant", ["instance", "objective", "hybrid", "cartesian"],
    )
    def test_state_round_trip_all_variants(
        self, tmp_path: Path, variant: str,
    ) -> None:
        state = EvolutionState(
            generation=1,
            frontier=[],
            instance_scores={},
            budget=BudgetState(),
            config_hash="h",
            frontier_type=variant,  # type: ignore[arg-type]
        )
        save_state(state, tmp_path)
        loaded = load_state(tmp_path)
        assert loaded is not None
        assert loaded.frontier_type == variant

    def test_legacy_state_without_field_defaults_to_instance(
        self, tmp_path: Path,
    ) -> None:
        """state.json from a pre-multi-axis HELIX run has no
        ``frontier_type`` key; load_state must default to
        ``"instance"`` (HELIX's historical single-axis behaviour)
        rather than raising or leaking ``None``."""
        helix_dir = tmp_path / ".helix"
        helix_dir.mkdir()
        legacy = {
            "schema_version": 1,
            "generation": 0,
            "frontier": [],
            "instance_scores": {},
            "budget": {"evaluations": 0},
            "config_hash": "x",
            "mutation_counter": 0,
            "merge_counter": 0,
            "total_merge_invocations": 0,
            "merge_attempted_pairs": [],
            "merge_description_triplets": [],
            "i": -1,
            "num_metric_calls_by_discovery": {},
            # frontier_type intentionally absent
        }
        (helix_dir / "state.json").write_text(json.dumps(legacy))
        loaded = load_state(tmp_path)
        assert loaded is not None
        assert loaded.frontier_type == "instance"

    def test_corrupt_frontier_type_defaults_to_instance(
        self, tmp_path: Path,
    ) -> None:
        """A state.json with a bogus frontier_type literal (e.g. from
        a hand-edit) should not crash — narrow to ``"instance"``."""
        helix_dir = tmp_path / ".helix"
        helix_dir.mkdir()
        corrupt = {
            "schema_version": 1,
            "generation": 0,
            "frontier": [],
            "instance_scores": {},
            "budget": {"evaluations": 0},
            "config_hash": "x",
            "frontier_type": "not-a-real-literal",
        }
        (helix_dir / "state.json").write_text(json.dumps(corrupt))
        loaded = load_state(tmp_path)
        assert loaded is not None
        assert loaded.frontier_type == "instance"


class TestReconstructFrontierUsesPersistedType:
    """``_reconstruct_frontier`` builds the display ``ParetoFrontier``
    with ``state.frontier_type``, NOT the ``ParetoFrontier.__init__``
    default (``"instance"``).  Guarantees display-path consistency
    after a post-evolve helix.toml edit."""

    def _build_state(
        self, tmp_path: Path, frontier_type: str = "hybrid",
    ) -> EvolutionState:
        # Seed a minimal on-disk artifact set so _reconstruct_frontier
        # picks up the single frontier member.
        base_dir = tmp_path / ".helix"
        _write_fake_worktree(tmp_path, "g0-s0")
        er = EvalResult(
            candidate_id="g0-s0",
            scores={},
            asi={},
            instance_scores={"a": 1.0, "b": 0.0},
            objective_scores=[{"obj_a": 0.9}, {"obj_a": 0.1}],
        )
        _write_evaluation(tmp_path, "g0-s0", er)
        state = EvolutionState(
            generation=0,
            frontier=["g0-s0"],
            instance_scores={"g0-s0": er.instance_scores},
            budget=BudgetState(),
            config_hash="h",
            frontier_type=frontier_type,  # type: ignore[arg-type]
        )
        save_state(state, tmp_path)
        return state

    def test_reconstruct_uses_hybrid_when_state_says_hybrid(
        self, tmp_path: Path,
    ) -> None:
        state = self._build_state(tmp_path, frontier_type="hybrid")
        base_dir = tmp_path / ".helix"
        frontier, _cands, _skipped = _reconstruct_frontier(base_dir, state)
        # ParetoFrontier exposes the active type via its property.
        assert frontier.frontier_type == "hybrid"

    def test_reconstruct_uses_objective_when_state_says_objective(
        self, tmp_path: Path,
    ) -> None:
        state = self._build_state(tmp_path, frontier_type="objective")
        base_dir = tmp_path / ".helix"
        frontier, _c, _s = _reconstruct_frontier(base_dir, state)
        assert frontier.frontier_type == "objective"

    def test_reconstruct_defaults_to_instance_without_attr(
        self, tmp_path: Path,
    ) -> None:
        """A state-like object lacking ``frontier_type`` (e.g. an
        ad-hoc test fixture) falls back to ``"instance"`` — the
        pre-Q display default, preserved."""
        _write_fake_worktree(tmp_path, "g0-s0")
        er = EvalResult(
            candidate_id="g0-s0",
            scores={},
            asi={},
            instance_scores={"a": 1.0},
        )
        _write_evaluation(tmp_path, "g0-s0", er)

        class _LegacyState:
            generation = 0
            frontier = ["g0-s0"]
            instance_scores: dict[str, dict[str, float]] = {"g0-s0": {"a": 1.0}}

        base_dir = tmp_path / ".helix"
        frontier, _c, _s = _reconstruct_frontier(base_dir, _LegacyState())
        assert frontier.frontier_type == "instance"


class TestDisplayIgnoresCurrentConfig:
    """End-to-end mock: state.json was written with ``frontier_type="hybrid"``;
    a subsequent display call must use ``"hybrid"`` for its frontier,
    even if the current helix.toml config says something else.

    This is the core L7 guarantee — display is driven by the PERSISTED
    run, not the live config.
    """

    def test_display_uses_state_type_not_config(self, tmp_path: Path) -> None:
        # Persist a "hybrid" frontier to state.json.
        _write_fake_worktree(tmp_path, "g0-s0")
        er = EvalResult(
            candidate_id="g0-s0",
            scores={},
            asi={},
            instance_scores={"a": 1.0, "b": 0.5},
            objective_scores=[{"obj_a": 0.9}, {"obj_a": 0.1}],
        )
        _write_evaluation(tmp_path, "g0-s0", er)
        state = EvolutionState(
            generation=0,
            frontier=["g0-s0"],
            instance_scores={"g0-s0": er.instance_scores},
            budget=BudgetState(),
            config_hash="h",
            frontier_type="hybrid",
        )
        save_state(state, tmp_path)

        # Reload state (simulating a fresh CLI invocation) and build
        # the display frontier — must still be "hybrid" even though
        # nothing about the current config has been consulted.
        reloaded = load_state(tmp_path)
        assert reloaded is not None
        base_dir = tmp_path / ".helix"
        frontier, _c, _s = _reconstruct_frontier(base_dir, reloaded)
        assert frontier.frontier_type == "hybrid", (
            "helix frontier / helix best must use the state.json axis, "
            "not whatever helix.toml currently says"
        )
