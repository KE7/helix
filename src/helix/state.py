"""HELIX evolution state persistence."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BudgetState:
    """Tracks resource consumption during evolution.

    Counts the total number of candidate evaluations performed.
    """
    evaluations: int = 0


class EvaluationCache:
    """Simple evaluation cache keyed by (candidate_id, split).

    GEPA parity: avoids re-evaluating identical candidates.
    GEPA uses (candidate_hash, example_id); HELIX uses (candidate_id, split)
    since HELIX evaluates whole candidates via shell commands.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def get(self, candidate_id: str, split: str) -> dict[str, Any] | None:
        """Return cached result dict or None."""
        return self._cache.get((candidate_id, split))

    def put(self, candidate_id: str, split: str, result_dict: dict[str, Any]) -> None:
        """Store a result in the cache."""
        self._cache[(candidate_id, split)] = result_dict

    def __len__(self) -> int:
        return len(self._cache)


@dataclass
class EvolutionState:
    """Persistent state for the HELIX evolution run.

    Tracks current generation, Pareto frontier, scores, budgets, and
    operation counters. Serialized to .helix/state.json for resumption.
    """
    generation: int
    frontier: list[str]
    instance_scores: dict[str, Any]  # dict[str, dict[str, float]] — candidate_id -> instance -> score
    budget: BudgetState
    config_hash: str
    mutation_counter: int = 0
    merge_counter: int = 0
    # Total merge invocations across the entire run (GEPA: lifetime cap).
    total_merge_invocations: int = 0
    # GEPA parity (Fix 12): track attempted merge pairs to avoid re-attempting.
    # Each entry is [cid_i, cid_j] sorted lexicographically.
    merge_attempted_pairs: list[list[str]] = field(default_factory=list)
    # GEPA parity (§5.1 minibatch integration): monotonic proposal counter.
    # Starts at -1 and is bumped to 0 before the first minibatch sample.
    # Mirrors GEPA ``state.i`` in core/state.py.
    i: int = -1


_STATE_FILENAME = "state.json"
_STATE_DIR = ".helix"


def _state_path(base_dir: Path) -> Path:
    return base_dir / _STATE_DIR / _STATE_FILENAME


def save_state(state: EvolutionState, base_dir: Path) -> None:
    """Atomically write the evolution state to .helix/state.json."""
    target = _state_path(base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "generation": state.generation,
        "frontier": state.frontier,
        "instance_scores": state.instance_scores,
        "budget": asdict(state.budget),
        "config_hash": state.config_hash,
        "mutation_counter": state.mutation_counter,
        "merge_counter": state.merge_counter,
        "total_merge_invocations": state.total_merge_invocations,
        "merge_attempted_pairs": state.merge_attempted_pairs,
        "i": state.i,
    }

    # Atomic write: write to tmp file in same directory, then rename
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state(base_dir: Path) -> EvolutionState | None:
    """Load evolution state from .helix/state.json, or return None if absent."""
    target = _state_path(base_dir)
    if not target.exists():
        return None

    with open(target) as f:
        data = json.load(f)

    budget_data = data.get("budget", {})
    budget = BudgetState(
        evaluations=budget_data.get("evaluations", 0),
    )

    return EvolutionState(
        generation=data["generation"],
        frontier=data["frontier"],
        instance_scores=data.get("instance_scores", {}),
        budget=budget,
        config_hash=data["config_hash"],
        mutation_counter=data.get("mutation_counter", 0),
        merge_counter=data.get("merge_counter", 0),
        total_merge_invocations=data.get("total_merge_invocations", 0),
        merge_attempted_pairs=data.get("merge_attempted_pairs", []),
        i=data.get("i", -1),
    )
