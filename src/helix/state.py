"""HELIX evolution state persistence."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from helix.population import FrontierType


# GEPA parity (audit-rng-state-persist D1):
# GEPA core/state.py:153 declares ``_VALIDATION_SCHEMA_VERSION: ClassVar[int] = 5``
# and migrates older state dicts on load (state.py:355-376).  HELIX previously
# had no schema version on ``state.json``; bumping to 1 marks the first
# versioned schema (the unversioned predecessor is treated as v0 on load).
SCHEMA_VERSION: int = 1


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
    # Each entry is [cid_i, cid_j] sorted lexicographically.  Kept for
    # backward-compat with existing state files; the within-propose retry
    # filter in ``lineage.find_merge_triplet`` reads this set to short-
    # circuit already-seen pairs (merge-pairing audit B2).
    merge_attempted_pairs: list[list[str]] = field(default_factory=list)
    # GEPA parity (merge-pairing audit C1, /tmp/audit_audit-merge-pairing.md:28-31):
    # mirrors GEPA ``merges_performed[1]`` at gepa/proposer/merge.py:195-203.
    # Each entry is [cid_i, cid_j, desc_hash] with cid_i <= cid_j
    # lexicographically and desc_hash = post-snapshot git SHA of the
    # merged worktree.  Blocks only the *same* (pair, output) triplet,
    # so the same pair can retry if a different ancestor/ordering yields
    # a different merged output.
    merge_description_triplets: list[list[str]] = field(default_factory=list)
    # GEPA parity (§5.1 minibatch integration): monotonic proposal counter.
    # Starts at -1 and is bumped to 0 before the first minibatch sample.
    # Mirrors GEPA ``state.i`` in core/state.py.
    i: int = -1
    # GEPA parity (audit-rng-state-persist C/§3): per-program discovery budget.
    # GEPA tracks ``num_metric_calls_by_discovery: list[int]`` indexed by
    # program_idx (state.py:177, appended at state.py:537).  HELIX uses
    # candidate_id strings, so the dict keys by id and stores the value of
    # ``state.budget.evaluations`` at the moment the candidate was added to
    # the frontier.  Empty by default; populated at every accept site (seed,
    # mutation, merge) in evolution.py.
    num_metric_calls_by_discovery: dict[str, int] = field(default_factory=dict)
    # Persisted ``evolution.frontier_type`` (GEPA ``FrontierType`` parity
    # — ``src/gepa/core/state.py:22-23``).  Captured at evolve-time so
    # read-only CLI commands (``helix frontier``, ``helix best``,
    # ``helix log``) display the frontier with the SAME dimensionality
    # the evolution run actually used — regardless of what
    # ``helix.toml`` currently says.  Legacy states without the field
    # fall back to ``"instance"`` (HELIX's historical single-axis
    # default) in ``load_state``.
    frontier_type: FrontierType = "instance"
    # GEPA parity (audit-rng-state-persist D1): persisted schema version.
    # Mirrors GEPA core/state.py:182 / class-var :153.  Bumped when the
    # serialized schema changes; ``load_state`` migrates older payloads by
    # supplying defaults for any missing fields.
    schema_version: int = SCHEMA_VERSION


_STATE_FILENAME = "state.json"
_STATE_DIR = ".helix"
# GEPA parity (audit-rng-state-persist C1): companion pickle for the
# per-(candidate_hash, example_id) eval cache.  GEPA pickles the whole state
# dict, which round-trips its tuple-keyed ``EvaluationCache._cache`` for free
# (gepa/core/state.py:306-340).  HELIX persists state as JSON, which cannot
# encode tuple keys, so the cache lives in a sibling pickle alongside
# ``state.json``.  Loaded conditionally on ``config.evolution.cache_evaluation``.
_EVAL_CACHE_FILENAME = "eval_cache.pkl"


def _state_path(base_dir: Path) -> Path:
    return base_dir / _STATE_DIR / _STATE_FILENAME


def _eval_cache_path(base_dir: Path) -> Path:
    return base_dir / _STATE_DIR / _EVAL_CACHE_FILENAME


def save_state(state: EvolutionState, base_dir: Path) -> None:
    """Atomically write the evolution state to .helix/state.json."""
    target = _state_path(base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    data = {
        # GEPA parity (audit-rng-state-persist D1): schema_version is written
        # FIRST so a stripped/legacy state.json without it loads as v0 and
        # triggers the migration branch in ``load_state``.
        "schema_version": SCHEMA_VERSION,
        "generation": state.generation,
        "frontier": state.frontier,
        "instance_scores": state.instance_scores,
        "budget": asdict(state.budget),
        "config_hash": state.config_hash,
        "mutation_counter": state.mutation_counter,
        "merge_counter": state.merge_counter,
        "total_merge_invocations": state.total_merge_invocations,
        "merge_attempted_pairs": state.merge_attempted_pairs,
        "merge_description_triplets": state.merge_description_triplets,
        "i": state.i,
        "num_metric_calls_by_discovery": state.num_metric_calls_by_discovery,
        "frontier_type": state.frontier_type,
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

    # GEPA parity (audit-rng-state-persist D1): migrate older payloads.
    # GEPA's analogue is ``GEPAState._upgrade_state_dict`` (state.py:402-420):
    # supply defaults for any missing fields, then bump the version stamp.
    # HELIX treats a missing ``schema_version`` as v0 (the unversioned
    # predecessor) and falls through into the same default-fill path.
    version = data.get("schema_version", 0)
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"state.json schema_version {version} is newer than supported "
            f"version {SCHEMA_VERSION}; upgrade HELIX or use a different run dir."
        )

    budget_data = data.get("budget", {})
    budget = BudgetState(
        evaluations=budget_data.get("evaluations", 0),
    )

    # Migrate legacy frontier_type: default to "instance" (HELIX's
    # historical single-axis behaviour) for states written before the
    # field existed.  Narrow the str → FrontierType via a whitelist so
    # a corrupted state.json can't produce an invalid literal.
    raw_frontier_type = data.get("frontier_type", "instance")
    frontier_type: FrontierType = (
        raw_frontier_type
        if raw_frontier_type in ("instance", "objective", "hybrid", "cartesian")
        else "instance"
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
        merge_description_triplets=data.get("merge_description_triplets", []),
        i=data.get("i", -1),
        num_metric_calls_by_discovery=data.get("num_metric_calls_by_discovery", {}),
        frontier_type=frontier_type,
        schema_version=SCHEMA_VERSION,
    )


def save_eval_cache(cache_dict: dict[Any, Any], base_dir: Path) -> None:
    """Atomically pickle the per-(candidate, example) eval cache.

    GEPA parity (audit-rng-state-persist C1): mirrors the cache-survival
    behaviour of ``GEPAState.save`` at gepa/core/state.py:306-340.  HELIX
    uses JSON for ``state.json`` (which cannot round-trip tuple keys), so the
    cache is written to a sibling pickle.  Caller should pass
    ``MinibatchEvalCache._cache`` directly.  No-op semantics for an empty
    cache: the file is still written so that resume can reliably distinguish
    "cache disabled in last run" from "cache enabled but empty".
    """
    target = _eval_cache_path(base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(cache_dict, f)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_eval_cache(base_dir: Path) -> dict[Any, Any] | None:
    """Load the per-(candidate, example) eval cache, or None if absent.

    GEPA parity (audit-rng-state-persist C1): mirrors the cache-restore
    behaviour at gepa/core/state.py:348-376.  Returns the raw dict so the
    caller can install it on a freshly constructed cache instance (the
    caller decides whether caching is enabled — see ``initialize_gepa_state``
    at gepa/core/state.py:683-687 for the equivalent gating).
    """
    target = _eval_cache_path(base_dir)
    if not target.exists():
        return None
    with open(target, "rb") as f:
        loaded = pickle.load(f)
    if not isinstance(loaded, dict):
        raise ValueError(
            f"eval_cache.pkl at {target} did not contain a dict "
            f"(got {type(loaded).__name__})."
        )
    return loaded
