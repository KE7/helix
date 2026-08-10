"""HELIX evolution state persistence."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from helix.population import FrontierType


# GEPA parity: GEPA core/state.py:153 declares
# ``_VALIDATION_SCHEMA_VERSION: ClassVar[int] = 5``
# and migrates older state dicts on load (state.py:355-376).  HELIX previously
# had no schema version on ``state.json``; subsequent bumps mark explicit
# JSON-native schema additions (the unversioned predecessor is treated as
# v0; ``load_state`` migrates by default-filling missing fields).
SCHEMA_VERSION: int = 2

# Schema version for the per-(candidate, example) eval cache pickle.
# Bumped to 1 when ``CachedEvaluation`` gained a per-example ``side_info``
# slot — pre-extension caches dropped the LIBERO reflection feedstock on
# every hit, so a resume that silently kept them around would silently
# keep dropping it.  ``load_eval_cache`` quarantines any payload without
# this version (treated as schema 0) instead of loading it, so the next
# eval pass repopulates the cache with the new slot filled.
EVAL_CACHE_SCHEMA_VERSION: int = 1


@dataclass
class BudgetState:
    """Tracks resource consumption during evolution.

    Counts metric calls. Dataset/minibatch paths add the number of uncached
    examples evaluated; single-task/no-example paths add 0/1
    (cached=0, uncached evaluator call=1 because no per-example ids exist).
    """
    evaluations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    reasoning_tokens: int = 0
    cost_usd: float = 0.0


class EvaluationCache:
    """Simple evaluation cache keyed by (candidate_content_key, split).

    GEPA parity: avoids re-evaluating identical candidate content.  GEPA uses
    ``(candidate_hash, example_id)``; HELIX's no-example/single-task path has
    no example ids, so it uses the content key plus split.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, str], dict[str, Any]] = {}

    def get(self, candidate_key: str, split: str) -> dict[str, Any] | None:
        """Return cached result dict or None."""
        return self._cache.get((candidate_key, split))

    def put(self, candidate_key: str, split: str, result_dict: dict[str, Any]) -> None:
        """Store a result in the cache."""
        self._cache[(candidate_key, split)] = result_dict

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
    # circuit already-seen pairs.
    merge_attempted_pairs: list[list[str]] = field(default_factory=list)
    # GEPA parity: mirrors GEPA ``merges_performed[1]`` at
    # gepa/proposer/merge.py:195-203.
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
    # GEPA parity: per-program discovery budget.
    # GEPA tracks ``num_metric_calls_by_discovery: list[int]`` indexed by
    # program_idx (state.py:177, appended at state.py:537).  HELIX uses
    # candidate_id strings, so the dict keys by id and stores the value of
    # ``state.budget.evaluations`` at the moment the candidate was added to
    # the frontier.  Empty by default; populated at every accept site (seed,
    # mutation, merge) in evolution.py.
    num_metric_calls_by_discovery: dict[str, int] = field(default_factory=dict)
    # Active Pareto-front snapshot for the selected ``frontier_type``.
    # ``frontier`` remains HELIX's append-only candidate id list; this
    # separate JSON-native field makes the retained fronts visible without
    # conflating them with all evaluated candidates.
    active_frontier: dict[str, list[str]] = field(default_factory=dict)
    # Persisted ``evolution.frontier_type`` (GEPA ``FrontierType`` parity
    # — ``src/gepa/core/state.py:22-23``).  Captured at evolve-time so
    # read-only CLI commands (``helix frontier``, ``helix best``,
    # ``helix log``) display the frontier with the SAME dimensionality
    # the evolution run actually used — regardless of what
    # ``helix.toml`` currently says.  Legacy states without the field
    # fall back to ``"instance"`` (HELIX's historical single-axis
    # default) in ``load_state``.
    frontier_type: FrontierType = "instance"
    # Resume compatibility metadata for settings that affect optimization
    # semantics.  This is intentionally a small JSON-native dict rather than
    # a GEPA-style single pickled artifact: HELIX still persists worktrees,
    # evaluations, lineage, and state as separate artifacts.
    resume_semantics: dict[str, Any] = field(default_factory=dict)
    # GEPA parity: persisted schema version.
    # Mirrors GEPA core/state.py:182 / class-var :153.  Bumped when the
    # serialized schema changes; ``load_state`` migrates older payloads by
    # supplying defaults for any missing fields.
    schema_version: int = SCHEMA_VERSION


_STATE_FILENAME = "state.json"
_STATE_DIR = ".helix"
# GEPA parity: companion pickle for the
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
        # GEPA parity: schema_version is written FIRST so a
        # stripped/legacy state.json without it loads as v0 and
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
        "active_frontier": state.active_frontier,
        "frontier_type": state.frontier_type,
        "resume_semantics": state.resume_semantics,
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

    # GEPA parity: migrate older payloads.
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
        input_tokens=budget_data.get("input_tokens", 0),
        output_tokens=budget_data.get("output_tokens", 0),
        cached_input_tokens=budget_data.get("cached_input_tokens", 0),
        cache_creation_input_tokens=budget_data.get("cache_creation_input_tokens", 0),
        cache_read_input_tokens=budget_data.get("cache_read_input_tokens", 0),
        reasoning_tokens=budget_data.get("reasoning_tokens", 0),
        cost_usd=budget_data.get("cost_usd", 0.0),
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
        active_frontier=data.get("active_frontier", {}),
        frontier_type=frontier_type,
        resume_semantics=data.get("resume_semantics", {}),
        schema_version=SCHEMA_VERSION,
    )


def save_eval_cache(cache_dict: dict[Any, Any], base_dir: Path) -> None:
    """Atomically pickle the per-(candidate, example) eval cache.

    GEPA parity: mirrors the cache-survival behaviour of
    ``GEPAState.save`` at gepa/core/state.py:306-340.  HELIX
    uses JSON for ``state.json`` (which cannot round-trip tuple keys), so the
    cache is written to a sibling pickle.  Caller should pass
    ``MinibatchEvalCache._cache`` directly.  No-op semantics for an empty
    cache: the file is still written so that resume can reliably distinguish
    "cache disabled in last run" from "cache enabled but empty".

    The payload is a ``{"schema_version": int, "entries": cache_dict}``
    envelope rather than the bare cache dict — ``load_eval_cache`` rejects
    older shapes so a pre-extension cache (no ``CachedEvaluation.side_info``
    slot) is not silently revived.  See ``EVAL_CACHE_SCHEMA_VERSION``.
    """
    target = _eval_cache_path(base_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=target.parent, suffix=".tmp")
    payload: dict[str, Any] = {
        "schema_version": EVAL_CACHE_SCHEMA_VERSION,
        "entries": cache_dict,
    }
    try:
        with os.fdopen(fd, "wb") as f:
            pickle.dump(payload, f)
        os.replace(tmp_path, target)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_eval_cache(base_dir: Path) -> dict[Any, Any] | None:
    """Load the per-(candidate, example) eval cache, or None if absent.

    GEPA parity: mirrors the cache-restore behaviour at
    gepa/core/state.py:348-376.  Returns the raw entries dict so the
    caller can install it on a freshly constructed cache instance (the
    caller decides whether caching is enabled — see ``initialize_gepa_state``
    at gepa/core/state.py:683-687 for the equivalent gating).

    Schema check: payloads without the current ``schema_version`` are
    quarantined and treated as absent so the next eval pass repopulates
    the cache.  This is deliberate, not "backwards compatibility": a
    pre-extension cache (``CachedEvaluation`` without the ``side_info``
    slot) silently reproduces the very LIBERO-reflection regression the
    slot was added to fix, so silently keeping it on resume would
    persist the bug.  The old payload is preserved on disk under a
    timestamped suffix for diagnostics.
    """
    target = _eval_cache_path(base_dir)
    if not target.exists():
        return None
    try:
        with open(target, "rb") as f:
            loaded = pickle.load(f)
    except Exception as exc:
        quarantined = _quarantine_corrupt_cache(target, reason="unreadable")
        warnings.warn(
            f"Ignoring unreadable eval cache at {target}: "
            f"{type(exc).__name__}: {exc}. Quarantined to {quarantined}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    if not isinstance(loaded, dict):
        quarantined = _quarantine_corrupt_cache(target, reason="non-dict")
        warnings.warn(
            f"Ignoring eval cache at {target}: expected dict, got "
            f"{type(loaded).__name__}. Quarantined to {quarantined}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    # Schema-version guard.  A payload without ``schema_version`` is a
    # pre-extension cache (bare ``dict[CacheKey, CachedEvaluation]`` with
    # no ``side_info`` slot); quarantine it so the next eval pass refills
    # the new slot rather than silently losing the reflection feedstock.
    version = loaded.get("schema_version") if "schema_version" in loaded else None
    if version != EVAL_CACHE_SCHEMA_VERSION:
        _quarantine_corrupt_cache(
            target, reason=f"schema-v{version}-expected-v{EVAL_CACHE_SCHEMA_VERSION}"
        )
        warnings.warn(
            f"Ignoring eval cache: schema_version={version!r} does not match "
            f"current {EVAL_CACHE_SCHEMA_VERSION}. The next eval pass will "
            f"repopulate the cache with the current shape; a diagnostic copy "
            f"was retained.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    entries = loaded.get("entries")
    if not isinstance(entries, dict):
        quarantined = _quarantine_corrupt_cache(target, reason="malformed-envelope")
        warnings.warn(
            f"Ignoring eval cache at {target}: envelope missing 'entries' "
            f"dict (got {type(entries).__name__}). Quarantined to {quarantined}.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return entries


def _quarantine_corrupt_cache(target: Path, *, reason: str) -> Path:
    """Move a corrupt eval cache file aside so the next save doesn't overwrite it.

    Returns the destination path (or the original if rename failed — in which
    case the file is left in place; the caller's warning will still surface
    the underlying error).  We use a unique timestamped suffix so repeated
    failed loads don't collide.
    """
    suffix = f".corrupt-{reason}-{int(time.time() * 1000)}"
    dest = target.with_name(target.name + suffix)
    try:
        os.replace(target, dest)
        return dest
    except OSError:
        # Best-effort: if we can't rename (e.g. cross-device, perms), leave
        # the file in place.  The save path uses ``os.replace`` to overwrite
        # atomically, so a future successful save still wins.
        return target


def clear_eval_cache(base_dir: Path) -> None:
    """Remove the persisted per-example eval cache if present.

    Used by ``run_evolution`` when ``cache_evaluation`` is disabled to make
    sure a stale pickle from a prior cache-enabled run does not get
    revived later.  Idempotent: a missing target is a no-op.
    """
    target = _eval_cache_path(base_dir)
    try:
        target.unlink()
    except FileNotFoundError:
        return
