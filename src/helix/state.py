"""HELIX evolution state persistence."""

from __future__ import annotations

import json
import os
import pickle
import tempfile
import time
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from helix.population import FrontierType


# GEPA parity (audit-rng-state-persist D1):
# GEPA core/state.py:153 declares ``_VALIDATION_SCHEMA_VERSION: ClassVar[int] = 5``
# and migrates older state dicts on load (state.py:355-376).  HELIX previously
# had no schema version on ``state.json``; subsequent bumps mark explicit
# JSON-native schema additions (the unversioned predecessor is treated as
# v0; ``load_state`` migrates by default-filling missing fields).
SCHEMA_VERSION: int = 4


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

    # Authentication overhead, tracked SEPARATELY from proposal accounting.
    #
    # The once-per-run auth preflight makes a real, billable authenticated
    # request. It must never enter ``evaluations``: every lane inspector
    # expresses budget conservation purely in terms of that counter, and
    # livebench asserts a HARD EQUALITY
    # (``int(final_budget_after) == evaluations``), so ANY non-proposal
    # increment breaks all four lanes at once. There is no slack to absorb an
    # accidental charge — the margin is one integer.
    #
    # Folding this into ``cost_usd`` / token counters would not break today's
    # inspectors (none assert on them), but it would silently misattribute
    # auth overhead to proposal cost — a quiet instance of exactly the
    # silent-flip class this release removes. Keep it separate and visible.
    auth_overhead_calls: int = 0
    auth_overhead_input_tokens: int = 0
    auth_overhead_output_tokens: int = 0
    auth_overhead_cost_usd: float = 0.0


ProposalTaskStatus = Literal[
    "planned",
    "running",
    "evaluated",
    "skipped",
    "failed",
    "tampered",
    "rejected",
    "applied",
    "interrupted",
]
ProposalSelectionResult = Literal[
    "pending", "not_applicable", "not_selected", "selected"
]
ProposalCleanupResult = Literal[
    "pending", "not_required", "removed", "missing", "failed"
]
ProposalBatchPhase = Literal[
    "planned", "dispatched", "applying", "complete", "interrupted"
]

TERMINAL_PROPOSAL_STATUSES: frozenset[ProposalTaskStatus] = frozenset(
    {"skipped", "failed", "tampered", "rejected", "applied", "interrupted"}
)
TERMINAL_CLEANUP_RESULTS: frozenset[ProposalCleanupResult] = frozenset(
    {"not_required", "removed", "missing", "failed"}
)

_PROPOSAL_TASK_STATUSES = frozenset(
    {
        "planned",
        "running",
        "evaluated",
        "skipped",
        "failed",
        "tampered",
        "rejected",
        "applied",
        "interrupted",
    }
)
_PROPOSAL_SELECTION_RESULTS = frozenset(
    {"pending", "not_applicable", "not_selected", "selected"}
)
_PROPOSAL_CLEANUP_RESULTS = frozenset(
    {"pending", "not_required", "removed", "missing", "failed"}
)
_PROPOSAL_BATCH_PHASES = frozenset(
    {"planned", "dispatched", "applying", "complete", "interrupted"}
)


def _budget_state_from_mapping(data: Mapping[str, Any]) -> BudgetState:
    return BudgetState(
        evaluations=int(data.get("evaluations", 0)),
        input_tokens=int(data.get("input_tokens", 0)),
        output_tokens=int(data.get("output_tokens", 0)),
        cached_input_tokens=int(data.get("cached_input_tokens", 0)),
        cache_creation_input_tokens=int(data.get("cache_creation_input_tokens", 0)),
        cache_read_input_tokens=int(data.get("cache_read_input_tokens", 0)),
        reasoning_tokens=int(data.get("reasoning_tokens", 0)),
        cost_usd=float(data.get("cost_usd", 0.0)),
        auth_overhead_calls=int(data.get("auth_overhead_calls", 0)),
        auth_overhead_input_tokens=int(data.get("auth_overhead_input_tokens", 0)),
        auth_overhead_output_tokens=int(data.get("auth_overhead_output_tokens", 0)),
        auth_overhead_cost_usd=float(data.get("auth_overhead_cost_usd", 0.0)),
    )


@dataclass
class ProposalTaskRecord:
    """Durable state for one parent-major P-by-N proposal slot.

    ``budget_charge`` is a monotonic runtime journal and may advance while the
    task is still running, including evaluator work whose outcome is a failure;
    ``budget_accounted`` closes that journal as an explicit crash barrier.  A
    resumed run can therefore conserve attempted evaluator work before terminal
    accounting without charging it again, and ``applied`` distinguishes a
    selected result from one already inserted into the frontier.  Candidate IDs
    remain reserved even when the task is interrupted or cleaned up.
    """

    batch_id: str
    p: int
    n: int
    task_index: int
    parent_group: int
    mutation_index: int
    parent_id: str
    child_id: str
    status: ProposalTaskStatus = "planned"
    score_delta: float | None = None
    selection: ProposalSelectionResult = "pending"
    cleanup: ProposalCleanupResult = "pending"
    budget_charge: BudgetState = field(default_factory=BudgetState)
    budget_accounted: bool = False
    applied: bool = False
    detail: str | None = None

    def is_terminal(self) -> bool:
        """Return whether both execution and cleanup reached a terminal state."""
        return (
            self.status in TERMINAL_PROPOSAL_STATUSES
            and self.cleanup in TERMINAL_CLEANUP_RESULTS
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "p": self.p,
            "n": self.n,
            "task_index": self.task_index,
            "parent_group": self.parent_group,
            "mutation_index": self.mutation_index,
            "parent_id": self.parent_id,
            "child_id": self.child_id,
            "status": self.status,
            "score_delta": self.score_delta,
            "selection": self.selection,
            "cleanup": self.cleanup,
            "budget_charge": asdict(self.budget_charge),
            "budget_accounted": self.budget_accounted,
            "applied": self.applied,
            "detail": self.detail,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProposalTaskRecord:
        raw_status = str(data.get("status", "planned"))
        raw_selection = str(data.get("selection", "pending"))
        raw_cleanup = str(data.get("cleanup", "pending"))
        if raw_status not in _PROPOSAL_TASK_STATUSES:
            raise ValueError(f"Invalid proposal task status: {raw_status!r}")
        if raw_selection not in _PROPOSAL_SELECTION_RESULTS:
            raise ValueError(f"Invalid proposal selection result: {raw_selection!r}")
        if raw_cleanup not in _PROPOSAL_CLEANUP_RESULTS:
            raise ValueError(f"Invalid proposal cleanup result: {raw_cleanup!r}")
        raw_budget = data.get("budget_charge", {})
        budget_mapping = raw_budget if isinstance(raw_budget, Mapping) else {}
        raw_delta = data.get("score_delta")
        return cls(
            batch_id=str(data["batch_id"]),
            p=int(data["p"]),
            n=int(data["n"]),
            task_index=int(data["task_index"]),
            parent_group=int(data["parent_group"]),
            mutation_index=int(data["mutation_index"]),
            parent_id=str(data["parent_id"]),
            child_id=str(data["child_id"]),
            status=cast(ProposalTaskStatus, raw_status),
            score_delta=None if raw_delta is None else float(raw_delta),
            selection=cast(ProposalSelectionResult, raw_selection),
            cleanup=cast(ProposalCleanupResult, raw_cleanup),
            budget_charge=_budget_state_from_mapping(budget_mapping),
            budget_accounted=bool(data.get("budget_accounted", False)),
            applied=bool(data.get("applied", False)),
            detail=None if data.get("detail") is None else str(data["detail"]),
        )


@dataclass
class ProposalBatchRecord:
    """Durable pre-dispatch plan and deterministic post-apply checkpoint."""

    batch_id: str
    generation: int
    p: int
    n: int
    tasks: list[ProposalTaskRecord]
    phase: ProposalBatchPhase = "planned"
    budget_before_dispatch: int = 0
    # Full resource snapshot paired with ``budget_before_dispatch``.  The
    # integer field remains for schema/backward compatibility and the explicit
    # overshoot contract; this snapshot lets resume conserve token/cost usage
    # when a process stopped after a worker returned but before its task row was
    # terminalized.
    budget_state_before_dispatch: BudgetState | None = None
    max_evaluations: int = 0
    max_in_flight_evaluations: int = 0
    maximum_overshoot: int = 0
    budget_after_apply: int | None = None

    def validate_plan(self) -> None:
        """Validate the exact parent-major P-by-N shape and unique IDs."""
        if not self.batch_id:
            raise ValueError("Proposal batch_id cannot be empty")
        if self.p < 1 or self.n < 1:
            raise ValueError("Proposal batch P and N must both be at least one")
        expected_count = self.p * self.n
        if len(self.tasks) != expected_count:
            raise ValueError(
                f"Proposal batch {self.batch_id!r} has {len(self.tasks)} tasks; "
                f"expected exactly P*N={expected_count}"
            )
        child_ids: set[str] = set()
        group_parents: dict[int, str] = {}
        for index, task in enumerate(self.tasks):
            expected_group, expected_mutation = divmod(index, self.n)
            if task.batch_id != self.batch_id or task.p != self.p or task.n != self.n:
                raise ValueError(f"Task {index} does not match its proposal batch")
            if (
                task.task_index != index
                or task.parent_group != expected_group
                or task.mutation_index != expected_mutation
            ):
                raise ValueError(
                    f"Task {index} is not in parent-major P-by-N order "
                    f"(expected group={expected_group}, mutation={expected_mutation})"
                )
            if not task.child_id:
                raise ValueError(f"Task {index} has an empty child_id")
            if task.child_id in child_ids:
                raise ValueError(f"Duplicate planned child_id: {task.child_id}")
            child_ids.add(task.child_id)
            previous_parent = group_parents.setdefault(
                task.parent_group, task.parent_id
            )
            if previous_parent != task.parent_id:
                raise ValueError(
                    f"Parent group {task.parent_group} contains different parents"
                )

    def all_tasks_terminal(self) -> bool:
        return all(task.is_terminal() for task in self.tasks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "generation": self.generation,
            "p": self.p,
            "n": self.n,
            "phase": self.phase,
            "budget_before_dispatch": self.budget_before_dispatch,
            "budget_state_before_dispatch": (
                asdict(self.budget_state_before_dispatch)
                if self.budget_state_before_dispatch is not None
                else None
            ),
            "max_evaluations": self.max_evaluations,
            "max_in_flight_evaluations": self.max_in_flight_evaluations,
            "maximum_overshoot": self.maximum_overshoot,
            "budget_after_apply": self.budget_after_apply,
            "tasks": [task.to_dict() for task in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProposalBatchRecord:
        raw_phase = str(data.get("phase", "planned"))
        if raw_phase not in _PROPOSAL_BATCH_PHASES:
            raise ValueError(f"Invalid proposal batch phase: {raw_phase!r}")
        raw_tasks = data.get("tasks", [])
        if not isinstance(raw_tasks, list):
            raise ValueError("Proposal batch tasks must be a list")
        tasks: list[ProposalTaskRecord] = []
        for raw_task in raw_tasks:
            if not isinstance(raw_task, Mapping):
                raise ValueError("Proposal batch task must be an object")
            tasks.append(ProposalTaskRecord.from_dict(raw_task))
        raw_after = data.get("budget_after_apply")
        raw_budget_before = data.get("budget_state_before_dispatch")
        budget_before_mapping = (
            raw_budget_before if isinstance(raw_budget_before, Mapping) else None
        )
        batch = cls(
            batch_id=str(data["batch_id"]),
            generation=int(data["generation"]),
            p=int(data["p"]),
            n=int(data["n"]),
            tasks=tasks,
            phase=cast(ProposalBatchPhase, raw_phase),
            budget_before_dispatch=int(data.get("budget_before_dispatch", 0)),
            budget_state_before_dispatch=(
                _budget_state_from_mapping(budget_before_mapping)
                if budget_before_mapping is not None
                else None
            ),
            max_evaluations=int(data.get("max_evaluations", 0)),
            max_in_flight_evaluations=int(data.get("max_in_flight_evaluations", 0)),
            maximum_overshoot=int(data.get("maximum_overshoot", 0)),
            budget_after_apply=None if raw_after is None else int(raw_after),
        )
        batch.validate_plan()
        return batch


@dataclass(frozen=True)
class BatchDispatchDecision:
    """Pre-dispatch budget decision and the batch's explicit overshoot bound."""

    allowed: bool
    evaluations_before: int
    max_evaluations: int
    max_in_flight_evaluations: int
    maximum_overshoot: int


@dataclass(frozen=True)
class BatchReconciliation:
    """Audit summary produced while terminalizing an interrupted batch."""

    batch_id: str
    reserved_child_ids: tuple[str, ...]
    applied_child_ids: tuple[str, ...]
    accounted_child_ids: tuple[str, ...]
    cleaned_child_ids: tuple[str, ...]
    missing_child_ids: tuple[str, ...]
    cleanup_failed_child_ids: tuple[str, ...]


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
    instance_scores: dict[
        str, Any
    ]  # dict[str, dict[str, float]] — candidate_id -> instance -> score
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
    # Step 7 parallel-proposal ledger.  Every planned P*N child id remains
    # here permanently, including skipped and interrupted slots, so resume
    # never rewinds into an already-issued id or infers completion from only
    # the first worker/artifact in a batch.
    proposal_batches: list[ProposalBatchRecord] = field(default_factory=list)
    # JSON-safe runtime scheduler checkpoint.  Kept separate from
    # ``resume_semantics`` because it is evolving execution position, not a
    # compatibility knob.  Legacy states default to an empty mapping.
    scheduler_state: dict[str, Any] = field(default_factory=dict)
    # GEPA parity (audit-rng-state-persist D1): persisted schema version.
    # Mirrors GEPA core/state.py:182 / class-var :153.  Bumped when the
    # serialized schema changes; ``load_state`` migrates older payloads by
    # supplying defaults for any missing fields.
    schema_version: int = SCHEMA_VERSION


CheckpointSaver = Callable[[EvolutionState], None]


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


def _persist_checkpoint(
    state: EvolutionState,
    base_dir: Path,
    saver: CheckpointSaver | None,
) -> None:
    """Persist through evolution's cache-aware wrapper when one is supplied."""
    if saver is None:
        save_state(state, base_dir)
    else:
        saver(state)


def _to_json_safe(value: Any) -> Any:
    """Normalize tuples/mappings into JSON-native scheduler state."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("Scheduler state mapping keys must be strings")
            normalized[key] = _to_json_safe(item)
        return normalized
    raise TypeError(f"Scheduler state value is not JSON-safe: {type(value).__name__}")


def encode_rng_state(rng_state: object) -> list[Any]:
    """Encode ``random.Random.getstate()`` output for ``state.json``."""
    normalized = _to_json_safe(rng_state)
    if not isinstance(normalized, list):
        raise TypeError("RNG state must be tuple/list shaped")
    return normalized


def _lists_to_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _lists_to_tuples(item) for key, item in value.items()}
    return value


def decode_rng_state(encoded: Sequence[Any]) -> tuple[Any, ...]:
    """Reconstruct the tuple shape required by ``random.Random.setstate``."""
    restored = _lists_to_tuples(list(encoded))
    if not isinstance(restored, tuple):
        raise TypeError("Encoded RNG state must restore to a tuple")
    return restored


def build_scheduler_checkpoint(
    *,
    frontier_rng_state: object,
    sampler_rng_state: object,
    sampler_epoch: int,
    sampler_shuffled_ids: Sequence[Any],
    sampler_last_trainset_size: int,
    sampler_id_frequencies: Mapping[str, int] | None = None,
    sampler_fallback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON-native frontier/sampler checkpoint runtime restores.

    ``sampler_fallback`` can contain the inner epoch sampler's equivalent
    fields when stratified sampling delegated to its fallback path.
    """
    if sampler_epoch < -1:
        raise ValueError("sampler_epoch cannot be less than -1")
    if sampler_last_trainset_size < 0:
        raise ValueError("sampler_last_trainset_size cannot be negative")
    checkpoint = {
        "frontier_rng_state": encode_rng_state(frontier_rng_state),
        "sampler": {
            "rng_state": encode_rng_state(sampler_rng_state),
            "epoch": sampler_epoch,
            "shuffled_ids": list(sampler_shuffled_ids),
            "last_trainset_size": sampler_last_trainset_size,
            "id_frequencies": dict(sampler_id_frequencies or {}),
            "fallback": dict(sampler_fallback or {}),
        },
    }
    normalized = _to_json_safe(checkpoint)
    assert isinstance(normalized, dict)
    return normalized


def checkpoint_scheduler_state(
    state: EvolutionState,
    base_dir: Path,
    scheduler_state: Mapping[str, Any],
    *,
    saver: CheckpointSaver | None = None,
) -> None:
    """Persist current RNG/sampler position through the cache-aware saver."""
    normalized = _to_json_safe(scheduler_state)
    if not isinstance(normalized, dict):
        raise TypeError("scheduler_state must be an object")
    state.scheduler_state = normalized
    _persist_checkpoint(state, base_dir, saver)


def check_batch_dispatch_budget(
    state: EvolutionState,
    *,
    max_evaluations: int,
    max_in_flight_evaluations: int,
) -> BatchDispatchDecision:
    """Check the budget once before dispatch and state the overshoot bound.

    A batch may start only while the current evaluation count is below a
    positive cap.  Once started, every completed in-flight call is accounted,
    even if that crosses the cap.  Consequently the permitted overshoot is
    bounded by ``max(0, before + max_in_flight_evaluations - cap)``.  A
    non-positive cap is unlimited and reports no finite overshoot.
    """
    if max_in_flight_evaluations < 0:
        raise ValueError("max_in_flight_evaluations cannot be negative")
    before = state.budget.evaluations
    allowed = max_evaluations <= 0 or before < max_evaluations
    overshoot = (
        0
        if max_evaluations <= 0
        else max(0, before + max_in_flight_evaluations - max_evaluations)
    )
    return BatchDispatchDecision(
        allowed=allowed,
        evaluations_before=before,
        max_evaluations=max_evaluations,
        max_in_flight_evaluations=max_in_flight_evaluations,
        maximum_overshoot=overshoot,
    )


def get_proposal_batch(
    state: EvolutionState, batch_id: str
) -> ProposalBatchRecord | None:
    """Return one persisted batch by id, or ``None`` when it is new."""
    return next(
        (batch for batch in state.proposal_batches if batch.batch_id == batch_id),
        None,
    )


def reserved_candidate_ids(state: EvolutionState) -> set[str]:
    """Return frontier and planned child IDs that may never be issued again."""
    reserved = set(state.frontier)
    reserved.update(
        task.child_id for batch in state.proposal_batches for task in batch.tasks
    )
    return reserved


def _batch_plan_signature(batch: ProposalBatchRecord) -> tuple[Any, ...]:
    return (
        batch.batch_id,
        batch.generation,
        batch.p,
        batch.n,
        tuple(
            (
                task.task_index,
                task.parent_group,
                task.mutation_index,
                task.parent_id,
                task.child_id,
            )
            for task in batch.tasks
        ),
    )


def checkpoint_batch_before_dispatch(
    state: EvolutionState,
    base_dir: Path,
    batch: ProposalBatchRecord,
    *,
    max_evaluations: int = 0,
    max_in_flight_evaluations: int = 0,
    saver: CheckpointSaver | None = None,
) -> ProposalBatchRecord:
    """Persist a complete P*N plan before any worker is submitted.

    Repeating the call with the same batch plan is idempotent.  Reusing a
    batch id for a different plan, or any child id reserved by another batch,
    is rejected before state is changed.
    """
    batch.validate_plan()
    existing = get_proposal_batch(state, batch.batch_id)
    if existing is not None:
        if _batch_plan_signature(existing) != _batch_plan_signature(batch):
            raise ValueError(f"Proposal batch id collision: {batch.batch_id}")
        return existing

    decision = check_batch_dispatch_budget(
        state,
        max_evaluations=max_evaluations,
        max_in_flight_evaluations=max_in_flight_evaluations,
    )
    if not decision.allowed:
        raise ValueError(
            f"Cannot dispatch proposal batch {batch.batch_id!r}: evaluation "
            f"budget {decision.evaluations_before}/{decision.max_evaluations} "
            "is exhausted"
        )

    already_reserved = reserved_candidate_ids(state)
    collisions = sorted(
        task.child_id for task in batch.tasks if task.child_id in already_reserved
    )
    if collisions:
        raise ValueError(
            "Planned proposal child IDs are already reserved: " + ", ".join(collisions)
        )

    batch.phase = "dispatched"
    batch.budget_before_dispatch = decision.evaluations_before
    batch.budget_state_before_dispatch = replace(state.budget)
    batch.max_evaluations = decision.max_evaluations
    batch.max_in_flight_evaluations = decision.max_in_flight_evaluations
    batch.maximum_overshoot = decision.maximum_overshoot
    state.proposal_batches.append(batch)
    _persist_checkpoint(state, base_dir, saver)
    return batch


_ALLOWED_TASK_TRANSITIONS: dict[ProposalTaskStatus, frozenset[ProposalTaskStatus]] = {
    "planned": frozenset(
        {
            "planned",
            "running",
            "evaluated",
            "skipped",
            "failed",
            "tampered",
            "rejected",
            "interrupted",
        }
    ),
    "running": frozenset(
        {
            "running",
            "evaluated",
            "skipped",
            "failed",
            "tampered",
            "rejected",
            "interrupted",
        }
    ),
    "evaluated": frozenset({"evaluated", "rejected", "applied", "interrupted"}),
    "skipped": frozenset({"skipped"}),
    "failed": frozenset({"failed"}),
    "tampered": frozenset({"tampered"}),
    "rejected": frozenset({"rejected"}),
    "applied": frozenset({"applied"}),
    "interrupted": frozenset({"interrupted"}),
}


def _task_by_index(batch: ProposalBatchRecord, task_index: int) -> ProposalTaskRecord:
    if task_index < 0 or task_index >= len(batch.tasks):
        raise IndexError(
            f"Proposal task index {task_index} is outside batch {batch.batch_id!r}"
        )
    return batch.tasks[task_index]


def checkpoint_batch_task(
    state: EvolutionState,
    base_dir: Path,
    *,
    batch_id: str,
    task_index: int,
    status: ProposalTaskStatus | None = None,
    score_delta: float | None = None,
    selection: ProposalSelectionResult | None = None,
    cleanup: ProposalCleanupResult | None = None,
    budget_charge: BudgetState | None = None,
    budget_accounted: bool | None = None,
    applied: bool | None = None,
    detail: str | None = None,
    expected_status: ProposalTaskStatus | None = None,
    saver: CheckpointSaver | None = None,
) -> ProposalTaskRecord:
    """Atomically checkpoint one ordered result/apply transition.

    Evolution remains the authoritative charger: it journals drained runtime
    results into ``budget_charge`` before a sibling checkpoint can expose their
    cache entries, then passes the computed delta here with
    ``budget_accounted=True`` at the terminal barrier.  This function never
    mutates the global budget.  Retrying an identical accounted marker is a
    no-op; retrying with a different charge is rejected.
    """
    batch = get_proposal_batch(state, batch_id)
    if batch is None:
        raise KeyError(f"Unknown proposal batch: {batch_id}")
    task = _task_by_index(batch, task_index)
    if expected_status is not None and task.status != expected_status:
        raise ValueError(
            f"Proposal task {batch_id}/{task_index} is {task.status!r}; "
            f"expected {expected_status!r}"
        )
    next_status = task.status if status is None else status
    if next_status not in _ALLOWED_TASK_TRANSITIONS[task.status]:
        raise ValueError(
            f"Invalid proposal task transition: {task.status!r} -> {next_status!r}"
        )
    if applied is True and next_status != "applied":
        raise ValueError("A task can be marked applied only with status='applied'")
    if next_status == "applied" and selection not in (None, "selected"):
        raise ValueError("An applied proposal task must have selection='selected'")

    if budget_charge is not None:
        if task.budget_accounted and task.budget_charge != budget_charge:
            raise ValueError(
                f"Proposal task {batch_id}/{task_index} was already charged "
                "with a different budget delta"
            )
        task.budget_charge = budget_charge
    if budget_accounted is False and task.budget_accounted:
        raise ValueError(
            f"Proposal task {batch_id}/{task_index} is already budget-accounted"
        )
    if budget_accounted is True:
        task.budget_accounted = True

    task.status = next_status
    if score_delta is not None:
        task.score_delta = score_delta
    if selection is not None:
        task.selection = selection
    if cleanup is not None:
        task.cleanup = cleanup
    if applied is not None:
        task.applied = applied
    if next_status == "applied":
        task.selection = "selected"
        task.cleanup = "not_required" if cleanup is None else cleanup
        task.applied = True
    elif next_status in {"skipped", "failed", "tampered"} and selection is None:
        task.selection = "not_applicable"
    elif next_status == "rejected" and selection is None:
        task.selection = "not_selected"
    if detail is not None:
        task.detail = detail
    if batch.phase == "dispatched" and next_status not in {"planned", "running"}:
        batch.phase = "applying"
    _persist_checkpoint(state, base_dir, saver)
    return task


def checkpoint_batch_after_apply(
    state: EvolutionState,
    base_dir: Path,
    *,
    batch_id: str,
    saver: CheckpointSaver | None = None,
) -> ProposalBatchRecord:
    """Persist the post-apply barrier after every planned slot is terminal."""
    batch = get_proposal_batch(state, batch_id)
    if batch is None:
        raise KeyError(f"Unknown proposal batch: {batch_id}")
    nonterminal = [task.task_index for task in batch.tasks if not task.is_terminal()]
    if nonterminal:
        raise ValueError(
            f"Cannot complete proposal batch {batch_id!r}; nonterminal slots: "
            + ", ".join(str(index) for index in nonterminal)
        )
    cleanup_failed = [
        task.task_index for task in batch.tasks if task.cleanup == "failed"
    ]
    if cleanup_failed:
        raise ValueError(
            f"Cannot complete proposal batch {batch_id!r}; cleanup failed for slots: "
            + ", ".join(str(index) for index in cleanup_failed)
        )
    unaccounted = [task.task_index for task in batch.tasks if not task.budget_accounted]
    if unaccounted:
        raise ValueError(
            f"Cannot complete proposal batch {batch_id!r}; unaccounted slots: "
            + ", ".join(str(index) for index in unaccounted)
        )
    batch_index = next(
        index
        for index, candidate_batch in enumerate(state.proposal_batches)
        if candidate_batch is batch
    )
    next_batch = (
        state.proposal_batches[batch_index + 1]
        if batch_index + 1 < len(state.proposal_batches)
        else None
    )
    evaluations_at_end = (
        next_batch.budget_before_dispatch
        if next_batch is not None
        else state.budget.evaluations
    )
    actual_in_flight = evaluations_at_end - batch.budget_before_dispatch
    if actual_in_flight < 0:
        raise ValueError(
            f"Proposal batch {batch_id!r} ended below its pre-dispatch budget"
        )
    recorded_in_flight = sum(task.budget_charge.evaluations for task in batch.tasks)
    if recorded_in_flight != actual_in_flight:
        raise ValueError(
            f"Proposal batch {batch_id!r} recorded {recorded_in_flight} "
            f"evaluation(s) across its tasks, but the global budget advanced "
            f"by {actual_in_flight}"
        )
    if (
        batch.max_in_flight_evaluations > 0
        and actual_in_flight > batch.max_in_flight_evaluations
    ):
        raise ValueError(
            f"Proposal batch {batch_id!r} accounted {actual_in_flight} evaluations; "
            f"declared in-flight bound is {batch.max_in_flight_evaluations}"
        )
    actual_overshoot = (
        0
        if batch.max_evaluations <= 0
        else max(0, evaluations_at_end - batch.max_evaluations)
    )
    if actual_overshoot > batch.maximum_overshoot:
        raise ValueError(
            f"Proposal batch {batch_id!r} overshot by {actual_overshoot}; "
            f"checkpoint bound is {batch.maximum_overshoot}"
        )
    batch.phase = "complete"
    batch.budget_after_apply = evaluations_at_end
    _persist_checkpoint(state, base_dir, saver)
    return batch


def reconcile_interrupted_batches(
    state: EvolutionState,
    base_dir: Path,
    *,
    worktrees_dir: Path,
    cleanup_worktree: Callable[[str, Path], bool],
    saver: CheckpointSaver | None = None,
) -> list[BatchReconciliation]:
    """Terminalize every slot in every interrupted persisted batch.

    Applied/frontier children are preserved and marked applied, preventing a
    second frontier insertion.  Every other planned worktree is passed to the
    caller's Git-aware cleanup callback.  The authoritative global budget is
    never changed; once cleanup succeeds, any durable delta not yet present in
    a task row is attributed deterministically so the batch ledger conserves
    that total.  All child IDs remain reserved through the ledger.
    """
    reconciled: list[BatchReconciliation] = []
    changed = False
    for batch_index, batch in enumerate(state.proposal_batches):
        next_batch = (
            state.proposal_batches[batch_index + 1]
            if batch_index + 1 < len(state.proposal_batches)
            else None
        )
        evaluations_at_end = (
            next_batch.budget_before_dispatch
            if next_batch is not None
            else state.budget.evaluations
        )
        retry_failed_cleanup = any(
            task.cleanup == "failed" and not task.applied for task in batch.tasks
        )
        if batch.phase == "complete" or (
            batch.phase == "interrupted" and not retry_failed_cleanup
        ):
            continue
        reserved_ids: list[str] = []
        applied_ids: list[str] = []
        accounted_ids: list[str] = []
        cleaned_ids: list[str] = []
        missing_ids: list[str] = []
        failed_ids: list[str] = []
        for task in batch.tasks:
            reserved_ids.append(task.child_id)

            if (
                task.applied
                or task.status == "applied"
                or task.child_id in state.frontier
            ):
                task.status = "applied"
                task.selection = "selected"
                task.cleanup = "not_required"
                task.applied = True
                applied_ids.append(task.child_id)
                continue

            worktree_path = worktrees_dir / task.child_id
            if not worktree_path.exists() and task.cleanup in {"removed", "missing"}:
                if task.cleanup == "removed":
                    cleaned_ids.append(task.child_id)
                else:
                    missing_ids.append(task.child_id)
                continue
            if worktree_path.exists():
                try:
                    removed = cleanup_worktree(task.child_id, worktree_path)
                except Exception as exc:
                    removed = False
                    task.detail = f"Interrupted-batch cleanup failed: {exc}"
                if removed and not worktree_path.exists():
                    task.cleanup = "removed"
                    cleaned_ids.append(task.child_id)
                else:
                    task.cleanup = "failed"
                    failed_ids.append(task.child_id)
            else:
                task.cleanup = "missing"
                missing_ids.append(task.child_id)
            task.status = "interrupted"
            if task.selection == "pending":
                task.selection = "not_applicable"
            task.applied = False

        # Cleanup is a resume barrier: only after every non-applied worktree is
        # gone may an interrupted batch be declared fully accounted.  A worker
        # can finish and update the global durable budget immediately before a
        # crash prevents its task row from being terminalized.  Preserve that
        # authoritative global total by assigning the unjournaled residual to
        # the first unaccounted slot in stable task order, then close every
        # remaining zero-residual slot.  This never charges the global budget a
        # second time and is idempotent on later resumes.
        if not failed_ids:
            unaccounted = [task for task in batch.tasks if not task.budget_accounted]
            if unaccounted:
                before = batch.budget_state_before_dispatch
                has_full_budget_baseline = before is not None
                if before is None:
                    # Legacy ledgers only persisted the evaluation baseline.
                    # Conserve that field exactly without guessing historical
                    # token/cost baselines that were never recorded.
                    before = BudgetState(evaluations=batch.budget_before_dispatch)

                durable_end = (
                    next_batch.budget_state_before_dispatch
                    if next_batch is not None
                    else state.budget
                )
                has_full_budget_end = durable_end is not None
                if durable_end is None:
                    # A legacy next batch still gives us an exact evaluation
                    # boundary, but never persisted token/cost counters.
                    assert next_batch is not None
                    durable_end = BudgetState(
                        evaluations=next_batch.budget_before_dispatch
                    )

                fields = (
                    (
                        "evaluations",
                        "input_tokens",
                        "output_tokens",
                        "cached_input_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                        "reasoning_tokens",
                        "cost_usd",
                    )
                    if has_full_budget_baseline and has_full_budget_end
                    else ("evaluations",)
                )
                residuals: dict[str, int | float] = {}
                for field_name in fields:
                    durable_total = getattr(durable_end, field_name)
                    baseline = getattr(before, field_name)
                    recorded = sum(
                        getattr(task.budget_charge, field_name) for task in batch.tasks
                    )
                    residual = durable_total - baseline - recorded
                    tolerance = 1e-12 if field_name == "cost_usd" else 0
                    if residual < -tolerance:
                        raise ValueError(
                            f"Interrupted proposal batch {batch.batch_id!r} "
                            f"over-recorded {field_name}: baseline={baseline}, "
                            f"recorded={recorded}, durable_total={durable_total}"
                        )
                    residuals[field_name] = (
                        0 if abs(residual) <= tolerance else residual
                    )

                residual_task = unaccounted[0]
                for field_name, residual in residuals.items():
                    setattr(
                        residual_task.budget_charge,
                        field_name,
                        getattr(residual_task.budget_charge, field_name) + residual,
                    )
                for task in unaccounted:
                    task.budget_accounted = True
                    suffix = "resume cleanup and accounting reconciled"
                    task.detail = f"{task.detail}; {suffix}" if task.detail else suffix

        accounted_ids.extend(
            task.child_id for task in batch.tasks if task.budget_accounted
        )

        batch.phase = "interrupted"
        batch.budget_after_apply = evaluations_at_end
        reconciled.append(
            BatchReconciliation(
                batch_id=batch.batch_id,
                reserved_child_ids=tuple(reserved_ids),
                applied_child_ids=tuple(applied_ids),
                accounted_child_ids=tuple(accounted_ids),
                cleaned_child_ids=tuple(cleaned_ids),
                missing_child_ids=tuple(missing_ids),
                cleanup_failed_child_ids=tuple(failed_ids),
            )
        )
        changed = True
    if changed:
        _persist_checkpoint(state, base_dir, saver)
    return reconciled


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
        "active_frontier": state.active_frontier,
        "frontier_type": state.frontier_type,
        "resume_semantics": state.resume_semantics,
        "proposal_batches": [batch.to_dict() for batch in state.proposal_batches],
        "scheduler_state": _to_json_safe(state.scheduler_state),
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

    raw_budget = data.get("budget", {})
    budget_data = raw_budget if isinstance(raw_budget, Mapping) else {}
    budget = _budget_state_from_mapping(budget_data)

    raw_batches = data.get("proposal_batches", [])
    if not isinstance(raw_batches, list):
        raise ValueError("state.json proposal_batches must be a list")
    proposal_batches: list[ProposalBatchRecord] = []
    for raw_batch in raw_batches:
        if not isinstance(raw_batch, Mapping):
            raise ValueError("state.json proposal batch must be an object")
        proposal_batches.append(ProposalBatchRecord.from_dict(raw_batch))
    raw_scheduler_state = data.get("scheduler_state", {})
    if not isinstance(raw_scheduler_state, Mapping):
        raise ValueError("state.json scheduler_state must be an object")
    normalized_scheduler_state = _to_json_safe(raw_scheduler_state)
    assert isinstance(normalized_scheduler_state, dict)

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
        proposal_batches=proposal_batches,
        scheduler_state=normalized_scheduler_state,
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
    return loaded


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
