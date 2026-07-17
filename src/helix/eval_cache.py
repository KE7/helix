"""HELIX evaluation cache — GEPA parity with concurrent single-flight.

The cache data model follows the layer described in /tmp/gepa_eval_spec.md §3
(originally at gepa/core/state.py:27-130).  HELIX additionally coordinates
concurrent cache misses per candidate-content/example key so parallel batches
cannot evaluate the same missing example twice.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, TypeAlias, TypeVar

from helix.trace import TRACE, EventType

CandidateHash: TypeAlias = str
DataId = TypeVar("DataId")
RolloutOutput = TypeVar("RolloutOutput")
CacheKey: TypeAlias = tuple[CandidateHash, Any]  # (hash, example_id)


@dataclass(frozen=True, slots=True)
class EvaluationBatchKey:
    """Identity of one evaluator request for cross-candidate deduplication.

    Candidate lineage ids are deliberately absent.  Two requests may share
    work only when their evaluation-relevant content, dataset split, and exact
    ordered minibatch are all identical.  Minibatch order is significant
    because evaluator side information is positional to ``helix_batch.json``.
    """

    content_key: str
    split: str
    instance_ids: tuple[str, ...] | None


def _candidate_hash(candidate: dict[str, str]) -> CandidateHash:
    """Deterministic hash of a candidate dict (order-independent over keys).

    GEPA §3.1: ``sha256(json.dumps(sorted(candidate.items())))``.
    """
    return hashlib.sha256(json.dumps(sorted(candidate.items())).encode()).hexdigest()


@dataclass
class CachedEvaluation(Generic[RolloutOutput]):
    output: RolloutOutput
    score: float
    objective_scores: dict[str, float] | None = None
    side_info: dict[str, Any] | None = None


@dataclass
class EvaluationCache(Generic[RolloutOutput, DataId]):
    _cache: dict[CacheKey, CachedEvaluation[RolloutOutput]] = field(
        default_factory=dict
    )
    # A bounded registry of keys currently being computed.  Entries exist only
    # for cache misses owned by active calls and are removed on both success and
    # failure.  Per-key events let overlapping batches share work without
    # serialising evaluations for independent examples.
    _in_flight: dict[CacheKey, threading.Event] = field(
        default_factory=dict, repr=False, compare=False
    )
    # Thread safety (audit-mutation §C4 / audit-budget-caching §C1): protect
    # both the persisted cache contents and the ephemeral single-flight
    # registry.  Evaluator work and event waits always happen outside the lock.
    _lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )

    def get(
        self, candidate: dict[str, str], example_id: DataId
    ) -> CachedEvaluation[RolloutOutput] | None:
        with self._lock:
            return self._cache.get((_candidate_hash(candidate), example_id))

    def put(
        self,
        candidate: dict[str, str],
        example_id: DataId,
        output: RolloutOutput,
        score: float,
        objective_scores: dict[str, float] | None = None,
        side_info: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._cache[(_candidate_hash(candidate), example_id)] = CachedEvaluation(
                output, score, objective_scores, side_info
            )

    def get_batch(
        self, candidate: dict[str, str], example_ids: list[DataId]
    ) -> tuple[dict[DataId, CachedEvaluation[RolloutOutput]], list[DataId]]:
        h = _candidate_hash(candidate)
        cached: dict[DataId, CachedEvaluation[RolloutOutput]] = {}
        uncached: list[DataId] = []
        with self._lock:
            for eid in example_ids:
                entry = self._cache.get((h, eid))
                if entry is not None:
                    cached[eid] = entry
                else:
                    uncached.append(eid)
        TRACE.emit(
            EventType.CACHE_GET,
            candidate_id=h,
            example_ids=list(example_ids),
            hit_ids=list(cached.keys()),
            miss_ids=list(uncached),
        )
        return cached, uncached

    def put_batch(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        outputs: list[RolloutOutput],
        scores: list[float],
        objective_scores_list: list[dict[str, float]] | None = None,
        side_info_list: list[dict[str, Any]] | None = None,
    ) -> None:
        _validate_batch_cardinality(
            example_ids,
            outputs,
            scores,
            objective_scores_list,
            side_info_list,
        )
        h = _candidate_hash(candidate)
        with self._lock:
            for i, eid in enumerate(example_ids):
                self._cache[(h, eid)] = CachedEvaluation(
                    outputs[i],
                    scores[i],
                    objective_scores_list[i] if objective_scores_list else None,
                    side_info_list[i] if side_info_list else None,
                )
        TRACE.emit(
            EventType.CACHE_PUT,
            candidate_id=h,
            example_ids=list(example_ids),
        )

    def evaluate_with_cache_full(
        self,
        candidate: dict[str, str],
        example_ids: list[DataId],
        fetcher: Callable[[list[DataId]], Any],
        evaluator: Callable[
            [Any, dict[str, str]],
            tuple[
                list[RolloutOutput],
                list[float],
                list[dict[str, float]] | None,
                list[dict[str, Any]] | None,
            ],
        ],
    ) -> tuple[
        dict[DataId, RolloutOutput],
        dict[DataId, float],
        dict[DataId, dict[str, float]] | None,
        dict[DataId, dict[str, Any]] | None,
        int,
    ]:
        candidate_hash = _candidate_hash(candidate)
        outputs_by_id: dict[DataId, RolloutOutput] = {}
        scores_by_id: dict[DataId, float] = {}
        objective_by_id: dict[DataId, dict[str, float]] | None = None
        side_info_by_id: dict[DataId, dict[str, Any]] | None = None
        num_actual_evaluations = 0
        first_partition = True

        # A call can own some missing keys while waiting for another call's
        # overlapping keys.  Compute the owned subset first so independent
        # examples keep running concurrently, then observe or retry any keys
        # whose owner failed.
        while True:
            cached: dict[DataId, CachedEvaluation[RolloutOutput]] = {}
            owned_ids: list[DataId] = []
            owned: dict[CacheKey, threading.Event] = {}
            waiting: dict[CacheKey, threading.Event] = {}

            with self._lock:
                for eid in example_ids:
                    key: CacheKey = (candidate_hash, eid)
                    entry = self._cache.get(key)
                    if entry is not None:
                        cached[eid] = entry
                        continue

                    # Preserve repeated slots within the owner's own batch.
                    # The cache intentionally collapses results by example id,
                    # but the evaluator and usage count retain multiplicity.
                    if key in owned:
                        owned_ids.append(eid)
                        continue

                    flight = self._in_flight.get(key)
                    if flight is None:
                        flight = threading.Event()
                        self._in_flight[key] = flight
                        owned[key] = flight
                        owned_ids.append(eid)
                    else:
                        waiting[key] = flight

            if first_partition:
                TRACE.emit(
                    EventType.CACHE_GET,
                    candidate_id=candidate_hash,
                    example_ids=list(example_ids),
                    hit_ids=list(cached.keys()),
                    miss_ids=[eid for eid in example_ids if eid not in cached],
                )
                first_partition = False

            for eid, entry in cached.items():
                outputs_by_id[eid] = entry.output
                scores_by_id[eid] = entry.score
                if entry.objective_scores is not None:
                    if objective_by_id is None:
                        objective_by_id = {}
                    objective_by_id[eid] = entry.objective_scores
                if entry.side_info is not None:
                    if side_info_by_id is None:
                        side_info_by_id = {}
                    side_info_by_id[eid] = entry.side_info

            if not owned_ids and not waiting:
                break

            if owned_ids:
                try:
                    batch = fetcher(owned_ids)
                    outputs, scores, obj_scores, side_infos = evaluator(
                        batch, candidate
                    )
                    _validate_batch_cardinality(
                        owned_ids,
                        outputs,
                        scores,
                        obj_scores,
                        side_infos,
                    )
                    # Keep the established public write path (including its
                    # cardinality guard and trace event), then wake waiters only
                    # after every result is visible in the cache.
                    self.put_batch(
                        candidate,
                        owned_ids,
                        outputs,
                        scores,
                        obj_scores,
                        side_infos,
                    )
                except BaseException:
                    self._release_owned(owned)
                    raise

                self._release_owned(owned)
                num_actual_evaluations += len(owned_ids)

            # Owners publish cache entries before signalling.  If an owner
            # failed, its event is still signalled after removing the claim;
            # the next loop then claims and retries the still-missing key.
            for flight in waiting.values():
                flight.wait()

        return (
            outputs_by_id,
            scores_by_id,
            objective_by_id,
            side_info_by_id,
            num_actual_evaluations,
        )

    def _release_owned(self, owned: dict[CacheKey, threading.Event]) -> None:
        """Drop completed claims and wake all waiters for those keys."""

        with self._lock:
            for key, flight in owned.items():
                if self._in_flight.get(key) is flight:
                    del self._in_flight[key]
                flight.set()


def _validate_batch_cardinality(
    example_ids: list[DataId],
    outputs: list[RolloutOutput],
    scores: list[float],
    objective_scores_list: list[dict[str, float]] | None,
    side_info_list: list[dict[str, Any]] | None,
) -> None:
    """Validate every positional evaluator output before cache/state mutation."""

    expected = len(example_ids)
    lengths = {
        "outputs": len(outputs),
        "scores": len(scores),
    }
    if objective_scores_list is not None:
        lengths["objective_scores"] = len(objective_scores_list)
    if side_info_list is not None:
        lengths["side_info"] = len(side_info_list)
    mismatched = {name: size for name, size in lengths.items() if size != expected}
    if mismatched:
        rendered = ", ".join(f"{name}={size}" for name, size in mismatched.items())
        raise ValueError(
            "Evaluator batch cardinality mismatch: "
            f"expected {expected} result(s) for {expected} example id(s); {rendered}."
        )
