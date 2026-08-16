"""HELIX evaluation cache — GEPA parity.

Ports the cache layer GEPA implements as ``_candidate_hash``,
``CachedEvaluation``, and ``EvaluationCache`` in ``core/state.py``.
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


def _candidate_hash(candidate: dict[str, str]) -> CandidateHash:
    """Deterministic hash of a candidate dict (order-independent over keys).

    GEPA §3.1: ``sha256(json.dumps(sorted(candidate.items())))``.
    """
    return hashlib.sha256(
        json.dumps(sorted(candidate.items())).encode()
    ).hexdigest()


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
    # Thread safety (mutation audit C4 / budget-caching audit C1): the
    # parent-minibatch eval now runs inside a ThreadPoolExecutor (see
    # evolution.py parent-eval parallel stage), so concurrent readers/writers
    # can race on ``_cache``.  A single lock serialises every access.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

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
        # Validate every positional list BEFORE taking the lock and mutating
        # the cache.  A mismatched batch must be rejected whole: writing the
        # prefix and then raising leaves the cache holding entries from a
        # batch the caller believes it never stored.
        _validate_batch_cardinality(
            example_ids, outputs, scores, objective_scores_list, side_info_list
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
        cached, uncached_ids = self.get_batch(candidate, example_ids)
        outputs_by_id: dict[DataId, RolloutOutput] = {
            eid: c.output for eid, c in cached.items()
        }
        scores_by_id: dict[DataId, float] = {
            eid: c.score for eid, c in cached.items()
        }
        objective_by_id: dict[DataId, dict[str, float]] | None = None
        for eid, c in cached.items():
            if c.objective_scores is not None:
                if objective_by_id is None:
                    objective_by_id = {}
                objective_by_id[eid] = c.objective_scores
        side_info_by_id: dict[DataId, dict[str, Any]] | None = None
        for eid, c in cached.items():
            if c.side_info is not None:
                if side_info_by_id is None:
                    side_info_by_id = {}
                side_info_by_id[eid] = c.side_info
        if uncached_ids:
            batch = fetcher(uncached_ids)
            outputs, scores, obj_scores, side_infos = evaluator(batch, candidate)
            # Validate before the projection loop below: an evaluator that
            # returns the wrong number of results would otherwise raise a bare
            # IndexError partway through, or silently drop trailing results.
            _validate_batch_cardinality(
                uncached_ids, outputs, scores, obj_scores, side_infos
            )
            for idx, eid in enumerate(uncached_ids):
                outputs_by_id[eid] = outputs[idx]
                scores_by_id[eid] = scores[idx]
                if obj_scores is not None:
                    if objective_by_id is None:
                        objective_by_id = {}
                    objective_by_id[eid] = obj_scores[idx]
                if side_infos is not None:
                    if side_info_by_id is None:
                        side_info_by_id = {}
                    side_info_by_id[eid] = side_infos[idx]
            self.put_batch(
                candidate,
                uncached_ids,
                outputs,
                scores,
                obj_scores,
                side_infos,
            )
        return outputs_by_id, scores_by_id, objective_by_id, side_info_by_id, len(uncached_ids)


def _validate_batch_cardinality(
    example_ids: list[DataId],
    outputs: list[RolloutOutput],
    scores: list[float],
    objective_scores_list: list[dict[str, float]] | None,
    side_info_list: list[dict[str, Any]] | None,
) -> None:
    """Reject a batch whose evaluator results do not match its example ids.

    Every list here is positional to ``example_ids``.  A short list used to
    surface as a bare ``IndexError`` partway through a write (leaving the cache
    holding a prefix of a rejected batch); a long one was silently truncated,
    discarding results the evaluator actually produced.  Both are evaluator
    contract violations and are reported as such, before any mutation.
    """
    expected = len(example_ids)
    lengths = {"outputs": len(outputs), "scores": len(scores)}
    if objective_scores_list is not None:
        lengths["objective_scores"] = len(objective_scores_list)
    if side_info_list is not None:
        lengths["side_info"] = len(side_info_list)
    mismatched = {name: size for name, size in lengths.items() if size != expected}
    if mismatched:
        rendered = ", ".join(f"{name}={size}" for name, size in mismatched.items())
        raise ValueError(
            "Evaluator batch cardinality mismatch: expected "
            f"{expected} result(s) for {expected} example id(s); {rendered}."
        )
