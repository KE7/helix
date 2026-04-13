"""HELIX evaluation cache — GEPA parity.

Line-for-line port of the cache layer described in
/tmp/gepa_eval_spec.md §3 (originally at gepa/core/state.py:27-130).
"""
from __future__ import annotations

import hashlib
import json
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


@dataclass
class EvaluationCache(Generic[RolloutOutput, DataId]):
    _cache: dict[CacheKey, CachedEvaluation[RolloutOutput]] = field(
        default_factory=dict
    )

    def get(
        self, candidate: dict[str, str], example_id: DataId
    ) -> CachedEvaluation[RolloutOutput] | None:
        return self._cache.get((_candidate_hash(candidate), example_id))

    def put(
        self,
        candidate: dict[str, str],
        example_id: DataId,
        output: RolloutOutput,
        score: float,
        objective_scores: dict[str, float] | None = None,
    ) -> None:
        self._cache[(_candidate_hash(candidate), example_id)] = CachedEvaluation(
            output, score, objective_scores
        )

    def get_batch(
        self, candidate: dict[str, str], example_ids: list[DataId]
    ) -> tuple[dict[DataId, CachedEvaluation[RolloutOutput]], list[DataId]]:
        h = _candidate_hash(candidate)
        cached: dict[DataId, CachedEvaluation[RolloutOutput]] = {}
        uncached: list[DataId] = []
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
    ) -> None:
        h = _candidate_hash(candidate)
        for i, eid in enumerate(example_ids):
            self._cache[(h, eid)] = CachedEvaluation(
                outputs[i],
                scores[i],
                objective_scores_list[i] if objective_scores_list else None,
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
            tuple[list[RolloutOutput], list[float], list[dict[str, float]] | None],
        ],
    ) -> tuple[
        dict[DataId, RolloutOutput],
        dict[DataId, float],
        dict[DataId, dict[str, float]] | None,
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
        if uncached_ids:
            batch = fetcher(uncached_ids)
            outputs, scores, obj_scores = evaluator(batch, candidate)
            for idx, eid in enumerate(uncached_ids):
                outputs_by_id[eid] = outputs[idx]
                scores_by_id[eid] = scores[idx]
                if obj_scores is not None:
                    if objective_by_id is None:
                        objective_by_id = {}
                    objective_by_id[eid] = obj_scores[idx]
            self.put_batch(candidate, uncached_ids, outputs, scores, obj_scores)
        return outputs_by_id, scores_by_id, objective_by_id, len(uncached_ids)
