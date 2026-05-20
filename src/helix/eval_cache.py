"""HELIX evaluation cache — GEPA parity.

Port of the cache layer from GEPA's ``gepa/core/state.py:27-130``
(public source at ``github.com/gepa-ai/gepa``), extended with a
per-(candidate, example) ``side_info`` slot.

The base GEPA ``CachedEvaluation`` stores only ``(output, score,
objective_scores)``.  GEPA's ``OptimizeAnythingAdapter._eval_cache``
(``gepa/adapters/optimize_anything_adapter/optimize_anything_adapter.py:92,
200-216``) caches the richer ``(score, output, side_info)`` tuple in a
*second* adapter-level cache, precisely because reflection prompts need
the side_info to survive a cache hit.  HELIX has no analogous adapter
cache — the engine-level cache is the only one — so we fold the
side_info slot into ``CachedEvaluation`` here.  This preserves the
LIBERO feedback signal (``evaluation_diagnostics``, ``judge_metrics``,
``evaluator_error``, ``video_path``, ...) across cache hits, which is
what reaches the mutator's ``## Diagnostics`` section
(see ``helix.mutator._render_per_example_diagnostics``).
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
    """Per-(candidate, example) cached evaluation entry.

    ``output``, ``score``, and ``objective_scores`` mirror GEPA's
    base ``CachedEvaluation`` (``gepa/core/state.py:37-43``).  ``side_info``
    is the HELIX-specific extension that lets per-example reflection
    diagnostics (LIBERO feedback, evaluator error traces, judge metrics,
    media references, ...) survive cache hits — see the module docstring
    for the GEPA OptimizeAnythingAdapter precedent.
    """

    output: RolloutOutput
    score: float
    objective_scores: dict[str, float] | None = None
    side_info: dict[str, Any] | None = None


@dataclass
class EvaluationCache(Generic[RolloutOutput, DataId]):
    _cache: dict[CacheKey, CachedEvaluation[RolloutOutput]] = field(
        default_factory=dict
    )
    # Thread safety: the parent-minibatch eval runs inside a
    # ThreadPoolExecutor (see
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
        """Evaluate ``candidate`` on ``example_ids`` with per-example caching.

        Evaluator return shape (4-tuple):

          ``(outputs, scores, objective_scores, side_infos)``

        ``objective_scores`` and ``side_infos`` are both optional
        (``None`` allowed).  When provided they must be positional to
        ``batch`` (length ``len(batch)``).

        Returns ``(outputs_by_id, scores_by_id, objective_by_id,
        side_info_by_id, num_actual_evals)``.  ``objective_by_id`` and
        ``side_info_by_id`` are ``None`` when no entry — cached or
        fresh — produced that axis; otherwise they cover the union of
        cache-hit and fresh-miss ids for which the axis was non-empty.
        """
        cached, uncached_ids = self.get_batch(candidate, example_ids)
        outputs_by_id: dict[DataId, RolloutOutput] = {
            eid: c.output for eid, c in cached.items()
        }
        scores_by_id: dict[DataId, float] = {
            eid: c.score for eid, c in cached.items()
        }
        objective_by_id: dict[DataId, dict[str, float]] | None = None
        side_info_by_id: dict[DataId, dict[str, Any]] | None = None
        for eid, c in cached.items():
            if c.objective_scores is not None:
                if objective_by_id is None:
                    objective_by_id = {}
                objective_by_id[eid] = c.objective_scores
            if c.side_info is not None:
                if side_info_by_id is None:
                    side_info_by_id = {}
                side_info_by_id[eid] = c.side_info
        if uncached_ids:
            batch = fetcher(uncached_ids)
            outputs, scores, obj_scores, side_infos = evaluator(batch, candidate)
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
        return (
            outputs_by_id,
            scores_by_id,
            objective_by_id,
            side_info_by_id,
            len(uncached_ids),
        )
