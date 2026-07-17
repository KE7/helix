"""Unit tests for helix.eval_cache."""
from __future__ import annotations

import dataclasses

import pytest

from helix.eval_cache import EvaluationBatchKey, EvaluationCache, _candidate_hash


def test_get_put_round_trip() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"a": "1", "b": "2"}
    assert cache.get(cand, 7) is None
    cache.put(
        cand,
        7,
        output="hello",
        score=0.5,
        objective_scores={"acc": 1.0},
        side_info={"trace": "cached"},
    )
    entry = cache.get(cand, 7)
    assert entry is not None
    assert entry.output == "hello"
    assert entry.score == 0.5
    assert entry.objective_scores == {"acc": 1.0}
    assert entry.side_info == {"trace": "cached"}


def test_get_batch_splits_cached_uncached() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"p": "x"}
    cache.put(cand, 1, "o1", 0.1)
    cache.put(cand, 3, "o3", 0.3)
    cached, uncached = cache.get_batch(cand, [1, 2, 3, 4])
    assert set(cached.keys()) == {1, 3}
    assert cached[1].output == "o1"
    assert cached[3].score == 0.3
    assert uncached == [2, 4]


def test_put_batch_stores_all_entries() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"k": "v"}
    cache.put_batch(
        cand,
        [10, 20, 30],
        ["a", "b", "c"],
        [0.1, 0.2, 0.3],
        [{"m": 1.0}, {"m": 2.0}, {"m": 3.0}],
        [{"log": "a"}, {"log": "b"}, {"log": "c"}],
    )
    for eid, out, score, m, log in [
        (10, "a", 0.1, 1.0, "a"),
        (20, "b", 0.2, 2.0, "b"),
        (30, "c", 0.3, 3.0, "c"),
    ]:
        e = cache.get(cand, eid)
        assert e is not None
        assert e.output == out
        assert e.score == score
        assert e.objective_scores == {"m": m}
        assert e.side_info == {"log": log}


def test_put_batch_no_objective_scores() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"k": "v"}
    cache.put_batch(cand, [1, 2], ["a", "b"], [0.5, 0.6])
    e = cache.get(cand, 1)
    assert e is not None
    assert e.objective_scores is None
    assert e.side_info is None


def test_evaluate_with_cache_full_calls_evaluator_only_for_uncached() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"k": "v"}
    # Pre-populate id=1
    cache.put(cand, 1, "cached_out", 0.9)

    calls: list[list[int]] = []

    def fetcher(ids: list[int]) -> list[int]:
        return list(ids)

    def evaluator(
        batch: list[int], _c: dict[str, str]
    ) -> tuple[
        list[str],
        list[float],
        list[dict[str, float]] | None,
        list[dict[str, str]] | None,
    ]:
        calls.append(list(batch))
        outs = [f"out{eid}" for eid in batch]
        scores = [float(eid) / 10 for eid in batch]
        obj = [{"acc": float(eid)} for eid in batch]
        side_info = [{"feedback": f"log{eid}"} for eid in batch]
        return outs, scores, obj, side_info

    outputs, scores, obj_by_id, side_info_by_id, n_uncached = cache.evaluate_with_cache_full(
        cand, [1, 2, 3], fetcher, evaluator
    )

    assert calls == [[2, 3]]
    assert n_uncached == 2
    assert outputs == {1: "cached_out", 2: "out2", 3: "out3"}
    assert scores == {1: 0.9, 2: 0.2, 3: 0.3}
    assert obj_by_id is not None
    assert obj_by_id == {2: {"acc": 2.0}, 3: {"acc": 3.0}}
    assert side_info_by_id == {
        2: {"feedback": "log2"},
        3: {"feedback": "log3"},
    }


def test_evaluate_with_cache_full_second_call_fully_cached() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"k": "v"}
    calls: list[list[int]] = []

    def fetcher(ids: list[int]) -> list[int]:
        return list(ids)

    def evaluator(
        batch: list[int], _c: dict[str, str]
    ) -> tuple[
        list[str],
        list[float],
        list[dict[str, float]] | None,
        list[dict[str, str]] | None,
    ]:
        calls.append(list(batch))
        return (
            [f"o{e}" for e in batch],
            [0.0 for _ in batch],
            None,
            [{"feedback": f"log{e}"} for e in batch],
        )

    cache.evaluate_with_cache_full(cand, [1, 2, 3], fetcher, evaluator)
    assert calls == [[1, 2, 3]]

    outputs, scores, obj, side_info, n_new = cache.evaluate_with_cache_full(
        cand, [1, 2, 3], fetcher, evaluator
    )
    assert calls == [[1, 2, 3]]  # not called again
    assert n_new == 0
    assert outputs == {1: "o1", 2: "o2", 3: "o3"}
    assert scores == {1: 0.0, 2: 0.0, 3: 0.0}
    assert obj is None
    assert side_info == {
        1: {"feedback": "log1"},
        2: {"feedback": "log2"},
        3: {"feedback": "log3"},
    }


def test_candidate_hash_order_independent() -> None:
    a = {"x": "1", "y": "2", "z": "3"}
    b = {"z": "3", "x": "1", "y": "2"}
    assert _candidate_hash(a) == _candidate_hash(b)


def test_different_candidates_different_hashes() -> None:
    a = {"x": "1"}
    b = {"x": "2"}
    c = {"y": "1"}
    assert _candidate_hash(a) != _candidate_hash(b)
    assert _candidate_hash(a) != _candidate_hash(c)
    assert _candidate_hash(b) != _candidate_hash(c)


def test_cache_isolation_by_candidate() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    a = {"k": "v1"}
    b = {"k": "v2"}
    cache.put(a, 1, "out_a", 0.1)
    assert cache.get(b, 1) is None
    cached, uncached = cache.get_batch(b, [1])
    assert cached == {}
    assert uncached == [1]


# ---------------------------------------------------------------------------
# Cardinality validation — must reject before any cache mutation
# ---------------------------------------------------------------------------


def test_put_batch_rejects_short_outputs_before_mutation() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"k": "v"}
    with pytest.raises(ValueError, match="cardinality mismatch"):
        cache.put_batch(cand, [1, 2, 3], ["a", "b"], [0.1, 0.2, 0.3])
    # No partial writes: the whole put was rejected up front.
    assert cache.get(cand, 1) is None
    assert cache.get(cand, 2) is None
    assert cache.get(cand, 3) is None


def test_put_batch_rejects_objective_and_side_info_mismatch() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"k": "v"}
    with pytest.raises(ValueError, match="objective_scores"):
        cache.put_batch(cand, [1, 2], ["a", "b"], [0.1, 0.2], [{"m": 1.0}])
    with pytest.raises(ValueError, match="side_info"):
        cache.put_batch(cand, [1, 2], ["a", "b"], [0.1, 0.2], None, [{"log": "x"}])
    assert cache.get(cand, 1) is None
    assert cache.get(cand, 2) is None


def test_evaluate_with_cache_full_rejects_bad_cardinality_before_mutation() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"k": "v"}

    def fetcher(ids: list[int]) -> list[int]:
        return list(ids)

    def evaluator(
        batch: list[int], _c: dict[str, str]
    ) -> tuple[
        list[str],
        list[float],
        list[dict[str, float]] | None,
        list[dict[str, str]] | None,
    ]:
        # Two outputs but only one score for a two-id request → mismatch.
        return ["o1", "o2"], [0.1], None, None

    with pytest.raises(ValueError, match="cardinality mismatch"):
        cache.evaluate_with_cache_full(cand, [1, 2], fetcher, evaluator)

    # The evaluator ran but its bad payload never reached the cache.
    assert cache.get(cand, 1) is None
    assert cache.get(cand, 2) is None


# ---------------------------------------------------------------------------
# EvaluationBatchKey — cross-candidate dedup identity
# ---------------------------------------------------------------------------


def test_evaluation_batch_key_equality_and_hash() -> None:
    a = EvaluationBatchKey("K", "val", ("1", "2"))
    b = EvaluationBatchKey("K", "val", ("1", "2"))
    assert a == b
    assert hash(a) == hash(b)
    assert a in {b}


def test_evaluation_batch_key_instance_id_order_is_significant() -> None:
    a = EvaluationBatchKey("K", "val", ("1", "2"))
    b = EvaluationBatchKey("K", "val", ("2", "1"))
    assert a != b


def test_evaluation_batch_key_content_split_and_none_distinguish() -> None:
    base = EvaluationBatchKey("K", "val", ("1",))
    assert base != EvaluationBatchKey("J", "val", ("1",))  # content_key
    assert base != EvaluationBatchKey("K", "train", ("1",))  # split
    assert base != EvaluationBatchKey("K", "val", None)  # whole-split vs minibatch


def test_evaluation_batch_key_is_frozen() -> None:
    key = EvaluationBatchKey("K", "val", None)
    with pytest.raises(dataclasses.FrozenInstanceError):
        key.split = "train"  # type: ignore[misc]
