"""Unit tests for helix.eval_cache."""
from __future__ import annotations


from helix.eval_cache import EvaluationCache, _candidate_hash


def test_get_put_round_trip() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"a": "1", "b": "2"}
    assert cache.get(cand, 7) is None
    cache.put(cand, 7, output="hello", score=0.5, objective_scores={"acc": 1.0})
    entry = cache.get(cand, 7)
    assert entry is not None
    assert entry.output == "hello"
    assert entry.score == 0.5
    assert entry.objective_scores == {"acc": 1.0}


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
    )
    for eid, out, score, m in [
        (10, "a", 0.1, 1.0),
        (20, "b", 0.2, 2.0),
        (30, "c", 0.3, 3.0),
    ]:
        e = cache.get(cand, eid)
        assert e is not None
        assert e.output == out
        assert e.score == score
        assert e.objective_scores == {"m": m}


def test_put_batch_no_objective_scores() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"k": "v"}
    cache.put_batch(cand, [1, 2], ["a", "b"], [0.5, 0.6])
    e = cache.get(cand, 1)
    assert e is not None
    assert e.objective_scores is None


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
    ) -> tuple[list[str], list[float], list[dict[str, float]] | None]:
        calls.append(list(batch))
        outs = [f"out{eid}" for eid in batch]
        scores = [float(eid) / 10 for eid in batch]
        obj = [{"acc": float(eid)} for eid in batch]
        return outs, scores, obj

    outputs, scores, obj_by_id, n_uncached = cache.evaluate_with_cache_full(
        cand, [1, 2, 3], fetcher, evaluator
    )

    assert calls == [[2, 3]]
    assert n_uncached == 2
    assert outputs == {1: "cached_out", 2: "out2", 3: "out3"}
    assert scores == {1: 0.9, 2: 0.2, 3: 0.3}
    assert obj_by_id is not None
    assert obj_by_id == {2: {"acc": 2.0}, 3: {"acc": 3.0}}


def test_evaluate_with_cache_full_second_call_fully_cached() -> None:
    cache: EvaluationCache[str, int] = EvaluationCache()
    cand = {"k": "v"}
    calls: list[list[int]] = []

    def fetcher(ids: list[int]) -> list[int]:
        return list(ids)

    def evaluator(
        batch: list[int], _c: dict[str, str]
    ) -> tuple[list[str], list[float], list[dict[str, float]] | None]:
        calls.append(list(batch))
        return [f"o{e}" for e in batch], [0.0 for _ in batch], None

    cache.evaluate_with_cache_full(cand, [1, 2, 3], fetcher, evaluator)
    assert calls == [[1, 2, 3]]

    outputs, scores, obj, n_new = cache.evaluate_with_cache_full(
        cand, [1, 2, 3], fetcher, evaluator
    )
    assert calls == [[1, 2, 3]]  # not called again
    assert n_new == 0
    assert outputs == {1: "o1", 2: "o2", 3: "o3"}
    assert scores == {1: 0.0, 2: 0.0, 3: 0.0}
    assert obj is None


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
