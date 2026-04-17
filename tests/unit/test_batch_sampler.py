"""Unit tests for EpochShuffledBatchSampler (GEPA parity) and StratifiedBatchSampler."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pytest

from helix.batch_sampler import EpochShuffledBatchSampler, StratifiedBatchSampler


@dataclass
class _Loader:
    ids: list[Any]

    def all_ids(self) -> list[Any]:
        return list(self.ids)

    def __len__(self) -> int:
        return len(self.ids)


@dataclass
class _State:
    i: int = 0


def test_basic_slices_size_3_of_9() -> None:
    loader = _Loader(list(range(9)))
    sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(42))
    state = _State(i=0)
    b0 = sampler.next_minibatch_ids(loader, state)
    assert len(b0) == 3
    state.i = 1
    b1 = sampler.next_minibatch_ids(loader, state)
    assert len(b1) == 3
    state.i = 2
    b2 = sampler.next_minibatch_ids(loader, state)
    assert len(b2) == 3
    # Full epoch covers every id exactly once (9 % 3 == 0, no padding).
    assert sorted(b0 + b1 + b2) == list(range(9))


def test_padding_7_examples_size_3() -> None:
    loader = _Loader(list(range(7)))
    sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(0))
    state = _State(i=0)
    sampler.next_minibatch_ids(loader, state)
    assert len(sampler.shuffled_ids) % 3 == 0
    assert len(sampler.shuffled_ids) == 9  # 7 rounded up to 9
    # All original ids still present.
    for eid in range(7):
        assert eid in sampler.shuffled_ids


def test_epoch_bump_triggers_reshuffle() -> None:
    loader = _Loader(list(range(6)))
    sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(1))
    state = _State(i=0)
    sampler.next_minibatch_ids(loader, state)
    first_shuffle = list(sampler.shuffled_ids)
    assert sampler.epoch == 0
    # i=2 -> base_idx=6, curr_epoch=1, triggers refresh
    state.i = 2
    sampler.next_minibatch_ids(loader, state)
    assert sampler.epoch == 1
    # Extremely likely different order (6! possibilities); assert at least the
    # shuffle was recomputed by checking list identity semantics: it's a fresh list.
    assert sampler.shuffled_ids is not first_shuffle


def test_seed_determinism() -> None:
    loader = _Loader(list(range(10)))
    s1 = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(123))
    s2 = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(123))
    seq1, seq2 = [], []
    for i in range(5):
        st1, st2 = _State(i=i), _State(i=i)
        seq1.append(s1.next_minibatch_ids(loader, st1))
        seq2.append(s2.next_minibatch_ids(loader, st2))
    assert seq1 == seq2


def test_empty_loader_raises() -> None:
    loader = _Loader([])
    sampler = EpochShuffledBatchSampler(minibatch_size=3)
    with pytest.raises(ValueError):
        sampler.next_minibatch_ids(loader, _State(i=0))


def test_returned_ids_subset_of_loader() -> None:
    loader = _Loader(["a", "b", "c", "d", "e"])
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(7))
    all_ids = set(loader.all_ids())
    for i in range(6):
        batch = sampler.next_minibatch_ids(loader, _State(i=i))
        for eid in batch:
            assert eid in all_ids


def test_non_overlapping_within_epoch() -> None:
    loader = _Loader(list(range(12)))
    sampler = EpochShuffledBatchSampler(minibatch_size=3, rng=random.Random(99))
    seen: list[int] = []
    # 12 / 3 = 4 steps per epoch, no padding
    for i in range(4):
        batch = sampler.next_minibatch_ids(loader, _State(i=i))
        seen.extend(batch)
    assert sorted(seen) == list(range(12))
    assert len(set(seen)) == 12


def test_state_i_slice_formula() -> None:
    """I2: slice returned equals shuffled_ids[(i*m) % L : (i*m)%L + m]."""
    loader = _Loader(list(range(8)))
    sampler = EpochShuffledBatchSampler(minibatch_size=2, rng=random.Random(5))
    # Prime shuffled_ids
    sampler.next_minibatch_ids(loader, _State(i=0))
    L = len(sampler.shuffled_ids)
    m = sampler.minibatch_size
    for i in range(4):
        batch = sampler.next_minibatch_ids(loader, _State(i=i))
        base = (i * m) % L
        assert batch == sampler.shuffled_ids[base : base + m]


# ---------------------------------------------------------------------------
# StratifiedBatchSampler tests
# ---------------------------------------------------------------------------


def _group_by_prefix(example_id: str, sep: str = "__") -> str:
    return example_id.split(sep, 1)[0]


def test_stratified_minibatch_contains_k_distinct_groups() -> None:
    """Every minibatch of size 3 should touch 3 distinct groups."""
    ids = ["a__0", "a__1", "b__0", "b__1", "c__0", "c__1"]
    loader = _Loader(ids)
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(0)
    )
    # Run across several minibatches, including epoch wrap-around.
    for i in range(8):
        batch = sampler.next_minibatch_ids(loader, _State(i=i))
        assert len(batch) == 3
        groups = {_group_by_prefix(eid) for eid in batch}
        assert groups == {"a", "b", "c"}, (
            f"minibatch {i} had groups {groups}, expected {{'a','b','c'}}"
        )


def test_stratified_epoch_covers_all_instances() -> None:
    """Each full epoch should emit every instance exactly once (balanced groups)."""
    ids = ["a__0", "a__1", "b__0", "b__1", "c__0", "c__1"]
    loader = _Loader(ids)
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(11)
    )
    # 6 ids / minibatch_size 3 = 2 steps per epoch.
    seen: list[str] = []
    for i in range(2):
        seen.extend(sampler.next_minibatch_ids(loader, _State(i=i)))
    assert sorted(seen) == sorted(ids)
    assert len(set(seen)) == len(ids)


def test_stratified_epoch_rotation_reshuffles() -> None:
    """Crossing into a new epoch should trigger a fresh shuffle."""
    ids = ["a__0", "a__1", "b__0", "b__1", "c__0", "c__1"]
    loader = _Loader(ids)
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(4)
    )
    sampler.next_minibatch_ids(loader, _State(i=0))
    first_schedule = list(sampler.shuffled_ids)
    assert sampler.epoch == 0
    # i=2 -> base_idx = 6 = len(shuffled_ids), crosses into epoch 1.
    sampler.next_minibatch_ids(loader, _State(i=2))
    assert sampler.epoch == 1
    # A fresh list is produced each epoch (identity check).
    assert sampler.shuffled_ids is not first_schedule


def test_stratified_fallback_when_groups_lt_minibatch_size() -> None:
    """Fewer groups than K → fall back to EpochShuffledBatchSampler semantics."""
    # 2 groups but minibatch_size=3: stratification impossible, fall back.
    ids = ["a__0", "a__1", "a__2", "b__0", "b__1", "b__2"]
    loader = _Loader(ids)
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(0)
    )
    batch = sampler.next_minibatch_ids(loader, _State(i=0))
    assert len(batch) == 3
    assert sampler._fallback is not None, "expected fallback sampler to be engaged"
    # All returned ids must come from the loader.
    all_ids = set(ids)
    for eid in batch:
        assert eid in all_ids


def test_stratified_fallback_epoch_covers_all_instances() -> None:
    """Fallback path should still cover every instance within an epoch."""
    ids = ["a__0", "a__1", "a__2", "b__0", "b__1", "b__2"]
    loader = _Loader(ids)
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(2)
    )
    seen: list[str] = []
    # 6 ids / 3 = 2 minibatches per epoch.
    for i in range(2):
        seen.extend(sampler.next_minibatch_ids(loader, _State(i=i)))
    assert sorted(seen) == sorted(ids)


def test_stratified_empty_loader_raises() -> None:
    loader = _Loader([])
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(0)
    )
    with pytest.raises(ValueError):
        sampler.next_minibatch_ids(loader, _State(i=0))


def test_stratified_determinism_same_seed() -> None:
    ids = [f"{g}__{i}" for g in ("a", "b", "c", "d") for i in range(3)]
    loader = _Loader(ids)
    s1 = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(42)
    )
    s2 = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(42)
    )
    seq1 = [s1.next_minibatch_ids(loader, _State(i=i)) for i in range(6)]
    seq2 = [s2.next_minibatch_ids(loader, _State(i=i)) for i in range(6)]
    assert seq1 == seq2


def test_stratified_unbalanced_group_sizes_still_stratifies() -> None:
    """With unbalanced groups (3/2/2), full stratified rounds still dominate."""
    ids = ["a__0", "a__1", "a__2", "b__0", "b__1", "c__0", "c__1"]
    loader = _Loader(ids)
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(7)
    )
    # Run through a couple of epochs — every minibatch still touches 3 groups.
    for i in range(6):
        batch = sampler.next_minibatch_ids(loader, _State(i=i))
        groups = {_group_by_prefix(eid) for eid in batch}
        assert len(groups) == 3


def test_stratified_num_groups_gt_m_unbalanced_no_crash() -> None:
    """Regression: num_groups > minibatch_size with uneven group sizes used to
    crash the trim logic (PR#3 review MUST-FIX).

    Reproducer: 4 groups of sizes (3, 2, 1, 3), m=2 — at least one round has
    `m <= len(round_slice) < num_groups`, so the previous flat-chunking trim
    misaligned and produced a `len(shuffled_ids)` that was not a multiple of
    `m`, tripping the `% m == 0` assert in `next_minibatch_ids`.
    """
    ids = [
        "a__0", "a__1", "a__2",
        "b__0", "b__1",
        "c__0",
        "d__0", "d__1", "d__2",
    ]
    # Run across every seed in 0..9 — review reports failure on all of them.
    for seed in range(10):
        loader = _Loader(ids)
        sampler = StratifiedBatchSampler(
            minibatch_size=2, group_fn=_group_by_prefix, rng=random.Random(seed)
        )
        # Drive multiple steps, including across epoch boundaries.
        for i in range(8):
            batch = sampler.next_minibatch_ids(loader, _State(i=i))
            assert len(batch) == 2
            groups = {_group_by_prefix(eid) for eid in batch}
            assert len(groups) == 2, (
                f"seed={seed} i={i} batch={batch} should have 2 distinct groups"
            )
        # Schedule must always be a multiple of minibatch_size.
        assert len(sampler.shuffled_ids) % 2 == 0


def test_stratified_num_groups_gt_m_balanced_eventual_coverage() -> None:
    """Across N epochs with `num_groups > m` balanced groups, every instance
    must appear at least once in the union of emitted minibatches.

    This guards against within-epoch group starvation regressing further
    (PR#3 review SHOULD-FIX #2): without per-round group rotation, the same
    `num_groups % m` groups would be dropped every round of every epoch
    until the per-epoch shuffle rotated them in by chance.
    """
    # 5 groups of size 3 each, m=3 -> num_groups % m == 2 dropped per round.
    ids = [f"{g}__{i}" for g in ("a", "b", "c", "d", "e") for i in range(3)]
    loader = _Loader(ids)
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(0)
    )
    seen: set[str] = set()
    # 5 epochs of 3 minibatches each — with rotation, this covers every id.
    for i in range(15):
        batch = sampler.next_minibatch_ids(loader, _State(i=i))
        seen.update(batch)
        assert len({_group_by_prefix(eid) for eid in batch}) == 3
    assert seen == set(ids), f"missing ids: {set(ids) - seen}"


def test_stratified_num_groups_gt_m_per_round_rotation_covers_all_groups() -> None:
    """Within a single epoch with `num_groups > m`, every group should appear
    at least once across the emitted schedule (per-round rotation guarantees
    this when there are enough rounds).
    """
    # 4 groups of size 4 each, m=3 -> 1 group dropped per round, 4 rounds
    # total -> rotation visits all groups within the epoch.
    ids = [f"{g}__{i}" for g in ("a", "b", "c", "d") for i in range(4)]
    loader = _Loader(ids)
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(0)
    )
    sampler.next_minibatch_ids(loader, _State(i=0))
    schedule_groups = {_group_by_prefix(eid) for eid in sampler.shuffled_ids}
    assert schedule_groups == {"a", "b", "c", "d"}, (
        "every group should appear at least once in the schedule "
        f"(got {schedule_groups})"
    )


def test_stratified_fallback_short_circuits_after_engagement() -> None:
    """Once the fallback is engaged and the trainset hasn't changed, calls
    should bypass `_update_shuffled` entirely (PR#3 review SHOULD-FIX #3).
    """
    ids = ["a__0", "a__1", "a__2", "b__0", "b__1", "b__2"]
    loader = _Loader(ids)
    sampler = StratifiedBatchSampler(
        minibatch_size=3, group_fn=_group_by_prefix, rng=random.Random(0)
    )
    # Engage the fallback.
    sampler.next_minibatch_ids(loader, _State(i=0))
    assert sampler._fallback is not None

    # Patch _update_shuffled so any further call would explode — proves we
    # short-circuit straight to the fallback when trainset_size is stable.
    call_count = {"n": 0}
    original = sampler._update_shuffled

    def _tripwire(*args: Any, **kwargs: Any) -> None:
        call_count["n"] += 1
        original(*args, **kwargs)

    sampler._update_shuffled = _tripwire  # type: ignore[method-assign]
    for i in range(1, 6):
        sampler.next_minibatch_ids(loader, _State(i=i))
    assert call_count["n"] == 0, (
        "fallback path should short-circuit and never re-bucket all_ids()"
    )
