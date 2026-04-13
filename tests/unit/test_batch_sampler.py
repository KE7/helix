"""Unit tests for EpochShuffledBatchSampler (GEPA parity)."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

import pytest

from helix.batch_sampler import EpochShuffledBatchSampler


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
