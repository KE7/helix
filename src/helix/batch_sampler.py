"""HELIX minibatch sampler — GEPA parity.

Line-for-line port of
  gepa.strategies.batch_sampler.EpochShuffledBatchSampler
(see /tmp/gepa_eval_spec.md §2).
"""
from __future__ import annotations

import random
from collections import Counter
from typing import Generic, Protocol, TypeVar, runtime_checkable

from helix.trace import TRACE, EventType

DataId = TypeVar("DataId")


@runtime_checkable
class DataLoader(Protocol[DataId]):
    def all_ids(self) -> list[DataId]: ...
    def __len__(self) -> int: ...


class _SamplerState(Protocol):
    i: int  # monotonic proposal counter, starts at -1 (bumped to 0 before first call)


@runtime_checkable
class BatchSampler(Protocol[DataId]):
    def next_minibatch_ids(
        self, loader: DataLoader[DataId], state: _SamplerState
    ) -> list[DataId]: ...


class EpochShuffledBatchSampler(Generic[DataId]):
    """GEPA-parity minibatch sampler.

    Invariants (GEPA §2.3):
      I1. `shuffled_ids` length is a multiple of `minibatch_size` after
          `_update_shuffled`.
      I2. The slice returned for `state.i` is exactly
          `shuffled_ids[(i*m) % L : (i*m) % L + m]` where
          `L = len(shuffled_ids)`, `m = minibatch_size`.
      I3. Shuffling is deterministic given `rng` seed.
      I4. Padding uses the least-frequent id
          (`id_freqs.most_common()[::-1][0][0]`).
    """

    def __init__(self, minibatch_size: int, rng: random.Random | None = None) -> None:
        self.minibatch_size = minibatch_size
        self.shuffled_ids: list[DataId] = []
        self.epoch: int = -1
        self.id_freqs: Counter[DataId] = Counter()
        self.last_trainset_size: int = 0
        self.rng = rng if rng is not None else random.Random(0)

    def _update_shuffled(self, loader: DataLoader[DataId]) -> None:
        all_ids = list(loader.all_ids())
        trainset_size = len(loader)
        self.last_trainset_size = trainset_size
        if trainset_size == 0:
            self.shuffled_ids = []
            self.id_freqs = Counter()
            return
        self.shuffled_ids = list(all_ids)
        self.rng.shuffle(self.shuffled_ids)
        self.id_freqs = Counter(self.shuffled_ids)
        mod = trainset_size % self.minibatch_size
        num_to_pad = (self.minibatch_size - mod) if mod != 0 else 0
        for _ in range(num_to_pad):
            selected_id = self.id_freqs.most_common()[::-1][0][0]
            self.shuffled_ids.append(selected_id)
            self.id_freqs[selected_id] += 1

    def next_minibatch_ids(
        self, loader: DataLoader[DataId], state: _SamplerState
    ) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")
        base_idx = state.i * self.minibatch_size
        curr_epoch = (
            0 if self.epoch == -1 else base_idx // max(len(self.shuffled_ids), 1)
        )
        needs_refresh = (
            not self.shuffled_ids
            or trainset_size != self.last_trainset_size
            or curr_epoch > self.epoch
        )
        if needs_refresh:
            self.epoch = curr_epoch
            self._update_shuffled(loader)
        assert len(self.shuffled_ids) >= self.minibatch_size
        assert len(self.shuffled_ids) % self.minibatch_size == 0
        base_idx = base_idx % len(self.shuffled_ids)
        end_idx = base_idx + self.minibatch_size
        assert end_idx <= len(self.shuffled_ids)
        _ids = self.shuffled_ids[base_idx:end_idx]
        TRACE.emit(
            EventType.SAMPLE_MINIBATCH,
            example_ids=list(_ids),
        )
        return _ids
