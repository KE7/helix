"""HELIX minibatch sampler — GEPA parity.

Line-for-line port of
  gepa.strategies.batch_sampler.EpochShuffledBatchSampler
(see /tmp/gepa_eval_spec.md §2).

Also provides :class:`StratifiedBatchSampler`, which is a HELIX extension
over EpochShuffledBatchSampler that guarantees each minibatch of size K
contains K instances drawn from K *distinct* groups (tasks), where the
group of an instance id is derived via a user-provided ``group_fn``.
This is useful when the training set contains multiple tasks with many
seeds per task: a uniform random sampler will often produce batches
dominated by one task, which is detrimental to generalisation-focused
evolutionary search.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Callable, Generic, Protocol, TypeVar, runtime_checkable

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


class StratifiedBatchSampler(Generic[DataId]):
    """Task-stratified minibatch sampler.

    Each epoch is pre-shuffled such that every minibatch of size ``K`` is a
    slice of ``K`` instances drawn from ``K`` distinct groups (per
    ``group_fn``).  This guarantees task diversity within every minibatch,
    which helps reflection-style evolutionary search reason about
    generalisation instead of overfitting to whichever task happens to
    dominate a random batch.

    Design notes:

    - Epoch layout: at the start of each epoch, we shuffle the instances
      *within* each group, then interleave across groups round-robin.  The
      resulting flat schedule (``self.shuffled_ids``) has the property that
      any contiguous window of ``minibatch_size`` indices begins at a
      multiple of ``minibatch_size`` and touches ``minibatch_size`` distinct
      groups — as long as at least ``minibatch_size`` groups exist.
    - Per-round rotation: when ``num_groups > minibatch_size``, each round
      drops the trailing ``num_groups % minibatch_size`` groups (padding
      would re-introduce a group collision).  We rotate ``group_keys`` by
      ``r`` positions on round ``r`` so the dropped slot rotates across all
      groups within a single epoch — preventing within-epoch group
      starvation.
    - Epoch bumps reshuffle via the shared ``rng``, mirroring
      :class:`EpochShuffledBatchSampler`'s determinism guarantees.
    - Padding: when a round is partial (some buckets exhausted), we trim to
      the largest multiple of ``minibatch_size`` that still fits in the
      round, dropping any trailing remainder rather than padding with
      duplicates (padding would re-introduce a group collision within the
      final minibatch).
    - Fallback: when ``len(groups) < minibatch_size``, a stratified minibatch
      is impossible, so the sampler transparently delegates to an internal
      :class:`EpochShuffledBatchSampler` for GEPA parity semantics.

    Invariants:
      S1. Every returned minibatch of size ``m`` contains exactly ``m``
          distinct group keys (when ``num_groups >= m``).
      S2. Within an epoch each instance is yielded at most once.  In each
          round of the round-robin interleave, only the first
          ``whole_rounds_per_round = (num_groups // m) * m`` entries are
          kept (rounded down to a multiple of ``m`` for partial rounds).
          When ``num_groups % m != 0``, the trailing ``num_groups % m``
          group slots in any given round are dropped — but the per-round
          rotation of ``group_keys`` ensures the dropped slot rotates
          across all groups, so no group is starved within an epoch.
      S3. Determinism given ``rng`` seed, matching
          :class:`EpochShuffledBatchSampler` semantics.
    """

    def __init__(
        self,
        minibatch_size: int,
        group_fn: Callable[[DataId], str],
        rng: random.Random | None = None,
    ) -> None:
        self.minibatch_size = minibatch_size
        self.group_fn = group_fn
        self.rng = rng if rng is not None else random.Random(0)
        self.shuffled_ids: list[DataId] = []
        self.epoch: int = -1
        self.last_trainset_size: int = 0
        # Fallback sampler used when num_groups < minibatch_size.  Lazily
        # created so the common case pays no extra allocations.
        self._fallback: EpochShuffledBatchSampler[DataId] | None = None

    def _update_shuffled(self, loader: DataLoader[DataId]) -> None:
        all_ids = list(loader.all_ids())
        self.last_trainset_size = len(loader)
        if not all_ids:
            self.shuffled_ids = []
            return

        # Bucket by group, preserving input order within each group.
        buckets: dict[str, list[DataId]] = defaultdict(list)
        for _id in all_ids:
            buckets[self.group_fn(_id)].append(_id)

        group_keys = sorted(buckets.keys())
        num_groups = len(group_keys)

        # Fallback path: not enough groups to guarantee stratification.
        if num_groups < self.minibatch_size:
            if self._fallback is None:
                self._fallback = EpochShuffledBatchSampler[DataId](
                    minibatch_size=self.minibatch_size, rng=self.rng
                )
            # Keep schedule empty so next_minibatch_ids delegates.
            self.shuffled_ids = []
            return

        # Shuffle within each group, then shuffle the group order.  Both
        # draws are consumed from the shared ``rng`` for determinism.
        for key in group_keys:
            self.rng.shuffle(buckets[key])
        self.rng.shuffle(group_keys)

        # Interleave round-robin and trim per-round.  Round r picks
        # buckets[k][r] for each key k where r is a valid index for that
        # bucket.  A "full round" touches every group exactly once — that's
        # our stratification guarantee.  Within a round, the first
        # ``whole_rounds_per_round = (num_groups // m) * m`` entries form
        # complete stratified minibatches; any trailing ``num_groups % m``
        # entries are dropped because padding them would require
        # duplicating a group.  When a round is partial (some buckets
        # exhausted), we further clip to a multiple of ``m`` so the slice
        # formula in ``next_minibatch_ids`` never overruns a boundary.
        #
        # Per-round rotation: rotating ``group_keys`` by ``r`` positions
        # before slicing rotates which group lands in the trimmed slot
        # across rounds, so no group is starved within an epoch when
        # ``num_groups > m`` and ``num_groups % m != 0``.
        max_rounds = max(len(buckets[k]) for k in group_keys)
        m = self.minibatch_size
        whole_rounds_per_round = (num_groups // m) * m
        trimmed: list[DataId] = []
        for r in range(max_rounds):
            offset = r % num_groups
            rotated_keys = group_keys[offset:] + group_keys[:offset]
            round_slice = [buckets[k][r] for k in rotated_keys if r < len(buckets[k])]
            # Take at most whole_rounds_per_round entries, then clip to a
            # multiple of m in case this round is partial.
            keep = min(len(round_slice), whole_rounds_per_round)
            keep -= keep % m
            trimmed.extend(round_slice[:keep])

        self.shuffled_ids = trimmed

    def next_minibatch_ids(
        self, loader: DataLoader[DataId], state: _SamplerState
    ) -> list[DataId]:
        trainset_size = len(loader)
        if trainset_size == 0:
            raise ValueError("Cannot sample a minibatch from an empty loader.")

        # Fast path: once the fallback is engaged and the trainset hasn't
        # changed, delegate directly without re-bucketing all_ids() on every
        # call.  ``self.epoch`` is left untouched in this branch — the inner
        # fallback maintains its own epoch state.
        if self._fallback is not None and trainset_size == self.last_trainset_size:
            return self._fallback.next_minibatch_ids(loader, state)

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

        # Fallback: delegate to the standard sampler when we can't stratify.
        if not self.shuffled_ids and self._fallback is not None:
            return self._fallback.next_minibatch_ids(loader, state)

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
