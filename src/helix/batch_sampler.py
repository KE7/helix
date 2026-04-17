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

    Each minibatch of size ``K`` is constructed by choosing ``K`` distinct
    groups (per ``group_fn``) and then picking one unused instance from each
    of those groups.  This guarantees task diversity within every minibatch,
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
    - Epoch bumps reshuffle via the shared ``rng``, mirroring
      :class:`EpochShuffledBatchSampler`'s determinism guarantees.
    - Padding: when the combined interleaved schedule is not a multiple of
      ``minibatch_size``, we drop the trailing partial round rather than
      pad with duplicates, because padding would re-introduce a group
      collision within the final minibatch.  Because all full rounds are
      emitted first, this only ever discards a tail of size
      ``< minibatch_size`` per epoch.
    - Fallback: when ``len(groups) < minibatch_size``, a stratified minibatch
      is impossible, so the sampler transparently delegates to an internal
      :class:`EpochShuffledBatchSampler` for GEPA parity semantics.

    Invariants:
      S1. Every returned minibatch of size ``m`` contains exactly ``m``
          distinct group keys (when ``num_groups >= m``).
      S2. Across a full epoch, each instance is yielded at most once
          (instances beyond ``floor(min_group_size * num_groups / m) * m``
          may be dropped in the final partial round).
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

        # Interleave round-robin: round r picks buckets[k][r] for each key k,
        # as long as index r is valid for that bucket.  A "full round" touches
        # every group exactly once — that's our stratification guarantee.
        max_rounds = max(len(buckets[k]) for k in group_keys)
        schedule: list[DataId] = []
        for r in range(max_rounds):
            round_slice: list[DataId] = []
            for key in group_keys:
                if r < len(buckets[key]):
                    round_slice.append(buckets[key][r])
            # Only emit the round if it fills at least one full minibatch of
            # distinct groups; otherwise stratification would break.
            if len(round_slice) >= self.minibatch_size:
                schedule.extend(round_slice)

        # Trim to a whole-number-of-minibatches length so the slice formula
        # in ``next_minibatch_ids`` never overruns a boundary.  Each full
        # round has length ``num_groups``; within a round, the first
        # ``(num_groups // m) * m`` entries form complete stratified
        # minibatches — any trailing ``num_groups % m`` entries are dropped
        # because padding them would require duplicating a group.
        m = self.minibatch_size
        whole_rounds_per_round = (num_groups // m) * m
        trimmed: list[DataId] = []
        # Split ``schedule`` back into rounds of size ``num_groups``.
        for start in range(0, len(schedule), num_groups):
            round_ids = schedule[start : start + num_groups]
            trimmed.extend(round_ids[:whole_rounds_per_round])

        self.shuffled_ids = trimmed

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
