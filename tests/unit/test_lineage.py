"""Unit tests for helix.lineage.find_merge_triplet.

Ported from GEPA's merge pair-selection behavior
(gepa/proposer/merge.py:87-115, 118-207).  Covers the post-align-merge
changes spelled out in the merge-pairing audit
(/tmp/audit_audit-merge-pairing.md):

- B1: overlap-floor filter moved INTO the retry loop.
- B2: attempted-pair filter moved INTO the retry loop.
- C3: canonical (lex-sorted) pair returned, mirroring GEPA
  ``merge.py:94-95`` ``if j < i: i, j = j, i``.
"""

from __future__ import annotations

import random


from helix.lineage import LineageEntry, find_merge_triplet


def _entry(cid: str, parents: list[str]) -> LineageEntry:
    return LineageEntry(
        id=cid,
        parent=parents[0] if parents else None,
        parents=parents,
        operation="mutate" if parents else "seed",
        generation=len(parents),
        files_changed=[],
    )


def _build_lineage(parents_by_id: dict[str, list[str]]) -> dict[str, LineageEntry]:
    return {cid: _entry(cid, parents) for cid, parents in parents_by_id.items()}


class TestFindMergeTripletCanonicalization:
    """GEPA parity (merge-pairing audit C3, merge.py:94-95).

    The sampled pair is lex-sorted inside ``find_merge_triplet`` so that
    ``(A, B)`` and ``(B, A)`` both surface as ``(A, B)`` regardless of the
    order ``rng.sample`` yielded.  This keeps the merge subprocess arg
    order deterministic across reseeds and makes the attempted-pair /
    description-triplet ledgers insensitive to sample order.
    """

    def test_returns_pair_in_lex_sorted_order(self):
        lineage = _build_lineage(
            {
                "anc": [],
                "z-child": ["anc"],
                "a-child": ["anc"],
            }
        )
        scores = {"anc": 0.0, "z-child": 0.5, "a-child": 0.5}
        # Force the rng to yield ("z-child", "a-child") first.
        rng = random.Random(0)
        # Deterministic seed-check: with frontier=[z, a] and seed 0,
        # rng.sample picks them in whatever internal order; regardless of
        # that order, the returned pair must be canonical.
        triplet = find_merge_triplet(
            lineage, ["z-child", "a-child"], scores, rng=rng, max_attempts=1,
        )
        assert triplet is not None
        i, j, _ancestor = triplet
        assert i < j, f"expected canonical (i <= j), got ({i}, {j})"
        assert i == "a-child" and j == "z-child"


class TestFindMergeTripletWithinRetryFilters:
    """GEPA parity (merge-pairing audit B1+B2, merge.py:147-148, 199-201).

    The attempted-pair and val-overlap filters live INSIDE the retry loop
    (``for _ in range(max_attempts)``).  A blocked sample resamples within
    the same call instead of forcing propose() to return ``None`` and
    burning the merge slot for the whole iteration.
    """

    def test_skips_attempted_pair_and_finds_unblocked(self):
        # Three non-dominated siblings of a single ancestor.
        lineage = _build_lineage(
            {
                "anc": [],
                "c1": ["anc"],
                "c2": ["anc"],
                "c3": ["anc"],
            }
        )
        scores = {"anc": 0.0, "c1": 0.5, "c2": 0.5, "c3": 0.5}
        # Block (c1, c2) — retry must land on a different canonical pair.
        attempted = {("c1", "c2")}
        # Run 30 seeds and verify we always get one of the unblocked pairs.
        for seed in range(30):
            rng = random.Random(seed)
            triplet = find_merge_triplet(
                lineage, ["c1", "c2", "c3"], scores,
                rng=rng, max_attempts=10,
                attempted_pairs=attempted,
            )
            assert triplet is not None
            i, j, _ = triplet
            assert (i, j) != ("c1", "c2"), (
                f"seed {seed}: returned blocked pair (c1,c2) instead of retrying"
            )
            assert (i, j) in {("c1", "c3"), ("c2", "c3")}

    def test_skips_low_overlap_pair_and_finds_unblocked(self):
        lineage = _build_lineage(
            {
                "anc": [],
                "c1": ["anc"],
                "c2": ["anc"],
                "c3": ["anc"],
            }
        )
        scores = {"anc": 0.0, "c1": 0.5, "c2": 0.5, "c3": 0.5}
        # Pretend (c1, c2) fails overlap floor; any other pair passes.
        def _overlap(i: str, j: str) -> bool:
            return (i, j) != ("c1", "c2")

        for seed in range(30):
            rng = random.Random(seed)
            triplet = find_merge_triplet(
                lineage, ["c1", "c2", "c3"], scores,
                rng=rng, max_attempts=10,
                has_val_support_overlap=_overlap,
            )
            assert triplet is not None
            i, j, _ = triplet
            assert (i, j) != ("c1", "c2"), (
                f"seed {seed}: returned low-overlap pair despite filter"
            )

    def test_returns_none_when_all_pairs_blocked(self):
        lineage = _build_lineage(
            {
                "anc": [],
                "c1": ["anc"],
                "c2": ["anc"],
            }
        )
        scores = {"anc": 0.0, "c1": 0.5, "c2": 0.5}
        rng = random.Random(0)
        triplet = find_merge_triplet(
            lineage, ["c1", "c2"], scores,
            rng=rng, max_attempts=10,
            attempted_pairs={("c1", "c2")},
        )
        assert triplet is None, (
            "all candidate pairs attempted → no valid triplet exists"
        )

    def test_returns_none_when_all_pairs_fail_overlap(self):
        lineage = _build_lineage(
            {
                "anc": [],
                "c1": ["anc"],
                "c2": ["anc"],
            }
        )
        scores = {"anc": 0.0, "c1": 0.5, "c2": 0.5}
        rng = random.Random(0)
        triplet = find_merge_triplet(
            lineage, ["c1", "c2"], scores,
            rng=rng, max_attempts=10,
            has_val_support_overlap=lambda _i, _j: False,
        )
        assert triplet is None, (
            "every candidate pair fails overlap → no valid triplet exists"
        )


class TestFindMergeTripletBackwardCompat:
    """GEPA parity (merge.py:87-115): without the new optional filters, the
    behavior is unchanged — attempted_pairs=None, has_val_support_overlap=None
    means "no filtering" and the canonical ``(i, j, ancestor)`` triplet is
    returned for any valid frontier.  The ``val_stage_size``-None invariant
    routes through this same path (merge acceptance is independent of
    ``val_stage_size``), so this test doubles as the no-op regression.
    """

    def test_no_filters_returns_valid_triplet(self):
        lineage = _build_lineage(
            {
                "anc": [],
                "c1": ["anc"],
                "c2": ["anc"],
            }
        )
        scores = {"anc": 0.0, "c1": 0.5, "c2": 0.5}
        rng = random.Random(42)
        triplet = find_merge_triplet(lineage, ["c1", "c2"], scores, rng=rng)
        assert triplet is not None
        i, j, anc = triplet
        # canonical order
        assert i <= j
        assert anc == "anc"
        assert {i, j} == {"c1", "c2"}
