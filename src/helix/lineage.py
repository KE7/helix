"""HELIX lineage tracking — records the ancestry of every candidate."""

from __future__ import annotations

import json
import random as _random
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class LineageEntry:
    """Records the ancestry and provenance of a single candidate.

    Tracks the candidate's ID, parent(s), operation type (mutation/merge),
    generation number, and which files were modified.
    """
    id: str
    parent: str | None
    parents: list[str]
    operation: str
    generation: int
    files_changed: list[str]


def record_entry(lineage_path: Path, entry: LineageEntry) -> None:
    """Append a LineageEntry to the lineage JSON file.

    The file is a JSON array.  We read it, append, and rewrite atomically
    using a temporary file so partial writes never corrupt the record.
    """
    lineage_path.parent.mkdir(parents=True, exist_ok=True)

    if lineage_path.exists():
        with open(lineage_path) as f:
            records: list[dict[str, Any]] = json.load(f)
    else:
        records = []

    records.append(asdict(entry))

    import os
    import tempfile

    fd, tmp = tempfile.mkstemp(dir=lineage_path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(records, f, indent=2)
        os.replace(tmp, lineage_path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_lineage(lineage_path: Path) -> dict[str, LineageEntry]:
    """Load lineage from the JSON file, returning a dict keyed by entry id."""
    if not lineage_path.exists():
        return {}

    with open(lineage_path) as f:
        records: list[dict[str, Any]] = json.load(f)

    return {
        r["id"]: LineageEntry(
            id=r["id"],
            parent=r.get("parent"),
            parents=r.get("parents", []),
            operation=r["operation"],
            generation=r["generation"],
            files_changed=r.get("files_changed", []),
        )
        for r in records
    }


# ---------------------------------------------------------------------------
# Common-ancestor merge selection (GEPA parity)
# ---------------------------------------------------------------------------


def get_ancestors(lineage: dict[str, LineageEntry], cid: str) -> set[str]:
    """Return the set of all ancestor IDs for *cid* (transitive closure)."""
    ancestors: set[str] = set()
    stack = [cid]
    while stack:
        current = stack.pop()
        entry = lineage.get(current)
        if entry is None:
            continue
        for pid in entry.parents:
            if pid and pid not in ancestors:
                ancestors.add(pid)
                stack.append(pid)
        if entry.parent and entry.parent not in ancestors:
            ancestors.add(entry.parent)
            stack.append(entry.parent)
    return ancestors


def find_merge_triplet(
    lineage: dict[str, LineageEntry],
    frontier_ids: list[str],
    frontier_scores: dict[str, float],
    rng: _random.Random | None = None,
    max_attempts: int = 10,
    attempted_pairs: set[tuple[str, str]] | None = None,
    has_val_support_overlap: Callable[[str, str], bool] | None = None,
) -> tuple[str, str, str] | None:
    """Find a (candidate_i, candidate_j, common_ancestor) triplet for merge.

    GEPA merge selection rule: both *i* and *j* must have improved on the
    common ancestor, and neither may be an ancestor of the other.

    GEPA parity (M3): uses random sampling with *max_attempts* retries
    instead of exhaustive O(n²) search.  Mirrors GEPA merge.py:87-90.

    GEPA parity (M4): ancestor selection is weighted random by ancestor
    score, not deterministic best.  Mirrors GEPA merge.py:108-112.

    GEPA parity (L1): improvement filter is non-strict (``>``) matching
    GEPA merge.py:59 — ancestor score must not exceed either candidate.

    GEPA parity (merge-pairing audit B1/B2/C3, /tmp/audit_audit-merge-pairing.md):
    - Canonicalize ``(i, j)`` inside the sampling loop so downstream
      consumers (the merge subprocess, the attempted-pair ledger) always
      see a lex-sorted tuple.  Mirrors GEPA ``merge.py:94-95``
      (``if j < i: i, j = j, i``).
    - Filter already-attempted pairs (``attempted_pairs``) and pairs
      that fail the val-support overlap floor (``has_val_support_overlap``)
      *inside* the retry loop, so a blocked sample triggers resampling
      rather than bailing out of the iteration.  Mirrors GEPA
      ``merge.py:147-148, 199-201`` (inside ``sample_and_attempt_...``).

    Parameters
    ----------
    lineage:
        Full lineage dict (id -> LineageEntry).
    frontier_ids:
        IDs of candidates currently on the Pareto frontier.
    frontier_scores:
        Mapping of candidate id -> aggregate score (for all frontier members
        **and** any ancestors that have known scores).
    rng:
        Seeded RNG instance.  Falls back to ``random.Random(0)`` if None.
    max_attempts:
        Maximum random sampling attempts before giving up.  Defaults to 10
        to match GEPA ``merge.py:76,126`` (``max_attempts=10``).
    attempted_pairs:
        Optional set of canonical ``(i, j)`` tuples (lex-sorted) that have
        already been attempted in earlier iterations.  Sampled pairs matching
        an entry are skipped and resampled within the same call.
    has_val_support_overlap:
        Optional callable; when provided, a candidate pair is skipped when
        ``has_val_support_overlap(i, j)`` returns ``False``.  Mirrors the
        ``has_val_support_overlap`` parameter in
        ``sample_and_attempt_merge_programs_by_common_predictors`` at
        GEPA ``merge.py:125,199-201``.

    Returns
    -------
    tuple[str, str, str] | None
        ``(i, j, common_ancestor)`` on success (with ``i <= j`` lex), or
        ``None`` if no valid triplet exists after ``max_attempts`` tries.
    """
    if len(frontier_ids) < 2:
        return None

    if rng is None:
        rng = _random.Random(0)

    # Pre-compute ancestor sets for each frontier candidate
    ancestor_sets: dict[str, set[str]] = {
        cid: get_ancestors(lineage, cid) for cid in frontier_ids
    }

    for _ in range(max_attempts):
        # GEPA parity (M3): random pair sampling (merge.py:87-90)
        i, j = rng.sample(frontier_ids, 2)
        if i == j:
            continue

        # GEPA parity (merge-pairing audit C3, merge.py:94-95): canonicalize
        # pair so (i, j) and (j, i) land on the same (i, j) tuple for all
        # downstream consumers (merge subprocess arg order, attempted-pair
        # ledger, description-triplet dedup).  HELIX ids are strings, so
        # lex comparison mirrors GEPA's integer ``j < i`` swap.
        if j < i:
            i, j = j, i

        # GEPA parity (merge-pairing audit B2, merge.py:147-148): skip
        # already-attempted pairs inside the retry loop instead of burning
        # the whole propose() call.
        if attempted_pairs is not None and (i, j) in attempted_pairs:
            continue

        # GEPA parity (merge-pairing audit B1, merge.py:199-201): skip
        # pairs with insufficient val-support overlap inside the retry
        # loop.  This lets the next sample win instead of consuming the
        # whole generation on the first unlucky draw.
        if has_val_support_overlap is not None and not has_val_support_overlap(i, j):
            continue

        anc_i = ancestor_sets[i]
        anc_j = ancestor_sets[j]

        # Neither may be an ancestor of the other
        if j in anc_i or i in anc_j:
            continue

        # Find common ancestors
        common = anc_i & anc_j
        if not common:
            continue

        # Filter valid common ancestors
        valid_ancestors: list[str] = []
        for ca in common:
            ca_score = frontier_scores.get(ca)
            if ca_score is None:
                continue
            i_score = frontier_scores.get(i)
            j_score = frontier_scores.get(j)
            if i_score is None or j_score is None:
                continue
            # GEPA parity (L1): non-strict improvement filter.
            # GEPA merge.py:59 uses ``agg_scores[ancestor] > agg_scores[i]``
            # to SKIP — meaning both must be >= ancestor (non-strict).
            if ca_score > i_score or ca_score > j_score:
                continue
            valid_ancestors.append(ca)

        if not valid_ancestors:
            continue

        # GEPA parity (M4): weighted random ancestor selection.
        # Mirrors GEPA merge.py:108-112 — rng.choices() with score weights.
        ancestor_weights = [
            max(frontier_scores.get(ca, 0.0), 1e-9)
            for ca in valid_ancestors
        ]
        selected_ancestor = rng.choices(
            valid_ancestors, weights=ancestor_weights, k=1
        )[0]

        return (i, j, selected_ancestor)

    return None
