"""Independent validity re-check of a circle packing in exact rational arithmetic.

Shares no code with the evaluator.  Every float coordinate is converted to the
exact rational it represents (``fractions.Fraction``), so no rounding can hide
a violation:

* containment: ``r <= x``, ``r <= y``, ``x + r <= 1``, ``y + r <= 1``
* disjointness: ``(x_i - x_j)^2 + (y_i - y_j)^2 >= (r_i + r_j)^2``

It reports whether the packing is exactly valid (tolerance 0), whether it is
valid at a given absolute tolerance (default 1e-6, the rule the evaluator
applies), the largest violations, and a certified radius sum: the sum after
shrinking every radius by the smallest uniform amount (found by bisection over
exact checks) that makes every circle with positive radius satisfy every
constraint exactly.  (A circle shrunk to radius 0 contributes nothing and can
be moved to any uncovered point.)  Given ``--claimed-score``, it also checks
that the claimed score does not exceed that certified sum.

Usage: ``python3 independent_check.py packing.json [--tolerance 1e-6]
[--claimed-score S]`` where ``packing.json`` holds a list of ``[x, y, r]``
rows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from fractions import Fraction
from typing import Any

N_CIRCLES = 26


def _rows(circles: Any, n: int) -> list[tuple[Fraction, Fraction, Fraction]]:
    if not isinstance(circles, list) or len(circles) != n:
        raise ValueError(f"expected a list of {n} rows")
    rows = []
    for row in circles:
        if not isinstance(row, list) or len(row) != 3:
            raise ValueError("every row must be [x, y, r]")
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("coordinates must be numbers")
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("coordinates must be finite")
        rows.append(tuple(Fraction(v) for v in row))  # type: ignore[arg-type]
    return rows  # type: ignore[return-value]


def _exactly_valid(
    rows: list[tuple[Fraction, Fraction, Fraction]], *, skip_zero: bool = False
) -> bool:
    if skip_zero:
        rows = [row for row in rows if row[2] != 0]
    for x, y, r in rows:
        if r < 0 or r > x or r > y or x + r > 1 or y + r > 1:
            return False
    for i in range(len(rows)):
        xi, yi, ri = rows[i]
        for j in range(i + 1, len(rows)):
            xj, yj, rj = rows[j]
            if (xi - xj) ** 2 + (yi - yj) ** 2 < (ri + rj) ** 2:
                return False
    return True


def _shrunk(
    rows: list[tuple[Fraction, Fraction, Fraction]], delta: Fraction
) -> list[tuple[Fraction, Fraction, Fraction]]:
    return [(x, y, max(Fraction(0), r - delta)) for x, y, r in rows]


def check(
    circles: Any,
    tolerance: float = 1e-6,
    n: int = N_CIRCLES,
    claimed_score: float | None = None,
) -> dict[str, Any]:
    rows = _rows(circles, n)
    tol = Fraction(tolerance)
    negative = sum(1 for _, _, r in rows if r < 0)
    worst_boundary = max(max(r - x, r - y, x + r - 1, y + r - 1) for x, y, r in rows)
    # Pair overlap r_i + r_j - d_ij, bounded exactly: d is irrational in
    # general, so compare squares.  An overlap exceeds t iff
    # r_i + r_j - t > 0 and (r_i + r_j - t)^2 > d^2.
    worst_overlap_estimate = -math.inf
    pairs_over_tol = 0
    pairs_over_zero = 0
    for i in range(n):
        xi, yi, ri = rows[i]
        for j in range(i + 1, n):
            xj, yj, rj = rows[j]
            d2 = (xi - xj) ** 2 + (yi - yj) ** 2
            s = ri + rj
            if s > 0 and s * s > d2:
                pairs_over_zero += 1
            if s - tol > 0 and (s - tol) ** 2 > d2:
                pairs_over_tol += 1
            worst_overlap_estimate = max(
                worst_overlap_estimate, float(s) - math.sqrt(float(d2))
            )
    boundary_over_tol = sum(
        1 for x, y, r in rows if max(r - x, r - y, x + r - 1, y + r - 1) > tol
    )
    exactly_valid = negative == 0 and _exactly_valid(rows)
    radius_sum = sum((r for _, _, r in rows), Fraction(0))

    # Smallest uniform shrink that makes the packing exactly valid, bisected
    # on exact checks; the returned delta is always one that passed.
    if exactly_valid:
        delta = Fraction(0)
    elif negative:
        delta = None
    else:
        hi = Fraction(1, 2)
        if not _exactly_valid(_shrunk(rows, hi), skip_zero=True):
            delta = None
        else:
            lo = Fraction(0)
            for _ in range(80):
                mid = (lo + hi) / 2
                if _exactly_valid(_shrunk(rows, mid), skip_zero=True):
                    hi = mid
                else:
                    lo = mid
            delta = hi
    certified = None
    certified_exact: Fraction | None = None
    if delta is not None:
        certified_exact = sum((r for _, _, r in _shrunk(rows, delta)), Fraction(0))
        certified = float(certified_exact)
    claim: dict[str, Any] = {}
    if claimed_score is not None:
        claim = {
            "claimed_score": claimed_score,
            # Bisection stops within 2**-81 of the minimal shrink; allow that
            # slack times n plus double rounding of the claimed value.
            "claimed_score_within_certified_sum": certified_exact is not None
            and Fraction(claimed_score)
            <= certified_exact + Fraction(n, 2**80) + Fraction(1, 10**12),
        }
    return {
        "n": n,
        "radius_sum": float(radius_sum),
        "exactly_valid": exactly_valid,
        "valid_at_tolerance": negative == 0
        and boundary_over_tol == 0
        and pairs_over_tol == 0,
        "tolerance": tolerance,
        "negative_radii": negative,
        "boundary_violations_over_tolerance": boundary_over_tol,
        "overlapping_pairs_over_tolerance": pairs_over_tol,
        "overlapping_pairs_exact": pairs_over_zero,
        "max_boundary_excess": float(worst_boundary),
        "max_pair_overlap_approx": worst_overlap_estimate,
        "uniform_shrink_for_exact_validity": None if delta is None else float(delta),
        "certified_exactly_valid_radius_sum": certified,
        **claim,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packing")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--claimed-score", type=float, default=None)
    args = parser.parse_args()
    with open(args.packing, encoding="utf-8") as handle:
        circles = json.load(handle)
    print(json.dumps(check(circles, args.tolerance, claimed_score=args.claimed_score), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
