"""Local validity checker for a circle packing.  It computes no score.

Run ``python3 check_packing.py`` to call ``solve.pack_circles(26)`` and check
the result, or import ``violations`` / ``residual_violation`` and pass them a
list of ``(x, y, r)`` rows.

A packing is valid when it has exactly 26 rows of three finite real numbers,
every radius is >= 0, and, with an absolute tolerance of 1e-6:

* ``r - x <= 1e-6``, ``r - y <= 1e-6``, ``x + r - 1 <= 1e-6``, ``y + r - 1 <= 1e-6``
* ``r_i + r_j - hypot(x_i - x_j, y_i - y_j) <= 1e-6`` for every pair ``i < j``

For a valid packing, ``residual_violation`` returns
``max(0, largest containment excess, largest pairwise overlap / 2)``; the
evaluator subtracts that amount from every radius (clipping at 0) before it
sums them.  A packing with no residual violation loses nothing.

This mirrors the rule the evaluator applies; the evaluator's own copy is the
one that counts.
"""

from __future__ import annotations

import math
import sys
from typing import Any

N_CIRCLES = 26
TOLERANCE = 1e-6


def _rows(circles: Any, n: int) -> tuple[list[tuple[float, float, float]], list[str]]:
    if not isinstance(circles, (list, tuple)):
        return [], [f"packing must be a list of rows, got {type(circles).__name__}"]
    if len(circles) != n:
        return [], [f"expected exactly {n} circles, got {len(circles)}"]
    problems: list[str] = []
    rows: list[tuple[float, float, float]] = []
    for index, row in enumerate(circles):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            problems.append(f"row {index} is not an (x, y, r) triple")
            continue
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in row):
            problems.append(f"row {index} contains a non-numeric value")
            continue
        try:
            x, y, r = (float(v) for v in row)
        except OverflowError:
            problems.append(f"row {index} contains a value too large for a float")
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            problems.append(f"row {index} contains NaN or an infinite value")
            continue
        rows.append((x, y, r))
    return rows, problems


def _worst(rows: list[tuple[float, float, float]]) -> tuple[float, float]:
    boundary = max(max(r - x, r - y, x + r - 1.0, y + r - 1.0) for x, y, r in rows)
    overlap = -math.inf
    for i, (xi, yi, ri) in enumerate(rows):
        for xj, yj, rj in rows[i + 1 :]:
            overlap = max(overlap, ri + rj - math.hypot(xi - xj, yi - yj))
    return boundary, overlap


def violations(circles: Any, n: int = N_CIRCLES, tol: float = TOLERANCE) -> list[str]:
    """Return a list of rule violations; an empty list means the packing is valid."""
    rows, problems = _rows(circles, n)
    if problems:
        return problems
    for i, (x, y, r) in enumerate(rows):
        if r < 0.0:
            problems.append(f"circle {i} has negative radius {r!r}")
        excess = max(r - x, r - y, x + r - 1.0, y + r - 1.0)
        if excess > tol:
            problems.append(f"circle {i} extends outside the unit square by {excess:.3e}")
    for i in range(n):
        xi, yi, ri = rows[i]
        for j in range(i + 1, n):
            xj, yj, rj = rows[j]
            overlap = ri + rj - math.hypot(xi - xj, yi - yj)
            if overlap > tol:
                problems.append(f"circles {i} and {j} overlap by {overlap:.3e}")
    return problems


def residual_violation(circles: Any, n: int = N_CIRCLES) -> float:
    """Amount the evaluator subtracts from every radius of a valid packing."""
    rows, problems = _rows(circles, n)
    if problems:
        raise ValueError("; ".join(problems))
    boundary, overlap = _worst(rows)
    return max(0.0, boundary, overlap / 2.0)


def _as_rows(circles: Any) -> Any:
    """Convert numpy arrays and other sequences to nested lists of floats."""
    try:
        return [[float(v) if not isinstance(v, bool) else v for v in row] for row in circles]
    except (TypeError, ValueError):
        return circles


def main() -> int:
    import solve

    packing = _as_rows(solve.pack_circles(N_CIRCLES))
    problems = violations(packing)
    if problems:
        print(f"INVALID: {len(problems)} problem(s)")
        for line in problems[:50]:
            print(f"  - {line}")
        return 1
    shrink = residual_violation(packing)
    if shrink > 0.0:
        print(f"VALID (residual violation {shrink:.3e} is subtracted from every radius)")
    else:
        print("VALID")
    return 0


if __name__ == "__main__":
    sys.exit(main())
