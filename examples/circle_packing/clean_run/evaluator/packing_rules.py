"""Validity and scoring rules for the n=26 circle-packing task.

Validity follows optimize_anything's circle-packing evaluator
(``validate_packing`` in the CAIS'26 artifact, ``atol=1e-6``).  A packing is a
sequence of exactly ``N_CIRCLES`` rows ``(x, y, r)``; it is valid when every
value is a finite real number (``bool`` is rejected), every radius is
``>= 0`` and, with the absolute tolerance ``TOLERANCE``:

* containment: ``r - x <= tol``, ``r - y <= tol``, ``x + r - 1 <= tol`` and
  ``y + r - 1 <= tol`` for every circle
* no overlap: ``r_i + r_j - hypot(x_i - x_j, y_i - y_j) <= tol`` for every
  pair ``i < j``

An invalid packing scores ``0.0``, as in optimize_anything.

The score of a valid packing is certified: a packing that is valid only up to
the tolerance must not score more than a packing that satisfies every
constraint exactly.  Let ``shrink`` be the largest residual violation,
``max(0, worst containment excess, worst pairwise overlap / 2)``.  Shrinking
every radius by ``shrink`` (and clipping at 0) removes every violation, so the
score is ``fsum(max(0, r - shrink))``.  For a packing with no residual
violation ``shrink`` is 0 and the score is the plain radius sum.  (A circle
clipped to radius 0 contributes nothing and can be moved to any uncovered
point, so the shrunk packing witnesses an exactly valid packing with that
radius sum, up to floating-point rounding of order 1e-16.)

Pure standard library; no randomness, clock or environment input, so the
same packing always receives the same verdict and score.
"""

from __future__ import annotations

import math
from typing import Any

N_CIRCLES = 26
TOLERANCE = 1e-6
MAX_LISTED = 5


def _is_real(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _fmt(value: float) -> str:
    return f"{value:.3e}"


def _invalid(errors: list[str], details: dict[str, Any]) -> dict[str, Any]:
    return {"valid": False, "score": 0.0, "errors": errors, "details": details}


def evaluate_packing(
    circles: Any, n: int = N_CIRCLES, tol: float = TOLERANCE
) -> dict[str, Any]:
    """Return ``{"valid", "score", "errors", "details"}`` for ``circles``.

    ``errors`` is a list of short human-readable strings (bounded in length);
    ``details`` holds counts, worst offenders, the certification shrink and
    radius statistics.  For an invalid packing ``score`` is ``0.0``.
    """
    errors: list[str] = []
    details: dict[str, Any] = {"expected_circles": n, "tolerance": tol}

    if not isinstance(circles, (list, tuple)):
        errors.append(f"packing must be a list of rows, got {type(circles).__name__}")
        return _invalid(errors, details)

    details["actual_circles"] = len(circles)
    if len(circles) != n:
        errors.append(f"expected exactly {n} circles, got {len(circles)}")
        return _invalid(errors, details)

    bad_rows: list[int] = []
    non_numeric: list[int] = []
    non_finite: list[int] = []
    rows: list[tuple[float, float, float]] = []
    for index, row in enumerate(circles):
        if not isinstance(row, (list, tuple)) or len(row) != 3:
            bad_rows.append(index)
            continue
        if not all(_is_real(value) for value in row):
            non_numeric.append(index)
            continue
        try:
            x, y, r = (float(value) for value in row)
        except OverflowError:  # an int too large for a float
            non_finite.append(index)
            continue
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(r)):
            non_finite.append(index)
            continue
        rows.append((x, y, r))
    if bad_rows:
        errors.append(
            f"{len(bad_rows)} row(s) are not [x, y, r] triples "
            f"(first: {bad_rows[:MAX_LISTED]})"
        )
    if non_numeric:
        errors.append(
            f"{len(non_numeric)} row(s) contain non-numeric values "
            f"(first: {non_numeric[:MAX_LISTED]})"
        )
    if non_finite:
        errors.append(
            f"{len(non_finite)} row(s) contain NaN or infinite values "
            f"(first: {non_finite[:MAX_LISTED]})"
        )
    if errors:
        return _invalid(errors, details)

    negative = [i for i, (_, _, r) in enumerate(rows) if r < 0.0]
    if negative:
        errors.append(
            f"{len(negative)} circle(s) have negative radius "
            f"(first: {negative[:MAX_LISTED]})"
        )

    boundary: list[tuple[float, int]] = []
    max_boundary_excess = -math.inf
    for i, (x, y, r) in enumerate(rows):
        excess = max(r - x, r - y, x + r - 1.0, y + r - 1.0)
        max_boundary_excess = max(max_boundary_excess, excess)
        if excess > tol:
            boundary.append((excess, i))
    boundary.sort(reverse=True)

    overlaps: list[tuple[float, int, int]] = []
    max_pair_overlap = -math.inf
    for i in range(n):
        xi, yi, ri = rows[i]
        for j in range(i + 1, n):
            xj, yj, rj = rows[j]
            overlap = ri + rj - math.hypot(xi - xj, yi - yj)
            max_pair_overlap = max(max_pair_overlap, overlap)
            if overlap > tol:
                overlaps.append((overlap, i, j))
    overlaps.sort(reverse=True)

    if boundary:
        errors.append(
            f"{len(boundary)} circle(s) extend outside the unit square by more "
            f"than {_fmt(tol)} (worst: circle {boundary[0][1]} by "
            f"{_fmt(boundary[0][0])})"
        )
    if overlaps:
        errors.append(
            f"{len(overlaps)} pair(s) overlap by more than {_fmt(tol)} "
            f"(worst: circles {overlaps[0][1]} and {overlaps[0][2]} by "
            f"{_fmt(overlaps[0][0])})"
        )

    radii = [r for _, _, r in rows]
    details.update(
        {
            "boundary_violations": len(boundary),
            "worst_boundary_violations": [
                f"circle {i} outside by {_fmt(e)}" for e, i in boundary[:MAX_LISTED]
            ],
            "overlap_violations": len(overlaps),
            "worst_overlaps": [
                f"circles {i} and {j} overlap by {_fmt(e)}"
                for e, i, j in overlaps[:MAX_LISTED]
            ],
            "negative_radii": len(negative),
            "max_boundary_excess": max_boundary_excess,
            "max_pair_overlap": max_pair_overlap,
            "min_radius": min(radii),
            "max_radius": max(radii),
        }
    )

    if errors:
        return _invalid(errors, details)

    shrink = max(0.0, max_boundary_excess, max_pair_overlap / 2.0)
    details["certification_shrink"] = shrink
    details["radius_sum_before_shrink"] = math.fsum(radii)
    score = math.fsum(max(0.0, r - shrink) for r in radii)
    return {"valid": True, "score": score, "errors": errors, "details": details}
