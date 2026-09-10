"""
Evaluator for circle packing solution.

Imports solve.pack_circles(26), checks validity constraints, computes total
radius sum as score, and prints one ``HELIX_RESULT=`` line. Exits with code 0
on success.

Score = sum of all radii (no penalty for violations — matching the
GEPA blog: https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)

``side_info`` carries the feedback the *next* mutation actually reasons from
(see the "Evaluator Contract" note in skills/helix/SKILL.md): which circles
overlap and by how much, which escape the unit square and by how much, radius
spread, which circle has the most room left to grow and what stops it, and
where the largest remaining empty gap is — the kind of thing a human
reviewing an arrangement would point at, derived only from data this
evaluator already computes about THIS candidate. None of it is a target to
match: every measure keeps producing a concrete next move no matter how good
the packing gets, because it describes the candidate's own geometry rather
than a distance to an external number. It stays bounded regardless of
instance size by listing only the worst few offenders (see
MAX_OFFENDERS_LISTED) rather than enumerating every circle or every pair.
"""
import json
import math
from pathlib import Path

N_CIRCLES = 26

# Cap how many individual offenders (overlapping pairs, out-of-bounds
# circles) side_info enumerates. Counts and aggregate totals are always
# exact; only the itemized "worst" lists are truncated, so side_info stays
# O(1) in n instead of O(n^2) pairs for a much larger instance -- well
# under the retained-evaluator-output cap
# (src/helix/change_summary.py::MAX_EVALUATOR_OUTPUT_CHARS), above which the
# output is shortened with a note in the text.
MAX_OFFENDERS_LISTED = 5

# Resolution of the grid search used to locate the largest still-empty gap
# (see _largest_gap). Fixed and independent of n, so cost stays O(GRID^2)
# instead of growing with instance size.
GRID_STEPS = 40


def _read_batch_ids() -> list[str]:
    """Read the positional example ids supplied by HELIX."""
    ids = json.loads((Path.cwd() / "helix_batch.json").read_text())
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("helix_batch.json must contain a JSON list[str]")
    return ids


def _bounds_escape(x: float, y: float, r: float) -> float:
    """How far a circle pokes outside [0, 1] x [0, 1]; 0.0 if fully inside."""
    return max(0.0, r - x, r - y, x + r - 1.0, y + r - 1.0)


def _wall_gap(x: float, y: float, r: float) -> tuple[float, str]:
    """Distance from a circle's edge to its nearest wall, and which wall."""
    gaps = {"left": x - r, "right": 1.0 - x - r, "bottom": y - r, "top": 1.0 - y - r}
    side = min(gaps, key=lambda k: gaps[k])
    return gaps[side], side


def _headroom_stats(circles: list[tuple[float, float, float]]) -> list[dict]:
    """How much each circle could grow before it touches something.

    For every circle, take the smaller of (a) its clearance to the nearest
    wall and (b) its clearance to its nearest neighbour. That is exactly how
    much its radius could increase in isolation before something has to
    give. Clamped at 0 for a circle that already overlaps or escapes the
    square -- that circle needs to move before it can grow, which the
    bounds/overlap violations already say. This is unbounded in the useful
    direction: it is strictly positive for any candidate that is not
    already a local optimum, so it never runs dry the way a fixed target
    would.
    """
    n = len(circles)
    stats = []
    for i, (x, y, r) in enumerate(circles):
        wall_val, wall_side = _wall_gap(x, y, r)
        nb_val = math.inf
        nb_idx = None
        for j, (xj, yj, rj) in enumerate(circles):
            if j == i:
                continue
            gap = math.hypot(x - xj, y - yj) - r - rj
            if gap < nb_val:
                nb_val, nb_idx = gap, j
        if n == 1 or wall_val <= nb_val:
            binder = f"the {wall_side} wall"
        else:
            binder = f"circle {nb_idx}"
        stats.append(
            {
                "circle": i,
                "headroom": max(0.0, min(wall_val, nb_val)),
                "binder": binder,
            }
        )
    return stats


def _largest_gap(
    circles: list[tuple[float, float, float]],
) -> tuple[float, float, float]:
    """Approximate radius and location of the largest still-empty region.

    Grid search over the unit square: at each point, the largest circle
    that could be centered there without touching a wall or an existing
    circle. Fixed grid resolution (GRID_STEPS), so cost does not grow with
    n. Like headroom, this stays positive for any candidate short of a
    perfect covering, so it keeps giving the agent somewhere to push
    circles toward instead of going silent once some milestone is passed.
    """
    best_r, best_x, best_y = -math.inf, 0.5, 0.5
    for gi in range(GRID_STEPS + 1):
        gx = gi / GRID_STEPS
        for gj in range(GRID_STEPS + 1):
            gy = gj / GRID_STEPS
            clearance = min(gx, gy, 1.0 - gx, 1.0 - gy)
            for x, y, r in circles:
                if clearance <= 0.0:
                    break
                d = math.hypot(gx - x, gy - y) - r
                if d < clearance:
                    clearance = d
            if clearance > best_r:
                best_r, best_x, best_y = clearance, gx, gy
    return max(0.0, best_r), best_x, best_y


def _analyze(circles: list[tuple[float, float, float]]) -> dict:
    """Derive actionable, bounded feedback from a packing.

    Everything here comes from data the evaluator already has after computing
    the score -- no extra work, no invented facts. An agent should be able to
    act on the returned dict (and its "feedback" summary) without ever seeing
    the numeric score.
    """
    n = len(circles)
    if n == 0:
        return {"feedback": "pack_circles returned no circles."}

    bounds_offenders = []
    for i, (x, y, r) in enumerate(circles):
        escape = _bounds_escape(x, y, r)
        if escape > 1e-9:
            bounds_offenders.append({"circle": i, "escapes_by": round(escape, 6)})
    bounds_offenders.sort(key=lambda o: -o["escapes_by"])

    overlap_offenders = []
    total_overlap = 0.0
    for i in range(n):
        x_i, y_i, r_i = circles[i]
        for j in range(i + 1, n):
            x_j, y_j, r_j = circles[j]
            dist = math.hypot(x_i - x_j, y_i - y_j)
            overlap = r_i + r_j - dist
            if overlap > 1e-9:  # same tolerance as the validity check below
                total_overlap += overlap
                overlap_offenders.append(
                    {"circles": [i, j], "overlap_by": round(overlap, 6)}
                )
    overlap_offenders.sort(key=lambda o: -o["overlap_by"])

    radii = [r for _, _, r in circles]
    sum_radii = sum(radii)

    feedback = []
    if not bounds_offenders and not overlap_offenders:
        feedback.append(f"All {n} circles are valid: inside the square, no overlaps.")
    else:
        if bounds_offenders:
            worst = bounds_offenders[0]
            feedback.append(
                f"{len(bounds_offenders)} circle(s) extend outside the unit square "
                f"(worst: circle {worst['circle']} by {worst['escapes_by']:.4f}) -- "
                "shrink or reposition those before anything else."
            )
        if overlap_offenders:
            worst = overlap_offenders[0]
            feedback.append(
                f"{len(overlap_offenders)} pair(s) overlap, total overlap depth "
                f"{total_overlap:.4f} (worst: circles {worst['circles']} by "
                f"{worst['overlap_by']:.4f}) -- push those centers apart or shrink "
                "one of each pair."
            )
    min_r, max_r = min(radii), max(radii)
    spread = f"{max_r / min_r:.2f}" if min_r > 1e-9 else "inf"
    feedback.append(
        f"Radii range {min_r:.4f}-{max_r:.4f} (mean {sum_radii / n:.4f}, "
        f"spread ratio {spread}) across {n} circles."
    )

    headroom = _headroom_stats(circles)
    total_headroom = sum(h["headroom"] for h in headroom)
    loosest = max(headroom, key=lambda h: h["headroom"])
    feedback.append(
        f"Circle {loosest['circle']} could grow {loosest['headroom']:.4f} "
        f"before {loosest['binder']} (total headroom {total_headroom:.4f})."
    )

    gap_r, gap_x, gap_y = _largest_gap(circles)
    feedback.append(
        f"Largest empty gap: r{gap_r:.4f} near ({gap_x:.2f}, {gap_y:.2f})."
    )

    # Render the itemized "worst" lists as plain strings rather than nested
    # dicts: HELIX's diagnostics renderer (mutator._render_side_info_value,
    # GEPA format_samples parity) turns a list of dicts into two more levels
    # of markdown headers per item, which buries the numbers the mutator
    # actually needs. A list of short strings renders as one flat "Item N"
    # per offender instead.
    bounds_worst = [
        f"circle {o['circle']} escapes the square by {o['escapes_by']:.4f}"
        for o in bounds_offenders[:MAX_OFFENDERS_LISTED]
    ]
    overlap_worst = [
        f"circles {o['circles'][0]} & {o['circles'][1]} overlap by {o['overlap_by']:.4f}"
        for o in overlap_offenders[:MAX_OFFENDERS_LISTED]
    ]

    return {
        "feedback": " ".join(feedback),
        "bounds_violations": {
            "count": len(bounds_offenders),
            "worst": bounds_worst,
        },
        "overlap_violations": {
            "count": len(overlap_offenders),
            "total_overlap_depth": round(total_overlap, 6),
            "worst": overlap_worst,
        },
        "radius_stats": {
            "n_circles": n,
            "min": round(min(radii), 6),
            "max": round(max(radii), 6),
            "mean": round(sum_radii / n, 6),
        },
    }


def _emit(score: float, *, side_info_extra: dict) -> None:
    side_info = {"scores": {"sum_radii": score}, **side_info_extra}
    payload = [[score, side_info] for _ in _read_batch_ids()]
    print("HELIX_RESULT=" + json.dumps(payload))


def evaluate():
    try:
        import solve
    except Exception as e:
        _emit(
            0.0,
            side_info_extra={
                "feedback": f"ERROR: Could not import solve.py: {type(e).__name__}: {e}",
            },
        )
        return

    try:
        circles = solve.pack_circles(N_CIRCLES)
    except Exception as e:
        _emit(
            0.0,
            side_info_extra={
                "feedback": (
                    f"ERROR: pack_circles({N_CIRCLES}) raised "
                    f"{type(e).__name__}: {e}"
                ),
            },
        )
        return

    # Score = sum of all radii (no violation penalty, matching GEPA blog)
    score = round(sum(r for _, _, r in circles), 6)
    _emit(score, side_info_extra=_analyze(circles))


if __name__ == "__main__":
    evaluate()
