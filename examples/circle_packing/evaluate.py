"""
Evaluator for circle packing solution.

Imports solve.pack_circles(26), checks validity constraints, computes total
radius sum as score, and prints one ``HELIX_RESULT=`` line. Exits with code 0
on success.

Score = sum of all radii (no penalty for violations — matching the
GEPA blog: https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)

``side_info`` carries the feedback the *next* mutation actually reasons from
(see the "Evaluator Contract" note in skills/helix/SKILL.md): which circles
overlap and by how much, which escape the unit square and by how much, the
radius spread, and the gap to the best known packing — the kind of thing a
human reviewing a bad arrangement would ask for, derived only from data this
evaluator already computes. It stays bounded regardless of instance size by
listing only the worst few offenders (see MAX_OFFENDERS_LISTED) rather than
enumerating every circle or every pair.
"""
import json
import math
from pathlib import Path

# Best published sum-of-radii for 26 circles in a unit square: matches
# solve_optimized.py's reference score and the figures already quoted in
# README.md / helix.toml (GEPA optimize_anything blog: 2.63598+; AlphaEvolve:
# 2.6358). Used only to report a distance-to-target in side_info -- it does
# not affect the score and is only meaningful for the N_CIRCLES instance
# actually evaluated here.
BEST_KNOWN_SUM_RADII = 2.635982
N_CIRCLES = 26

# Cap how many individual offenders (overlapping pairs, out-of-bounds
# circles) side_info enumerates. Counts and aggregate totals are always
# exact; only the itemized "worst" lists are truncated, so side_info stays
# O(1) in n instead of O(n^2) pairs for a much larger instance -- well
# under the 32 KiB retained-evaluator-output cap (MAX_EVALUATOR_OUTPUT_BYTES
# in src/helix/change_summary.py).
MAX_OFFENDERS_LISTED = 5


def _read_batch_ids() -> list[str]:
    """Read the positional example ids supplied by HELIX."""
    ids = json.loads((Path.cwd() / "helix_batch.json").read_text())
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("helix_batch.json must contain a JSON list[str]")
    return ids


def _bounds_escape(x: float, y: float, r: float) -> float:
    """How far a circle pokes outside [0, 1] x [0, 1]; 0.0 if fully inside."""
    return max(0.0, r - x, r - y, x + r - 1.0, y + r - 1.0)


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
    feedback.append(
        f"Radii range {min(radii):.4f}-{max(radii):.4f} (mean "
        f"{sum_radii / n:.4f}) across {n} circles."
    )
    if n == N_CIRCLES:
        # Round before comparing: the score itself is reported rounded to 6
        # decimals, so a candidate that lands on the published best must not
        # read as "0.0000 below" (true float sum) instead of "reached".
        gap = round(BEST_KNOWN_SUM_RADII - round(sum_radii, 6), 6)
        feedback.append(
            f"Sum of radii is {gap:.4f} below the best known packing for "
            f"{n} circles ({BEST_KNOWN_SUM_RADII}), "
            f"{100.0 * gap / BEST_KNOWN_SUM_RADII:.2f}% short."
            if gap > 1e-9
            else "Sum of radii has reached the best known packing for "
            f"{n} circles ({BEST_KNOWN_SUM_RADII})."
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
