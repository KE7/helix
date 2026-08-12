"""
Evaluator for circle packing solution.

Imports solve.pack_circles(26), checks validity constraints,
computes total radius sum as score, and prints one ``HELIX_RESULT=`` line.
Exits with code 0 on success.

Score = sum of all radii (no penalty for violations — matching the
GEPA blog: https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/)
"""
import json
import math
from pathlib import Path


def _read_batch_ids() -> list[str]:
    """Read the positional example ids supplied by HELIX."""
    ids = json.loads((Path.cwd() / "helix_batch.json").read_text())
    if not isinstance(ids, list) or not all(isinstance(item, str) for item in ids):
        raise ValueError("helix_batch.json must contain a JSON list[str]")
    return ids


def _emit(score: float, *, violations: int, arrangement: str) -> None:
    side_info = {
        "scores": {"sum_radii": score},
        "violations": violations,
        "arrangement": arrangement,
    }
    payload = [[score, side_info] for _ in _read_batch_ids()]
    print("HELIX_RESULT=" + json.dumps(payload))


def evaluate():
    try:
        import solve
    except Exception as e:
        _emit(
            0.0,
            violations=1,
            arrangement=f"ERROR: Could not import solve.py: {type(e).__name__}: {e}",
        )
        return

    try:
        circles = solve.pack_circles(26)
    except Exception as e:
        _emit(0.0, violations=1, arrangement=f"ERROR: pack_circles(26) raised: {e}")
        return

    violations = 0

    # Check bounds: each circle must be fully within the unit square
    for i, (x, y, r) in enumerate(circles):
        if not (r <= x <= 1 - r and r <= y <= 1 - r):
            violations += 1

    # Check non-overlap: distance between any two centers >= r_i + r_j
    for i in range(len(circles)):
        x_i, y_i, r_i = circles[i]
        for j in range(i + 1, len(circles)):
            x_j, y_j, r_j = circles[j]
            dist = math.sqrt((x_i - x_j) ** 2 + (y_i - y_j) ** 2)
            if dist < r_i + r_j - 1e-9:  # small tolerance for floating point
                violations += 1

    # Score = sum of all radii (no violation penalty, matching GEPA blog)
    score = sum(r for _, _, r in circles)

    n = len(circles)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    avg_r = score / n if n > 0 else 0.0
    arrangement = (
        f"{n} circles in {cols}x{rows} grid, avg_r={avg_r:.4f}, "
        f"violations={violations}"
    )

    _emit(round(score, 6), violations=violations, arrangement=arrangement)


if __name__ == "__main__":
    evaluate()
