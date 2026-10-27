"""Does a circle-packing score depend on validate_packing's atol=1e-6?

For each packing: the largest pairwise overlap and boundary excess it uses,
and the best radius sum at the same centers with zero tolerance (an LP:
maximize sum r subject to r_i + r_j <= d_ij and r_i <= distance to each wall).
"""

import json
import re
import sys
from itertools import combinations

import numpy as np
from scipy.optimize import linprog


def parse(solution: object) -> np.ndarray:
    if isinstance(solution, str):
        nums = [float(x) for x in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", solution)]
        return np.array(nums).reshape(-1, 3)
    return np.asarray(solution, dtype=float).reshape(-1, 3)


def report(name: str, circles: np.ndarray, logged: float) -> dict:
    c, r = circles[:, :2], circles[:, 2]
    n = len(r)
    overlaps = [
        r[i] + r[j] - np.linalg.norm(c[i] - c[j]) for i, j in combinations(range(n), 2)
    ]
    walls = np.concatenate([r - c[:, 0], c[:, 0] + r - 1, r - c[:, 1], c[:, 1] + r - 1])
    pairs = list(combinations(range(n), 2))
    a_ub, b_ub = [], []
    for i, j in pairs:
        row = np.zeros(n)
        row[i] = row[j] = 1
        a_ub.append(row)
        b_ub.append(np.linalg.norm(c[i] - c[j]))
    wall_dist = np.minimum.reduce([c[:, 0], 1 - c[:, 0], c[:, 1], 1 - c[:, 1]])
    # HiGHS's default feasibility tolerance (1e-7) is larger than the overlaps
    # being measured, so tighten it and then repair any residual violation.
    lp = linprog(
        -np.ones(n),
        A_ub=np.array(a_ub),
        b_ub=np.array(b_ub),
        bounds=[(0, w) for w in wall_dist],
        method="highs",
        options={
            "primal_feasibility_tolerance": 1e-10,
            "dual_feasibility_tolerance": 1e-10,
        },
    )
    out = {
        "name": name,
        "logged_score": logged,
        "sum_radii": float(r.sum()),
        "max_pair_overlap": float(max(overlaps)),
        "pairs_overlapping": int(sum(o > 0 for o in overlaps)),
        "max_wall_excess": float(walls.max()),
        "circles_past_wall": int(sum(w > 0 for w in walls)),
        "strict_lp_sum_same_centers": float(strict(c, lp.x).sum()),
        "strict_shrink_sum_same_radii": float(strict(c, r).sum()),
    }
    out["tolerance_gain_vs_lp"] = out["sum_radii"] - out["strict_lp_sum_same_centers"]
    return out


def strict(c: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Shrink each circle by half its worst overlap (and any wall excess): exactly feasible."""
    n = len(r)
    delta = np.zeros(n)
    for i, j in combinations(range(n), 2):
        v = r[i] + r[j] - np.linalg.norm(c[i] - c[j])
        if v > 0:
            delta[i] = max(delta[i], v / 2)
            delta[j] = max(delta[j], v / 2)
    wall = np.minimum.reduce([c[:, 0], 1 - c[:, 0], c[:, 1], 1 - c[:, 1]])
    out = np.minimum(r - delta, wall)
    for i, j in combinations(range(n), 2):
        assert out[i] + out[j] <= np.linalg.norm(c[i] - c[j]) + 1e-15
    return out


if __name__ == "__main__":
    results = []
    for name, path in zip(sys.argv[1::2], sys.argv[2::2]):
        last = json.load(open(path))[-1]
        results.append(report(name, parse(last["best_solution"]), last["best_score"]))
    print(json.dumps(results, indent=2))
