import numpy as np


def pack_circles(n: int) -> list[tuple[float, float, float]]:
    """Pack n circles in unit square to maximize sum of radii.

    Naive seed implementation matching the GEPA blog post:
    https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/

    Uses a concentric-ring heuristic: one center circle, a ring of 8,
    then an outer ring for the remaining circles.  Radii are computed
    greedily so that no two circles overlap and every circle stays
    fully inside the unit square.
    """
    centers = np.zeros((n, 2))

    # Center circle
    centers[0] = [0.5, 0.5]

    # Ring of 8 around center
    for i in range(min(8, n - 1)):
        angle = 2 * np.pi * i / 8
        centers[i + 1] = [0.5 + 0.3 * np.cos(angle), 0.5 + 0.3 * np.sin(angle)]

    # Outer ring for remaining
    if n > 9:
        remaining = n - 9
        for i in range(remaining):
            angle = 2 * np.pi * i / remaining
            centers[i + 9] = [0.5 + 0.7 * np.cos(angle), 0.5 + 0.7 * np.sin(angle)]

    centers = np.clip(centers, 0.01, 0.99)
    radii = _compute_max_radii(centers)
    circles_arr = np.hstack([centers, radii.reshape(-1, 1)])

    return [
        (float(circles_arr[i, 0]), float(circles_arr[i, 1]), float(circles_arr[i, 2]))
        for i in range(n)
    ]


def _compute_max_radii(centers: np.ndarray) -> np.ndarray:
    """Compute maximum non-overlapping radii that stay inside the unit square."""
    n = centers.shape[0]
    radii = np.ones(n)

    # Limit by distance to borders
    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    # Limit by distance to other circles
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if radii[i] + radii[j] > dist:
                scale = dist / (radii[i] + radii[j])
                radii[i] *= scale
                radii[j] *= scale

    return radii


if __name__ == "__main__":
    circles = pack_circles(26)
    print(f"Total radius sum: {sum(r for _, _, r in circles):.6f}")
