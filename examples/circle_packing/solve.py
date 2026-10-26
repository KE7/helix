
import numpy as np

def main(timeout, current_best_solution):
    """
    Circle packing optimization.

    Args:
        timeout: Time budget in seconds
        current_best_solution: Previous best circles array (n, 3) or None

    Returns:
        dict with 'circles' (n, 3) array and 'all_scores' list
    """
    n = 26

    # Use current_best_solution if provided, otherwise start fresh
    if current_best_solution is not None:
        circles = current_best_solution.copy()
    else:
        # Simple initial placement
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
        radii = compute_max_radii(centers)
        circles = np.hstack([centers, radii.reshape(-1, 1)])

    score = float(np.sum(circles[:, 2]))
    return {'circles': circles, 'all_scores': [score]}


def compute_max_radii(centers):
    """Compute maximum radii that don't overlap and stay in unit square."""
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
