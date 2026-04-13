"""Parse JSON stdout with 'accuracy' field into scores and instance_scores."""

from __future__ import annotations

import json


def parse(returncode: int, stdout: str, stderr: str) -> tuple[dict[str, float], dict[str, float]]:
    """Parse JSON output containing accuracy and instance_scores.

    Expects stdout to be a JSON object with:
        - 'accuracy': float in [0, 1]
        - 'instance_scores': dict mapping puzzle id to 1.0 or 0.0 (optional)

    Returns:
        scores: dict with "accuracy" key
        instance_scores: dict mapping puzzle id to score
    """
    instance_scores: dict[str, float] = {}
    accuracy = 0.0

    try:
        # Try full stdout as JSON first (handles pretty-printed multi-line JSON)
        data = None
        try:
            data = json.loads(stdout.strip())
        except (json.JSONDecodeError, ValueError):
            # Fallback: find the last complete JSON object line in stdout
            lines = stdout.strip().splitlines()
            for line in reversed(lines):
                line = line.strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        break
                    except json.JSONDecodeError:
                        continue
        if data is None:
            data = {}
        accuracy = float(data.get("accuracy", 0.0))
        raw_instances = data.get("instance_scores", {})
        if isinstance(raw_instances, dict):
            for k, v in raw_instances.items():
                try:
                    instance_scores[str(k)] = float(v)
                except (TypeError, ValueError):
                    instance_scores[str(k)] = 0.0
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    scores: dict[str, float] = {"accuracy": accuracy}
    return scores, instance_scores
