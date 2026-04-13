"""Parse JSON stdout to extract a 'score' field as a float metric."""

from __future__ import annotations

import json


def parse(returncode: int, stdout: str, stderr: str) -> tuple[dict[str, float], dict[str, float]]:
    """Extract 'score' from JSON stdout.

    Expects stdout to contain a JSON object with a 'score' key (float).
    Returns scores={"score": value} and instance_scores={"score": value}.

    Falls back to 0.0 on any parse error.
    """
    try:
        data = json.loads(stdout.strip())
        score = float(data.get("score", 0.0))
    except (json.JSONDecodeError, ValueError, TypeError):
        score = 0.0

    # Penalize if process failed
    if returncode != 0:
        score = 0.0

    return {"score": score}, {"score": score}
