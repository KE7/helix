"""Parse exit code into a simple success score."""

from __future__ import annotations


def parse(returncode: int, stdout: str, stderr: str) -> tuple[dict[str, float], dict[str, float]]:
    """Return success as both an aggregate and synthetic instance score."""
    success = 1.0 if returncode == 0 else 0.0
    return {"success": success}, {"success": success}
