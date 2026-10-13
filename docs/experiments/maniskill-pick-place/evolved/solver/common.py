"""Shared helpers for the task routines."""

from __future__ import annotations

from typing import Any


def observe(ctx: Any) -> dict[str, Any]:
    """Convenience accessor for the full observation."""
    return ctx.get_observation()
