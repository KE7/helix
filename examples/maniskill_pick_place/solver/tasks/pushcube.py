"""Naive seed routine for the ``pushcube`` family."""

from __future__ import annotations

from typing import Any

from solver.common import observe


def solve(ctx: Any) -> None:
    obs = observe(ctx)
    cube = obs["objects"]["cube"]["position"]
    goal = obs["goal"]["position"]
    ctx.move_to(cube[0], cube[1], 0.08)
    ctx.move_to(goal[0], goal[1], 0.08)
