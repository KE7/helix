"""Naive seed routine for the ``stackcube`` family."""

from __future__ import annotations

from typing import Any

from solver.common import observe


def solve(ctx: Any) -> None:
    obs = observe(ctx)
    cube_a = obs["objects"]["cube_a"]["position"]
    ctx.move_to(cube_a[0], cube_a[1], 0.12)
    ctx.close_gripper()
    cube_b = obs["objects"]["cube_b"]["position"]
    ctx.move_to(cube_b[0], cube_b[1], 0.12)
    ctx.open_gripper()
