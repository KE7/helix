"""Seed routine for the ``pickcube`` family."""

from __future__ import annotations

from typing import Any

from solver.common import observe


def solve(ctx: Any) -> None:
    obs = observe(ctx)
    cube = obs["objects"]["cube"]["position"]
    goal = obs["goal"]["position"]
    safe_z = max(cube[2], goal[2]) + 0.1

    ctx.open_gripper()
    ctx.move_to(cube[0], cube[1], safe_z)
    ctx.move_to(cube[0], cube[1], cube[2])
    ctx.close_gripper()

    obs = observe(ctx)
    if not obs["gripper"]["grasped"]:
        # Re-attempt: nudge down slightly and close again.
        ctx.move_to(cube[0], cube[1], cube[2] - 0.005)
        ctx.close_gripper()

    ctx.move_to(cube[0], cube[1], safe_z)
    ctx.move_to(goal[0], goal[1], safe_z)
    ctx.move_to(goal[0], goal[1], goal[2])
