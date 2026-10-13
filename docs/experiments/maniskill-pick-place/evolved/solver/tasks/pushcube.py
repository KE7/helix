"""Seed routine for the ``pushcube`` family."""

from __future__ import annotations

import math
from typing import Any

from solver.common import observe


def solve(ctx: Any) -> None:
    obs = observe(ctx)
    cube = obs["objects"]["cube"]
    cube_pos = cube["position"]
    half = cube["half_size"]
    goal = obs["goal"]["position"]
    z = cube_pos[2]

    dx = goal[0] - cube_pos[0]
    dy = goal[1] - cube_pos[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist

    offset = half + 0.04
    start_x = cube_pos[0] - ux * offset
    start_y = cube_pos[1] - uy * offset

    ctx.close_gripper()
    ctx.move_to(start_x, start_y, z + 0.08)
    ctx.move_to(start_x, start_y, z)

    # Push through in small segments, slightly overshooting the goal so the
    # cube's center actually reaches it despite contact offset.
    end_x = goal[0] + ux * half * 0.3
    end_y = goal[1] + uy * half * 0.3
    segments = 6
    for i in range(1, segments + 1):
        t = i / segments
        px = start_x + (end_x - start_x) * t
        py = start_y + (end_y - start_y) * t
        ctx.move_to(px, py, z)
        obs = observe(ctx)
        if obs["steps_left"] <= 0:
            return

    # Retreat straight up and away so the gripper doesn't foul the final check.
    ctx.move_to(end_x, end_y, z + 0.08)
