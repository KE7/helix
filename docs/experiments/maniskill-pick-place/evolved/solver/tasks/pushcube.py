"""``pushcube`` routine: pusher behind the cube, then push through it."""

from __future__ import annotations

import math
from typing import Any

from solver.common import observe

SAFE_Z = 0.10
PUSH_Z = 0.02
BEHIND = 0.06


def solve(ctx: Any) -> None:
    obs = observe(ctx)
    cube = obs["objects"]["cube"]["position"]
    goal = obs["goal"]["position"]
    dx, dy = goal[0] - cube[0], goal[1] - cube[1]
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    ctx.close_gripper()
    bx, by = cube[0] - ux * BEHIND, cube[1] - uy * BEHIND
    ctx.move_to(bx, by, SAFE_Z)
    ctx.move_to(bx, by, PUSH_Z)
    # Push in segments, re-aiming from the observed cube position.
    for _ in range(8):
        obs = observe(ctx)
        cube = obs["objects"]["cube"]["position"]
        rem = math.hypot(goal[0] - cube[0], goal[1] - cube[1])
        if rem < 0.01 or obs["steps_left"] < 20:
            break
        dx, dy = goal[0] - cube[0], goal[1] - cube[1]
        ux, uy = dx / rem, dy / rem
        tcp = obs["tcp"]["position"]
        # TCP should end ~ on the goal (cube center ends at goal).
        tx, ty = goal[0] - ux * 0.0, goal[1] - uy * 0.0
        step = min(rem + 0.02, 0.08)
        ctx.move_to(tcp[0] + ux * step, tcp[1] + uy * step, PUSH_Z)
