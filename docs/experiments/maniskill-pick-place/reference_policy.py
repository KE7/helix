"""Scripted reference policy for the maniskill_pick_place tasks.

test_benchmark.py runs it on every train and val instance to check that each
one is solvable with the ctx primitives. It lives outside examples/ so the
mutation agent never sees it.
"""

from __future__ import annotations

import math
from typing import Any

HOVER = 0.08  # approach height above an object (metres)
CUBE_HALF = 0.02


def _grasp_cube(ctx: Any, key: str) -> list[float]:
    """Grasp the cube named *key*, retrying once; return its position."""
    cube = ctx.get_observation()["objects"][key]["position"]
    ctx.open_gripper()
    ctx.move_to(cube[0], cube[1], cube[2] + HOVER)
    ctx.move_to(cube[0], cube[1], cube[2])
    result = ctx.close_gripper()
    if not result.get("grasped"):
        cube = ctx.get_observation()["objects"][key]["position"]
        ctx.open_gripper()
        ctx.move_to(cube[0], cube[1], cube[2])
        ctx.close_gripper()
        cube = ctx.get_observation()["objects"][key]["position"]
    return cube


def _pick(ctx: Any) -> None:
    cube = _grasp_cube(ctx, "cube")
    goal = ctx.get_observation()["goal"]["position"]
    ctx.move_to(cube[0], cube[1], cube[2] + HOVER)
    ctx.move_to(goal[0], goal[1], goal[2])


def _stack(ctx: Any) -> None:
    cube_a = _grasp_cube(ctx, "cube_a")
    ctx.move_to(cube_a[0], cube_a[1], cube_a[2] + HOVER + 0.04)
    cube_b = ctx.get_observation()["objects"]["cube_b"]["position"]
    ctx.move_to(cube_b[0], cube_b[1], cube_b[2] + HOVER + 0.04)
    ctx.move_to(cube_b[0], cube_b[1], cube_b[2] + 2 * CUBE_HALF + 0.002)
    ctx.open_gripper()
    ctx.move_to(cube_b[0], cube_b[1], cube_b[2] + HOVER)


def _push(ctx: Any) -> None:
    """Close the gripper into a pusher and push the cube toward the goal in segments."""
    ctx.close_gripper()
    obs = ctx.get_observation()
    goal = obs["goal"]["position"]
    threshold = obs["goal"]["threshold"]
    cube = obs["objects"]["cube"]["position"]
    dx, dy = goal[0] - cube[0], goal[1] - cube[1]
    dist = math.hypot(dx, dy)
    ux, uy = dx / dist, dy / dist
    # Start behind the cube (opposite side from the goal), then descend.
    ctx.move_to(cube[0] - ux * 0.06, cube[1] - uy * 0.06, cube[2] + HOVER)
    ctx.move_to(cube[0] - ux * 0.06, cube[1] - uy * 0.06, cube[2])
    for _ in range(8):
        obs = ctx.get_observation()
        cube = obs["objects"]["cube"]["position"]
        dx, dy = goal[0] - cube[0], goal[1] - cube[1]
        dist = math.hypot(dx, dy)
        if dist <= threshold * 0.6:
            break
        ux, uy = dx / dist, dy / dist
        step = min(0.06, dist * 0.8)
        ctx.move_to(cube[0] + ux * step, cube[1] + uy * step, cube[2])
    # Retreat so the arm is not resting on the cube at scoring time.
    tcp = ctx.get_observation()["tcp"]["position"]
    ctx.move_to(tcp[0], tcp[1], tcp[2] + HOVER)


def solve(ctx: Any) -> None:
    """Complete one trial using only the four primitives."""
    {"pickcube": _pick, "stackcube": _stack, "pushcube": _push}[ctx.task_family](ctx)
