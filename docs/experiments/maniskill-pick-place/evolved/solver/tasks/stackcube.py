"""Seed routine for the ``stackcube`` family."""

from __future__ import annotations

from typing import Any

from solver.common import observe


def solve(ctx: Any) -> None:
    obs = observe(ctx)
    cube_a = obs["objects"]["cube_a"]
    cube_b = obs["objects"]["cube_b"]
    a_pos = cube_a["position"]
    b_pos = cube_b["position"]
    half_a = cube_a["half_size"]
    half_b = cube_b["half_size"]
    safe_z = max(a_pos[2], b_pos[2]) + 0.1
    place_z = b_pos[2] + half_a + half_b

    ctx.open_gripper()
    ctx.move_to(a_pos[0], a_pos[1], safe_z)
    ctx.move_to(a_pos[0], a_pos[1], a_pos[2])
    ctx.close_gripper()

    obs = observe(ctx)
    if not obs["gripper"]["grasped"]:
        ctx.move_to(a_pos[0], a_pos[1], a_pos[2] - 0.005)
        ctx.close_gripper()

    ctx.move_to(a_pos[0], a_pos[1], safe_z)
    ctx.move_to(b_pos[0], b_pos[1], safe_z)
    ctx.move_to(b_pos[0], b_pos[1], place_z)
    ctx.open_gripper()
    ctx.move_to(b_pos[0], b_pos[1], safe_z)
