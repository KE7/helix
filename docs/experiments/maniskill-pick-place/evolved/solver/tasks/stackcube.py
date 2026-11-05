"""``stackcube`` routine: grasp cube A, place it on cube B, release."""

from __future__ import annotations

from typing import Any

from solver.common import observe


def solve(ctx: Any) -> None:
    obs = observe(ctx)
    a = obs["objects"]["cube_a"]["position"]
    ctx.open_gripper()
    ctx.move_to(a[0], a[1], 0.10)
    ctx.move_to(a[0], a[1], a[2])
    ctx.close_gripper()
    ctx.move_to(a[0], a[1], 0.12)
    obs = observe(ctx)
    b = obs["objects"]["cube_b"]["position"]
    ctx.move_to(b[0], b[1], 0.12)
    # Correct for A's offset inside the grip.
    obs = observe(ctx)
    a = obs["objects"]["cube_a"]["position"]
    t = obs["tcp"]["position"]
    ox, oy = a[0] - t[0], a[1] - t[1]
    zt = b[2] + 2 * obs["objects"]["cube_b"]["half_size"] + 0.003
    ctx.move_to(b[0] - ox, b[1] - oy, zt)
    ctx.open_gripper()
    t = observe(ctx)["tcp"]["position"]
    ctx.move_to(t[0], t[1], 0.15)
