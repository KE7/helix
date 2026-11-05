"""``pickcube`` routine: grasp from above and carry to the 3-D goal."""

from __future__ import annotations

from typing import Any

from solver.common import observe


def solve(ctx: Any) -> None:
    obs = observe(ctx)
    cube = obs["objects"]["cube"]["position"]
    ctx.open_gripper()
    ctx.move_to(cube[0], cube[1], 0.10)
    ctx.move_to(cube[0], cube[1], cube[2])
    ctx.close_gripper()
    goal = obs["goal"]["position"]
    ctx.move_to(cube[0], cube[1], 0.12)
    ctx.move_to(goal[0], goal[1], goal[2])
    # Refine using the observed cube-vs-goal offset.
    obs = observe(ctx)
    c = obs["objects"]["cube"]["position"]
    t = obs["tcp"]["position"]
    ctx.move_to(t[0] + goal[0] - c[0], t[1] + goal[1] - c[1], t[2] + goal[2] - c[2])
