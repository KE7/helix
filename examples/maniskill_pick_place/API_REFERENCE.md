# Solver runtime API

The benchmark calls `solver.run(ctx)` once per trial. `ctx` is the only
interface to the world; do not import ManiSkill, gymnasium, or the `sim`
package.

## Context fields

| Field | Type | Meaning |
|---|---|---|
| `ctx.task_family` | `str` | `"pushcube"`, `"pickcube"`, or `"stackcube"` |
| `ctx.instruction` | `str` | The natural-language task statement |
| `ctx.instance_id` | `str` | e.g. `"pickcube__3"` |

## Primitives

### `ctx.get_observation() -> dict`

```python
{
  "tcp": {"position": [x, y, z]},        # tool-center-point (between fingertips)
  "gripper": {"command": "open"|"closed", "grasped": bool},
  "objects": {                            # per family:
    "cube":   {"position": [x, y, z], "half_size": 0.02},   # pushcube, pickcube
    # or
    "cube_a": {"position": [...], "half_size": 0.02},        # stackcube
    "cube_b": {"position": [...], "half_size": 0.02},
  },
  "goal": {"position": [x, y, z], "threshold": r},           # pushcube, pickcube
  # or     {"description": "cube A resting on top of cube B, gripper released"},
  "workspace": {"x_range": [...], "y_range": [...], "z_range": [...]},
  "steps_left": int,
}
```

`grasped` is the environment's grasp check (for stackcube, whether cube A is
grasped). It is always `False` for pushcube.

### `ctx.move_to(x, y, z) -> dict`

Closed-loop Cartesian move of the TCP (orientation is held fixed, pointing
down). Requests are clamped to the workspace. Returns
`{"status": "converged" | "timeout", "position": [...], "requested": [...]}`,
where `requested` is the clamped target. Each call steps until the TCP is
within 0.01 m of the target (`converged`) or 120 control steps have passed
(`timeout`).

### `ctx.close_gripper() / ctx.open_gripper() -> dict`

Close returns `{"status": "closed", "grasped": bool}`; open returns
`{"status": "open"}`. Each call spends 12 control steps with the arm held in
place. The gripper holds its commanded state during subsequent moves.

## Budget and scoring

- Each trial has a 400-control-step budget (`steps_left` in the observation).
  `move_to`, `open_gripper`, and `close_gripper` spend it; `get_observation`
  does not. A call that runs out of budget partway returns
  `{"status": "budget_exhausted", "position": [...]}`. Once `steps_left` is 0,
  the next call to any of the three ends `solver.run`, and the trial is scored
  as it stands.
- A trial scores 1.0 if the environment's success check passes on the final
  state; otherwise it scores the best shaped reward reached, capped at 0.95.
  Success means:
  - pushcube: the cube is within `threshold` of the goal in xy and is still
    on the table.
  - pickcube: the cube is within `threshold` of the 3-D goal and the robot is
    still.
  - stackcube: cube A is centered on cube B within about 3.3 cm, rests on its
    top within 5 mm, is still, and is not grasped.
- After `solver.run` returns, the benchmark holds the arm still for up to 40
  more control steps and then judges, so end in the state you want judged.
  These steps do not reduce `steps_left`, but the `steps_used` diagnostic
  includes them.
- `print()` output from solver code is discarded during evaluation.
