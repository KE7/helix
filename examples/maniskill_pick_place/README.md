# maniskill_pick_place

## Problem

Tabletop manipulation on [ManiSkill 3](https://maniskill.ai) with a Franka
Panda. HELIX evolves a multi-file `solver/` package (a task router plus one
routine per task family) built on four primitives: `get_observation`,
`move_to`, `open_gripper`, and `close_gripper`. Trials are scored by each
environment's shaped reward and success check.

Instance ids are `<family>__<trial>`. The id seeds `env.reset`, so
`pickcube__3` starts from the same state on every run.

| Family | ManiSkill env | Task |
|---|---|---|
| `pushcube`  | `PushCube-v1`  | push the cube into the goal region |
| `pickcube`  | `PickCube-v1`  | grasp the cube and hold it at a 3-D goal point |
| `stackcube` | `StackCube-v1` | stack cube A on cube B and release it |

The success checks are listed in `API_REFERENCE.md`. There are 9 train trials
(`*__0..2`) and 9 validation trials (`*__3..5`); the ids are the file stems
under `splits/`.

## How to run

Copy the example to a standalone directory, then run HELIX there:

```bash
work_dir="$(mktemp -d)/maniskill_pick_place"
cp -R "$(git rev-parse --show-toplevel)/examples/maniskill_pick_place" "$work_dir"
cd "$work_dir"
helix evolve
```

`helix evolve` initializes the copy as a git repository and evolves `solver/`
against `evaluate.py` using `helix.toml`. This setup has no sandbox: the agent
runs on your machine, can read the whole copy, including `evaluate.py` and
`sim/`, and can use the network.

To score the seed `solver/` by hand without writing into the repository, copy
the example and run the evaluator there:

```bash
tmp_dir="$(mktemp -d)/maniskill_pick_place"
cp -R "$(git rev-parse --show-toplevel)/examples/maniskill_pick_place" "$tmp_dir"
cd "$tmp_dir"
printf '["pushcube__0", "pickcube__0", "stackcube__0"]' > helix_batch.json
uv run --no-project --python 3.12 --with mani_skill==3.0.1 --with pin python3 evaluate.py
```

The evaluator prints one `HELIX_RESULT=` line. Each trial's `feedback` names
the stage the rollout reached (for example `never_reached` or
`grasped_no_transport`) and gives a hint; the other `side_info` keys are
diagnostics such as `ever_grasped`, `closest_object_to_goal_m` and
`steps_used`.

## Results

The seed `solver/` scores 0.138 on validation. See
[`docs/experiments/maniskill-pick-place.md`](../../docs/experiments/maniskill-pick-place.md)
for a full run.

## Files

```
maniskill_pick_place/
├── helix.toml           # HELIX config
├── evaluate.py          # runs the batch in one child process
├── API_REFERENCE.md     # the ctx contract solvers code against
├── sim/                 # ManiSkill adapter
│   ├── __init__.py      #   macOS Vulkan setup, applied before SAPIEN loads
│   ├── tasks.py         #   instance id -> (ManiSkill env, reset seed)
│   └── env.py           #   primitives, scoring, per-trial diagnostics
├── solver/              # the evolving package (naive seed)
└── splits/{train,val}/  # train and val instance ids
```

`evaluate.py`, `sim/`, `splits/`, `API_REFERENCE.md` and `helix.toml` are in
`[evaluator].protected_files`, so a candidate that changes any of them is
rejected.

## Requirements

The evaluator command in `helix.toml` pins Python 3.12 and installs
`mani_skill==3.0.1` and `pin` (Pinocchio, for the arm's inverse-kinematics
controller) through [`uv`](https://docs.astral.sh/uv/) on first use.
ManiSkill also needs a Vulkan driver, even though this benchmark renders
nothing.

- macOS: run `brew install molten-vk` once. `sim/__init__.py` points the
  Vulkan loader at it.
- Linux: x86_64 only, because ManiSkill 3.0.1 depends on mplib, which has no
  ARM Linux wheels. On Debian 13, install
  `libx11-6 libgl1 libglib2.0-0 libgomp1 libvulkan1 mesa-vulkan-drivers`.
  Debian 12's Mesa is too old: SAPIEN does not accept its CPU Vulkan driver.

If the simulator raises an error while starting, `evaluate.py` exits with it
instead of printing a result, so `helix evolve` stops at the seed evaluation. The
simulator's error is in `.helix/helix.log`.
