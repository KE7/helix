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

The success checks are listed in `API_REFERENCE.md`. There are 33 train
trials (`*__0..10`) and 18 validation trials (`*__26..31`); the ids are the
file stems under `splits/`.

## How to run

The mutation agent runs in HELIX's Docker sandbox and each trial's solver runs
in a throwaway container, so Docker must be running. Log the sandbox in to
Claude once:

```bash
helix sandbox login claude
```

Then copy the example to a standalone directory and start it with `run.sh`:

```bash
work_dir="$(mktemp -d)/maniskill_pick_place"
cp -R "$(git rev-parse --show-toplevel)/examples/maniskill_pick_place" "$work_dir"
cd "$work_dir"
./run.sh
```

`run.sh` starts the check service, runs `helix evolve` with any arguments you
pass it, and stops the service when HELIX exits. `helix evolve` initializes
the copy as a git repository and evolves `solver/` against `evaluate.py`
using `helix.toml`.

The agent sees only `solver/`, the training ids under `splits/train`,
`API_REFERENCE.md` and `check.py` (`[sandbox].omit_from_agent` keeps the rest
out of its copy). To try a change it runs `python3 check.py <training ids>`,
which sends the candidate to the check service (`check_server.py`) on the
host. The service runs those trials through `evaluate.py` and returns the
feedback HELIX would show, and it refuses any id outside `splits/train`.

`evaluate.py` runs each trial's solver in a container built from
`solver.Dockerfile`, with no network, a read-only filesystem and only a copy
of `solver/` mounted, so solver code cannot change the simulator, the scorer
or anything else on your machine. `--unsafe-local` runs solvers as plain
local processes instead, for machines without Docker; they then run as your
own user with no isolation.

To score the seed `solver/` by hand without writing into the repository, copy
the example and run the evaluator there:

```bash
tmp_dir="$(mktemp -d)/maniskill_pick_place"
cp -R "$(git rev-parse --show-toplevel)/examples/maniskill_pick_place" "$tmp_dir"
cd "$tmp_dir"
printf '["pushcube__0", "pickcube__0", "stackcube__0"]' > helix_batch.json
uv run --no-project --python 3.12 --with-requirements requirements.txt python3 -I evaluate.py
```

The evaluator prints one `HELIX_RESULT=` line. Each trial's `feedback` names
the stage the rollout reached (for example `never_reached` or
`grasped_no_transport`) and gives a hint; the other `side_info` keys are
diagnostics such as `ever_grasped`, `closest_object_to_goal_m` and
`steps_used`.

## Files

```
maniskill_pick_place/
├── helix.toml           # HELIX config
├── run.sh               # starts the check service, then helix evolve
├── evaluate.py          # runs each solver in a container, scores trials
├── solver.Dockerfile    # the solver container image
├── check.py             # the agent's training-only check command
├── check_server.py      # host service behind check.py
├── API_REFERENCE.md     # the ctx contract solvers code against
├── requirements.txt     # pinned simulator stack for the evaluator command
├── sim/                 # ManiSkill adapter
│   ├── __init__.py      #   macOS Vulkan setup, applied before SAPIEN loads
│   ├── tasks.py         #   instance id -> (ManiSkill env, reset seed)
│   └── env.py           #   primitives, scoring, per-trial diagnostics
├── solver/              # the evolving package (naive seed)
└── splits/{train,val}/  # train and val instance ids
```

Every file except `solver/` is in `[evaluator].protected_files`, so a
candidate that changes one is rejected.

## Requirements

The evaluator command in `helix.toml` pins Python 3.12 and installs the
packages in `requirements.txt` through [`uv`](https://docs.astral.sh/uv/) on
first use: ManiSkill 3.0.1 and the exact versions of the packages that carry
its physics and the arm's inverse-kinematics controller, such as SAPIEN,
PyTorch and Pinocchio (`pin`).
ManiSkill also needs a Vulkan driver, even though this benchmark renders
nothing. The agent sandbox and the solver containers need Docker (Docker
Desktop on macOS).

- macOS: run `brew install molten-vk` once. `sim/__init__.py` points the
  Vulkan loader at it.
- Linux: x86_64 only, because ManiSkill 3.0.1 depends on mplib, which has no
  ARM Linux wheels. On Debian 13, install
  `libx11-6 libgl1 libglib2.0-0 libgomp1 libvulkan1 mesa-vulkan-drivers`.
  Debian 12's Mesa is too old: SAPIEN does not accept its CPU Vulkan driver.

If the simulator raises an error while starting, `evaluate.py` exits with it
instead of printing a result, so `helix evolve` stops at the seed evaluation. The
simulator's error is in `.helix/helix.log`.
