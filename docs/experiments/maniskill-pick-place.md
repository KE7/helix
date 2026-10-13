# ManiSkill Pick-and-Place Experiment

> **Objective:** Evolve a multi-file Python solver that completes push, pick and stack tasks on ManiSkill 3 with a Franka Panda, using four primitives for observation, motion and the gripper.

---

## Overview

HELIX evolves a `solver/` package that controls a Franka Panda on three ManiSkill 3 tasks. HELIX's coding-agent mutator rewrites the package, and the evaluator runs `solver.run(ctx)` once per trial and scores the rollout. Neither the evaluator nor the seed or evolved solver calls an LLM. HELIX is also the optimizer in [RHO](https://rho-robotics.github.io) (Robotics Harness Optimization, arXiv:2606.16458).

The example is [`examples/maniskill_pick_place/`](../../examples/maniskill_pick_place/).

| | Value |
|---|---|
| **Simulator** | ManiSkill 3.0.1, CPU simulation, state observations |
| **Robot** | Franka Panda, `pd_ee_delta_pose` controller |
| **Families** | `pushcube`, `pickcube`, `stackcube` (`PushCube-v1`, `PickCube-v1`, `StackCube-v1`) |
| **Primitives** | `get_observation`, `move_to(x, y, z)`, `open_gripper`, `close_gripper` |
| **Score per trial** | 1.0 on success, otherwise capped partial credit (see [Evaluation](#evaluation)) |
| **Splits** | 9 train trials (`*__0..2`), 9 validation trials (`*__3..5`) |
| **Held-out check** | 60 trials (`*__6..25`), never seen by HELIX or the agent |

In the recorded run one mutation took validation from 0.138 to 1.0 and solved all 60 held-out trials. It also solved all 9 training trials, so HELIX skipped the remaining generations (see [Why the search stopped](#why-the-search-stopped)). The run was not sandboxed, so it is an end-to-end check of the example, not an isolated benchmark result.

---

## Configuration

The settings in [`examples/maniskill_pick_place/helix.toml`](../../examples/maniskill_pick_place/helix.toml), without the `objective`, the `[agent].background` text and the comments:

```toml
[project]
passthrough_env = ["USER"]

[evaluator]
command = "uv run --no-project --python 3.12 --with mani_skill==3.0.1 --with pin python3 evaluate.py"
protected_files = ["evaluate.py", "sim", "splits", "API_REFERENCE.md", "helix.toml"]

[seedless]
enabled = false
train_path = "splits/train"
val_path = "splits/val"

[evolution]
max_generations = 15
max_evaluations = 400
minibatch_size = 3
perfect_score_threshold = 1.0
batch_sampler = "stratified"
cache_evaluation = true
merge_enabled = false

[agent]
backend = "claude"
model = "sonnet"
effort = "medium"
max_turns = 30
```

`acceptance_criterion` is not set, so it is the default, `strict_improvement`.

`_scrub_environment` in `src/helix/executor.py` passes the evaluator and the agent only an allow-list of variables, and `USER` is not on it by default. Claude Code on macOS looks up its Keychain login by `$USER`, so without `passthrough_env = ["USER"]` non-sandboxed `claude` mutations fail at once with "Not logged in".

`[agent].background` tells the agent to read `API_REFERENCE.md` first, lists the protected files, and gives the command for checking a change on one trial per family.

---

## Seed Solution

The seed [`solver/`](../../examples/maniskill_pick_place/solver/) is a task router (`main.py`), a shared `common.py`, and one routine per family:

- `pushcube` moves to a height of 8 cm over the cube, then over the goal, and never touches the cube.
- `pickcube` closes the gripper at a height of 12 cm over the cube.
- `stackcube` closes at 12 cm over cube A and opens over cube B.

The seed scores 0.138 on validation and succeeds on 0 of 9 trials. Its family means are 0.168 (push), 0.110 (pick) and 0.135 (stack).

---

## Evaluation

`evaluate.py` runs the batch in one spawned child process. The child first points file descriptor 1 at `/dev/null`, so nothing the solver or a native library prints can reach the stdout that carries `HELIX_RESULT`.

The solver still runs in the same process as the scoring code, so solver code that imports or patches `sim` could change its own score. `API_REFERENCE.md` tells solvers not to import `sim`, but nothing in this example enforces it. The recorded run's evolved solver does not import `sim`.

An instance id must be `<family>__<n>`, where the family is `pushcube`, `pickcube` or `stackcube` and `n` is a non-negative integer. Any other id scores 0 with stage `benchmark_error`. The CRC32 of a valid id seeds `env.reset`, so every id, including held-out ids not listed under `splits/`, starts from the same state on every run.

Each trial rebuilds the ManiSkill scene on reset (`reconfigure=True`), at about 0.05 s per trial. Without the rebuild, `pushcube__10` ended with a different cube position when it ran after `pushcube__4` in the same process. A test runs `pushcube__10` under the reference policy before and after `pushcube__4` and checks that the final cube position is the same. Runs 1 to 5 did not have the rebuild (see [Earlier Runs](#earlier-runs)).

The child then calls `solver.run(ctx)`. An exception or `sys.exit` from the solver becomes feedback, and the trial is still scored.

A trial has a budget of 400 control steps, spent by `move_to` and the gripper primitives. [`API_REFERENCE.md`](../../examples/maniskill_pick_place/API_REFERENCE.md) gives the cost of each call and what a call returns when the budget runs out. Once `steps_left` is 0, the next call to `move_to` or a gripper primitive raises `BudgetExhausted`, which ends `solver.run`, and the trial is scored as it stands. `BudgetExhausted` derives from `BaseException`, so a solver's `except Exception` cannot swallow it.

After `solver.run` ends, `sim/env.py` holds the arm still for up to 40 settle steps, stopping early once the environment reports success. The trial scores 1.0 if the environment's success flag is set. Otherwise it scores the highest normalized dense reward reached at any step of the rollout, capped at 0.95 so that a failed trial always scores below a successful one.

Each completed trial's `side_info` contains:

- `scores`: `reward` and `success` (1.0 or 0.0).
- `feedback`: the instance id, the stage the rollout reached (for example `never_reached`, `grasped_no_transport` or `pushed_partway`), the reward, a hint for that stage, and any solver traceback.
- Diagnostics: the environment's info flags, `ever_grasped`, `closest_tcp_to_object_m`, `closest_object_to_goal_m`, `final_object_position`, `steps_used` and `primitive_calls`.

A trial that does not report within 150 s is killed and scores 0 with a timeout message. If the child process dies, the trial it was running scores 0 and its feedback gives the exit code. In both cases every later trial in the batch scores 0 with "Not run: the evaluator stopped at trial <id>." If the simulator cannot start, `evaluate.py` exits without a `HELIX_RESULT` line and `helix evolve` stops at the seed evaluation.

---

## Recorded Run

The recorded run is run 6, the last of six (see [Earlier Runs](#earlier-runs)). It is the only run made on the example as committed here. It ran on 2026-10-15. HELIX was at `origin/main` commit `81fde06`, and ManiSkill 3.0.1 ran in CPU simulation on macOS on an Apple M3 Pro.

The mutation agent worked in a copy of `examples/maniskill_pick_place`. No file in that copy mentions the reference policy, the tests or the evolved solver, but its README links to this write-up by a relative path. This example has no sandboxed setup, so the agent ran on the host with access to files outside its workspace and to the network.

| | Value |
|---|---|
| **Validation score** | 0.138 (seed) -> 1.0 (`g1-s1`), 9 of 9 trials |
| **Mutations** | 1 accepted (`g1-s1`, generation 1), 0 rejected |
| **`g1-s1` agent session** | 13 turns, 12 tool calls, 65 s |
| **Output / cache-read tokens** | 5,722 / 271k |
| **Cost** | $0.36, as reported by Claude Code |
| **Evaluation budget** | 66 of `max_evaluations = 400` |
| **Wall clock** | 124 s |

These figures come from the run's `.helix/` directory and the agent's transcript, which are not committed.

### Why the search stopped

HELIX skipped generations 2 to 15. The 14 skip records in `.helix/skips/`, which `helix attempts --skips` lists, all give the reason `perfect_subsample` with parent `g1-s1`. `g1-s1` succeeds on all 9 training trials, so every sampled training minibatch scored 1.0 and `perfect_score_threshold = 1.0` skipped the mutation.

With 3 training trials per family, one mutation is enough to saturate the training set. In runs 3 and 5 the search stopped this way while a validation push still failed. Removing the threshold would not be enough, because under the default `strict_improvement` acceptance a child cannot beat a parent that scores 1.0 on its training minibatch. More or harder training trials would keep the search going.

### Held-out check

The evolved solver succeeds on all 60 held-out trials, 20 in each family. With the train and validation trials it succeeds on 78 of 78. The benchmark test `test_recorded_evolved_solver_reproduces` replays this solver on all 78 trials and fails on any miss (see [Reproducing](#reproducing)).

The scripted [reference policy](maniskill-pick-place/reference_policy.py) is outside the copied example. It solves all 18 train and validation trials, and the tests use it to check that each of them can be solved with the four primitives. On the held-out trials it scores 58 of 60, missing `pushcube__12` and `pushcube__23`.

### Transcript audit

The agent's transcript has 12 tool calls. It read the task prompt, listed the workspace, and read `API_REFERENCE.md`, `solver/common.py`, `solver/main.py` and the three task routines. It then rewrote the three task routines and ran the evaluator once on `pushcube__0`, `pickcube__0` and `stackcube__0`. That evaluator run is outside HELIX's evaluation budget. It never opened `evaluate.py`, `sim/`, `README.md`, or any path outside its workspace.

---

## What the Agent Wrote

The evolved package is in [`maniskill-pick-place/evolved/solver/`](maniskill-pick-place/evolved/solver/). The `solver/` diff against the seed is 3 files, +77/-12 lines, all in the three task routines. `main.py`, `common.py` and the `__init__.py` files are unchanged.

- `pickcube`: the safe height is the higher of the cube's and the goal's z, plus 10 cm. It opens, moves over the cube at the safe height, descends to the cube's center height and closes. If `grasped` is false, it descends 5 mm more and closes again, once, without reopening first. It then rises to the safe height, moves over the goal and descends to the goal point.
- `stackcube`: the same open, approach and close on cube A, with the same single retry 5 mm lower. The place height is cube B's z plus half of B plus half of A. It rises, moves over cube B at the safe height, descends to the place height, opens and rises.
- `pushcube`: the routine closes the gripper to make a pusher. It starts behind the cube on the side away from the goal, the half-size plus 4 cm from its center, approaching from 8 cm above. It then pushes along the straight line from that start to a point 0.3 half-sizes (6 mm) past the goal, in 6 equal segments. After each segment it reads the observation but uses only `steps_left`, so it never re-aims at the cube's new position. Finally it lifts 8 cm.

In the agent's prompt, the seed's feedback gave `pickcube__2` and `stackcube__0` the stage `never_reached`, with a hint to move above the cube at a safe height, open the gripper and descend to the cube's z before closing. It gave `pushcube__0` the stage `never_moved_object`, with a hint to close the gripper, start just behind the cube and push toward the goal in small segments, re-observing between pushes. The new routines follow these hints, except that the push routine does not re-aim between segments.

---

## Earlier Runs

Between runs, the feedback wording, comments, docstrings, README, `API_REFERENCE.md` and objective wording changed. Before run 5, a call made after the budget was spent returned `budget_exhausted` without ending the solver, and the push and stack hints were worded differently. The per-trial scene rebuild was added before run 6.

| Run | Date | Validation | Held-out |
|---|---|---|---|
| 1 | 2026-10-13 | 1.0 (9 of 9) | 60 of 60 |
| 2 | 2026-10-14 | 1.0 (9 of 9) | 60 of 60 |
| 3 | 2026-10-14 | 0.933 (8 of 9, misses `pushcube__4`) | 58 of 60 (misses `pushcube__7`, `__10`) |
| 4 | 2026-10-15 | 1.0 (9 of 9) | not measured (a one-generation check of the README steps) |
| 5 | 2026-10-15 | 0.994 (8 of 9, misses `pushcube__4`) | 58 of 60 (misses `pushcube__7`, `__10`) |

A validation score is the mean of the per-trial scores, so a missed trial still adds its partial credit.

All six runs used the settings above, except that run 4 was started with `helix evolve --generations 1`. Every run accepted exactly one mutation.

Per-trial scores were checked on all 78 trials whenever the scoring code changed. They were identical for the seed and the reference policy across every version, for run 1's solver across a cleanup of that code, for run 3's solver across the budget and hint change, and for run 5's solver across the scene-rebuild fix.

### The far-goal pushes in runs 3 and 5

Runs 3 and 5 both missed `pushcube__4`, `__7` and `__10`. The measurements below are from run 5's solver.

Among `pushcube__0` to `__25`, these are three of the four trials whose goals lie farthest out (largest x), at goal x = 0.2996 (`pushcube__4`), 0.2966 (`pushcube__7`) and 0.2942 (`pushcube__10`). The fourth, `pushcube__8` (goal x = 0.2983), succeeded. The gripper's workspace stops at x = 0.25 m (`WORKSPACE_X` in `sim/env.py`, about the Panda's reach).

Run 5's push routine re-aims each round and pushes up to 8 cm past the cube in one move. On these three trials the first full push sends the cube past the goal to x = 0.42 to 0.43. The routine then makes two moves to get behind the cube again and one push. Each of these moves targets a point beyond x = 0.25, is clamped to 0.25, and times out at 120 steps. These three moves use up the rest of the 400-step budget. The cubes end 12.3 cm (`pushcube__4`) and 13.4 cm (`pushcube__10`) from the goal, where success needs under 10 cm.

Run 6's solver and the reference policy solve all three. On these trials the end of run 6's push line lies beyond x = 0.25 and is clamped to it.

---

## Limitations

The task is easy for Claude Code with Sonnet at medium effort. In every run the first mutation reached 8 or 9 of 9 on validation. Besides more or harder training trials (see [Why the search stopped](#why-the-search-stopped)), the task could be made harder with:

- Tasks that need orientation control or contact precision, such as `PegInsertionSide-v1`, `PlugCharger-v1` and `LiftPegUpright-v1`. These need a `move_to` that also accepts an orientation.
- Less prescriptive `side_info`: keep the stage and distances, and drop the per-stage hint.
- RGB-D observations with a perception primitive in place of ground-truth object poses.

---

## Reproducing

The simulator was tested on macOS (Apple M3 Pro) and in Docker on Debian 13 (amd64). Both need a Vulkan driver. The example README's [Requirements](../../examples/maniskill_pick_place/README.md#requirements) section lists the setup. `helix evolve` uses the `claude` backend, so the Claude Code CLI must be installed and logged in.

The benchmark tests are in [`maniskill-pick-place/test_benchmark.py`](maniskill-pick-place/test_benchmark.py). Run them from the repository root:

```bash
uv run --no-project --python 3.12 --with mani_skill==3.0.1 --with pin \
    --with pytest pytest docs/experiments/maniskill-pick-place -q
```

The 33 tests take about 40 s on CPU once ManiSkill's caches are built. The first run can take over a minute while it builds them. They cover the ids and splits, deterministic reset, independence from earlier trials, the primitives and step budget, the `HELIX_RESULT` protocol and evaluator startup failure, the seed and reference policy, and a replay of the evolved solver on all 78 trials. The tests sit outside pytest's `testpaths`, so a default `pytest` run skips them.

To run the evolution, follow the example README's [How to run](../../examples/maniskill_pick_place/README.md#how-to-run).

After the run, `helix best` prints the best candidate's id and scores, and `helix best --export <path>` copies its tree. `helix frontier` shows the Pareto frontier and `helix history` the lineage tree. `helix attempts` lists rejected attempts, if there were any, and `helix attempts --skips` lists the skipped generations.

---

<div align="center">
<sub><a href="../../README.md">← Back to HELIX</a></sub>
</div>
