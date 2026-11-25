# ManiSkill pick-and-place

HELIX evolves a `solver/` package that controls a Franka Panda on three ManiSkill 3 tasks. HELIX's coding-agent mutator rewrites the package, and the evaluator runs `solver.run(ctx)` once per trial and scores the rollout. Neither the evaluator nor the seed or evolved solver calls an LLM. HELIX is also the optimizer in [RHO](https://rho-robotics.github.io) (Robotics Harness Optimization, arXiv:2606.16458).

The example is [`examples/maniskill_pick_place/`](https://github.com/KE7/helix/tree/main/examples/maniskill_pick_place).

| | Value |
|---|---|
| **Simulator** | ManiSkill 3.0.1, CPU simulation, state observations; the physics and IK packages are pinned in `requirements.txt` |
| **Robot** | Franka Panda, `pd_ee_delta_pose` controller |
| **Families** | `pushcube`, `pickcube`, `stackcube` (`PushCube-v1`, `PickCube-v1`, `StackCube-v1`) |
| **Primitives** | `get_observation`, `move_to(x, y, z)`, `open_gripper`, `close_gripper` |
| **Score per trial** | 1.0 on success, otherwise capped partial credit (see [Evaluation](#evaluation)) |
| **Splits** | 33 train trials (`*__0..10`), 18 validation trials (`*__26..31`) |
| **Held-out check** | 90 trials: the same 30 trial numbers for each family, drawn at random from 1,000,000 to 9,999,999 and kept out of the example |

In the recorded run, with the agent in HELIX's Docker sandbox and every solver in a throwaway container, one mutation took validation from 0.064 to 1.0 and solved all 141 train, validation and held-out trials, so HELIX skipped the remaining generations (see [Search progress](#search-progress)).


## Configuration

The settings in [`examples/maniskill_pick_place/helix.toml`](https://github.com/KE7/helix/blob/main/examples/maniskill_pick_place/helix.toml), without the `objective`, the `[agent].background` text and the comments:

```toml
[evaluator]
command = "uv run --no-project --python 3.12 --with-requirements requirements.txt python3 -I evaluate.py"
protected_files = ["evaluate.py", "sim", "splits", "requirements.txt", "solver.Dockerfile", "check.py", "check_server.py", "run.sh", "API_REFERENCE.md", "README.md", "helix.toml"]

[seedless]
enabled = false
train_path = "splits/train"
val_path = "splits/val"

[evolution]
max_generations = 15
max_evaluations = 600
batch_sampler = "stratified"
num_sampled_groups = 3
num_examples_per_group = 2
minibatch_size = 6
perfect_score_threshold = 1.0
cache_evaluation = true
merge_enabled = false

[sandbox]
enabled = true
evaluator = false
add_host_gateway = true
omit_from_agent = ["evaluate.py", "sim", "splits/val", "README.md", "requirements.txt", "solver.Dockerfile", "check_server.py", "run.sh"]

[agent]
backend = "claude"
model = "sonnet"
effort = "medium"
max_turns = 30
```

`acceptance_criterion` is not set, so it is the default, `strict_improvement`. Each minibatch holds two training trials from each of the three families.

The mutation agent runs in HELIX's Docker sandbox, and the evaluator runs on the host as your own user (`evaluator = false`), with each solver it scores in a container of its own (see [Evaluation](#evaluation)). The agent works on a copy of the candidate without the files listed in `omit_from_agent`, so it sees only `solver/`, the training ids under `splits/train`, `API_REFERENCE.md` and `check.py`. It logs in to Claude through HELIX's sandbox auth volume (`helix sandbox login claude`).

`[agent].background` tells the agent to read `API_REFERENCE.md` first, says every file but `solver/` is protected, asks it not to branch on trial ids or memorize coordinates, and gives `python3 check.py <training ids>` as the way to check a change.

### The check service

The sandboxed agent cannot run ManiSkill: its container has no Vulkan driver, and on Apple Silicon it is arm64 Linux, for which ManiSkill's mplib dependency has no wheels. So [`check.py`](https://github.com/KE7/helix/blob/main/examples/maniskill_pick_place/check.py) sends the candidate's `solver/` files and the ids to [`check_server.py`](https://github.com/KE7/helix/blob/main/examples/maniskill_pick_place/check_server.py) on the host, at `http://host.docker.internal:8765`. The service refuses any id outside `splits/train` and any file that is not a `.py` file under `solver/`. It writes the files into a private copy of the example, runs `evaluate.py` there, so the solver goes through the same container as in HELIX's own evaluations, and returns each trial's score and side_info. [`run.sh`](https://github.com/KE7/helix/blob/main/examples/maniskill_pick_place/run.sh) starts the service, runs `helix evolve` and stops the service when HELIX exits. On macOS the service listens on 127.0.0.1, which Docker Desktop's `host.docker.internal` reaches; on Linux, `run.sh` binds it to the Docker bridge gateway.

### Splits

The training set holds the first 11 trials of each family. That includes the four push trials among `pushcube__0` to `__25` whose goals lie farthest out (`__4`, `__7`, `__8` and `__10`, at x = 0.29 to 0.30): the arm cannot reach that far at table height, so the cube has to end close to the farthest point the pusher can push it. Validation uses `*__26..31`. The held-out trial numbers were drawn once with `random.Random(20261027).sample(range(1_000_000, 10_000_000), 30)` and are listed in [`test_benchmark.py`](https://github.com/KE7/helix/blob/main/docs/experiments/maniskill-pick-place/test_benchmark.py) and `recorded_run.json`, outside the example. The agent's copy holds only the training ids, and its `check.py` scores nothing else.


## Seed solver

The seed [`solver/`](https://github.com/KE7/helix/tree/main/examples/maniskill_pick_place/solver) is a task router (`main.py`), a shared `common.py`, and one routine per family:

- `pushcube` moves to z = 8 cm over the cube, then over the goal, and never touches the cube.
- `pickcube` closes the gripper at z = 12 cm over the cube, about 10 cm above its center, then moves to the goal.
- `stackcube` closes at z = 12 cm over cube A and opens at the same height over cube B.

The seed scores 0.064 on validation and succeeds on 0 of 18 trials. Its family means are 0.055 (push), 0.051 (pick) and 0.085 (stack). It solves none of the 33 training or 90 held-out trials.


## Evaluation

[`evaluate.py`](https://github.com/KE7/helix/blob/main/examples/maniskill_pick_place/evaluate.py) runs as `python3 -I evaluate.py`. Isolated mode leaves the working directory off `sys.path`, so a top-level file that a candidate adds, such as a `json.py`, cannot shadow a module the evaluator imports. Three kinds of process take part:

- **The parent** builds the solver image from [`solver.Dockerfile`](https://github.com/KE7/helix/blob/main/examples/maniskill_pick_place/solver.Dockerfile) if it does not exist yet (Python 3.12 with numpy, run as a non-root user), reads `helix_batch.json`, starts an env process for the batch and prints the one `HELIX_RESULT` line. It never imports `sim` or `solver`.
- **The env process** points file descriptors 1 and 2 at `/dev/null`, so nothing it or its children print can reach HELIX's pipes. It loads `sim` by file path, builds the ManiSkill environments the batch needs and tells the parent it is ready, so loading the simulator does not count against any trial's time limit. It copies `solver/` and `evaluate.py` into a private temporary directory, then builds each trial, serves the solver's requests and scores the trial.
- **A solver container per trial** runs `python3 -I evaluate.py --solver-client`, which loads `solver` by file path and calls `solver.run(ctx)`. The container has no network, a read-only filesystem apart from a small `/tmp`, no capabilities, a pid and memory limit, and only the private copy mounted, read-only. The solver's `ctx` carries `task_family`, `instruction` and the four primitives, and each primitive is a JSON line to the env process over the container's stdin and stdout. The solver never has `sim` or the trial, and it cannot write to the example, the uv cache or anything else on the host, so a candidate can change its score only by moving the robot.

The env process reads requests as JSON, never by unpickling, accepts only the four primitive names, and requires three finite numbers for `move_to`. A request it cannot read ends the solver, and the trial is scored as it stands. An exception or `sys.exit` from the solver becomes feedback, with the 8 innermost traceback frames, and the trial is still scored. Starting a container costs about 0.2 s per trial on an Apple M3 Pro.

`--unsafe-local` runs each solver as a plain local process instead, for machines without Docker, and prints a warning: the solver then runs as your own user and could change any file you can, including the scorer. If Docker is not available and `--unsafe-local` is not given, the evaluator exits with an error instead of printing a result.

An instance id must be `<family>__<n>`, where the family is `pushcube`, `pickcube` or `stackcube` and `n` is a non-negative integer written in ASCII digits without leading zeros. Any other id scores 0 with stage `benchmark_error`. The CRC32 of a valid id seeds `env.reset`, so every id starts from the same state on every run.

Each trial rebuilds the ManiSkill scene on reset (`reconfigure=True`), at about 0.05 s per trial. Without the rebuild, `pushcube__10` ended with a different cube position when it ran after `pushcube__4` in the same process. A test runs `pushcube__10` under the reference policy before and after `pushcube__4` and checks that the final cube position is the same.

A trial has a budget of 400 control steps, spent by `move_to` and the gripper primitives. [`API_REFERENCE.md`](https://github.com/KE7/helix/blob/main/examples/maniskill_pick_place/API_REFERENCE.md) gives the cost of each call and what a call returns when the budget runs out. Once `steps_left` is 0, the next call to `move_to` or a gripper primitive raises `BudgetExhausted` in the solver process, which ends `solver.run`, and the trial is scored as it stands. `BudgetExhausted` derives from `BaseException`, so a solver's `except Exception` cannot swallow it, and a solver that swallows it anyway gets no further replies.

`move_to` drives the TCP with ManiSkill's `pd_ee_delta_pose` controller, one control step at a time toward the target. Near the edge of the arm's reach the controller finds no IK solution for a long step and leaves the arm in place. So after a step that does not move the TCP, `move_to` halves the step length, and it doubles it again after each step that does, so one brief stall does not slow the rest of the move. It ends the move as `stalled` when 5 mm steps do not move the TCP. With the gripper pointing down, the TCP reaches about x = 0.18 at table height near y = 0, short of the x = 0.25 that requests are clamped to, and `API_REFERENCE.md` says so.

After `solver.run` ends, `sim/env.py` holds the arm still for 40 settle steps and judges the final state. The trial scores 1.0 if the environment's success flag is set on the last step. Otherwise it scores the environment's normalized dense reward on the last step, capped at 0.95 so that a failed trial always scores below a successful one. A cube that slides through the goal and out again, or a stack that topples while settling, is judged where it ends up.

Each completed trial's `side_info` contains:

- `scores`: `reward` and `success` (1.0 or 0.0).
- `feedback`: the instance id, the stage the rollout reached (for example `never_reached`, `grasped_no_transport`, `pushed_partway` or `overshot_goal`), the reward, a hint for that stage, and any note on how the solver ended.
- Diagnostics: the environment's info flags, `ever_grasped`, `closest_tcp_to_object_m`, `closest_object_to_goal_m`, `final_object_to_goal_m`, `final_object_position`, `best_reward_seen`, `steps_used` and `primitive_calls`.

A push whose cube got within 10 cm of the goal but ended outside it gets the stage `overshot_goal`, and its hint says to use shorter final pushes and stop once the cube is inside the threshold.

A solver that has not finished within 60 s, for example because it loops without calling a primitive or swallows `BudgetExhausted` with a bare `except:`, is stopped with `docker kill`, and its trial is scored as it stands with a timeout note. A solver that exits early gets the same treatment, with its exit code in the note. Everything a solver starts lives in its container, which `--rm` removes, and the parent also kills any container labelled with the evaluation's id that is still running when an env process ends. So a solver that ignores SIGTERM or leaves a daemon behind cannot hang `helix evolve`, which runs the evaluator without a timeout. If the env process itself dies, or does not score a trial within 120 s, that trial scores 0 and the rest of the batch runs in a new env process. If the simulator cannot start, `evaluate.py` exits without a `HELIX_RESULT` line and `helix evolve` stops at the seed evaluation.


## Recorded run

The recorded run is the only run made on the example as committed here (for the ten before it, see [Earlier runs](#earlier-runs)). It ran on 2026-10-29 through `./run.sh`, with HELIX at `origin/main` commit `ad9f020` and the packages in `requirements.txt`, with ManiSkill 3.0.1 in CPU simulation on macOS on an Apple M3 Pro.

| | Value |
|---|---|
| **Validation score** | 0.064 (seed) -> 1.0 (`g1-s1`, 18 of 18) |
| **Mutations** | 1 accepted (`g1-s1`, generation 1), 0 rejected; generations 2 to 15 skipped |
| **Agent session** | 6 turns, 5 tool calls, 40 s |
| **Output / cache-read tokens** | 2,766 / 144k |
| **Cost** | $0.14, as reported by Claude Code |
| **Evaluation budget** | 132 of `max_evaluations = 600` |
| **Wall clock** | 2 min 29 s (`helix timing`): 42 s in the agent, 1 min 44 s in the evaluator |

`helix timing` reports "2 accepted, 14 rejected" because it counts the 14 skipped generations as rejected; no mutation was rejected.

[`recorded_run.json`](https://github.com/KE7/helix/blob/main/docs/experiments/maniskill-pick-place/recorded_run.json) holds these figures except the mutation counts and the wall clock, trimmed from the run's `.helix/` directory and the agent's transcript. It also has the lineage, the per-trial validation scores, the skip records with their minibatches, the agent's tool calls, and the replay of `g1-s1` on every train, validation and held-out trial.

### Search progress

The seed's minibatch feedback gave `pushcube__1` and `__4` the stage `never_moved_object`, and `pickcube__1`, `__8`, `stackcube__5` and `__6` the stage `never_reached`. In generation 1 the agent rewrote the three task routines and checked them with `check.py` in two calls, on 6 chosen and then 14 random training trials (17 distinct ids), all of which succeeded. Then it finished. Its tool calls, listed in `recorded_run.json`, read the workspace, rewrote the three routines and ran `check.py` on training ids; its container held no evaluator or simulator. `g1-s1` succeeds on all 33 training trials, so every later minibatch scored 1.0, and `perfect_score_threshold = 1.0` skipped generations 2 to 15.

With `perfect_score_threshold = 1.0`, HELIX skips a generation whose parent is already perfect on the minibatch, so a first mutation that solves every training trial ends the search. Without the threshold, no child could beat a perfect parent at the gate either. In runs 9 and 10, on the same splits, the first mutation overshot a training push and a second one fixed it. Getting more out of the search needs harder training trials (see [Harder variants](#harder-variants)).

### Held-out check

`g1-s1` succeeds on all 90 held-out trials, 30 in each family, and with the train and validation trials on 141 of 141. The benchmark test `test_recorded_evolved_solver_reproduces` replays this solver on all 141 trials and fails if any trial's outcome changes (see [Reproducing](#reproducing)).

The scripted [reference policy](https://github.com/KE7/helix/blob/main/docs/experiments/maniskill-pick-place/reference_policy.py) is outside the copied example. It solves all 33 training and 18 validation trials, and the tests use it to check that each training trial can be solved with the four primitives. On the held-out trials it scores 89 of 90, missing `pushcube__9141871`.

## Evolved solver

The evolved package is [`g1-s1`](https://github.com/KE7/helix/tree/main/docs/experiments/maniskill-pick-place/evolved/solver). Its `solver/` diff against the seed is 3 files, +55/-10 lines, all in the three task routines. `main.py`, `common.py` and the `__init__.py` files are unchanged.

- `pushcube` closes the gripper to make a pusher, moves to z = 10 cm over a point 6 cm behind the cube on the side away from the goal, and descends to z = 2 cm. In up to 8 rounds it then reads the cube's position and stops once the cube is within 1 cm of the goal or fewer than 20 steps are left. Otherwise it moves the TCP from where it is toward the goal, along the line from the cube to the goal, by the remaining distance plus 2 cm, at most 8 cm.
- `pickcube` opens, moves to z = 10 cm over the cube, descends to its center and closes. It rises to z = 12 cm, moves to the goal point, then reads where the cube actually is and moves once more by the cube's offset from the goal.
- `stackcube` grasps cube A the same way and rises to z = 12 cm. Over cube B it reads where A sits in the gripper. It moves to the place point, B's z plus one cube height plus 3 mm, shifted to cancel that offset, then opens and rises to z = 15 cm.

None of the routines checks `grasped`, and the push keeps going until the cube is within 1 cm of the goal, well inside the 10 cm threshold.


## Independent reproduction

Karim Elmaaroufi ran an earlier version of the example on an Apple M4 Pro, with HELIX `main` at `91bcc9f` and Codex `gpt-6-luna` at medium effort as the mutator, for 10 generations ([comment](https://github.com/KE7/helix/pull/77#issuecomment-6448160846), with rollout GIFs [below it](https://github.com/KE7/helix/pull/77#issuecomment-6448249934)). One mutation took validation from 0.138 to 1.0, and the evolved solver solved 57 of 60 held-out trials. Three things that run found shaped the example: two of the misses were pushes that slid through the goal and still scored 0.95, which led to final-state scoring and the `overshot_goal` stage; its agent read `sim/` and ran the evaluator on held-out ids, which led to the random held-out ids and the sandbox; and a solver sharing the scorer's process could reach `ctx._trial`, which led to the solver container.


## Earlier runs

Runs 1 to 10 used earlier versions of the example. The agent ran on the host and could read `sim/` and `evaluate.py`. Runs 1 to 8 ran the solver inside the scorer's process, and runs 9 and 10 in a separate local process. Run 10 repeated run 9 after the objective in `helix.toml` was corrected to describe final-state scoring. Runs 1 to 8 also used the old splits (train `*__0..2`, validation `*__3..5`, held-out `*__6..25`) and scored a failed trial by its best reward at any step, so their numbers are not comparable with run 9's. Every run accepted one or two mutations.

| Run | Date | Validation | Held-out |
|---|---|---|---|
| 1 | 2026-10-13 | 1.0 (9 of 9) | 60 of 60 |
| 2 | 2026-10-14 | 1.0 (9 of 9) | 60 of 60 |
| 3 | 2026-10-14 | 0.933 (8 of 9, misses `pushcube__4`) | 58 of 60 (misses `pushcube__7`, `__10`) |
| 4 | 2026-10-15 | 1.0 (9 of 9) | not measured (a one-generation check of the README steps) |
| 5 | 2026-10-15 | 0.994 (8 of 9, misses `pushcube__4`) | 58 of 60 (misses `pushcube__7`, `__10`) |
| 6 | 2026-10-15 | 1.0 (9 of 9) | 60 of 60 |
| 7 | 2026-10-26 | 0.934 (8 of 9, misses `pushcube__4`) | 56 of 60 (misses `pushcube__7`, `__8`, `__10`, `__19`) |
| 8 | 2026-10-26 | 1.0 (9 of 9) | 60 of 60 |
| 9 | 2026-10-27 | 1.0 (18 of 18, new splits) | 89 of 90 (new held-out ids; misses `pushcube__9141871`) |
| 10 | 2026-10-27 | 1.0 (18 of 18) | 90 of 90 |
| 11 | 2026-10-29 | 1.0 (18 of 18) | 90 of 90 (the recorded run, above) |

A validation score is the mean of the per-trial scores, so a missed trial still adds its partial credit.


## Harder variants

The task is easy for Claude Code with Sonnet at medium effort: in every run, one or two mutations solved all or nearly all validation trials. It could be made harder with:

- Tasks that need orientation control or contact precision, such as `PegInsertionSide-v1`, `PlugCharger-v1` and `LiftPegUpright-v1`. These need a `move_to` that also accepts an orientation.
- Less prescriptive `side_info`: keep the stage and distances, and drop the per-stage hint.
- RGB-D observations with a perception primitive in place of ground-truth object poses.

## Reproducing

The simulator was tested on macOS (Apple M3 Pro) and in Docker on Debian 13 (amd64). The Debian check ran on the example as it was before run 7; the changes made since were tested on macOS only. Both platforms need a Vulkan driver, and the example now needs Docker for the agent sandbox and the solver containers. The example README's [Requirements](https://github.com/KE7/helix/blob/main/examples/maniskill_pick_place/README.md#requirements) section lists the setup. Before the first run, log the sandbox in to Claude with `helix sandbox login claude`.

The benchmark tests are in [`maniskill-pick-place/test_benchmark.py`](https://github.com/KE7/helix/blob/main/docs/experiments/maniskill-pick-place/test_benchmark.py). Run them from the repository root:

```bash
uv run --no-project --python 3.12 \
    --with-requirements examples/maniskill_pick_place/requirements.txt \
    --with pytest pytest docs/experiments/maniskill-pick-place -q
```

The 61 tests take about 2 minutes on CPU once ManiSkill's caches are built and the solver image exists; the Docker-based tests are skipped if Docker is not running. The first run takes longer while uv installs the pinned packages, ManiSkill builds its caches and Docker builds the solver image. The 141-trial replay takes about 70 s, so it raises the repository's 60 s per-test limit to 600 s. The tests sit outside pytest's `testpaths`, so a default `pytest` run skips them. They check:

- the benchmark itself: the ids and splits, deterministic resets, that a trial does not depend on the ones before it,
  the primitives, the arm's reach, stall recovery, the step budget, and final-state scoring with `overshot_goal`;
- the results: the `HELIX_RESULT` line, the reference policy on all 33 training trials, the seed, and a replay of the
  evolved solver on all 141 train, validation and held-out trials;
- that a solver cannot cheat or break the evaluator: by printing a fake `HELIX_RESULT` line, importing `sim`, writing
  to its copy of the example or to the image, opening a network connection, adding a top-level `json.py`, sending a
  pickled request or a non-finite `move_to`, ignoring SIGTERM, leaving processes running, hanging or exiting early;
- that the evaluator stops cleanly when the simulator cannot start, `--unsafe-local` mode, and that the check service
  refuses non-training ids and files outside `solver/`.

To run the evolution, copy the example out of the repository and run `./run.sh` there, as the example README's [How to run](https://github.com/KE7/helix/blob/main/examples/maniskill_pick_place/README.md#how-to-run) shows. `run.sh` passes its arguments on to `helix evolve`.

After the run, `helix best --export <path>` exports the best candidate's tree, including `solver/`, and `helix attempts --skips` lists the generations skipped as perfect.

