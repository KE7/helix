# HELIX TOML Reference

Use this file when creating or editing `helix.toml`.

## Contents

- Root Fields
- Evaluator
- Dataset
- Seedless
- Evolution
- Agent
- Sandbox
- Worktree

## Root Fields

```toml
objective = "Concrete optimization objective."
seed = "."        # optional, default current project
rng_seed = 0      # deterministic selection/sampling
passthrough_env = ["CUDA_VISIBLE_DEVICES", "HF_HOME", "MUJOCO_GL"]
```

`passthrough_env` is the only normal path for evaluator/agent subprocesses to
receive host env vars after HELIX scrubs the environment. Do not pass secrets to
agents unless the user intentionally accepts that risk. Prefer local proxy URLs
over raw API keys in evaluator code.

## Evaluator

```toml
[evaluator]
command = "uv run python evaluate.py"
score_parser = "helix_result"
include_stdout = true
include_stderr = true
extra_commands = []
protected_files = ["evaluate.py", "data/gold.json"]
```

Rules:

- `command` is split with `shlex.split()` and run with `shell=false`.
- Prefer `uv run python evaluate.py`, `python -m package.eval`, or
  `bash run_eval.sh`; avoid bare `python3` unless dependencies are guaranteed.
- `extra_commands` run after the main command; in sandbox mode they share the
  same temporary evaluator workspace as the main command.
- `protected_files` are hashed at run start and checked after mutations/merges.
  Add evaluator scripts, benchmark data, goldens, and scoring helpers.

Score parsers:

- `pytest`: score from pytest-like output.
- `exitcode`: success is based on process exit code.
- `json_accuracy`: parse JSON output with accuracy-like fields.
- `json_score`: parse JSON output with score-like fields.
- `helix_result`: GEPA-compatible per-example contract; use for datasets,
  minibatches, side info, and multi-objective frontiers.

## Dataset

```toml
[dataset]
train_size = 100
val_size = 100
```

HELIX does not own train/val files here. It only needs cardinality. For each
evaluation, HELIX writes a JSON `list[str]` to `helix_batch.json`; the evaluator
loads its own dataset and filters by those ids.

Single-task mode: omit `[dataset]` or leave sizes unset.

Stratified task ids: use ids shaped like `task__trial`; HELIX derives group
keys by splitting on `evolution.group_key_separator`.

## Seedless

```toml
[seedless]
enabled = false
train_path = "data/train.jsonl"
val_path = "data/val.jsonl"
```

Seedless mode asks the backend to create the initial candidate from the
objective instead of starting from the current working tree. `train_path` and
`val_path` are prompt-grounding paths, not runtime dataset ownership.

## Evolution

```toml
[evolution]
max_generations = 20
perfect_score_threshold = 1.0
max_evaluations = -1

minibatch_size = 3
num_parallel_proposals = 4  # or "auto"
max_workers = 32
cache_evaluation = true
acceptance_criterion = "strict_improvement"
val_stage_size = 10

batch_sampler = "epoch_shuffled"  # or "stratified"
group_key_separator = "__"

frontier_type = "hybrid"  # "instance" | "objective" | "hybrid" | "cartesian"

merge_enabled = false
max_merge_invocations = 5
merge_val_overlap_floor = 5
merge_subsample_size = 5
```

Important choices:

- `frontier_type = "hybrid"` is the HELIX default and matches GEPA
  `optimize_anything`'s multi-axis intent.
- Non-`instance` frontiers need `score_parser = "helix_result"` and
  per-example `side_info["scores"]`.
- `num_parallel_proposals` controls concurrent agent mutations per generation.
  Keep it low for expensive agent backends.
- `max_workers` bounds evaluation/mutation thread pools.
- Enable `merge_enabled` only when candidate changes can be meaningfully merged.

## Agent

```toml
[agent]
backend = "claude"  # claude | codex | cursor | gemini | opencode
model = "sonnet"
effort = "medium"
max_turns = 20
allowed_tools = ["Read", "Edit", "Write", "Bash", "Glob", "Grep"]
background = "Only modify src/solver.py. Do not edit evaluate.py."
```

Use `background` as the agent's project-specific guardrail. It is not a
security boundary; use `protected_files` and sandboxing for stronger controls.

Backend notes:

- Claude: usually `claude`, `max_turns`, optional model/effort.
- Codex: usually `codex`, model optional.
- Cursor/Gemini/OpenCode: check CLI support and model naming before pinning.

## Sandbox

```toml
[sandbox]
enabled = true
image = "ghcr.io/ke7/helix-evo-runner-claude:latest" # optional
network = "bridge"
cpus = 4.0
memory = "16g"
timeout_seconds = 3600
pids_limit = 512
add_host_gateway = false
skip_special_files = true
```

Recommended new-project workflow:

```bash
helix sandbox login claude
helix sandbox status claude
```

If `image` is unset, HELIX chooses the image from `agent.backend`.

Sandbox behavior:

- Agent runs in a copied `/workspace`; agent changes sync back.
- Evaluator runs in a fresh copied `/workspace`; evaluator changes are
  discarded.
- Agent auth volume mounts at `/home/node`.
- Evaluator containers do not get auth volumes.
- `.env`, `.env.*`, `.git`, and HELIX artifacts do not sync back.
- Special files are skipped by default; set `skip_special_files = false` to
  raise on unsupported file types instead.

Use `add_host_gateway = true` on Linux when evaluator code must reach a
host-side local proxy at `http://host.docker.internal:<port>/...`.

## Worktree

```toml
[worktree]
base_dir = ".helix/worktrees"
```

Leave this alone unless the project needs worktrees on a faster or larger disk.
