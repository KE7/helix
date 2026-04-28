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

Without Docker sandboxing, `[evaluator].command` runs directly in the candidate
workspace. With `[sandbox].enabled = true`, `[evaluator.sidecar]` is required:
HELIX starts that evaluator service once per `helix evolve`, then runs
short-lived evaluator-runner containers that can reach it.

```toml
[evaluator]
command = "python /runner/evaluate_client.py"
score_parser = "helix_result"
include_stdout = true
include_stderr = true
extra_commands = []

[evaluator.sidecar]
image = "my-private-evaluator:latest"
runner_image = "my-evaluator-runner:latest"
command = "python -m benchmark_server"
endpoint = "http://helix-evaluator:8080/evaluate"
startup_timeout_seconds = 60
# Optional when the default HTTP reachability probe is not enough:
# healthcheck_command = "python /runner/healthcheck.py"
```

Rules:

- `command` is split with `shlex.split()` and run with `shell=false`.
- In sandbox mode, `command` is the evaluator-runner command. The runner sees a
  copied `/workspace`, receives `HELIX_EVALUATOR_ENDPOINT`, calls the sidecar,
  prints `HELIX_RESULT`, and exits.
- `runner_image` is the short-lived evaluator-runner image. It should contain
  the runner script and client dependencies, not private benchmark data.
- `image` is the long-lived private evaluator service image. It should contain
  evaluator code, benchmark data, and expensive simulator dependencies.
- The long-lived sidecar does not mount `/workspace`; the runner must stream
  the needed candidate files/data over RPC, or execute candidate code itself and
  call the sidecar only for private judging/simulation.
- The sidecar container stays warm for the full HELIX run, has no published
  host ports, and is attached only to the private evaluator Docker network.
- Mutation agent containers do not join the evaluator network and do not
  receive the sidecar endpoint.
- `extra_commands` run after the main command in the same evaluator-runner
  workspace.
- `protected_files` are hashed at run start and checked after mutations/merges.
  They are only for non-sandboxed local prototypes; do not use repo-local
  evaluator files as the Docker sandbox security boundary.

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
background = "Only modify src/solver.py and supporting implementation files."
```

Use `background` as the agent's project-specific guardrail. It is not a
security boundary. In sandboxed runs, evaluator secrecy comes from the sidecar
network boundary, not from telling the agent not to edit evaluator files.

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
- Evaluator sidecar starts once per `helix evolve` on a private internal Docker
  network.
- Evaluator-runner containers run in fresh copied `/workspace` directories,
  use `[evaluator.sidecar].runner_image`, join the private evaluator network,
  call the sidecar, and exit.
- The sidecar sees only what the runner sends over the endpoint.
- Evaluator-runner changes are discarded.
- Agent auth volume mounts at `/home/node`.
- Evaluator containers do not get agent auth volumes.
- `helix.toml`, `.env`, `.env.*`, `.git`, and HELIX artifacts are excluded
  from sandbox workspace copies/sync-back.
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
