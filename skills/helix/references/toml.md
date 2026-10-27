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

Score parser:

- `helix_result`: the only supported parser. This is the GEPA-compatible
  per-example contract: emit one positional
  `[score, side_info]` pair per id in `helix_batch.json`; use it for datasets,
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
# stop_at_score = 0.95  # end the run once the best val score reaches it; unset = off

minibatch_size = 3
num_parallel_proposals = 4
max_workers = 32
cache_evaluation = true
acceptance_criterion = "strict_improvement"
val_stage_size = 10
sibling_history = false  # show the mutator every earlier child of the same parent
sibling_summary_max_words = 150  # word budget for each of those summaries
# sibling_history_max_entries = 5  # prompt shows only the N most recent; unset = all

batch_sampler = "epoch_shuffled"  # or "stratified"
group_key_separator = "__"

frontier_type = "instance"  # "instance" | "objective" | "hybrid" | "cartesian"

merge_enabled = false
max_merge_invocations = 5
merge_val_overlap_floor = 5
merge_subsample_size = 5
```

Important choices:

- A run ends on `max_generations`, on `max_evaluations` (`<= 0` disables it),
  or on `stop_at_score`, whichever comes first. `stop_at_score` (unset by
  default; `helix evolve --stop-at-score X` overrides it) is checked at the
  start of every generation: once the best mean validation score of any
  accepted candidate is `>= stop_at_score`, the run stops the same way budget
  exhaustion does, with state saved and the frontier printed. It mirrors GEPA's
  `ScoreThresholdStopper`. `helix resume` under the same target stops again
  straight away; raise or unset it to keep evolving.
- `perfect_score_threshold` is not a stop. It is GEPA's skip-perfect gate: a
  parent whose minibatch scores all reach it is not mutated that generation,
  and the run carries on. Use `stop_at_score` to end a run at a score.
- The four frontier modes mirror GEPA's `FrontierType`; `frontier_type =
  "instance"` is the HELIX default and needs no objective
  scores beyond the per-example results from the built-in `helix_result` parser.
- Non-`instance` frontiers need per-example `side_info["scores"]`. Missing
  objective scores warn and no-op the objective axis for `hybrid`, which
  continues on its instance axis; `objective` and `cartesian` raise
  `MissingObjectiveScoresError` when parent selection needs that axis.
- `num_parallel_proposals` controls concurrent agent mutations per generation.
  Keep it low for expensive agent backends.
- `max_workers` bounds evaluation/mutation thread pools.
- Enable `merge_enabled` only when candidate changes can be meaningfully merged.

## Agent

```toml
[agent]
backend = "claude"  # agy | claude | codex | cursor | opencode
model = "sonnet"
effort = "medium"
max_turns = 20
background = "Only modify src/solver.py and supporting implementation files."
# web_access = false  # turn off the backend's built-in web tools
```

Use `background` as the agent's project-specific guardrail. It is not a
security boundary. In sandboxed runs, evaluator secrecy comes from the sidecar
network boundary, not from telling the agent not to edit evaluator files.

Backend notes:

- Agy: usually `agy`, optional model/effort.
- Claude: usually `claude`, `max_turns`, optional model/effort.
- Codex: usually `codex`, model optional.
- Cursor/OpenCode: check CLI support and model naming before pinning.

No web access:

- `web_access = false` turns off the backend's built-in web tools only. Web
  access is on by default. Your own settings still load, including MCP servers
  and plugins you configured, and some of those can reach the web (for
  example codex's browser plugin, a fetch MCP server, or claude.ai
  connectors). The shell can still reach the network too.
- A strict no-web guarantee is the Docker sandbox's job: a fresh home with no
  user config, plus a `[sandbox].network` with no route to the internet.
- Each backend's own flags apply it:
  - claude: `--disallowedTools=WebSearch,WebFetch`.
  - codex: `-c web_search="disabled" -c features.apps=false`. The account-side
    `codex_apps` connector searches the web even with `web_search` disabled,
    so this also turns off your ChatGPT app connectors for the run.
  - opencode: `OPENCODE_PERMISSION` denying `webfetch`/`websearch`, which
    overrides every config file.
  - agy and cursor have no such switch, so `web_access = false` is a
    configuration error for them, including through `helix evolve --backend`.
- `web_access = false` is pinned for resume: resuming with it changed in
  either direction is rejected, because a no-web comparison holds only if
  every proposal ran without web tools.

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
- After each agent invocation HELIX copies the CLI's own transcript for that
  session (claude `projects/-workspace/<id>.jsonl`, codex `sessions/.../
  rollout-*-<thread_id>.jsonl`, cursor `projects/*/agent-transcripts/<id>/`,
  agy `brain/<id>/.system_generated/logs/transcript*.jsonl`, opencode this
  session's rows exported from `opencode.db`) out of the auth volume into
  `.helix_artifacts/backend_transcripts/<backend>/<session_id>.jsonl`. This
  always happens for every backend; override `transcript_artifact_dir` /
  `claude_transcript_root` under `[sandbox]` for custom layouts. See
  `run-debug.md` for the full per-backend table.
- Sandboxed runs reset the backend CLI's session state in its auth volume at
  the start of every generation. Credentials and operator configuration are
  never touched; the copied transcripts are the only durable record of what
  the agent did.
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
