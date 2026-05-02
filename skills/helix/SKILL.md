---
name: helix
description: Use when creating, validating, running, debugging, or migrating HELIX optimization projects, including GEPA Optimize Anything migrations, Docker sandbox setup, evaluator sidecars, TOML authoring, run monitoring, and output interpretation.
metadata:
  short-description: Build, run, debug, and migrate HELIX configs
---

# HELIX

Use this skill for operator-side HELIX setup. It is for the person writing
`helix.toml`, evaluator runners, and migration wrappers. Do not give this
context to a mutation agent.

## First Moves

1. Locate the project root: find `helix.toml`, `.helix/`, or ask for the target
   directory if neither exists.
2. Inspect `helix.toml`, evaluator sidecar settings, and the current git status.
3. Inspect local source before assuming schema or CLI behavior:
   `src/helix/config.py`, `src/helix/sandbox.py`, `src/helix/executor.py`,
   `src/helix/parsers/helix_result.py`, `src/helix/mutator.py`, `README.md`,
   and `examples/*/helix.toml`.
4. Prefer Docker sandboxing for new setups:
   `helix sandbox login <backend>`, then `[sandbox].enabled = true`.
5. Use `uv run helix ...` inside this repo, or `helix ...` for installed
   user projects.
6. For Docker sandboxing, require an evaluator sidecar. The evaluator must not
   live in the candidate workspace or be visible to mutation agents.

## Reference Routing

- Creating or fixing `helix.toml`: read `references/toml.md`.
- Running, monitoring, resuming, and debugging HELIX: read
  `references/run-debug.md`.
- Migrating from `../gepa/src/gepa/optimize_anything.py`: read
  `references/gepa-migration.md`.

## Core Model

HELIX evolves whole repositories. A mutation backend edits a detached candidate
worktree, then HELIX evaluates that candidate. Accepted candidates enter a
Pareto frontier. `helix best --export PATH` copies the best candidate out when
the run is done.

With Docker sandboxing enabled, HELIX copies each candidate into a temporary
agent container workspace. Agent changes sync back to the candidate worktree;
evaluator-side changes are discarded. HELIX also starts a warm evaluator
sidecar once per `helix evolve` on a private Docker network. Mutation agents
run on the normal agent network and cannot reach the sidecar. Evaluator-runner
containers run only during evaluation, join the private network, call the
sidecar, print `HELIX_RESULT`, and exit. Agent containers get a persistent
backend auth volume such as `helix-auth-claude`; candidate workspaces do not
persist between mutations except through HELIX's accepted worktree sync.

## Security Boundary

For new projects, set up two directories:

- A mutable candidate workspace with only the files the agent may inspect and
  edit.
- A private evaluator/benchmark directory or image with tests, goldens, data,
  secrets, services, and scoring logic.

When `[sandbox].enabled = true`, evaluator execution should use a sidecar. The
agent container should never receive evaluator source, evaluator secrets,
`helix.toml`, `.env*`, `.git`, or HELIX state. It should receive feedback only
through parsed `HELIX_RESULT` scores and side information.

Do not put evaluator endpoints, credentials, goldens, answer keys, or private
benchmark details in `agent.background`, candidate files, or mutation prompts.
Use `passthrough_env` only for non-secret runtime settings such as device and
cache hints unless the user accepts the exposure.

If a private sidecar directly executes candidate code, that candidate code can
observe anything in the process it runs inside. Prefer a narrow runner that
copies or mounts only the candidate artifacts it needs, runs them with a limited
interface, and returns sanitized scores, stdout, stderr, and feedback.

## Docker Sidecar TOML

Use this shape for new sandboxed projects. The evaluator command is a client
that runs in `runner_image` and talks to the long-lived private sidecar service.

```toml
objective = "Describe exactly what the candidate should improve."
seed = "."
rng_seed = 0

# Optional: preserve dependency/cache/device variables through HELIX's env scrub.
passthrough_env = ["CUDA_VISIBLE_DEVICES", "HF_HOME"]

[evaluator]
command = "python /runner/evaluate_client.py"
score_parser = "helix_result"
include_stdout = true
include_stderr = true

[evaluator.sidecar]
image = "my-private-evaluator:latest"
runner_image = "my-evaluator-runner:latest"
command = "python -m evaluator_server"
endpoint = "http://helix-evaluator:8080/evaluate"
startup_timeout_seconds = 120
# Optional when the default endpoint probe is not enough.
# healthcheck_command = "python /runner/healthcheck.py"

[dataset]
train_size = 100
val_size = 50

[evolution]
max_generations = 20
max_evaluations = 500
minibatch_size = 3
max_workers = 32
num_parallel_proposals = 4
cache_evaluation = true
acceptance_criterion = "strict_improvement"
frontier_type = "hybrid"

[agent]
backend = "codex"
model = "gpt-5.5"
max_turns = 20
background = """
Only modify the mutable candidate implementation files.
Preserve public function names and output artifact formats.
"""

[sandbox]
enabled = true
network = "bridge"
skip_special_files = true

[worktree]
base_dir = ".helix/worktrees"
```

Required top-level fields are `objective` and `[evaluator].command`. Nested
config models reject unknown keys, so validate TOML instead of assuming typos
are ignored.

Use repo-local evaluator files only for quick non-sandbox prototypes. In that
mode, set `evaluator.protected_files` for evaluator code, datasets, metrics,
and output wrappers, but do not treat that as the security boundary for
untrusted mutation agents.

## Dataset Handoff

`[dataset]` holds cardinalities, not file paths. If `dataset.train_size` or
`dataset.val_size` is set, HELIX writes `helix_batch.json` in the candidate
workspace before each evaluator run:

```json
["0", "1", "2"]
```

The ids are opaque strings. The evaluator client or sidecar must map each id to
its private data store. With `evolution.batch_sampler = "stratified"`, ids may
look like `"group__case_7"`; do not cast ids to integers unless your evaluator
intentionally uses range-index ids.

HELIX also sets:

- `HELIX_SPLIT=train|val`
- `HELIX_INSTANCE_IDS` as a comma-separated copy of requested ids
- `HELIX_EVALUATOR_ENDPOINT` for sidecar evaluator clients

Prefer `helix_batch.json` for `score_parser = "helix_result"` because the
parser uses that same file to zip positional results back to ids.

## Evaluator Contract

For `score_parser = "helix_result"`, HELIX writes `helix_batch.json` in the
candidate workspace before evaluation. The evaluator-runner reads it, calls the
sidecar endpoint from `HELIX_EVALUATOR_ENDPOINT`, and must print one line:

```text
HELIX_RESULT=[[0.8, {"example_id": "0", "feedback": "..."}], [1.0, {"example_id": "1"}]]
```

The payload must be valid JSON and must be a list with the same length and
order as `helix_batch.json`. Each entry may be a bare numeric score, a rich pair
such as `[0.75, {"feedback": "..."}]`, or mixed bare/rich entries in one list.
Scores must be finite numbers; bools normalize to `1.0` and `0.0`.
`side_info` must be a dict or `null`.

For objective, hybrid, or cartesian frontiers, put named objective values in
`side_info["scores"]`:

```python
side_info = {
    "scores": {
        "accuracy": accuracy,
        "latency": latency_score,
    },
    "feedback": feedback,
}
```

Do not emit legacy batch-level shapes such as:

```text
HELIX_RESULT=[0.8, {"scores": {"accuracy": 0.8}}]
HELIX_RESULT=[[0.8, 1.0, 0.5], {"details": "..."}]
```

Those shapes are intentionally rejected. Emit one entry per example instead.

## Running And Monitoring

Common commands:

```bash
helix init
helix sandbox login codex
helix sandbox status codex
helix evolve --config helix.toml --dir .
helix resume --config helix.toml --dir .
helix frontier --dir .
helix best --dir .
helix best --export ./best-worktree --dir .
helix history --dir .
helix log --dir .
helix clean --dir .
```

Use `helix clean` only when the user wants to discard saved state and worktrees.
Before a long run, smoke-test the evaluator command on a tiny batch and verify
that stdout contains exactly one `HELIX_RESULT=` line, the payload length equals
the batch length, each score is finite, and objective frontier runs include
numeric `side_info["scores"]`.

## GEPA Migration Map

Map source concepts to HELIX this way:

| GEPA Optimize Anything source | HELIX destination |
| --- | --- |
| `seed_candidate="..."` | A seed file such as `prompt.txt`, `program.py`, `solve.py`, or `candidate.json` |
| `seed_candidate={"program": "..."}` | One file per mutable component, or a JSON file plus wrappers |
| `seed_candidate=None` | `[seedless] enabled = true` with `objective` and optional `seedless.train_path` |
| `evaluator(candidate, example)` | A private evaluator service plus runner client that prints `HELIX_RESULT=...` |
| `dataset`, `valset` | Evaluator-owned data plus `[dataset].train_size` and `[dataset].val_size` |
| `objective=...` | top-level `objective = "..."` |
| `background=...` | `[agent].background = """..."""` |
| `EngineConfig(seed=N)` | top-level `rng_seed = N` |
| `EngineConfig(max_metric_calls=N)` | `[evolution].max_evaluations = N` |
| `EngineConfig(max_workers=N)` | `[evolution].max_workers = N` |
| `EngineConfig(num_parallel_proposals=N|"auto")` | `[evolution].num_parallel_proposals = N` or `"auto"` |
| `ReflectionConfig(reflection_minibatch_size=K)` | `[evolution].minibatch_size = K` |
| `EngineConfig(cache_evaluation=True)` | `[evolution].cache_evaluation = true` |
| `EngineConfig(acceptance_criterion=...)` | `[evolution].acceptance_criterion = "..."` |
| `EngineConfig(frontier_type=...)` | `[evolution].frontier_type = "..."` |
| `EngineConfig(run_dir=...)` outputs | `helix best --export` plus a post-run artifact writer |

HELIX does not directly implement every GEPA Optimize Anything feature. When
source runs use `track_best_outputs`, `best_example_evals`, `opt_state` warm
starts, refiner loops, callbacks, external trackers, or custom output
directories, preserve the observable result with wrapper code and note any
behavioral gap in the migration.

## Pitfalls

- Do not make the evaluator file part of the mutable workspace for sandboxed
  runs.
- Do not key `HELIX_RESULT` by ids. It is positional to `helix_batch.json`.
- Do not print more than one `HELIX_RESULT=` line.
- Do not put dataset file paths under `[dataset]`; use evaluator-owned paths or
  `[seedless].train_path` only for seed prompt grounding.
- Do not assume `HELIX_INSTANCE_IDS` is enough for `helix_result`; the parser
  reads `helix_batch.json`.
- Do not rely on source `run_dir` state files being interchangeable with
  `.helix` state. Preserve user-facing artifacts through export/post-processing.
- Do not switch `frontier_type` on a resumed run without understanding stored
  state; use a clean `.helix` directory or a separate run directory.
