# Migrating From GEPA `optimize_anything.py` To HELIX

Use this file when translating a GEPA `optimize_anything(...)` workflow into a
HELIX project.

## Contents

- Mental Model Shift
- Basic Mapping
- Evaluator Translation
- Single-Task GEPA Mode
- Multi-Objective Migration
- Config Translation Example
- Candidate Representation
- Dataset ID Strategy
- Docker Sandbox In Migration
- Migration Checklist

## Mental Model Shift

GEPA `optimize_anything`:

```text
candidate object/string
  -> Python evaluator(candidate, example)
  -> GEPA proposer mutates candidate text
```

HELIX:

```text
whole repository candidate
  -> evaluator command in a worktree
  -> coding agent mutates files
```

Migration usually means:

1. Put the candidate artifact into repo files, e.g. `solve.py`, prompts,
   config, model wrapper, or policy code.
2. Put evaluator code and benchmark data in a private sidecar image; keep them
   out of the candidate workspace.
3. Configure `helix.toml` so HELIX owns evolution and calls the evaluator.

## Basic Mapping

| GEPA optimize_anything | HELIX |
|---|---|
| `candidate` initial value | initial repo contents, or `[seedless]` |
| `evaluate(candidate)` | evaluator-runner command calling the sidecar |
| `dataset` | `[dataset].train_size`, `[dataset].val_size`; evaluator owns actual data |
| per-example score | `HELIX_RESULT` entry score |
| `side_info` / trajectory | per-example `side_info` in `HELIX_RESULT` |
| `side_info["scores"]` objective metrics | reserved `side_info["scores"]` dict |
| `reflection_minibatch_size` | `[evolution].minibatch_size` |
| `num_parallel_proposals` | `[evolution].num_parallel_proposals` |
| `max_metric_calls` | `[evolution].max_evaluations` |
| `frontier_type` | `[evolution].frontier_type` |
| `cache_evaluation` | `[evolution].cache_evaluation` |
| `merge=MergeConfig(...)` | `[evolution].merge_enabled = true` plus merge fields |
| GEPA proposer model | `[agent]` coding backend/model/effort |

## Evaluator Translation

GEPA single-example evaluator:

```python
def evaluate(candidate, example):
    output = run_candidate(candidate, example)
    return score, {"trace": output, "scores": {"accuracy": score}}
```

HELIX sidecar-backed evaluator runner:

```python
import json
from pathlib import Path
import os
import requests

ENDPOINT = os.environ["HELIX_EVALUATOR_ENDPOINT"]

ids = json.loads(Path("helix_batch.json").read_text())
response = requests.post(
    ENDPOINT,
    json={
        "ids": ids,
        "candidate": Path("solve.py").read_text(),
    },
    timeout=300,
)
response.raise_for_status()
payload = response.json()["results"]

print("HELIX_RESULT=" + json.dumps(payload))
```

Configure:

```toml
[evaluator]
command = "python /runner/evaluate_client.py"
score_parser = "helix_result"

[evaluator.sidecar]
image = "my-private-evaluator:latest"
runner_image = "my-evaluator-runner:latest"
command = "python -m benchmark_server"
endpoint = "http://helix-evaluator:8080/evaluate"

[dataset]
train_size = 100
val_size = 100
```

## Single-Task GEPA Mode

If GEPA had no dataset and only optimized one aggregate objective, HELIX can use
a simpler parser:

```python
import json
score = evaluate_candidate()
print(json.dumps({"score": score}))
```

```toml
[evaluator]
command = "python /runner/evaluate_client.py"
score_parser = "json_score"

[evaluator.sidecar]
image = "my-private-evaluator:latest"
runner_image = "my-evaluator-runner:latest"
command = "python -m benchmark_server"
endpoint = "http://helix-evaluator:8080/evaluate"
```

Use `helix_result` anyway when you want side info in reflection prompts or
future multi-axis scoring.

## Multi-Objective Migration

GEPA `optimize_anything` supports `FrontierType` values:

```text
instance | objective | hybrid | cartesian
```

HELIX supports the same values:

```toml
[evolution]
frontier_type = "hybrid"
```

For objective/hybrid/cartesian, emit per-example named objective scores:

```python
payload.append([primary_score, {
    "scores": {
        "success": success,
        "latency": -latency,      # higher is better
        "format": format_score,
    },
    "notes": "diagnostic text for reflection",
}])
```

All HELIX scores are treated as higher-is-better. Negate costs/losses.

## Config Translation Example

GEPA:

```python
config = GEPAConfig(
    engine=EngineConfig(
        max_metric_calls=500,
        num_parallel_proposals=4,
        max_workers=32,
        frontier_type="hybrid",
        cache_evaluation=True,
    ),
    reflection=ReflectionConfig(reflection_minibatch_size=3),
    merge=MergeConfig(max_merge_invocations=5),
)
```

HELIX:

```toml
[evolution]
max_evaluations = 500
num_parallel_proposals = 4
max_workers = 32
frontier_type = "hybrid"
cache_evaluation = true
minibatch_size = 3
merge_enabled = true
max_merge_invocations = 5
```

## Candidate Representation

For GEPA text candidates, choose one:

- Put text in `candidate.txt`; evaluator reads it.
- Put prompts in `prompts/*.md`; agent edits prompt files.
- Put executable code in `solve.py`; evaluator imports and tests it.
- For multi-file systems, let the whole repo be candidate state.

Give the agent a precise `[agent].background`:

```toml
[agent]
background = """
The candidate is solve.py and prompts/system.md.
Improve correctness on the benchmark while preserving the public API.
"""
```

## Dataset ID Strategy

HELIX writes string ids. For positional datasets:

```python
ids = json.loads(Path("helix_batch.json").read_text())
examples = [DATA[int(i)] for i in ids]
```

For task/trial datasets:

```python
task, trial = eid.split("__", 1)
```

Then:

```toml
[evolution]
batch_sampler = "stratified"
group_key_separator = "__"
```

Use stratified sampling when each minibatch should cover multiple tasks.

## Docker Sandbox In Migration

For GEPA workflows that relied on a local Python process, HELIX adds coding
agents and should normally use Docker sandboxing:

```toml
[sandbox]
enabled = true
network = "bridge"
skip_special_files = true
```

Run once per backend:

```bash
helix sandbox login claude
```

If the old GEPA evaluator depended on local packages, either:

- build them into the sidecar image if they are evaluator/benchmark
  dependencies, or
- build them into `runner_image` if they are evaluator-runner dependencies.

## Migration Checklist

1. Identify the candidate artifact and put it in repo files.
2. Build a private evaluator sidecar image with evaluator code and data.
3. Write an evaluator-runner command that reads `helix_batch.json`, calls
   `HELIX_EVALUATOR_ENDPOINT`, and prints `HELIX_RESULT=...`.
3. Emit `HELIX_RESULT=` with one entry per requested id.
4. Add `[evaluator.sidecar]` with image, command, and endpoint.
5. Add `[dataset]` cardinalities.
6. Map GEPA engine/reflection/merge config into `[evolution]`.
7. Configure `[agent]` with backend, model, max turns, and background.
8. Enable `[sandbox]` and log in the backend.
9. Run evaluator manually, then `helix evolve`.
10. Inspect `helix frontier`, `helix best`, and `helix log`.
