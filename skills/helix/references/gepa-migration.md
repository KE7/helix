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
2. Write `evaluate.py` that loads the current repo candidate and evaluates
   selected examples.
3. Configure `helix.toml` so HELIX owns evolution and calls the evaluator.

## Basic Mapping

| GEPA optimize_anything | HELIX |
|---|---|
| `candidate` initial value | initial repo contents, or `[seedless]` |
| `evaluate(candidate)` | `[evaluator].command` running `evaluate.py` |
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

HELIX evaluator:

```python
import json
from pathlib import Path
from solve import solve

DATA = load_dataset()

ids = json.loads(Path("helix_batch.json").read_text())
payload = []
for eid in ids:
    example = DATA[int(eid)]  # or use opaque ids directly
    output = solve(example["input"])
    score = grade(output, example)
    payload.append([score, {
        "output": output,
        "expected": example.get("expected"),
        "scores": {"accuracy": score},
    }])

print("HELIX_RESULT=" + json.dumps(payload))
```

Configure:

```toml
[evaluator]
command = "uv run python evaluate.py"
score_parser = "helix_result"
protected_files = ["evaluate.py", "data/train.json", "data/val.json"]

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
command = "uv run python evaluate.py"
score_parser = "json_score"
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
Do not edit evaluate.py, data/, or expected outputs.
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

- make `[evaluator].command` set up/run the environment reproducibly, or
- build a custom sandbox image with those dependencies and set `[sandbox].image`.

## Migration Checklist

1. Identify the candidate artifact and put it in repo files.
2. Write `evaluate.py` that reads `helix_batch.json`.
3. Emit `HELIX_RESULT=` with one entry per requested id.
4. Add `protected_files` for evaluator and benchmark data.
5. Add `[dataset]` cardinalities.
6. Map GEPA engine/reflection/merge config into `[evolution]`.
7. Configure `[agent]` with backend, model, max turns, and background.
8. Enable `[sandbox]` and log in the backend.
9. Run evaluator manually, then `helix evolve`.
10. Inspect `helix frontier`, `helix best`, and `helix log`.
