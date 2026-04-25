---
name: helix
description: Use when creating, validating, or migrating HELIX optimization projects, especially when converting GEPA Optimize Anything examples to HELIX TOML while preserving evaluator behavior, metrics, Pareto semantics, and output artifacts.
metadata:
  short-description: Build and migrate HELIX optimization configs
---

# HELIX

Use this skill when a user wants to run HELIX, write a `helix.toml`, validate a HELIX setup, or migrate a `gepa.optimize_anything.optimize_anything(...)` benchmark to HELIX.

## Start With Local Source

HELIX behavior is defined by the checkout in front of you. Before writing or changing a migration, inspect the local files that govern the behavior you need:

- `src/helix/config.py`: authoritative TOML schema and defaults.
- `src/helix/parsers/helix_result.py`: per-example `HELIX_RESULT=` parser and strict evaluator contract.
- `src/helix/executor.py`: evaluator cwd, env scrub, `HELIX_SPLIT`, `HELIX_INSTANCE_IDS`, stdout parsing, and zero-fill warning behavior.
- `src/helix/mutator.py`: backend CLI arguments, prompt shape, diagnostics rendering, and seedless generation.
- `README.md` and `examples/*/helix.toml`: current user-facing CLI and config examples.
- `src/gepa/optimize_anything.py`, `src/gepa/adapters/optimize_anything_adapter/optimize_anything_adapter.py`, and the source example's `main.py`: source-side evaluator signature, config, outputs, and run-directory artifacts.

Do not edit benchmark examples unless that is explicitly part of the task. For migrations, prefer adding a HELIX wrapper/config around the existing benchmark code and protecting the evaluator, datasets, metrics, and output scripts from mutation.

## HELIX Model

HELIX evolves a whole working tree in isolated git worktrees under `.helix/worktrees/`. There is no `target_file`. The configured backend may read, edit, create, or delete files unless constrained by `agent.background` and by protected evaluator files.

Each generation generally follows:

1. Evaluate the current candidate with `evaluator.command`.
2. Select a parent from the Pareto frontier.
3. Mutate that worktree through the configured coding-agent backend.
4. Gate the mutation on a train minibatch.
5. Evaluate accepted candidates on validation ids and update the frontier.

Use `helix clean` only when the user wants to discard saved state and worktrees.

## Core `helix.toml`

Use the current `[agent]` section. Older local examples may still show backend-specific section names; prefer `[agent]` for new migrations because that is the schema in `src/helix/config.py`.

```toml
objective = "Describe exactly what the candidate should improve."
seed = "."
rng_seed = 0

# Optional: preserve dependency/cache/device variables through HELIX's env scrub.
# passthrough_env = ["OPENAI_API_KEY", "CUDA_VISIBLE_DEVICES", "HF_HOME"]

[evaluator]
command = "uv run python evaluate.py"
score_parser = "helix_result"
include_stdout = true
include_stderr = true
extra_commands = []
protected_files = [
  "evaluate.py",
  "main.py",
  "utils.py",
]

[dataset]
train_size = 100
val_size = 50

[evolution]
max_generations = 20
max_evaluations = 500  # metric-call budget: uncached examples, or 1 per uncached single-task/no-example eval call
minibatch_size = 3
max_workers = 32
num_parallel_proposals = 1
cache_evaluation = true
acceptance_criterion = "strict_improvement"
merge_enabled = false
frontier_type = "hybrid"

[agent]
backend = "codex"
model = "gpt-5.5"
max_turns = 20
background = """
Only modify the mutable candidate implementation files.
Do not edit evaluate.py, datasets, metrics, output writers, or benchmark fixtures.
Preserve public function names and output artifact formats.
"""

[worktree]
base_dir = ".helix/worktrees"
```

Required top-level fields are `objective` and `[evaluator].command`. All nested config models reject unknown keys, so validate TOML instead of assuming typos are ignored.

## Codex Backend

For Codex-backed HELIX runs:

```toml
[agent]
backend = "codex"
model = "gpt-5.5"    # Passed verbatim to `codex exec --model ...`; choose a model your Codex CLI supports.
max_turns = 20       # Rendered into the mutation prompt as a turn budget.
background = "Concrete mutation constraints and domain guidance."
```

HELIX invokes Codex as:

```bash
codex exec --json --dangerously-bypass-approvals-and-sandbox --model <model> <prompt>
```

`agent.effort` and `agent.allowed_tools` are part of the shared agent schema, but the current Codex invocation only passes `model` plus the prompt. GPT-5.5 defaults to medium reasoning effort when the host does not override it, so do not add effort settings to HELIX configs unless you have eval evidence for a different tradeoff. Put operational constraints in `agent.background`.

## Dataset Handoff

`[dataset]` holds cardinalities, not file paths. If `dataset.train_size` or `dataset.val_size` is set, HELIX writes `helix_batch.json` in the candidate worktree before each evaluator run:

```json
["0", "1", "2"]
```

The ids are opaque strings. The evaluator must read `helix_batch.json` from its current working directory and map each id to its own data store. With `evolution.batch_sampler = "stratified"`, ids may look like `"group__case_7"`; do not cast ids to integers unless your evaluator intentionally uses range-index ids.

HELIX also sets:

- `HELIX_SPLIT=train|val`
- `HELIX_INSTANCE_IDS` as a comma-separated copy of requested ids

Prefer `helix_batch.json` for `score_parser = "helix_result"` because the parser uses that same file to zip positional results back to ids.

## Evaluator Contract

For migrations from GEPA Optimize Anything, default to:

```toml
[evaluator]
command = "uv run python evaluate.py"
score_parser = "helix_result"
protected_files = ["evaluate.py", "main.py", "utils.py"]
```

The evaluator runs as a subprocess with `cwd` set to the candidate worktree. It must emit exactly one `HELIX_RESULT=` line on stdout when using `helix_result`:

```text
HELIX_RESULT=[[0.8, {"example_id": "0", "feedback": "..."}], [1.0, {"example_id": "1"}]]
```

The payload must be valid JSON and must be a list with the same length and order as `helix_batch.json`. Each entry may be:

- a bare numeric score, e.g. `0.75`
- a rich pair, e.g. `[0.75, {"feedback": "..."}]`
- mixed bare/rich entries in one list

HELIX normalizes bare scores to empty `side_info` dicts. Scores must be finite numbers; bools are accepted as `1.0` and `0.0`. `side_info` must be a dict or `null`.

Do not emit legacy batch-level shapes such as:

```text
HELIX_RESULT=[0.8, {"scores": {"accuracy": 0.8}}]
HELIX_RESULT=[[0.8, 1.0, 0.5], {"details": "..."}]
```

Those shapes are intentionally rejected. Emit one entry per example instead.

## Objective Scores And Frontier

GEPA Optimize Anything forwards multi-objective scores from `side_info["scores"]`. HELIX mirrors that with the `helix_result` parser:

```python
side_info = {
    "scores": {
        "sum_radii": score,
        "validity": validity_score,
    },
    "validation_details": details,
    "stdout": captured_stdout,
}
```

Use `evolution.frontier_type` according to the source run:

- `"instance"`: retain candidates by per-example validation score.
- `"objective"`: retain by objective names from `side_info["scores"]`.
- `"hybrid"`: retain by instance or objective axes. This is the HELIX default and matches GEPA Optimize Anything's default.
- `"cartesian"`: retain by `(example_id, objective_name)` pairs.

The train minibatch gate still compares primary per-example scores regardless of frontier type. Non-instance modes need `score_parser = "helix_result"` plus numeric `side_info["scores"]`; otherwise objective axes have no data.

## Migration Map

Map source concepts to HELIX this way:

| GEPA Optimize Anything source | HELIX destination |
| --- | --- |
| `seed_candidate="..."` | A seed file such as `prompt.txt`, `program.py`, `solve.py`, or `candidate.json` |
| `seed_candidate={"program": "..."}` | One file per mutable component, or a JSON file plus wrappers |
| `seed_candidate=None` | `[seedless] enabled = true` with `objective` and optional `seedless.train_path` |
| `evaluator(candidate, example)` | `evaluate.py` reads candidate files and selected ids, then prints `HELIX_RESULT=...` |
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
| `EngineConfig(run_dir=...)` outputs | `main.py` or post-run export wrapper writes the same artifacts after `helix best --export` |

HELIX does not directly implement every GEPA Optimize Anything feature. When source runs use `track_best_outputs`, `best_example_evals`, `opt_state` warm starts, refiner loops, callbacks, external trackers, or custom output directories, preserve the observable result with wrapper code and note any behavioral gap in the migration.

## Wrapper Patterns

### Mutable Program File

Use when `seed_candidate={"program": INITIAL_PROGRAM}` or `seed_candidate=SEED_CODE`.

Project layout:

```text
helix.toml
program.py              # mutable candidate
evaluate.py             # protected
main.py                 # protected export/test wrapper
utils/                  # protected helpers, datasets, simulators
```

`evaluate.py` should import or execute `program.py`, select examples from `helix_batch.json`, and emit one result per selected example:

```python
import json
import os
from pathlib import Path

from utils.dataset import train, val
from utils.metrics import evaluate_program


def selected_examples():
    split = os.environ.get("HELIX_SPLIT", "val")
    data = train if split == "train" else val
    ids = json.loads(Path("helix_batch.json").read_text())
    return [(eid, data[int(eid)]) for eid in ids]


def main():
    payload = []
    for eid, example in selected_examples():
        score, side_info = evaluate_program(Path("program.py"), example)
        side_info = dict(side_info or {})
        side_info.setdefault("example_id", eid)
        payload.append([float(score), side_info])
    print("HELIX_RESULT=" + json.dumps(payload))


if __name__ == "__main__":
    main()
```

Only cast `eid` to `int` for range-index datasets. For file-stem, grouped, or externally supplied ids, build an id-to-example map instead.

### Prompt File

Use when the source candidate is a string prompt.

```text
prompt.txt
evaluate.py
main.py
```

`evaluate.py` reads `prompt.txt` for each selected example, calls the original scoring helper, and places model outputs, reasoning, and feedback in per-example `side_info` so HELIX mutation prompts get useful diagnostics.

### JSON Candidate

Use when the source candidate is a dict of multiple text components.

```text
candidate.json
evaluate.py
main.py
```

Keep the JSON schema stable. If only one component should mutate, state that in `agent.background`. If several components may mutate, tell the backend which keys are mutable and which are fixed.

### `main.py` Export Wrapper

GEPA Optimize Anything examples often write `best_program.py`, `metrics.json`, candidate dumps, charts, or baseline/test scores after `result = optimize_anything(...)`. Preserve those artifacts by turning `main.py` into a HELIX run/export wrapper:

1. Resolve the same run directory flags or env vars as the source script.
2. Run `helix evolve --config helix.toml --dir .` or assume the run already exists.
3. Export the best worktree with `helix best --export <tmp-or-run-dir>/best-worktree --dir .`.
4. Copy the evolved mutable file to the historical artifact path, e.g. `best_program.py`.
5. Re-run baseline, optimized, and test-set reporting with the original helper functions.
6. Write `metrics.json`, candidates, charts, and output schemas expected by downstream scripts.

If preserving every candidate dump is important, document that HELIX stores candidate worktrees and state under `.helix/`; add an explicit exporter rather than changing evaluator behavior.

## Seedless Mode

When source uses `seed_candidate=None`, configure:

```toml
[seedless]
enabled = true
train_path = "data/train.jsonl"
```

`seedless.train_path` is only prompt grounding for seed generation. HELIX reads up to three examples from a JSON array, JSONL file, or directory of JSON files and includes them in the seed-generation prompt. Runtime sampling still comes from `[dataset]` and `helix_batch.json`.

## Validation

Run focused checks before a full evolution:

```bash
PYTHONPATH=/home/kd/research/helix/src python -c "from pathlib import Path; from helix.config import load_config; print(load_config(Path('helix.toml')))"
python -m py_compile evaluate.py main.py
printf '["0"]\n' > helix_batch.json
HELIX_SPLIT=train uv run python evaluate.py
rm -f helix_batch.json
```

For the evaluator smoke test, verify:

- stdout contains exactly one `HELIX_RESULT=` line.
- the JSON payload length equals `helix_batch.json` length.
- each entry is a finite score or `[finite_score, side_info_dict]`.
- objective frontier runs include numeric `side_info["scores"]`.
- protected files cover evaluators, datasets, metrics, wrappers, and stable output code.
- required API keys, package managers, datasets, and cache paths are covered by `passthrough_env` or documented as blockers.

Use HELIX commands from the project directory:

```bash
PYTHONPATH=/home/kd/research/helix/src helix evolve --config helix.toml --dir .
PYTHONPATH=/home/kd/research/helix/src helix resume --config helix.toml --dir .
PYTHONPATH=/home/kd/research/helix/src helix frontier --dir .
PYTHONPATH=/home/kd/research/helix/src helix best --dir .
PYTHONPATH=/home/kd/research/helix/src helix best --export ./best-worktree --dir .
PYTHONPATH=/home/kd/research/helix/src helix history --dir .
PYTHONPATH=/home/kd/research/helix/src helix log --dir .
```

## Pitfalls

- Do not use aggregate JSON parsers for Optimize Anything minibatch migrations unless the evaluator already returns correct id-keyed `instance_scores`; `helix_result` is safer for per-example parity.
- Do not key `HELIX_RESULT` by ids. It is positional to `helix_batch.json`.
- Do not print more than one `HELIX_RESULT=` line. HELIX treats that as an evaluator error.
- Do not let the backend mutate `evaluate.py`, datasets, metrics, simulation helpers, or output writers.
- Do not put dataset file paths under `[dataset]`; use evaluator-owned paths or `[seedless].train_path` only for seed prompt grounding.
- Do not assume `HELIX_INSTANCE_IDS` is enough for `helix_result`; the parser reads `helix_batch.json`.
- Do not forget HELIX's scrubbed environment. Add `passthrough_env` for API keys, GPU settings, cache dirs, or benchmark-specific env vars.
- Do not rely on source `run_dir` state files being interchangeable with `.helix` state. Preserve user-facing artifacts through export/post-processing.
- Do not switch `frontier_type` on a resumed run without understanding stored state; use a clean `.helix` directory or a separate run directory for a different frontier.
- Do not promise full equivalence for source features such as refiner loops, callbacks, tracking integrations, or `opt_state` warm starts unless the migration implements an explicit HELIX-side substitute.
