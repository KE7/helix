# Datasets and evaluator integrity

## Dataset Modes

HELIX takes the train and validation instance ids from one of two TOML sections:

| Section | Purpose |
|---|---|
| `[dataset]` | Cardinality only: `train_size` / `val_size`. The ids are `"0"` to `"n-1"`, handed to the evaluator in `helix_batch.json` (Architecture A). |
| `[seedless]` | `train_path` / `val_path`: a JSON array file, a JSONL file, or a directory of `*.json` files. HELIX reads its train and val ids from them **whether or not `enabled` is true**. With `enabled = true`, the first 3 train examples also go into the seed-generation prompt. |

| Mode | Config | Description |
|---|---|---|
| **Single-task / no-example** | none of the four keys set | GEPA O.A. Single-Task Search (`dataset=None`, `valset=None`): HELIX writes `helix_batch.json` holding the single id `["0"]` (and leaves `HELIX_INSTANCE_IDS` unset), and the evaluator prints one `[score, side_info]` pair for it; uncached eval calls count as 1 metric call. |
| **Example-id handoff, index ids** | `dataset.train_size` / `dataset.val_size` | The ids are the stringified indices `"0"` to `"train_size-1"` for training (sampled into minibatches) and `"0"` to `"val_size-1"` for validation; the evaluator reads them from `helix_batch.json` (a JSON `list[str]`) in cwd and filters its own dataset. |
| **Example-id handoff, named ids** | `seedless.train_path` / `seedless.val_path` pointing at directories | The ids are the sorted file stems of the `*.json` files, such as `"pushcube__0"`. A JSON array or JSONL file gives index ids `"0"` to `"n-1"` instead. The evaluator still loads the examples itself; HELIX reads only the ids (and, in seedless mode, the first 3 train examples). |
| **Seedless multi-task** | `seedless.enabled = true`, `seedless.train_path` set | As above, and the seed-generation prompt includes the first 3 training examples for grounding. |

Where each split's ids come from, when more than one key is set:

- Train ids: `seedless.train_path` if set, otherwise `dataset.train_size`.
- Validation ids: `dataset.val_size` if set, otherwise `seedless.val_path`,
  otherwise `seedless.train_path` (validation then reuses the train ids).

### Grouped ids for stratified sampling

`evolution.batch_sampler = "stratified"` groups ids by the text before the
first `evolution.group_key_separator` (default `"__"`): `"pushcube__0"` and
`"pushcube__1"` are both in group `"pushcube"`. HELIX never invents grouped
ids. To get them, put one `*.json` file per instance in a directory, named
`<group><separator><anything>.json`, and point `seedless.train_path` (and
`val_path`) at it; `enabled` can stay `false`.
[`examples/maniskill_pick_place`](../experiments/maniskill-pick-place.md)
does this with `splits/train/pushcube__0.json` and so on.

With only `dataset.train_size`, every id is a bare index such as `"7"`, which
has no separator, so each id is its own group of one and stratification
groups nothing.

Your evaluator remains the source of truth for the examples themselves.
During evolution HELIX sets `HELIX_SPLIT` (`train` or `val`)
so evaluator-owned datasets can switch behavior by phase, mirroring GEPA's
`trainset` / `valset` duality.

When `evolution.val_stage_size` is positive and validation ids exist (from `dataset.val_size` or the `[seedless]` paths), accepted mutation proposals first evaluate the deterministic first-N validation ids, then evaluate only the remaining ids and compose the per-id results. This is valid only when each per-id score and objective is independent of the requested id set; do not enable it for batch-relative metrics, cross-example normalization, shared warm-up, or objectives derived from an aggregate metric. Stage-only results are never added to the frontier; HELIX persists only the composed full-val result for Pareto ranking and resume stability. An evaluator may treat `HELIX_EVALUATION_PHASE=val_stage` as a score-only call and defer feedback or other non-score side effects until the promoted tail/full call.

## Evaluator Integrity

For local prototypes, HELIX can lock evaluator-critical files so
mutations and merges cannot game the score by editing the benchmark itself.
Runs with sandboxed evaluation (`[sandbox].evaluator = true`) should use
`[evaluator.sidecar]` instead of repo-local evaluator files.

```toml
[evaluator]
command = "uv run python evaluate.py"
protected_files = [
  "evaluate.py",
  "goldens.json",
  "helpers/evaluator_utils.py",
]
```

At run start, HELIX hashes the evaluator command target plus any
`evaluator.protected_files` entries and writes the manifest to
`.helix/evaluator_manifest.json`. Candidates that modify any protected file are
rejected before evaluation.
