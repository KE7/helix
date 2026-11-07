# Datasets and evaluator integrity

## Dataset Modes

HELIX splits dataset concerns across two TOML sections:

| Section | Purpose |
|---|---|
| `[dataset]` | Cardinality only — `train_size` / `val_size` — drives the minibatch sampler when the evaluator owns the dataset and HELIX hands off example ids via `helix_batch.json` (Architecture A). |
| `[seedless]` | Seedless-mode toggle + optional prompt-grounding paths (`train_path` / `val_path`) — used only during seed generation to show the LLM representative inputs. |

| Mode | Config | Description |
|---|---|---|
| **Single-task / no-example** | neither set | GEPA O.A. Single-Task Search (`dataset=None`, `valset=None`): evaluator runs without example-id handoff; uncached eval calls count as 1 metric call. |
| **Example-id handoff** | `dataset.train_size` / `dataset.val_size` set | HELIX samples example ids — stringified indices into `range(train_size)` by default, or opaque task-prefixed ids like `"group_alpha__case_3"` under `evolution.batch_sampler = "stratified"`; the evaluator reads them from `helix_batch.json` (a JSON `list[str]`) in cwd and filters its own dataset. |
| **Seedless multi-task** | `seedless.enabled = true`, `seedless.train_path` set | Seed generation prompt includes the first 3 training examples for grounding. |

HELIX does not own separate dataset files for train/val; your evaluator remains
the source of truth. During evolution HELIX sets `HELIX_SPLIT` (`train` or `val`)
so evaluator-owned datasets can switch behavior by phase, mirroring GEPA's
`trainset` / `valset` duality.

When `evolution.val_stage_size` is positive and `dataset.val_size` is set, accepted mutation proposals first evaluate the deterministic first-N validation ids, then evaluate only the remaining ids and compose the per-id results. This is valid only when each per-id score and objective is independent of the requested id set; do not enable it for batch-relative metrics, cross-example normalization, shared warm-up, or objectives derived from an aggregate metric. Stage-only results are never added to the frontier; HELIX persists only the composed full-val result for Pareto ranking and resume stability. An evaluator may treat `HELIX_EVALUATION_PHASE=val_stage` as a score-only call and defer feedback or other non-score side effects until the promoted tail/full call.

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
