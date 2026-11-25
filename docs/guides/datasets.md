# Datasets and evaluator integrity

## Dataset modes

Your evaluator loads and scores the examples. HELIX hands it a list of instance ids in `helix_batch.json` and sets
`HELIX_SPLIT` to `train` or `val`; your evaluator maps each id to an example. The ids come from one of three setups:

| Setup | Config | Ids |
|---|---|---|
| Single task | no `[dataset]` sizes and no `[seedless]` paths | one id, `"0"`, as in GEPA's single-task search. Each uncached evaluation counts as one metric call. |
| Numbered instances | `dataset.train_size`, `dataset.val_size` | `"0"` to `"train_size-1"` for training (sampled into minibatches) and `"0"` to `"val_size-1"` for validation. |
| Named instances | `seedless.train_path`, `seedless.val_path` | the sorted file names, without `.json`, of a directory of `*.json` files, such as `"pushcube__0"`. A JSON array or JSONL file gives numbered ids instead. |

HELIX reads the `[seedless]` paths for ids whether or not `seedless.enabled` is true. With `enabled = true`, the first
3 training examples also go into the prompt that generates the seed.

When more than one is set:

- Training ids come from `seedless.train_path` if set, otherwise from `dataset.train_size`.
- Validation ids come from `dataset.val_size` if set, otherwise from `seedless.val_path`, otherwise from
  `seedless.train_path` (validation then reuses the training ids).

## Grouped ids for stratified sampling

`evolution.batch_sampler = "stratified"` draws each minibatch across groups of instances. An id's group is the text
before the first `evolution.group_key_separator` (default `"__"`), so `"pushcube__0"` and `"pushcube__1"` are both in
group `pushcube`.

Only named ids have groups. Put one JSON file per instance in a directory, named `<group>__<anything>.json`, and
point `seedless.train_path` (and `val_path`) at it; `seedless.enabled` can stay `false`.
[`examples/maniskill_pick_place`](../experiments/maniskill-pick-place.md) does this with
`splits/train/pushcube__0.json` and so on. Numbered ids such as `"7"` have no separator, so each is a group of one and
there are no groups to balance.

## Staged validation

With validation ids and `evolution.val_stage_size = N`, a proposal that passes the gate is first scored on the first
`N` validation ids. Only if it passes that stage is it scored on the remaining ids, and HELIX combines the two into
one validation result. Only the combined result goes on the frontier and into the saved state.

Use it only when each instance's score is the same whatever other ids are in the batch. Leave it off if your
evaluator normalizes across examples, shares a warm-up between them, or derives an objective from the batch as a
whole.

During the first stage HELIX sets `HELIX_EVALUATION_PHASE=val_stage`. Your evaluator can return scores only then and
save feedback and other side effects for the full call.

## Evaluator integrity

List the files your score depends on in `evaluator.protected_files`, so a mutation or merge cannot raise its score by
editing the benchmark:

```toml
[evaluator]
command = "uv run python evaluate.py"
protected_files = [
  "evaluate.py",
  "goldens.json",
  "helpers/evaluator_utils.py",
]
```

At the start of a run, HELIX hashes the script in `[evaluator].command` and these files, and writes the hashes to
`.helix/evaluator_manifest.json`. A candidate that changes any of them is rejected before it is scored. Merge prompts
leave out protected files and `[sandbox].omit_from_agent` paths, even when a candidate's history holds them.

Protected files can still be read by an agent that runs without the sandbox. To keep them out of its sight, use the
[Docker sandbox](sandboxing.md) with `omit_from_agent`. To keep scoring code and data off your machine's candidate
copies entirely, run the evaluator in a sidecar (`[sandbox].evaluator = true` with `[evaluator.sidecar]`).
