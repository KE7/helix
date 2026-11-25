# Writing an evaluator

The evaluator is the one piece of code you write for HELIX. HELIX runs it in each candidate's worktree and reads one line from its output: the score, plus feedback the next mutation can learn from.

!!! note "The contract"
    - Print exactly **one** line that starts with `HELIX_RESULT=`, followed by a JSON list with one entry per example, in the order of `helix_batch.json`. Each entry is a score or a `[score, side_info]` pair.
    - **Higher is better.** Scores must be finite numbers (no `NaN`); `true` and `false` count as 1 and 0. They need not lie in [0, 1], but `perfect_score_threshold` and `stop_at_score`, if you set them, use the same scale (see [Configuration](../guides/configuration.md)).
    - Without a `HELIX_RESULT=` line there are no scores, and HELIX treats the evaluation as an error. A non-zero exit keeps the per-example scores it printed (the minibatch gate still uses them), and its stdout and stderr go to the next mutation as debug context.
    - HELIX writes the ids to evaluate into `helix_batch.json` in the working directory and sets `HELIX_SPLIT` to `train` or `val`. You own the dataset: map each id to an example the same way every time.
    - The evaluator runs with a scrubbed environment and no shell. List API keys and other variables it needs in `passthrough_env`.

## Single task

With no `[dataset]` section, the candidate is scored on one task, and `helix_batch.json` holds a single id:

```python
# evaluate.py
import json

score, feedback = run_my_benchmark()  # your code; higher is better
print("HELIX_RESULT=" + json.dumps([[score, {"feedback": feedback}]]))
```

## Train and validation splits

Set the split sizes, and HELIX samples ids from them: `"0"` to `"199"` for training here, as strings. To use named ids such as `"pushcube__0"` instead (which `evolution.batch_sampler = "stratified"` needs to group examples), point `[seedless].train_path` / `val_path` at directories of `*.json` files; the file stems become the ids (see [Datasets](../guides/datasets.md)). Either way, key your examples by id rather than by position:

```toml
[dataset]
train_size = 200
val_size = 50
```

```python
# evaluate.py
import json, os

split = os.environ["HELIX_SPLIT"]                  # "train" or "val"
examples = load_examples(split)                    # your code: a dict from id to example
ids = json.load(open("helix_batch.json"))          # e.g. ["3", "17", "42"]

results = []
for i in ids:
    ok, why = run_one(examples[i])                # your code
    results.append([1.0 if ok else 0.0, {"feedback": why}])
print("HELIX_RESULT=" + json.dumps(results))
```

## Testing by hand

Before `helix evolve`, run the command once from your project directory the way HELIX will:

```bash
echo '["0"]' > helix_batch.json
HELIX_SPLIT=train uv run python evaluate.py | grep '^HELIX_RESULT='
rm helix_batch.json
```

## Project environment

`[evaluator].command` runs without a shell, in a scrubbed environment. Point it at the interpreter that has your
evaluator's dependencies, for example:

```toml
[evaluator]
command = "uv run python evaluate.py"
```

or a wrapper script such as `bash run_eval.sh`. A bare `python3 evaluate.py` works only if that interpreter already
has everything your evaluator imports.

## Feedback

The next mutation reads `side_info`: HELIX puts it in the Diagnostics section of the agent's task prompt, after the
scores. A score alone tells the agent only that a case failed. Say why:

```python
# Not much to act on
[0.0, {"scores": {"accuracy": 0.0}}]
# Something to fix
[0.0, {"scores": {"accuracy": 0.0}, "feedback": "expected '3.13.2', got '3.12.0': used a cached version list"}]
```

Put in `feedback` what you would want to know when looking at a bad result. [`examples/web_researcher/evaluate.py`](https://github.com/KE7/helix/blob/main/examples/web_researcher/evaluate.py)
and [`examples/circle_packing/evaluate.py`](https://github.com/KE7/helix/blob/main/examples/circle_packing/evaluate.py)
are worked examples.

To score several objectives, put them in `side_info["scores"]` as named numbers, as above; an `objective`, `hybrid` or
`cartesian` frontier uses them.

For notes about the whole evaluation rather than one instance, call `from helix import log` and
`log("what happened", key=value)` in your evaluator. They appear as Evaluator Notes in the next task prompt.

The agent never sees the raw `HELIX_RESULT=` line. HELIX saves your evaluator's stdout and stderr with each
evaluation for debugging; the agent gets them when the evaluator exits non-zero (see the contract above), unless `evaluator.include_stdout` or `include_stderr` is off.
