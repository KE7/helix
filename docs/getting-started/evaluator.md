# Writing an evaluator

The evaluator is the one piece of code you write for HELIX. HELIX runs it in each candidate's worktree and reads one line from its output: the score, plus feedback the next mutation can learn from.

!!! note "The contract"
    - Print exactly **one** line that starts with `HELIX_RESULT=`, followed by a JSON list with one entry per example, in the order of `helix_batch.json`. Each entry is a score or a `[score, side_info]` pair.
    - **Higher is better.** Scores must be finite numbers (no `NaN`, no booleans). They need not lie in [0, 1], but `perfect_score_threshold` and `stop_at_score`, if you set them, use the same scale (see [Configuration](../guides/configuration.md)).
    - A non-zero exit code or a missing `HELIX_RESULT=` line marks the evaluation as failed; its stdout and stderr go to the next mutation as debug context.
    - HELIX writes the ids to evaluate into `helix_batch.json` in the working directory and sets `HELIX_SPLIT` to `train` or `val`. You own the dataset: map each id to an example the same way every time.
    - The evaluator runs with a scrubbed environment and no shell. List API keys and other variables it needs in `passthrough_env`.

## A single task

With no `[dataset]` section, the candidate is scored on one task, and `helix_batch.json` holds a single id:

```python
# evaluate.py
import json

score, feedback = run_my_benchmark()  # your code; higher is better
print("HELIX_RESULT=" + json.dumps([[score, {"feedback": feedback}]]))
```

## A dataset with train and validation splits

Set the split sizes, and HELIX samples ids from them: `"0"` to `"199"` for training here, as strings. With `evolution.batch_sampler = "stratified"`, ids are opaque strings such as `"group_alpha__case_3"` instead, so key your examples by id rather than by position:

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

## Test it by hand

Before `helix evolve`, run the command once from your project directory the way HELIX will:

```bash
echo '["0"]' > helix_batch.json
HELIX_SPLIT=train uv run python evaluate.py | grep '^HELIX_RESULT='
rm helix_batch.json
```

## Minimal Example

```toml
objective = "Maximize test pass rate and code coverage"

[evaluator]
# evaluate.py must print one HELIX_RESULT=... line for HELIX to score it.
command = "uv run python evaluate.py"
```

When your evaluator needs project dependencies, make `evaluator.command` use the
same environment those dependencies are installed in. Good patterns are
`uv run python evaluate.py` or a wrapper like `bash run_eval.sh`. Avoid bare
`python3 evaluate.py` unless that interpreter already has everything your
evaluator imports.

## Score Parser

HELIX has one score parser: `helix_result`. Evaluators read the positional ids
from `helix_batch.json` and emit one `[score, side_info]` pair per id:

```python
print("HELIX_RESULT=" + json.dumps([
    [2.63, {"scores": {"sum_radii": 2.63}, "feedback": "..."}],
]))
```

For a batch, the payload must have the same length and order as
`helix_batch.json`. `side_info` is retained for reflection; its optional
`scores` dictionary supplies named objective axes.

`side_info` — not the score — is what the *next* mutation actually reasons
from: HELIX renders it into the mutation prompt's Diagnostics section, which
the agent reads before the number. `{"scores": {"accuracy": 0.0}}` alone
tells the next mutation only that it failed, not why; `{"scores": {"accuracy":
0.0}, "feedback": "expected '3.13.2', got '3.12.0' -- used a cached version
list"}` gives it something to act on. See
[`examples/web_researcher/evaluate.py`](https://github.com/KE7/helix/blob/main/examples/web_researcher/evaluate.py)
and [`examples/circle_packing/evaluate.py`](https://github.com/KE7/helix/blob/main/examples/circle_packing/evaluate.py)
for evaluators that emit this shape.

For GEPA-style reflective feedback, prefer structured per-example
`side_info` with the built-in result parser. For additional
human-readable notes that are not tied to one example, evaluators may call
`from helix import log` and then `log("what happened", key=value)`. HELIX
captures these notes through `HELIX_ASI_LOG`, not stdout, and renders them as
Evaluator Notes in mutation prompts.

Raw `HELIX_RESULT=...` JSON is only consumed by HELIX as a machine protocol and
is not shown to mutators. Evaluator stdout/stderr remain stored for debugging,
but successful mutation prompts omit them when structured diagnostics or
`helix.log()` notes are available; failed evaluations still include stdout and
stderr as fallback debug context.
