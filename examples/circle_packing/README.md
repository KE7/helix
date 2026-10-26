# circle_packing

optimize_anything's circle-packing experiment, run by HELIX. Pack 26
non-overlapping circles inside the unit square so as to maximize the sum of
their radii.

The experiment is GEPA's own: the seed, the candidate interface, the scoring,
the refiner and the budget come from
[gepa-ai/gepa@133d185a `examples/circle_packing`](https://github.com/gepa-ai/gepa/tree/133d185a/examples/circle_packing).
Only the proposer differs: HELIX's mutation agent edits the candidate where
GEPA's reflection LM proposes new text.

## The candidate

A candidate has two components, each in a file:

- `solve.py` (`code`) defines `main(timeout, current_best_solution)` and
  returns `{'circles': <(26, 3) array of (x, y, r)>, 'all_scores': [...]}`.
  `current_best_solution` is the best packing found so far in the run, or
  `None`.
- `refiner_prompt.txt` (`refiner_prompt`) instructs the refiner.

The seed files are GEPA's `SEED_CODE` and `SEED_REFINEMENT_PROMPT`, byte for
byte. No other file of the candidate is evaluated.

## Scoring

`evaluate.py` scores a candidate with GEPA's fitness function
(`gepa_fitness.py`):

1. `solve.py` runs in a fresh Python subprocess with a 600-second limit,
   warm-started with the run's best packing.
2. Its packing is valid when it has 26 rows, no NaN, no negative radius, every
   circle inside the square and no two overlapping, with an absolute tolerance
   of 1e-6. A valid packing scores the sum of its radii; anything else scores 0.
3. The refiner (`openai/gpt-5.1`, temperature 1.0, 32,000 tokens) is given
   `refiner_prompt.txt`, the code and its results, and returns rewritten code,
   which runs and is scored the same way.
4. The candidate scores the better of the two. `side_info["scores"]` carries
   both scores and their maximum, the objectives of the Pareto frontier.

The run's StateTracker caches code and `(code, refiner_prompt)` results, so
an unchanged candidate never runs again, and keeps the run's best packing. It
lives in `.helix/gepa_state_tracker/`; `state_tracker_logs.json` there has
GEPA's log format, including the code that produced the best packing.

## How to run

The mutation agent runs in HELIX's Docker sandbox with no web access, on the
network that [`../circle_packing_sandbox/up.sh`](../circle_packing_sandbox/up.sh)
starts: an `--internal` network whose only route out is a gateway for the
Anthropic Messages API. The evaluator runs on the host. Copy the example to a
standalone directory, then run HELIX there:

```bash
helix sandbox login claude
examples/circle_packing_sandbox/up.sh
export OPENAI_API_KEY=...
work_dir="$(mktemp -d)/circle_packing"
cp -R examples/circle_packing "$work_dir"
cd "$work_dir"
helix evolve
```

`helix.toml` sets GEPA's budget of 200 metric calls, its cache, an objective
frontier, a minibatch of 1 and no merge. The objective and the agent
background are GEPA's text. The agent is Claude Haiku 4.5 in Claude Code, with
its built-in web tools off (`web_access = false`). See
[`docs/experiments/circle-packing.md`](../../docs/experiments/circle-packing.md)
for what is identical to GEPA and what is not.

## Files

- `solve.py`, `refiner_prompt.txt`: the seed candidate.
- `evaluate.py`: HELIX evaluator; prints one `HELIX_RESULT=` pair per id in
  `helix_batch.json`.
- `gepa_fitness.py`: GEPA's fitness function, StateTracker, executor,
  `validate_packing` and refiner.
- `run_main.py`: runs one candidate subprocess.
- `requirements.in`, `requirements.txt`: the experiment's Python environment
  and its lock.
- `helix.toml`: run configuration. Every evaluator file is protected, so a
  candidate that changes one is rejected.
