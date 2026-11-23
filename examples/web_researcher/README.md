# web_researcher

HELIX evolves a small question-answering agent for software-engineering questions whose answers change over time,
such as the latest version of a package.

## The task

There are 10 questions, 5 in `train/` and 5 in `val/`, about library versions, API behaviour and language changes.
For example:

- *What is the latest stable version of the `requests` Python package on PyPI?*
- *What HTTP status code does the GitHub REST API return when a client exceeds the secondary rate limit?*
- *In Python 3, which module replaced `urllib2` for `urlopen`?*

The seed agent, `solve()` in `agent.py`, asks Claude Haiku with no web access. It answers from the model's training
data, so its version answers go stale as new releases ship. To do better, a mutation has to make the agent look the
answer up. The candidate is the whole directory, so the agent may add modules, for example one that fetches and
retries URLs or one that caches answers, and it can run `agent.py` on a question to check an answer before HELIX
scores it. With a backend that has web tools, the mutation agent itself can also look things up while it edits.

## How answers are scored

`evaluate.py` reads the ids in `helix_batch.json` (`"0"` to `"4"`, from the `[dataset]` sizes in `helix.toml`) and
maps them to the split's questions in file-name order. Expected answers for version questions are fetched when the
evaluator runs, so they follow new releases:

- `FETCH:pypi:<package>`: the package's current version on PyPI.
- `FETCH:cpython`: the latest CPython release on endoflife.date.

If a source cannot be reached, the evaluator uses the question's `ground_truth_static_fallback`.

Each question has a match mode:

- `exact`: the normalized strings are equal (status codes, module names).
- `version_prefix`: the answer is the expected version, or gives at least major.minor matching its leading parts.
  For an expected `3.13.4`, `3.13` and `3.13.4` match; `3`, `3.1` and `3.13.40` do not.

An exception or an empty answer scores 0 for that question, and the exception is in its feedback under `error`.

`evaluate.py`, `train/` and `val/` are protected: a candidate that changes them is rejected. The sandbox also leaves
them out of the agent's copy (`omit_from_agent`). The evaluator runs on your machine, in the candidate's worktree.

## Running it

The seed agent calls the Anthropic API, so set `ANTHROPIC_API_KEY`, with credit on it. `helix.toml` lists it in
`passthrough_env`, so the evaluator receives it, and the `claude` mutation agent uses it too.

The mutation agents run in Docker. Start Docker and log the sandbox in once:

```bash
helix sandbox login claude
helix sandbox status claude
```

Copy the example out of the repository and run HELIX in the copy:

```bash
work_dir="$(mktemp -d)/web_researcher"
cp -R "$(git rev-parse --show-toplevel)/examples/web_researcher" "$work_dir"
cd "$work_dir"

# Score the seed on the validation split (default), then on the training split.
# evaluate.py uses helix.timed, so it needs helix-evo 0.3.0 or newer.
uv run --no-project --python 3.12 --with anthropic --with 'helix-evo>=0.3.0' python evaluate.py
HELIX_SPLIT=train uv run --no-project --python 3.12 --with anthropic --with 'helix-evo>=0.3.0' python evaluate.py

helix evolve
```

`helix evolve` makes the copy a git repository and commits it as the seed. No recorded run is published for this
example yet.

## Files

```
web_researcher/
├── helix.toml     # HELIX project config
├── agent.py       # the seed agent: asks Claude Haiku, no web access
├── evaluate.py    # the evaluator; fetches current versions as it runs
├── train/         # 5 training questions, one JSON file each
└── val/           # 5 validation questions
```
