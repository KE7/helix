# web_researcher

This example evolves a small question-answering agent for software engineering
questions whose answers change over time.

## The Problem

Software engineering constantly changes: libraries release new versions weekly,
APIs deprecate methods, status codes get updated in documentation, and language
features get added or removed. An agent that only uses LLM training knowledge will
give stale answers as soon as its training cutoff passes.

**The task:** Answer 10 software engineering questions about library versions,
API behavior, and language changes, 5 in `train/` and 5 in `val/`. Examples:
- *"What is the latest stable version of the `anthropic` Python package?"*
- *"What HTTP status code does GitHub return for secondary rate limit violations?"*
- *"What is the latest version of `numpy` on PyPI?"*

The **seed agent** (`agent.py`) asks Claude Haiku directly, with no web access.
Its answers come from the model's training data, so answers to version questions
fall behind as new releases ship.

---

## HELIX Features Used in This Example

HELIX's search loop is ported from GEPA `optimize_anything`; see the
[main README](../../README.md#design).

Three HELIX properties matter for this task.

### 1. The mutation agent can use its backend's tools

HELIX does not restrict the backend's tools. With a backend that can fetch web
pages, the mutation agent can consult live sources while it edits the
candidate.

### 2. The candidate is the whole directory

HELIX evolves the *entire working directory*. A mutation may split the agent
into modules, for example `fetcher.py` for URL fetching and retries or
`cache.py` for a TTL cache, instead of keeping everything in `agent.py`.

### 3. The agent can run its own code mid-mutation

Inside a mutation, the agent can run `agent.py` on a question to check an
answer before the candidate is scored. After every mutation, HELIX runs the
configured evaluator command, which scores an exception or an empty answer as 0
for that question.

## Results

Run the example to produce results.

---

## Evaluator Design

`evaluate.py` is itself web-aware. It reads requested ids from
`helix_batch.json` and emits one positional `HELIX_RESULT=[score, side_info]`
pair per id. HELIX writes positional ids (`"0"` to `"4"`) because `helix.toml`
sets `[dataset]` sizes; the evaluator maps them to the split's questions sorted
by file name. A question whose `ground_truth` is `FETCH:pypi:<package>` expects
the package's current version on PyPI, and `FETCH:cpython` expects the latest
CPython release listed by endoflife.date. The evaluator fetches these values
when it runs. This means:

- Expected answers track current releases. If a source cannot be reached, the
  evaluator falls back to the question's `ground_truth_static_fallback` value.
- An answer is scored against the value published at evaluation time.

Match modes:
- `exact`: normalized string equality (for status codes, module names)
- `version_prefix`: the answer equals the expected version, or gives at least
  major.minor and those components equal the leading components of the expected
  version. For an expected `"3.13.4"`, `"3.13"` and `"3.13.4"` match; `"3"`,
  `"3.1"`, and `"3.13.40"` do not.

`helix.toml` protects `evaluate.py`, `train/`, and `val/`, so a candidate that
modifies them is rejected. Its `[sandbox]` section runs mutation agents in
Docker workspace copies that leave out those paths
(`omit_from_agent`); evaluation runs on the host in the candidate worktree.

---

## Running the Example

The seed agent calls the Anthropic API, so set `ANTHROPIC_API_KEY`. HELIX scrubs
the environment of evaluator and agent subprocesses, and `helix.toml` lists
`ANTHROPIC_API_KEY` in `passthrough_env` so the evaluator receives it. HELIX
also passes `ANTHROPIC_API_KEY` to the `claude` mutation backend, so mutation
sessions authenticate with this API key.

Mutation agents run in the Docker sandbox. Install and start Docker, then log
in to the backend's sandbox auth volume once:

```bash
helix sandbox login claude
helix sandbox status claude
```

Copy the example to a standalone directory, then run HELIX there:

```bash
work_dir="$(mktemp -d)/web_researcher"
cp -R "$(git rev-parse --show-toplevel)/examples/web_researcher" "$work_dir"
cd "$work_dir"

# Score the seed on the validation split (default) or the training split
uv run --no-project --python 3.12 --with anthropic python evaluate.py
HELIX_SPLIT=train uv run --no-project --python 3.12 --with anthropic python evaluate.py

# Start evolution
helix evolve
```

`helix evolve` initializes the copy as a git repository and commits it as the
seed.

---

## File Layout

```
web_researcher/
├── helix.toml          # HELIX project config
├── agent.py            # Seed agent: asks Claude Haiku directly, no web access
├── evaluate.py         # Evaluator with live ground truth fetching
├── README.md           # This file
├── train/              # 5 training questions (used during HELIX mutation)
│   ├── q01_python_version.json
│   ├── q02_requests_version.json
│   ├── q03_numpy_version.json
│   ├── q04_github_ratelimit_status.json
│   └── q05_urllib2_replacement.json
└── val/                # 5 validation questions
    ├── q06_anthropic_version.json
    ├── q07_httpx_version.json
    ├── q08_github_ratelimit_count.json
    ├── q09_asyncio_coroutine_removal.json
    └── q10_pydantic_version.json
```
