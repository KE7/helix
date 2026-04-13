# web_researcher — HELIX Integration Test

This test demonstrates a class of problem that **GEPA fundamentally cannot solve**:
evolving an agent that must browse the internet to stay accurate over time.

## The Problem

Software engineering constantly changes: libraries release new versions weekly,
APIs deprecate methods, status codes get updated in documentation, and language
features get added or removed. An agent that only uses LLM training knowledge will
give stale answers as soon as its training cutoff passes.

**The task:** Answer 10 software engineering questions requiring current knowledge.
Examples:
- *"What is the latest stable version of the `anthropic` Python package?"*
- *"What HTTP status code does GitHub return for secondary rate limit violations?"*
- *"What is the latest version of `numpy` on PyPI?"*

The **naive seed agent** (`agent.py`) asks Claude Haiku directly — no web access.
It gets easy/stable questions right but fails on version-specific ones as time passes.

In a fresh 2-generation HELIX run from this naive seed, the best evolved
candidate learned to fetch live data from PyPI and other current sources and
reached `1.0` validation accuracy. That snapshot is saved as
`agent_optimized.py`.

---

## Why GEPA Cannot Do This

GEPA is a *blind one-shot code generator*: it receives a spec and emits code, but
cannot iterate, verify, or access the internet during the mutation step. This creates
three insurmountable barriers:

### 1. GEPA Cannot Browse the Web During Mutation

GEPA generates code in a single pass using only the system prompt and one code file.
It cannot:
- Discover what URLs to hit (PyPI JSON API, python.org download page, endoflife.date)
- Test whether a `requests.get(url)` call actually returns the expected JSON shape
- Iterate on the fetch logic until it correctly extracts the version number

HELIX's mutation step is *Claude Code with web browsing enabled*. During mutation,
HELIX can open `https://pypi.org/pypi/requests/json` in a browser, inspect the
JSON structure, and write the correct `data["info"]["version"]` extraction logic.
It can verify the agent returns `"2.32.3"` (or whatever the live version is) before
committing the change.

### 2. GEPA Operates on One File — Cannot Coordinate a Multi-File Agent

An effective web-fetching agent benefits from decomposition:
- `agent.py` — orchestration, LLM fallback, answer normalization
- `fetcher.py` — URL fetching, retry logic, HTTP error handling
- `cache.py` — disk-based TTL cache to avoid re-fetching during evaluation

GEPA mutates a single file atomically. It cannot create new modules, split
responsibilities across files, or maintain a shared cache between modules. A
single-file solution that inlines HTTP logic, caching, and LLM fallback becomes
unwieldy and fragile.

HELIX evolution operates on the *entire working directory*. It can create
`fetcher.py` on generation 2, extract cache logic to `cache.py` on generation 4,
and refine the orchestration in `agent.py` independently. Each file evolves
with a clear responsibility.

### 3. GEPA Cannot Run the Agent Mid-Mutation to Verify Web Requests Work

The hardest part of writing a web-fetching agent is not the concept — it's the
plumbing: handling timeouts, 404s, unexpected JSON shapes, rate limits, and
network failures gracefully. GEPA cannot run `python agent.py` inside the mutation
loop to check that `solve("What is the latest numpy version?")` returns `"2.2.2"`
rather than `""` or raising an exception.

HELIX's inner loop runs the configured evaluator command after *every mutation*.
In this example that should be `uv run python evaluate.py`, not bare
`python3 evaluate.py`. An agent
that raises `ConnectionError` on a bad URL gets score 0 and is discarded.
An agent with a 10-second timeout that correctly falls back to a cached value
gets a higher score and survives. This self-verification loop is what produces
robust, production-quality web integration — not just code that looks right.

---

## How HELIX Overcomes These Barriers

| Capability | GEPA | HELIX |
|---|---|---|
| Web browsing during mutation | ✗ | ✓ Claude Code with browser |
| Multi-file coordination | ✗ (1 file) | ✓ Entire directory |
| Run agent to verify mid-mutation | ✗ | ✓ evaluate.py after each mutation |
| Retry failing mutations | ✗ | ✓ Frontier selection + crossover |
| Test on train set, validate on val | ✗ | ✓ SPLIT env var |

**Observed fresh run:**

- **Generation 0 (seed):** `agent.py` scored `0.4` on the held-out validation set.
  It got the stable facts right and the live version questions wrong.
- **Generation 1:** HELIX evolved a candidate that replaced stale hardcoded
  package versions with live fetches.
- **Generation 2:** The best frontier candidate (`g1-s3`) reached `1.0`
  validation accuracy and is preserved in `agent_optimized.py`.

---

## Evaluator Design

`evaluate.py` is itself web-aware. For questions with `"ground_truth": "FETCH:..."`,
the evaluator fetches the live answer from PyPI or other sources. This means:

- The **ground truth is always current** — no stale expected values.
- An agent that fetches from web will **always match** the evaluator's reference.
- A naive agent using training data will **drift** as time passes.

Match modes:
- `exact`: normalized string equality (for status codes, module names)
- `version_prefix`: agent can answer `"3.13"` when current is `"3.13.4"` — a
  correct major.minor is accepted for partial credit

---

## Running the Test

```bash
cd examples/web_researcher

# Run on validation split (default)
uv run python evaluate.py

# Run on training split
HELIX_SPLIT=train uv run python evaluate.py

# Initialize HELIX and start evolution
helix init
helix evolve
```

---

## File Layout

```
web_researcher/
├── helix.toml          # HELIX project config
├── agent.py            # Naive seed: Claude Haiku, no web access
├── agent_optimized.py  # Best candidate from a fresh 2-generation run
├── evaluate.py         # Evaluator with live ground truth fetching
├── README.md           # This file
├── train/              # 5 training questions (used during HELIX mutation)
│   ├── q01_python_version.json
│   ├── q02_requests_version.json
│   ├── q03_numpy_version.json
│   ├── q04_github_ratelimit_status.json
│   └── q05_urllib2_replacement.json
└── val/                # 5 validation questions (held-out evaluation)
    ├── q06_anthropic_version.json
    ├── q07_httpx_version.json
    ├── q08_github_ratelimit_count.json
    ├── q09_asyncio_coroutine_removal.json
    └── q10_pydantic_version.json
```
