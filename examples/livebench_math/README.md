# HELIX LiveBench-Math parallel-proposals smoke demo

This is a faithful, laptop-scaled **smoke/subset experiment**, not a reproduction
of the GEPA publication. The publication used LiveBench-Math splits of 100 train,
100 validation, and 168 held-out test questions with a 5,000-metric-call budget.
It compared `P=2,N=2` with `P=1,N=1`, using `gpt-4.1-mini` as solver and
`gpt-5-mini` as proposer. This demo preserves those proposal shapes, the
Terrarium seed prompt and split algorithm, and the official LiveBench scorers,
but uses four train and four validation rows and does not claim a held-out test
result. It also has an explicit proposer-model deviation: Codex ChatGPT OAuth
returned HTTP 400 for `gpt-5-mini` in the pinned runner, so the actual smoke
uses supported proposer `gpt-5.4`. It is not described as equivalent to the
publication model. See [SOURCES.md](SOURCES.md) for immutable pins, IDs, and
attribution.

The protected evaluator uses the dated solver snapshot
`gpt-4.1-mini-2025-04-14`, temperature 1, maximum 32,000 output tokens, a
180-second request timeout, and zero retries. The HELIX mutation agent is
`gpt-5.4` with low reasoning effort and eight turns. Candidate code receives
only opaque split positions; questions, ground truth, scoring code, and the API
credential remain in a run-private sidecar. The sidecar has outbound network
access only because the pinned solver is an external API.
`OPENAI_API_KEY` is passed only through `[evaluator.sidecar].passthrough_env`;
the mutation sandbox uses the plain Codex runner image and cannot read the
baked dataset, official scorer modules, ground truth, or solver credential.

## Prerequisites

- Docker Desktop with at least 4 GiB available to a worker container
- Python 3.11+ and `uv`
- a clean Git worktree at HELIX commit `84c7bcd2b82a56c8dd5c18b7fe5828101b6a7023`
- `OPENAI_API_KEY` exported in the shell; never put it in this directory
- access to `ghcr.io/ke7/helix-evo-runner-codex:0.2.0`, GitHub, Hugging Face,
  and the OpenAI API during setup/run

Run every command below from this directory.

## Setup and scorer gate

```sh
docker build --progress=plain \
  -t helix-livebench-math:e2c8b590-smoke .
docker run --rm helix-livebench-math:e2c8b590-smoke \
  python3 /opt/livebench-math/smoke_score.py
uv run pytest ../../tests/examples/test_livebench_math.py -q
```

The one-row scorer command must report validation question
`4dc5a69ba4f2038bd73182b69e13d3669a77bfdc5fdaf8e41e615fafc51eb359`,
`correct_score: 1.0`, and `incorrect_score: 0.0`. It invokes the official pinned
LiveBench AIME scorer against the dataset's pinned ground truth; it does not call
the solver and therefore incurs no API cost.

## Run P=2,N=2

```sh
/usr/bin/time -p uv run helix evolve --config helix.toml --dir .
uv run helix log --dir .
uv run helix history --dir .
uv run helix frontier --dir .
uv run helix attempts --dir .
uv run helix best --dir .
python3 inspect_run.py --require-terminal
```

Expected durable artifacts are `.helix/state.json`, the proposal ledger embedded
in that state, candidate/evaluation records, and HELIX-managed candidate
worktrees. With `P=2,N=2`, every scheduled proposal wave must have two proposal
batches of two mutations, four distinct candidate IDs, and isolated Docker
containers/worktrees. Scores are official LiveBench per-row scores averaged by
HELIX; solver token usage is recorded in evaluator details. A 32-call ceiling is
configured, but cache hits and scheduler stopping can make actual calls lower.

## Resume and compare P=1,N=1

After an interruption, inspect `.helix/state.json`, then resume the same run:

```sh
uv run helix resume --dir .
```

Repeated `resume` after the terminal state must not spend more budget or add
candidate IDs. Run `python3 inspect_run.py --require-terminal` immediately
before and after; its state digest, candidate IDs, scores, ledger totals, and
metric-call budget must remain identical.

Clean the first run before the serial comparison; the two configs intentionally
share `.helix/`:

```sh
echo y | uv run helix clean --dir .
/usr/bin/time -p uv run helix evolve --config helix.1x1.toml --dir .
uv run helix log --dir .
uv run helix best --dir .
```

The comparison is only between two tiny stochastic smoke runs. It cannot be
used to reproduce or refute the publication's validation/test scores.

## Cleanup and zero-resource proof

Inspect anything needed first, then remove HELIX state/worktrees and the derived
image (the published base runner image is not task-created):

```sh
echo y | uv run helix clean --dir .
docker image rm helix-livebench-math:e2c8b590-smoke
test ! -e .helix
test -z "$(git worktree list --porcelain | grep '^worktree ' | grep '/.helix/' || true)"
test -z "$(docker ps -aq --filter label=org.helix.demo=livebench-math)"
test -z "$(docker network ls -q --filter label=org.helix.demo=livebench-math)"
! docker image inspect helix-livebench-math:e2c8b590-smoke >/dev/null 2>&1
git status --short -- .
```

Do not commit `.helix/`, benchmark rows, results, images/layers, caches, logs,
credentials, or temporary evidence. Local ignore rules enforce the common cases.
