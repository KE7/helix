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
baked dataset, official scorer modules, ground truth, solver credential, or
non-prompt demo files. `sandbox.omit_from_agent` makes `prompt.txt` the only
candidate artifact visible to the mutator.

Terrarium's reflective evaluator may expose answer-oriented feedback. This demo
intentionally deviates at that interface for security: incorrect answers return
only the official numeric score and a generic retry message. Ground truth never
enters serialized side info or the rendered mutation/reflection prompt. The
official scorer computation itself is unchanged.

## Prerequisites

- Docker Desktop with at least 4 GiB available to a worker container
- Python 3.11+ and `uv`
- a clean HELIX 0.3.0 worktree built directly on the canonical release tip
  `c9371f4`, which carries release metadata `4622413` and core fixes
  `94f9751`, `402dcc8`, and `e5c260f`. Verify with:
  `git merge-base --is-ancestor c9371f4 HEAD && python -c "import helix; print(helix.__version__)"`
  (expect exit 0 and `0.3.0`). This lane adds only
  `examples/livebench_math/` and its test module on top of that tip.
- `OPENAI_API_KEY` exported in the shell; never put it in this directory
- a one-time `helix sandbox login claude`, which stores credentials in the
  `helix-auth-claude` Docker volume. No host key is passed into the runner.
- access to the pinned runner
  `ghcr.io/ke7/helix-evo-runner-claude@sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba`, GitHub, Hugging Face, and the OpenAI API during setup/run

### Runner provenance

The runner is pinned by a **registry manifest digest**, verifiable from any
host with `docker manifest inspect <image>`. This matters: an earlier revision
of this demo pinned a codex runner digest that was correctly *shaped* but had
never been published to GHCR. That hash was the local image **config ID**, a
different hash space from a registry manifest digest, so the demo only worked
on the one machine that happened to have the image cached. `RepoDigests` was
empty, which is the tell. `test_runner_pin_is_registry_resolvable_not_a_local_image_id`
now performs the network resolution so string-shaped fakes cannot pass again.

This image is published for `linux/amd64` only. On arm64 hosts it runs under
emulation, which is fine for the correctness-oriented run below but means
wall-clock timings from an arm64 host are not comparable to native ones.

The mutation engine is the claude runner rather than a codex runner. This does
not change what the benchmark measures: the agent only edits `prompt.txt`,
while the solver (`gpt-4.1-mini`) is invoked directly by the protected sidecar
and graded by the official LiveBench scorers. The proposer is outside the
scored path.

### Divergence: this lane containerizes the evaluator

This is the only HELIX 0.3.0 demo lane that sets `[sandbox] evaluator = true`.
The other lanes (`formulacode`, `swebench`, `algotune`) set it to `false`. The
divergence is required here because scoring must happen inside the protected
sidecar that holds the ground truth, never in the agent's trust domain.

It is sound despite the amd64 emulation described above because **nothing in
the scored path is time-dependent**: the official LiveBench scorers perform
exact numeric and string matching against ground truth, so emulation changes
how long scoring takes but not what it computes. If emulation ever pushed work
past a timeout, the sidecar's `except` branch returns a hard `0.0` with an
`error` field, so the failure surfaces as fail-closed and is distinguishable in
the ledger from a genuinely wrong answer (which has no `error` field and a
non-empty `output`). For that reason **no wall-clock numbers are reported from
this lane**.

The timeout chain is nested strictly outward, so no inner bound can outlive its
outer one:

| Bound | Value | Source |
|---|---|---|
| Solver call (no retries) | 180s | `SOLVER_TIMEOUT_SECONDS` |
| Evaluator batch call | 240s | `evaluate.run_client` |
| Sandbox container | 600s | `sandbox.timeout_seconds` |

### Credential isolation

The solver credential is sidecar-only, exercising core fix `94f9751`.
`OPENAI_API_KEY` is passed via `[evaluator.sidecar] passthrough_env` and is
absent from the agent's environment and from the agent's docker argv.

Note that HELIX re-adds each backend's auth variables **after** scrubbing, via
`BACKEND_AUTH_ENV`. Asserting only on `_scrub_environment` therefore stops one
step too early, which is why the tests here assert on the final docker argv.
That table lists `OPENAI_API_KEY` under the `opencode` backend, so switching
this demo to `opencode` would hand the solver credential to the mutation agent
and silently defeat this isolation while every config still looked correct.
`test_backend_choice_is_what_keeps_the_solver_key_out_of_the_agent` guards that
assumption.

Run every command below from this directory.

## Setup and scorer gate

```sh
docker manifest inspect \
  ghcr.io/ke7/helix-evo-runner-claude@sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba \
  > /dev/null && echo "runner pin resolves in registry"
docker pull \
  ghcr.io/ke7/helix-evo-runner-claude@sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba
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
uv run helix attempts --path .helix
uv run helix best --dir .
python3 inspect_run.py --require-terminal
```

Expected durable artifacts are `.helix/state.json`, the proposal ledger embedded
in that state, candidate/evaluation records, and HELIX-managed candidate
worktrees. With `P=2,N=2`, the generation has one proposal batch containing two
parent groups times two mutations: four tasks with distinct candidate IDs and
isolated Docker containers/worktrees. Scores are official LiveBench per-row
scores averaged by HELIX; solver token usage is recorded in evaluator details. A
32-call ceiling is configured, but cache hits and scheduler stopping can make
actual calls lower.

## Resume and compare P=1,N=1

After an interruption, inspect `.helix/state.json`, then resume the same run:

```sh
python3 inspect_run.py --require-terminal
../../.venv/bin/helix resume --dir .
python3 inspect_run.py --require-terminal
../../.venv/bin/helix resume --dir .
python3 inspect_run.py --require-terminal
```

Both consecutive `resume` invocations after terminal state must spend no budget
and add no candidate IDs. The three inspector snapshots must have identical
state digests, candidate IDs, scores, ledger totals, and metric-call budgets.

For release evidence, export deterministic manifests outside `.helix/` before
and after each resume, then compare the files byte-for-byte. The manifest is
intentionally non-vacuous: it records proposal identities and terminal fields,
budget accounting, durable-file counts and sizes, and three independent
digests (exact state-file bytes, a stable semantic projection, and the durable
artifact inventory). It excludes only the append-only diagnostic log and
ephemeral candidate worktrees.

```sh
mkdir -p ../../.helix-evidence/livebench-release
python3 inspect_run.py --resume-manifest \
  > ../../.helix-evidence/livebench-release/resume0-before.json
../../.venv/bin/helix resume --dir .
python3 inspect_run.py --resume-manifest \
  > ../../.helix-evidence/livebench-release/resume1-after.json
cmp ../../.helix-evidence/livebench-release/{resume0-before,resume1-after}.json
../../.venv/bin/helix resume --dir .
python3 inspect_run.py --resume-manifest \
  > ../../.helix-evidence/livebench-release/resume2-after.json
cmp ../../.helix-evidence/livebench-release/{resume1-after,resume2-after}.json
```

Invoke HELIX through its installed console script as shown and capture each
actual process status in a dedicated evidence file. Do not use
`python -m helix`: HELIX intentionally has no `helix.__main__` entry point.

Clean the first run before the serial comparison; the two configs intentionally
share `.helix/`:

```sh
echo y | uv run helix clean --dir .
/usr/bin/time -p uv run helix evolve --config helix.1x1.toml --dir .
uv run helix log --dir .
uv run helix best --dir .
python3 inspect_run.py
```

The comparison is only between two tiny stochastic smoke runs. It cannot be
used to reproduce or refute the publication's validation/test scores. The
argument-free inspector is the shape-agnostic 1x1 summary; only
`--require-terminal` applies the fixed P=2,N=2 release gate.

## Cleanup and resource accounting

This section accounts for **task-created resources**. It is deliberately no
longer titled "zero-resource proof": the checks below verify that this run's
own worktrees, containers, networks and derived image are gone, and they never
inspected any Docker **volume** at all.

**Persistent auth-store state is a separate matter.** A sandboxed run's
credential path is `sandbox.auth`. This lane runs `auth = "env"`, so it mounts
no persistent auth volume and leaves no auth-store state behind. Under
`auth = "volume"` — which is what an omitted `auth` key silently resolves to —
the backend CLI reads and writes a persistent `helix-auth-<backend>` volume
that is shared **across runs**, and that volume is mutable runtime state which
cleanup neither removes nor accounts for. See
`docs/design/sandbox-home-isolation.md`.

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
