# HELIX FormulaCode benchmark demo

This is a self-contained, **FormulaCode-compatible smoke demo** for HELIX's
P×N proposal scheduler. It evolves the real NetworkX repository at the pinned
base of verified FormulaCode task `networkx_networkx_7971`, runs focused
upstream correctness tests, measures the official PR's adversarial workload,
and applies fc-eval's speedup, advantage, and correctness-fallback semantics.

It is not a full FormulaCode leaderboard run. A leaderboard-comparable run uses
fc-eval's complete generated task, all covered ASV workloads, its official
`linux/amd64` task image, and preferably isolated x86_64 AWS hardware. This demo
uses two public training cases and two larger validation cases so a laptop can
exercise a genuine HELIX P=2, N=2 run. Its scores must be labeled **smoke-subset
scores**, never FormulaCode leaderboard scores.

## Recorded result of the accepted 0.3.0 demo run

The verification run that qualified this lane for the 0.3.0 release produced a
**negative optimization result**, recorded here so it is not overstated
elsewhere:

- **0 of 4 proposals were accepted.** All four failed correctness, fell back to
  baseline (`failure_kind=correctness_failure`, `success=false`,
  `correctness=0.0`, speedup clamped to `1.0`).
- The frontier **retained the seed** `g0-s0` at advantage `-107.7423`; no
  candidate improved on it.
- Total cost: **$0.2487627** across **12 evaluations**.

What this run demonstrates is **scheduler, ledger, resume, and cleanup
integrity** — a real P=2, N=2 with four distinct proposal IDs, worktrees,
concurrent containers, and candidate-keyed transcript roots; a terminal
fail-closed ledger with conserved budget (task charges `4+0+4+2 = 10`, equal to
the batch delta `12 - 2`, with the global total `12 = 10` proposal `+ 2`
non-proposal); two byte-identical resumes; and zero task-created residue. It
does **not** demonstrate any improvement on FormulaCode, and these numbers are
smoke-subset figures, not leaderboard performance. A run in which every
proposal is rejected is a valid and honest outcome for an integrity demo: the
fail-closed path is exactly what is being exercised.

### How to read these numbers

- **The four candidate advantage values are not agent measurements.** Every
  candidate records `agent_median_seconds=null` with
  `fallback_to_baseline=true`, so its advantage (`-65.0810198304571`, identical
  across all four) is derived from the oracle/nop baseline *after* correctness
  failed. Four identical values are the expected consequence of universal
  fallback, not evidence of cross-candidate contamination. Only the **seed**
  advantage is a genuine host-timed measurement.
- **Timing is indicative, not precise.** This host carries roughly 1.3 cores of
  uncontrolled background load (macOS Spotlight/media-analysis daemons) that
  cannot be quiesced. The run was made with no competing HELIX lane work, and
  load was sampled and recorded, but the seed advantage should be read as
  indicative rather than a controlled benchmark figure.
- **The cost is reproducible but unexplained.** Two independent post-fix runs
  agree within 1.3% ($0.2488 and $0.2521), both roughly 51% above an earlier
  pre-fix run ($0.1651). The increase is therefore systematic rather than
  noise, but **no cause has been established** and the runs are not presented
  as equivalent.
- **`g1-s2` is charged 0 evaluations by design.** It is a content-dedup
  *follower*: its committed tree is byte-identical to `g1-s1`, so the executor
  reuses the leader's outcome and charges the follower zero
  (`num_actual_evaluations == 0`), including when the shared outcome is a
  failure. The zero charge is correct accounting, not a skipped task.

### Known limitations (pre-existing, non-blocking)

These are properties of HELIX that this demo surfaced. Neither is introduced by
the demo, and neither blocks the release:

1. **Dedup provenance is not durable across cleanup.** The follower-to-leader
   relationship above is reconstructible while the run's artifacts exist, but
   `helix clean` removes them, and the persisted ledger does not record the
   donor. After cleanup, a legitimately deduplicated zero-charge follower is
   not distinguishable from a skipped task using persisted state alone.
2. **Full token/cost conservation is not a persisted guarantee.** Exact
   conservation is enforced for `evaluations` (verified in this run), but the
   six token counters and `cost_usd` are not conserved relationally in the
   persisted state.

### Runner architecture and emulation

The pinned runner image `ghcr.io/ke7/helix-evo-runner-claude@sha256:6be6fef…`
is a **single-platform `linux/amd64`** image — the manifest is not a multi-arch
index, so there is no `arm64` variant to select. On an `aarch64` host it runs
**under emulation** (`uname -m` inside the container reports `x86_64`), and
HELIX passes no `--platform` flag, relying on the host default.

This does **not** affect any measured result here, and the reason is
structural rather than incidental: `[sandbox] evaluator = false`, so the
timing and correctness evaluator runs **natively on the host**. Only the
agent's code-writing container is emulated. Emulation changes how long the
agent takes to write code; it does not change how fast the resulting code
runs.

Two consequences worth stating plainly:

- Emulation slows the agent. This run used `model = "haiku"`, `effort = "low"`,
  `max_turns = 6`, `timeout_seconds = 360`. A slower agent is a **plausible
  contributing factor** to the 0/4 correctness failures above. This is offered
  as a hypothesis, not a measured conclusion.
- Any lane that intends to publish wall-clock numbers from an `aarch64` host
  must confirm its evaluator is host-side. Container-measured timings under
  emulation would not be comparable to native and must not be reported.

**Pre-pull the runner before a timed run.** HELIX never calls `docker pull`; it
passes the image reference straight to `docker run`. With a cold cache, Docker
pulls implicitly at dispatch, requiring live network and registry auth
mid-run — a failure that would abort the run at its first container.

## Exact sources and licenses

All machine-readable pins live in [`pins.json`](pins.json):

- HELIX 0.3.0 runtime core `c9371f4c91a50fa7196c85826a0e08e546aa05bc`,
  with canonical ancestry `94f9751` → `402dcc8` → `e5c260f` → `c9371f4`.
- fc-eval `0.1.0` source commit
  `c08f665e7bf3b4de225b72dc02ce9b15b7aaba2b` (BSD-3-Clause).
- FormulaCode verified-dataset commit
  `897e48cab8a27d32ba20ddb970b1fc397d96ee95`, artifact
  `verified/train-00000-of-00001.parquet`, LFS SHA-256
  `d872c4f3025e2331c012ce311e4330c73a72b87034c287fc9ce5f4d1b23e81d7`.
- NetworkX task `networkx_networkx_7971`, base
  `a986762f2a1919126df2174644232c92c58be2be`, human-oracle merge
  `3d0bb212f9fa4bac168c3b8c3f512a5f69b7920c` (BSD-3-Clause).
- Official task image digest
  `sha256:6e7f9d3cc7ec5020b8156038eea0832bc2d993953f5f03e13e38dcaccaffb0fe`.

The scorer ports fc-eval's pinned benchmark-name deconstruction, median ratios,
geometric aggregation, level groupings, and revert-to-baseline failure rule.
See [`LICENSES.md`](LICENSES.md) for attribution and the FormulaCode citation.
No dataset row, solution patch, benchmark clone, image, or generated result is
tracked in this repository.

## Prerequisites

- macOS or Linux with Git, Docker, `uv`, and `jq`.
- Docker with at least 4 CPUs and 6 GiB available; the demonstrated machine had
  14 CPUs and 31,490,187,264 bytes in the shared VM.
- At least 8 GiB free disk for setup and the HELIX runner. The official
  FormulaCode task image is not pulled by this smoke workflow.
- A valid `ANTHROPIC_API_KEY` in the environment. Never place it in a file or
  command-line argument. HELIX supplies it to isolated mutation containers and
  redacts Docker diagnostics. Each protected evaluator subprocess removes
  provider credential variables from its own environment before any candidate
  import/execution; its correctness child receives only PATH and PYTHONPATH.
- This HELIX checkout must contain the canonical 0.3.0 ancestry above.

The pinned HELIX runner is `linux/amd64`; Docker Desktop on Apple Silicon uses
emulation. That affects wall time, but performance measurement itself runs on
the host and is serialized with a shared lock, not inside the emulated agent
containers.

## Setup

From the HELIX repository root:

```bash
git merge-base --is-ancestor 94f9751de22b42d5a3140d69dec1d1a9ffd329e5 HEAD
git merge-base --is-ancestor 402dcc8cfb2c461144de8f019e6ec49811dc2da9 HEAD
git merge-base --is-ancestor e5c260f08948cf33abf61716f31b25f455de0dd0 HEAD
git merge-base --is-ancestor c9371f4c91a50fa7196c85826a0e08e546aa05bc HEAD
uv run helix --version
uv run python examples/formulacode/manage.py preflight
uv run python examples/formulacode/manage.py setup
```

`setup` is idempotent after success. It:

1. records the Docker/disk resource snapshot;
2. clones NetworkX only into ignored `examples/formulacode/.work/`;
3. checks out the pinned base and a temporary, mutation-inaccessible oracle;
4. creates a shared Python 3.12 venv with `pytest==8.4.1`;
5. runs the two pinned upstream component test files on base and oracle;
6. calibrates nop/oracle samples with 3 warmups, 7 repeats, and 12 loops;
7. removes the oracle worktree and commits only the local HELIX seed overlay.

Inspect calibration without exposing hidden validation definitions to the
mutation agent:

```bash
jq '{task_id,base_commit,oracle_commit,correctness,machine}' \
  examples/formulacode/.work/networkx_networkx_7971/.formulacode/baselines.json
```

The mutation sandbox omits `.formulacode/` and `helix.toml`; HELIX also hashes
both as protected paths. Candidates therefore cannot read or change held-out
cases, baselines, scorer code, or configuration.

## Run the real P=2, N=2 batch

First take a one-shot resource check. Do not overlap unrelated benchmark/timing
containers on a shared Docker VM.

```bash
uv run python examples/formulacode/manage.py preflight
docker pull --platform linux/amd64 \
  ghcr.io/ke7/helix-evo-runner-claude@sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba
uv run helix evolve \
  --dir examples/formulacode/.work/networkx_networkx_7971
```

The generated configuration fixes:

- `num_parallel_proposals = 2` (P),
- `mutations_per_parent = 2` (N),
- `max_workers = 4`, `max_generations = 1`,
- deterministic `rng_seed = 7971`,
- deterministic `proposal_selection = "best_improvement"`,
- two train and two validation cases,
- a 40-unit evaluator budget,
- one isolated Git worktree and one isolated Docker mutation container per
  candidate.

## Inspect durable evidence

```bash
uv run python examples/formulacode/manage.py inspect | tee /tmp/formulacode-inspect.json
jq '.state | {generation,budget,batches,tasks,accounting,distinct_child_ids,parent_major_order,terminal_p2n2}' \
  /tmp/formulacode-inspect.json
uv run helix frontier --dir examples/formulacode/.work/networkx_networkx_7971
uv run helix history --dir examples/formulacode/.work/networkx_networkx_7971
uv run helix attempts \
  --path examples/formulacode/.work/networkx_networkx_7971/.helix --json
```

Expected durable artifacts (all ignored and local):

- `.helix/state.json` — generation, frontier, scheduler checkpoint, cumulative
  token/cost/evaluation budget, and P×N proposal ledger;
- `.helix/evaluations/*.json` — candidate scores and FormulaCode side info;
- `.helix/attempts/*.json` — rejected candidates and gate evidence;
- `.helix/lineage.json` and trajectory/log files;
- `.helix/worktrees/<candidate-id>/` — isolated candidate repositories;
- `.helix/eval_cache.pkl` — durable content/example cache.

`inspect` fails closed unless state is generation 1 with exactly one complete
P=2,N=2 batch, four globally distinct reserved IDs in parent-major order, and
every task has terminal status/selection/cleanup plus `budget_accounted=true`.
For each complete batch it reports the sum of task evaluation charges and the
`budget_after_apply - budget_before_dispatch` delta and requires equality. It
also reports total global, proposal, and non-proposal evaluations, requires the
terminal batch boundary to equal the global budget, lists actual worktree
paths, and scans HELIX artifacts for configured API-key values without printing
them.

## Resume and idempotence

Resume uses the exact same generated config; changing P, N, selection, sampler,
or evaluator semantics is intentionally rejected by HELIX.

```bash
uv run python examples/formulacode/manage.py inspect > /tmp/formulacode-resume0.json
jq -S '{fingerprint,budget:.state.budget,accounting:.state.accounting,batches:.state.batches,tasks:.state.tasks}' \
  /tmp/formulacode-resume0.json > /tmp/formulacode-resume0.stable.json

uv run helix resume --dir examples/formulacode/.work/networkx_networkx_7971
uv run python examples/formulacode/manage.py inspect > /tmp/formulacode-resume1.json
jq -S '{fingerprint,budget:.state.budget,accounting:.state.accounting,batches:.state.batches,tasks:.state.tasks}' \
  /tmp/formulacode-resume1.json > /tmp/formulacode-resume1.stable.json
cmp /tmp/formulacode-resume0.stable.json /tmp/formulacode-resume1.stable.json

uv run helix resume --dir examples/formulacode/.work/networkx_networkx_7971
uv run python examples/formulacode/manage.py inspect > /tmp/formulacode-resume2.json
jq -S '{fingerprint,budget:.state.budget,accounting:.state.accounting,batches:.state.batches,tasks:.state.tasks}' \
  /tmp/formulacode-resume2.json > /tmp/formulacode-resume2.stable.json
cmp /tmp/formulacode-resume1.stable.json /tmp/formulacode-resume2.stable.json
```

For a completed one-generation run, both consecutive resumes must schedule no
new child ID, consume no additional evaluator budget, and leave the durable
fingerprint and terminal ledger/accounting unchanged.

## Cleanup and resource accounting

This section accounts for **task-created resources**. `manage.py cleanup`
verifies that no HELIX worktree/branch or task-created container remains, which
is correct as written — but the previous title, "zero-resource proof", promised
more than the checks deliver, and none of them inspects a Docker volume.

**Persistent auth-store state is a separate matter.** This lane runs
`sandbox.auth = "env"`, so it mounts no persistent auth volume and leaves no
auth-store state behind. Under `auth = "volume"` — which an omitted `auth` key
silently resolves to — the backend CLI reads and writes a persistent
`helix-auth-<backend>` volume shared **across runs**; that is mutable runtime
state which cleanup neither removes nor accounts for. See
`docs/design/sandbox-home-isolation.md`.

Run HELIX cleanup first, then remove the ignored benchmark clone/venv and the
runner image if this demo pulled it:

```bash
uv run helix clean --dir examples/formulacode/.work/networkx_networkx_7971
uv run python examples/formulacode/manage.py cleanup --remove-runner-image
uv run python examples/formulacode/manage.py verify-clean
rm -f /tmp/formulacode-inspect.json \
  /tmp/formulacode-resume0.json /tmp/formulacode-resume0.stable.json \
  /tmp/formulacode-resume1.json /tmp/formulacode-resume1.stable.json \
  /tmp/formulacode-resume2.json /tmp/formulacode-resume2.stable.json
git worktree list --porcelain
docker ps --format '{{.ID}} {{.Names}} {{.Image}}'
git status --short -- examples/formulacode tests/examples/test_formulacode.py
```

`cleanup` refuses broad paths, verifies no HELIX worktree/branch or task-created
container remains, then deletes only the owned `.work/` tree. Generated clones,
calibration data, state, results, caches, logs, and the local venv are not
recoverable after this step.

## Full official fc-eval reference

For a leaderboard-comparable evaluation, use fc-eval at the pinned commit and
its generated verified task on isolated x86_64 hardware:

```bash
git clone https://github.com/formula-code/fc-eval.git /tmp/fc-eval-c08f665
git -C /tmp/fc-eval-c08f665 checkout c08f665e7bf3b4de225b72dc02ce9b15b7aaba2b
cd /tmp/fc-eval-c08f665
uv sync --extra formulacode
uv run fc-eval run --dataset formulacode --remote-build \
  --config examples/config.json --task-id networkx_networkx_7971
```

That command uses fc-eval's full correctness/snapshot/ASV pipeline and produces
an official score. It requires a real fc-eval agent config and remote-machine
credentials and is intentionally separate from this HELIX scheduler smoke.
