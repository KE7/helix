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

## Exact sources and licenses

All machine-readable pins live in [`pins.json`](pins.json):

- HELIX base `84c7bcd2b82a56c8dd5c18b7fe5828101b6a7023` plus upstream
  relative-path fix `402dcc8cfb2c461144de8f019e6ec49811dc2da9` (HELIX 0.3.0).
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
  14 CPUs and 8,216,301,568 bytes in the shared VM.
- At least 8 GiB free disk for setup and the HELIX runner. The official
  FormulaCode task image is not pulled by this smoke workflow.
- A valid `ANTHROPIC_API_KEY` in the environment. Never place it in a file or
  command-line argument. HELIX passes it through only to isolated mutation
  containers and redacts Docker diagnostics.
- This HELIX checkout must contain the pinned base and relative-path fix above.

The pinned HELIX runner is `linux/amd64`; Docker Desktop on Apple Silicon uses
emulation. That affects wall time, but performance measurement itself runs on
the host and is serialized with a shared lock, not inside the emulated agent
containers.

## Setup

From the HELIX repository root:

```bash
git cat-file -e 84c7bcd2b82a56c8dd5c18b7fe5828101b6a7023^{commit}
git cat-file -e 402dcc8cfb2c461144de8f019e6ec49811dc2da9^{commit}
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
jq '.state | {generation,budget,batches,tasks,distinct_child_ids,parent_major_order}' \
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

`inspect` asserts unique reserved child IDs, parent-major ordering, reports every
task's status/selection/cleanup/budget charge, lists actual worktree paths, and
scans HELIX artifacts for configured API-key values without printing them.

## Resume and idempotence

Resume uses the exact same generated config; changing P, N, selection, sampler,
or evaluator semantics is intentionally rejected by HELIX.

```bash
before=$(uv run python examples/formulacode/manage.py fingerprint | jq -r .fingerprint)
uv run helix resume --dir examples/formulacode/.work/networkx_networkx_7971
after=$(uv run python examples/formulacode/manage.py fingerprint | jq -r .fingerprint)
test "$before" = "$after"
```

For a completed one-generation run, resume must schedule no new child ID,
consume no additional evaluator budget, and leave the durable fingerprint
unchanged.

## Cleanup and zero-resource proof

Run HELIX cleanup first, then remove the ignored benchmark clone/venv and the
runner image if this demo pulled it:

```bash
uv run helix clean --dir examples/formulacode/.work/networkx_networkx_7971
uv run python examples/formulacode/manage.py cleanup --remove-runner-image
uv run python examples/formulacode/manage.py verify-clean
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
