# SWE-bench-Live: isolated HELIX P×N demo

This self-contained smoke demo evolves a real coding-agent program against one
pinned official SWE-bench-Live task. It is not a full-benchmark claim: the only
benchmark score is official binary resolution for
`capstone-engine__capstone-2743`. See `SOURCE.md` for immutable commits, object
hashes, image digest, licensing, and the already-passing upstream gold command.

## Prerequisites and measured portability

- macOS or Linux; Python 3.11+; `uv`; Git and Git LFS; Docker 24+.
- Claude Code authenticated on the host (`claude auth status`).
- At least 4 CPUs and enough space for the pinned 2.49 GB unpacked image.
- The official image is `linux/amd64`. Docker Desktop emulation was validated on
  an Apple Silicon host (Darwin ARM64, 14 CPUs, 48 GiB host RAM, 8.216 GiB
  Docker RAM). Each task container is capped at 4 GiB so the configured
  two-worker window fits the VM. Gold resolved three times, including two
  concurrent task containers.

No benchmark clone, row, patch, test patch, result, HELIX state, credentials,
logs, image, or cache is committed. `prepare.py` checks out only the pinned
dataset revision in a temporary directory, validates the parquet size/hash and
row contract, writes private fields to a labeled Docker volume, and deletes the
temporary checkout. The mutation context receives no private task row or
credentials. The task-container network is disabled, and candidate code runs as
an unprivileged user that cannot read the root-only private task volume.

## Exact setup and smoke

The image must already exist locally; setup never pulls or substitutes a tag.

```bash
cd examples/swebench_live
docker image inspect \
  docker.io/starryzhang/sweb.eval.x86_64.capstone-engine_1776_capstone-2743@sha256:c3d6222106db9afce1eaf6036f67d540011e46ea8e59419097c32d0555032ed9
uv run --with pyarrow python prepare.py
python3 evaluate.py --gold-smoke
```

The smoke must report `"accuracy": 1.0`, `"resolved": true`, and
`"gold_smoke_only": true`. It validates the private adapter but is not a HELIX
result. The seed agent intentionally emits an empty patch, so a standalone
`python3 evaluate.py` should report official score 0 before evolution.

## Exact P=2, N=2 run

Use a disposable project copy so this repository does not gain a nested `.git`:

```bash
run_dir="$(mktemp -d /tmp/helix-swebench-live.XXXXXX)"
cp -R examples/swebench_live/. "$run_dir/"
cd "$run_dir"
uv run --with pyarrow python prepare.py
uv run --project /path/to/helix-repo helix evolve --dir .
```

`helix.toml` fixes P=2, N=2, `max_workers=2`, one generation, Haiku/low,
best-improvement selection, no merge, and no evaluation cache. HELIX therefore
reserves four distinct child IDs/worktrees in deterministic parent-major order;
at most two isolated official task containers run concurrently. Each non-empty
candidate receives a fresh container and the exact official harness rule.

Before the heavy evolution window, verify no other benchmark containers are
running and record resources:

```bash
docker ps --format '{{.ID}} {{.Names}} {{.Image}} {{.Status}}'
docker info --format 'cpus={{.NCPU}} memory={{.MemTotal}} arch={{.Architecture}}'
df -h .
```

## Inspect, resume, and expected artifacts

```bash
uv run --project /path/to/helix-repo helix frontier --dir .
uv run --project /path/to/helix-repo helix attempts --path .helix --json
python3 inspect_run.py --dir . > artifacts/run-audit.json
uv run --project /path/to/helix-repo helix resume --dir .
python3 inspect_run.py --dir . > artifacts/resume-audit.json
cmp artifacts/run-audit.json artifacts/resume-audit.json
```

Expected uncommitted artifacts are `.helix/state.json`, batch/task journals,
attempt records, HELIX logs, candidate worktrees, and
`artifacts/evaluations/*.json`. `inspect_run.py` records the state hash,
frontiers, P/N batch ledger, distinct ordered candidate IDs, task cleanup
states, evaluation budget, and artifact count. With the generation budget
already exhausted, resume must be idempotent: no new task container, candidate,
or evaluation charge, and byte-identical state.

## Exact cleanup and zero-resource proof

Export a best candidate first if desired, then remove HELIX and benchmark state:

```bash
uv run --project /path/to/helix-repo helix best --export best-export --dir .
uv run --project /path/to/helix-repo helix clean --dir .
python3 cleanup.py --remove-image > cleanup-proof.json
git worktree list
docker ps -a --filter label=com.helix.demo=swebench-live-capstone-2743
docker volume ls --filter label=com.helix.demo=swebench-live-capstone-2743
docker image inspect \
  docker.io/starryzhang/sweb.eval.x86_64.capstone-engine_1776_capstone-2743@sha256:c3d6222106db9afce1eaf6036f67d540011e46ea8e59419097c32d0555032ed9
```

The last image inspection must fail after cleanup. `cleanup-proof.json` records
the before/after image IDs, removed labeled containers/volume, and zero remaining
labeled containers. Remove the disposable `run_dir` only after retaining any
audit evidence needed for the release report.
