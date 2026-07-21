# SWE-bench-Live: isolated HELIX P×N demo

This self-contained smoke demo evolves a real coding-agent program against one
pinned official SWE-bench-Live task. It is not a full-benchmark claim: the only
benchmark score is official binary resolution for
`capstone-engine__capstone-2743`. See `SOURCE.md` for immutable commits, object
hashes, image digest, licensing, and the already-passing upstream gold command.

## Prerequisites and measured portability

- macOS or Linux; Python 3.11+; `uv`; Git and Git LFS; Docker 24+.
- Claude Code authenticated on the host (`claude auth status`).
- The content-addressed Claude runner image
  `sha256:016259fef07b7344f924fad0129a19cb2541248e5f4c9af98ef462579aeb8d1b`
  (local `ghcr.io/ke7/helix-evo-runner-claude:0.2.0`, `linux/arm64`) and
  an authenticated `helix-auth-claude` Docker volume.
- At least 8 CPUs and enough space for the pinned 2.49 GB official task image.
- The official image is `linux/amd64`. Docker Desktop emulation was validated on
  an Apple Silicon host with Docker capacity 31,490,187,264 bytes and 14 CPUs.
  Each task container is capped at 4 GiB; each mutation container is capped at
  2 GiB/2 CPUs/256 PIDs and 900 seconds. Gold resolved three times, including
  two concurrent task containers.

No benchmark clone, row, patch, test patch, result, HELIX state, credentials,
logs, image, or cache is committed. `prepare.py` checks out only the pinned
dataset revision in a temporary directory, validates the parquet size/hash and
row contract, writes private fields to a labeled Docker volume, and deletes the
temporary checkout. Mutation runs in the pinned Claude Docker runner and sees
only `coding_agent.py` plus the public `TASK.md`; it receives no private task
row, credentials, evaluator/config/setup/source files, Docker socket, or task
volume. The evaluator remains host-side so it can launch the official sandbox.
The task-container network is disabled, and candidate code runs as an
unprivileged user that cannot read the root-only private task volume.

## Exact setup and smoke

The image must already exist locally; setup never pulls or substitutes a tag.

```bash
cd examples/swebench_live
docker image inspect \
  docker.io/starryzhang/sweb.eval.x86_64.capstone-engine_1776_capstone-2743@sha256:c3d6222106db9afce1eaf6036f67d540011e46ea8e59419097c32d0555032ed9
docker image inspect \
  sha256:016259fef07b7344f924fad0129a19cb2541248e5f4c9af98ef462579aeb8d1b
uv run --project /path/to/helix-repo helix sandbox status claude \
  --image sha256:016259fef07b7344f924fad0129a19cb2541248e5f4c9af98ef462579aeb8d1b
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
git archive HEAD examples/swebench_live | tar -x -C "$run_dir" --strip-components=2
cd "$run_dir"
uv run --with pyarrow python prepare.py
uv run --project /path/to/helix-repo helix evolve --dir .
```

`helix.toml` fixes P=2, N=2, `max_workers=2`, one generation, Haiku/low,
best-improvement selection, no merge, and no evaluation cache. HELIX therefore
reserves four distinct child IDs/worktrees in deterministic parent-major order;
at most two isolated official task containers run concurrently. Each non-empty
candidate receives a fresh container and the exact official harness rule. Only
official correctness contributes to selection; timing and resource data are
auxiliary diagnostics.

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
python3 inspect_run.py --dir . > artifacts/resume-1-audit.json
cmp artifacts/run-audit.json artifacts/resume-1-audit.json
uv run --project /path/to/helix-repo helix resume --dir .
python3 inspect_run.py --dir . > artifacts/resume-2-audit.json
cmp artifacts/run-audit.json artifacts/resume-2-audit.json
```

Expected uncommitted artifacts are `.helix/state.json`, batch/task journals,
attempt records, HELIX logs, candidate worktrees, and
`artifacts/evaluations/*.json`. `inspect_run.py` fails closed unless generation
1 contains exactly one complete current-schema P=2/N=2 ledger with four
globally distinct IDs,
parent-major indices, terminal status/cleanup, `budget_accounted=true`, and an
exact per-batch task-charge/budget-delta match. It records the global proposal
charge and explicit non-proposal remainder, and requires the final batch budget
to equal the global counter. With the generation budget already
exhausted, both consecutive resumes must be idempotent: no new task container,
candidate, or evaluation charge, and byte-identical state/accounting.

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
labeled containers. After recording the audit evidence, remove and verify the
exact disposable directory with a guarded path check:

```bash
cd /path/to/helix-repo
python3 - "$run_dir" <<'PY'
import shutil
import sys
import tempfile
from pathlib import Path

target = Path(sys.argv[1]).resolve()
temp_root = Path(tempfile.gettempdir()).resolve()
if target.parent != temp_root or not target.name.startswith("helix-swebench-live."):
    raise SystemExit(f"refusing unexpected cleanup target: {target}")
shutil.rmtree(target)
PY
test ! -e "$run_dir"
```
