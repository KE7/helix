# Running, Monitoring, And Debugging HELIX

Use this file when a user asks how to run HELIX, inspect progress, resume, or
debug failures.

## Contents

- Preflight
- Running
- Monitoring
- Resuming
- Cleaning
- Debugging Evaluator Failures
- Debugging Agent Failures
- Debugging Docker Sandboxing
- Common Repair Actions

## Preflight

1. Confirm project root:
   ```bash
   pwd
   test -f helix.toml
   ```
2. Validate evaluator runner and sidecar wiring:
   ```bash
   docker run --rm my-private-evaluator:latest python -m benchmark_server
   ```
   Then verify the `[evaluator].command` runner can read `helix_batch.json`,
   call `HELIX_EVALUATOR_ENDPOINT`, and print `HELIX_RESULT=...`.
3. If sandboxed, verify Docker and auth:
   ```bash
   docker ps
   helix sandbox status
   helix sandbox login <backend>
   ```
4. If sandboxed, check sidecar config:
   ```toml
   [evaluator]
   command = "python /runner/evaluate_client.py"

   [evaluator.sidecar]
   image = "my-private-evaluator:latest"
   runner_image = "my-evaluator-runner:latest"
   command = "python -m benchmark_server"
   endpoint = "http://helix-evaluator:8080/evaluate"
   ```

## Running

```bash
helix evolve
helix evolve --dir path/to/project
helix evolve --config path/to/helix.toml
```

Inside this repository, prefer:

```bash
uv run helix evolve --dir path/to/project
```

HELIX creates `.helix/` state and candidate worktrees. Do not manually edit
`.helix/state.json` unless repairing a run with explicit user approval.

## Monitoring

Common commands:

```bash
helix frontier
helix best
helix best --export ./best-solution
helix history
helix log
helix attempts
helix attempts --skips
```

Useful files:

```text
.helix/
  config.toml
  evaluator_manifest.json
  state.json
  helix.log
  log/*.json
  worktrees/gN-sN/
  attempts/           (per-rejected-candidate JSON records)
  skips/              (per-generation perfect-skip event lists)
```

For backend diagnostics in a candidate worktree, inspect:

```text
.helix_mutation_prompt.md
.helix_backend_result.json
.helix_backend_stdout.txt
.helix_backend_stderr.txt
.helix_artifacts/backend_transcripts/<backend>/<session_id>.jsonl
```

The backend result JSON includes normalized usage metadata and
`transcript_artifacts` entries describing the backend CLI's own transcript
for the invocation (`available: false` plus a `reason` when it could not be
found).  HELIX always copies that native transcript for every backend, in
Docker sandbox mode out of the `helix-auth-<backend>` volume before syncing
changes back, otherwise from the operator's `$HOME`:

| backend | native source (relative to `$HOME`) | artifact |
| --- | --- | --- |
| claude | `.claude/projects/-workspace/<session_id>.jsonl` (`claude_transcript_root`) | `claude/<session_id>.jsonl` |
| codex | `.codex/sessions/YYYY/MM/DD/rollout-<ts>-<thread_id>.jsonl` | `codex/<thread_id>.jsonl` |
| cursor | `.cursor/projects/<cwd-slug>/agent-transcripts/<session_id>/<session_id>.jsonl` | `cursor/<session_id>.jsonl` |
| agy | `.gemini/antigravity-cli/brain/<conversation_id>/.system_generated/logs/transcript{,_full}.jsonl` | `agy/<id>.jsonl`, `agy/<id>.full.jsonl` |
| opencode | this session's `session`/`message`/`part` rows of `.local/share/opencode/opencode.db` | `opencode/<sessionID>.jsonl` (one `{"table", "row"}` object per line) |

The structured stdout of the JSONL backends is still kept verbatim in
`.helix_backend_stdout.txt`; the native transcript is preserved in addition
because it carries the full per-turn record (tool payloads, reasoning items)
that the streamed events elide.

In Docker sandbox mode the backend's session state in `helix-auth-<backend>`
is wiped at the start of every generation (see `helix.backends.
BACKEND_STATE_PATHS`; one `reset <backend> agent state in ...` INFO line per
generation, a WARNING if the wipe fails), so a transcript that could not be
copied is logged at WARNING with the candidate id and the path looked for,
and cannot be recovered from the volume afterwards.

## Resuming

Use:

```bash
helix resume
```

Resume loads `.helix/state.json`, existing candidate worktrees, and cached
evaluations. If config changed, check `.helix/config.toml` and current
`helix.toml`; HELIX may preserve persisted frontier dimensionality for display
stability.

## Cleaning

```bash
helix clean
```

Use `helix best --export PATH` before cleaning if the user wants to keep the
best candidate outside `.helix/`.

## Debugging Evaluator Failures

Symptoms and checks:

- No score: verify the evaluator emits the required result line.
- `helix_result` failure: ensure `helix_batch.json` is read from cwd and output
  has exactly one final `HELIX_RESULT=` line.
- Length mismatch: `len(HELIX_RESULT payload) == len(helix_batch.json)`.
- All zero minibatch scores: evaluator probably ignored `helix_batch.json` ids
  or keyed results by metric names instead of example ids.
- Hangs: add evaluator-level timeouts or configure `[sandbox].timeout_seconds`.
- Missing dependencies in sandbox: build private evaluator/simulator
  dependencies into `[evaluator.sidecar].image` and evaluator-client
  dependencies into `[evaluator.sidecar].runner_image`.

Good `helix_result` smoke evaluator pattern:

```python
import json
from pathlib import Path

ids = json.loads(Path("helix_batch.json").read_text())
payload = []
for eid in ids:
    score = evaluate_one(eid)
    payload.append([score, {"id": eid, "scores": {"success": score}}])
print("HELIX_RESULT=" + json.dumps(payload))
```

## Debugging Agent Failures

Checks:

- Auth: `helix sandbox status <backend>` or run backend CLI status outside
  sandbox for local mode.
- Backend command output: inspect `.helix_backend_stderr.txt`.
- Prompt quality: inspect `.helix_mutation_prompt.md`.
- Protected file rejection: check `.helix/helix.log` for manifest errors.
- Rate limits: HELIX reports backend rate-limit failures and can resume later.
- Sandbox write issues: check Docker daemon, image tag, and native Linux UID
  ownership behavior. HELIX chowns the temporary workspace to `node` and back.

## Debugging Docker Sandboxing

Auth volume commands:

```bash
helix sandbox login claude
helix sandbox status claude
helix sandbox logout claude
```

Backend login behavior:

- Agy: full interactive `agy` launch (no dedicated login subcommand)
- Claude: `claude setup-token`
- Codex: `codex login --device-auth`
- Cursor: `cursor-agent login`
- OpenCode: full `opencode` TUI for provider/model/login setup

Network:

- Default `bridge` is needed for agent model endpoints.
- Evaluator sidecars use a private internal Docker network created by HELIX.
  Mutator containers are not attached to it.
- For host proxy access on Docker Desktop, use `host.docker.internal`.
- On Linux, set `[sandbox].add_host_gateway = true`.

Custom images:

```bash
docker build -t helix-runner-base:latest -f docker/base.Dockerfile .
docker build -t helix-runner-codex:latest -f docker/codex.Dockerfile .
```

Then:

```toml
[sandbox]
enabled = true
image = "helix-runner-codex:latest"
```

For heavy benchmarks, bake dependencies into the image. Avoid mounting host
venvs directly unless the user accepts portability and sandbox tradeoffs.

## Common Repair Actions

- Add or fix `[sandbox]` and run `helix sandbox login <backend>`.
- Add `[evaluator.sidecar]` when `[sandbox].enabled = true`.
- Add evaluator service dependencies to the sidecar image and runner
  dependencies to `runner_image`.
- Add `passthrough_env` for non-secret runtime vars such as CUDA/HF caches.
- Use `protected_files` only for non-sandboxed local prototypes.
- Emit one positional result pair per example for per-example datasets.
- Lower `num_parallel_proposals` if backend rate limits or resource contention
  dominate.
- Raise `timeout_seconds`, `memory`, or custom image resources for simulators.
