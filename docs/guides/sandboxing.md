# Docker sandboxing

With `[sandbox].enabled = true`, HELIX runs each mutation and merge agent in a Docker container that holds a copy of
the candidate, not the candidate itself. When the agent exits, HELIX copies its changes back into the candidate's
worktree. The evaluator still runs on your machine, in the candidate's worktree, unless you also run it in Docker
(see [Running the evaluator in Docker](#running-the-evaluator-in-docker)).

## What the agent's container holds

- `/workspace`: a copy of the candidate, without `helix.toml`, `.env` and `.env.*` files, `.git`, HELIX's own run
  files, and anything you list in `[sandbox].omit_from_agent`. FIFOs, sockets and device files are skipped (set
  `skip_special_files = false` to make them an error instead).
- `/home/node`: the backend's login volume (see [Logging in](#logging-in)).

Nothing else from your machine is mounted: not the project root, its parent directories, or your home directory.
Because every agent works in `/workspace`, nothing it sees names its candidate. Without the sandbox, an agent works in
`.helix/worktrees/<candidate>` (such as `g3-s12`), so its working directory names the candidate and generation.

## Logging in

Each backend has its own login volume, mounted at `/home/node` in agent containers: `helix-auth-agy`,
`helix-auth-claude`, `helix-auth-codex`, `helix-auth-cursor`, `helix-auth-opencode` or `helix-auth-pi`. Log in once
per backend, inside the same Linux container HELIX uses:

```bash
helix sandbox login claude
helix sandbox status claude
```

How each backend signs in:

| Backend | `helix sandbox login` runs | What you do |
|---|---|---|
| `claude` | `claude auth login --claudeai` | Open the printed URL, then paste the code back into the terminal. |
| `codex` | `codex login --device-auth` | Open the printed URL and enter the code. |
| `agy`, `opencode` | the CLI itself | Pick a provider and model and sign in. |
| `pi` | the CLI itself | Run `/login`, pick a provider, then quit. |
| `cursor` | `cursor-agent login` | Follow its prompts. |

HELIX logs in inside the container instead of copying your machine's login, because desktop logins often live in the
macOS Keychain or a Linux keyring, which a Linux container cannot use.

If your evaluator calls a service on your machine, Docker Desktop provides `host.docker.internal`. On Linux, set
`add_host_gateway = true`.

## Agent timeouts

`[sandbox].timeout_seconds` caps each agent call's wall-clock time (no limit by default). A call that runs out of time
is a failed mutation, reported as `timed out after Ns (sandbox.timeout_seconds)`, even if the backend had already
printed a result such as Claude's max-turns message. The container's edits are discarded and the candidate is
dropped. Tokens the backend reported before the kill still count against the budget.

## Running the evaluator in Docker

With `[sandbox].evaluator = true` as well, the evaluator runs in Docker too, and you must configure
`[evaluator.sidecar]`:

- The sidecar (`[evaluator.sidecar].image`) is a long-running container that holds your private benchmark data and
  scoring code. HELIX starts it once per `helix evolve`, on a private Docker network that agent containers cannot
  reach. It does not see the candidate's files.
- For each evaluation, HELIX starts a short-lived runner container (`runner_image`) with a copy of the candidate. It
  runs `[evaluator].command`, which calls the sidecar over that network, prints the result and exits. Files the
  runner changes are discarded.

The runner either sends the sidecar the candidate files or data it needs, or runs the candidate's code itself and asks
the sidecar only to judge or simulate. Keep private data in the sidecar image, not the runner image. Neither container
gets the agents' login volume.

## What carries over between agents on the login volume

At the start of every generation, HELIX deletes the backend's session state from the login volume: its transcripts,
memories, shell snapshots, session indexes and prompt history. For OpenCode it deletes only the session rows of
`opencode.db`, which also holds the account's tokens. The full list per backend is
`helix.backends.BACKEND_STATE_PATHS`. Logins and your CLI configuration are never touched. Each call's transcript is
copied out of the volume when the call ends, to
`.helix_artifacts/backend_transcripts/<backend>/<session_id>.jsonl` in the candidate's worktree.

The reset deletes a list of paths; it does not give each agent a fresh home directory. So:

- Anything an agent writes under `/home/node` outside those paths stays on the
  volume and reaches later candidates, later generations and later runs that
  mount the same volume. Examples: `~/.codex/AGENTS.md`, `~/.claude/CLAUDE.md`,
  skills, plugins, or hook settings in `~/.claude/settings.json`.
- Within one generation, the candidates running at the same time
  (`num_parallel_proposals` x `mutations_per_parent` above 1) mount the same
  volume read-write and can see each other's files there.
- Two HELIX runs at the same time on the same volume, such as the default
  `helix-auth-<backend>`, can delete each other's live sessions, and with them
  transcripts not yet copied: each run's per-generation reset clears the
  session state of the whole volume.

For concurrent runs, or when no instruction file may carry over between runs,
set a separate `[sandbox].auth_volume` for each run. It is a Docker volume
name (letters, digits, `_`, `.`, `-`, starting with a letter or digit) mounted
at `/home/node` in agent containers in place of `helix-auth-<backend>`, and the
transcript copy and the reset use it too. `helix sandbox login` never writes
it, so the credential has to reach the agent another way: for example
copying a logged-in volume into it with Docker before the run, or a gateway
that holds the key, as in `examples/circle_packing_sandbox`.

## Symlinks the agent creates

When HELIX copies the agent's changes back, it copies symlinks as symlinks and never follows them. It then removes
any new or changed symlink that is absolute or points outside the candidate's worktree, and logs a warning naming it
(`dropping symlink ... it points outside the workspace`). Symlinks the candidate already had, pointing to the same
place, are kept. Transcripts are copied only from real directories and regular files, and a `.gitignore` symlink
that leads outside the worktree is replaced by a regular file instead of being written through.

## Resource limits and host names

These `[sandbox]` keys apply to agent and evaluator-runner containers:

| Key | Default | Effect |
|---|---|---|
| `cpus` | unset | `docker run --cpus` |
| `memory` | unset | `docker run --memory`, e.g. `"16g"` |
| `pids_limit` | `512` | `docker run --pids-limit` |
| `timeout_seconds` | unset | Wall-clock limit on each sandboxed command (one agent call, or one evaluator-runner command). When it expires the container is removed and that call fails. A host-run evaluator is not limited. |
| `add_host_gateway` | `false` | Adds `--add-host host.docker.internal:host-gateway` (needed on Linux; Docker Desktop has the name already). |
| `extra_hosts` | `{}` | A table of `name = "address"` pairs, each passed as `--add-host name:address`. Also applied to the evaluator sidecar. |

The evaluator sidecar gets only `extra_hosts` from this list.

## Agent container images

By default HELIX chooses a published backend-specific mutator image from
`agent.backend`: `ghcr.io/ke7/helix-evo-runner-agy`,
`ghcr.io/ke7/helix-evo-runner-claude`,
`ghcr.io/ke7/helix-evo-runner-codex`,
`ghcr.io/ke7/helix-evo-runner-cursor`,
`ghcr.io/ke7/helix-evo-runner-opencode`, or
`ghcr.io/ke7/helix-evo-runner-pi`. To build locally instead, build the
shared base first with
`docker build -t helix-runner-base:latest -f docker/base.Dockerfile .`, then
build the backend image you need, for example
`docker build -t helix-runner-codex:latest -f docker/codex.Dockerfile .`, and
set `[sandbox].image` to that local tag.

Evaluator sidecar images are benchmark-specific and are not published by HELIX.
Use `ghcr.io/ke7/helix-evo-runner-base` or `docker/base.Dockerfile` as a base
for your own `runner_image`; build/publish the private sidecar service image
from your evaluator repository.
