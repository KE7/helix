# Docker sandboxing


When `[sandbox].enabled = true`, HELIX runs mutation and merge agents in
Docker containers with copied workspaces on the normal agent network. The
evaluator command still runs on the host in the candidate worktree unless
sandboxed evaluation is also enabled.

When `[sandbox].evaluator = true` is set as well, HELIX requires
`[evaluator.sidecar]` and starts it once per `helix evolve` on a private
internal Docker network that agent containers cannot reach.
Evaluator-runner containers run only during evaluation, join the private
network, call the sidecar, print evaluator output, and exit.
The long-lived sidecar does not mount the candidate workspace; the runner must
stream the needed candidate files/data over RPC, or execute candidate code
itself and call the sidecar only for private judging/simulation.
`[evaluator.sidecar].image` is the private service image; `runner_image` is the
short-lived client image used for `[evaluator].command`. Keep private benchmark
data in the service image, not in the runner image.

Agent changes are synced back to the real candidate worktree after the backend
exits; evaluator-runner file changes are discarded.
`[sandbox].timeout_seconds` (unset by default: no limit) caps each agent
container's wall-clock time. An agent call that runs out of it is a failed
mutation ("timed out after Ns (sandbox.timeout_seconds)"), whatever the
backend printed before the kill, even a claude max-turns result: the
container's edits are discarded, the candidate is dropped, and the tokens
the backend reported before the kill are still charged to the budget. HELIX never mounts the host
project root, parent directories, or home directory by default.
`helix.toml`, `.env`, `.env.*`, `.git`, and HELIX runtime artifacts are also
excluded from sandbox workspace copies/sync-back so agents cannot read sidecar
configuration or mutate run settings.
During copy and sync, HELIX skips unsupported special files such as FIFOs,
sockets, and device nodes by default. Set `skip_special_files = false` only if
you want unsupported workspace file types to raise instead of being ignored.

Agent containers mount a persistent Docker auth volume at `/home/node`;
evaluator sidecar and runner containers never receive it. Run
`helix sandbox login <backend>` once per backend to complete that CLI's normal
login flow inside the same Linux container environment HELIX will use later:

```bash
helix sandbox login claude
helix sandbox status claude
```

For Claude, HELIX uses `claude setup-token` for sandbox login because that is
the flow that works cleanly in browserless Docker/SSH-style environments; it
prints a URL and then accepts the code from the browser in the terminal.
Codex similarly uses `codex login --device-auth` so the callback does not depend
on a localhost server inside the container. Antigravity CLI (`agy`) has no
dedicated login subcommand, so HELIX starts its normal interactive CLI, same
as OpenCode, which starts the normal TUI so you can choose the provider/model
and complete provider login in one setup session. pi also signs in only from
its TUI: `helix sandbox login pi` starts it, you run `/login`, pick a provider,
and quit.

The volume names are `helix-auth-agy`, `helix-auth-claude`, `helix-auth-codex`,
`helix-auth-cursor`, `helix-auth-opencode`, and `helix-auth-pi`.

Logging in inside the container avoids copying host credential stores into Docker. On macOS, Claude/Cursor
browser-login tokens may live in Keychain; on Linux they may live in
Secret Service/libsecret, GNOME Keyring, KWallet, or another desktop keyring.
Those stores are session- and OS-specific, so copying their databases into a
Linux Docker image is not a reliable authentication mechanism. If your
evaluator uses a local proxy, keep that endpoint in your evaluator code as
usual. Docker Desktop supports `host.docker.internal`; Linux users can set
`add_host_gateway = true`.

## What persists on the auth volume

At the start of every generation HELIX deletes the backend CLI's session state
from the auth volume: the paths listed for that backend in
`helix.backends.BACKEND_STATE_PATHS` (transcripts, memories, shell snapshots,
session indexes, prompt history; for OpenCode, only the session rows of
`opencode.db`, which also holds the account's tokens). Credentials and
operator configuration are never touched. Each invocation's native transcript
(every backend) is copied out of the volume when the call ends and kept at
`.helix_artifacts/backend_transcripts/<backend>/<session_id>.jsonl` in the
candidate worktree, where tool-call accounting also reads it.

The reset is a list of paths, not a fresh home directory, so it does not keep
candidates apart completely:

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

## Sync-back and symlinks

The agent's workspace is writable by the agent, so sync-back treats its
symlinks with suspicion. Symlinks are copied back as symlinks and never
followed, but after the copy HELIX removes every symlink that is new or
changed and is absolute or resolves outside the candidate worktree, and logs a
warning naming it (`dropping symlink ... it points outside the workspace`).
Symlinks the candidate already had, pointing to the same place, are kept.
Transcript artifacts are copied only from real directories and regular files,
and a `.gitignore` symlink that leaves the worktree is replaced by a regular
file instead of written through.

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

## Images

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
