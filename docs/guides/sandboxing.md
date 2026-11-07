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
exits; evaluator-runner file changes are discarded. HELIX never mounts the host
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

Backend state in the sandbox never survives a generation; the run's artifacts
are the record. At the start of every generation HELIX wipes the backend
CLI's session state in its auth volume (transcripts, memories, shell
snapshots, session indexes; never credentials or operator configuration) so
no candidate sees another candidate's history. Each invocation's native
transcript (every backend) is copied out of the volume first and kept at
`.helix_artifacts/backend_transcripts/<backend>/<session_id>.jsonl` in the
candidate worktree, where tool-call accounting also reads it.
This avoids copying host credential stores into Docker. On macOS, Claude/Cursor
browser-login tokens may live in Keychain; on Linux they may live in
Secret Service/libsecret, GNOME Keyring, KWallet, or another desktop keyring.
Those stores are session- and OS-specific, so copying their databases into a
Linux Docker image is not a reliable authentication mechanism. If your
evaluator uses a local proxy, keep that endpoint in your evaluator code as
usual. Docker Desktop supports `host.docker.internal`; Linux users can set
`add_host_gateway = true`.

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
