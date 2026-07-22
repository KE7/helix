# Sandbox backend authentication

How a sandboxed mutation agent obtains backend credentials, what HELIX
verifies before a run starts, and what each mode costs you.

## The two modes

Sandboxed agent authentication is **explicit**. There is no auto-detection and
**no automatic fallback in either direction** — a mode is chosen in a config
file and never engages because something else failed.

```toml
[sandbox]
enabled = true
auth = "volume"          # "volume" (default) | "env"
auth_env_allow = []      # meaningful only when auth = "env"
agent_passthrough_env = []
require_cli_match = false
```

> ## `auth = "volume"` is UNSUPPORTED FOR AGENT EXECUTION in 0.3.0
>
> Omitting `sandbox.auth` still resolves to `"volume"`, and a sandboxed agent
> run in that mode now **raises `VolumeModeUnsupportedError` before any
> container starts**, on **every** backend. A config that says nothing about
> `sandbox.auth` will therefore fail. Set `auth = "env"` explicitly.
>
> Why: the persistent auth store is shared **across runs**, and every supported
> CLI keeps per-run state inside it that HELIX cannot relocate, so a later
> candidate can be causally influenced by an earlier one. HELIX refuses rather
> than report an isolated run that is not isolated.
>
> `helix sandbox login` / `status` / `logout` are **unaffected** and still use
> the volume — that is what it is for. So no statement of the form "HELIX never
> mounts the auth volume over HOME" is true; the accurate statement is that **no
> AGENT container mounts it**.

| | `auth = "volume"` | `auth = "env"` |
|---|---|---|
| Agent execution | **UNSUPPORTED — raises** | supported |
| Credential path | n/a for agents; `login`/`status` only | the variables in `auth_env_allow` |
| Credentials on the container argv | n/a | exactly `auth_env_allow`, nothing else |
| Auth volume mount (agent container) | n/a | **none at all** |
| Container-side OAuth refresh | n/a | **disabled** |

`sandbox.enabled = false` is unchanged: non-sandboxed runs always authenticate
from the environment, and setting `sandbox.auth` there is a config error.

## Why `auth = "env"` is a tradeoff, not an alternative

Setting `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` turns OAuth mode **off**
in the backend CLI. An env-mode run therefore performs **no container-side
token refresh at all**. If an auth volume exists for that backend, an env-mode
run consumes the credential path without maintaining it, and **the volume's
stored token will go stale**.

**Env mode mounts NO auth volume at all.** An earlier version of this document
said it mounted the volume `:ro`, reasoning that a run which cannot refresh has
no need of a writable mount. That reasoning addressed the wrong risk and the
code now explicitly repudiates it: a read-only whole-HOME mount still exposes
every prior run's transcripts, sessions and caches **for reading**, and read
access is the cross-candidate channel — write access never was. Env mode
therefore mounts no persistent store, so the channel does not exist rather than
being masked.

`CLAUDE_CODE_OAUTH_TOKEN` is **rejected** in `auth_env_allow`. It does not
merely bypass refresh — it makes the credential accessor return a record with
a null refresh token, disabling refresh permanently. The
`..._FILE_DESCRIPTOR` variants are rejected for the same reason.

The mode prints a non-suppressible disclosure at startup naming the variables
(never their values) and the container's network exposure.

## What reaches a container, and why

Every variable entering a container carries a recorded **origin** and an
explicit set of authorized **scopes**. This table is the whole policy.

| Origin | Config surface | Agent | Evaluator | Sidecar |
|---|---|---|---|---|
| `helix_internal` | none (code) | registered names only | ✓ | ✓ |
| `config_env` | `env = {...}` | ✓ | ✓ | ✓ |
| `config_passthrough` | `passthrough_env` | ✗ under a sandbox | ✓ | ✓ |
| `agent_passthrough` | `sandbox.agent_passthrough_env` | ✓ | ✗ | ✗ |
| `sidecar_passthrough` | `evaluator.sidecar.passthrough_env` | ✗ | ✗ | ✓ |
| `auth_env_allow` | `sandbox.auth_env_allow` | only if `auth = "env"` | ✗ | ✗ |
| `backend_auth_env` | `helix/backends.py` | only if `sandbox.enabled = false` | ✗ | ✗ |

Two consequences follow **mechanically** rather than by convention:

- **Renaming the backend cannot change agent credential flow.** Under a
  sandbox the per-backend table grants nothing to agent scope.
- **A sidecar credential cannot reach an agent.** `sidecar_passthrough` has no
  agent scope, and the cross-field union that previously merged top-level
  `passthrough_env` into the sidecar grant is gone.

Naming the same variable for both the agent and a credentialed sidecar is a
**hard config error**, for every lane and every backend.

### Scope limit

"No credential on the container argv" is **not** "no credential in the
container". `/home/node` is mounted from the auth volume and the workspace
mount carries whatever is in the candidate repo.

## The preflight

Volume mode verifies authentication **once per run, before the first mutation
is dispatched**, with no proposal, budget, ledger, or run-state side effects.
A failure aborts pre-dispatch with a redacted, actionable message.

1. **Existence** — `docker volume inspect`. Never `docker run -v`, which
   *creates* a missing named volume.
2. **Provenance** — a HELIX-owned stamp at `/home/node/.helix-auth-meta.json`
   recording backend, CLI version, image and timestamp. A **missing stamp
   means unknown**, and unknown is never reported as valid. Absence does not
   fail a run; a wrong-backend volume does.
3. **Sufficiency** — a **real authenticated request**, using the exact
   configured runner image, against the real volume at `:rw`.

Neither a non-empty credentials file nor the backend's own status text is
treated as evidence of working authentication. Both were observed reporting
success against credentials that a real request rejected.

Failures distinguish a failed **token refresh** (remedy: `helix sandbox
login`) from a failed **request after a successful refresh** (remedy: quota,
model access, or network) from a **transport** failure. Where the signals are
ambiguous the message says so rather than sending you to a remedy that will
not help.

### Preflight side effects, and what protects you

The preflight starts a real container with the auth volume mounted **`:rw`**,
and that has real consequences you should know about:

- **It writes non-credential state into the auth volume.** The backend CLI
  creates its own files on startup — session state, caches, log directories,
  and a backup of its config. This is normal CLI behaviour, not something
  HELIX does, but it means the volume is not read-only during a run.
- **It may refresh your credential, and a successful refresh rotates the
  stored refresh token.** That is the *intended* repair path — it is what
  keeps volume auth self-sustaining — but it means the volume must never be
  probed through a copy. A copy would absorb the rotation while the real
  volume kept the now-invalid token.
- **It makes one billable inference call**, recorded separately as *auth
  overhead*. It does **not** enter the evaluation budget, so budget
  conservation checks are unaffected.

Safeguards, so this cannot happen by accident:

- The probe's Docker runner is a **required dependency** with no default. A
  caller that does not supply one cannot reach Docker at all.
- The probe's argv is built by the **same** code path as the production
  mutation container, so there is no separate probe mount to point elsewhere.
- HELIX's own test suite **denies real Docker by default** and refuses any
  `helix-auth-*` volume outright, with no override.
- Concurrent runs are serialised by an advisory lock on the volume name and
  **fail loudly** rather than proceeding unverified. This cannot coordinate
  with a non-HELIX CLI touching the same account.

The verdict is cached **in-process only**, for one run. Nothing is persisted:
a stored verdict with a timestamp would be exactly the kind of unsound
sufficiency signal this design removes.

## Volume lifecycle

- `helix sandbox status` **never creates a volume**. It reports `not
  provisioned` and does not run a container. (It previously started one, and
  because `docker run -v` silently creates, observing a volume provisioned
  it.) It reports existence and provenance only — never validity.
- `helix sandbox login` **may** create a volume, and announces it when it
  does.
- `login`, `status`, `logout` and the run preflight all resolve to the **same
  runner image**. These commands now read your project's `helix.toml`; when no
  image can be determined they **fail with an actionable error** rather than
  defaulting to `:latest`, which can be a different CLI build than your runs
  consume.

## Migration

- `sandbox.enabled = false` — unaffected.
- Sandboxed runs that relied on host credentials reaching the agent must
  declare it: `auth = "env"` plus `auth_env_allow`. Loading such a config
  without the declaration is an error naming the remedy.
- Top-level `passthrough_env` no longer grants agent scope under a sandbox.
  Use `sandbox.agent_passthrough_env` for non-credential variables and
  `sandbox.auth_env_allow` for credentials.
- Existing volumes have no provenance stamp and report **unknown** until
  re-provisioned with `helix sandbox login <backend>`.
- Scripts that relied on `status` provisioning a volume must call `login`.

### `HELIX_*` variables

The `HELIX_*` prefix wildcard still propagates to evaluator and sidecar
scope. For **sandboxed agent** scope it is replaced by an explicit registry
(`HELIX_SPLIT`, `HELIX_INSTANCE_IDS`, `HELIX_EVALUATOR_ENDPOINT`,
`HELIX_ASI_LOG`, `HELIX_ASI_LOG_ENV`, `HELIX_ARTIFACT_NAMES`,
`HELIX_CLAUDE_TRANSCRIPT_ROOT`, `HELIX_DIR`, `HELIX_RESULT`,
`HELIX_TOML_TEMPLATE`) plus `sandbox.agent_passthrough_env`. A prefix is a
namespace, not a boundary: nothing stopped a credential being named
`HELIX_OPENAI_KEY`, and no config file recorded that it had been.
