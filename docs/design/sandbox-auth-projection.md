# Sandbox auth projection: login volume as source, private state per candidate

**Status:** proposed; implementation requires approval.  This document changes
no runtime behavior.

## Decision in one paragraph

Keep `helix sandbox login <backend>` and the backend-specific
`helix-auth-<backend>` volume.  They remain the operator-facing login store and
the source of truth.  A mutation agent must never mount that shared volume.
For `sandbox.auth = "login"`, HELIX will make a new, random,
candidate-owned Docker volume; a short-lived seeding helper copies an allowlist
of credential files from the login volume into it; and the agent mounts only
that candidate volume inside an otherwise private tmpfs `HOME`.  The candidate
volume starts with credentials only and is removed after the candidate.  Thus
both sequential and concurrent candidates have disjoint writable auth state.

This is an allowlist, not a denylist: the initial candidate auth store is
constructed from the credential manifest alone.  An agent can create arbitrary
files beside its credential, including rotated credentials, but no other
candidate can mount that volume and it is destroyed at cleanup.

## Scope and non-goals

- Login, status, and logout continue to mount `helix-auth-<backend>` exactly as
  they do today.  The guarantee is **no agent container** mounts it, not that
  HELIX never mounts it.
- The login volume remains writable to its login command.  Agent containers do
  not write it, and the seeding helper mounts it read-only.
- This design is about agent authentication.  Evaluators and credentialed
  sidecars retain their existing, separately scoped configuration.
- It is not a mechanism for returning a refreshed credential to the login
  volume.  That limitation is stated in [Rotation](#rotation-and-its-open-question).

## Why a shared writable auth directory is retired

The prior rejection of shared-volume agent execution remains correct, even
though the rejection exception itself will be removed.  A writable auth
directory cannot be made candidate-independent with a denylist: OAuth rotation
requires write and rename authority in the credential directory, so an agent
can create an unenumerated file which a later agent reads.  Resetting a *single*
shared directory between samples can address sequential residue, but cannot
address P×N candidates that run at the same time.

The backend evidence behind that decision is retained here so that a future
change does not reintroduce a direct shared mount:

| Backend | Finding under the retired shared-store design |
| --- | --- |
| Claude | Per-run state lives beside the credential; it cannot be relocated as a split store. |
| Gemini | Per-run state lives beside the credential; its available home knob moves the credential too. |
| Cursor | A config/data split looks plausible but was not verified. Plausible is not an isolation proof. |
| OpenCode | Its session DB shares the only relocation knob with the credential. |
| Codex | Agent-memory databases do isolate: across three clean full-layout runs, `CODEX_SQLITE_HOME` left the shared directory untouched and redirected six files. Codex nevertheless fails on `models_cache.json` alone; it must not be described as leaking agent memory. |

`models_cache.json` is carrying state: its presence and version change later
control flow by skipping or refetching network work.  That result, and the
structural writable-directory argument above, are sufficient reasons never to
mount the login volume in an agent container again.

## Credential-transfer options considered

The credential must cross from the login volume to a candidate without making
the login volume visible to that candidate.  The working image prototype is
important evidence, not a reason to choose its transport by default.

| Option | Secret in agent env / Docker metadata | Agent-image requirement | Runtime-path impact | Rotation and isolation | Verdict |
| --- | --- | --- | --- | --- | --- |
| (a) Base64 environment variable plus entrypoint | Yes: visible in `docker inspect`, PID 1 environment, and process listings until the entrypoint unsets it | Yes, per backend | Small | Per-candidate tmpfs gives good isolation; refreshed state dies with the container | Not a HELIX mechanism. An operator may arrange this inside their own image over an allowlisted `auth = "env"` variable; HELIX does not encode, name, decode, select, or document that arrangement as a path. |
| (b) `docker create` → `docker cp` → `docker start` | No | No | High: replaces one `docker run` with create/copy/start, including stdin, output capture, timeout, and cleanup paths | Can isolate, but needs a helper/source copy and a short-lived host file | Do not choose |
| (c) Candidate auth volume seeded from login volume | No secret in env or container metadata | No per-backend runner image | Moderate, localized to agent launch and cleanup | Writable private auth store; no shared mount; rotation is candidate-local | **Recommended** |
| (d) Read-only bind of the credential file | No | No | Low only in the happy path | Cannot preserve credential-file rotation; parent-path ownership is also problematic | Reject |
| (e) Credential baked into a local derived image | No | One local derivative per backend/credential revision | Low at launch; requires local build, image selection, and rotation rebuild lifecycle | **Does satisfy candidate isolation**, but persists the secret in image layers/cache | Reject as default |

### Evidence gathered for this choice

The stock local Claude runner (`2.1.120`) and Codex runner (`0.125.0`) were
probed without any real credential, network access, or login volume mount.
Both lack their target dot-directory in the base image.  A direct file bind
causes Docker to create the intervening directory as `root:root`; Codex then
fails at startup with `Error loading configuration: Permission denied`.  The
same topology gives Claude a root-owned `.claude` parent.  This repeats the
mount-layout lesson from the working Claude prototype: tests that omit HELIX's
real nested mounts and ownership topology are not integration evidence.

A `docker create` followed by `docker cp` of a Claude credential file also
failed before start because `/home/node/.claude` is absent in the unmodified
image.  Making (b) work requires a bootstrap container or injected startup
command, plus materializing the source credential on the host between two
copies.  When `docker cp` was instead aimed at an already-existing target
directory, it installed the synthetic file as root-owned `0644`, so (b) also
needs a privileged ownership/mode repair before a normal `node` agent can
rotate it.  In contrast, a synthetic source-volume → candidate-volume seed,
followed by a node-owned private write and atomic rename, completed locally in
327 ms.  That is a setup measurement, not a claim about OAuth; it is about
0.4% of a 90-second mutation budget and should be remeasured in the integration
suite.

The read-only bind probe cannot establish live OAuth rotation because it used a
synthetic, unauthenticated record.  It does establish that (d) is not a trivial
drop-in topology.  More fundamentally, a read-only credential file cannot
support a successful atomic refresh write, so it cannot meet the desired normal
CLI semantics even if a particular command tolerates it before expiry.  A
dedicated-grant test remains required for the exact Claude and Codex refresh
behavior.

### Option (e): local credential image

This option genuinely satisfies the isolation requirement.  Every candidate
would start from the same read-only credential layer and can write only to its
own container overlay; it receives no shared writable store.  Isolation is not
the reason to reject it.

Its operational cost is also lower than it may first appear.  Against the
cached local Claude base, a synthetic 516-byte credential image took 558 ms for
the first build and 371 ms after changing only the credential `COPY` input.
Copying that 516-byte file into a stopped container took 35 ms, or 0.039% of a
90-second mutation.  The complete synthetic candidate-volume seed and private
atomic-write check took 327 ms, or 0.36% of the same budget.  These are local
Docker measurements, not portability promises, but they show that avoiding
candidate-time plumbing saves milliseconds rather than meaningful mutation
time.

The baked design loses on lifecycle and at-rest exposure, not latency:

- The credential is recoverable from a local image layer by exporting or
  extracting layers.  `docker history` identifies the credential-bearing
  layer; build cache can retain it even after the visible image tag is removed.
  A candidate volume is credential-bearing only until its narrow cleanup path
  runs, while a tmpfs home dies with its container.
- An observed access-token rotation requires rebuilding the local derivative
  and switching future candidates to its new identity.  Access credentials
  rotated roughly every eight hours in two observations.  If a stale baked
  record cannot refresh in a fresh candidate, every candidate using that one
  image fails together.  Whether a candidate can self-refresh remains
  unmeasured, so this is a real synchronized-risk path rather than a claim that
  refresh is broken.
- Published runner images are registry artifacts pinned by digest.  Baking
  cannot modify those artifacts: it creates a local `FROM <published digest>`
  derivative and requires HELIX to track its base digest, local image id, and
  source-volume revision.  That is a second identity and rotation-update
  lifecycle, rather than the existing one source-of-truth volume.

The 35–327 ms avoided by (e) does not justify adding a long-lived credential
layer and an image-rebuild state machine.  Preserve (e) as an operator-managed
experimental option if needed, but do not make it the default.

## Does the credential need to move at all?

The following alternatives avoid one kind of transfer, but none is a better
current default:

| Alternative | Assessment |
| --- | --- |
| Per-user local derived image | This is option (e): no candidate-time copy, but local secret layers, cache retention, and a rotation/image-identity lifecycle. |
| Long-lived credential sidecar | No supported common CLI protocol lets Claude, Codex, Cursor, Gemini, and OpenCode delegate their local credential lookup to a broker.  A new HTTPS/token proxy would be backend-specific, network-visible, and itself a long-lived shared stateful service.  It is a new auth product, not a smaller projection mechanism. |
| Each candidate logs itself in | Not viable for interactive OAuth.  The installed Claude login help offers subscription/console, email, and SSO choices but no headless token input.  Codex offers `--device-auth` (user-mediated) and `--with-api-key` (a distinct explicit API-key path).  Cursor exposes `NO_OPEN_BROWSER`, which suppresses browser opening rather than supplying a credential.  Gemini's `-p` is headless prompting, not login; HELIX's login command remains interactive.  OpenCode exposes provider management but no verified generic non-interactive OAuth import. |

The one non-interactive path found is Codex `--with-api-key`; it is an explicit
environment/API-key workflow and remains covered by `auth = "env"`.  It does
not preserve the interactive login volume as source of truth.  Therefore the
credential must be made available to a login-authenticated candidate somehow;
the candidate-volume seed is the smallest current mechanism that avoids an
agent environment secret, Docker configuration secret, custom runner
entrypoint, and persistent credential image.

## Recommended mechanism: per-candidate credential-only volume

For each sandboxed agent candidate using `auth = "login"`:

1. HELIX creates a random, labelled Docker volume owned by that candidate.
   The name and labels contain run/candidate identity and backend, never a
   credential.  Collision is an error; a pre-existing volume is never reused.
2. HELIX runs a fixed, short-lived generic seeding helper with network disabled,
   `no-new-privileges`, the login volume mounted read-only at a private source
   path, and the new candidate volume mounted read-write at a private
   destination path.  It copies only the backend manifest's credential files,
   creates the needed directories with the runner uid/gid, applies the required
   file modes, validates the manifest's minimal backend-specific record schema,
   and exits.  It does not list, archive, or copy the source home.  A malformed
   or incomplete record aborts here, before the agent container is created.
3. HELIX starts the ordinary backend runner with a uid/gid-correct tmpfs
   `HOME`, the candidate volume mounted only at that backend's auth directory,
   and any existing per-candidate transcript bind at its nested transcript
   location.  It does not receive the login volume, an environment credential,
   or a credential-bearing command argument.
4. After success, failure, cancellation, or timeout, HELIX removes the agent
   container and then removes that exact candidate volume.  Failed removal is
   reported as a credential-cleanup failure and includes a safe manual cleanup
   identifier; it must never be treated as successful cleanup.  It must refuse
   to remove any volume not carrying the candidate label/prefix and must never
   target `helix-auth-*`.

The candidate volume starts as an allowlisted auth directory, rather than a
copy of the source home.  During a run it may accumulate backend state, but it
is inaccessible to sibling agents and removed afterwards.  The surrounding
tmpfs makes unrelated `HOME` state container-local as well.

The source manifest is code-owned, versioned with the runner contract, and
uses relative paths only.  Initial candidates are:

| Backend | Login-volume source → candidate auth-volume destination | Readiness |
| --- | --- | --- |
| Claude | `.claude/.credentials.json` → `.credentials.json` mounted at `$HOME/.claude` | Ready, subject to real-grant end-to-end test |
| Codex | `.codex/auth.json` → `auth.json` mounted at `$HOME/.codex` | Ready, subject to real-grant end-to-end test |
| Cursor | `.cursor/cli-config.json` → `cli-config.json` mounted at `$HOME/.cursor` | Needs credential-shape and live-login verification |
| Gemini | `.gemini/oauth_creds.json` plus any login-required account record | Needs a measured login-volume manifest; current observed source was not OAuth-backed |
| OpenCode | `.local/share/opencode/auth.json` → `auth.json` mounted at its XDG data auth directory | Needs live-login verification of the complete required file set |

The mechanism is common; its manifest is backend-specific.  No backend needs a
custom agent image merely to receive the credential.  The helper is one generic
HELIX utility action, not five entrypoints; it must be pinned and tested like
other runtime tooling.  A backend is not enabled
for `auth = "login"` until its manifest is confirmed by an authenticated
candidate test.  In particular, Cursor's unverified split must not be promoted
to a claim of readiness merely because a candidate volume would isolate it.

## Earlier image-side experiments

An earlier image experiment authenticated in a private home, removed its
input from the process environment before agent exec, wrote mode `0600`, and
failed closed on malformed data. It also exposed the root-owned nested
transcript-parent failure that this design avoids with a uid/gid-correct tmpfs
HOME and pre-created nested mount path. Its nine-hour, 37-generation run is
useful evidence for the livebench write-up.

It is not a HELIX auth mechanism. HELIX has exactly two paths: `auth =
"login"` and `auth = "env"`. HELIX contains no projection-variable name,
encoding, decoding, runner-image selection, or compatibility branch. An
operator-owned image may independently interpret an allowlisted environment
value under `auth = "env"`; that image behavior is outside HELIX's contract.

## Configuration surface

`auth` describes the **source of the credential**, not the mount or transport
mechanism.  The production surface stays small:

```toml
[sandbox]
enabled = true
auth = "login"  # "login" (from helix-auth-<backend>) or "env" (explicit host value)
auth_env_allow = []
```

- `auth = "login"` selects the candidate-volume projection described here.
  `auth_env_allow` must be empty: no credential transport variable is needed.
- `auth = "env"` means an explicit host environment value is the credential
  source; `auth_env_allow` remains the non-empty scrub allowlist of names that
  may reach the agent.  It is not inferred from backend or host state.
- No `auth_projection_*`, file path, volume name, or backend-specific
  transport key is exposed in configuration.  Those would make a small source
  choice into a second configuration language.

This retains `auth_env_allow` because it is the actual final scrub allowlist
for named values.  It does not use that key as hidden plumbing for login
credentials.  The validation/documentation change should stay roughly 30–50
lines; backend manifests and runtime cleanup are implementation details, not
new user knobs.

## Deletions and documentation to retain

The implementation removes:

- `VolumeModeUnsupportedError` and its long retired-mode remediation text;
- the agent-execution paths that mount `helix-auth-<backend>`, including any
  post-run transcript copy that remounts it;
- `OAUTH_SUPPRESSING_ENV`;
- `FORBIDDEN_AUTH_ENV_NAMES` and every policy branch that specially handles
  `CLAUDE_CODE_OAUTH_TOKEN`.

HELIX must not forbid or otherwise manipulate a variable it does not set.  The
existing warning for both credential forms being present remains the right
diagnostic for an ambiguous explicit-env configuration.

The history in [Why a shared writable auth directory is retired](#why-a-shared-writable-auth-directory-is-retired)
stays as product documentation.  Delete the retirement code, not the reason
for structural isolation.

## Rotation and its open question

The candidate auth volume is writable, so a backend can attempt its normal
in-container OAuth refresh and atomic rename.  Any refreshed credential stays
only in that candidate volume and dies when the volume is removed; it is never
written back to the login volume. The earlier image experiment observed the
same candidate-local rotation boundary during a nine-hour, 37-generation run.

Whether in-container OAuth refresh works at all is **unmeasured**.  Two prior
attempts were void: the supervising process shared the grant and its own API
activity revoked the grant under observation.  They do not show refresh working
or broken.  Resolving the question requires a dedicated grant whose observer
does not use it.

## Verification plan

Unit and integration tests must make the structural boundary observable on the
final Docker argv and live Docker objects, not merely on a helper function.

1. **No shared source mount.** For every login-enabled backend, assert the
   final agent argv and `docker inspect` contain no `helix-auth-<backend>`
   mount.  The seeding helper may contain that name only as a read-only source;
   it must not be the agent container.
2. **Credential-only seed.** Seed a disposable source with a credential canary
   and unrelated state canaries.  The candidate volume contains the credential
   paths and no unrelated source state before the agent starts.
3. **No credential in agent environment or metadata.** Assert the agent
   environment, command, labels, and inspect configuration omit credential
   values. The candidate-volume name alone is not secret.
4. **Malformed and missing source fail closed.** Missing, malformed, oversized,
   or wrong-mode source material must prevent the agent from starting with a
   stable, redacted error.  This is mandatory: silently continuing turns a
   credential transport bug into a misleading `not logged in` failure.
5. **Sibling isolation.** In concurrent C1/C2 and sequential A→B canaries,
   make A/C1 write a state marker next to its credential; B/C2 must not read
   it.  Assert different candidate volume identities and successful removal.
6. **Cleanup failure is loud and narrow.** Force volume-removal failure and
   verify the result names only the candidate volume, retains no shared-volume
   deletion path, and makes manual cleanup actionable.
7. **Real mount topology.** Run the Claude test with tmpfs `HOME`, the nested
   transcript bind, workspace bind, and runner user.  This guards against the
   root-owned intervening-directory failure found by the proven runner.
8. **Dedicated-grant rotation test.** For Claude and Codex, start a candidate
   after its access token expires while retaining a valid refresh token, then
   distinguish authenticated refresh from a revoked-grant failure.  Confirm no
   changed record reaches the source volume.  Do not run the observer against
   the same grant.

## Migration and errors

`sandbox.auth = "volume"` is invalid after this change because it describes
the retired mount mechanism, not a source.  Config loading should fail before
Docker is invoked with:

```
sandbox.auth = "volume" has been replaced by "login".
  auth = "login" reads the credential from helix-auth-<backend> at candidate
  start, copies only the credential into a private candidate store, and never
  mounts the shared login volume in an agent container.
  Set sandbox.auth = "login"; do not use environment credential transport.
```

Existing `auth = "env"` configurations continue to mean explicit named host
values and continue to require a non-empty `auth_env_allow`.  Configurations
that set `auth_env_allow` with `auth = "login"` fail with an error explaining
that login copying has no environment transport. `auth = "env"` remains the
separate, explicit API-key mechanism and is gated by `auth_env_allow`.

This makes the ordinary login-volume workflow structural and image-neutral. It
also introduces the small `sandbox.auth` and `auth_env_allow` configuration
contract needed for an explicit choice of credential source.
