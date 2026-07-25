# Mutation-agent runner supply chain

The five mutation-agent images are built from `docker/runner-versions.json`.
That file is the reviewable source manifest; it records the exact upstream
version, download URL, and digest used by a local build. The protected workflow
also retains its freshly resolved manifest and change plan for 90 days.

## Upstream sources

| Backend | Authoritative version/install source | Checked-in version |
| --- | --- | --- |
| Claude Code | [`@anthropic-ai/claude-code` on npm](https://www.npmjs.com/package/@anthropic-ai/claude-code) | 2.1.218 |
| Codex | [`@openai/codex` on npm](https://www.npmjs.com/package/@openai/codex) and the matching OpenAI platform package | 0.145.0 |
| Cursor Agent | [Cursor's official installer](https://cursor.com/install), parsed only for the version and official artifact URL | 2026.07.20-8cc9c0b |
| Gemini CLI | [`@google/gemini-cli` on npm](https://www.npmjs.com/package/@google/gemini-cli) | 0.52.0 |
| OpenCode | [`opencode-ai` on npm](https://www.npmjs.com/package/opencode-ai) | 1.18.4 |

The resolver accepts npm tarballs only from `registry.npmjs.org`, Cursor
artifacts only from `downloads.cursor.com`, HTTPS only, stable versions only,
and SHA-512 (npm) or SHA-256 (Cursor) digests of the expected length. An
ambiguous Cursor installer, a prerelease npm `latest`, malformed metadata, a
timeout, or a host change fails closed.

Cursor's installer is never executed in an image. The workflow hashes the
installer, extracts its single embedded release version, downloads both
official Linux architecture archives, and hashes those archives before a
build. No backend Dockerfile runs `npm install`. Claude, Codex, and OpenCode
extract their verified launcher plus the exact Linux native package selected
for each architecture. OpenCode's amd64 image includes both its AVX2 and
baseline binaries and chooses at runtime. Gemini extracts its verified bundle
plus the exact `@lydell/node-pty` selector and architecture package; its
documented child-process fallback remains available without the unneeded
optional keychain and legacy PTY modules. The manifest records the launcher's
full optional-dependency map and every extracted child package URL and SHA-512
digest, so neither a build nor a same-version upstream republish can introduce
an unmeasured package.

The base is pinned to
`node:22-bookworm-slim@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3`,
Debian snapshot `20260720T000000Z`, and `uv==0.11.7`. The two `uv` wheels have
architecture-specific `files.pythonhosted.org` URLs and SHA-256 digests.
The CA-less slim base bootstraps from the snapshot's HTTP endpoint while APT
still verifies Debian's signed release metadata; all later artifact downloads
use HTTPS plus the recorded content digest.
Changing any base recipe input requires a new base recipe tag; backend
immutable tags are bound to that tag, the backend Dockerfile hash, and every
backend artifact URL/checksum by a digest suffix. The resulting OCI digest,
BuildKit SBOM, and signed provenance remain the byte-exact release identity.

## Daily publication sequence

`.github/workflows/publish-runners.yml` runs daily and by
`workflow_dispatch`. One repository-wide concurrency group is serialized with
`cancel-in-progress: false`.

1. Resolve all five official upstreams and validate their checksums.
2. Derive a collision-resistant `cli-<version>-r<recipe-hash>` tag bound to
   the exact base and backend recipes, including Dockerfile and upstream
   source digests. An existing tag counts as published only after exact
   two-platform manifest validation and GitHub attestation verification.
   Otherwise the workflow fails closed. Build absent versions on native
   amd64 (`ubuntu-latest`) and arm64 (`ubuntu-24.04-arm`) runners.
3. Push each architecture by an otherwise untagged digest. Run the backend
   version smoke on that exact digest, natively, with no network, a read-only
   root filesystem, private tmpfs HOME, uid/gid 1000, and no credential mount.
4. For Codex, additionally parse `codex debug models --bundled` and require the
   exact model slug `gpt-5.6-luna` with ordered reasoning levels
   `low, medium, high, xhigh, max`. This proves that `xhigh` is present and is
   second-highest in the image's shipped catalog.
5. Merge only two architecture-qualified smoke records, verify exact runtime
   platform parity, and attach BuildKit SBOM/provenance plus an OIDC-signed
   GitHub build-provenance attestation. The serialized publication step then
   queries GHCR directly: only an authenticated HTTP 404 is treated as an
   absent tag; authorization, network, and registry errors fail closed.
   Collision validation and tag creation happen together after attestation,
   followed by an exact digest re-read. Every v4 workflow artifact name also
   includes `run_attempt`, so “re-run all jobs” writes a new artifact family
   instead of colliding with the prior attempt's immutable artifacts. Consumers
   download all attempt-qualified artifacts from the same workflow run and
   select the newest producer at or before the current attempt independently
   per architecture or backend. This lets a failed-job or specific-job rerun
   reuse successful sibling artifacts from an earlier attempt while replacing
   only producers that reran. Selection fails closed on future, duplicate,
   missing, unexpected, symlinked, or file-colliding artifact inputs, and on
   symlinked or canonically overlapping input/output roots before any copy.
6. Preflight all eligible convenience-tag targets together. Durable
   `rollback-before-<run>-<attempt>` tags are created before mutation, and the
   complete rollback plan is retained before any `latest` tag moves. A single
   promotion job moves `latest` tags and compensates by restoring every
   already-moved tag if a later move fails. If retaining the committed ledger
   fails after a move, the same compensation restores the previous tags; the
   pre-move evidence and durable rollback tags remain.

The tag-move step replaces its shell with the transaction process so GitHub's
cancel signals reach it directly. It treats ordinary command errors, `SIGINT`,
and `SIGTERM` as compensated failures, restores attempted moves in reverse, and
re-reads every restored digest. [GitHub's cancellation
reference](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-cancellation)
states that the runner first sends `SIGINT`, waits 7.5 seconds, then sends
`SIGTERM`, waits 2.5 seconds, and may then kill the process tree; the server
also has a five-minute forced-cancellation limit. A forced kill cannot be made
transactionally recoverable by an in-process handler. The workflow therefore
retains the complete plan and creates rollback tags before mutation, so that
hard-cancellation recovery remains evidence-backed even when the process is not
allowed to finish compensation.

## Promotion is a separate, maintainer-approved phase

Publishing an immutable tag is automatic. Moving the mutable `latest` tag that
`src/helix/backends.py` actually runs is not. Each backend in
`docker/runner-versions.json` carries two maintainer-owned guard fields:

| Field | Meaning |
| --- | --- |
| `promotion_guard_version` | The upstream CLI version a maintainer has reviewed |
| `promotion_guard_immutable_tag` | The exact `cli-<version>-r<recipe-hash>` tag that review covered, or `null` for "nothing approved" |

`latest` moves only when the freshly built immutable tag equals
`promotion_guard_immutable_tag` **and** the discovered version equals
`promotion_guard_version`. Both comparisons are exact string equality, so a
same-version content republish (a new launcher tarball, a changed Dockerfile,
or a new base recipe) mints a different immutable tag and cannot inherit the
previous approval. The immutable image is still published and retained;
`latest` keeps pointing at the last approved image.

Every backend currently ships `promotion_guard_immutable_tag: null` and a guard
version behind its pinned version, so **no automatic promotion happens today**.
That is deliberate. To keep it from rotting silently, the `resolve` job emits a
`promotion_stalled` output, a `::warning::` annotation per stalled backend, and
a step-summary table naming each backend, its built immutable tag, and its
stale guard values; the same detail is retained in `promotion-stall.json`
inside the `resolved-runner-plan-*` artifact.

To approve a promotion, a maintainer:

1. Reads the stall table (or runs
   `python tools/runner_images.py image-tag --catalog docker/runner-versions.json --backend <backend>`)
   to get the built immutable tag.
2. Verifies the built image — its digest, SBOM, provenance attestation, and
   the smoke evidence retained by the run.
3. Sets that backend's `promotion_guard_version` to the pinned `version` and
   `promotion_guard_immutable_tag` to the built tag in
   `docker/runner-versions.json`, and merges the change to `main` through
   normal review.

The next scheduled run then sees an exact guard match and promotes that one
backend inside the compensated transaction described above. Reverting the guard
fields is enough to stop future promotions; it does not move `latest` back, for
which the manual `rollback` dispatch exists.

Pull requests cannot invoke this publishing workflow. Its discovery, change
matrix, collision policy, platform parity, model-catalog contract, and static
no-PR trigger are covered by offline unit tests in
`tests/unit/test_runner_images.py`.

Schedule and manual events are accepted only in `KE7/helix` on its reviewed
`main` default branch. Every third-party action is pinned to a full commit SHA.
Attestation verification requires this workflow path, `refs/heads/main`, and
GitHub-hosted runners. Release builds do not consume shared GitHub Actions build
caches, so a cache produced by another ref cannot bypass a download-integrity
step.

Only build jobs and promotion jobs receive `packages: write`; discovery has
read-only package access. Only attestation jobs receive `id-token: write` and
`attestations: write`. Only the failure notifier receives `issues: write`.
A build or merge failure prevents the single promotion job from starting. A
promotion-time failure triggers compensating rollback and creates or updates
one repository issue.

## Rollback and retention

Before every convenience-tag promotion, a 90-day rollback-plan artifact records
the previous and new digests, immutable tag, durable rollback tag, workflow run
ID, and attempt. A successful move additionally records a committed ledger. If
that second upload fails, the workflow restores the prior `latest` digest and
fails. Immutable version and rollback tags are not deleted. A manual rollback
uses the same prepare-retain-move-retain-compensate transaction: it accepts an
explicit backend and `sha256:` manifest digest only through validated
environment variables, runs the smoke (and Codex catalog assertion) on both
native architectures, verifies the target's GitHub attestation and exact
amd64/arm64 parity, creates another durable rollback tag, and retains its plan
before moving `latest`. The workflow has no image-delete step and never removes
runner images.

## Local Codex proof

Use new task-specific tags; do not reuse `latest` or any preserved tag. Build
the exact base recipe first, record its local content ID, then pass its unique
task tag to the backend build. (BuildKit does not portably accept a daemon
image ID in `FROM`; the protected workflow instead uses the registry digest
directly.)

```sh
docker build \
  --file docker/base.Dockerfile \
  --tag helix-runner-base:luna-proof \
  .

docker image inspect --format '{{.Id}}' \
  helix-runner-base:luna-proof
docker build \
  --build-arg BASE_IMAGE=helix-runner-base:luna-proof \
  --tag helix-runner-codex:0.145.0-luna-proof \
  --file docker/codex.Dockerfile .

docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev \
  --tmpfs /home/node:rw,nosuid,nodev,uid=1000,gid=1000 \
  --user 1000:1000 \
  helix-runner-codex:0.145.0-luna-proof codex --version

docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev \
  --tmpfs /home/node:rw,nosuid,nodev,uid=1000,gid=1000 \
  --user 1000:1000 \
  helix-runner-codex:0.145.0-luna-proof \
  codex debug models --bundled > /tmp/codex-models.json

python tools/runner_images.py verify-codex-catalog \
  --input /tmp/codex-models.json
```

This proves the exact Dockerfiles and source pins on the host's native
architecture. It is not the workflow's published multiarch artifact: only the
registry index digest, its two child digests, and its attestations can establish
that identity. The catalog proof is offline and does not claim that an
authenticated service request succeeded. A service canary, if required, must
use a dedicated disposable credential source and a metered request; it must
never mount a shared `helix-auth-*` volume.
