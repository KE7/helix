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
build. Codex is stricter still: its launcher tarball and its exact
`linux-x64`/`linux-arm64` native tarballs are independently verified and
extracted without a second npm resolution.

The base is pinned to
`node:22-bookworm-slim@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3`,
Debian snapshot `20260720T000000Z`, and `uv==0.11.7`. The two `uv` wheels have
architecture-specific `files.pythonhosted.org` URLs and SHA-256 digests.
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
   platform parity, attach BuildKit SBOM/provenance and an OIDC-signed GitHub
   build-provenance attestation, and only then create the immutable tag.
6. Preflight all eligible convenience-tag targets together. Durable
   `rollback-before-<run>-<attempt>` tags are created before mutation. A
   single promotion job moves `latest` tags and compensates by restoring every
   already-moved tag if a later move fails.

The daily job deliberately does **not** move `latest` when a CLI version is
newer than the checked-in HOME-layout measurement in
`src/helix/backend_layout.py`. In that case the immutable image is published
and retained, but `latest` remains the known rollback image until the layout is
remeasured and its guard version is updated. This prevents a routine upstream
release from silently invalidating HELIX's shared-auth isolation evidence.

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

Every successful convenience-tag promotion records the previous and new
digests, immutable tag, durable rollback tag, and workflow run ID in a 90-day
artifact. Immutable version and rollback tags are not deleted. A manual
rollback accepts an explicit backend and `sha256:` manifest digest only through
validated environment variables, runs the smoke (and Codex catalog assertion)
on both native architectures, verifies the target's GitHub attestation and
exact amd64/arm64 parity, creates another durable rollback tag, and only then
moves `latest`. The workflow has no image-delete step and never removes runner
images.

## Local Codex proof

Use new task-specific tags; do not reuse `latest` or any preserved tag. Build
the exact base recipe first, then pass that content by local image ID to the
backend build:

```sh
docker build \
  --file docker/base.Dockerfile \
  --tag helix-runner-base:luna-proof \
  .

base_id="$(docker image inspect --format '{{.Id}}' \
  helix-runner-base:luna-proof)"
docker build \
  --build-arg "BASE_IMAGE=${base_id}" \
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
