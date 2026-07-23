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
`node:22-bookworm-slim@sha256:6c74791e557ce11fc957704f6d4fe134a7bc8d6f5ca4403205b2966bd488f6b3`
and `uv==0.11.7`. Debian package installation can still change if the base is
intentionally rebuilt against a later Debian repository snapshot; the
resulting OCI digest, BuildKit SBOM, and signed provenance are therefore the
byte-exact identity. Source pins provide a functional rebuild recipe, while
the digest is the immutable rollback/reuse handle.

## Daily publication sequence

`.github/workflows/publish-runners.yml` runs daily and by
`workflow_dispatch`. One repository-wide concurrency group is serialized with
`cancel-in-progress: false`.

1. Resolve all five official upstreams and validate their checksums.
2. Treat an existing `cli-<version>` tag as unchanged; build only absent
   versions. Build amd64 on `ubuntu-latest` and arm64 on
   `ubuntu-24.04-arm`.
3. Push each architecture by an otherwise untagged digest. Run the backend
   version smoke on that exact digest, natively, with no network, a read-only
   root filesystem, private tmpfs HOME, uid/gid 1000, and no credential mount.
4. For Codex, additionally parse `codex debug models --bundled` and require the
   exact model slug `gpt-5.6-luna` with ordered reasoning levels
   `low, medium, high, xhigh, max`. This proves that `xhigh` is present and is
   second-highest in the image's shipped catalog.
5. Merge only two passing runtime platforms, publish the immutable
   `cli-<version>` manifest, attach BuildKit SBOM/provenance and an
   OIDC-signed GitHub build-provenance attestation, then consider mutable-tag
   promotion.

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

Only build jobs and promotion jobs receive `packages: write`; discovery has
read-only package access. Only attestation jobs receive `id-token: write` and
`attestations: write`. Only the failure notifier receives `issues: write`.
A failed run creates or updates one repository issue and never promotes a
mutable tag.

## Rollback and retention

Every successful convenience-tag promotion records the previous and new
digests, immutable tag, and workflow run ID in a 90-day artifact. Immutable
version tags are not deleted. A manual rollback requires an explicit backend
and `sha256:` manifest digest, runs the smoke (and Codex catalog assertion) on
both native architectures, verifies exact amd64/arm64 parity, and only then
moves `latest`. The workflow has no image-delete step and never removes runner
images.

## Local Codex proof

Use a new task-specific tag; do not reuse `latest` or any preserved tag:

```sh
docker build \
  --build-arg BASE_IMAGE=helix-runner-base:latest \
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

The catalog proof is offline and does not claim that an authenticated service
request succeeded. A service canary, if required, must use a dedicated
disposable credential source and a metered request; it must never mount a
shared `helix-auth-*` volume.
