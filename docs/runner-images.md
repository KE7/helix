# Mutation-agent runner supply chain

HELIX runs each mutation agent inside a per-backend container image. This
document covers how those images are built, what is pinned, and how to change
or roll back what they run.

## What is pinned, and why

This is the point of the whole pipeline. No image ever runs
`npm install -g <cli>@latest` or `curl https://cursor.com/install | bash`.
Every byte that lands in an image is fetched from an expected host over HTTPS
and verified against a recorded checksum before it is unpacked. Those pins live
in two files, split by who owns the format:

| Layer | Pin | Where |
| --- | --- | --- |
| Base OS | `node:22-bookworm-slim` by `sha256:` digest | `docker/base.Dockerfile` |
| Base packages | Debian snapshot `20260720T000000Z` — a frozen apt archive | `docker/runner-versions.json` |
| `uv` | version-pinned wheel URL + SHA-256, per architecture | `docker/runner-versions.json` |
| claude / codex / opencode CLIs | tarball URL + sha512 integrity, plus each native subpackage | `docker/package-lock.json` |
| Gemini CLI | bundled parent tarball URL + SHA-512 | `docker/runner-versions.json` |
| `@lydell/node-pty` (Gemini's native dep) | tarball URL + sha512, per architecture | `docker/package-lock.json` |
| Cursor CLI | archive URL + SHA-256, per architecture | `docker/runner-versions.json` |

### Why a lockfile

`package-lock.json` already stores, for every platform-native binary, exactly
what a pin catalog would store by hand: the resolved URL, the sha512 integrity,
and the `cpu`/`os`/`libc` constraints — including transitive optional
dependencies. Maintaining a second copy of that by hand was the single largest
source of code in this pipeline.

It is generated with `npm install --package-lock-only --ignore-scripts`, which
resolves without installing. **This is a pin store, not an install step.** The
Dockerfiles still hand-extract three verified tarballs each. `npm ci` is
deliberately not used: it would pull 655 packages for Gemini alone, where the
image today holds one bundled `bundle/gemini.js` plus node-pty, and it would
introduce install-script execution the build has never had.

Gemini's own tarball stays a hand-maintained pin because its unbundled
dependency tree is 655 packages and would make the lockfile 8,459 lines, even
though its published tarball ships self-contained. Cursor stays hand-maintained
because it belongs to no package ecosystem at all: its version lives inside a
shell script, and its archives must be downloaded and hashed directly.

Renovate maintains the lockfile natively, so a version bump and its integrity
hash arrive in the same PR. Against the previous bespoke catalog it could only
have bumped version strings and left the digests stale.

### What the tool enforces

`tools/runner_images.py validate` checks both files: npm downloads only from
`registry.npmjs.org`, Cursor archives only from `downloads.cursor.com`, HTTPS
only, digests of the correct length, and — importantly — that every artifact an
image extracts is named in `LOCKFILE_ARTIFACTS` and really is a `linux` binary
for the expected CPU and a glibc one.

That allowlist is what stops a build from extracting a package nobody reviewed.
npm records whatever optional dependencies upstream declared; the tool records
what a human approved. A new native optional dependency appearing upstream
cannot enter an image without a line changing in the tool. It is named per
package rather than queried by `cpu`/`os`/`libc`, because those fields do not
identify a unique artifact — claude publishes `linux-x64` and `linux-x64-musl`,
opencode publishes `linux-x64` and `linux-x64-baseline`.

The residual catalog rejects a version or digest for a lockfile-owned backend,
so the two files cannot drift into disagreeing about the same fact.

Cursor's installer is downloaded but never executed. It is parsed for its
single embedded release version and the official archive URLs, which are then
fetched and checksummed directly. Because those archives are version-addressed
and immutable, an unchanged version reuses the reviewed checksum in the catalog
instead of re-downloading hundreds of megabytes every night; any drift in the
version or the URL falls back to a full re-download and re-hash.

The base bootstraps apt over `http://snapshot.debian.org` because the slim base
has no CA certificates yet — `ca-certificates` is installed *by* that same apt
run. APT still verifies Debian's signed release metadata, and the residual risk
of being served an older signed snapshot is bounded by the pinned timestamp.
Every later download uses HTTPS plus a recorded content digest.

## When images get rebuilt

**Nightly is the check cadence, not the release cadence.** The scheduled run
asks each upstream for its latest release and compares it against the registry:

```
resolved_version = discover upstream latest        (npm dist-tags / cursor installer)
if <image>:<resolved_version> already exists  ->  skip
else                                          ->  build and publish
```

If nothing shipped upstream, the run is a clean no-op. A backend is rebuilt
because Codex released 0.146.0, not because a day passed.

Tags published per backend:

- `ghcr.io/ke7/helix-evo-runner-<backend>:<upstream-version>` — e.g. `codex:0.145.0`
- `ghcr.io/ke7/helix-evo-runner-<backend>:latest` — moved on a successful publish

and for the shared runtime:

- `ghcr.io/ke7/helix-evo-runner-base:node22-uv<uv>-snapshot<date>-r<6hex>`
- `ghcr.io/ke7/helix-evo-runner-base:latest`

The base tag's six-hex suffix covers the whole base recipe — `base.Dockerfile`
itself plus the node digest, snapshot, uv version, and wheel digests. Without
it, editing `base.Dockerfile` would leave the tag unchanged, the "already
published?" check would answer *yes*, and every backend would silently build
`FROM` a stale base.

## Publication sequence

`.github/workflows/publish-runners.yml` is six jobs:

1. **`guard`** — accept only `KE7/helix` on its `main` default branch, from a
   schedule or a manual dispatch. Pull requests and forks cannot publish.
2. **`resolve`** — discover upstream versions, ask the registry what already
   exists, emit the build matrix. Absence is decided by `imagetools inspect`,
   where *only* a definitive "no such tag" counts as absent; an auth, network,
   rate-limit, or server error aborts rather than being mistaken for "nothing
   published yet".
3. **`base`** — build and publish the base runtime, only when its recipe tag is
   absent (or `force` is set).
4. **`backend`** — one job per changed backend. Each builds **both**
   architectures in a single `--platform linux/amd64,linux/arm64` invocation
   under QEMU. This is cheap because the Dockerfiles only download, verify, and
   untar prebuilt binaries — nothing compiles.
5. **`rollback`** — manual dispatch only.
6. **`notify`** — file or update one labelled issue on failure *or*
   cancellation.

Jobs 3 and 4 both follow the same order, and the order is the point:

```
build and push BY DIGEST (no tag exists yet)
  -> assert the index really has linux/amd64 and linux/arm64
  -> smoke each architecture: --network none, --read-only, private tmpfs HOME,
     uid/gid 1000, no credential mount, and an anchored version assertion
  -> for Codex, also parse `codex debug models --bundled` and require the exact
     slug gpt-5.6-luna with reasoning levels low, medium, high, xhigh, max
  -> attach BuildKit SBOM + provenance and an OIDC-signed GitHub attestation
  -> verify that attestation
  -> only now create the :<version> and :latest tags, then read them back
```

**No tag ever names an image that has not passed both smokes.** A failed or
cancelled run leaves every previously published tag exactly where it was.

Because absence is decided by a tag's existence rather than by verifying its
attestation, the discovery path trusts that only this workflow can write to
these repositories — it holds the only `packages: write` grant. Attestation is
still generated and verified on every publish and every rollback.

The per-backend matrix uses `fail-fast: false`, so one backend failing does not
block the others; each backend publishes independently.

## Changing a recipe

Editing a Dockerfile, bumping the Debian snapshot, or moving a uv wheel is not
an upstream release, so the nightly check will not notice it. Ship it with a
manual dispatch:

```
workflow_dispatch(operation: refresh, force: true)                   # everything
workflow_dispatch(operation: refresh, force: true, backend: codex)   # one backend
```

A base recipe change is picked up automatically, because the base tag is bound
to the recipe.

If the upstream version has already been published, a forced rebuild does
**not** overwrite that tag. It publishes `<version>-r<run_id>` and moves
`latest` there, so `codex:0.145.0` always means the bytes it meant on the day
it was first published. This matters because a version tag may already be
pinned in someone's `sandbox.image` config.

## Rolling back

Immutable version tags and every prior digest are never deleted, so a rollback
is a retag against something the registry already holds:

```
workflow_dispatch(operation: rollback, backend: codex, digest: sha256:...)
```

The job validates the backend against the five literals and the digest against
`^sha256:[0-9a-f]{64}$`, asserts both platforms are present, verifies the
target's GitHub attestation, runs the smoke on both architectures, moves
`latest`, and reads it back. No ledger, no compensating transaction, no signal
handling — there is nothing to compensate, because nothing is destroyed.

Rollback has its own concurrency group, so an incident is never queued behind
an in-flight nightly build.

## Local proof

Build the base first, then a backend against it:

```sh
docker build --file docker/base.Dockerfile --tag helix-runner-base:proof .

docker build \
  --build-arg BASE_IMAGE=helix-runner-base:proof \
  --file docker/codex.Dockerfile \
  --tag helix-runner-codex:proof .

docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev \
  --tmpfs /home/node:rw,nosuid,nodev,uid=1000,gid=1000 \
  --user 1000:1000 \
  helix-runner-codex:proof codex debug models --bundled > /tmp/codex-models.json

python tools/runner_images.py verify-codex-catalog --input /tmp/codex-models.json
```

To regenerate the npm pins after editing `docker/package.json` (Renovate
normally does this for you):

```sh
cd docker && npm install --package-lock-only --ignore-scripts
python tools/runner_images.py validate --catalog docker/runner-versions.json
```

Use a task-specific tag; do not reuse `latest`. This proves the Dockerfiles and
source pins on your native architecture only — the published multi-arch index
digest and its attestations are the real release identity.

## Tests

`tests/unit/test_runner_images.py` covers all of this offline — no test
reaches the network. It validates the catalog and the lockfile, resolves every
backend, checks that the build arguments cover exactly the ARGs each Dockerfile
declares (in both directions) and that their digests are the lockfile's own
integrity fields, exercises the change-detection policy, the forced-rebuild tag
rule, and the base-tag recipe binding, and invokes all seven CLI subcommands
through `main()`.

Two workflow properties are asserted here because no linter knows them: that no
tag is created before the image passes both smokes, and that the registry
absence check is fail-closed. Everything else about the workflow — YAML
validity, expression types, shell correctness, template injection, unpinned
actions, credential persistence, permission scope — is checked by **actionlint**
and **zizmor**, which run as a blocking `lint-workflows` job in CI. Those
replaced roughly 130 lines of hand-rolled source-text grep tests.
