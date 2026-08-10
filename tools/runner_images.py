#!/usr/bin/env python3
"""Resolve and validate content-addressed HELIX mutation-runner inputs.

Network discovery and pure planning are deliberately separate.  Unit tests use
captured registry/installer fixtures; only the protected release workflow calls
``discover`` against upstream services or checks registry tags.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, TypeVar


BACKENDS = ("claude", "codex", "cursor", "gemini", "opencode")
PLATFORMS = ("amd64", "arm64")
# The exact lockfile entry each image extracts, per backend and platform.
#
# This stays a checked-in decision rather than a query over the lockfile's
# cpu/os/libc fields, because those fields do not identify a unique artifact:
# claude publishes linux-x64 (libc glibc) *and* linux-x64-musl for the same
# cpu/os, and opencode publishes linux-x64 (AVX2) *and* linux-x64-baseline with
# identical cpu/os/libc. Naming the package is also what preserves the property
# that a build extracts exactly the artifacts someone reviewed and nothing
# else: a new optional dependency appearing upstream cannot enter an image
# without a line changing here.
#
# ``cli`` is the wrapper package; ``shared`` is architecture-independent.
# Gemini has no ``cli`` entry -- its own tarball is a residual catalog pin,
# because its 655-package unbundled dependency tree would make the lockfile
# 8,459 lines even though the published tarball ships a self-contained bundle.
LOCKFILE_ARTIFACTS: dict[str, dict[str, str]] = {
    "claude": {
        "cli": "@anthropic-ai/claude-code",
        "amd64": "@anthropic-ai/claude-code-linux-x64",
        "arm64": "@anthropic-ai/claude-code-linux-arm64",
    },
    "codex": {
        "cli": "@openai/codex",
        # npm aliases of @openai/codex itself, not separate packages:
        # `npm:@openai/codex@<version>-linux-x64`.
        "amd64": "@openai/codex-linux-x64",
        "arm64": "@openai/codex-linux-arm64",
    },
    "gemini": {
        "shared": "@lydell/node-pty",
        "amd64": "@lydell/node-pty-linux-x64",
        "arm64": "@lydell/node-pty-linux-arm64",
    },
    "opencode": {
        "cli": "opencode-ai",
        # Baseline runs on every x86-64 CPU; see docker/opencode.Dockerfile for
        # why the AVX2 build and its runtime /proc/cpuinfo branch were dropped.
        "amd64": "opencode-linux-x64-baseline",
        "arm64": "opencode-linux-arm64",
    },
}
LOCKFILE_CPU = {"amd64": "x64", "arm64": "arm64"}
# Backends whose own tarball stays a hand-maintained catalog pin.
RESIDUAL_PARENTS = frozenset(
    name for name, group in LOCKFILE_ARTIFACTS.items() if "cli" not in group
)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA512_RE = re.compile(r"^[0-9a-f]{128}$")
VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]*$")
# ``smoke_command`` reaches ``sh -lc`` inside the build container.  The catalog
# is an in-repo trust boundary, so this is defense in depth rather than an
# exploitable path, but the charset deliberately excludes ``;``, backticks,
# ``$``, ``(``, ``)``, ``<``, ``>``, backslash, and newlines.
SMOKE_COMMAND_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z _.=/&|'\"-]{0,255}$")
TAG_RE = re.compile(r"^[0-9A-Za-z_][0-9A-Za-z_.-]{0,127}$")
CURSOR_VERSION_RE = re.compile(r"2026\.[0-9]{2}\.[0-9]{2}-[0-9a-f]+")
# HELIX needs a Codex that ships gpt-5.6-luna with an xhigh reasoning level.
CODEX_MINIMUM_VERSION = (0, 145, 0)

DEFAULT_LOCKFILE = (
    Path(__file__).resolve().parent.parent / "docker" / "package-lock.json"
)

T = TypeVar("T")
Sleep = Callable[[float], None]
NETWORK_RETRY_ATTEMPTS = 3
RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})


class RunnerPlanError(ValueError):
    """A release input is ambiguous, malformed, or violates promotion policy."""


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RunnerPlanError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _require_url(value: object, host: str) -> str:
    if not isinstance(value, str):
        raise RunnerPlanError("expected URL string")
    if not value or any(ord(character) <= 0x20 for character in value):
        raise RunnerPlanError(f"URL contains whitespace or controls: {value!r}")
    parsed = urllib.parse.urlparse(value)
    try:
        port = parsed.port
    except ValueError as exc:
        raise RunnerPlanError(f"untrusted upstream URL: {value!r}") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != host
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
        or not parsed.path.startswith("/")
        or parsed.fragment
    ):
        raise RunnerPlanError(f"untrusted upstream URL: {value!r}")
    return value


def _semver_tuple(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise RunnerPlanError(f"expected stable semantic version, got {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _require_source(
    source: object,
    host: str,
    digest: re.Pattern[str],
    digest_field: str,
    context: str,
) -> None:
    """Require a download to be an HTTPS URL on ``host`` plus a content digest."""
    if not isinstance(source, dict):
        raise RunnerPlanError(f"{context}: expected an object")
    _require_url(source.get("tarball") or source.get("url"), host)
    if not digest.fullmatch(str(source.get(digest_field, ""))):
        raise RunnerPlanError(f"{context}: invalid {digest_field}")


def _require_version(value: object, context: str) -> str:
    if not isinstance(value, str) or not VERSION_RE.fullmatch(value):
        raise RunnerPlanError(f"{context}: invalid version")
    return value


def _require_smoke_command(value: object, context: str) -> str:
    if not isinstance(value, str) or not SMOKE_COMMAND_RE.fullmatch(value):
        raise RunnerPlanError(f"{context}: smoke command is missing or unsafe")
    return value


def base_tag(base: Mapping[str, Any], dockerfile: Path) -> str:
    """Derive the base image tag from the whole base recipe.

    The readable part names what a human cares about.  The six-hex suffix binds
    every input that can change the built bytes -- the Dockerfile itself plus
    each pinned catalog value -- so editing ``docker/base.Dockerfile`` or
    bumping the node digest cannot silently reuse an already-published, stale
    base.  Without it, "does this tag exist?" would answer *yes* for a recipe
    that no longer matches the checkout, and every backend would build FROM a
    stale base with nothing reporting a problem.
    """
    material = json.dumps(
        {
            "dockerfile": hashlib.sha256(dockerfile.read_bytes()).hexdigest(),
            "node_image": base["node_image"],
            "debian_snapshot": base["debian_snapshot"],
            "uv_version": base["uv_version"],
            "uv_wheels": base["uv_wheels"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(material.encode("utf-8")).hexdigest()[:6]
    snapshot = str(base["debian_snapshot"])[:8]
    tag = f"node22-uv{base['uv_version']}-snapshot{snapshot}-r{fingerprint}"
    if not TAG_RE.fullmatch(tag):
        raise RunnerPlanError(f"derived base image tag is unsafe: {tag!r}")
    return tag


def validate_catalog(catalog: dict[str, Any]) -> None:
    """Check the checked-in source pins a build actually depends on.

    Scope is deliberately narrow: every value a Dockerfile consumes must be
    present, come from the expected host over HTTPS, and carry a digest of the
    right shape.  Anything a build does not read is not validated here.
    """
    if catalog.get("schema_version") != 2:
        raise RunnerPlanError("unsupported runner catalog schema")
    if catalog.get("registry_prefix") != "ghcr.io/ke7/helix-evo-runner":
        raise RunnerPlanError("unexpected runner registry prefix")
    backends = catalog.get("backends")
    if not isinstance(backends, dict) or tuple(sorted(backends)) != tuple(
        sorted(BACKENDS)
    ):
        raise RunnerPlanError("runner catalog must contain exactly the five backends")

    base = catalog.get("base")
    if not isinstance(base, dict):
        raise RunnerPlanError("missing base configuration")
    if not re.fullmatch(
        r"node:[^@\s]+@sha256:[0-9a-f]{64}", str(base.get("node_image", ""))
    ):
        raise RunnerPlanError("base node image must be digest-pinned")
    if base.get("dockerfile") != "docker/base.Dockerfile":
        raise RunnerPlanError("unexpected base dockerfile")
    if not isinstance(base.get("uv_version"), str):
        raise RunnerPlanError("base uv version is required")
    if not re.fullmatch(r"\d{8}T\d{6}Z", str(base.get("debian_snapshot", ""))):
        raise RunnerPlanError("base Debian snapshot timestamp is invalid")
    uv_wheels = base.get("uv_wheels")
    if not isinstance(uv_wheels, dict) or tuple(sorted(uv_wheels)) != PLATFORMS:
        raise RunnerPlanError("base requires exact amd64/arm64 uv wheels")
    for platform in PLATFORMS:
        _require_source(
            uv_wheels[platform],
            "files.pythonhosted.org",
            SHA256_RE,
            "sha256",
            f"base uv/{platform}",
        )
    _require_smoke_command(base.get("smoke_command"), "base")

    for name in BACKENDS:
        item = backends[name]
        if not isinstance(item, dict):
            raise RunnerPlanError(f"{name}: expected a backend object")
        _require_smoke_command(item.get("smoke_command"), name)
        if item.get("dockerfile") != f"docker/{name}.Dockerfile":
            raise RunnerPlanError(f"{name}: unexpected dockerfile")

        if name == "cursor":
            # Cursor belongs to no package ecosystem: its version lives inside
            # a shell script and its archives must be hashed by hand.
            if item.get("installer") != "https://cursor.com/install":
                raise RunnerPlanError("cursor: unexpected installer")
            _require_version(item.get("version"), name)
            platforms = item.get("platforms")
            if not isinstance(platforms, dict) or tuple(sorted(platforms)) != PLATFORMS:
                raise RunnerPlanError("cursor: exact amd64/arm64 inputs required")
            for platform in PLATFORMS:
                _require_source(
                    platforms[platform],
                    "downloads.cursor.com",
                    SHA256_RE,
                    "sha256",
                    f"cursor/{platform}",
                )
        elif name in RESIDUAL_PARENTS:
            # Gemini's own tarball is pinned here rather than in the lockfile;
            # only its node-pty artifacts come from the lockfile.
            _require_version(item.get("version"), name)
            if item.get("package") != "@google/gemini-cli":
                raise RunnerPlanError(f"{name}: unexpected npm package")
            _require_source(item, "registry.npmjs.org", SHA512_RE, "sha512", name)
        elif set(item) - {"dockerfile", "smoke_command"}:
            # Everything else is owned by docker/package-lock.json. A stray
            # version or digest here would be a second source of truth that
            # nothing keeps in step with the lockfile.
            raise RunnerPlanError(
                f"{name}: version and digests belong to the lockfile, "
                f"not the catalog: {sorted(set(item) - {'dockerfile', 'smoke_command'})}"
            )


def validate_lockfile(lock: Mapping[str, Any]) -> None:
    """Check every artifact an image extracts is present and on-platform.

    npm records whatever the upstream optional-dependency closure happened to
    contain.  ``LOCKFILE_ARTIFACTS`` records what a human reviewed.  This is
    where the two must agree, and it is what stops a new native optional
    dependency from entering an image just because upstream declared one.
    """
    if lock.get("lockfileVersion") != 3:
        raise RunnerPlanError("unsupported npm lockfile version")
    packages = lock.get("packages")
    if not isinstance(packages, dict):
        raise RunnerPlanError("lockfile has no packages map")

    for name, group in sorted(LOCKFILE_ARTIFACTS.items()):
        for slot, package in sorted(group.items()):
            node = packages.get(f"node_modules/{package}")
            if not isinstance(node, dict):
                raise RunnerPlanError(
                    f"{name}/{slot}: {package} is not in the lockfile"
                )
            context = f"{name}/{slot}"
            _require_url(node.get("resolved"), "registry.npmjs.org")
            _lockfile_sha512(node, context)
            if slot in PLATFORMS:
                if node.get("cpu") != [LOCKFILE_CPU[slot]]:
                    raise RunnerPlanError(f"{context}: wrong cpu {node.get('cpu')!r}")
                if node.get("os") != ["linux"]:
                    raise RunnerPlanError(f"{context}: wrong os {node.get('os')!r}")
                # The base image is Debian, so a musl build would not run.
                if node.get("libc") not in (None, ["glibc"]):
                    raise RunnerPlanError(f"{context}: wrong libc {node.get('libc')!r}")

    codex = packages.get("node_modules/@openai/codex")
    if not isinstance(codex, dict):
        raise RunnerPlanError("codex is not in the lockfile")
    # HELIX requires a Codex that ships gpt-5.6-luna with an xhigh level.
    if _semver_tuple(str(codex.get("version"))) < CODEX_MINIMUM_VERSION:
        raise RunnerPlanError("codex must remain at least 0.145.0")


def _lockfile_sha512(node: Mapping[str, Any], context: str) -> str:
    """Return the hex sha512 behind a lockfile ``integrity`` field."""
    integrity = node.get("integrity")
    if not isinstance(integrity, str) or not integrity.startswith("sha512-"):
        raise RunnerPlanError(f"{context}: missing sha512 integrity")
    try:
        digest = base64.b64decode(
            integrity.removeprefix("sha512-"), validate=True
        ).hex()
    except ValueError as exc:
        raise RunnerPlanError(f"{context}: malformed sha512 integrity") from exc
    if not SHA512_RE.fullmatch(digest):
        raise RunnerPlanError(f"{context}: malformed sha512 digest")
    return digest


def _lockfile_source(
    lock: Mapping[str, Any], package: str, context: str
) -> dict[str, str]:
    """Return the ``{package, version, tarball, sha512}`` a Dockerfile needs."""
    node = lock["packages"][f"node_modules/{package}"]
    return {
        "package": package,
        # Aliased entries carry the aliased package's own version.
        "version": str(node["version"]),
        "tarball": _require_url(node["resolved"], "registry.npmjs.org"),
        "sha512": _lockfile_sha512(node, context),
    }


def parse_cursor_installer(payload: bytes) -> dict[str, str]:
    text = payload.decode("utf-8")
    versions = sorted(set(CURSOR_VERSION_RE.findall(text)))
    if len(versions) != 1:
        raise RunnerPlanError(
            f"cursor installer must name exactly one version; found {versions!r}"
        )
    version = versions[0]
    return {
        "version": version,
        "installer_sha256": hashlib.sha256(payload).hexdigest(),
        "amd64_tarball": (
            f"https://downloads.cursor.com/lab/{version}/linux/x64/"
            "agent-cli-package.tar.gz"
        ),
        "arm64_tarball": (
            f"https://downloads.cursor.com/lab/{version}/linux/arm64/"
            "agent-cli-package.tar.gz"
        ),
    }


def _is_retryable(exc: OSError) -> bool:
    """Report whether an idempotent GET may be safely repeated.

    Only transient conditions qualify.  These reads resolve upstream npm
    metadata and Cursor archives, where a 404 means the package or release
    genuinely is not there and 401/403 mean the request is malformed or
    blocked; retrying either would turn a decisive answer into a slower,
    noisier one without ever changing it.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return exc.code in RETRYABLE_HTTP_STATUS
    return isinstance(exc, (urllib.error.URLError, TimeoutError))


def _retry_get(
    operation: Callable[[], T],
    *,
    description: str,
    attempts: int = NETWORK_RETRY_ATTEMPTS,
    sleep: Sleep = time.sleep,
) -> T:
    """Run one idempotent GET with bounded exponential backoff.

    ``sleep`` is injected so unit tests observe the backoff schedule without
    spending wall-clock time.  A non-retryable error, and the final attempt's
    error, propagate unchanged so every existing fail-closed handler still
    sees the original exception.
    """
    if attempts < 1:
        raise RunnerPlanError(f"{description}: retry budget must be positive")
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except OSError as exc:
            if attempt == attempts or not _is_retryable(exc):
                raise
            sleep(float(2 ** (attempt - 1)))
    raise RunnerPlanError(  # pragma: no cover - loop always returns or raises
        f"{description}: exhausted {attempts} attempts"
    )


def _fetch(url: str, timeout: float = 30.0, *, sleep: Sleep = time.sleep) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "helix-runner-version-audit/1"},
    )

    def read() -> bytes:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload: bytes = response.read()
        return payload

    return _retry_get(read, description=f"fetch {url}", sleep=sleep)


def _fetch_sha256(
    url: str, timeout: float = 180.0, *, sleep: Sleep = time.sleep
) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "helix-runner-version-audit/1"},
    )

    def read() -> str:
        # The digest is rebuilt per attempt: a stream that died halfway must
        # never contribute its partial bytes to the recorded checksum.
        digest = hashlib.sha256()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    return _retry_get(read, description=f"hash {url}", sleep=sleep)


def resolve_cursor_checksums(
    tarballs: Mapping[str, str],
    version: str,
    catalog_item: Mapping[str, Any],
    *,
    fetch_sha256: Callable[[str], str] = _fetch_sha256,
) -> dict[str, str]:
    """Return the SHA-256 of each discovered Cursor platform archive.

    Cursor's release archives are addressed by version and are
    content-immutable, so streaming both multi-hundred-megabyte tarballs
    through a hash every single day proves nothing new once the version has
    not moved.  A reviewed catalog digest is reused only when the discovered
    version *and* the derived URL both still match it; any drift at all falls
    back to a full re-download and re-hash.
    """
    recorded = catalog_item.get("platforms")
    checksums: dict[str, str] = {}
    for platform in PLATFORMS:
        tarball = tarballs[platform]
        cached = recorded.get(platform) if isinstance(recorded, dict) else None
        if (
            version == catalog_item.get("version")
            and isinstance(cached, dict)
            and cached.get("tarball") == tarball
            and SHA256_RE.fullmatch(str(cached.get("sha256", "")))
        ):
            checksums[platform] = str(cached["sha256"])
        else:
            checksums[platform] = fetch_sha256(tarball)
    return checksums


def discover(
    catalog: dict[str, Any], lock: dict[str, Any], *, cursor_checksums: bool
) -> dict[str, Any]:
    """Resolve every pinned input into the flat map the workflow builds from.

    Everything except Cursor is already pinned in git -- npm packages by
    ``docker/package-lock.json``, Gemini's bundled tarball and the base runtime
    by the catalog -- so this reaches the network only for Cursor, whose
    version lives inside a shell script at ``https://cursor.com/install``.
    """
    validate_catalog(catalog)
    validate_lockfile(lock)
    resolved: dict[str, Any] = {
        "schema_version": 1,
        "base": catalog["base"],
        "backends": {},
    }
    for name in BACKENDS:
        item = catalog["backends"][name]
        common = {
            "kind": "codex"
            if name == "codex"
            else "cursor"
            if name == "cursor"
            else "npm",
            "dockerfile": item["dockerfile"],
            "smoke_command": item["smoke_command"],
        }
        if name == "cursor":
            resolved["backends"][name] = _resolve_cursor(item, common, cursor_checksums)
            continue

        group = LOCKFILE_ARTIFACTS[name]
        if name in RESIDUAL_PARENTS:
            parent = {
                "package": item["package"],
                "version": item["version"],
                "tarball": item["tarball"],
                "sha512": item["sha512"],
            }
        else:
            parent = _lockfile_source(lock, group["cli"], f"{name}/cli")
        entry: dict[str, Any] = {**parent, **common}

        if name == "codex":
            # Codex's platform binaries are aliases of @openai/codex itself,
            # so each one's lockfile version is `<version>-linux-<arch>`.
            entry["platforms"] = {
                platform: {
                    "package_version": (
                        source := _lockfile_source(
                            lock, group[platform], f"{name}/{platform}"
                        )
                    )["version"],
                    "tarball": source["tarball"],
                    "sha512": source["sha512"],
                }
                for platform in PLATFORMS
            }
        else:
            entry["artifacts"] = {
                slot: [_lockfile_source(lock, group[slot], f"{name}/{slot}")]
                for slot in ("shared", "amd64", "arm64")
                if slot in group
            }
        resolved["backends"][name] = entry
    return resolved


def _resolve_cursor(
    item: Mapping[str, Any], common: Mapping[str, str], cursor_checksums: bool
) -> dict[str, Any]:
    cursor: dict[str, Any] = dict(
        parse_cursor_installer(_fetch(str(item["installer"])))
    )
    tarballs = {
        platform: str(cursor.pop(f"{platform}_tarball")) for platform in PLATFORMS
    }
    cursor.update(
        {
            **common,
            "installer": item["installer"],
            "platforms": {
                platform: {"tarball": tarballs[platform]} for platform in PLATFORMS
            },
        }
    )
    if cursor_checksums:
        checksums = resolve_cursor_checksums(tarballs, str(cursor["version"]), item)
        for platform in PLATFORMS:
            cursor["platforms"][platform]["sha256"] = checksums[platform]
    return cursor


def build_arguments(item: Mapping[str, Any], base_image: str) -> list[str]:
    """Return the exact ``KEY=VALUE`` build arguments for one backend.

    This lived as a fifty-line ``jq`` case statement inside the workflow, where
    it could not be tested and where a value containing a newline would have
    injected an extra build argument into the heredoc. Every value is checked
    here instead.
    """
    if not re.fullmatch(
        r"ghcr\.io/[a-z0-9][a-z0-9._/-]*@sha256:[0-9a-f]{64}", base_image
    ):
        raise RunnerPlanError(f"base image must be digest-pinned: {base_image!r}")
    kind = item.get("kind")
    arguments = {
        "BASE_IMAGE": base_image,
        "CLI_VERSION": str(item["version"]),
    }
    if kind in {"npm", "codex"}:
        arguments["CLI_TARBALL"] = str(item["tarball"])
        arguments["CLI_SHA512"] = str(item["sha512"])
    if kind == "npm":
        artifacts = item["artifacts"]
        # Only Gemini declares a shared (architecture-independent) artifact, so
        # only Gemini's Dockerfile declares the matching ARGs. Emitting the pair
        # for claude/opencode would pass buildx two arguments their recipes
        # never read.
        if "shared" in artifacts:
            shared = artifacts["shared"][0]
            arguments["CLI_SHARED_TARBALL"] = str(shared["tarball"])
            arguments["CLI_SHARED_SHA512"] = str(shared["sha512"])
        for platform in PLATFORMS:
            entries = artifacts[platform]
            upper = platform.upper()
            arguments[f"CLI_{upper}_TARBALL"] = str(entries[0]["tarball"])
            arguments[f"CLI_{upper}_SHA512"] = str(entries[0]["sha512"])
    elif kind in {"codex", "cursor"}:
        digest_field = "sha512" if kind == "codex" else "sha256"
        for platform in PLATFORMS:
            source = item["platforms"][platform]
            upper = platform.upper()
            arguments[f"CLI_{upper}_TARBALL"] = str(source["tarball"])
            arguments[f"CLI_{upper}_{digest_field.upper()}"] = str(source[digest_field])
    else:
        raise RunnerPlanError(f"unknown backend kind {kind!r}")

    for key, value in arguments.items():
        if any(ord(character) < 0x20 for character in value):
            raise RunnerPlanError(f"{key}: build argument contains a control byte")
    return [f"{key}={value}" for key, value in arguments.items()]


def change_plan(
    resolved: Mapping[str, Any],
    published: Mapping[str, Any],
    *,
    force: bool = False,
    run_id: str = "",
) -> list[dict[str, str]]:
    """Decide which backends to build, and under which tag.

    The whole policy: an upstream release that is not in the registry yet gets
    built; anything already published is skipped.  Nightly is therefore the
    *check* cadence and an upstream release is the *publish* trigger, so a
    quiet night is a clean no-op.

    ``force`` rebuilds regardless -- the escape hatch for a recipe change that
    upstream did not trigger.  When the version tag already exists, a forced
    rebuild publishes ``<version>-r<run_id>`` instead of overwriting it: a tag
    someone has pinned must never silently change meaning.
    """
    builds: list[dict[str, str]] = []
    for name in BACKENDS:
        item = resolved["backends"][name]
        version = str(item["version"])
        if not VERSION_RE.fullmatch(version) or len(version) > 80:
            raise RunnerPlanError(f"{name}: unsafe image version {version!r}")
        exists = bool(published.get(name, False))
        if exists and not force:
            continue
        tag = version
        if exists:
            if not re.fullmatch(r"[0-9]+", run_id):
                raise RunnerPlanError(
                    f"{name}: forcing a rebuild of a published version needs "
                    "a numeric run id for the replacement tag"
                )
            tag = f"{version}-r{run_id}"
        if not TAG_RE.fullmatch(tag):
            raise RunnerPlanError(f"{name}: derived image tag is unsafe: {tag!r}")
        builds.append(
            {
                "name": name,
                "dockerfile": str(item["dockerfile"]),
                "version": version,
                "tag": tag,
            }
        )
    return builds


def verify_codex_catalog(payload: dict[str, Any]) -> None:
    models = payload.get("models")
    if not isinstance(models, list):
        raise RunnerPlanError("codex model catalog has no models array")
    matches = [model for model in models if model.get("slug") == "gpt-5.6-luna"]
    if len(matches) != 1:
        raise RunnerPlanError("expected exactly one gpt-5.6-luna catalog entry")
    levels = matches[0].get("supported_reasoning_levels")
    if not isinstance(levels, list):
        raise RunnerPlanError("Luna catalog entry has no reasoning levels")
    efforts = [entry.get("effort") for entry in levels]
    expected = ["low", "medium", "high", "xhigh", "max"]
    # The equality above already pins xhigh as the second-highest effort, which
    # is the property HELIX actually depends on.
    if efforts != expected:
        raise RunnerPlanError(
            f"unexpected Luna reasoning order: {efforts!r}; expected {expected!r}"
        )


def verify_platforms(payload: dict[str, Any]) -> None:
    manifests = payload.get("manifests")
    if not isinstance(manifests, list):
        raise RunnerPlanError("manifest list has no manifests")
    platforms: list[dict[str, Any]] = []
    for manifest in manifests:
        if not isinstance(manifest, dict):
            raise RunnerPlanError("manifest list entry is not an object")
        platform = manifest.get("platform", {})
        if not isinstance(platform, dict):
            raise RunnerPlanError("manifest platform is not an object")
        platforms.append(platform)
    runtime_platforms = sorted(
        (platform.get("os"), platform.get("architecture"))
        for platform in platforms
        if platform.get("os") != "unknown"
    )
    if runtime_platforms != [("linux", "amd64"), ("linux", "arm64")]:
        raise RunnerPlanError(
            f"expected exact linux/amd64+linux/arm64 parity, got {runtime_platforms!r}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--catalog", type=Path, required=True)
    validate.add_argument("--lockfile", type=Path, default=DEFAULT_LOCKFILE)
    discovery = subparsers.add_parser("discover")
    discovery.add_argument("--catalog", type=Path, required=True)
    discovery.add_argument("--lockfile", type=Path, default=DEFAULT_LOCKFILE)
    discovery.add_argument("--output", type=Path, required=True)
    discovery.add_argument("--cursor-checksums", action="store_true")
    plan = subparsers.add_parser("plan")
    plan.add_argument("--resolved", type=Path, required=True)
    plan.add_argument("--published", type=Path, required=True)
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--force", action="store_true")
    plan.add_argument("--run-id", default="")
    base = subparsers.add_parser("base-tag")
    base.add_argument("--catalog", type=Path, required=True)
    args_command = subparsers.add_parser("build-args")
    args_command.add_argument("--resolved", type=Path, required=True)
    args_command.add_argument("--backend", choices=BACKENDS, required=True)
    args_command.add_argument("--base-image", required=True)
    catalog = subparsers.add_parser("verify-codex-catalog")
    catalog.add_argument("--input", type=Path, required=True)
    platforms = subparsers.add_parser("verify-platforms")
    platforms.add_argument("--input", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            validate_catalog(load_json(args.catalog))
            validate_lockfile(load_json(args.lockfile))
        elif args.command == "discover":
            write_json(
                args.output,
                discover(
                    load_json(args.catalog),
                    load_json(args.lockfile),
                    cursor_checksums=args.cursor_checksums,
                ),
            )
        elif args.command == "plan":
            write_json(
                args.output,
                change_plan(
                    load_json(args.resolved),
                    load_json(args.published),
                    force=args.force,
                    run_id=args.run_id,
                ),
            )
        elif args.command == "build-args":
            resolved = load_json(args.resolved)
            for argument in build_arguments(
                resolved["backends"][args.backend], args.base_image
            ):
                print(argument)
        elif args.command == "base-tag":
            catalog = load_json(args.catalog)
            validate_catalog(catalog)
            # ``validate_catalog`` has already pinned ``base.dockerfile`` to the
            # literal ``docker/base.Dockerfile``, so the path is a constant
            # relative to the repository root and needs no traversal guard.
            root = args.catalog.resolve().parent.parent
            print(base_tag(catalog["base"], root / str(catalog["base"]["dockerfile"])))
        elif args.command == "verify-codex-catalog":
            verify_codex_catalog(load_json(args.input))
        elif args.command == "verify-platforms":
            verify_platforms(load_json(args.input))
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(args.command)
    except (
        OSError,
        json.JSONDecodeError,
        RunnerPlanError,
        # A catalog that is well-formed JSON but structurally wrong (a list
        # where an object belongs, a null where a string belongs) must still
        # exit 2 with the standard message rather than a raw traceback.
        AttributeError,
        TypeError,
    ) as exc:
        print(f"runner image gate failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
