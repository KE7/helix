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
import os
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
NPM_PACKAGES = {
    "claude": "@anthropic-ai/claude-code",
    "codex": "@openai/codex",
    "gemini": "@google/gemini-cli",
    "opencode": "opencode-ai",
}
NPM_ARTIFACT_PACKAGES: dict[str, dict[str, tuple[str, ...]]] = {
    "claude": {
        "amd64": ("@anthropic-ai/claude-code-linux-x64",),
        "arm64": ("@anthropic-ai/claude-code-linux-arm64",),
    },
    "gemini": {
        "shared": ("@lydell/node-pty",),
        "amd64": ("@lydell/node-pty-linux-x64",),
        "arm64": ("@lydell/node-pty-linux-arm64",),
    },
    "opencode": {
        "amd64": ("opencode-linux-x64", "opencode-linux-x64-baseline"),
        "arm64": ("opencode-linux-arm64",),
    },
}
DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
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


def _optional_dependencies(release: Mapping[str, Any], context: str) -> dict[str, str]:
    raw = release.get("optionalDependencies", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise RunnerPlanError(f"{context}: optional dependencies must be an object")
    dependencies: dict[str, str] = {}
    for package, version in sorted(raw.items()):
        if not isinstance(package, str) or not package or any(
            ord(character) <= 0x20 for character in package
        ):
            raise RunnerPlanError(f"{context}: invalid optional package name")
        if not isinstance(version, str):
            raise RunnerPlanError(f"{context}: invalid version for {package}")
        try:
            _semver_tuple(version)
        except RunnerPlanError:
            alias = re.fullmatch(
                r"npm:(@[0-9A-Za-z._-]+/[0-9A-Za-z._-]+|[0-9A-Za-z._-]+)"
                r"@([0-9A-Za-z][0-9A-Za-z._+-]*)",
                version,
            )
            if alias is None or not VERSION_RE.fullmatch(alias.group(2)):
                raise RunnerPlanError(
                    f"{context}: optional dependency {package} is not exact"
                ) from None
        dependencies[package] = version
    return dependencies


def _require_smoke_command(value: object, context: str) -> str:
    if not isinstance(value, str) or not SMOKE_COMMAND_RE.fullmatch(value):
        raise RunnerPlanError(f"{context}: smoke command is missing or unsafe")
    return value


def base_immutable_tag(base: Mapping[str, Any]) -> str:
    identity = {
        key: base[key]
        for key in (
            "dockerfile",
            "dockerfile_sha256",
            "node_image",
            "debian_snapshot",
            "uv_version",
            "uv_wheels",
        )
    }
    material = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    snapshot_date = str(base["debian_snapshot"])[:8]
    tag = (
        f"runtime-node22-uv{base['uv_version']}-"
        f"snapshot{snapshot_date}-r{fingerprint}"
    )
    if not TAG_RE.fullmatch(tag):
        raise RunnerPlanError(f"derived base image tag is unsafe: {tag!r}")
    return tag


def _repo_file(catalog_path: Path, entry: str) -> Path:
    """Resolve a catalog-declared Dockerfile inside the repository."""
    catalog = load_json(catalog_path)
    item = catalog["base"] if entry == "base" else catalog["backends"][entry]
    relative = Path(str(item["dockerfile"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise RunnerPlanError(f"{entry}: unsafe dockerfile path")
    root = catalog_path.resolve().parent.parent
    path = (root / relative).resolve()
    if root not in path.parents:
        raise RunnerPlanError(f"{entry}: dockerfile escapes the repository")
    return path


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
    if catalog.get("schema_version") != 1:
        raise RunnerPlanError("unsupported runner catalog schema")
    if catalog.get("registry_prefix") not in {
        None,
        "ghcr.io/ke7/helix-evo-runner",
    }:
        raise RunnerPlanError("unexpected runner registry prefix")
    backends = catalog.get("backends")
    if not isinstance(backends, dict) or tuple(sorted(backends)) != tuple(
        sorted(BACKENDS)
    ):
        raise RunnerPlanError("runner catalog must contain exactly the five backends")
    base = catalog.get("base")
    if not isinstance(base, dict):
        raise RunnerPlanError("missing base configuration")
    node_image = base.get("node_image")
    if not isinstance(node_image, str) or not re.fullmatch(
        r"node:[^@\s]+@sha256:[0-9a-f]{64}", node_image
    ):
        raise RunnerPlanError("base node image must be digest-pinned")
    if base.get("dockerfile") != "docker/base.Dockerfile":
        raise RunnerPlanError("unexpected base dockerfile")
    if not SHA256_RE.fullmatch(str(base.get("dockerfile_sha256", ""))):
        raise RunnerPlanError("base dockerfile sha256 is required")
    if not isinstance(base.get("uv_version"), str):
        raise RunnerPlanError("base uv version is required")
    if not re.fullmatch(r"\d{8}T\d{6}Z", str(base.get("debian_snapshot", ""))):
        raise RunnerPlanError("base Debian snapshot timestamp is invalid")
    uv_wheels = base.get("uv_wheels")
    if not isinstance(uv_wheels, dict) or tuple(sorted(uv_wheels)) != PLATFORMS:
        raise RunnerPlanError("base requires exact amd64/arm64 uv wheels")
    for platform in PLATFORMS:
        wheel = uv_wheels[platform]
        if not isinstance(wheel, dict):
            raise RunnerPlanError(f"base uv/{platform}: expected an object")
        _require_url(wheel.get("url"), "files.pythonhosted.org")
        if not SHA256_RE.fullmatch(str(wheel.get("sha256", ""))):
            raise RunnerPlanError(f"base uv/{platform}: invalid sha256")
    if not TAG_RE.fullmatch(str(base.get("immutable_tag", ""))):
        raise RunnerPlanError("base immutable tag is invalid")
    if base["immutable_tag"] != base_immutable_tag(base):
        raise RunnerPlanError("base immutable tag is not bound to all recipe inputs")
    _require_smoke_command(base.get("smoke_command"), "base")

    for name in BACKENDS:
        item = backends[name]
        if not isinstance(item, dict):
            raise RunnerPlanError(f"{name}: expected a backend object")
        version = item.get("version")
        if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
            raise RunnerPlanError(f"{name}: invalid version")
        promotion_guard = item.get("promotion_guard_version")
        if not isinstance(promotion_guard, str) or not VERSION_RE.fullmatch(
            promotion_guard
        ):
            raise RunnerPlanError(f"{name}: invalid promotion guard version")
        promotion_guard_tag = item.get("promotion_guard_immutable_tag")
        if promotion_guard_tag is not None and (
            not isinstance(promotion_guard_tag, str)
            or not TAG_RE.fullmatch(promotion_guard_tag)
        ):
            raise RunnerPlanError(f"{name}: invalid promotion guard immutable tag")
        _require_smoke_command(item.get("smoke_command"), name)
        if item.get("dockerfile") != f"docker/{name}.Dockerfile":
            raise RunnerPlanError(f"{name}: unexpected dockerfile")
        if not SHA256_RE.fullmatch(str(item.get("dockerfile_sha256", ""))):
            raise RunnerPlanError(f"{name}: missing dockerfile sha256")
        kind = item.get("kind")
        expected_kind = (
            "codex" if name == "codex" else "cursor" if name == "cursor" else "npm"
        )
        if kind != expected_kind:
            raise RunnerPlanError(
                f"{name}: expected backend kind {expected_kind!r}, got {kind!r}"
            )
        if kind in {"npm", "codex"}:
            if item.get("package") != NPM_PACKAGES[name]:
                raise RunnerPlanError(f"{name}: unexpected npm package")
            _require_url(item.get("tarball"), "registry.npmjs.org")
            if not SHA512_RE.fullmatch(str(item.get("sha512", ""))):
                raise RunnerPlanError(f"{name}: invalid sha512")
            optional_dependencies = item.get("optional_dependencies")
            if not isinstance(optional_dependencies, dict):
                raise RunnerPlanError(f"{name}: optional dependency map is required")
            _optional_dependencies(
                {"optionalDependencies": optional_dependencies}, f"{name} launcher"
            )
        if kind == "npm":
            artifacts = item.get("artifacts")
            expected_groups = NPM_ARTIFACT_PACKAGES[name]
            if not isinstance(artifacts, dict) or tuple(sorted(artifacts)) != tuple(
                sorted(expected_groups)
            ):
                raise RunnerPlanError(
                    f"{name}: exact content-pinned artifact groups are required"
                )
            for group, expected_packages in expected_groups.items():
                group_artifacts = artifacts[group]
                if not isinstance(group_artifacts, list) or tuple(
                    artifact.get("package")
                    for artifact in group_artifacts
                    if isinstance(artifact, dict)
                ) != expected_packages:
                    raise RunnerPlanError(
                        f"{name}/{group}: exact artifact packages are required"
                    )
                for artifact in group_artifacts:
                    if not isinstance(artifact, dict):
                        raise RunnerPlanError(
                            f"{name}/{group}: artifact must be an object"
                        )
                    version_value = artifact.get("version")
                    if not isinstance(version_value, str):
                        raise RunnerPlanError(
                            f"{name}/{group}: artifact version is required"
                        )
                    _semver_tuple(version_value)
                    _require_url(artifact.get("tarball"), "registry.npmjs.org")
                    if not SHA512_RE.fullmatch(str(artifact.get("sha512", ""))):
                        raise RunnerPlanError(
                            f"{name}/{group}: invalid artifact sha512"
                        )
                    artifact_optional = artifact.get("optional_dependencies")
                    if not isinstance(artifact_optional, dict):
                        raise RunnerPlanError(
                            f"{name}/{group}: artifact dependency map is required"
                        )
                    _optional_dependencies(
                        {"optionalDependencies": artifact_optional},
                        f"{name}/{group}/{artifact['package']}",
                    )
        if kind in {"codex", "cursor"}:
            platforms = item.get("platforms")
            if not isinstance(platforms, dict) or tuple(sorted(platforms)) != PLATFORMS:
                raise RunnerPlanError(f"{name}: exact amd64/arm64 inputs required")
        if kind == "codex":
            if _semver_tuple(version) < _semver_tuple(str(item["minimum_version"])):
                raise RunnerPlanError("codex is below its declared minimum version")
            if _semver_tuple(str(item["minimum_version"])) < (0, 145, 0):
                raise RunnerPlanError("codex minimum must remain at least 0.145.0")
            if item.get("model_catalog_command") != "codex debug models --bundled":
                raise RunnerPlanError("codex model catalog command changed")
            if item.get("required_model") != "gpt-5.6-luna":
                raise RunnerPlanError("codex required model must be gpt-5.6-luna")
            if item.get("required_reasoning_effort") != "xhigh":
                raise RunnerPlanError("codex required reasoning effort must be xhigh")
            for platform in PLATFORMS:
                source = item["platforms"][platform]
                if not isinstance(source, dict):
                    raise RunnerPlanError(f"codex/{platform}: expected an object")
                suffix = "linux-x64" if platform == "amd64" else "linux-arm64"
                if source.get("package_version") != f"{version}-{suffix}":
                    raise RunnerPlanError(
                        f"codex/{platform}: unexpected platform package version"
                    )
                _require_url(source.get("tarball"), "registry.npmjs.org")
                if not SHA512_RE.fullmatch(str(source.get("sha512", ""))):
                    raise RunnerPlanError(f"codex/{platform}: invalid sha512")
        elif kind == "cursor":
            if item.get("installer") != "https://cursor.com/install":
                raise RunnerPlanError("cursor: unexpected installer")
            _require_url(item.get("installer"), "cursor.com")
            if not SHA256_RE.fullmatch(str(item.get("installer_sha256", ""))):
                raise RunnerPlanError("cursor: invalid installer sha256")
            for platform in PLATFORMS:
                source = item["platforms"][platform]
                if not isinstance(source, dict):
                    raise RunnerPlanError(f"cursor/{platform}: expected an object")
                _require_url(source.get("tarball"), "downloads.cursor.com")
                if not SHA256_RE.fullmatch(str(source.get("sha256", ""))):
                    raise RunnerPlanError(f"cursor/{platform}: invalid sha256")


def validate_catalog_files(catalog: dict[str, Any], catalog_path: Path) -> None:
    """Prove that every declared Dockerfile digest matches the checkout."""
    validate_catalog(catalog)
    repository_root = catalog_path.resolve().parent.parent
    entries = [("base", catalog["base"])]
    entries.extend((name, catalog["backends"][name]) for name in BACKENDS)
    for name, item in entries:
        relative = Path(str(item["dockerfile"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise RunnerPlanError(f"{name}: unsafe dockerfile path")
        dockerfile = (repository_root / relative).resolve()
        if repository_root not in dockerfile.parents:
            raise RunnerPlanError(f"{name}: dockerfile escapes repository")
        actual = hashlib.sha256(dockerfile.read_bytes()).hexdigest()
        if actual != item["dockerfile_sha256"]:
            raise RunnerPlanError(
                f"{name}: dockerfile sha256 mismatch: expected "
                f"{item['dockerfile_sha256']}, got {actual}"
            )


def parse_npm_metadata(package: str, payload: bytes) -> dict[str, Any]:
    metadata = json.loads(payload)
    try:
        version = metadata["dist-tags"]["latest"]
        release = metadata["versions"][version]
        dist = release["dist"]
        tarball = _require_url(dist["tarball"], "registry.npmjs.org")
        integrity = dist["integrity"]
    except (KeyError, TypeError) as exc:
        raise RunnerPlanError(f"{package}: incomplete npm metadata") from exc
    if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
        raise RunnerPlanError(f"{package}: invalid latest version")
    _semver_tuple(version)
    if not isinstance(integrity, str) or not integrity.startswith("sha512-"):
        raise RunnerPlanError(f"{package}: missing sha512 integrity")
    try:
        sha512 = base64.b64decode(
            integrity.removeprefix("sha512-"), validate=True
        ).hex()
    except ValueError as exc:
        raise RunnerPlanError(f"{package}: malformed sha512 integrity") from exc
    if not SHA512_RE.fullmatch(sha512):
        raise RunnerPlanError(f"{package}: malformed sha512 digest")
    return {
        "package": package,
        "version": version,
        "tarball": tarball,
        "sha512": sha512,
        "optional_dependencies": _optional_dependencies(
            release, f"{package}@{version}"
        ),
    }


def parse_npm_release(package: str, version: str, payload: bytes) -> dict[str, Any]:
    metadata = json.loads(payload)
    try:
        actual_version = metadata["version"]
        dist = metadata["dist"]
        tarball = _require_url(dist["tarball"], "registry.npmjs.org")
        integrity = dist["integrity"]
    except (KeyError, TypeError) as exc:
        raise RunnerPlanError(
            f"{package}@{version}: incomplete npm release metadata"
        ) from exc
    if actual_version != version:
        raise RunnerPlanError(
            f"{package}: requested {version!r}, registry returned {actual_version!r}"
        )
    if not VERSION_RE.fullmatch(version):
        raise RunnerPlanError(f"{package}: unsafe release version {version!r}")
    if not isinstance(integrity, str) or not integrity.startswith("sha512-"):
        raise RunnerPlanError(f"{package}@{version}: missing sha512 integrity")
    try:
        sha512 = base64.b64decode(
            integrity.removeprefix("sha512-"), validate=True
        ).hex()
    except ValueError as exc:
        raise RunnerPlanError(
            f"{package}@{version}: malformed sha512 integrity"
        ) from exc
    if not SHA512_RE.fullmatch(sha512):
        raise RunnerPlanError(f"{package}@{version}: malformed sha512 digest")
    return {
        "package": package,
        "version": version,
        "tarball": tarball,
        "sha512": sha512,
        "optional_dependencies": _optional_dependencies(
            metadata, f"{package}@{version}"
        ),
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

    Only transient conditions qualify.  A 404 is a *meaningful answer* for
    ``inspect_ghcr_tag`` (the tag is absent) and 401/403 mean the credential is
    wrong; retrying either would turn a decisive result into a slower, noisier
    one without ever changing it.
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

    payload = _retry_get(read, description=f"fetch {url}", sleep=sleep)
    if not isinstance(payload, bytes):
        raise RunnerPlanError(f"upstream returned non-bytes payload for {url!r}")
    return payload


def _fetch_sha256(url: str, timeout: float = 180.0, *, sleep: Sleep = time.sleep) -> str:
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


def discover(catalog: dict[str, Any], *, cursor_checksums: bool) -> dict[str, Any]:
    validate_catalog(catalog)
    resolved: dict[str, Any] = {
        "schema_version": 1,
        "base": catalog["base"],
        "backends": {},
    }
    for name in BACKENDS:
        item = catalog["backends"][name]
        if item["kind"] in {"npm", "codex"}:
            package = item["package"]
            encoded = urllib.parse.quote(package, safe="")
            npm: dict[str, Any] = dict(
                parse_npm_metadata(
                    package,
                    _fetch(f"https://registry.npmjs.org/{encoded}"),
                )
            )
            npm.update(
                {
                    key: item[key]
                    for key in (
                        "kind",
                        "dockerfile",
                        "dockerfile_sha256",
                        "smoke_command",
                        "minimum_version",
                        "promotion_guard_version",
                        "promotion_guard_immutable_tag",
                        "model_catalog_command",
                        "required_model",
                        "required_reasoning_effort",
                    )
                    if key in item
                }
            )
            if item["kind"] == "codex":
                if _semver_tuple(npm["version"]) < _semver_tuple(
                    str(item["minimum_version"])
                ):
                    raise RunnerPlanError("upstream codex is below required minimum")
                npm["platforms"] = {}
                for platform, suffix in (
                    ("amd64", "linux-x64"),
                    ("arm64", "linux-arm64"),
                ):
                    platform_version = f"{npm['version']}-{suffix}"
                    platform_metadata = parse_npm_release(
                        package,
                        platform_version,
                        _fetch(
                            "https://registry.npmjs.org/"
                            f"{encoded}/{urllib.parse.quote(platform_version, safe='')}"
                        ),
                    )
                    npm["platforms"][platform] = {
                        "package_version": platform_version,
                        "tarball": platform_metadata["tarball"],
                        "sha512": platform_metadata["sha512"],
                    }
            else:
                artifact_groups = NPM_ARTIFACT_PACKAGES[name]
                npm["artifacts"] = {}
                transitive_optional: dict[str, str] = {}
                for group in ("shared", "amd64", "arm64"):
                    if group not in artifact_groups:
                        continue
                    npm["artifacts"][group] = []
                    for artifact_package in artifact_groups[group]:
                        version_source = (
                            transitive_optional
                            if name == "gemini" and group != "shared"
                            else npm["optional_dependencies"]
                        )
                        artifact_version = version_source.get(artifact_package)
                        if artifact_version is None:
                            raise RunnerPlanError(
                                f"{name}: upstream optional dependency closure "
                                f"does not contain {artifact_package}"
                            )
                        artifact_encoded = urllib.parse.quote(
                            artifact_package, safe=""
                        )
                        artifact_metadata = parse_npm_release(
                            artifact_package,
                            artifact_version,
                            _fetch(
                                "https://registry.npmjs.org/"
                                f"{artifact_encoded}/"
                                f"{urllib.parse.quote(artifact_version, safe='')}"
                            ),
                        )
                        npm["artifacts"][group].append(artifact_metadata)
                        if name == "gemini" and group == "shared":
                            transitive_optional.update(
                                artifact_metadata["optional_dependencies"]
                            )
            resolved["backends"][name] = npm
        else:
            cursor: dict[str, Any] = dict(
                parse_cursor_installer(_fetch(str(item["installer"])))
            )
            cursor.update(
                {
                    "kind": "cursor",
                    "dockerfile": item["dockerfile"],
                    "dockerfile_sha256": item["dockerfile_sha256"],
                    "installer": item["installer"],
                    "promotion_guard_version": item["promotion_guard_version"],
                    "promotion_guard_immutable_tag": item[
                        "promotion_guard_immutable_tag"
                    ],
                    "smoke_command": item["smoke_command"],
                    "platforms": {
                        platform: {
                            "tarball": cursor[f"{platform}_tarball"],
                        }
                        for platform in PLATFORMS
                    },
                }
            )
            if cursor_checksums:
                checksums = resolve_cursor_checksums(
                    {
                        platform: str(cursor["platforms"][platform]["tarball"])
                        for platform in PLATFORMS
                    },
                    str(cursor["version"]),
                    item,
                )
                for platform in PLATFORMS:
                    cursor["platforms"][platform]["sha256"] = checksums[platform]
            cursor.pop("amd64_tarball")
            cursor.pop("arm64_tarball")
            resolved["backends"][name] = cursor
    return resolved


def _backend_source_identity(item: Mapping[str, Any]) -> dict[str, Any]:
    identity: dict[str, Any] = {
        key: item[key]
        for key in ("kind", "dockerfile", "dockerfile_sha256", "version")
    }
    if item["kind"] in {"npm", "codex"}:
        identity.update(
            {
                "package": item["package"],
                "tarball": item["tarball"],
                "sha512": item["sha512"],
                "optional_dependencies": item["optional_dependencies"],
            }
        )
    if item["kind"] == "codex":
        identity["platforms"] = item["platforms"]
    elif item["kind"] == "npm":
        identity["artifacts"] = item["artifacts"]
    elif item["kind"] == "cursor":
        # ``installer_sha256`` is deliberately absent.  The installer is
        # fetched only to discover the version and the official artifact URLs;
        # it is never executed in an image.  The per-platform tarball URLs and
        # their SHA-256 digests already bind the exact shipped content, so
        # including the installer hash would mint a new immutable tag — and a
        # full two-architecture rebuild and republish of byte-identical CLI
        # content — for a comment-only edit to Cursor's install script.  The
        # hash is still recorded in the resolved manifest as evidence.
        identity.update(
            {
                "installer": item["installer"],
                "platforms": item["platforms"],
            }
        )
    return identity


def immutable_tag(item: Mapping[str, Any], base_tag: str) -> str:
    version = item.get("version")
    if not isinstance(version, str):
        raise RunnerPlanError("backend version is missing")
    if not VERSION_RE.fullmatch(version):
        raise RunnerPlanError(f"unsafe image version {version!r}")
    if len(version) > 80:
        raise RunnerPlanError("image version is too long")
    if not TAG_RE.fullmatch(base_tag):
        raise RunnerPlanError(f"unsafe base image tag {base_tag!r}")
    # Docker tags cannot contain "+". Bind the readable version to all source
    # inputs and the base recipe so ambiguity or recipe drift changes the
    # immutable identity.
    readable = re.sub(r"[^0-9A-Za-z_.-]", "-", version)
    material = json.dumps(
        {
            "base_tag": base_tag,
            "backend": _backend_source_identity(item),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    fingerprint = hashlib.sha256(material.encode("utf-8")).hexdigest()[:12]
    tag = f"cli-{readable}-r{fingerprint}"
    if not TAG_RE.fullmatch(tag):
        raise RunnerPlanError(f"derived image tag is unsafe: {tag!r}")
    return tag


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


def assert_immutable_collision(existing: str | None, candidate: str) -> None:
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", candidate):
        raise RunnerPlanError("candidate image digest is malformed")
    if existing is None:
        return
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", existing):
        raise RunnerPlanError("existing image digest is malformed")
    if existing != candidate:
        raise RunnerPlanError(
            f"immutable tag collision: existing {existing}, candidate {candidate}"
        )


def inspect_ghcr_tag(
    image: str,
    tag: str,
    *,
    actor: str,
    token: str,
    urlopen: Any | None = None,
    sleep: Sleep = time.sleep,
) -> str | None:
    """Return a GHCR tag digest, treating only an authenticated 404 as absent."""
    prefix = "ghcr.io/"
    if not image.startswith(prefix):
        raise RunnerPlanError("registry tag inspection only supports ghcr.io")
    repository = image.removeprefix(prefix)
    if not re.fullmatch(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*", repository):
        raise RunnerPlanError("GHCR repository is malformed")
    if not TAG_RE.fullmatch(tag):
        raise RunnerPlanError("GHCR tag is malformed")
    if not actor or not token:
        raise RunnerPlanError("GHCR inspection credentials are missing")
    opener = urlopen or urllib.request.urlopen
    credentials = base64.b64encode(f"{actor}:{token}".encode()).decode()
    scope = urllib.parse.quote(f"repository:{repository}:pull", safe="")
    token_request = urllib.request.Request(
        f"https://ghcr.io/token?service=ghcr.io&scope={scope}",
        headers={
            "Authorization": f"Basic {credentials}",
            "User-Agent": "helix-runner-publish/1",
        },
    )
    def exchange_token() -> Any:
        with opener(token_request, timeout=30.0) as response:
            return json.loads(response.read())

    try:
        token_payload = _retry_get(
            exchange_token, description="GHCR token exchange", sleep=sleep
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RunnerPlanError("GHCR token exchange failed") from exc
    if not isinstance(token_payload, dict):
        raise RunnerPlanError("GHCR token exchange returned malformed JSON")
    bearer = token_payload.get("token") or token_payload.get("access_token")
    if not isinstance(bearer, str) or not bearer:
        raise RunnerPlanError("GHCR token exchange returned no token")

    manifest_request = urllib.request.Request(
        f"https://ghcr.io/v2/{repository}/manifests/{tag}",
        headers={
            "Accept": (
                "application/vnd.oci.image.index.v1+json,"
                "application/vnd.docker.distribution.manifest.list.v2+json,"
                "application/vnd.oci.image.manifest.v1+json,"
                "application/vnd.docker.distribution.manifest.v2+json"
            ),
            "Authorization": f"Bearer {bearer}",
            "User-Agent": "helix-runner-publish/1",
        },
    )
    def read_manifest() -> tuple[bytes, str]:
        with opener(manifest_request, timeout=30.0) as response:
            body: bytes = response.read()
            header: str = response.headers.get("Docker-Content-Digest", "")
        return body, header

    try:
        manifest, digest = _retry_get(
            read_manifest, description="GHCR manifest inspection", sleep=sleep
        )
    except urllib.error.HTTPError as exc:
        # Never retried: an authenticated 404 is the answer "this tag does not
        # exist", which the whole publication gate depends on being decisive.
        if exc.code == 404:
            return None
        raise RunnerPlanError(
            f"GHCR manifest inspection failed with HTTP {exc.code}"
        ) from exc
    except OSError as exc:
        raise RunnerPlanError("GHCR manifest inspection failed") from exc
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise RunnerPlanError("GHCR returned a malformed manifest digest")
    actual = f"sha256:{hashlib.sha256(manifest).hexdigest()}"
    if actual != digest:
        raise RunnerPlanError("GHCR manifest body does not match its digest header")
    return digest


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
    if efforts != expected:
        raise RunnerPlanError(
            f"unexpected Luna reasoning order: {efforts!r}; expected {expected!r}"
        )
    if efforts[-2] != "xhigh":
        raise RunnerPlanError("xhigh is not Luna's second-highest reasoning effort")


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
    discovery = subparsers.add_parser("discover")
    discovery.add_argument("--catalog", type=Path, required=True)
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
    catalog = subparsers.add_parser("verify-codex-catalog")
    catalog.add_argument("--input", type=Path, required=True)
    platforms = subparsers.add_parser("verify-platforms")
    platforms.add_argument("--input", type=Path, required=True)
    collision = subparsers.add_parser("check-collision")
    collision.add_argument("--existing")
    collision.add_argument("--candidate", required=True)
    inspect_tag = subparsers.add_parser("inspect-ghcr-tag")
    inspect_tag.add_argument("--image", required=True)
    inspect_tag.add_argument("--tag", required=True)
    tag = subparsers.add_parser("image-tag")
    tag.add_argument("--catalog", type=Path, required=True)
    tag.add_argument("--backend", choices=BACKENDS, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            catalog = load_json(args.catalog)
            validate_catalog_files(catalog, args.catalog)
        elif args.command == "discover":
            write_json(
                args.output,
                discover(
                    load_json(args.catalog),
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
        elif args.command == "base-tag":
            catalog = load_json(args.catalog)
            validate_catalog(catalog)
            print(base_tag(catalog["base"], _repo_file(args.catalog, "base")))
        elif args.command == "verify-codex-catalog":
            verify_codex_catalog(load_json(args.input))
        elif args.command == "verify-platforms":
            verify_platforms(load_json(args.input))
        elif args.command == "check-collision":
            assert_immutable_collision(args.existing or None, args.candidate)
        elif args.command == "inspect-ghcr-tag":
            digest = inspect_ghcr_tag(
                args.image,
                args.tag,
                actor=os.environ.get("GITHUB_ACTOR", ""),
                token=os.environ.get("GH_TOKEN", ""),
            )
            print(digest or "absent")
        elif args.command == "image-tag":
            catalog = load_json(args.catalog)
            validate_catalog(catalog)
            print(
                immutable_tag(
                    catalog["backends"][args.backend],
                    str(catalog["base"]["immutable_tag"]),
                )
            )
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
