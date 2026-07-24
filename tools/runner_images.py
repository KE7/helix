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
import signal
import shutil
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


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
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA512_RE = re.compile(r"^[0-9a-f]{128}$")
VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]*$")
TAG_RE = re.compile(r"^[0-9A-Za-z_][0-9A-Za-z_.-]{0,127}$")
CURSOR_VERSION_RE = re.compile(r"2026\.[0-9]{2}\.[0-9]{2}-[0-9a-f]+")


class RunnerPlanError(ValueError):
    """A release input is ambiguous, malformed, or violates promotion policy."""


class PromotionInterrupted(BaseException):
    """A termination signal interrupted a convenience-tag transaction."""

    def __init__(self, signum: int) -> None:
        self.signum = signum
        super().__init__(f"promotion interrupted by signal {signum}")


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


def validate_catalog(catalog: dict[str, Any]) -> None:
    if catalog.get("schema_version") != 1:
        raise RunnerPlanError("unsupported runner catalog schema")
    if catalog.get("registry_prefix") not in {
        None,
        "ghcr.io/ke7/helix-evo-runner",
    }:
        raise RunnerPlanError("unexpected runner registry prefix")
    if tuple(sorted(catalog.get("backends", {}))) != tuple(sorted(BACKENDS)):
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
        _require_url(wheel.get("url"), "files.pythonhosted.org")
        if not SHA256_RE.fullmatch(str(wheel.get("sha256", ""))):
            raise RunnerPlanError(f"base uv/{platform}: invalid sha256")
    if not TAG_RE.fullmatch(str(base.get("immutable_tag", ""))):
        raise RunnerPlanError("base immutable tag is invalid")
    if base["immutable_tag"] != base_immutable_tag(base):
        raise RunnerPlanError("base immutable tag is not bound to all recipe inputs")
    if not isinstance(base.get("smoke_command"), str):
        raise RunnerPlanError("base smoke command is required")

    for name in BACKENDS:
        item = catalog["backends"][name]
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
        if not isinstance(item.get("smoke_command"), str):
            raise RunnerPlanError(f"{name}: missing smoke command")
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


def _fetch(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "helix-runner-version-audit/1"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    if not isinstance(payload, bytes):
        raise RunnerPlanError(f"upstream returned non-bytes payload for {url!r}")
    return payload


def _fetch_sha256(url: str, timeout: float = 180.0) -> str:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "helix-runner-version-audit/1"},
    )
    digest = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while chunk := response.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
                for platform in PLATFORMS:
                    cursor["platforms"][platform]["sha256"] = _fetch_sha256(
                        cursor["platforms"][platform]["tarball"]
                    )
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
        identity.update(
            {
                "installer": item["installer"],
                "installer_sha256": item["installer_sha256"],
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
    resolved: dict[str, Any], published_tags: dict[str, str]
) -> dict[str, Any]:
    validate_catalog(resolved)
    changed: list[dict[str, Any]] = []
    builds: list[dict[str, str]] = []
    for name in BACKENDS:
        item = resolved["backends"][name]
        version = item["version"]
        tag = immutable_tag(item, str(resolved["base"]["immutable_tag"]))
        if published_tags.get(name) == tag:
            continue
        changed.append(
            {
                "name": name,
                "dockerfile": item["dockerfile"],
                "version": version,
                "immutable_tag": tag,
                "promotion_approved": (
                    version == item["promotion_guard_version"]
                    and tag == item["promotion_guard_immutable_tag"]
                ),
            }
        )
        for platform, runner in (
            ("amd64", "ubuntu-latest"),
            ("arm64", "ubuntu-24.04-arm"),
        ):
            builds.append(
                {
                    "name": name,
                    "dockerfile": item["dockerfile"],
                    "version": version,
                    "platform": f"linux/{platform}",
                    "arch": platform,
                    "runner": runner,
                }
            )
    return {"changed": changed, "builds": builds}


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
    try:
        with opener(token_request, timeout=30.0) as response:
            token_payload = json.loads(response.read())
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
    try:
        with opener(manifest_request, timeout=30.0) as response:
            manifest = response.read()
            digest = response.headers.get("Docker-Content-Digest", "")
    except urllib.error.HTTPError as exc:
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


CommandRunner = Callable[[list[str]], str]


def _registry_command(argv: list[str]) -> str:
    try:
        process = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
    except OSError as exc:
        raise RunnerPlanError("registry command could not start") from exc
    try:
        stdout, _stderr = process.communicate()
    except BaseException:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.communicate(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            process.wait()
        raise
    if process.returncode != 0:
        raise RunnerPlanError("registry command failed")
    return stdout


def _promotion_records(directory: Path) -> list[dict[str, Any]]:
    records = [load_json(path) for path in sorted(directory.glob("*.json"))]
    for record in records:
        backend = record.get("backend")
        if backend not in BACKENDS:
            raise RunnerPlanError("promotion record has an invalid backend")
        for field in ("previous_digest", "promoted_digest"):
            if not re.fullmatch(
                r"sha256:[0-9a-f]{64}", str(record.get(field, ""))
            ):
                raise RunnerPlanError(f"{backend}: malformed {field}")
    return records


def _move_tag(
    record: Mapping[str, Any],
    digest_field: str,
    *,
    run_command: CommandRunner,
) -> None:
    backend = str(record["backend"])
    digest = str(record[digest_field])
    image = f"ghcr.io/ke7/helix-evo-runner-{backend}"
    run_command(
        [
            "docker",
            "buildx",
            "imagetools",
            "create",
            "-t",
            f"{image}:latest",
            f"{image}@{digest}",
        ]
    )
    payload = json.loads(
        run_command(
            [
                "docker",
                "buildx",
                "imagetools",
                "inspect",
                f"{image}:latest",
                "--format",
                "{{json .Manifest}}",
            ]
        )
    )
    if not isinstance(payload, dict):
        raise RunnerPlanError(f"{backend}: registry returned a malformed manifest")
    if payload.get("digest") != digest:
        raise RunnerPlanError(
            f"{backend}: latest digest verification failed after tag move"
        )


def restore_latest_tags(
    records: Sequence[Mapping[str, Any]],
    *,
    run_command: CommandRunner = _registry_command,
) -> None:
    failures: list[str] = []
    for record in reversed(records):
        try:
            _move_tag(record, "previous_digest", run_command=run_command)
        except (OSError, json.JSONDecodeError, RunnerPlanError):
            failures.append(str(record.get("backend", "unknown")))
    if failures:
        raise RunnerPlanError(
            "compensating rollback failed for: " + ", ".join(failures)
        )


def promote_latest_tags(
    records: Sequence[Mapping[str, Any]],
    *,
    ledger_dir: Path,
    moved_file: Path,
    output_file: Path,
    run_command: CommandRunner = _registry_command,
    before_restore: Callable[[], None] | None = None,
) -> None:
    ledger_dir.mkdir(parents=True, exist_ok=True)
    moved_file.write_text("", encoding="utf-8")
    attempted: list[Mapping[str, Any]] = []
    try:
        for record in records:
            backend = str(record["backend"])
            attempted.append(record)
            with moved_file.open("a", encoding="utf-8") as stream:
                stream.write(f"{backend}\n")
            _move_tag(record, "promoted_digest", run_command=run_command)
            committed = dict(record)
            committed["state"] = "committed"
            write_json(ledger_dir / f"{backend}.json", committed)
        with output_file.open("a", encoding="utf-8") as stream:
            stream.write(f"promoted={'true' if records else 'false'}\n")
    except BaseException as exc:
        if before_restore is not None:
            before_restore()
        try:
            restore_latest_tags(attempted, run_command=run_command)
        except RunnerPlanError as restore_exc:
            raise RunnerPlanError(
                "promotion failed and compensation was incomplete"
            ) from restore_exc
        raise exc


def select_retry_artifacts(
    input_dir: Path,
    output_dir: Path,
    *,
    family: str,
    current_attempt: int,
    backend: str | None = None,
    required: Sequence[str] = (),
) -> dict[str, int]:
    if current_attempt < 1:
        raise RunnerPlanError("current workflow attempt must be positive")
    if family == "resolved-plan":
        matcher = re.compile(r"^resolved-runner-plan-([0-9]+)$")
        logical_group = None
        expected = {"plan"}
    elif family == "base-digests":
        matcher = re.compile(r"^digests-base-(amd64|arm64)-([0-9]+)$")
        logical_group = 1
        expected = set(PLATFORMS)
    elif family == "backend-digests":
        if backend not in BACKENDS:
            raise RunnerPlanError("backend artifact selection needs a valid backend")
        matcher = re.compile(
            rf"^digests-{re.escape(backend)}-(amd64|arm64)-([0-9]+)$"
        )
        logical_group = 1
        expected = set(PLATFORMS)
    elif family == "releases":
        matcher = re.compile(
            r"^release-(claude|codex|cursor|gemini|opencode)-([0-9]+)$"
        )
        logical_group = 1
        expected = set(required)
        if not expected or not expected.issubset(BACKENDS):
            raise RunnerPlanError("release selection needs valid required backends")
    else:
        raise RunnerPlanError(f"unknown retry artifact family: {family}")

    selected: dict[str, tuple[int, Path]] = {}
    for artifact in sorted(input_dir.iterdir()):
        if not artifact.is_dir() or artifact.is_symlink():
            raise RunnerPlanError(f"unexpected artifact entry: {artifact.name}")
        match = matcher.fullmatch(artifact.name)
        if match is None:
            raise RunnerPlanError(
                f"artifact name does not match {family}: {artifact.name}"
            )
        logical = "plan" if logical_group is None else match.group(logical_group)
        attempt = int(match.group(match.lastindex or 1))
        if attempt > current_attempt:
            raise RunnerPlanError("artifact comes from a future workflow attempt")
        previous = selected.get(logical)
        if previous is None or attempt > previous[0]:
            selected[logical] = (attempt, artifact)
        elif attempt == previous[0]:
            raise RunnerPlanError(
                f"duplicate {logical} artifacts for attempt {attempt}"
            )
    if set(selected) != expected:
        missing = sorted(expected - set(selected))
        extra = sorted(set(selected) - expected)
        raise RunnerPlanError(
            f"incomplete {family} artifacts; missing={missing}, extra={extra}"
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RunnerPlanError("retry artifact output directory is not empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    for logical in sorted(selected):
        _, artifact = selected[logical]
        for source in sorted(artifact.rglob("*")):
            if source.is_symlink():
                raise RunnerPlanError("artifact contains a symbolic link")
            if not source.is_file():
                continue
            target = output_dir / source.relative_to(artifact)
            if target.exists():
                raise RunnerPlanError(f"artifact file collision: {target.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    return {logical: selected[logical][0] for logical in sorted(selected)}


def _ignore_termination_signals() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)


def _run_promotion_cli(args: argparse.Namespace) -> int:
    records = _promotion_records(args.input_dir)
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }

    def interrupt(signum: int, _frame: object) -> None:
        _ignore_termination_signals()
        raise PromotionInterrupted(signum)

    for signum in previous_handlers:
        signal.signal(signum, interrupt)
    try:
        promote_latest_tags(
            records,
            ledger_dir=args.ledger_dir,
            moved_file=args.moved_file,
            output_file=args.github_output,
            before_restore=_ignore_termination_signals,
        )
    except PromotionInterrupted as exc:
        print(
            f"runner image gate failed: promotion interrupted by signal {exc.signum}",
            file=sys.stderr,
        )
        return 128 + exc.signum
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


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
    runtime_platforms = sorted(
        (
            manifest.get("platform", {}).get("os"),
            manifest.get("platform", {}).get("architecture"),
        )
        for manifest in manifests
        if manifest.get("platform", {}).get("os") != "unknown"
    )
    if runtime_platforms != [("linux", "amd64"), ("linux", "arm64")]:
        raise RunnerPlanError(
            f"expected exact linux/amd64+linux/arm64 parity, got {runtime_platforms!r}"
        )


def verify_build_evidence(
    directory: Path,
    *,
    backend: str,
    version: str,
) -> list[str]:
    """Validate the two native smoke records and return their digests.

    Artifact filenames are architecture-qualified so downloading multiple
    artifacts into one directory cannot overwrite evidence.  The merge job
    trusts only the JSON fields after this exact-set validation; it never
    derives a digest from an uploaded filename.
    """
    if backend not in BACKENDS:
        raise RunnerPlanError(f"unknown backend {backend!r}")
    records: dict[str, str] = {}
    paths = sorted(directory.glob("smoke-*.json"))
    if len(paths) != 2:
        raise RunnerPlanError(
            f"{backend}: expected exactly two smoke records, found {len(paths)}"
        )
    for path in paths:
        record = load_json(path)
        if record.get("backend") != backend or record.get("version") != version:
            raise RunnerPlanError(f"{path}: backend/version mismatch")
        platform = record.get("platform")
        digest = record.get("digest")
        if platform not in {"linux/amd64", "linux/arm64"}:
            raise RunnerPlanError(f"{path}: invalid platform {platform!r}")
        if not isinstance(digest, str) or not re.fullmatch(
            r"sha256:[0-9a-f]{64}", digest
        ):
            raise RunnerPlanError(f"{path}: invalid digest")
        if platform in records:
            raise RunnerPlanError(f"{backend}: duplicate evidence for {platform}")
        records[platform] = digest
    if set(records) != {"linux/amd64", "linux/arm64"}:
        raise RunnerPlanError(f"{backend}: incomplete platform evidence")
    return [records["linux/amd64"], records["linux/arm64"]]


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
    promote = subparsers.add_parser("promote-latest")
    promote.add_argument("--input-dir", type=Path, required=True)
    promote.add_argument("--ledger-dir", type=Path, required=True)
    promote.add_argument("--moved-file", type=Path, required=True)
    promote.add_argument("--github-output", type=Path, required=True)
    restore = subparsers.add_parser("restore-latest")
    restore.add_argument("--input-dir", type=Path, required=True)
    select_artifacts = subparsers.add_parser("select-retry-artifacts")
    select_artifacts.add_argument("--input-dir", type=Path, required=True)
    select_artifacts.add_argument("--output-dir", type=Path, required=True)
    select_artifacts.add_argument(
        "--family",
        choices=("resolved-plan", "base-digests", "backend-digests", "releases"),
        required=True,
    )
    select_artifacts.add_argument("--current-attempt", type=int, required=True)
    select_artifacts.add_argument("--backend", choices=BACKENDS)
    select_artifacts.add_argument("--required-json", default="[]")
    tag = subparsers.add_parser("image-tag")
    tag.add_argument("--catalog", type=Path, required=True)
    tag.add_argument("--backend", choices=BACKENDS, required=True)
    evidence = subparsers.add_parser("verify-build-evidence")
    evidence.add_argument("--input-dir", type=Path, required=True)
    evidence.add_argument("--backend", required=True)
    evidence.add_argument("--version", required=True)
    evidence.add_argument("--output", type=Path, required=True)
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
                change_plan(load_json(args.resolved), load_json(args.published)),
            )
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
        elif args.command == "promote-latest":
            return _run_promotion_cli(args)
        elif args.command == "restore-latest":
            _ignore_termination_signals()
            restore_latest_tags(_promotion_records(args.input_dir))
        elif args.command == "select-retry-artifacts":
            required = json.loads(args.required_json)
            if not isinstance(required, list) or not all(
                isinstance(item, str) for item in required
            ):
                raise RunnerPlanError("required artifact names must be a JSON array")
            print(
                json.dumps(
                    select_retry_artifacts(
                        args.input_dir,
                        args.output_dir,
                        family=args.family,
                        current_attempt=args.current_attempt,
                        backend=args.backend,
                        required=required,
                    ),
                    sort_keys=True,
                )
            )
        elif args.command == "image-tag":
            catalog = load_json(args.catalog)
            validate_catalog(catalog)
            print(
                immutable_tag(
                    catalog["backends"][args.backend],
                    str(catalog["base"]["immutable_tag"]),
                )
            )
        elif args.command == "verify-build-evidence":
            write_json(
                args.output,
                {
                    "backend": args.backend,
                    "version": args.version,
                    "digests": verify_build_evidence(
                        args.input_dir,
                        backend=args.backend,
                        version=args.version,
                    ),
                },
            )
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(args.command)
    except (OSError, json.JSONDecodeError, RunnerPlanError) as exc:
        print(f"runner image gate failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
