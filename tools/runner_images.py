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
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


BACKENDS = ("claude", "codex", "cursor", "gemini", "opencode")
PLATFORMS = ("amd64", "arm64")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHA512_RE = re.compile(r"^[0-9a-f]{128}$")
VERSION_RE = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]*$")
CURSOR_VERSION_RE = re.compile(r"2026\.[0-9]{2}\.[0-9]{2}-[0-9a-f]+")


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
    parsed = urllib.parse.urlparse(value)
    if parsed.scheme != "https" or parsed.hostname != host:
        raise RunnerPlanError(f"untrusted upstream URL: {value!r}")
    return value


def _semver_tuple(value: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", value)
    if not match:
        raise RunnerPlanError(f"expected stable semantic version, got {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def validate_catalog(catalog: dict[str, Any]) -> None:
    if catalog.get("schema_version") != 1:
        raise RunnerPlanError("unsupported runner catalog schema")
    if tuple(sorted(catalog.get("backends", {}))) != tuple(sorted(BACKENDS)):
        raise RunnerPlanError("runner catalog must contain exactly the five backends")
    base = catalog.get("base")
    if not isinstance(base, dict):
        raise RunnerPlanError("missing base configuration")
    node_image = base.get("node_image")
    if not isinstance(node_image, str) or "@sha256:" not in node_image:
        raise RunnerPlanError("base node image must be digest-pinned")

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
        if not isinstance(item.get("smoke_command"), str):
            raise RunnerPlanError(f"{name}: missing smoke command")
        kind = item.get("kind")
        if kind in {"npm", "codex"}:
            _require_url(item.get("tarball"), "registry.npmjs.org")
            if not SHA512_RE.fullmatch(str(item.get("sha512", ""))):
                raise RunnerPlanError(f"{name}: invalid sha512")
        if kind in {"codex", "cursor"}:
            platforms = item.get("platforms")
            if not isinstance(platforms, dict) or tuple(sorted(platforms)) != PLATFORMS:
                raise RunnerPlanError(f"{name}: exact amd64/arm64 inputs required")
        if kind == "codex":
            if _semver_tuple(version) < _semver_tuple(str(item["minimum_version"])):
                raise RunnerPlanError("codex is below its declared minimum version")
            for platform in PLATFORMS:
                source = item["platforms"][platform]
                _require_url(source.get("tarball"), "registry.npmjs.org")
                if not SHA512_RE.fullmatch(str(source.get("sha512", ""))):
                    raise RunnerPlanError(f"codex/{platform}: invalid sha512")
        elif kind == "cursor":
            _require_url(item.get("installer"), "cursor.com")
            if not SHA256_RE.fullmatch(str(item.get("installer_sha256", ""))):
                raise RunnerPlanError("cursor: invalid installer sha256")
            for platform in PLATFORMS:
                source = item["platforms"][platform]
                _require_url(source.get("tarball"), "downloads.cursor.com")
                if not SHA256_RE.fullmatch(str(source.get("sha256", ""))):
                    raise RunnerPlanError(f"cursor/{platform}: invalid sha256")


def parse_npm_metadata(package: str, payload: bytes) -> dict[str, str]:
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
    if "-" in version and not version.startswith("2026."):
        raise RunnerPlanError(f"{package}: prerelease latest is not publishable")
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
    }


def parse_npm_release(package: str, version: str, payload: bytes) -> dict[str, str]:
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
        return response.read()


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
            npm = parse_npm_metadata(
                package,
                _fetch(f"https://registry.npmjs.org/{encoded}"),
            )
            npm.update(
                {
                    key: item[key]
                    for key in (
                        "kind",
                        "dockerfile",
                        "smoke_command",
                        "minimum_version",
                        "promotion_guard_version",
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
            resolved["backends"][name] = npm
        else:
            cursor = parse_cursor_installer(_fetch(str(item["installer"])))
            cursor.update(
                {
                    "kind": "cursor",
                    "dockerfile": item["dockerfile"],
                    "installer": item["installer"],
                    "promotion_guard_version": item["promotion_guard_version"],
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
            resolved["backends"][name] = cursor
    return resolved


def immutable_tag(version: str) -> str:
    if not VERSION_RE.fullmatch(version):
        raise RunnerPlanError(f"unsafe image version {version!r}")
    return f"cli-{version.replace('+', '_')}"


def change_plan(
    resolved: dict[str, Any], published_versions: dict[str, str]
) -> dict[str, Any]:
    validate_catalog(resolved)
    changed: list[dict[str, Any]] = []
    builds: list[dict[str, str]] = []
    for name in BACKENDS:
        item = resolved["backends"][name]
        version = item["version"]
        if published_versions.get(name) == version:
            continue
        changed.append(
            {
                "name": name,
                "dockerfile": item["dockerfile"],
                "version": version,
                "immutable_tag": immutable_tag(version),
                "promotion_approved": version == item["promotion_guard_version"],
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            validate_catalog(load_json(args.catalog))
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
        else:  # pragma: no cover - argparse enforces the command set
            raise AssertionError(args.command)
    except (OSError, json.JSONDecodeError, RunnerPlanError) as exc:
        print(f"runner image gate failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
