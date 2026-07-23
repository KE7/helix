"""Offline release-safety tests for mutation-agent runner images."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.runner_images import (
    RunnerPlanError,
    assert_immutable_collision,
    change_plan,
    parse_cursor_installer,
    parse_npm_metadata,
    validate_catalog,
    verify_codex_catalog,
    verify_platforms,
)


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "docker" / "runner-versions.json"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "publish-runners.yml"


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _npm_payload(
    *,
    version: str = "0.145.0",
    tarball: str = "https://registry.npmjs.org/@openai/codex/-/codex-0.145.0.tgz",
    integrity: str = (
        "sha512-/PSPSFujjjmiyVFvG2yu/grOFhsWdokTH8t2KGWhXSo/"
        "M5n/dIDsnbsnO82/7bLtIoDuzQf7ATBUMWqPWQINlQ=="
    ),
) -> bytes:
    return json.dumps(
        {
            "dist-tags": {"latest": version},
            "versions": {
                version: {
                    "dist": {
                        "tarball": tarball,
                        "integrity": integrity,
                    }
                }
            },
        }
    ).encode()


def test_checked_in_runner_catalog_is_complete_and_content_pinned() -> None:
    validate_catalog(_catalog())


def test_npm_discovery_parses_version_tarball_and_integrity() -> None:
    resolved = parse_npm_metadata("@openai/codex", _npm_payload())
    assert resolved["version"] == "0.145.0"
    assert resolved["sha512"].startswith("fcf48f485ba38e39")
    assert len(resolved["sha512"]) == 128


@pytest.mark.parametrize(
    ("tarball", "integrity"),
    [
        ("https://evil.invalid/codex.tgz", "sha512-YQ=="),
        (
            "https://registry.npmjs.org/@openai/codex/-/codex.tgz",
            "sha256-not-accepted",
        ),
    ],
)
def test_npm_discovery_fails_closed_on_untrusted_or_malformed_sources(
    tarball: str, integrity: str
) -> None:
    with pytest.raises(RunnerPlanError):
        parse_npm_metadata(
            "@openai/codex",
            _npm_payload(tarball=tarball, integrity=integrity),
        )


def test_npm_discovery_rejects_prerelease_latest() -> None:
    with pytest.raises(RunnerPlanError, match="prerelease"):
        parse_npm_metadata(
            "@openai/codex",
            _npm_payload(version="0.146.0-beta.1"),
        )


def test_cursor_installer_requires_one_unambiguous_version() -> None:
    installer = b"""
    VERSION=2026.07.20-8cc9c0b
    DOWNLOAD=https://downloads.cursor.com/lab/2026.07.20-8cc9c0b/linux/x64/a.tgz
    """
    resolved = parse_cursor_installer(installer)
    assert resolved["version"] == "2026.07.20-8cc9c0b"
    assert len(resolved["installer_sha256"]) == 64
    with pytest.raises(RunnerPlanError, match="exactly one"):
        parse_cursor_installer(installer + b"\nVERSION=2026.07.21-deadbee\n")


def test_change_plan_builds_only_changed_backends_on_both_native_arches() -> None:
    catalog = _catalog()
    published = {name: item["version"] for name, item in catalog["backends"].items()}
    published["codex"] = "0.130.0"
    plan = change_plan(catalog, published)
    assert plan["changed"] == [
        {
            "name": "codex",
            "dockerfile": "docker/codex.Dockerfile",
            "version": "0.145.0",
            "immutable_tag": "cli-0.145.0",
            "promotion_approved": False,
        }
    ]
    assert [(item["arch"], item["runner"]) for item in plan["builds"]] == [
        ("amd64", "ubuntu-latest"),
        ("arm64", "ubuntu-24.04-arm"),
    ]


def test_change_plan_is_empty_when_every_published_version_matches() -> None:
    catalog = _catalog()
    published = {name: item["version"] for name, item in catalog["backends"].items()}
    assert change_plan(catalog, published) == {"changed": [], "builds": []}


def test_immutable_tag_collision_is_idempotent_or_fails_hard() -> None:
    digest = "sha256:" + "a" * 64
    assert_immutable_collision(None, digest)
    assert_immutable_collision(digest, digest)
    with pytest.raises(RunnerPlanError, match="collision"):
        assert_immutable_collision("sha256:" + "b" * 64, digest)


def test_codex_catalog_requires_luna_and_exact_second_highest_xhigh() -> None:
    catalog = {
        "models": [
            {
                "slug": "gpt-5.6-luna",
                "supported_reasoning_levels": [
                    {"effort": effort}
                    for effort in ("low", "medium", "high", "xhigh", "max")
                ],
            }
        ]
    }
    verify_codex_catalog(catalog)
    catalog["models"][0]["supported_reasoning_levels"][-2]["effort"] = "high"
    with pytest.raises(RunnerPlanError, match="reasoning order"):
        verify_codex_catalog(catalog)


def test_manifest_parity_requires_exact_linux_amd64_and_arm64() -> None:
    payload = {
        "manifests": [
            {"platform": {"os": "linux", "architecture": "amd64"}},
            {"platform": {"os": "linux", "architecture": "arm64"}},
            {"platform": {"os": "unknown", "architecture": "unknown"}},
        ]
    }
    verify_platforms(payload)
    payload["manifests"][1]["platform"]["architecture"] = "amd64"
    with pytest.raises(RunnerPlanError, match="parity"):
        verify_platforms(payload)


def test_dockerfiles_do_not_install_a_floating_backend_cli() -> None:
    for backend in ("claude", "codex", "cursor", "gemini", "opencode"):
        text = (ROOT / "docker" / f"{backend}.Dockerfile").read_text()
        assert "@latest" not in text
        assert "curl https://cursor.com/install" not in text
        assert "CLI_VERSION=" in text
        checksum = "SHA256=" if backend == "cursor" else "SHA512="
        assert checksum in text


def test_publish_workflow_cannot_publish_from_a_pull_request() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = text.split("env:", 1)[0]
    assert "pull_request:" not in trigger_block
    assert "schedule:" in trigger_block
    assert "workflow_dispatch:" in trigger_block
    assert "cancel-in-progress: false" in text
    assert "packages: write" in text
    assert "verify-codex-catalog" in text
    assert "check-collision" in text
    assert "attest-build-provenance" in text
