"""Offline release-safety tests for mutation-agent runner images."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.runner_images import (
    RunnerPlanError,
    assert_immutable_collision,
    change_plan,
    inspect_ghcr_tag,
    immutable_tag,
    parse_cursor_installer,
    parse_npm_metadata,
    validate_catalog,
    validate_catalog_files,
    verify_build_evidence,
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
    validate_catalog_files(_catalog(), CATALOG_PATH)


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
            "https://registry.npmjs.org/codex.tgz\nCLI_SHA512=bad",
            (
                "sha512-/PSPSFujjjmiyVFvG2yu/grOFhsWdokTH8t2KGWhXSo/"
                "M5n/dIDsnbsnO82/7bLtIoDuzQf7ATBUMWqPWQINlQ=="
            ),
        ),
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


@pytest.mark.parametrize("version", ["0.146.0-beta.1", "2026.1.0-beta.1"])
def test_npm_discovery_rejects_prerelease_latest(version: str) -> None:
    with pytest.raises(RunnerPlanError, match="stable semantic version"):
        parse_npm_metadata(
            "@openai/codex",
            _npm_payload(version=version),
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
    published = {
        name: immutable_tag(item, catalog["base"]["immutable_tag"])
        for name, item in catalog["backends"].items()
    }
    published["codex"] = "cli-0.130.0-rold"
    plan = change_plan(catalog, published)
    assert plan["changed"] == [
        {
            "name": "codex",
            "dockerfile": "docker/codex.Dockerfile",
            "version": "0.145.0",
            "immutable_tag": immutable_tag(
                catalog["backends"]["codex"], catalog["base"]["immutable_tag"]
            ),
            "promotion_approved": False,
        }
    ]
    assert [(item["arch"], item["runner"]) for item in plan["builds"]] == [
        ("amd64", "ubuntu-latest"),
        ("arm64", "ubuntu-24.04-arm"),
    ]


def test_change_plan_is_empty_when_every_published_version_matches() -> None:
    catalog = _catalog()
    published = {
        name: immutable_tag(item, catalog["base"]["immutable_tag"])
        for name, item in catalog["backends"].items()
    }
    assert change_plan(catalog, published) == {"changed": [], "builds": []}


def test_same_version_content_drift_rebuilds_and_cannot_promote() -> None:
    catalog = _catalog()
    item = catalog["backends"]["gemini"]
    old_tag = immutable_tag(item, catalog["base"]["immutable_tag"])
    item["promotion_guard_version"] = item["version"]
    item["promotion_guard_immutable_tag"] = old_tag
    published = {
        name: immutable_tag(backend, catalog["base"]["immutable_tag"])
        for name, backend in catalog["backends"].items()
    }
    item["sha512"] = "a" * 128
    plan = change_plan(catalog, published)
    changed = next(entry for entry in plan["changed"] if entry["name"] == "gemini")
    assert changed["immutable_tag"] != old_tag
    assert changed["promotion_approved"] is False


def test_immutable_tag_collision_is_idempotent_or_fails_hard() -> None:
    digest = "sha256:" + "a" * 64
    assert_immutable_collision(None, digest)
    assert_immutable_collision(digest, digest)
    with pytest.raises(RunnerPlanError, match="collision"):
        assert_immutable_collision("sha256:" + "b" * 64, digest)


class _HTTPResponse:
    def __init__(self, payload: bytes, headers: dict[str, str] | None = None) -> None:
        self.payload = payload
        self.headers = headers or {}

    def __enter__(self) -> "_HTTPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.payload


def test_ghcr_tag_inspection_distinguishes_absence_from_registry_failure() -> None:
    manifest = b'{"schemaVersion":2}'
    digest = f"sha256:{hashlib.sha256(manifest).hexdigest()}"

    def present(
        request: urllib.request.Request, timeout: float
    ) -> _HTTPResponse:
        assert timeout == 30.0
        if request.full_url.startswith("https://ghcr.io/token?"):
            return _HTTPResponse(b'{"token":"registry-token"}')
        return _HTTPResponse(manifest, {"Docker-Content-Digest": digest})

    assert (
        inspect_ghcr_tag(
            "ghcr.io/ke7/helix-evo-runner-codex",
            "cli-0.145.0-rabc",
            actor="ci",
            token="secret",
            urlopen=present,
        )
        == digest
    )

    def mismatched(
        request: urllib.request.Request, timeout: float
    ) -> _HTTPResponse:
        if request.full_url.startswith("https://ghcr.io/token?"):
            return _HTTPResponse(b'{"token":"registry-token"}')
        return _HTTPResponse(
            manifest,
            {"Docker-Content-Digest": "sha256:" + "b" * 64},
        )

    with pytest.raises(RunnerPlanError, match="does not match"):
        inspect_ghcr_tag(
            "ghcr.io/ke7/helix-evo-runner-codex",
            "cli-0.145.0-rabc",
            actor="ci",
            token="secret",
            urlopen=mismatched,
        )

    def response_error(status: int) -> object:
        def fail(
            request: urllib.request.Request, timeout: float
        ) -> _HTTPResponse:
            if request.full_url.startswith("https://ghcr.io/token?"):
                return _HTTPResponse(b'{"token":"registry-token"}')
            raise urllib.error.HTTPError(
                request.full_url,
                status,
                "registry error",
                {},
                io.BytesIO(),
            )

        return fail

    assert (
        inspect_ghcr_tag(
            "ghcr.io/ke7/helix-evo-runner-codex",
            "cli-0.145.0-rabc",
            actor="ci",
            token="secret",
            urlopen=response_error(404),
        )
        is None
    )
    with pytest.raises(RunnerPlanError, match="HTTP 401"):
        inspect_ghcr_tag(
            "ghcr.io/ke7/helix-evo-runner-codex",
            "cli-0.145.0-rabc",
            actor="ci",
            token="secret",
            urlopen=response_error(401),
        )
    with pytest.raises(RunnerPlanError, match="HTTP 500"):
        inspect_ghcr_tag(
            "ghcr.io/ke7/helix-evo-runner-codex",
            "cli-0.145.0-rabc",
            actor="ci",
            token="secret",
            urlopen=response_error(500),
        )

    def token_404(
        request: urllib.request.Request, timeout: float
    ) -> _HTTPResponse:
        raise urllib.error.HTTPError(
            request.full_url,
            404,
            "token endpoint missing",
            {},
            io.BytesIO(),
        )

    with pytest.raises(RunnerPlanError, match="token exchange failed"):
        inspect_ghcr_tag(
            "ghcr.io/ke7/helix-evo-runner-codex",
            "cli-0.145.0-rabc",
            actor="ci",
            token="secret",
            urlopen=token_404,
        )


def test_immutable_version_tags_are_collision_resistant_and_base_bound() -> None:
    catalog = _catalog()
    base = catalog["base"]["immutable_tag"]
    item = copy.deepcopy(catalog["backends"]["codex"])
    item["version"] = "1.0+foo"
    first = immutable_tag(item, base)
    item["version"] = "1.0-foo"
    assert first != immutable_tag(item, base)
    item["version"] = "1.0+foo"
    assert first != immutable_tag(item, base + "-next")
    item["sha512"] = "a" * 128
    assert first != immutable_tag(item, base)
    assert re.fullmatch(r"[0-9A-Za-z_][0-9A-Za-z_.-]{0,127}", first)


def test_checked_in_dockerfile_hashes_fail_closed_on_recipe_drift() -> None:
    catalog = _catalog()
    catalog["backends"]["codex"]["dockerfile_sha256"] = "0" * 64
    with pytest.raises(RunnerPlanError, match="dockerfile sha256 mismatch"):
        validate_catalog_files(catalog, CATALOG_PATH)


def test_base_immutable_identity_changes_with_catalog_only_inputs() -> None:
    catalog = _catalog()
    catalog["base"]["uv_wheels"]["arm64"]["sha256"] = "a" * 64
    with pytest.raises(RunnerPlanError, match="all recipe inputs"):
        validate_catalog(catalog)


def test_catalog_rejects_unknown_backend_kind() -> None:
    catalog = _catalog()
    catalog["backends"]["claude"]["kind"] = "unexpected"
    with pytest.raises(RunnerPlanError, match="backend kind"):
        validate_catalog(catalog)


def test_catalog_rejects_backend_source_or_luna_contract_drift() -> None:
    catalog = _catalog()
    catalog["backends"]["claude"]["package"] = "lookalike-package"
    with pytest.raises(RunnerPlanError, match="npm package"):
        validate_catalog(catalog)
    catalog = _catalog()
    catalog["backends"]["codex"]["required_reasoning_effort"] = "high"
    with pytest.raises(RunnerPlanError, match="must be xhigh"):
        validate_catalog(catalog)


def test_current_promotion_guard_approves_only_exact_content_identity() -> None:
    catalog = _catalog()
    item = catalog["backends"]["claude"]
    item["promotion_guard_version"] = item["version"]
    item["promotion_guard_immutable_tag"] = immutable_tag(
        item, catalog["base"]["immutable_tag"]
    )
    plan = change_plan(catalog, {})
    changed = next(entry for entry in plan["changed"] if entry["name"] == "claude")
    assert changed["promotion_approved"] is True
    item["sha512"] = "a" * 128
    plan = change_plan(catalog, {})
    changed = next(entry for entry in plan["changed"] if entry["name"] == "claude")
    assert changed["promotion_approved"] is False


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


def test_build_evidence_requires_two_unique_native_smoke_records(
    tmp_path: Path,
) -> None:
    for arch in ("amd64", "arm64"):
        (tmp_path / f"smoke-{arch}.json").write_text(
            json.dumps(
                {
                    "backend": "codex",
                    "version": "0.145.0",
                    "platform": f"linux/{arch}",
                    "digest": "sha256:" + ("a" if arch == "amd64" else "b") * 64,
                }
            ),
            encoding="utf-8",
        )
    assert verify_build_evidence(tmp_path, backend="codex", version="0.145.0") == [
        "sha256:" + "a" * 64,
        "sha256:" + "b" * 64,
    ]
    (tmp_path / "smoke-arm64.json").unlink()
    with pytest.raises(RunnerPlanError, match="exactly two"):
        verify_build_evidence(tmp_path, backend="codex", version="0.145.0")


def test_dockerfiles_do_not_install_a_floating_backend_cli() -> None:
    for backend in ("claude", "codex", "cursor", "gemini", "opencode"):
        text = (ROOT / "docker" / f"{backend}.Dockerfile").read_text()
        assert "@latest" not in text
        assert "curl https://cursor.com/install" not in text
        assert "CLI_VERSION=" in text
        checksum = "SHA256=" if backend == "cursor" else "SHA512="
        assert checksum in text
        if backend in {"claude", "gemini", "opencode"}:
            assert "npm install" not in text
            assert "npm cache" not in text
            assert "TARGETARCH" in text


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
    assert "github.repository == 'KE7/helix'" in text
    assert "github.event.repository.default_branch == 'main'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert (
        "ATTESTATION_SIGNER_WORKFLOW: "
        "KE7/helix/.github/workflows/publish-runners.yml"
    ) in text
    assert "ATTESTATION_SIGNER_WORKFLOW: http" not in text
    assert '--signer-workflow "$ATTESTATION_SIGNER_WORKFLOW"' in text
    assert '--source-ref "$ATTESTATION_SOURCE_REF"' in text
    assert "--deny-self-hosted-runners" in text
    assert "cache-from:" not in text
    assert "cache-to:" not in text
    uses = re.findall(r"uses:\s+([^@\s]+)@([^\s#]+)", text)
    assert uses
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for _, revision in uses)
    assert "smoke-${{ matrix.arch }}.json" in text
    assert "verify-build-evidence" in text
    assert "promotion_guard_immutable_tag" in text
    assert "CLI_AMD64_FALLBACK_SHA512" in text
    assert "TARGET_DIGEST: ${{ inputs.target_digest }}" in text
    assert "BACKEND: ${{ inputs.backend }}" in text


def test_immutable_tag_collision_check_is_fail_closed_at_publish_time() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "EXISTING_DIGEST:" not in text
    assert 'echo "existing=${existing}"' not in text
    assert "if docker buildx imagetools inspect" not in text
    for start, end in (
        (
            "- name: Publish immutable base tag only after attestation succeeds",
            "  build-backends:",
        ),
        (
            "- name: Publish immutable tag and release record after attestation",
            "      - uses: actions/upload-artifact@",
        ),
    ):
        segment = text[text.index(start) : text.index(end, text.index(start))]
        inspect_at = segment.index("inspect-ghcr-tag")
        collision_at = segment.index("check-collision")
        publish_at = segment.index("docker buildx imagetools create")
        verify_at = segment.rindex("docker buildx imagetools inspect")
        assert inspect_at < collision_at < publish_at < verify_at
        assert 'if [[ "$existing" == "absent" ]]' in segment


def test_latest_moves_are_bracketed_by_retained_evidence_and_compensation() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    normalized = " ".join(text.split())
    prepared = text.index(
        "- name: Retain promotion rollback plan before moving latest tags"
    )
    moved = text.index(
        "- name: Move latest tags with compensating rollback on any failure"
    )
    committed = text.index("- name: Retain committed promotion ledger")
    compensated = text.index(
        "- name: Restore latest tags if committed ledger retention fails"
    )
    assert prepared < moved < committed < compensated
    assert (
        "failure() && steps.move.outputs.promoted == 'true' && "
        "steps.committed_ledger.outcome == 'failure'"
    ) in normalized

    rollback_prepared = text.index(
        "- name: Retain manual rollback plan before moving latest"
    )
    rollback_moved = text.index("- name: Move latest to the verified rollback digest")
    rollback_committed = text.index("- name: Retain committed manual rollback ledger")
    rollback_compensated = text.index(
        "- name: Restore latest if manual rollback ledger retention fails"
    )
    assert (
        rollback_prepared
        < rollback_moved
        < rollback_committed
        < rollback_compensated
    )
