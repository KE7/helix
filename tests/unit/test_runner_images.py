"""Offline release-safety tests for mutation-agent runner images."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import signal
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.runner_images import (
    PromotionInterrupted,
    RunnerPlanError,
    _promotion_records,
    assert_immutable_collision,
    change_plan,
    inspect_ghcr_tag,
    immutable_tag,
    main as runner_images_main,
    parse_cursor_installer,
    parse_npm_metadata,
    promote_latest_tags,
    restore_latest_tags,
    select_retry_artifacts,
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


def test_shipped_catalog_deliberately_approves_no_promotion() -> None:
    """The current posture is intent, not accident.

    Every backend ships an unset ``promotion_guard_immutable_tag`` and a guard
    version behind the pinned version, so ``latest`` never moves automatically.
    A maintainer approves a promotion by updating both guard fields.  This test
    fails the moment that posture changes, forcing the change to be reviewed.
    """
    catalog = _catalog()
    plan = change_plan(catalog, {})
    assert [entry["name"] for entry in plan["changed"]] == sorted(
        catalog["backends"]
    )
    assert all(entry["promotion_approved"] is False for entry in plan["changed"])
    for name, item in catalog["backends"].items():
        assert item["promotion_guard_immutable_tag"] is None, name
        assert item["promotion_guard_version"] != item["version"], name


def test_workflow_makes_a_stalled_promotion_gate_loud_and_machine_readable() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "promotion_stalled: ${{ steps.plan.outputs.promotion_stalled }}" in text
    assert 'echo "promotion_stalled=${promotion_stalled}" >> "$GITHUB_OUTPUT"' in text
    assert "::warning title=Runner promotion stalled::" in text
    assert "### Convenience-tag promotion is stalled" in text
    assert "/tmp/promotion-stall.json" in text
    # The stall report must be retained with the rest of the plan evidence.
    retained = text[
        text.index("- name: Retain the exact resolved build plan") :
    ].split("retention-days:", 1)[0]
    assert "/tmp/promotion-stall.json" in retained
    # Reporting must not weaken the gate: promotion still requires an exact
    # guard match computed by the planner and re-checked in the promote job.
    assert 'select(.promotion_approved == true)' in text


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


def test_failure_notifier_covers_cancellation_and_dedupes_beyond_one_page() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    notifier = text[text.index("  notify-failure:") :]
    # A cancelled run is exactly the hard-kill-mid-transaction case.
    assert "(failure() || cancelled())" in notifier
    assert "RUN_OUTCOME: ${{ cancelled() && 'cancelled' || 'failed' }}" in notifier
    assert "was cancelled" in notifier
    assert "rollback-before-" in notifier
    # Dedupe must not silently stop at the first page of open issues.
    assert "per_page: 100" in notifier
    assert "listForRepo({" not in notifier
    assert "github.paginate(github.rest.issues.listForRepo" in notifier
    assert 'const label = "runner-image-refresh";' in notifier
    assert "labels: label" in notifier
    assert "labels: [label]" in notifier


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
    assert text.count("exec python tools/runner_images.py promote-latest") == 2
    assert text.count("python tools/runner_images.py restore-latest") == 2

    # Both preflights must decide "no previous latest" from the authenticated
    # 404-only probe, never from a failed `imagetools inspect`.
    assert text.count('--image "$image" --tag latest') == 2
    assert text.count('if [[ "$latest_state" == "absent" ]]; then') == 2
    assert text.count("bootstrap=true") == 2
    assert text.count("--argjson bootstrap") == 2
    assert text.count('[[ "$previous" == "$latest_state" ]]') == 2

    # The registry prefix has exactly one source of truth: promotion records
    # carry the image they target, so a prefix change cannot make the
    # compensating rollback address a different repository.
    assert text.count('--arg image "$image"') == 2
    assert text.count("backend:$backend,image:$image,") == 2
    assert 'f"ghcr.io/ke7/helix-evo-runner-{backend}"' not in (
        (ROOT / "tools" / "runner_images.py").read_text(encoding="utf-8")
    )


def test_v4_artifact_names_are_unique_for_same_run_retries() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    attempt = "${{ github.run_attempt }}"
    upload_blocks = text.split("uses: actions/upload-artifact@")[1:]
    upload_names = [
        re.search(r"\n\s+name:\s+([^\n]+)", block).group(1)
        for block in upload_blocks
    ]
    assert upload_names
    assert all(attempt in name for name in upload_names)
    templates = {
        f"resolved-runner-plan-{attempt}",
        f"digests-base-${{{{ matrix.arch }}}}-{attempt}",
        f"digests-${{{{ matrix.name }}}}-${{{{ matrix.arch }}}}-{attempt}",
        f"release-${{{{ matrix.name }}}}-{attempt}",
    }
    assert all(f"name: {template}" in text for template in templates)
    first_attempt = {template.replace(attempt, "1") for template in templates}
    second_attempt = {template.replace(attempt, "2") for template in templates}
    assert first_attempt.isdisjoint(second_attempt)
    assert "pattern: resolved-runner-plan-*" in text
    assert "pattern: digests-base-*" in text
    assert "pattern: digests-${{ matrix.name }}-*" in text
    assert "pattern: release-*" in text
    assert text.count("select-retry-artifacts") == 5
    assert text.count("merge-multiple: false") == 5


def test_selective_retry_uses_latest_available_producer_per_logical_input(
    tmp_path: Path,
) -> None:
    def artifact(root: Path, name: str, filename: str, value: str) -> None:
        directory = root / name
        directory.mkdir(parents=True)
        (directory / filename).write_text(value, encoding="utf-8")

    plan_input = tmp_path / "plan-input"
    artifact(plan_input, "resolved-runner-plan-1", "plan.json", "attempt-1")
    assert select_retry_artifacts(
        plan_input,
        tmp_path / "plan-output",
        family="resolved-plan",
        current_attempt=2,
    ) == {"plan": 1}
    assert (tmp_path / "plan-output" / "plan.json").read_text() == "attempt-1"

    base_input = tmp_path / "base-input"
    artifact(base_input, "digests-base-amd64-1", "amd64", "amd64-attempt-1")
    artifact(base_input, "digests-base-arm64-1", "arm64-old", "old")
    artifact(base_input, "digests-base-arm64-2", "arm64", "arm64-attempt-2")
    assert select_retry_artifacts(
        base_input,
        tmp_path / "base-output",
        family="base-digests",
        current_attempt=2,
    ) == {"amd64": 1, "arm64": 2}
    assert (tmp_path / "base-output" / "amd64").read_text() == "amd64-attempt-1"
    assert (tmp_path / "base-output" / "arm64").read_text() == "arm64-attempt-2"
    assert not (tmp_path / "base-output" / "arm64-old").exists()

    backend_input = tmp_path / "backend-input"
    artifact(
        backend_input,
        "digests-codex-amd64-1",
        "smoke-amd64.json",
        "amd64-attempt-1",
    )
    artifact(
        backend_input,
        "digests-codex-arm64-2",
        "smoke-arm64.json",
        "arm64-attempt-2",
    )
    assert select_retry_artifacts(
        backend_input,
        tmp_path / "backend-output",
        family="backend-digests",
        backend="codex",
        current_attempt=2,
    ) == {"amd64": 1, "arm64": 2}

    release_input = tmp_path / "release-input"
    artifact(release_input, "release-codex-1", "codex.json", "codex-old")
    artifact(release_input, "release-codex-2", "codex.json", "codex-new")
    artifact(release_input, "release-claude-1", "claude.json", "claude-old")
    assert select_retry_artifacts(
        release_input,
        tmp_path / "release-output",
        family="releases",
        current_attempt=2,
        required=("codex", "claude"),
    ) == {"claude": 1, "codex": 2}
    assert (tmp_path / "release-output" / "codex.json").read_text() == "codex-new"
    assert (
        tmp_path / "release-output" / "claude.json"
    ).read_text() == "claude-old"


def test_rerun_all_replaces_every_logical_input_with_current_attempt(
    tmp_path: Path,
) -> None:
    source = tmp_path / "all-input"
    for arch in ("amd64", "arm64"):
        for attempt in (1, 2):
            directory = source / f"digests-base-{arch}-{attempt}"
            directory.mkdir(parents=True)
            (directory / arch).write_text(
                f"{arch}-attempt-{attempt}",
                encoding="utf-8",
            )
    assert select_retry_artifacts(
        source,
        tmp_path / "all-output",
        family="base-digests",
        current_attempt=2,
    ) == {"amd64": 2, "arm64": 2}
    assert (tmp_path / "all-output" / "amd64").read_text() == "amd64-attempt-2"
    assert (tmp_path / "all-output" / "arm64").read_text() == "arm64-attempt-2"


def test_retry_artifact_selection_rejects_ambiguous_or_unsafe_inputs(
    tmp_path: Path,
) -> None:
    future = tmp_path / "future"
    (future / "resolved-runner-plan-3").mkdir(parents=True)
    with pytest.raises(RunnerPlanError, match="future"):
        select_retry_artifacts(
            future,
            tmp_path / "future-output",
            family="resolved-plan",
            current_attempt=2,
        )

    duplicate = tmp_path / "duplicate"
    for suffix in ("1", "01"):
        directory = duplicate / f"resolved-runner-plan-{suffix}"
        directory.mkdir(parents=True)
        (directory / suffix).write_text(suffix, encoding="utf-8")
    with pytest.raises(RunnerPlanError, match="duplicate"):
        select_retry_artifacts(
            duplicate,
            tmp_path / "duplicate-output",
            family="resolved-plan",
            current_attempt=2,
        )

    missing = tmp_path / "missing"
    (missing / "digests-base-amd64-1").mkdir(parents=True)
    with pytest.raises(RunnerPlanError, match="missing"):
        select_retry_artifacts(
            missing,
            tmp_path / "missing-output",
            family="base-digests",
            current_attempt=1,
        )

    unexpected = tmp_path / "unexpected"
    unexpected.mkdir()
    (unexpected / "not-an-artifact").write_text("bad", encoding="utf-8")
    with pytest.raises(RunnerPlanError, match="unexpected"):
        select_retry_artifacts(
            unexpected,
            tmp_path / "unexpected-output",
            family="resolved-plan",
            current_attempt=1,
        )

    linked = tmp_path / "linked"
    linked.mkdir()
    real = tmp_path / "symlink-target"
    real.mkdir()
    (linked / "resolved-runner-plan-1").symlink_to(real, target_is_directory=True)
    with pytest.raises(RunnerPlanError, match="unexpected"):
        select_retry_artifacts(
            linked,
            tmp_path / "linked-output",
            family="resolved-plan",
            current_attempt=1,
        )

    real_input = tmp_path / "real-input"
    real_artifact = real_input / "resolved-runner-plan-1"
    real_artifact.mkdir(parents=True)
    (real_artifact / "plan.json").write_text("immutable", encoding="utf-8")
    linked_input = tmp_path / "linked-input"
    linked_input.symlink_to(real_input, target_is_directory=True)
    with pytest.raises(RunnerPlanError, match="symlinked input"):
        select_retry_artifacts(
            linked_input,
            tmp_path / "linked-input-output",
            family="resolved-plan",
            current_attempt=1,
        )
    assert not (tmp_path / "linked-input-output").exists()

    output_target = tmp_path / "output-target"
    output_target.mkdir()
    linked_output = tmp_path / "linked-output-root"
    linked_output.symlink_to(output_target, target_is_directory=True)
    with pytest.raises(RunnerPlanError, match="symlinked output"):
        select_retry_artifacts(
            real_input,
            linked_output,
            family="resolved-plan",
            current_attempt=1,
        )
    assert not any(output_target.iterdir())

    with pytest.raises(RunnerPlanError, match="overlap"):
        select_retry_artifacts(
            real_input,
            real_artifact / "nested-output",
            family="resolved-plan",
            current_attempt=1,
        )
    assert not (real_artifact / "nested-output").exists()

    collision = tmp_path / "collision"
    for arch in ("amd64", "arm64"):
        directory = collision / f"digests-base-{arch}-1"
        directory.mkdir(parents=True)
        (directory / "same").write_text(arch, encoding="utf-8")
    with pytest.raises(RunnerPlanError, match="collision"):
        select_retry_artifacts(
            collision,
            tmp_path / "collision-output",
            family="base-digests",
            current_attempt=1,
        )


@pytest.mark.parametrize(
    "layout",
    (
        "equal",
        "nested-input",
        "nested-artifact",
        "ancestor",
        "canonical-equal",
        "canonical-nested",
    ),
)
def test_retry_artifact_selection_rejects_overlapping_roots_before_mutation(
    tmp_path: Path,
    layout: str,
) -> None:
    case = tmp_path / layout
    if layout == "ancestor":
        output = case / "output"
        input_dir = output / "input"
    else:
        input_dir = case / "input"
        output = case / "safe-placeholder"
    artifact = input_dir / "resolved-runner-plan-1"
    artifact.mkdir(parents=True)
    source = artifact / "plan.json"
    source.write_text("immutable", encoding="utf-8")

    if layout == "equal":
        output = input_dir
    elif layout == "nested-input":
        output = input_dir / "new-output"
    elif layout == "nested-artifact":
        output = artifact / "new-output"
    elif layout == "canonical-equal":
        alias = case / "alias"
        alias.symlink_to(case, target_is_directory=True)
        output = alias / "input"
    elif layout == "canonical-nested":
        alias = case / "alias"
        alias.symlink_to(input_dir, target_is_directory=True)
        output = alias / "new-output"

    source_tree_before = sorted(
        path.relative_to(input_dir) for path in input_dir.rglob("*")
    )
    with pytest.raises(RunnerPlanError, match="overlap"):
        select_retry_artifacts(
            input_dir,
            output,
            family="resolved-plan",
            current_attempt=1,
        )
    assert source.read_text(encoding="utf-8") == "immutable"
    assert sorted(path.relative_to(input_dir) for path in input_dir.rglob("*")) == (
        source_tree_before
    )


def test_retry_artifact_selection_accepts_disjoint_sibling_roots(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    artifact = input_dir / "resolved-runner-plan-1"
    artifact.mkdir(parents=True)
    (artifact / "plan.json").write_text("immutable", encoding="utf-8")

    nonexistent_output = tmp_path / "nonexistent-output"
    assert select_retry_artifacts(
        input_dir,
        nonexistent_output,
        family="resolved-plan",
        current_attempt=1,
    ) == {"plan": 1}
    assert (nonexistent_output / "plan.json").read_text() == "immutable"

    empty_output = tmp_path / "empty-output"
    empty_output.mkdir()
    assert select_retry_artifacts(
        input_dir,
        empty_output,
        family="resolved-plan",
        current_attempt=1,
    ) == {"plan": 1}
    assert (empty_output / "plan.json").read_text() == "immutable"


def test_retry_artifact_selection_cli_rejects_symlinked_input_root(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_dir = tmp_path / "input"
    artifact = input_dir / "resolved-runner-plan-1"
    artifact.mkdir(parents=True)
    (artifact / "plan.json").write_text("immutable", encoding="utf-8")
    linked_input = tmp_path / "linked-input"
    linked_input.symlink_to(input_dir, target_is_directory=True)
    output = tmp_path / "output"

    assert runner_images_main(
        [
            "select-retry-artifacts",
            "--input-dir",
            str(linked_input),
            "--output-dir",
            str(output),
            "--family",
            "resolved-plan",
            "--current-attempt",
            "1",
        ]
    ) == 2
    assert "symlinked input" in capsys.readouterr().err
    assert not output.exists()


def _promotion_record(
    backend: str,
    previous: str | None,
    promoted: str,
) -> dict:
    bootstrap = previous is None
    return {
        "state": "prepared",
        "backend": backend,
        "image": f"ghcr.io/ke7/helix-evo-runner-{backend}",
        "previous_digest": previous,
        "promoted_digest": promoted,
        "immutable_tag": f"cli-test-{backend}",
        "rollback_tag": None if bootstrap else "rollback-before-1-1",
        "bootstrap": bootstrap,
        "run_id": "1",
        "run_attempt": "1",
    }


class _FakeRegistry:
    def __init__(
        self,
        state: dict[str, str],
        *,
        fail_digest: str | None = None,
        interrupt_digest: str | None = None,
        interrupt_signal: int = signal.SIGTERM,
        mismatch_after_digest: str | None = None,
    ) -> None:
        self.state = state
        self.fail_digest = fail_digest
        self.interrupt_digest = interrupt_digest
        self.interrupt_signal = interrupt_signal
        self.mismatch_after_digest = mismatch_after_digest
        self.last_digest: str | None = None
        self.commands: list[list[str]] = []

    @staticmethod
    def _backend(reference: str) -> str:
        repository = reference.split("@", 1)[0].split(":", 1)[0]
        return repository.rsplit("/", 1)[1].rsplit("-", 1)[1]

    def __call__(self, argv: list[str]) -> str:
        self.commands.append(argv)
        if "create" in argv:
            reference = argv[-1]
            backend = self._backend(reference)
            digest = reference.rsplit("@", 1)[1]
            self.last_digest = digest
            if digest == self.interrupt_digest:
                raise PromotionInterrupted(self.interrupt_signal)
            if digest == self.fail_digest:
                raise RunnerPlanError("simulated registry write failure")
            self.state[backend] = digest
            return ""
        reference = argv[4]
        backend = self._backend(reference)
        digest = self.state[backend]
        if self.last_digest == self.mismatch_after_digest:
            digest = "sha256:" + "f" * 64
        return json.dumps({"digest": digest})


def _write_records(directory: Path, *records: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    for record in records:
        (directory / f"{record['backend']}.json").write_text(
            json.dumps(record), encoding="utf-8"
        )
    return directory


def test_tag_moves_target_the_repository_named_in_the_promotion_record(
    tmp_path: Path,
) -> None:
    """The registry prefix must have exactly one source of truth.

    A prefix change previously left ``_move_tag`` pointing at the old
    repository while the workflow published to the new one, so promotion and
    its compensating rollback would disagree mid-transaction.
    """
    previous = "sha256:" + "a" * 64
    promoted = "sha256:" + "c" * 64

    relocated = _promotion_record("codex", previous, promoted)
    relocated["image"] = "ghcr.io/ke7/helix-next-runner-codex"
    registry = _FakeRegistry({"codex": previous})
    registry.state = {"codex": previous}
    promote_latest_tags(
        [relocated],
        ledger_dir=tmp_path / "ledger",
        moved_file=tmp_path / "moved",
        output_file=tmp_path / "output",
        run_command=registry,
    )
    created = [argv[-1] for argv in registry.commands if "create" in argv]
    assert created == [f"ghcr.io/ke7/helix-next-runner-codex@{promoted}"]

    for bad in (
        None,
        "helix-evo-runner-codex",
        "docker.io/ke7/helix-evo-runner-codex",
        "ghcr.io/ke7/helix-evo-runner-claude",
        "ghcr.io/ke7/helix-evo-runner-codex:latest",
        "ghcr.io/ke7/helix-evo-runner-codex@sha256:" + "a" * 64,
        "ghcr.io/helix-evo-runner-codex",
    ):
        record = _promotion_record("codex", previous, promoted)
        record["image"] = bad
        directory = tmp_path / f"bad-{abs(hash(str(bad)))}"
        with pytest.raises(RunnerPlanError, match="promotion image"):
            _promotion_records(_write_records(directory, record))


def test_bootstrap_promotion_records_are_accepted_and_stay_fail_closed(
    tmp_path: Path,
) -> None:
    promoted = "sha256:" + "c" * 64
    previous = "sha256:" + "a" * 64

    accepted = _write_records(
        tmp_path / "bootstrap", _promotion_record("codex", None, promoted)
    )
    records = _promotion_records(accepted)
    assert records[0]["previous_digest"] is None
    assert records[0]["bootstrap"] is True

    # A null previous digest without the explicit bootstrap marker is a
    # malformed record, not an implicit first publication.
    unmarked = _promotion_record("codex", None, promoted)
    unmarked["bootstrap"] = False
    with pytest.raises(RunnerPlanError, match="malformed previous_digest"):
        _promotion_records(_write_records(tmp_path / "unmarked", unmarked))

    # A bootstrap record that also names a previous digest is contradictory.
    contradictory = _promotion_record("codex", previous, promoted)
    contradictory["bootstrap"] = True
    with pytest.raises(RunnerPlanError, match="null previous_digest"):
        _promotion_records(
            _write_records(tmp_path / "contradictory", contradictory)
        )

    # Bootstrap never relaxes the requirement on the digest being promoted.
    malformed = _promotion_record("codex", None, "sha256:not-a-digest")
    with pytest.raises(RunnerPlanError, match="malformed promoted_digest"):
        _promotion_records(_write_records(tmp_path / "malformed", malformed))

    non_boolean = _promotion_record("codex", None, promoted)
    non_boolean["bootstrap"] = "true"
    with pytest.raises(RunnerPlanError, match="bootstrap marker"):
        _promotion_records(_write_records(tmp_path / "non-boolean", non_boolean))


def test_restore_skips_bootstrap_records_without_counting_them_as_failures() -> None:
    promoted = "sha256:" + "c" * 64
    record = _promotion_record("codex", None, promoted)
    registry = _FakeRegistry({"codex": promoted})
    restore_latest_tags([record], run_command=registry)
    assert registry.commands == []
    assert registry.state == {"codex": promoted}


def test_mixed_bootstrap_and_normal_batch_compensates_only_what_it_can(
    tmp_path: Path,
) -> None:
    """One first publication plus one ordinary promotion in the same batch.

    The ordinary backend must be restored to its previous digest; the
    bootstrap backend has nothing to restore and must not turn the
    compensation into a reported failure.
    """
    claude_previous = "sha256:" + "b" * 64
    promoted = {"codex": "sha256:" + "c" * 64, "claude": "sha256:" + "d" * 64}
    records = [
        _promotion_record("codex", None, promoted["codex"]),
        _promotion_record("claude", claude_previous, promoted["claude"]),
    ]
    registry = _FakeRegistry(
        {"claude": claude_previous}, fail_digest=promoted["claude"]
    )
    with pytest.raises(RunnerPlanError, match="simulated"):
        promote_latest_tags(
            records,
            ledger_dir=tmp_path / "ledger",
            moved_file=tmp_path / "moved",
            output_file=tmp_path / "output",
            run_command=registry,
        )
    assert registry.state["claude"] == claude_previous
    assert registry.state["codex"] == promoted["codex"]
    created = [argv[-1] for argv in registry.commands if "create" in argv]
    assert not any(reference.startswith("null") for reference in created)
    assert sum("helix-evo-runner-codex@" in ref for ref in created) == 1


def test_promotion_transaction_compensates_error_and_signal_paths(
    tmp_path: Path,
) -> None:
    previous = {
        "codex": "sha256:" + "a" * 64,
        "claude": "sha256:" + "b" * 64,
    }
    promoted = {
        "codex": "sha256:" + "c" * 64,
        "claude": "sha256:" + "d" * 64,
    }
    records = [
        _promotion_record("codex", previous["codex"], promoted["codex"]),
        _promotion_record("claude", previous["claude"], promoted["claude"]),
    ]

    failed = _FakeRegistry(previous.copy(), fail_digest=promoted["claude"])
    with pytest.raises(RunnerPlanError, match="simulated"):
        promote_latest_tags(
            records,
            ledger_dir=tmp_path / "failed-ledger",
            moved_file=tmp_path / "failed-moved",
            output_file=tmp_path / "failed-output",
            run_command=failed,
        )
    assert failed.state == previous

    for signum in (signal.SIGINT, signal.SIGTERM):
        before_restore: list[str] = []
        interrupted = _FakeRegistry(
            previous.copy(),
            interrupt_digest=promoted["claude"],
            interrupt_signal=signum,
        )
        with pytest.raises(PromotionInterrupted) as caught:
            promote_latest_tags(
                records,
                ledger_dir=tmp_path / f"signal-ledger-{signum}",
                moved_file=tmp_path / f"signal-moved-{signum}",
                output_file=tmp_path / f"signal-output-{signum}",
                run_command=interrupted,
                before_restore=lambda: before_restore.append("called"),
            )
        assert caught.value.signum == signum
        assert before_restore == ["called"]
        assert interrupted.state == previous

    bad_output = tmp_path / "bad-output"
    bad_output.mkdir()
    output_failed = _FakeRegistry(previous.copy())
    with pytest.raises(OSError):
        promote_latest_tags(
            records,
            ledger_dir=tmp_path / "output-failure-ledger",
            moved_file=tmp_path / "output-failure-moved",
            output_file=bad_output,
            run_command=output_failed,
        )
    assert output_failed.state == previous


def test_promotion_transaction_writes_ledgers_and_detects_bad_restore(
    tmp_path: Path,
) -> None:
    previous = "sha256:" + "a" * 64
    promoted = "sha256:" + "c" * 64
    record = _promotion_record("codex", previous, promoted)
    registry = _FakeRegistry({"codex": previous})
    output = tmp_path / "github-output"
    ledger = tmp_path / "ledger"
    promote_latest_tags(
        [record],
        ledger_dir=ledger,
        moved_file=tmp_path / "moved",
        output_file=output,
        run_command=registry,
    )
    assert registry.state["codex"] == promoted
    assert output.read_text(encoding="utf-8") == "promoted=true\n"
    assert json.loads((ledger / "codex.json").read_text())["state"] == "committed"

    mismatched = _FakeRegistry(
        {"codex": promoted},
        mismatch_after_digest=previous,
    )
    with pytest.raises(RunnerPlanError, match="compensating rollback failed"):
        restore_latest_tags([record], run_command=mismatched)

    failed_restore = _FakeRegistry(
        {"codex": promoted},
        fail_digest=previous,
    )
    with pytest.raises(RunnerPlanError, match="compensating rollback failed"):
        restore_latest_tags([record], run_command=failed_restore)
