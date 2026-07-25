"""Offline release-safety tests for mutation-agent runner images."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from tools.runner_images import (
    RunnerPlanError,
    _fetch,
    _fetch_sha256,
    base_tag,
    change_plan,
    main as runner_images_main,
    parse_cursor_installer,
    parse_npm_metadata,
    resolve_cursor_checksums,
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


def test_upstream_fetches_retry_and_rehash_the_whole_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"cursor-archive-bytes"
    attempts: list[int] = []

    class _Stream:
        def __init__(self, chunks: list[bytes]) -> None:
            self.chunks = chunks

        def __enter__(self) -> "_Stream":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self, _size: int = -1) -> bytes:
            if not self.chunks:
                return b""
            chunk = self.chunks.pop(0)
            if chunk is None:  # pragma: no cover - defensive
                raise AssertionError
            return chunk

    def urlopen(request: object, timeout: float = 0.0) -> _Stream:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            # A truncated first attempt must not contribute partial bytes.
            raise urllib.error.URLError("connection reset")
        return _Stream([payload[:5], payload[5:]])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    delays: list[float] = []
    assert (
        _fetch_sha256("https://downloads.cursor.com/a.tgz", sleep=delays.append)
        == hashlib.sha256(payload).hexdigest()
    )
    assert delays == [1.0]
    assert len(attempts) == 2

    attempts.clear()
    delays = []

    def always_fails(request: object, timeout: float = 0.0) -> _Stream:
        attempts.append(len(attempts))
        raise urllib.error.URLError("connection reset")

    monkeypatch.setattr(urllib.request, "urlopen", always_fails)
    with pytest.raises(urllib.error.URLError):
        _fetch("https://registry.npmjs.org/x", sleep=delays.append)
    assert len(attempts) == 3
    assert delays == [1.0, 2.0]


def test_cursor_archives_are_rehashed_only_when_their_identity_moves() -> None:
    cursor = _catalog()["backends"]["cursor"]
    tarballs = {
        platform: cursor["platforms"][platform]["tarball"]
        for platform in ("amd64", "arm64")
    }
    hashed: list[str] = []

    def fetch(url: str) -> str:
        hashed.append(url)
        return "e" * 64

    # Unchanged version and unchanged URLs: reuse the reviewed digests and
    # download nothing.
    assert resolve_cursor_checksums(
        tarballs, cursor["version"], cursor, fetch_sha256=fetch
    ) == {
        platform: cursor["platforms"][platform]["sha256"]
        for platform in ("amd64", "arm64")
    }
    assert hashed == []

    # A new upstream version derives new URLs, so both archives are re-hashed.
    moved = {
        platform: url.replace(cursor["version"], "2026.07.21-deadbee")
        for platform, url in tarballs.items()
    }
    assert resolve_cursor_checksums(
        moved, "2026.07.21-deadbee", cursor, fetch_sha256=fetch
    ) == {"amd64": "e" * 64, "arm64": "e" * 64}
    assert sorted(hashed) == sorted(moved.values())

    # Same version but a URL the catalog never recorded still re-hashes.
    hashed.clear()
    tampered = dict(tarballs)
    tampered["arm64"] = "https://downloads.cursor.com/lab/x/linux/arm64/other.tgz"
    resolved = resolve_cursor_checksums(
        tampered, cursor["version"], cursor, fetch_sha256=fetch
    )
    assert hashed == [tampered["arm64"]]
    assert resolved["amd64"] == cursor["platforms"]["amd64"]["sha256"]
    assert resolved["arm64"] == "e" * 64

    # A catalog whose recorded digest is malformed is never trusted.
    hashed.clear()
    broken = copy.deepcopy(cursor)
    broken["platforms"]["amd64"]["sha256"] = "not-a-digest"
    resolve_cursor_checksums(
        tarballs, cursor["version"], broken, fetch_sha256=fetch
    )
    assert hashed == [tarballs["amd64"]]


def test_smoke_commands_are_restricted_to_a_conservative_charset() -> None:
    catalog = _catalog()
    # Every shipped command must still validate.
    validate_catalog(catalog)
    assert catalog["base"]["smoke_command"].count("&&") == 3

    for injected in (
        "claude --version; rm -rf /",
        "claude --version `id`",
        "claude --version $(id)",
        "claude --version > /etc/passwd",
        "claude --version\nid",
        "$(id)",
        "",
    ):
        catalog = _catalog()
        catalog["backends"]["claude"]["smoke_command"] = injected
        with pytest.raises(RunnerPlanError, match="smoke command"):
            validate_catalog(catalog)

    catalog = _catalog()
    catalog["base"]["smoke_command"] = "python --version; id"
    with pytest.raises(RunnerPlanError, match="base: smoke command"):
        validate_catalog(catalog)


def test_malformed_catalog_structures_exit_cleanly_instead_of_tracebacking(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Well-formed JSON with the wrong shapes must still exit 2."""
    for mutate in (
        lambda c: c["base"]["uv_wheels"].__setitem__("amd64", ["not", "a", "map"]),
        lambda c: c["backends"].__setitem__("cursor", "not-an-object"),
        lambda c: c["backends"]["codex"]["platforms"].__setitem__("amd64", 7),
        lambda c: c["backends"]["cursor"]["platforms"].__setitem__("arm64", None),
    ):
        catalog = _catalog()
        mutate(catalog)
        path = tmp_path / "runner-versions.json"
        path.write_text(json.dumps(catalog), encoding="utf-8")
        assert runner_images_main(["validate", "--catalog", str(path)]) == 2
        assert "runner image gate failed" in capsys.readouterr().err

    malformed_manifest = tmp_path / "manifest.json"
    malformed_manifest.write_text(
        json.dumps({"manifests": [{"platform": "linux/amd64"}]}), encoding="utf-8"
    )
    assert (
        runner_images_main(["verify-platforms", "--input", str(malformed_manifest)])
        == 2
    )
    assert "runner image gate failed" in capsys.readouterr().err


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
        if backend in {"claude", "gemini", "opencode"}:
            assert "npm install" not in text
            assert "npm cache" not in text
            assert "TARGETARCH" in text


def _run_bodies(text: str) -> list[tuple[int, str]]:
    """Return (line number, body) for every ``run:`` step in the workflow."""
    lines = text.splitlines()
    bodies: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if not stripped.startswith("run:"):
            index += 1
            continue
        indent = len(line) - len(stripped)
        body = [stripped.removeprefix("run:").strip()]
        index += 1
        while index < len(lines):
            following = lines[index]
            if following.strip() and (
                len(following) - len(following.lstrip()) <= indent
            ):
                break
            body.append(following)
            index += 1
        bodies.append((index, "\n".join(body)))
    return bodies


def test_base_tag_binds_the_whole_base_recipe(tmp_path: Path) -> None:
    """Editing base.Dockerfile or any pinned input must change the tag.

    Without this the "does the tag already exist?" check answers *yes* for a
    recipe that no longer matches the checkout, the base build is skipped, and
    every backend silently builds FROM a stale base.
    """
    base = _catalog()["base"]
    dockerfile = tmp_path / "base.Dockerfile"
    dockerfile.write_text("FROM node:22\n", encoding="utf-8")
    original = base_tag(base, dockerfile)
    assert original.startswith("node22-uv0.11.7-snapshot20260720-r")
    assert re.fullmatch(r"[0-9A-Za-z_][0-9A-Za-z_.-]{0,127}", original)

    dockerfile.write_text("FROM node:22\n# a comment\n", encoding="utf-8")
    assert base_tag(base, dockerfile) != original

    dockerfile.write_text("FROM node:22\n", encoding="utf-8")
    assert base_tag(base, dockerfile) == original

    for mutate in (
        lambda b: b.__setitem__("node_image", "node:22-bookworm-slim@sha256:" + "a" * 64),
        lambda b: b["uv_wheels"]["arm64"].__setitem__("sha256", "b" * 64),
        lambda b: b.__setitem__("uv_version", "0.11.8"),
        lambda b: b.__setitem__("debian_snapshot", "20260721T000000Z"),
    ):
        drifted = copy.deepcopy(base)
        mutate(drifted)
        assert base_tag(drifted, dockerfile) != original


def test_plan_builds_only_versions_the_registry_does_not_have() -> None:
    """Nightly is the check cadence; an upstream release is the publish trigger."""
    catalog = _catalog()
    everything_published = {name: True for name in catalog["backends"]}
    assert change_plan(catalog, everything_published) == []

    codex_released = dict(everything_published, codex=False)
    assert change_plan(catalog, codex_released) == [
        {
            "name": "codex",
            "dockerfile": "docker/codex.Dockerfile",
            "version": "0.145.0",
            "tag": "0.145.0",
        }
    ]

    # A registry with nothing in it builds all five.
    assert [entry["name"] for entry in change_plan(catalog, {})] == sorted(
        catalog["backends"]
    )


def test_forced_rebuild_never_overwrites_a_published_version_tag() -> None:
    """A tag someone pinned must not silently change meaning.

    Forcing a rebuild of a version that already shipped publishes
    ``<version>-r<run_id>`` and moves ``latest`` there; the original version
    tag keeps its original bytes.
    """
    catalog = _catalog()
    published = {name: True for name in catalog["backends"]}
    forced = change_plan(catalog, published, force=True, run_id="90210")
    assert [entry["tag"] for entry in forced] == [
        f"{catalog['backends'][entry['name']]['version']}-r90210" for entry in forced
    ]
    assert all(entry["version"] != entry["tag"] for entry in forced)

    # Forcing something that was never published just uses the plain version.
    fresh = change_plan(catalog, {}, force=True, run_id="90210")
    assert [entry["tag"] for entry in fresh] == [
        entry["version"] for entry in fresh
    ]

    # A replacement tag requires a usable run id rather than silently colliding.
    with pytest.raises(RunnerPlanError, match="numeric run id"):
        change_plan(catalog, published, force=True)
    with pytest.raises(RunnerPlanError, match="numeric run id"):
        change_plan(catalog, published, force=True, run_id="../evil")


def test_plan_rejects_an_unsafe_upstream_version() -> None:
    catalog = _catalog()
    catalog["backends"]["codex"]["version"] = "0.145.0 --build-arg=evil"
    with pytest.raises(RunnerPlanError, match="unsafe image version"):
        change_plan(catalog, {})


def _run_bodies(text: str) -> list[tuple[int, str]]:
    """Return (line number, body) for every ``run:`` step in the workflow."""
    lines = text.splitlines()
    bodies: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        stripped = line.lstrip()
        if not stripped.startswith("run:"):
            index += 1
            continue
        indent = len(line) - len(stripped)
        body = [stripped.removeprefix("run:").strip()]
        index += 1
        while index < len(lines):
            following = lines[index]
            if following.strip() and (
                len(following) - len(following.lstrip()) <= indent
            ):
                break
            body.append(following)
            index += 1
        bodies.append((index, "\n".join(body)))
    return bodies


def test_publish_workflow_cannot_publish_from_a_pull_request() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = text.split("env:", 1)[0]
    assert "pull_request:" not in trigger_block
    assert "push:" not in trigger_block
    assert "schedule:" in trigger_block
    assert "workflow_dispatch:" in trigger_block
    assert "github.repository == 'KE7/helix'" in text
    assert "github.event.repository.default_branch == 'main'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert (
        "ATTESTATION_SIGNER_WORKFLOW: KE7/helix/.github/workflows/publish-runners.yml"
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


def test_workflow_is_five_jobs_with_one_multiarch_build_per_backend() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    jobs = re.findall(r"^  ([a-z][a-z-]*):$", text[text.index("\njobs:\n") :], re.M)
    assert jobs == ["guard", "resolve", "base", "backend", "rollback", "notify"]

    # Both arches in one job: no per-architecture matrix, no digest handoff.
    assert text.count("platforms: linux/amd64,linux/arm64") == 2
    assert "ubuntu-24.04-arm" not in text
    assert "matrix.arch" not in text
    assert "select-retry-artifacts" not in text
    assert "verify-build-evidence" not in text
    # Emulation is what makes that possible.
    assert text.count("docker/setup-qemu-action@") == 3


def test_no_tag_is_created_before_the_image_passes_both_smokes() -> None:
    """The PR's best property: a tag never names an unvalidated image."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    for job, subject in (("base", "base"), ("backend", "backend")):
        start = text.index(f"\n  {job}:\n")
        end = len(text)
        for candidate in ("\n  backend:\n", "\n  rollback:\n", "\n  notify:\n"):
            position = text.find(candidate, start + 1)
            if position != -1:
                end = min(end, position)
        segment = text[start:end]
        assert "push-by-digest=true" in segment, subject
        build_at = segment.index("push-by-digest=true")
        smoke_at = segment.index("Smoke both architectures before any tag exists")
        attest_at = segment.index("actions/attest-build-provenance@")
        tag_at = segment.index("docker buildx imagetools create -t")
        assert build_at < smoke_at < attest_at < tag_at, subject
        # Both architectures really are exercised.
        assert "for platform in linux/amd64 linux/arm64" in segment, subject
        assert "verify-platforms" in segment, subject


def test_registry_absence_check_is_fail_closed() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "tag_exists() {" in text
    # Only a definitive "no such tag" is absence; everything else aborts.
    assert "grep -qiE 'not found|manifest unknown|no such manifest|404'" in text
    assert "registry inspection failed for" in text
    helper = text[text.index("tag_exists() {") : text.index("base_tag=\"$(")]
    assert helper.count("exit 1") == 1
    assert "return 1" in helper


def test_rollback_dispatch_is_not_queued_behind_the_nightly_build() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    concurrency = text[text.index("concurrency:") : text.index("permissions:")]
    assert (
        "group: runner-image-"
        "${{ github.event.inputs.operation || 'refresh' }}-"
        "${{ github.repository }}"
    ) in concurrency
    assert "cancel-in-progress: false" in concurrency
    # Rollback is dispatch-only and still verifies before it retags.
    rollback = text[text.index("\n  rollback:\n") : text.index("\n  notify:\n")]
    assert "inputs.operation == 'rollback'" in rollback
    assert "verify-platforms" in rollback
    assert "gh attestation verify" in rollback
    assert 'for platform in linux/amd64 linux/arm64' in rollback
    assert '[[ "$actual" == "$TARGET_DIGEST" ]]' in rollback


def test_failure_notifier_covers_cancellation_and_dedupes_beyond_one_page() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    notifier = text[text.index("\n  notify:\n") :]
    assert "(failure() || cancelled())" in notifier
    # `cancelled()` is only legal in a job/step `if:`, so the notifier
    # classifies the run from the upstream job results.
    assert "JOB_RESULTS: ${{ toJSON(needs) }}" in notifier
    assert 'job.result === "cancelled"' in notifier
    assert "was cancelled" in notifier
    assert "github.paginate(github.rest.issues.listForRepo" in notifier
    assert 'const label = "runner-image-refresh";' in notifier
    assert "labels: label" in notifier
    assert "labels: [label]" in notifier


def test_no_workflow_expression_is_interpolated_into_a_shell_body() -> None:
    """Every ``${{ }}`` value reaches a shell through a quoted env var.

    ``change_plan`` constrains the matrix to five literal backend names, so
    there is no live injection today, but interpolating an expression straight
    into a ``run:`` body is what actionlint and zizmor flag, and it is the only
    thing standing between this workflow and a future unconstrained value.
    """
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    bodies = _run_bodies(text)
    assert len(bodies) >= 8
    offenders = [line for line, body in bodies if "${{" in body]
    assert offenders == [], f"expressions interpolated into run: at {offenders}"
    for declaration in (
        "BACKEND: ${{ matrix.name }}",
        "VERSION: ${{ matrix.version }}",
        "IMAGE_TAG: ${{ matrix.tag }}",
        "BUILD_DIGEST: ${{ steps.build.outputs.digest }}",
        "BASE_TAG: ${{ needs.resolve.outputs.base_tag }}",
    ):
        assert declaration in text
    assert (
        len(re.findall(
            r'\[\[ "\$BACKEND" =~ \^\(claude\|codex\|cursor\|gemini\|opencode\)\$ \]\]',
            text,
        ))
        == 4
    )


def test_workflow_version_smoke_is_boundary_aware() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    # A bare substring match would let a CLI reporting 1.10 satisfy 1.1.
    assert 'grep -F "$version"' not in text
    assert '(^|[^0-9A-Za-z.])v?${escaped}' in text
    assert "([^0-9A-Za-z.]|$)" in text
    assert "escaped=" in text
