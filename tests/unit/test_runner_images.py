"""Offline release-safety tests for mutation-agent runner images."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest

import tools.runner_images as runner_images
from tools.runner_images import (
    BACKENDS,
    RunnerPlanError,
    _fetch,
    _fetch_sha256,
    base_tag,
    build_arguments,
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
    resolve_cursor_checksums(tarballs, cursor["version"], broken, fetch_sha256=fetch)
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


def test_catalog_rejects_untrusted_or_unmeasured_backend_sources() -> None:
    """The security win: every download is host-pinned and digest-pinned."""
    catalog = _catalog()
    catalog["backends"]["claude"]["package"] = "lookalike-package"
    with pytest.raises(RunnerPlanError, match="npm package"):
        validate_catalog(catalog)

    catalog = _catalog()
    catalog["backends"]["gemini"]["artifacts"]["amd64"][0]["tarball"] = (
        "https://evil.invalid/node-pty.tgz"
    )
    with pytest.raises(RunnerPlanError, match="untrusted upstream URL"):
        validate_catalog(catalog)

    catalog = _catalog()
    catalog["backends"]["cursor"]["platforms"]["arm64"]["sha256"] = "nope"
    with pytest.raises(RunnerPlanError, match="cursor/arm64: invalid sha256"):
        validate_catalog(catalog)

    catalog = _catalog()
    catalog["backends"]["opencode"]["artifacts"]["amd64"].pop()
    with pytest.raises(RunnerPlanError, match="exact artifact packages"):
        validate_catalog(catalog)

    catalog = _catalog()
    catalog["base"]["node_image"] = "node:22-bookworm-slim"
    with pytest.raises(RunnerPlanError, match="digest-pinned"):
        validate_catalog(catalog)

    catalog = _catalog()
    catalog["backends"]["codex"]["version"] = "0.144.0"
    catalog["backends"]["codex"]["platforms"]["amd64"]["package_version"] = (
        "0.144.0-linux-x64"
    )
    catalog["backends"]["codex"]["platforms"]["arm64"]["package_version"] = (
        "0.144.0-linux-arm64"
    )
    with pytest.raises(RunnerPlanError, match="at least 0.145.0"):
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
        lambda b: b.__setitem__(
            "node_image", "node:22-bookworm-slim@sha256:" + "a" * 64
        ),
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
    assert [entry["tag"] for entry in fresh] == [entry["version"] for entry in fresh]

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
    helper = text[text.index("tag_exists() {") : text.index('base_tag="$(')]
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
    assert "for platform in linux/amd64 linux/arm64" in rollback
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


def test_workflow_version_smoke_is_boundary_aware() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    # A bare substring match would let a CLI reporting 1.10 satisfy 1.1.
    assert 'grep -F "$version"' not in text
    assert "(^|[^0-9A-Za-z.])v?${escaped}" in text
    assert "([^0-9A-Za-z.]|$)" in text
    assert "escaped=" in text


def _fake_upstream(catalog: dict) -> Callable[..., bytes]:
    """Serve npm/cursor responses from the checked-in pins, offline."""
    releases: dict[str, dict] = {}
    for name, item in catalog["backends"].items():
        if item["kind"] in {"npm", "codex"}:
            optional = {}
            if item["kind"] == "npm":
                for group, artifacts in item["artifacts"].items():
                    for artifact in artifacts:
                        version = (
                            artifact["tarball"].rsplit("-", 1)[1].removesuffix(".tgz")
                        )
                        optional[artifact["package"]] = version
                        releases[f"{artifact['package']}@{version}"] = {
                            "version": version,
                            "dist": {
                                "tarball": artifact["tarball"],
                                "integrity": "sha512-"
                                + base64.b64encode(
                                    bytes.fromhex(artifact["sha512"])
                                ).decode(),
                            },
                            "optionalDependencies": (
                                optional if group == "shared" else {}
                            ),
                        }
            if item["kind"] == "codex":
                for platform in ("amd64", "arm64"):
                    source = item["platforms"][platform]
                    optional[item["package"]] = item["version"]
                    releases[f"{item['package']}@{source['package_version']}"] = {
                        "version": source["package_version"],
                        "dist": {
                            "tarball": source["tarball"],
                            "integrity": "sha512-"
                            + base64.b64encode(
                                bytes.fromhex(source["sha512"])
                            ).decode(),
                        },
                    }
            releases[item["package"]] = {
                "dist-tags": {"latest": item["version"]},
                "versions": {
                    item["version"]: {
                        "dist": {
                            "tarball": item["tarball"],
                            "integrity": "sha512-"
                            + base64.b64encode(bytes.fromhex(item["sha512"])).decode(),
                        },
                        "optionalDependencies": optional,
                    }
                },
            }

    cursor = catalog["backends"]["cursor"]

    def fetch(url: str, *args: object, **kwargs: object) -> bytes:
        if url == "https://cursor.com/install":
            return (
                f"VERSION={cursor['version']}\n"
                f"URL={cursor['platforms']['amd64']['tarball']}\n"
            ).encode()
        path = urllib.parse.unquote(url.removeprefix("https://registry.npmjs.org/"))
        if "/" in path and not path.startswith("@"):
            package, version = path.rsplit("/", 1)
            return json.dumps(releases[f"{package}@{version}"]).encode()
        if path.count("/") == 2:
            scope, name, version = path.split("/")
            return json.dumps(releases[f"{scope}/{name}@{version}"]).encode()
        return json.dumps(releases[path]).encode()

    return fetch


def test_discovery_resolves_every_build_argument_the_workflow_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """discover() must emit exactly what the backend job feeds to buildx."""
    catalog = _catalog()
    monkeypatch.setattr(runner_images, "_fetch", _fake_upstream(catalog))
    monkeypatch.setattr(
        runner_images,
        "_fetch_sha256",
        lambda url, *a, **k: pytest.fail(f"unexpected re-hash of {url}"),
    )
    resolved = runner_images.discover(catalog, cursor_checksums=True)

    assert sorted(resolved["backends"]) == sorted(BACKENDS)
    for name in BACKENDS:
        item = resolved["backends"][name]
        assert item["version"] == catalog["backends"][name]["version"], name
        assert item["dockerfile"] == f"docker/{name}.Dockerfile", name
        assert item["smoke_command"], name
        if item["kind"] == "npm":
            for group in item["artifacts"].values():
                for artifact in group:
                    assert artifact["tarball"].startswith("https://registry.npmjs.org/")
                    assert re.fullmatch(r"[0-9a-f]{128}", artifact["sha512"])
        if item["kind"] in {"codex", "cursor"}:
            assert sorted(item["platforms"]) == ["amd64", "arm64"], name

    # Cursor keeps its installer hash as evidence and, because the version did
    # not move, reuses the reviewed archive checksums instead of re-hashing.
    cursor = resolved["backends"]["cursor"]
    assert re.fullmatch(r"[0-9a-f]{64}", cursor["installer_sha256"])
    for platform in ("amd64", "arm64"):
        assert (
            cursor["platforms"][platform]["sha256"]
            == catalog["backends"]["cursor"]["platforms"][platform]["sha256"]
        )

    # The resolved manifest is what the plan is computed from.
    assert change_plan(resolved, {name: True for name in BACKENDS}) == []


def test_cursor_smoke_covers_both_entry_points_mutator_uses() -> None:
    """`src/helix/mutator.py` invokes `cursor agent`, not `cursor-agent`.

    Only the auth commands in `backends.py` call `cursor-agent` directly; every
    mutation goes through the `/usr/local/bin/cursor` shim in
    `docker/cursor.Dockerfile`. Deleting that shim would break every Cursor run
    inside the sandbox, so the smoke has to exercise both entry points.
    """
    mutator = (ROOT / "src" / "helix" / "mutator.py").read_text(encoding="utf-8")
    assert '"cursor",\n            "agent",' in mutator

    dockerfile = (ROOT / "docker" / "cursor.Dockerfile").read_text(encoding="utf-8")
    assert "/usr/local/bin/cursor-agent" in dockerfile
    assert "> /usr/local/bin/cursor" in dockerfile

    smoke = _catalog()["backends"]["cursor"]["smoke_command"]
    assert smoke == "cursor-agent --version && cursor agent --version"


def test_build_arguments_cover_every_arg_each_dockerfile_declares() -> None:
    """Every ARG a Dockerfile reads must be supplied, and nothing else.

    This logic used to be a jq case statement inside the workflow, where a
    missing key silently produced an empty build argument and the image was
    built from whatever the Dockerfile's default happened to be.
    """
    catalog = _catalog()
    base_image = "ghcr.io/ke7/helix-evo-runner-base@sha256:" + "a" * 64
    for name, item in catalog["backends"].items():
        arguments = dict(
            entry.split("=", 1) for entry in build_arguments(item, base_image)
        )
        dockerfile = (ROOT / "docker" / f"{name}.Dockerfile").read_text()
        declared = set(re.findall(r"^ARG ([A-Z0-9_]+)", dockerfile, re.M))
        declared -= {"TARGETARCH"}  # supplied by buildx, not by us
        missing = declared - set(arguments)
        assert not missing, f"{name}: unsupplied build args {sorted(missing)}"
        assert arguments["BASE_IMAGE"] == base_image
        assert arguments["CLI_VERSION"] == item["version"]
        # Absent artifacts must be explicit empty strings, never omitted: the
        # Dockerfile ARG defaults are stale pinned URLs, so an omitted key
        # would silently build from the previous release.
        if name == "opencode":
            assert arguments["CLI_AMD64_FALLBACK_TARBALL"].endswith(".tgz")
        elif name in {"claude", "gemini"}:
            assert arguments["CLI_AMD64_FALLBACK_TARBALL"] == ""
        # Only Gemini has a shared artifact.
        if name == "gemini":
            assert "node-pty" in arguments["CLI_SHARED_TARBALL"]
        elif name in {"claude", "opencode"}:
            assert arguments["CLI_SHARED_TARBALL"] == ""


def test_build_arguments_fail_closed_on_a_floating_base_or_injected_value() -> None:
    catalog = _catalog()
    item = catalog["backends"]["codex"]
    for unpinned in (
        "ghcr.io/ke7/helix-evo-runner-base:latest",
        "ghcr.io/ke7/helix-evo-runner-base",
        "docker.io/ke7/helix-evo-runner-base@sha256:" + "a" * 64,
        "ghcr.io/ke7/helix-evo-runner-base@sha256:nope",
    ):
        with pytest.raises(RunnerPlanError, match="digest-pinned"):
            build_arguments(item, unpinned)

    # A newline in any value would inject an extra build argument into the
    # heredoc the workflow writes to $GITHUB_OUTPUT.
    injected = copy.deepcopy(item)
    injected["platforms"]["amd64"]["sha512"] = "a" * 128 + "\nCLI_SHA512=evil"
    with pytest.raises(RunnerPlanError, match="control byte"):
        build_arguments(
            injected, "ghcr.io/ke7/helix-evo-runner-base@sha256:" + "a" * 64
        )

    with pytest.raises(RunnerPlanError, match="unknown backend kind"):
        build_arguments(
            {"kind": "rogue", "version": "1.0.0"},
            "ghcr.io/ke7/helix-evo-runner-base@sha256:" + "a" * 64,
        )
