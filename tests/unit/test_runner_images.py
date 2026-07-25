"""Offline release-safety tests for mutation-agent runner images."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pytest

from tools.runner_images import (
    RunnerPlanError,
    _fetch,
    _fetch_sha256,
    assert_immutable_collision,
    base_tag,
    change_plan,
    inspect_ghcr_tag,
    immutable_tag,
    main as runner_images_main,
    parse_cursor_installer,
    parse_npm_metadata,
    resolve_cursor_checksums,
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
            sleep=lambda _seconds: None,
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


def _flaky_ghcr(
    statuses: list[int],
    manifest: bytes,
    digest: str,
) -> Callable[[urllib.request.Request, float], _HTTPResponse]:
    """A GHCR opener whose manifest endpoint fails with ``statuses`` in order."""
    remaining = list(statuses)

    def opener(
        request: urllib.request.Request, timeout: float
    ) -> _HTTPResponse:
        if request.full_url.startswith("https://ghcr.io/token?"):
            return _HTTPResponse(b'{"token":"registry-token"}')
        if remaining:
            raise urllib.error.HTTPError(
                request.full_url,
                remaining.pop(0),
                "registry error",
                {},  # type: ignore[arg-type]
                io.BytesIO(),
            )
        return _HTTPResponse(manifest, {"Docker-Content-Digest": digest})

    return opener


def test_registry_reads_retry_only_transient_failures() -> None:
    manifest = b'{"schemaVersion":2}'
    digest = f"sha256:{hashlib.sha256(manifest).hexdigest()}"

    for statuses in ([503], [429, 500]):
        delays: list[float] = []
        assert (
            inspect_ghcr_tag(
                "ghcr.io/ke7/helix-evo-runner-codex",
                "cli-0.145.0-rabc",
                actor="ci",
                token="secret",
                urlopen=_flaky_ghcr(statuses, manifest, digest),
                sleep=delays.append,
            )
            == digest
        )
        assert delays == [1.0, 2.0][: len(statuses)]

    # Three consecutive transient failures exhaust the cap and fail closed.
    delays = []
    with pytest.raises(RunnerPlanError, match="HTTP 503"):
        inspect_ghcr_tag(
            "ghcr.io/ke7/helix-evo-runner-codex",
            "cli-0.145.0-rabc",
            actor="ci",
            token="secret",
            urlopen=_flaky_ghcr([503, 503, 503], manifest, digest),
            sleep=delays.append,
        )
    assert delays == [1.0, 2.0]

    # A 404 is the decisive "absent" answer and 401/403 are credential
    # problems: neither is ever retried.
    for status, expected in ((404, None), (401, "HTTP 401"), (403, "HTTP 403")):
        delays = []
        opener = _flaky_ghcr([status, status, status], manifest, digest)
        if expected is None:
            assert (
                inspect_ghcr_tag(
                    "ghcr.io/ke7/helix-evo-runner-codex",
                    "cli-0.145.0-rabc",
                    actor="ci",
                    token="secret",
                    urlopen=opener,
                    sleep=delays.append,
                )
                is None
            )
        else:
            with pytest.raises(RunnerPlanError, match=expected):
                inspect_ghcr_tag(
                    "ghcr.io/ke7/helix-evo-runner-codex",
                    "cli-0.145.0-rabc",
                    actor="ci",
                    token="secret",
                    urlopen=opener,
                    sleep=delays.append,
                )
        assert delays == []


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


def test_cursor_immutable_tag_tracks_content_not_installer_noise() -> None:
    """A comment-only edit to Cursor's install script must not force a rebuild.

    The installer is never executed in an image; it is parsed for the version
    and the official artifact URLs. The tarball URLs plus their SHA-256 digests
    are what actually bind the shipped content, so those — and not the
    installer hash — decide the immutable tag.
    """
    catalog = _catalog()
    base = catalog["base"]["immutable_tag"]
    item = copy.deepcopy(catalog["backends"]["cursor"])
    original = immutable_tag(item, base)

    item["installer_sha256"] = "f" * 64
    assert immutable_tag(item, base) == original

    item = copy.deepcopy(catalog["backends"]["cursor"])
    item["platforms"]["arm64"]["sha256"] = "f" * 64
    assert immutable_tag(item, base) != original

    item = copy.deepcopy(catalog["backends"]["cursor"])
    item["platforms"]["amd64"]["tarball"] = (
        "https://downloads.cursor.com/lab/2026.07.20-8cc9c0b/linux/x64/other.tar.gz"
    )
    assert immutable_tag(item, base) != original

    item = copy.deepcopy(catalog["backends"]["cursor"])
    item["version"] = "2026.07.21-deadbee"
    assert immutable_tag(item, base) != original


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


def test_retry_artifact_families_index_their_groups_by_name(
    tmp_path: Path,
) -> None:
    """Attempt and logical group come from named regex groups.

    Positional indexing (or ``match.lastindex``) silently picks the wrong group
    the moment a pattern gains an alternation, and the release families differ
    in how many groups they have.
    """
    releases = tmp_path / "releases"
    for backend, attempt in (("cursor", 3), ("opencode", 11)):
        directory = releases / f"release-{backend}-{attempt}"
        directory.mkdir(parents=True)
        (directory / f"{backend}.json").write_text(backend, encoding="utf-8")
    assert select_retry_artifacts(
        releases,
        tmp_path / "releases-out",
        family="releases",
        current_attempt=11,
        required=("cursor", "opencode"),
    ) == {"cursor": 3, "opencode": 11}

    # The single-group family still reports the attempt, not the whole name.
    plan = tmp_path / "plan"
    (plan / "resolved-runner-plan-42").mkdir(parents=True)
    (plan / "resolved-runner-plan-42" / "plan.json").write_text("x")
    assert select_retry_artifacts(
        plan,
        tmp_path / "plan-out",
        family="resolved-plan",
        current_attempt=42,
    ) == {"plan": 42}

    source = (ROOT / "tools" / "runner_images.py").read_text(encoding="utf-8")
    assert "int(match.group(match.lastindex" not in source
    assert source.count("(?P<attempt>[0-9]+)") == 4
    assert source.count("(?P<logical>") == 3


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
