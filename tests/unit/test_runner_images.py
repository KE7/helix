"""Offline release-safety tests for mutation-agent runner images.

Every CLI subcommand is exercised through ``main()``. The previous suite was
851 lines and left 21% of the tool unexecuted, with four of seven subcommands
never invoked at all -- which is how twenty lines of duplicated dead code
reached the branch. Source-text greps over the workflow are gone: actionlint
and zizmor run in CI and cover that ground properly. What survives here either
calls the tool and asserts an outcome, or encodes a property no linter knows.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from helix.config import AgentConfig
from helix.mutator import _build_backend_args
from tools.runner_images import (
    BACKENDS,
    LOCKFILE_ARTIFACTS,
    PLATFORMS,
    RunnerPlanError,
    _fetch,
    _fetch_sha256,
    base_tag,
    build_arguments,
    change_plan,
    discover,
    main as runner_images_main,
    parse_cursor_installer,
    resolve_cursor_checksums,
    validate_catalog,
    validate_lockfile,
    verify_codex_catalog,
    verify_platforms,
)

ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "docker" / "runner-versions.json"
LOCKFILE_PATH = ROOT / "docker" / "package-lock.json"
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "publish-runners.yml"
BASE_IMAGE = "ghcr.io/ke7/helix-evo-runner-base@sha256:" + "a" * 64


def _catalog() -> dict:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def _lock() -> dict:
    return json.loads(LOCKFILE_PATH.read_text(encoding="utf-8"))


CURSOR_INSTALLER = (
    b"#!/bin/sh\nVERSION=2026.07.20-8cc9c0b\n"
    b"https://downloads.cursor.com/lab/2026.07.20-8cc9c0b/linux/x64/a.tgz\n"
)


def _resolved() -> dict:
    """Resolve every backend with no network access at all.

    Everything but Cursor is already pinned in git, so only Cursor's installer
    needs a fixture. Its archive digests come from the reviewed catalog pins,
    which is exactly what `--cursor-checksums` reuses in production when the
    version has not moved.
    """
    import tools.runner_images as module

    original = module._fetch
    module._fetch = lambda *args, **kwargs: CURSOR_INSTALLER  # type: ignore[assignment]
    try:
        resolved = discover(_catalog(), _lock(), cursor_checksums=False)
    finally:
        module._fetch = original  # type: ignore[assignment]

    recorded = _catalog()["backends"]["cursor"]["platforms"]
    for platform in PLATFORMS:
        resolved["backends"]["cursor"]["platforms"][platform]["sha256"] = recorded[
            platform
        ]["sha256"]
    return resolved


# --------------------------------------------------------------------------
# The checked-in pins
# --------------------------------------------------------------------------


def test_checked_in_catalog_and_lockfile_are_complete_and_content_pinned() -> None:
    validate_catalog(_catalog())
    validate_lockfile(_lock())


def test_lockfile_covers_every_artifact_an_image_extracts() -> None:
    """Each allowlisted package resolves to an on-platform linux/glibc binary."""
    packages = _lock()["packages"]
    for backend, group in LOCKFILE_ARTIFACTS.items():
        for slot, package in group.items():
            node = packages[f"node_modules/{package}"]
            assert node["resolved"].startswith("https://registry.npmjs.org/")
            assert node["integrity"].startswith("sha512-")
            if slot in PLATFORMS:
                assert node["cpu"] == [{"amd64": "x64", "arm64": "arm64"}[slot]]
                assert node["os"] == ["linux"], (backend, slot)
                # A musl build would not run on the Debian base.
                assert node.get("libc") in (None, ["glibc"]), (backend, slot)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda c: c["backends"]["claude"].update(version="9.9.9"), "lockfile"),
        (
            lambda c: c["backends"]["cursor"].update(installer="https://evil.invalid"),
            "installer",
        ),
        (
            lambda c: c["backends"]["gemini"].update(
                tarball="https://evil.invalid/x.tgz"
            ),
            "untrusted",
        ),
        (lambda c: c["backends"]["gemini"].update(sha512="beef"), "sha512"),
        (
            lambda c: c["backends"]["codex"].update(
                dockerfile="docker/evil.Dockerfile"
            ),
            "dockerfile",
        ),
        (
            lambda c: c["backends"]["codex"].update(smoke_command="codex; rm -rf /"),
            "smoke command",
        ),
        (lambda c: c["base"].update(node_image="node:22"), "digest-pinned"),
        (lambda c: c["base"]["uv_wheels"]["amd64"].update(sha256="nope"), "sha256"),
        (lambda c: c["backends"].pop("gemini"), "exactly the five backends"),
    ],
)
def test_catalog_fails_closed_on_untrusted_or_unmeasured_input(mutate, match) -> None:
    catalog = _catalog()
    mutate(catalog)
    with pytest.raises(RunnerPlanError, match=match):
        validate_catalog(catalog)


@pytest.mark.parametrize(
    ("package", "field", "value", "match"),
    [
        ("@anthropic-ai/claude-code-linux-x64", "cpu", ["arm64"], "cpu"),
        ("@anthropic-ai/claude-code-linux-x64", "os", ["darwin"], "os"),
        ("@anthropic-ai/claude-code-linux-x64", "libc", ["musl"], "libc"),
        ("opencode-linux-arm64", "integrity", "sha256-abc", "integrity"),
        ("opencode-linux-arm64", "resolved", "https://evil.invalid/x.tgz", "untrusted"),
        ("@openai/codex", "version", "0.144.9", "0.145.0"),
    ],
)
def test_lockfile_fails_closed_on_off_platform_or_unmeasured_artifacts(
    package: str, field: str, value: object, match: str
) -> None:
    lock = _lock()
    lock["packages"][f"node_modules/{package}"][field] = value
    with pytest.raises(RunnerPlanError, match=match):
        validate_lockfile(lock)


def test_lockfile_fails_closed_when_an_allowlisted_artifact_disappears() -> None:
    """A removed optional dependency must stop the build, not shrink the image."""
    lock = _lock()
    del lock["packages"]["node_modules/@lydell/node-pty-linux-arm64"]
    with pytest.raises(RunnerPlanError, match="not in the lockfile"):
        validate_lockfile(lock)


# --------------------------------------------------------------------------
# Resolution and build arguments
# --------------------------------------------------------------------------


def test_discover_resolves_every_backend_from_pins_already_in_git() -> None:
    resolved = _resolved()
    assert sorted(resolved["backends"]) == sorted(BACKENDS)
    claude = resolved["backends"]["claude"]
    assert claude["version"] == "2.1.218"
    assert len(claude["sha512"]) == 128
    # Codex's platform binaries are aliases carrying their own version string.
    codex = resolved["backends"]["codex"]
    assert codex["platforms"]["amd64"]["package_version"] == "0.145.0-linux-x64"
    # Gemini takes its own tarball from the catalog and node-pty from the lock.
    gemini = resolved["backends"]["gemini"]
    assert "gemini-cli" in gemini["tarball"]
    assert "node-pty" in gemini["artifacts"]["shared"][0]["tarball"]


def test_build_arguments_cover_every_arg_each_dockerfile_declares() -> None:
    """Every ARG a Dockerfile reads must be supplied, and nothing else.

    The expectation is derived from the Dockerfiles rather than restated, so
    this survives a refactor and still catches the real bug: a key the tool
    stops emitting silently builds from a stale ARG default instead of failing.
    """
    resolved = _resolved()
    for name in BACKENDS:
        arguments = dict(
            entry.split("=", 1)
            for entry in build_arguments(resolved["backends"][name], BASE_IMAGE)
        )
        dockerfile = (ROOT / "docker" / f"{name}.Dockerfile").read_text()
        declared = set(re.findall(r"^ARG ([A-Z0-9_]+)", dockerfile, re.M))
        declared -= {"TARGETARCH"}  # supplied by buildx, not by us
        assert not declared - set(arguments), f"{name}: unsupplied build args"
        # ...and nothing else. buildx accepts a surplus argument silently, so
        # only this direction catches the tool drifting from the recipe.
        assert not set(arguments) - declared - {"BASE_IMAGE"}, f"{name}: surplus args"
        assert arguments["CLI_VERSION"] == resolved["backends"][name]["version"]


def test_dockerfile_arg_defaults_match_the_pins_the_tool_resolves() -> None:
    """Every pinned ARG default equals the pin resolved from the lockfile.

    The defaults exist so ``docker build`` works off a plain checkout with no
    tooling. That makes them a second copy of every pin, and a second copy is
    only safe if it cannot drift, so this asserts the recipes and the pin store
    agree. A lockfile bump that does not also update the recipe fails here,
    rather than leaving a default that silently builds last release's tarball
    for anyone building locally.
    """
    resolved = _resolved()
    for name in BACKENDS:
        arguments = dict(
            entry.split("=", 1)
            for entry in build_arguments(resolved["backends"][name], BASE_IMAGE)
        )
        dockerfile = (ROOT / "docker" / f"{name}.Dockerfile").read_text()
        # Enumerated from the recipe, never from a hand-written list: an ARG
        # added later is covered the moment it is declared, so the guarantee
        # cannot narrow silently as pins are added.
        declared = [
            arg
            for arg in re.findall(r"^ARG ([A-Z0-9_]+)", dockerfile, re.M)
            # TARGETARCH is supplied by buildx; BASE_IMAGE's default is a local
            # convenience tag rather than a content pin.
            if arg not in {"TARGETARCH", "BASE_IMAGE"}
        ]
        assert declared, f"{name}: no pinned ARGs found in the recipe"
        defaults = dict(re.findall(r"^ARG ([A-Z0-9_]+)=(.*)$", dockerfile, re.M))
        for arg in declared:
            assert arg in defaults, (
                f"{name}.Dockerfile: ARG {arg} has no default, so a plain "
                f"`docker build` would build it empty"
            )
            assert defaults[arg] == arguments[arg], (
                f"{name}.Dockerfile: ARG {arg} default has drifted from the "
                f"resolved pin\n"
                f"  recipe default: {defaults[arg]}\n"
                f"  resolved pin:   {arguments[arg]}"
            )


def test_base_dockerfile_arg_defaults_match_the_catalog() -> None:
    """The base recipe's defaults are the same pins the workflow passes it."""
    base = _catalog()["base"]
    defaults = dict(
        re.findall(
            r"^ARG ([A-Z0-9_]+)=(.*)$",
            (ROOT / "docker" / "base.Dockerfile").read_text(),
            re.M,
        )
    )
    expected = {
        "NODE_BASE": base["node_image"],
        "DEBIAN_SNAPSHOT": base["debian_snapshot"],
        "UV_VERSION": base["uv_version"],
    } | {
        f"UV_{platform.upper()}_{field}": base["uv_wheels"][platform][key]
        for platform in PLATFORMS
        for field, key in (("WHEEL", "url"), ("SHA256", "sha256"))
    }
    # Equality both ways: a new pinned ARG that this test does not know about
    # is a failure, not a silent gap.
    assert defaults == expected, (
        "docker/base.Dockerfile defaults have drifted from the catalog\n"
        f"  only in recipe:  {sorted(defaults.items() - expected.items())}\n"
        f"  only in catalog: {sorted(expected.items() - defaults.items())}"
    )


def test_build_arguments_digests_match_the_lockfile_byte_for_byte() -> None:
    """The build args are the lockfile's own integrity fields, hex-decoded."""
    resolved = _resolved()
    packages = _lock()["packages"]
    for name, group in LOCKFILE_ARTIFACTS.items():
        arguments = dict(
            entry.split("=", 1)
            for entry in build_arguments(resolved["backends"][name], BASE_IMAGE)
        )
        for slot, package in group.items():
            integrity = packages[f"node_modules/{package}"]["integrity"]
            expected = base64.b64decode(integrity.removeprefix("sha512-")).hex()
            key = {"cli": "CLI_SHA512", "shared": "CLI_SHARED_SHA512"}.get(
                slot, f"CLI_{slot.upper()}_SHA512"
            )
            assert arguments[key] == expected, f"{name}/{slot}"


@pytest.mark.parametrize(
    "base_image",
    [
        "ghcr.io/ke7/helix-evo-runner-base:latest",
        "ghcr.io/ke7/helix-evo-runner-base",
        "docker.io/library/node@sha256:" + "a" * 64,
    ],
)
def test_build_arguments_reject_a_floating_base(base_image: str) -> None:
    with pytest.raises(RunnerPlanError, match="digest-pinned"):
        build_arguments(_resolved()["backends"]["codex"], base_image)


def test_build_arguments_reject_an_injected_control_byte() -> None:
    resolved = _resolved()
    item = copy.deepcopy(resolved["backends"]["codex"])
    item["tarball"] = "https://registry.npmjs.org/a.tgz\nCLI_SHA512=bad"
    with pytest.raises(RunnerPlanError, match="control byte"):
        build_arguments(item, BASE_IMAGE)


# --------------------------------------------------------------------------
# Cursor: the one backend in no package ecosystem
# --------------------------------------------------------------------------


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


def test_cursor_archives_are_rehashed_only_when_their_identity_moves() -> None:
    cursor = _catalog()["backends"]["cursor"]
    tarballs = {p: cursor["platforms"][p]["tarball"] for p in PLATFORMS}
    hashed: list[str] = []

    def fetch(url: str) -> str:
        hashed.append(url)
        return "e" * 64

    # Unchanged version and URLs: reuse the reviewed digests, download nothing.
    assert resolve_cursor_checksums(
        tarballs, cursor["version"], cursor, fetch_sha256=fetch
    ) == {p: cursor["platforms"][p]["sha256"] for p in PLATFORMS}
    assert hashed == []

    # Same version but a URL the catalog never recorded still re-hashes.
    tampered = dict(tarballs)
    tampered["arm64"] = "https://downloads.cursor.com/lab/x/linux/arm64/other.tgz"
    resolved = resolve_cursor_checksums(
        tampered, cursor["version"], cursor, fetch_sha256=fetch
    )
    assert hashed == [tampered["arm64"]]
    assert resolved["amd64"] == cursor["platforms"]["amd64"]["sha256"]


def test_cursor_smoke_covers_both_entry_points_the_mutator_uses() -> None:
    """The mutator runs `cursor agent`; auth paths run `cursor-agent`.

    `docker/cursor.Dockerfile` installs `cursor-agent` and writes a `cursor`
    shim beside it. Deleting the shim would break every Cursor mutation inside
    the sandbox, so the smoke command has to exercise both spellings. The
    expectation is read out of the mutator's own argv rather than grepped out
    of its source text.
    """
    argv = _build_backend_args("/workspace", AgentConfig(backend="cursor"), "prompt.md")
    assert argv[:2] == ["cursor", "agent"]

    dockerfile = (ROOT / "docker" / "cursor.Dockerfile").read_text()
    assert "/usr/local/bin/cursor-agent" in dockerfile
    assert "> /usr/local/bin/cursor" in dockerfile

    smoke = _catalog()["backends"]["cursor"]["smoke_command"]
    assert smoke == "cursor-agent --version && cursor agent --version"


# --------------------------------------------------------------------------
# Network behaviour
# --------------------------------------------------------------------------


def test_upstream_fetches_retry_and_rehash_the_whole_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A stream that dies halfway must never contribute partial bytes."""
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
            return self.chunks.pop(0) if self.chunks else b""

    def urlopen(request: object, timeout: float = 0.0) -> _Stream:
        attempts.append(len(attempts))
        if len(attempts) == 1:
            raise urllib.error.URLError("connection reset")
        return _Stream([payload[:5], payload[5:]])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    delays: list[float] = []
    assert (
        _fetch_sha256("https://downloads.cursor.com/a.tgz", sleep=delays.append)
        == hashlib.sha256(payload).hexdigest()
    )
    assert delays == [1.0]

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


# --------------------------------------------------------------------------
# Planning and post-build assertions
# --------------------------------------------------------------------------


def test_plan_builds_only_versions_the_registry_does_not_have() -> None:
    resolved = _resolved()
    published = {name: True for name in BACKENDS}
    assert change_plan(resolved, published) == []

    published["codex"] = False
    builds = change_plan(resolved, published)
    assert [b["name"] for b in builds] == ["codex"]
    assert builds[0]["tag"] == resolved["backends"]["codex"]["version"]


def test_forced_rebuild_never_overwrites_a_published_version_tag() -> None:
    resolved = _resolved()
    published = {name: True for name in BACKENDS}
    builds = change_plan(resolved, published, force=True, run_id="42")
    assert all(b["tag"] == f"{b['version']}-r42" for b in builds)
    # A forced rebuild of a published version needs a run id to tag with.
    with pytest.raises(RunnerPlanError, match="run id"):
        change_plan(resolved, published, force=True)


def test_plan_rejects_an_unsafe_upstream_version() -> None:
    resolved = _resolved()
    resolved["backends"]["codex"]["version"] = "0.145.0 --build-arg=x"
    with pytest.raises(RunnerPlanError, match="unsafe image version"):
        change_plan(resolved, {})


def test_base_tag_binds_the_whole_base_recipe(tmp_path: Path) -> None:
    """Editing the Dockerfile or any pin must change the tag."""
    base = _catalog()["base"]
    dockerfile = tmp_path / "base.Dockerfile"
    dockerfile.write_text("FROM node:22\n")
    original = base_tag(base, dockerfile)

    dockerfile.write_text("FROM node:22\nRUN echo drift\n")
    assert base_tag(base, dockerfile) != original
    dockerfile.write_text("FROM node:22\n")
    assert base_tag(base, dockerfile) == original

    for field, value in (
        ("node_image", base["node_image"].replace("6c74", "6c75")),
        ("debian_snapshot", "20260721T000000Z"),
        ("uv_version", "0.11.8"),
    ):
        drifted = dict(base)
        drifted[field] = value
        assert base_tag(drifted, dockerfile) != original


def test_codex_catalog_requires_luna_with_xhigh_second_highest() -> None:
    def payload(efforts: list[str]) -> dict:
        return {
            "models": [
                {
                    "slug": "gpt-5.6-luna",
                    "supported_reasoning_levels": [{"effort": e} for e in efforts],
                }
            ]
        }

    verify_codex_catalog(payload(["low", "medium", "high", "xhigh", "max"]))
    with pytest.raises(RunnerPlanError, match="reasoning order"):
        verify_codex_catalog(payload(["low", "medium", "high", "max"]))
    with pytest.raises(RunnerPlanError, match="gpt-5.6-luna"):
        verify_codex_catalog({"models": []})


def test_manifest_parity_requires_exact_linux_amd64_and_arm64() -> None:
    def manifest(platforms: list[tuple[str, str]]) -> dict:
        return {
            "manifests": [
                {"platform": {"os": os, "architecture": arch}} for os, arch in platforms
            ]
        }

    verify_platforms(manifest([("linux", "amd64"), ("linux", "arm64")]))
    # Attestation entries carry os "unknown" and must be ignored, not counted.
    verify_platforms(
        manifest([("linux", "amd64"), ("linux", "arm64"), ("unknown", "unknown")])
    )
    with pytest.raises(RunnerPlanError, match="parity"):
        verify_platforms(manifest([("linux", "amd64")]))


# --------------------------------------------------------------------------
# The CLI itself -- every subcommand, through main()
# --------------------------------------------------------------------------


def test_every_cli_subcommand_runs_and_reports_its_exit_status(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The gap that let twenty lines of dead code into main() unnoticed."""
    resolved_path = tmp_path / "resolved.json"
    resolved_path.write_text(json.dumps(_resolved()))
    published = tmp_path / "published.json"
    published.write_text(json.dumps({name: False for name in BACKENDS}))

    assert runner_images_main(["validate", "--catalog", str(CATALOG_PATH)]) == 0

    assert (
        runner_images_main(
            [
                "plan",
                "--resolved",
                str(resolved_path),
                "--published",
                str(published),
                "--output",
                str(tmp_path / "builds.json"),
            ]
        )
        == 0
    )
    builds = json.loads((tmp_path / "builds.json").read_text())
    assert [b["name"] for b in builds] == list(BACKENDS)

    assert runner_images_main(["base-tag", "--catalog", str(CATALOG_PATH)]) == 0
    assert capsys.readouterr().out.strip().startswith("node22-uv")

    assert (
        runner_images_main(
            [
                "build-args",
                "--resolved",
                str(resolved_path),
                "--backend",
                "gemini",
                "--base-image",
                BASE_IMAGE,
            ]
        )
        == 0
    )
    assert "CLI_SHARED_TARBALL=" in capsys.readouterr().out

    catalog_input = tmp_path / "models.json"
    catalog_input.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "slug": "gpt-5.6-luna",
                        "supported_reasoning_levels": [
                            {"effort": e}
                            for e in ("low", "medium", "high", "xhigh", "max")
                        ],
                    }
                ]
            }
        )
    )
    assert (
        runner_images_main(["verify-codex-catalog", "--input", str(catalog_input)]) == 0
    )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "manifests": [
                    {"platform": {"os": "linux", "architecture": "amd64"}},
                    {"platform": {"os": "linux", "architecture": "arm64"}},
                ]
            }
        )
    )
    assert runner_images_main(["verify-platforms", "--input", str(manifest)]) == 0


def test_discover_subcommand_writes_a_resolved_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`discover` is what the workflow depends on to produce a file on disk."""
    installer = (
        b"VERSION=2026.07.20-8cc9c0b\n"
        b"https://downloads.cursor.com/lab/2026.07.20-8cc9c0b/linux/x64/a.tgz\n"
    )
    monkeypatch.setattr("tools.runner_images._fetch", lambda *a, **k: installer)
    output = tmp_path / "resolved.json"
    assert (
        runner_images_main(
            ["discover", "--catalog", str(CATALOG_PATH), "--output", str(output)]
        )
        == 0
    )
    resolved = json.loads(output.read_text())
    assert sorted(resolved["backends"]) == sorted(BACKENDS)
    assert resolved["backends"]["cursor"]["version"] == "2026.07.20-8cc9c0b"


@pytest.mark.parametrize(
    "payload",
    ['["not", "an", "object"]', '{"schema_version": 2, "backends": null}', "{}"],
)
def test_malformed_input_exits_cleanly_instead_of_tracebacking(
    tmp_path: Path, payload: str
) -> None:
    broken = tmp_path / "catalog.json"
    broken.write_text(payload)
    assert runner_images_main(["validate", "--catalog", str(broken)]) == 2


# --------------------------------------------------------------------------
# The one workflow property no linter knows
# --------------------------------------------------------------------------


def test_no_tag_is_created_before_the_image_passes_both_smokes() -> None:
    """A tag must never name an image that has not been validated."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    for job in ("base", "backend"):
        start = text.index(f"\n  {job}:\n")
        end = min(
            (
                p
                for p in (
                    text.find(c, start + 1)
                    for c in ("\n  backend:\n", "\n  rollback:\n", "\n  notify:\n")
                )
                if p != -1
            ),
            default=len(text),
        )
        segment = text[start:end]
        build_at = segment.index("push-by-digest=true")
        smoke_at = segment.index("Smoke both architectures before any tag exists")
        attest_at = segment.index("actions/attest-build-provenance@")
        tag_at = segment.index("docker buildx imagetools create -t")
        assert build_at < smoke_at < attest_at < tag_at, job
        assert "for platform in linux/amd64 linux/arm64" in segment, job
        assert "verify-platforms" in segment, job


def test_registry_absence_check_is_fail_closed() -> None:
    """Only a definitive "no such tag" is absence; everything else aborts."""
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    helper = text[text.index("tag_exists() {") : text.index('base_tag="$(')]
    assert "grep -qiE 'not found|manifest unknown|no such manifest|404'" in text
    assert "registry inspection failed for" in text
    assert helper.count("exit 1") == 1
    assert "return 1" in helper


def test_publish_workflow_cannot_publish_from_a_pull_request() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    trigger_block = text.split("env:", 1)[0]
    assert "pull_request:" not in trigger_block
    assert "github.repository == 'KE7/helix'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert (
        "ATTESTATION_SIGNER_WORKFLOW: KE7/helix/.github/workflows/publish-runners.yml"
    ) in text
    assert '--signer-workflow "$ATTESTATION_SIGNER_WORKFLOW"' in text
    assert "--deny-self-hosted-runners" in text
