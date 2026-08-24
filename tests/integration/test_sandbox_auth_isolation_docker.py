"""Real-Docker proof that per-candidate auth volumes actually isolate credentials.

``tests/unit/test_sandbox.py`` covers the same design by construction: it
patches ``helix.sandbox.subprocess.run`` and inspects the argv HELIX builds, so
no container ever runs and no volume is ever created. This module covers the
runtime behaviour instead -- a shared credential mutated in place under
concurrent use is a runtime failure, not a static argv defect -- using a real
Docker daemon and HELIX's own volume-creation and seeding functions
(``_create_candidate_auth_volume``, ``_seed_candidate_auth_volume``,
``run_sandboxed_command``), never a hand-rolled reimplementation of the
copy/mount logic under test.

Both tests are parametrized over *every* backend key in
``helix.sandbox.AUTH_CREDENTIAL_MANIFEST``, not just ``claude``, because each
backend's manifest entry (a distinct source path, target filename, and mount
destination) is a place the copy/mount logic can fail independently. A backend
is not enabled for ``auth = "volume"`` until a real test confirms its manifest,
so this parametrization is the gate, not a convenience. It is driven directly
off ``AUTH_CREDENTIAL_MANIFEST`` (backend list) and ``AUTH_MOUNT_DESTINATIONS``
(looked up by key, never copied into a second hardcoded list), so a new
manifest entry is covered automatically and a backend present in one dict but
missing from the other fails with a loud ``KeyError`` rather than being
silently skipped.

Credential safety
------------------
Every credential used here is synthetic. These tests never read, mount, or
reference an operator's real ``~/.claude``, ``~/.codex``, ``~/.cursor``,
``~/.gemini``, or ``~/.local/share/opencode``, and never touch the real
``helix-auth-<backend>`` volumes: ``helix.sandbox.sandbox_auth_volume_name``
is monkeypatched for the duration of each test to point at a throwaway
``helix-test-synthetic-login-*`` volume instead. That is the sole seam
touched; every other code path under test runs unmodified.

Credential shape: ``claude``'s login-volume shape (``claudeAiOauth`` with an
OAuth token pair) and ``cursor``'s (an access/refresh token pair) are
reflected in the fake fixtures below. Neither was obtained by reading an
operator's credential file -- which the paragraph above forbids; each was read
out of the shipped runner image's own CLI bundle, where the code that writes
the file states the shape directly. The on-disk shapes of ``codex``'s
``auth.json``, ``gemini``'s ``oauth_creds.json``, and ``opencode``'s
``auth.json`` are deliberately not reproduced; those three get a minimal,
valid, non-empty JSON object (see ``_fake_credential`` below) instead.
Everything the isolation tests assert -- byte-for-byte isolation across
candidate volumes, mode and ownership, manifest-only contents, and that a
write to one candidate reaches neither another candidate nor the login volume
-- depends on the manifest's source/target *paths* and on file *bytes*
surviving unchanged, not on any credential's internal schema. The shape
matters only to the backend-specific test at the end of this module, which
hands a real CLI the seeded record and reads what it says about it.

These tests carry the ``docker_integration`` marker (see ``pyproject.toml``)
and skip when the daemon or a fixture image is unavailable, so a machine
without Docker is not a failure. Set ``HELIX_DOCKER_TESTS_STRICT=1``, as CI
does, to turn every such skip into a failure.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import uuid
from pathlib import Path

import pytest

import helix.sandbox as sandbox_module
from helix.backends import DEFAULT_BACKEND_IMAGES
from helix.config import SandboxConfig
from helix.sandbox import (
    AUTH_CREDENTIAL_MANIFEST,
    AUTH_MOUNT_DESTINATIONS,
    CandidateAuthVolume,
    _create_candidate_auth_volume,
    _remove_candidate_auth_volume,
    _run_docker,
    _seed_candidate_auth_volume,
    run_sandboxed_command,
)


pytestmark = pytest.mark.docker_integration

# Every backend HELIX's manifest declares -- the sole source of truth for
# which backends these tests cover. Sorted only for stable, readable test IDs.
_BACKENDS = sorted(AUTH_CREDENTIAL_MANIFEST)

# Every backend other than ``claude`` gets this placeholder rather than a
# shape-accurate fixture; see the module docstring's "Credential shape"
# section for why that is a safety requirement, not an omission.
_GENERIC_SYNTHETIC_SHAPE_NOTE = (
    "real on-disk shape unknown -- placeholder for byte-isolation testing only, "
    "see test_sandbox_auth_isolation_docker.py module docstring"
)


# Shapes taken from the code inside each shipped runner image's CLI bundle
# that writes the file, never from an operator's credential. Every token value
# below is a fixed placeholder that no service ever issued.
_SYNTHETIC_TOKEN = "synthetic-not-a-real-token"


def _fake_credential(backend: str) -> str:
    """A synthetic seed credential, backend-shaped where the shape is known."""
    if backend == "claude":
        return json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "fake-not-a-real-token",
                    "refreshToken": "fake",
                    "expiresAt": 9999999999,
                    "scopes": ["user:inference"],
                }
            }
        )
    if backend == "cursor":
        # The CLI's credential writer stores exactly these keys.
        return json.dumps(
            {
                "accessToken": _SYNTHETIC_TOKEN,
                "refreshToken": f"{_SYNTHETIC_TOKEN}-refresh",
                "apiKey": None,
            }
        )
    return json.dumps(
        {
            "synthetic_credential": True,
            "backend": backend,
            "state": "seed",
            "note": _GENERIC_SYNTHETIC_SHAPE_NOTE,
        }
    )


def _rotated_credential(backend: str) -> str:
    """A synthetic "rotated" value simulating an in-place token refresh."""
    if backend == "cursor":
        rotated = json.loads(_fake_credential(backend))
        rotated["accessToken"] = f"{_SYNTHETIC_TOKEN}-rotated-by-candidate-a"
        return json.dumps(rotated)
    if backend == "claude":
        return json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "fake-rotated-token-from-candidate-a",
                    "refreshToken": "fake-rotated",
                    "expiresAt": 9999999999,
                    "scopes": ["user:inference"],
                }
            }
        )
    return json.dumps(
        {
            "synthetic_credential": True,
            "backend": backend,
            "state": "rotated-by-candidate-a",
            "note": _GENERIC_SYNTHETIC_SHAPE_NOTE,
        }
    )


def _fixture_image(backend: str) -> str:
    """Resolve the runner image for *backend*, honoring test-only overrides."""
    per_backend_override = os.environ.get(f"HELIX_DOCKER_TEST_IMAGE_{backend.upper()}")
    if per_backend_override:
        return per_backend_override
    # Legacy single-backend override, kept for anyone still setting it locally.
    if backend == "claude":
        legacy_override = os.environ.get("HELIX_DOCKER_TEST_IMAGE")
        if legacy_override:
            return legacy_override
    return DEFAULT_BACKEND_IMAGES[backend]


def _docker(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], check=check, capture_output=True, text=True, timeout=30
    )


def _require_docker_fixture(image: str) -> None:
    """Skip when Docker or the fixture image is unavailable.

    Set ``HELIX_DOCKER_TESTS_STRICT=1`` to turn every skip into a failure. CI
    sets it, because a Docker job that silently skips every test still reports
    green -- which is worse than not running the job at all.
    """
    strict = os.environ.get("HELIX_DOCKER_TESTS_STRICT") == "1"

    def _bail(reason: str) -> None:
        if strict:
            pytest.fail(f"{reason} [HELIX_DOCKER_TESTS_STRICT=1 forbids skipping]")
        pytest.skip(reason)

    try:
        daemon = _docker("info")
        inspected = _docker("image", "inspect", image)
    except (OSError, subprocess.SubprocessError) as exc:
        _bail(f"Docker daemon unavailable: {exc}")
        return
    if daemon.returncode != 0:
        _bail(f"Docker daemon unavailable: {daemon.stderr.strip()}")
        return
    if inspected.returncode != 0:
        _bail(f"fixture image {image!r} is not installed locally")


def _diagnostic_docker_args(
    volume_name: str, mount: str, script: str, *, writable: bool, image: str
) -> list[str]:
    """Build argv for a throwaway inspection container.

    *image* is always one of the already-pulled backend runner images (see
    ``_fixture_image``) -- these diagnostics never pull a separate
    third-party image just to poke at a volume.
    """
    mode = "rw" if writable else "ro"
    return [
        "docker",
        "run",
        "--rm",
        "--user",
        "root",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        "-v",
        f"{volume_name}:{mount}:{mode}",
        image,
        "sh",
        "-c",
        script,
    ]


def _make_synthetic_login_volume(
    backend: str,
    credential_json: str,
    source_rel: str,
    *,
    image: str,
    mode: str = "600",
) -> str:
    """Create a throwaway, HELIX-shaped login volume holding a fake credential.

    This mimics the *shape* the real login volume has after a real login (a
    manifest-declared relative path, mode 0600 by default, uid/gid 1000) but
    is a brand-new randomly-named volume -- it is never
    ``helix-auth-<backend>`` and never derived from any real credential.

    *mode* defaults to the real login volume's own mode, ``600``. Tests
    exercising ``_seed_command``'s wrong-mode guard (ADR verification
    requirement 4) pass a different value deliberately.
    """
    name = f"helix-test-synthetic-login-{backend}-{uuid.uuid4().hex}"
    _run_docker(["docker", "volume", "create", name])
    container_path = f"/home/node/{source_rel}"
    parent = str(Path(container_path).parent)
    script = (
        f"set -eu; mkdir -p {shlex.quote(parent)}; "
        f"printf '%s' {shlex.quote(credential_json)} > {shlex.quote(container_path)}; "
        f"chmod {shlex.quote(mode)} {shlex.quote(container_path)}; "
        "chown -R 1000:1000 /home/node"
    )
    _run_docker(
        _diagnostic_docker_args(name, "/home/node", script, writable=True, image=image)
    )
    return name


def _read_credential(
    volume_name: str, rel_path: str, *, mount: str = "/check", image: str
) -> tuple[str, str]:
    """Return ``(file content, "mode uid:gid")`` read from inside *volume_name*."""
    path = f"{mount}/{rel_path}"
    script = (
        f"cat {shlex.quote(path)}; "
        "printf '\\n===MODE===\\n'; "
        f"stat -c '%a %u:%g' {shlex.quote(path)}"
    )
    result = _run_docker(
        _diagnostic_docker_args(volume_name, mount, script, writable=False, image=image)
    )
    content, mode = result.stdout.split("\n===MODE===\n")
    return content, mode.strip()


def _write_credential(
    volume_name: str, rel_path: str, content: str, *, mount: str = "/check", image: str
) -> None:
    path = f"{mount}/{rel_path}"
    script = f"printf '%s' {shlex.quote(content)} > {shlex.quote(path)}"
    _run_docker(
        _diagnostic_docker_args(volume_name, mount, script, writable=True, image=image)
    )


def _list_top_level(volume_name: str, *, mount: str = "/check", image: str) -> set[str]:
    result = _run_docker(
        _diagnostic_docker_args(
            volume_name, mount, f"ls -a {mount}", writable=False, image=image
        )
    )
    return {entry for entry in result.stdout.split() if entry not in {".", ".."}}


def _volume_exists(name: str) -> bool:
    return _docker("volume", "inspect", name).returncode == 0


def _force_remove_volume(name: str) -> None:
    _docker("volume", "rm", "-f", name)


@pytest.fixture(autouse=True)
def _no_leaked_test_volumes():
    """Belt-and-braces cleanup: remove any stray test-created volumes.

    Scoped to prefixes HELIX itself only ever assigns to throwaway
    credential volumes (``helix-candidate-auth-``) or that this test module
    invents for its synthetic login fixture (``helix-test-synthetic-login-``).
    Neither prefix can collide with a real ``helix-auth-<backend>`` volume.
    """
    yield
    listing = _docker("volume", "ls", "--format", "{{.Name}}")
    for name in listing.stdout.splitlines():
        if name.startswith("helix-candidate-auth-") or name.startswith(
            "helix-test-synthetic-login-"
        ):
            _force_remove_volume(name)


# --------------------------------------------------------------------------
# Manifest/destination parity -- makes drift between the two dicts loud.
# --------------------------------------------------------------------------


def test_auth_manifest_and_mount_destinations_cover_the_same_backends() -> None:
    """Every manifest backend has a mount destination and vice versa.

    The parametrized tests below look up ``AUTH_MOUNT_DESTINATIONS[backend]``
    directly (never a second hardcoded backend list), so a missing entry
    already fails those tests loudly with a ``KeyError`` at setup time. This
    assertion makes that guarantee explicit and independent of Docker.
    """
    assert set(AUTH_CREDENTIAL_MANIFEST) == set(AUTH_MOUNT_DESTINATIONS)


# --------------------------------------------------------------------------
# TEST 1 -- runtime isolation via HELIX's own volume-creation/seeding code
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_candidate_auth_volumes_isolate_synthetic_credential_under_rotation(
    backend: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_image = _fixture_image(backend)
    _require_docker_fixture(fixture_image)

    real_login_volume_name = f"helix-auth-{backend}"  # never created/read/written below
    source_rel, target_rel = AUTH_CREDENTIAL_MANIFEST[backend][0]
    fake_credential = _fake_credential(backend)
    rotated_credential = _rotated_credential(backend)

    login_volume = _make_synthetic_login_volume(
        backend, fake_credential, source_rel, image=fixture_image
    )

    # The only seam we touch: redirect the seeding step's *source* volume
    # away from the real "helix-auth-<backend>" toward our synthetic fixture.
    # `_seed_candidate_auth_volume` looks this up by calling
    # `sandbox_auth_volume_name(volume.backend)` from `helix.sandbox`'s own
    # module globals, so patching the module attribute here also redirects
    # the call made from inside `_seed_candidate_auth_volume` itself.
    monkeypatch.setattr(sandbox_module, "sandbox_auth_volume_name", lambda _backend: login_volume)

    candidate_a: CandidateAuthVolume | None = None
    candidate_b: CandidateAuthVolume | None = None
    try:
        # 1 & 2: real HELIX volume-creation + seeding code, called directly.
        candidate_a = _create_candidate_auth_volume(backend)
        candidate_b = _create_candidate_auth_volume(backend)
        assert candidate_a.name != candidate_b.name
        assert candidate_a.name != login_volume
        assert candidate_b.name != login_volume
        assert real_login_volume_name not in (candidate_a.name, candidate_b.name, login_volume)

        _seed_candidate_auth_volume(candidate_a, fixture_image)
        _seed_candidate_auth_volume(candidate_b, fixture_image)

        # 3: both candidates contain the credential, mode 0600, owned 1000:1000.
        content_a, mode_a = _read_credential(candidate_a.name, target_rel, image=fixture_image)
        content_b, mode_b = _read_credential(candidate_b.name, target_rel, image=fixture_image)
        assert content_a == fake_credential
        assert content_b == fake_credential
        assert mode_a == "600 1000:1000"
        assert mode_b == "600 1000:1000"

        # 4: the central assertion. Write a different "rotated" credential
        # into candidate A's own copy -- what a backend does when it refreshes
        # a token in place -- then assert candidate B is byte-unchanged.
        _write_credential(candidate_a.name, target_rel, rotated_credential, image=fixture_image)
        rotated_content_a, _ = _read_credential(candidate_a.name, target_rel, image=fixture_image)
        assert rotated_content_a == rotated_credential

        unaffected_content_b, unaffected_mode_b = _read_credential(
            candidate_b.name, target_rel, image=fixture_image
        )
        assert unaffected_content_b == fake_credential, (
            "candidate B's credential changed after candidate A rotated its "
            "own copy -- candidate auth volumes are not isolated"
        )
        assert unaffected_mode_b == "600 1000:1000"

        # The synthetic login volume itself is also untouched by A's write.
        login_content, _ = _read_credential(
            login_volume, source_rel, mount="/home/node", image=fixture_image
        )
        assert login_content == fake_credential

        # 5: candidate volumes only contain the manifest-declared targets --
        # no trace of the login volume's own nested layout leaks through.
        top_level_a = _list_top_level(candidate_a.name, image=fixture_image)
        top_level_b = _list_top_level(candidate_b.name, image=fixture_image)
        source_top_component = source_rel.split("/", 1)[0]
        assert source_top_component not in top_level_a  # source's directory name, not the target's
        assert source_top_component not in top_level_b
        expected_top_level = {target for _, target in AUTH_CREDENTIAL_MANIFEST[backend]}
        # claude's transcript bind is deliberately pre-created by the seed
        # helper (see `_seed_command`'s claude special case in sandbox.py);
        # every other backend's candidate volume holds only its credential.
        if backend == "claude":
            expected_top_level.add("projects")
        assert top_level_a == expected_top_level
        assert top_level_b == expected_top_level

        login_mountpoint = _docker(
            "volume", "inspect", "-f", "{{.Mountpoint}}", login_volume
        ).stdout.strip()
        for candidate in (candidate_a, candidate_b):
            candidate_mountpoint = _docker(
                "volume", "inspect", "-f", "{{.Mountpoint}}", candidate.name
            ).stdout.strip()
            assert candidate_mountpoint != login_mountpoint
    finally:
        _force_remove_volume(login_volume)

    # 6: cleanup actually destroys the candidate volumes (real Docker check).
    assert _volume_exists(candidate_a.name)
    assert _volume_exists(candidate_b.name)
    _remove_candidate_auth_volume(candidate_a)
    _remove_candidate_auth_volume(candidate_b)
    assert not _volume_exists(candidate_a.name)
    assert not _volume_exists(candidate_b.name)


# --------------------------------------------------------------------------
# TEST 2 -- E2E through HELIX's real sandbox entry point
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_e2e_sandboxed_agent_command_sees_isolated_seeded_credential(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_image = _fixture_image(backend)
    _require_docker_fixture(fixture_image)

    real_login_volume_name = f"helix-auth-{backend}"
    source_rel, target_rel = AUTH_CREDENTIAL_MANIFEST[backend][0]
    mount_dest = AUTH_MOUNT_DESTINATIONS[backend]
    fake_credential = _fake_credential(backend)
    rotated_credential = _rotated_credential(backend)

    login_volume = _make_synthetic_login_volume(
        backend, fake_credential, source_rel, image=fixture_image
    )
    monkeypatch.setattr(sandbox_module, "sandbox_auth_volume_name", lambda _backend: login_volume)

    observed_docker_args: list[list[str]] = []
    original_docker_args = sandbox_module._docker_args

    def _spy(*args: object, **kwargs: object) -> list[str]:
        built = original_docker_args(*args, **kwargs)  # type: ignore[arg-type]
        observed_docker_args.append(built)
        return built

    monkeypatch.setattr(sandbox_module, "_docker_args", _spy)

    source = tmp_path / "candidate"
    source.mkdir()
    dest_path = f"{mount_dest}/{target_rel}"

    # A harmless command that never calls a model or any backend CLI: it only
    # reads the seeded credential and process environment, then writes a
    # "rotated" value the way a real refresh would -- purely local file I/O.
    command = [
        "sh",
        "-c",
        (
            f"cat {shlex.quote(dest_path)}; echo; "
            f"echo MODE:$(stat -c '%a %u:%g' {shlex.quote(dest_path)}); "
            "echo HOME_ENV:$HOME; "
            "if mount | grep -q \" on /home/node type tmpfs\"; then "
            "echo HOME_MOUNT:tmpfs; else echo HOME_MOUNT:not-tmpfs; fi; "
            f"printf '%s' {shlex.quote(rotated_credential)} > {shlex.quote(dest_path)}"
        ),
    ]

    result = run_sandboxed_command(
        command,
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True, auth="volume", network="none"),
        scope="agent",
        sync_back=False,
        image=fixture_image,
        agent_backend=backend,
    )

    try:
        assert result.returncode == 0, (
            f"sandboxed command failed for backend {backend!r}: {result.stderr}"
        )
        lines = result.stdout.splitlines()
        assert lines[0] == fake_credential, "seeded credential not present/readable at mount dest"
        assert lines[1] == "MODE:600 1000:1000", "credential not owned/moded correctly in container"
        assert lines[2] == "HOME_ENV:/home/node"
        assert lines[3] == "HOME_MOUNT:tmpfs", "HOME is not the private per-container tmpfs"

        # The agent container's own argv never mentions the login volume, by
        # name or by our monkeypatched stand-in -- it only ever sees a
        # `helix-candidate-auth-*` volume.
        agent_call = next(
            call for call in observed_docker_args if call[:2] == ["docker", "run"]
        )
        assert not any(login_volume in item for item in agent_call)
        assert not any(real_login_volume_name in item for item in agent_call)
        mount_arg = next(
            item for item in agent_call if item.endswith(f":{mount_dest}:rw")
        )
        candidate_volume_in_argv = mount_arg.split(":", 1)[0]
        assert candidate_volume_in_argv.startswith(f"helix-candidate-auth-{backend}-")

        # The candidate volume the agent actually used is gone once
        # `run_sandboxed_command` returns (its own cleanup path ran).
        assert not _volume_exists(candidate_volume_in_argv)

        # The central assertion, replayed through the real end-to-end entry
        # point: the in-container "rotation" write landed on the candidate's
        # private copy, not on the synthetic login volume.
        login_content, _ = _read_credential(
            login_volume, source_rel, mount="/home/node", image=fixture_image
        )
        assert login_content == fake_credential, (
            "the login volume was mutated by a write made inside the agent "
            "container -- credentials are not isolated"
        )
    finally:
        _force_remove_volume(login_volume)


# --------------------------------------------------------------------------
# TEST 3 & 4 -- fail-closed guards on malformed source material
#
# ADR verification requirement 4: "Missing, malformed, oversized, or
# wrong-mode source material must prevent the agent from starting with a
# stable, redacted error. This is mandatory." Neither guard has real-Docker
# coverage: `_seed_command`'s checks run inside the seed helper container, so
# a unit test that only inspects argv (as `tests/unit/test_sandbox.py` does
# for the rest of this module) never actually exercises them.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_wrong_mode_credential_fails_closed_before_any_agent_container(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A credential source at the wrong mode must fail closed, not silently.

    ADR verification requirement 4 names "wrong-mode source material" as one
    of the cases that "must prevent the agent from starting with a stable,
    redacted error. This is mandatory." `_seed_command` enforces this with
    ``test $(stat -c %a <source>) = 600``; this test proves that guard
    actually runs against a real Docker daemon, not just that the argv
    contains the check.
    """
    fixture_image = _fixture_image(backend)
    _require_docker_fixture(fixture_image)

    source_rel, _target_rel = AUTH_CREDENTIAL_MANIFEST[backend][0]
    secret_marker = f"SECRET-MARKER-{uuid.uuid4().hex}"
    credential = json.dumps({"marker": secret_marker, "backend": backend})

    # 0644, not the required 0600 -- the guard `_seed_command` enforces.
    login_volume = _make_synthetic_login_volume(
        backend, credential, source_rel, image=fixture_image, mode="644"
    )
    monkeypatch.setattr(sandbox_module, "sandbox_auth_volume_name", lambda _backend: login_volume)

    observed_docker_args: list[list[str]] = []
    original_docker_args = sandbox_module._docker_args

    def _spy(*args: object, **kwargs: object) -> list[str]:
        built = original_docker_args(*args, **kwargs)  # type: ignore[arg-type]
        observed_docker_args.append(built)
        return built

    monkeypatch.setattr(sandbox_module, "_docker_args", _spy)

    source = tmp_path / "candidate"
    source.mkdir()

    try:
        with pytest.raises(RuntimeError) as excinfo:
            run_sandboxed_command(
                ["sh", "-c", "echo should-not-run"],
                cwd=source,
                env={},
                sandbox=SandboxConfig(enabled=True, auth="volume", network="none"),
                scope="agent",
                sync_back=False,
                image=fixture_image,
                agent_backend=backend,
            )

        message = str(excinfo.value)
        assert "credential seed failed" in message
        # The message names *this* cause, not the generic one and not the
        # cause of a different guard.
        assert "not a private, regular" in message
        assert "holds no credential" not in message
        assert secret_marker not in message
        assert secret_marker not in repr(excinfo.value)
        cause = excinfo.value.__cause__
        if isinstance(cause, subprocess.CalledProcessError):
            assert secret_marker not in (cause.stdout or "")
            assert secret_marker not in (cause.stderr or "")

        # The seed step raised before `run_sandboxed_commands`'s command loop
        # ever reaches `_docker_args` -- no agent container was created.
        assert observed_docker_args == []
    finally:
        _force_remove_volume(login_volume)


@pytest.mark.parametrize("backend", _BACKENDS)
def test_malformed_json_credential_fails_closed_before_any_agent_container(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Non-JSON credential content must fail closed, not silently.

    ADR verification requirement 4 names "malformed source material" as one
    of the cases that "must prevent the agent from starting with a stable,
    redacted error. This is mandatory." `_seed_command` enforces this with a
    ``python -c`` step that parses every manifest entry as JSON; this test
    proves that guard actually runs against a real Docker daemon.
    """
    fixture_image = _fixture_image(backend)
    _require_docker_fixture(fixture_image)

    source_rel, _target_rel = AUTH_CREDENTIAL_MANIFEST[backend][0]
    secret_marker = f"SECRET-MARKER-{uuid.uuid4().hex}"
    non_json_content = f"not-json-at-all::{secret_marker}"

    # Correct mode, but content that fails `python -c ... json.loads(...)`.
    login_volume = _make_synthetic_login_volume(
        backend, non_json_content, source_rel, image=fixture_image, mode="600"
    )
    monkeypatch.setattr(sandbox_module, "sandbox_auth_volume_name", lambda _backend: login_volume)

    observed_docker_args: list[list[str]] = []
    original_docker_args = sandbox_module._docker_args

    def _spy(*args: object, **kwargs: object) -> list[str]:
        built = original_docker_args(*args, **kwargs)  # type: ignore[arg-type]
        observed_docker_args.append(built)
        return built

    monkeypatch.setattr(sandbox_module, "_docker_args", _spy)

    source = tmp_path / "candidate"
    source.mkdir()

    try:
        with pytest.raises(RuntimeError) as excinfo:
            run_sandboxed_command(
                ["sh", "-c", "echo should-not-run"],
                cwd=source,
                env={},
                sandbox=SandboxConfig(enabled=True, auth="volume", network="none"),
                scope="agent",
                sync_back=False,
                image=fixture_image,
                agent_backend=backend,
            )

        message = str(excinfo.value)
        assert "credential seed failed" in message
        assert "not a readable credential record" in message
        assert "holds no credential" not in message
        assert secret_marker not in message
        assert secret_marker not in repr(excinfo.value)
        cause = excinfo.value.__cause__
        if isinstance(cause, subprocess.CalledProcessError):
            assert secret_marker not in (cause.stdout or "")
            assert secret_marker not in (cause.stderr or "")

        assert observed_docker_args == []
    finally:
        _force_remove_volume(login_volume)


# --------------------------------------------------------------------------
# TEST 5 -- an empty login volume names itself
#
# The most likely real-world seeding failure is not a corrupt credential: it
# is a login volume nobody has signed in yet. Signing in on the host does not
# sign the sandbox in, and the two are easy to confuse, so the error has to
# say which one is missing and how to fix it.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_empty_login_volume_names_its_cause_and_the_command_that_fixes_it(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_image = _fixture_image(backend)
    _require_docker_fixture(fixture_image)

    # A login volume that exists but holds no credential at all -- exactly
    # what a never-signed-in or signed-out backend looks like.
    login_volume = f"helix-test-synthetic-login-{backend}-{uuid.uuid4().hex}"
    _run_docker(["docker", "volume", "create", login_volume])
    monkeypatch.setattr(
        sandbox_module, "sandbox_auth_volume_name", lambda _backend: login_volume
    )

    observed_docker_args: list[list[str]] = []
    original_docker_args = sandbox_module._docker_args

    def _spy(*args: object, **kwargs: object) -> list[str]:
        built = original_docker_args(*args, **kwargs)  # type: ignore[arg-type]
        observed_docker_args.append(built)
        return built

    monkeypatch.setattr(sandbox_module, "_docker_args", _spy)

    source = tmp_path / "candidate"
    source.mkdir()

    try:
        with pytest.raises(RuntimeError) as excinfo:
            run_sandboxed_command(
                ["sh", "-c", "echo should-not-run"],
                cwd=source,
                env={},
                sandbox=SandboxConfig(enabled=True, auth="volume", network="none"),
                scope="agent",
                sync_back=False,
                image=fixture_image,
                agent_backend=backend,
            )

        message = str(excinfo.value)
        # The failure must name *this* cause, and must not be confusable with
        # the malformed or wrong-mode cases the tests above cover.
        assert "holds no credential" in message
        assert f"helix sandbox login {backend}" in message
        assert "not a readable credential record" not in message
        assert "not a private, regular" not in message
        # No agent container was created.
        assert observed_docker_args == []
    finally:
        _force_remove_volume(login_volume)


# --------------------------------------------------------------------------
# TEST 6 -- the cursor CLI reads the credential from the path we mount it at
#
# `cursor-agent status` grades a *local parse*: it reads the credential the
# CLI's own resolver points at and reports whether an access/refresh pair is
# there. Under `--network none` it cannot and does not check the grant with
# any service, so this test establishes that HELIX mounts the credential
# where the CLI looks -- not that any grant is valid.
#
# The negative half matters as much as the positive half. A test that only
# asserted "the signed-out message is absent" would pass just as well if no
# credential reached the container at all, so this asserts the credential is
# present at the exact path the CLI reads, asserts the CLI reports the
# signed-in message positively, and runs the CLI a second time against the
# directory the manifest used to point at to show the check can tell the two
# apart.
# --------------------------------------------------------------------------


_CURSOR_SIGNED_IN = "Login successful!"
_CURSOR_SIGNED_OUT = "Not logged in"

# What the manifest said before it was read out of the CLI bundle: the
# settings file, in the settings directory. Kept here only as the control
# arm of the test below.
_CURSOR_SETTINGS_ENTRY = (".cursor/cli-config.json", "cli-config.json")
_CURSOR_SETTINGS_DESTINATION = "/home/node/.cursor"


def test_cursor_cli_reads_the_credential_from_the_path_helix_mounts_it_at(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = "cursor"
    fixture_image = _fixture_image(backend)
    _require_docker_fixture(fixture_image)

    source_rel, target_rel = AUTH_CREDENTIAL_MANIFEST[backend][0]
    destination = AUTH_MOUNT_DESTINATIONS[backend]
    credential = _fake_credential(backend)

    def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
        workspace = tmp_path / f"candidate-{uuid.uuid4().hex[:8]}"
        workspace.mkdir()
        return run_sandboxed_command(
            command,
            cwd=workspace,
            env={},
            sandbox=SandboxConfig(enabled=True, auth="volume", network="none"),
            scope="agent",
            sync_back=False,
            image=fixture_image,
            agent_backend=backend,
        )

    login_volume = _make_synthetic_login_volume(
        backend, credential, source_rel, image=fixture_image
    )
    monkeypatch.setattr(
        sandbox_module, "sandbox_auth_volume_name", lambda _backend: login_volume
    )

    try:
        # 1. The credential is at the exact path, mode, and ownership the CLI
        #    needs -- asserted directly, not inferred from the CLI's verdict.
        credential_path = f"{destination}/{target_rel}"
        probe = _run(
            [
                "sh",
                "-c",
                (
                    f"cat {shlex.quote(credential_path)}; echo; "
                    f"echo MODE:$(stat -c '%a %u:%g' {shlex.quote(credential_path)}); "
                    # The corrected destination sits two components below the
                    # tmpfs $HOME, so Docker would synthesise its parent as
                    # root-owned unless HELIX pre-owns it.
                    "echo PARENT:$(stat -c '%a %u:%g' /home/node/.config)"
                ),
            ]
        )
        assert probe.returncode == 0, probe.stderr
        probe_lines = probe.stdout.splitlines()
        assert probe_lines[0] == credential, "credential is not readable at the mount destination"
        assert probe_lines[1] == "MODE:600 1000:1000"
        assert probe_lines[2] == "PARENT:700 1000:1000", (
            "the credential directory's parent is not owned by the agent -- "
            "Docker synthesised it as root"
        )

        # 2. The CLI itself, handed that credential, reports signed in.
        status = _run(["cursor-agent", "status"])
        assert _CURSOR_SIGNED_IN in status.stdout, (
            f"cursor-agent did not find the seeded credential: {status.stdout!r} "
            f"{status.stderr!r}"
        )
        assert _CURSOR_SIGNED_OUT not in status.stdout

        # 3. The control. Point HELIX at the CLI's *settings* file in the
        #    settings directory -- the pre-correction manifest -- and the same
        #    check reports signed out, so step 2 is not vacuous.
        settings_login_volume = _make_synthetic_login_volume(
            backend,
            json.dumps({"version": 1, "editor": {"vimMode": False}}),
            _CURSOR_SETTINGS_ENTRY[0],
            image=fixture_image,
        )
        monkeypatch.setattr(
            sandbox_module,
            "sandbox_auth_volume_name",
            lambda _backend: settings_login_volume,
        )
        monkeypatch.setitem(
            sandbox_module.AUTH_CREDENTIAL_MANIFEST, backend, (_CURSOR_SETTINGS_ENTRY,)
        )
        monkeypatch.setitem(
            sandbox_module.AUTH_MOUNT_DESTINATIONS, backend, _CURSOR_SETTINGS_DESTINATION
        )
        try:
            control = _run(["cursor-agent", "status"])
            assert _CURSOR_SIGNED_OUT in control.stdout, (
                "the settings file was accepted as a credential -- this check "
                f"cannot tell the two apart: {control.stdout!r}"
            )
        finally:
            _force_remove_volume(settings_login_volume)
    finally:
        _force_remove_volume(login_volume)
