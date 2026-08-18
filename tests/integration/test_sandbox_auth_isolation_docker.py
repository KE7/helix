"""Real-Docker proof that per-candidate auth volumes actually isolate credentials.

``tests/unit/test_sandbox.py`` proves HELIX's auth-volume design *by
construction*: it patches ``helix.sandbox.subprocess.run`` and inspects the
argv HELIX builds, so no container ever runs and no volume is ever created.
That is necessary but not sufficient -- the credential-rotation incident this
design exists to prevent was an *observed* runtime failure (a shared
credential mutated in place under concurrent use), not a static argv defect.

This module closes that gap using a real Docker daemon and HELIX's own
volume-creation/seeding functions (``_create_candidate_auth_volume``,
``_seed_candidate_auth_volume``, ``run_sandboxed_command``) -- never a
hand-rolled reimplementation of the copy/mount logic under test.

Both tests below are parametrized over *every* backend key in
``helix.sandbox.AUTH_CREDENTIAL_MANIFEST`` -- not just ``claude`` -- because
each backend's manifest entry (a distinct source path, target filename, and
mount destination) is a place the copy/mount logic can fail independently.
This resolves review finding #5 against
``docs/design/sandbox-auth-projection.md``: that document marks cursor,
gemini, and opencode as UNVERIFIED and states that a backend is not enabled
for ``auth = "volume"`` until its manifest is confirmed by a real test. The
parametrization is driven directly off ``AUTH_CREDENTIAL_MANIFEST`` (backend
list) and ``AUTH_MOUNT_DESTINATIONS`` (looked up by key, never copied into a
second hardcoded list), so a future manifest entry is covered automatically
and a backend present in one dict but missing from the other fails with a
loud ``KeyError`` rather than being silently skipped.

Credential safety
------------------
Every credential used here is synthetic. The tests never read, mount, or
reference an operator's real ``~/.claude``, ``~/.codex``, ``~/.cursor``,
``~/.gemini``, or ``~/.local/share/opencode``, and never touch the real
``helix-auth-<backend>`` volumes: ``helix.sandbox.sandbox_auth_volume_name``
is monkeypatched for the duration of each test to point at a throwaway
``helix-test-synthetic-login-*`` volume instead. That is the sole seam
touched; every other code path under test runs unmodified.

Credential shape: only ``claude``'s real login-volume shape
(``claudeAiOauth`` with an OAuth token pair) is reflected in the fake
fixture below, matching what earlier claude-only auth-volume work already
established. The real on-disk shape of ``codex``'s ``auth.json``,
``cursor``'s ``cli-config.json``, ``gemini``'s ``oauth_creds.json``, and
``opencode``'s ``auth.json`` was **not** determined for this change -- doing
so would require reading an operator's real credential file, which is
exactly what this module must never do. Those four backends instead get a
minimal, valid, non-empty JSON object (``_GENERIC_SYNTHETIC_SHAPE`` below).
This is a deliberate substitution, not an oversight: everything these tests
assert -- byte-for-byte isolation across candidate volumes, mode/ownership,
manifest-only contents, and that a write to one candidate never reaches
another candidate or the login volume -- depends on the manifest's
source/target *paths* and on file *bytes* being preserved unchanged, not on
the credential JSON's internal schema. A schema-accurate fixture would not
change what byte-isolation demonstrates; it would only matter for a test
that also validates backend-specific JSON semantics, which none of these
do.

Convention: follows ``docker_integration`` (see ``pyproject.toml`` markers)
and the ``_require_docker_fixture``-style skip used historically for
Docker-gated tests in this repo -- skip cleanly when the daemon or fixture
image is unavailable rather than fail CI/a laptop without Docker.
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

# ``claude``'s real login-volume shape (verified, not read from an operator
# credential -- this shape is already established by prior claude-only auth
# work). Every other backend's real shape is unknown here by design; see the
# module docstring's "Credential shape" section.
_GENERIC_SYNTHETIC_SHAPE_NOTE = (
    "real on-disk shape unknown -- placeholder for byte-isolation testing only, "
    "see test_sandbox_auth_isolation_docker.py module docstring"
)


def _fake_credential(backend: str) -> str:
    """A synthetic, backend-shaped-where-known seed credential. All fake."""
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

    Set ``HELIX_DOCKER_TESTS_STRICT=1`` to turn every skip into a failure.
    CI sets it: a Docker job that silently skips every test still reports
    green, which is worse than not running it at all.
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
    backend: str, credential_json: str, source_rel: str, *, image: str
) -> str:
    """Create a throwaway, HELIX-shaped login volume holding a fake credential.

    This mimics the *shape* the real login volume has after a real login (a
    manifest-declared relative path, mode 0600, uid/gid 1000) but is a
    brand-new randomly-named volume -- it is never ``helix-auth-<backend>``
    and never derived from any real credential.
    """
    name = f"helix-test-synthetic-login-{backend}-{uuid.uuid4().hex}"
    _run_docker(["docker", "volume", "create", name])
    container_path = f"/home/node/{source_rel}"
    parent = str(Path(container_path).parent)
    script = (
        f"set -eu; mkdir -p {shlex.quote(parent)}; "
        f"printf '%s' {shlex.quote(credential_json)} > {shlex.quote(container_path)}; "
        f"chmod 600 {shlex.quote(container_path)}; "
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

        # 4: THE assertion that matters most. Write a different "rotated"
        # credential into candidate A's own copy (simulating a backend
        # refreshing its token in place, exactly as observed in the
        # incident), then assert candidate B is byte-unchanged.
        _write_credential(candidate_a.name, target_rel, rotated_credential, image=fixture_image)
        rotated_content_a, _ = _read_credential(candidate_a.name, target_rel, image=fixture_image)
        assert rotated_content_a == rotated_credential

        unaffected_content_b, unaffected_mode_b = _read_credential(
            candidate_b.name, target_rel, image=fixture_image
        )
        assert unaffected_content_b == fake_credential, (
            "candidate B's credential changed after candidate A rotated its "
            "own copy -- this is exactly the shared-volume rotation incident "
            "this design exists to prevent"
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

        # THE assertion that matters most, replayed through the real E2E
        # entry point: the in-container "rotation" write landed on the
        # candidate's private copy, not on the synthetic login volume.
        login_content, _ = _read_credential(
            login_volume, source_rel, mount="/home/node", image=fixture_image
        )
        assert login_content == fake_credential, (
            "the login volume was mutated by a write made inside the agent "
            "container -- credentials are not isolated"
        )
    finally:
        _force_remove_volume(login_volume)
