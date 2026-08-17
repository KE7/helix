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

Credential safety
------------------
Every credential used here is synthetic (``fake-not-a-real-token``, shaped
like ``AUTH_CREDENTIAL_MANIFEST["claude"]`` expects). The tests never read,
mount, or reference an operator's real ``~/.claude`` et al., and never touch
the real ``helix-auth-<backend>`` volumes: ``helix.sandbox.sandbox_auth_volume_name``
is monkeypatched for the duration of each test to point at a throwaway
``helix-test-synthetic-login-*`` volume instead. That is the sole seam
touched; every other code path under test runs unmodified.

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
    _SEED_IMAGE,
    run_sandboxed_command,
)


pytestmark = pytest.mark.docker_integration

_BACKEND = "claude"
_FIXTURE_IMAGE = os.environ.get("HELIX_DOCKER_TEST_IMAGE", DEFAULT_BACKEND_IMAGES[_BACKEND])
_REAL_LOGIN_VOLUME_NAME = f"helix-auth-{_BACKEND}"  # never created/read/written below

_SOURCE_REL, _TARGET_REL = AUTH_CREDENTIAL_MANIFEST[_BACKEND][0]
_MOUNT_DEST = AUTH_MOUNT_DESTINATIONS[_BACKEND]

_FAKE_CREDENTIAL = json.dumps(
    {
        "claudeAiOauth": {
            "accessToken": "fake-not-a-real-token",
            "refreshToken": "fake",
            "expiresAt": 9999999999,
            "scopes": ["user:inference"],
        }
    }
)
_ROTATED_CREDENTIAL = json.dumps(
    {
        "claudeAiOauth": {
            "accessToken": "fake-rotated-token-from-candidate-a",
            "refreshToken": "fake-rotated",
            "expiresAt": 9999999999,
            "scopes": ["user:inference"],
        }
    }
)


def _docker(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], check=check, capture_output=True, text=True, timeout=30
    )


def _require_docker_fixture() -> None:
    try:
        daemon = _docker("info")
        image = _docker("image", "inspect", _FIXTURE_IMAGE)
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.skip(f"Docker daemon unavailable: {exc}")
    if daemon.returncode != 0:
        pytest.skip(f"Docker daemon unavailable: {daemon.stderr.strip()}")
    if image.returncode != 0:
        pytest.skip(f"fixture image {_FIXTURE_IMAGE!r} is not installed locally")


def _diagnostic_docker_args(
    volume_name: str, mount: str, script: str, *, writable: bool
) -> list[str]:
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
        _SEED_IMAGE,
        "sh",
        "-c",
        script,
    ]


def _make_synthetic_login_volume(backend: str, credential_json: str) -> str:
    """Create a throwaway, HELIX-shaped login volume holding a fake credential.

    This mimics the *shape* the real login volume has after ``claude auth
    login`` (manifest-declared relative path, mode 0600, uid/gid 1000) but is
    a brand-new randomly-named volume -- it is never ``helix-auth-<backend>``
    and never derived from any real credential.
    """
    name = f"helix-test-synthetic-login-{backend}-{uuid.uuid4().hex}"
    _run_docker(["docker", "volume", "create", name])
    container_path = f"/home/node/{_SOURCE_REL}"
    parent = str(Path(container_path).parent)
    script = (
        f"set -eu; mkdir -p {shlex.quote(parent)}; "
        f"printf '%s' {shlex.quote(credential_json)} > {shlex.quote(container_path)}; "
        f"chmod 600 {shlex.quote(container_path)}; "
        "chown -R 1000:1000 /home/node"
    )
    _run_docker(_diagnostic_docker_args(name, "/home/node", script, writable=True))
    return name


def _read_credential(volume_name: str, rel_path: str, *, mount: str = "/check") -> tuple[str, str]:
    """Return ``(file content, "mode uid:gid")`` read from inside *volume_name*."""
    path = f"{mount}/{rel_path}"
    script = (
        f"cat {shlex.quote(path)}; "
        "printf '\\n===MODE===\\n'; "
        f"stat -c '%a %u:%g' {shlex.quote(path)}"
    )
    result = _run_docker(_diagnostic_docker_args(volume_name, mount, script, writable=False))
    content, mode = result.stdout.split("\n===MODE===\n")
    return content, mode.strip()


def _write_credential(volume_name: str, rel_path: str, content: str, *, mount: str = "/check") -> None:
    path = f"{mount}/{rel_path}"
    script = f"printf '%s' {shlex.quote(content)} > {shlex.quote(path)}"
    _run_docker(_diagnostic_docker_args(volume_name, mount, script, writable=True))


def _list_top_level(volume_name: str, *, mount: str = "/check") -> set[str]:
    result = _run_docker(
        _diagnostic_docker_args(volume_name, mount, f"ls -a {mount}", writable=False)
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
# TEST 1 -- runtime isolation via HELIX's own volume-creation/seeding code
# --------------------------------------------------------------------------


def test_candidate_auth_volumes_isolate_synthetic_credential_under_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_docker_fixture()

    login_volume = _make_synthetic_login_volume(_BACKEND, _FAKE_CREDENTIAL)

    # The only seam we touch: redirect the seeding step's *source* volume
    # away from the real "helix-auth-claude" toward our synthetic fixture.
    # `_seed_candidate_auth_volume` looks this up by calling
    # `sandbox_auth_volume_name(volume.backend)` from `helix.sandbox`'s own
    # module globals, so patching the module attribute here also redirects
    # the call made from inside `_seed_candidate_auth_volume` itself.
    monkeypatch.setattr(sandbox_module, "sandbox_auth_volume_name", lambda _backend: login_volume)

    candidate_a: CandidateAuthVolume | None = None
    candidate_b: CandidateAuthVolume | None = None
    try:
        # 1 & 2: real HELIX volume-creation + seeding code, called directly.
        candidate_a = _create_candidate_auth_volume(_BACKEND)
        candidate_b = _create_candidate_auth_volume(_BACKEND)
        assert candidate_a.name != candidate_b.name
        assert candidate_a.name != login_volume
        assert candidate_b.name != login_volume
        assert _REAL_LOGIN_VOLUME_NAME not in (candidate_a.name, candidate_b.name, login_volume)

        _seed_candidate_auth_volume(candidate_a)
        _seed_candidate_auth_volume(candidate_b)

        # 3: both candidates contain the credential, mode 0600, owned 1000:1000.
        content_a, mode_a = _read_credential(candidate_a.name, _TARGET_REL)
        content_b, mode_b = _read_credential(candidate_b.name, _TARGET_REL)
        assert content_a == _FAKE_CREDENTIAL
        assert content_b == _FAKE_CREDENTIAL
        assert mode_a == "600 1000:1000"
        assert mode_b == "600 1000:1000"

        # 4: THE assertion that matters most. Write a different "rotated"
        # credential into candidate A's own copy (simulating a backend
        # refreshing its token in place, exactly as observed in the
        # incident), then assert candidate B is byte-unchanged.
        _write_credential(candidate_a.name, _TARGET_REL, _ROTATED_CREDENTIAL)
        rotated_content_a, _ = _read_credential(candidate_a.name, _TARGET_REL)
        assert rotated_content_a == _ROTATED_CREDENTIAL

        unaffected_content_b, unaffected_mode_b = _read_credential(candidate_b.name, _TARGET_REL)
        assert unaffected_content_b == _FAKE_CREDENTIAL, (
            "candidate B's credential changed after candidate A rotated its "
            "own copy -- this is exactly the shared-volume rotation incident "
            "this design exists to prevent"
        )
        assert unaffected_mode_b == "600 1000:1000"

        # The synthetic login volume itself is also untouched by A's write.
        login_content, _ = _read_credential(login_volume, _SOURCE_REL, mount="/home/node")
        assert login_content == _FAKE_CREDENTIAL

        # 5: candidate volumes only contain the manifest-declared targets --
        # no trace of the login volume's own nested layout leaks through.
        top_level_a = _list_top_level(candidate_a.name)
        top_level_b = _list_top_level(candidate_b.name)
        assert ".claude" not in top_level_a  # source's directory name, not the target's
        assert ".claude" not in top_level_b
        assert top_level_a == {".credentials.json", "projects"}
        assert top_level_b == {".credentials.json", "projects"}

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


def test_e2e_sandboxed_agent_command_sees_isolated_seeded_credential(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _require_docker_fixture()

    login_volume = _make_synthetic_login_volume(_BACKEND, _FAKE_CREDENTIAL)
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
    dest_path = f"{_MOUNT_DEST}/{_TARGET_REL}"

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
            f"printf '%s' {shlex.quote(_ROTATED_CREDENTIAL)} > {shlex.quote(dest_path)}"
        ),
    ]

    result = run_sandboxed_command(
        command,
        cwd=source,
        env={},
        sandbox=SandboxConfig(enabled=True, auth="volume", network="none"),
        scope="agent",
        sync_back=False,
        image=_FIXTURE_IMAGE,
        agent_backend=_BACKEND,
    )

    try:
        assert result.returncode == 0, f"sandboxed command failed: {result.stderr}"
        lines = result.stdout.splitlines()
        assert lines[0] == _FAKE_CREDENTIAL, "seeded credential not present/readable at mount dest"
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
        assert not any(_REAL_LOGIN_VOLUME_NAME in item for item in agent_call)
        mount_arg = next(
            item for item in agent_call if item.endswith(f":{_MOUNT_DEST}:rw")
        )
        candidate_volume_in_argv = mount_arg.split(":", 1)[0]
        assert candidate_volume_in_argv.startswith("helix-candidate-auth-claude-")

        # The candidate volume the agent actually used is gone once
        # `run_sandboxed_command` returns (its own cleanup path ran).
        assert not _volume_exists(candidate_volume_in_argv)

        # THE assertion that matters most, replayed through the real E2E
        # entry point: the in-container "rotation" write landed on the
        # candidate's private copy, not on the synthetic login volume.
        login_content, _ = _read_credential(login_volume, _SOURCE_REL, mount="/home/node")
        assert login_content == _FAKE_CREDENTIAL, (
            "the login volume was mutated by a write made inside the agent "
            "container -- credentials are not isolated"
        )
    finally:
        _force_remove_volume(login_volume)
