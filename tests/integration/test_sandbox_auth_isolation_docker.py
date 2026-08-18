"""Real-Docker proof that per-candidate auth volumes actually isolate credentials.

``tests/unit/test_sandbox.py`` proves the design *by construction*: it patches
``helix.sandbox.subprocess.run`` and inspects the argv HELIX builds, so no
container ever runs and no volume is ever created. That is necessary but not
sufficient -- the credential-rotation incident this design exists to prevent
was an *observed* runtime failure (a shared credential mutated in place under
concurrent use), not a static argv defect.

This module closes that gap with a real Docker daemon and HELIX's own
functions (``_create_candidate_auth_volume``, ``_seed_candidate_auth_volume``,
``run_sandboxed_command``) -- never a reimplementation of the logic under
test. Every test is parametrized off ``AUTH_CREDENTIAL_MANIFEST`` and looks
``AUTH_MOUNT_DESTINATIONS`` up by key, so a new manifest entry is covered
automatically and a backend present in one dict but not the other fails
loudly rather than being silently skipped.

Credential safety
------------------
Every credential here is synthetic. The tests never read, mount, or reference
an operator's real ``~/.claude``, ``~/.codex``, ``~/.cursor``, ``~/.gemini``,
or ``~/.local/share/opencode``, and never touch the real
``helix-auth-<backend>`` volumes: ``helix.sandbox.sandbox_auth_volume_name``
is monkeypatched per test to a throwaway ``helix-test-synthetic-login-*``
volume. That is the sole seam touched; every other path runs unmodified.

Each fixture is written in that backend's *own* record shape
(``_CREDENTIAL_SHAPES`` below) with invented token values, because a generic
JSON blob is exactly what let a wrong manifest entry pass every byte-isolation
assertion in this file -- see
``test_backend_cli_self_reports_authenticated_from_the_seeded_manifest``.

Cost and quota: nothing here can spend. Every credential is synthetic and
every container runs ``--network none``; the commands used are each backend's
free local status command. ``gemini`` ships no status subcommand and is
asserted negatively instead (see ``_SELF_ATTESTATION``).

Convention: ``docker_integration`` (see ``pyproject.toml`` markers), skipping
cleanly when the daemon or fixture image is unavailable -- unless
``HELIX_DOCKER_TESTS_STRICT=1``, which turns every skip into a failure.
"""

from __future__ import annotations

import base64
import json
import os
import shlex
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

import helix.sandbox as sandbox_module
from helix.backends import DEFAULT_BACKEND_IMAGES
from helix.config import SandboxConfig
from helix.sandbox import (
    AUTH_CREDENTIAL_MANIFEST,
    AUTH_MOUNT_DESTINATIONS,
    AUTH_PRECREATED_DIRECTORIES,
    AUTH_SYNTHESIZED_FILES,
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

# --------------------------------------------------------------------------
# Synthetic credentials, in each backend's own record shape
# --------------------------------------------------------------------------
#
# Every token value below is invented. ``example.invalid`` is the reserved
# never-resolvable TLD (RFC 2606), and the marker string is threaded through
# each record so a seeded copy and a "rotated" copy are distinguishable by
# value while remaining structurally identical.


def _unsigned_jwt(claims: dict[str, object]) -> str:
    """A structurally valid, cryptographically worthless JWT.

    ``codex login status`` parses ``tokens.id_token`` before it reports
    anything, and rejects a malformed one with "invalid ID token format" --
    which is itself useful, since it proves the CLI is reading that exact
    file at that exact path. Three segments are required; the signature
    segment is never verified locally, so a constant placeholder suffices.
    """

    def _segment(payload: dict[str, object]) -> str:
        raw = json.dumps(payload, separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    return f"{_segment({'alg': 'none', 'typ': 'JWT'})}.{_segment(claims)}.bm90LWEtc2ln"


def _claude_credential(marker: str) -> str:
    return json.dumps(
        {
            "claudeAiOauth": {
                "accessToken": f"fake-not-a-real-token-{marker}",
                "refreshToken": f"fake-{marker}",
                "expiresAt": 9999999999999,
                "scopes": ["user:inference"],
                "subscriptionType": "max",
            }
        }
    )


def _codex_credential(marker: str) -> str:
    return json.dumps(
        {
            "OPENAI_API_KEY": None,
            "tokens": {
                "id_token": _unsigned_jwt(
                    {
                        "email": "synthetic@example.invalid",
                        "https://api.openai.com/auth": {
                            "chatgpt_plan_type": "plus",
                            "chatgpt_account_id": "00000000-0000-0000-0000-000000000000",
                        },
                        "exp": 9999999999,
                    }
                ),
                "access_token": f"fake-access-{marker}",
                "refresh_token": f"fake-refresh-{marker}",
                "account_id": "00000000-0000-0000-0000-000000000000",
            },
            "last_refresh": "2099-01-01T00:00:00.000000Z",
        }
    )


def _cursor_credential(marker: str) -> str:
    return json.dumps(
        {
            "accessToken": f"fake-access-{marker}",
            "refreshToken": f"fake-refresh-{marker}",
            "apiKey": f"fake-key-{marker}",
        }
    )


def _gemini_credential(marker: str) -> str:
    return json.dumps(
        {
            "access_token": f"fake-access-{marker}",
            "refresh_token": f"fake-refresh-{marker}",
            "scope": "https://www.googleapis.com/auth/cloud-platform",
            "token_type": "Bearer",
            "expiry_date": 9999999999999,
        }
    )


def _opencode_credential(marker: str) -> str:
    return json.dumps(
        {
            "anthropic": {
                "type": "oauth",
                "refresh": f"fake-refresh-{marker}",
                "access": f"fake-access-{marker}",
                "expires": 9999999999999,
            }
        }
    )


_CREDENTIAL_SHAPES: dict[str, Callable[[str], str]] = {
    "claude": _claude_credential,
    "codex": _codex_credential,
    "cursor": _cursor_credential,
    "gemini": _gemini_credential,
    "opencode": _opencode_credential,
}


def _fake_credential(backend: str) -> str:
    """A synthetic seed credential in *backend*'s own record shape."""
    return _CREDENTIAL_SHAPES[backend]("seed")


def _rotated_credential(backend: str) -> str:
    """A synthetic "rotated" value simulating an in-place token refresh."""
    return _CREDENTIAL_SHAPES[backend]("rotated-by-candidate-a")


# --------------------------------------------------------------------------
# Per-backend self-attestation: what the CLI itself says about the manifest
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _SelfAttestation:
    """A free, local command that makes a backend report its own auth state.

    *required* and *forbidden* are substrings of the command's combined
    stdout/stderr. Both are matched against the CLI's own words, never
    against credential content.
    """

    command: tuple[str, ...]
    required: tuple[str, ...]
    forbidden: tuple[str, ...]


_SELF_ATTESTATION: dict[str, _SelfAttestation] = {
    "claude": _SelfAttestation(
        command=("claude", "auth", "status"),
        required=('"loggedIn": true',),
        forbidden=('"loggedIn": false',),
    ),
    "codex": _SelfAttestation(
        command=("codex", "login", "status"),
        required=("Logged in",),
        forbidden=("Not logged in", "invalid ID token"),
    ),
    "cursor": _SelfAttestation(
        command=("cursor-agent", "status"),
        required=("Logged in",),
        forbidden=("Not logged in",),
    ),
    # The Gemini CLI ships no status subcommand (`gemini --help`), so this is
    # the negative form: the run must get *past* the auth-method gate. With a
    # synthetic token and `--network none` it then fails locally inside
    # `initOauthClient`, which is the expected, free, request-free outcome.
    "gemini": _SelfAttestation(
        command=("gemini", "--skip-trust", "-p", "ping"),
        required=(),
        forbidden=("Please set an Auth method",),
    ),
    "opencode": _SelfAttestation(
        command=("opencode", "auth", "list"),
        required=("1 credentials",),
        forbidden=("0 credentials",),
    ),
}


def _fixture_image(backend: str) -> str:
    """Resolve the runner image for *backend*, honoring a test-only override."""
    override = os.environ.get(f"HELIX_DOCKER_TEST_IMAGE_{backend.upper()}")
    return override or DEFAULT_BACKEND_IMAGES[backend]


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
    # Every manifest backend also has a self-attestation command below, so a
    # newly added backend cannot silently skip the one check that would catch
    # a wrong path. The optional tables are subsets, not parallel lists: a
    # backend needs a synthesised sibling or a pre-created directory only if
    # its CLI does.
    assert set(AUTH_CREDENTIAL_MANIFEST) == set(_SELF_ATTESTATION)
    assert set(AUTH_SYNTHESIZED_FILES) <= set(AUTH_CREDENTIAL_MANIFEST)
    assert set(AUTH_PRECREATED_DIRECTORIES) <= set(AUTH_CREDENTIAL_MANIFEST)


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
        # A candidate volume holds exactly what sandbox.py declares for the
        # backend and nothing else: the manifest's copied credentials, the
        # non-secret siblings the seeder synthesises, and the directories it
        # pre-creates. All three come from sandbox.py's own tables rather than
        # a second hardcoded list here, so a new declaration is covered
        # automatically and an undeclared stray file fails loudly.
        expected_top_level = {target for _, target in AUTH_CREDENTIAL_MANIFEST[backend]}
        expected_top_level |= {
            target for target, _ in AUTH_SYNTHESIZED_FILES.get(backend, ())
        }
        expected_top_level |= {
            directory.split("/", 1)[0]
            for directory in AUTH_PRECREATED_DIRECTORIES.get(backend, ())
        }
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


# --------------------------------------------------------------------------
# TEST 3 -- manifest self-attestation: the CLI itself confirms the manifest
# --------------------------------------------------------------------------


@pytest.mark.parametrize("backend", _BACKENDS)
def test_backend_cli_self_reports_authenticated_from_the_seeded_manifest(
    backend: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seed through HELIX's real path, then let the backend grade the manifest.

    The two tests above prove that whatever ``AUTH_CREDENTIAL_MANIFEST``
    names is copied intact, privately, to ``AUTH_MOUNT_DESTINATIONS``. They
    cannot prove the manifest names the right thing: a generic blob copied to
    a generic path satisfies every one of their assertions even when the CLI
    reads a completely different file. Two manifest entries were in fact
    wrong under exactly those passing tests -- cursor named its *settings*
    file (``.cursor/cli-config.json``) at the wrong destination, and gemini
    named a credential its CLI refuses without an auth-method sibling.

    So this test asks the only authority that can settle it: the backend's
    own CLI, running as the agent user, inside the candidate container, over
    the volume HELIX seeded. It needs no real grant, no network, and no
    quota (see the module docstring's "Cost and quota"), which is what makes
    it a CI-able regression guard rather than a manual ritual.
    """
    fixture_image = _fixture_image(backend)
    _require_docker_fixture(fixture_image)

    attestation = _SELF_ATTESTATION[backend]
    source_rel, _target_rel = AUTH_CREDENTIAL_MANIFEST[backend][0]
    fake_credential = _fake_credential(backend)

    login_volume = _make_synthetic_login_volume(
        backend, fake_credential, source_rel, image=fixture_image
    )
    monkeypatch.setattr(
        sandbox_module, "sandbox_auth_volume_name", lambda _backend: login_volume
    )

    workspace = tmp_path / "candidate"
    workspace.mkdir()

    try:
        result = run_sandboxed_command(
            list(attestation.command),
            cwd=workspace,
            env={},
            sandbox=SandboxConfig(
                enabled=True, auth="volume", network="none", timeout_seconds=300
            ),
            scope="agent",
            sync_back=False,
            image=fixture_image,
            agent_backend=backend,
        )
        # The CLIs disagree about exit codes for a reported-but-unusable
        # credential, so the contract is what they *say*, not what they
        # return. Both streams are searched because they disagree about that
        # too.
        output = f"{result.stdout}\n{result.stderr}"

        for needle in attestation.required:
            assert needle in output, (
                f"{backend}'s own CLI did not report itself authenticated from "
                f"the seeded manifest: expected {needle!r} in the output of "
                f"{' '.join(attestation.command)}. The credential was placed at "
                f"AUTH_CREDENTIAL_MANIFEST[{backend!r}] -> "
                f"{AUTH_MOUNT_DESTINATIONS[backend]}. Output was:\n{output}"
            )
        for needle in attestation.forbidden:
            assert needle not in output, (
                f"{backend}'s own CLI rejected the seeded manifest: found "
                f"{needle!r} in the output of {' '.join(attestation.command)}. "
                f"Output was:\n{output}"
            )
    finally:
        _force_remove_volume(login_volume)
