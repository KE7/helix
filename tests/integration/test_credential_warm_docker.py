"""Container proof that the credential warm is real, and that it is free.

The warm exists to move a due token refresh out of N racing candidates and into
one writer.  Two claims have to hold against a real backend container or the
change is worse than useless:

(a) it costs nothing -- no model call, no quota, on an operator's paid account
    that would otherwise be charged once per generation; and
(b) it does nothing at all when the credential is already fresh.

Both are asserted here by running the registered warm command with
``--network none``.  A command that completes with no network cannot have
reached a model; a warm that leaves the volume byte-identical did no work.

Credentials are synthetic and live in throwaway volumes (see ``conftest``); no
test logs in, and none can reach a real ``helix-auth-*`` volume.
"""

from __future__ import annotations

import subprocess

import pytest

from helix.backends import (
    BACKEND_AUTH_COMMANDS,
    BACKENDS,
    backend_credential_warm_skip_reason,
)


pytestmark = pytest.mark.docker_integration


CODEX_IMAGE = "ghcr.io/ke7/helix-evo-runner-codex:latest"

# A well-shaped credential that no real service would ever accept.  Only two
# things matter: that codex reads it as a stored ChatGPT login, and that its
# ``last_refresh`` is stamped now, so no refresh is due.  The id_token is an
# unsigned JWT over invented claims -- codex parses it for the account fields
# and nothing here is a real token or can rotate a real grant.
SYNTHETIC_CODEX_ID_TOKEN = (
    "eyJhbGciOiJub25lIiwidHlwIjoiSldUIn0.eyJzdWIiOiJTWU5USEVUSUMiLCJleHAiOjQxMDI0NDQ4MDAsImVtYWlsIjoic3ludGhldGljQGV4YW1wbGUuaW52YWxpZCIsImh0dHBzOi8vYXBpLm9wZW5haS5jb20vYXV0aCI6eyJjaGF0Z3B0X3BsYW5fdHlwZSI6InBsdXMiLCJjaGF0Z3B0X2FjY291bnRfaWQiOiJTWU5USEVUSUMiLCJjaGF0Z3B0X3VzZXJfaWQiOiJTWU5USEVUSUMiLCJ1c2VyX2lkIjoiU1lOVEhFVElDIn19.SYNTHETIC"
)

SYNTHETIC_FRESH_CODEX_AUTH = (
    "mkdir -p /home/node/.codex; "
    "NOW=$(date -u +%Y-%m-%dT%H:%M:%S.000000Z); "
    "printf '{\"OPENAI_API_KEY\":null,\"tokens\":{"
    "\"id_token\":\"%s\",\"access_token\":\"SYNTHETIC\","
    "\"refresh_token\":\"SYNTHETIC\",\"account_id\":\"SYNTHETIC\"},"
    "\"last_refresh\":\"%s\"}' "
    f"\"{SYNTHETIC_CODEX_ID_TOKEN}\" \"$NOW\" "
    "> /home/node/.codex/auth.json"
)


def _run_warm(*, image: str, volume: str, backend: str, timeout: int = 180):
    """Run the backend's registered warm command the way ``helix.sandbox`` does.

    Same single-writer shape as ``sandbox_auth_docker_args``: one container,
    the login volume read-write at ``/home/node``, nothing else attached.  The
    only deliberate difference is ``--network none``, which turns "this makes
    no model call" from a claim into something the container can prove.
    """
    args = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "node",
        "-v",
        f"{volume}:/home/node:rw",
        "-e",
        "HOME=/home/node",
        "-e",
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        image,
        *BACKEND_AUTH_COMMANDS[backend]["warm"],
    ]
    return subprocess.run(
        args, capture_output=True, text=True, check=False, timeout=timeout
    )


def _files(listing: set[str]) -> set[str]:
    """Keep only regular files from a ``volume_listing`` result.

    Directories are filtered out deliberately: the codex CLI scaffolds a few
    empty ones (``.codex/tmp``, ``.codex/memories``) the first time it starts
    in a fresh volume, and every candidate would create the same ones on its
    own.  What must not move is the credential and the files beside it.
    """
    return {line for line in listing if line.startswith("-")}


def _auth_digest(volume: str, image: str) -> str:
    """Hash the stored credential without reading it.

    The digest is the whole point: it proves the file did or did not change
    without anything ever printing, decoding, or logging its contents.
    """
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{volume}:/home/node:ro",
            "--user",
            "node",
            image,
            "sh",
            "-c",
            "sha256sum /home/node/.codex/auth.json | cut -d' ' -f1",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return result.stdout.strip()


@pytest.mark.timeout(300)
def test_codex_warm_is_free(require_image, throwaway_volume) -> None:
    """No model call, no quota -- proved by completing with no network at all.

    This is the constraint that decides whether the warm may ship: it runs on
    the operator's paid account once per generation, so a status/whoami command
    that quietly cost a request would be worse than the race it prevents.  A
    container with no network namespace cannot reach any endpoint, so a clean
    exit here is proof rather than assertion.
    """
    image = require_image(CODEX_IMAGE)
    volume = throwaway_volume(image, SYNTHETIC_FRESH_CODEX_AUTH)

    result = _run_warm(image=image, volume=volume, backend="codex")

    assert result.returncode == 0, result.stderr


@pytest.mark.timeout(300)
def test_codex_warm_is_a_no_op_on_a_fresh_credential(
    require_image, throwaway_volume, volume_listing
) -> None:
    """A credential that is not due for refresh must come back untouched.

    If the warm rewrote a fresh credential it would be doing the very thing it
    exists to prevent -- spending a single-use refresh token nobody needed to
    spend -- once per generation.
    """
    image = require_image(CODEX_IMAGE)
    volume = throwaway_volume(image, SYNTHETIC_FRESH_CODEX_AUTH)
    before_files = _files(volume_listing(volume, image))
    before_digest = _auth_digest(volume, image)

    _run_warm(image=image, volume=volume, backend="codex")

    assert _auth_digest(volume, image) == before_digest
    assert _files(volume_listing(volume, image)) == before_files


@pytest.mark.timeout(300)
def test_repeated_codex_warms_leave_no_residue(
    require_image, throwaway_volume, volume_listing
) -> None:
    """The warm runs every generation, so it must not accumulate.

    It also must not behave like a candidate: a command that opened a session
    or wrote a rollout into the shared volume would hand the next candidate
    state it did not create -- the contamination the per-candidate state work
    on this branch removes.
    """
    image = require_image(CODEX_IMAGE)
    volume = throwaway_volume(image, SYNTHETIC_FRESH_CODEX_AUTH)

    _run_warm(image=image, volume=volume, backend="codex")
    after_first = volume_listing(volume, image)
    digest_first = _auth_digest(volume, image)

    for _ in range(2):
        _run_warm(image=image, volume=volume, backend="codex")

    assert volume_listing(volume, image) == after_first
    assert _auth_digest(volume, image) == digest_first


@pytest.mark.timeout(120)
def test_skipped_backends_have_no_warm_command_to_run() -> None:
    """Skipping is a decision this suite is allowed to see.

    If a backend ever gains a warm command, it must gain container proof in
    this file at the same time -- this assertion is what makes that fail loudly
    instead of shipping an unmeasured per-generation call on a paid account.
    """
    warmed = [b for b in BACKENDS if backend_credential_warm_skip_reason(b) is None]
    assert warmed == ["codex"]
