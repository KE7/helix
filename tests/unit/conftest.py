"""Unit-test safety guards.

The guard here exists because of a real incident, not a hypothetical one.

``preflight_auth`` performs a REAL authenticated operation against the REAL
backend auth volume at ``:rw`` — that is deliberate and correct in production
(see ``helix/authpreflight.py``: probing a copy or mounting ``:ro`` produces a
silently wrong result).  But it means any unit test that reaches that code
path with Docker available will start a container against the shared
``helix-auth-*`` volume and attempt a token refresh with the operator's real
refresh token.  A *successful* refresh rotates that token, which would
invalidate the shared credential for every lane.

Wiring the preflight into ``run_evolution`` made exactly that reachable from
``tests/unit/test_evolution.py``, whose fixture mocks the sidecar but not
``subprocess``.  Mocking that one test would have left the same landmine for
the next test someone writes.

So the guard is structural: unit tests may not execute Docker at all.  Any
attempt fails loudly and names the offending argv.  Tests that need to observe
docker argv must mock ``subprocess.run`` (most already do) or pass an injected
runner; tests that genuinely need a container belong in the integration tier,
where disposable volumes with synthetic credentials are used.
"""

from __future__ import annotations

import subprocess

import pytest


_REAL_RUN = subprocess.run
_REAL_POPEN = subprocess.Popen


def _looks_like_docker(args: object) -> bool:
    if isinstance(args, str):
        return args.strip().startswith("docker")
    if isinstance(args, (list, tuple)) and args:
        first = args[0]
        return isinstance(first, str) and first.rsplit("/", 1)[-1] == "docker"
    return False


class _DockerInUnitTest(RuntimeError):
    """Raised when a unit test tries to start a real container."""


class _SharedAuthVolumeTouched(RuntimeError):
    """Raised when any test names a shared ``helix-auth-*`` volume for real."""


def _names_shared_auth_volume(args: object) -> str | None:
    """Return the shared auth volume named in ``args``, if any.

    Belt-and-braces against the general Docker block: even a command that did
    not look like ``docker`` to the check above must never reach one of the
    real credential volumes.  This mirrors the refusal rule the synthetic
    refresh harness already adopts — a test refuses to run at all if its
    target volume name matches ``helix-auth-*``.
    """
    items = args if isinstance(args, (list, tuple)) else [args]
    for item in items:
        if not isinstance(item, str):
            continue
        for token in item.replace(":", " ").replace("=", " ").split():
            if token.startswith("helix-auth-"):
                return token
    return None


def _describe(args: object) -> str:
    if isinstance(args, (list, tuple)):
        return " ".join(str(a) for a in args)
    return str(args)


@pytest.fixture(autouse=True)
def _no_real_docker_in_unit_tests(monkeypatch: pytest.MonkeyPatch):
    """Fail loudly if a unit test invokes Docker for real.

    This is a safety control, not a tidiness rule: the auth preflight's probe
    mounts the shared credential volume ``:rw`` and can rotate a real token.
    """

    def guarded_run(args, *a, **kw):  # type: ignore[no-untyped-def]
        if volume := _names_shared_auth_volume(args):
            raise _SharedAuthVolumeTouched(
                f"a test named the SHARED auth volume {volume!r} in a real "
                f"subprocess:\n    {_describe(args)}\n"
                "Tests must never touch helix-auth-* volumes. Re-authenticating "
                "or refreshing shared state rotates the stored token and "
                "invalidates it for every lane. Use a DISPOSABLE volume with "
                "SYNTHETIC credentials."
            )
        if _looks_like_docker(args):
            raise _DockerInUnitTest(
                "a unit test attempted to run Docker for real:\n"
                f"    {_describe(args)}\n"
                "Unit tests must not start containers. The auth preflight "
                "probes the SHARED helix-auth-* volume at :rw with a real "
                "authenticated request, and a successful refresh ROTATES the "
                "stored token — invalidating that volume for every lane.\n"
                "Fix: mock subprocess.run / the injected docker runner, or "
                "mock helix.evolution.preflight_auth. Tests that genuinely "
                "need a container belong in the integration tier and must use "
                "a DISPOSABLE volume with SYNTHETIC credentials."
            )
        return _REAL_RUN(args, *a, **kw)

    def guarded_popen(args, *a, **kw):  # type: ignore[no-untyped-def]
        if _looks_like_docker(args):
            raise _DockerInUnitTest(
                f"a unit test attempted to Popen Docker for real: "
                f"{_describe(args)}"
            )
        return _REAL_POPEN(args, *a, **kw)

    monkeypatch.setattr(subprocess, "run", guarded_run)
    monkeypatch.setattr(subprocess, "Popen", guarded_popen)
    yield
