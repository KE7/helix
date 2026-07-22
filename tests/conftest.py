"""Root pytest conftest for HELIX.

Responsibilities:
- Ensure the HELIX TraceBus is disabled between tests (a test that forgets
  to exit ``TRACE.record()`` must not leak state into the next test).
- Prepend the GEPA differential-testing fixture root to ``sys.path`` so the
  new ``tests/unit/gepa_diff/`` package imports cleanly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make the diff-harness fixture package importable as a top-level module tree.
_DIFF_ROOT = Path(__file__).parent / "unit" / "gepa_diff"
if str(_DIFF_ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(_DIFF_ROOT.parent.parent))


@pytest.fixture(autouse=True)
def _trace_bus_clean_between_tests():
    """Guarantee ``TRACE.enabled`` is False and ``TRACE.events`` empty per test.

    Tests that use the diff harness opt into ``TRACE.record()`` explicitly;
    any leak (e.g. a raised exception inside ``record()``) would otherwise
    keep the bus enabled and contaminate subsequent tests.
    """
    try:
        from helix.trace import TRACE
    except Exception:
        # HELIX not importable in this environment — nothing to guard.
        yield
        return
    _prev_enabled = TRACE.enabled
    _prev_events = list(TRACE.events)
    try:
        yield
    finally:
        TRACE.enabled = False
        TRACE.events = []
        # Restore whatever the test harness set up pre-test (should be idle).
        TRACE.enabled = _prev_enabled if not TRACE.enabled else False
        if not TRACE.events:
            TRACE.events = _prev_events if _prev_events else []


# ---------------------------------------------------------------------------
# Docker / shared-credential safety guard (session-wide, default-deny)
# ---------------------------------------------------------------------------
#
# This guard exists because of a real breach, not a hypothetical one.
#
# ``helix.authpreflight.preflight_auth`` performs a REAL authenticated
# operation against the REAL backend auth volume, mounted ``:rw``.  That is
# deliberate and correct in production: probing a copy, or mounting ``:ro``,
# produces a silently WRONG result (a refresh rotates the server-side token,
# so a copy would absorb the rotation while the real volume kept the dead
# token; and ``:ro`` makes the refresh lockfile fail *silently*).
#
# The consequence is that any test reaching that path with Docker available
# starts a container against the shared ``helix-auth-*`` volume and attempts a
# token refresh with a real refresh token.  A SUCCESSFUL refresh ROTATES that
# token and invalidates the shared credential for every lane.
#
# Wiring the preflight into ``run_evolution`` made exactly that reachable from
# ``pytest tests/unit/``.  Classification of that event, recorded here so the
# guard's purpose is not softened over time: prohibition BREACHED; credential
# record UNCHANGED; non-credential shared-volume state MUTATED; OAuth POST
# CANNOT BE RULED OUT.
#
# Design of the guard:
#   * DEFAULT-DENY every real Docker invocation, across EVERY subprocess
#     alias, not just the one that happened to be used.
#   * HARD-DENY any ``helix-auth-*`` volume name with NO OVERRIDE, even for
#     opted-in integration tests. There is no legitimate test reason to touch
#     a shared credential volume; integration work uses DISPOSABLE volumes
#     with SYNTHETIC credentials.
#   * Real Docker is available only to tests explicitly marked
#     ``@pytest.mark.docker_integration`` AND only when
#     ``HELIX_ALLOW_DOCKER_TESTS=1`` is set in the environment.

_SHARED_AUTH_VOLUME_PREFIX = "helix-auth-"
_DOCKER_OPT_IN_ENV = "HELIX_ALLOW_DOCKER_TESTS"


class SharedAuthVolumeTouched(RuntimeError):
    """A test named a shared ``helix-auth-*`` volume. Never permitted."""


class RealDockerInTest(RuntimeError):
    """A test tried to invoke Docker without opting in."""


def _flatten_command(args: object) -> list[str]:
    if isinstance(args, str):
        return args.split()
    if isinstance(args, (list, tuple)):
        return [str(a) for a in args]
    return [str(args)]


def _describe(args: object) -> str:
    return " ".join(_flatten_command(args))


def _named_shared_auth_volume(args: object) -> str | None:
    """Return a shared auth volume named anywhere in ``args``, if present."""
    for item in _flatten_command(args):
        for token in item.replace(":", " ").replace("=", " ").replace(",", " ").split():
            if token.startswith(_SHARED_AUTH_VOLUME_PREFIX):
                return token
    return None


def _is_docker_command(args: object) -> bool:
    tokens = _flatten_command(args)
    if not tokens:
        return False
    exe = tokens[0].rsplit("/", 1)[-1]
    return exe in {"docker", "docker-compose", "podman", "nerdctl"} or (
        exe in {"sh", "bash", "sh.exe"} and any("docker" in t for t in tokens[1:])
    )


def _check_command(args: object, *, docker_allowed: bool) -> None:
    # HARD DENY: no override, not even under an opted-in integration test.
    if volume := _named_shared_auth_volume(args):
        raise SharedAuthVolumeTouched(
            f"a test named the SHARED auth volume {volume!r}:\n"
            f"    {_describe(args)}\n"
            "This denial has NO override. Touching a shared credential volume "
            "can trigger an OAuth refresh, and a successful refresh ROTATES "
            "the stored token, invalidating it for every lane. Use a "
            "DISPOSABLE volume with SYNTHETIC credentials."
        )
    if _is_docker_command(args) and not docker_allowed:
        raise RealDockerInTest(
            f"a test attempted to run Docker for real:\n    {_describe(args)}\n"
            "Real Docker is denied by default. Mock subprocess/the injected "
            "docker runner, or mark the test @pytest.mark.docker_integration "
            f"and set {_DOCKER_OPT_IN_ENV}=1. Integration tests must use "
            "DISPOSABLE volumes with SYNTHETIC credentials."
        )


def pytest_configure(config):  # noqa: D103
    config.addinivalue_line(
        "markers",
        "docker_integration: test may invoke real Docker when "
        f"{_DOCKER_OPT_IN_ENV}=1. Shared helix-auth-* volumes remain denied.",
    )


@pytest.fixture(autouse=True)
def _docker_safety_guard(request, monkeypatch):
    """Default-deny real Docker across every subprocess alias.

    Alias coverage is enumerated explicitly rather than relying on the one
    entry point that happened to be used, because partial coverage is
    indistinguishable from full coverage until the day it isn't.
    """
    import os
    import subprocess

    docker_allowed = bool(
        request.node.get_closest_marker("docker_integration")
    ) and os.environ.get(_DOCKER_OPT_IN_ENV) == "1"

    # subprocess.* aliases
    for name in ("run", "call", "check_call", "check_output", "Popen"):
        original = getattr(subprocess, name)

        def make(orig=original):
            def guarded(args, *a, **kw):
                _check_command(args, docker_allowed=docker_allowed)
                return orig(args, *a, **kw)

            return guarded

        monkeypatch.setattr(subprocess, name, make())

    for name in ("getoutput", "getstatusoutput"):
        original = getattr(subprocess, name, None)
        if original is None:
            continue

        def make_str(orig=original):
            def guarded(cmd, *a, **kw):
                _check_command(cmd, docker_allowed=docker_allowed)
                return orig(cmd, *a, **kw)

            return guarded

        monkeypatch.setattr(subprocess, name, make_str())

    # os.* process aliases
    for name in ("system", "popen"):
        original = getattr(os, name, None)
        if original is None:
            continue

        def make_os(orig=original):
            def guarded(cmd, *a, **kw):
                _check_command(cmd, docker_allowed=docker_allowed)
                return orig(cmd, *a, **kw)

            return guarded

        monkeypatch.setattr(os, name, make_os())

    for name in (
        "execv",
        "execve",
        "execvp",
        "execvpe",
        "spawnv",
        "spawnve",
        "spawnvp",
        "spawnvpe",
        "posix_spawn",
    ):
        original = getattr(os, name, None)
        if original is None:
            continue

        def make_exec(orig=original):
            def guarded(path, args, *a, **kw):
                _check_command([path, *_flatten_command(args)[1:]], docker_allowed=docker_allowed)
                return orig(path, args, *a, **kw)

            return guarded

        monkeypatch.setattr(os, name, make_exec())

    # docker SDK, if installed
    try:
        import docker as docker_sdk
    except Exception:
        docker_sdk = None
    if docker_sdk is not None:
        def denied(*a, **kw):
            raise RealDockerInTest(
                "a test attempted to use the Docker SDK. Real Docker is "
                "denied by default; mock it or use an injected runner."
            )

        for attr in ("from_env", "DockerClient"):
            if hasattr(docker_sdk, attr) and not docker_allowed:
                monkeypatch.setattr(docker_sdk, attr, denied)

    yield
