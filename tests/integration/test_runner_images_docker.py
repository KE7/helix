"""Real-Docker proof that every shipped runner image actually runs its CLI.

``DEFAULT_BACKEND_IMAGES`` in ``src/helix/backends.py`` names one image per
mutation backend, and ``docker/<backend>.Dockerfile`` builds it. Nothing in the
unit suite runs those images: it patches ``helix.sandbox.subprocess.run`` and
inspects the argv HELIX builds, so a Dockerfile that installs its CLI somewhere
the container's unprivileged user cannot reach still passes every unit test.

This module covers the gap. It is parametrized directly off
``DEFAULT_BACKEND_IMAGES`` -- never a second hardcoded list -- so a backend
added to the registry is covered automatically, and a backend whose image is
missing fails loudly rather than being silently skipped.

The command each image is checked with is the backend's own version probe. It
is deliberately the cheapest call the CLI offers: it needs no credential, makes
no model call, and reaches no network. These tests never read, mount, or
reference an operator's credential or a ``helix-auth-*`` volume.

Ownership matters as much as presence. Each check runs as ``node``, the same
unprivileged user ``_docker_args`` in ``src/helix/sandbox.py`` gives the agent
container, because an installer that drops its binary under ``/root`` produces
an image that works when probed as root and fails for every real run.

These tests carry the ``docker_integration`` marker (see ``pyproject.toml``)
and skip when the daemon or an image is unavailable, so a machine without
Docker is not a failure. Set ``HELIX_DOCKER_TESTS_STRICT=1``, as CI does, to
turn every such skip into a failure.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from helix.backends import DEFAULT_BACKEND_IMAGES


pytestmark = pytest.mark.docker_integration

# The version probe for each backend's CLI, keyed by backend name. Kept beside
# the image registry it is checked against rather than derived from
# ``BACKEND_AUTH_COMMANDS``: those commands authenticate, which is exactly what
# a smoke test must not do.
_VERSION_COMMANDS: dict[str, str] = {
    "agy": "agy --version",
    "claude": "claude --version",
    "codex": "codex --version",
    "cursor": "cursor agent --version",
    "opencode": "opencode --version",
}

_BACKENDS = sorted(DEFAULT_BACKEND_IMAGES)


def _image(backend: str) -> str:
    """Resolve the runner image for *backend*, honoring a test-only override."""
    override = os.environ.get(f"HELIX_DOCKER_TEST_IMAGE_{backend.upper()}")
    return override or DEFAULT_BACKEND_IMAGES[backend]


def _docker(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], check=False, capture_output=True, text=True, timeout=timeout
    )


def _require_docker_image(image: str) -> None:
    """Skip when Docker or the image is unavailable.

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
        daemon = _docker("info", timeout=30)
        inspected = _docker("image", "inspect", image, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        _bail(f"Docker daemon unavailable: {exc}")
        return
    if daemon.returncode != 0:
        _bail(f"Docker daemon unavailable: {daemon.stderr.strip()}")
        return
    if inspected.returncode != 0:
        _bail(f"runner image {image!r} is not installed locally")


def test_every_backend_in_the_registry_has_a_version_command() -> None:
    """A backend added to the registry must also be added to this module."""
    assert sorted(_VERSION_COMMANDS) == _BACKENDS


@pytest.mark.timeout(180)
@pytest.mark.parametrize("backend", _BACKENDS)
def test_runner_image_cli_runs_as_the_unprivileged_agent_user(backend: str) -> None:
    """The backend CLI is on PATH and executable for the container's agent."""
    image = _image(backend)
    _require_docker_image(image)
    result = _docker(
        "run",
        "--rm",
        "--user",
        "node",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        image,
        "sh",
        "-lc",
        _VERSION_COMMANDS[backend],
        timeout=120,
    )
    assert result.returncode == 0, (
        f"{_VERSION_COMMANDS[backend]!r} failed in {image} as user node "
        f"(exit {result.returncode})\nstdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert result.stdout.strip(), f"{image} produced no version output"
