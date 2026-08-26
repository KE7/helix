"""Shared fixtures for container-backed integration tests.

These tests run real backend containers.  They never touch a real
``helix-auth-*`` login volume and never perform a login: every credential they
write is synthetic and lives in a throwaway volume that is removed again in
teardown.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from collections.abc import Iterator

import pytest


REAL_AUTH_VOLUME_PREFIX = "helix-auth-"
TEST_VOLUME_PREFIX = "helix-agent-state-test-"


def _strict() -> bool:
    return os.environ.get("HELIX_DOCKER_TESTS_STRICT") == "1"


def _unavailable(reason: str) -> None:
    """Fail under ``HELIX_DOCKER_TESTS_STRICT=1``, otherwise skip.

    Strict mode exists so CI cannot silently turn this suite into a no-op:
    a missing daemon or image becomes a failure to fix rather than a green run.
    """
    if _strict():
        pytest.fail(f"HELIX_DOCKER_TESTS_STRICT=1 but {reason}")
    pytest.skip(reason)


def _docker(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args], capture_output=True, text=True, check=check, timeout=300
    )


@pytest.fixture(scope="session")
def docker_available() -> None:
    try:
        _docker("info")
    except (OSError, subprocess.SubprocessError) as exc:
        _unavailable(f"Docker daemon is not usable: {exc}")


@pytest.fixture
def require_image(docker_available: None):
    def _require(image: str) -> str:
        result = _docker("image", "inspect", image, check=False)
        if result.returncode != 0:
            _unavailable(f"image {image} is not present locally")
        return image

    return _require


@pytest.fixture
def throwaway_volume(docker_available: None) -> Iterator[object]:
    """Create synthetic-credential volumes and guarantee their removal.

    The name prefix is asserted on every call so a bug in a test can never
    address, mutate, or delete one of the real ``helix-auth-*`` volumes.
    """
    created: list[str] = []

    def _create(image: str, seed_script: str) -> str:
        name = f"{TEST_VOLUME_PREFIX}{uuid.uuid4().hex[:12]}"
        assert not name.startswith(REAL_AUTH_VOLUME_PREFIX)
        _docker("volume", "create", name)
        created.append(name)
        # Seed as root, then hand the tree to the container user the backends
        # run as, mirroring how a real login volume ends up owned.
        _docker(
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{name}:/home/node",
            "--user",
            "root",
            image,
            "sh",
            "-c",
            f"set -eu; {seed_script}; chown -R node:node /home/node",
        )
        return name

    try:
        yield _create
    finally:
        for name in created:
            assert name.startswith(TEST_VOLUME_PREFIX)
            _docker("volume", "rm", "-f", name, check=False)


@pytest.fixture
def volume_listing(docker_available: None):
    """Return names/modes/sizes of a volume's contents -- never file contents."""

    def _list(volume: str, image: str) -> set[str]:
        # ``~/.cache`` is pruned: the CLIs drop version-keyed caches there (for
        # example cursor's V8 compile cache). Those are rebuilt from the image
        # and carry no candidate state, so including them would make the
        # comparison flap without telling us anything about contamination.
        result = _docker(
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
            'find /home/node -mindepth 1 '
            '-not -path /home/node/.cache -not -path "/home/node/.cache/*" '
            '-printf "%M %s %p\\n" | sort -k3',
        )
        return {line for line in result.stdout.splitlines() if line.strip()}

    return _list
