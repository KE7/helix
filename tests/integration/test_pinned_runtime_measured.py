"""Pinned layout guards, measured against the IMAGES rather than a constant.

``test_layout_is_pinned_to_the_runtime_it_was_measured_on`` compares the
registry to a hardcoded dict in the test file -- a DECLARATION checked against
another DECLARATION. It cannot detect the case that matters: the registry and
the constant agreeing with each other while both disagree with the image that
actually runs.

That is not hypothetical here. The claude layout was originally measured
against ``:latest`` (2.1.120) while the demos run the pinned DIGEST (2.1.138),
and the two differ in a load-bearing way -- the ``.last-cleanup`` literal is
absent from one and present in the other. A declaration-vs-declaration check
was structurally incapable of noticing.

These tests run the real images, so they are marked ``docker_integration``.
"""

from __future__ import annotations

import shutil
import subprocess

import pytest

from helix.backend_layout import BACKEND_LAYOUTS
from helix.sandbox_home import NODE_GID, NODE_UID


pytestmark = pytest.mark.docker_integration


# The version-reporting command per backend, and the image to read it from.
# claude is pinned to the DIGEST the demos run, not to ``:latest``.
_PROBES = {
    "claude": (
        "ghcr.io/ke7/helix-evo-runner-claude"
        "@sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba",
        "claude --version",
    ),
    "codex": ("ghcr.io/ke7/helix-evo-runner-codex:latest", "codex --version"),
    "cursor": (
        "ghcr.io/ke7/helix-evo-runner-cursor:latest",
        "cursor-agent --version",
    ),
    "gemini": ("ghcr.io/ke7/helix-evo-runner-gemini:latest", "gemini --version"),
    "opencode": ("ghcr.io/ke7/helix-evo-runner-opencode:latest", "opencode --version"),
}


def _image_present(image: str) -> bool:
    """Positively identify an ABSENT image, rather than inferring it."""
    return (
        subprocess.run(
            ["docker", "image", "inspect", image],
            capture_output=True,
            text=True,
            timeout=60,
        ).returncode
        == 0
    )


def _run_in_image(image: str, command: str) -> str:
    """Run *command* in *image*.

    SKIP/FAIL DISCIPLINE (F-11). The earlier version skipped on ANY nonzero
    exit with "image unavailable locally", so a renamed CLI, a broken
    entrypoint or a changed ``--version`` flag degraded the whole suite to SKIP
    and the gate reported GREEN. That is the same missing-vs-failed conflation
    ``helix.transcripts`` was rewritten to eliminate, reintroduced inside the
    guard that fixed a declaration-vs-declaration check.

    Availability is now decided BEFORE the probe runs, by positively
    identifying an absent image. Once the probe is SELECTED, a nonzero exit is
    a FAILURE.
    """
    if shutil.which("docker") is None:
        pytest.skip("docker unavailable")
    if not _image_present(image):
        pytest.skip(f"image not present locally: {image}")
    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--network",
            "none",
            "--user",
            "1000:1000",
            "--entrypoint",
            "sh",
            image,
            "-c",
            command,
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"probe failed in {image}: exit {result.returncode}\n"
        f"  command: {command}\n"
        f"  stderr:  {result.stderr.strip()[:400]}\n"
        f"  The image IS present, so this is a real failure -- a renamed CLI, a "
        f"broken entrypoint, or a changed flag. Skipping here would report the "
        f"gate GREEN while measuring nothing."
    )
    return result.stdout.strip()


@pytest.mark.parametrize("backend", sorted(_PROBES))
def test_pinned_cli_version_matches_the_actual_image(backend: str) -> None:
    """The registry's pin must match what the image REPORTS.

    Catches: a base-image or digest bump landing without re-measuring the
    layout -- which silently converts a measurement into an assumption, and is
    how a newly shared path gets introduced.
    """
    image, command = _PROBES[backend]
    reported = _run_in_image(image, command)
    declared = BACKEND_LAYOUTS[backend].pinned_cli_version
    assert declared in reported, (
        f"{backend}: registry pins {declared!r} but {image} reports "
        f"{reported!r} -- re-measure the layout against the image that runs"
    )


@pytest.mark.parametrize("backend", sorted(_PROBES))
def test_node_uid_matches_the_actual_image(backend: str) -> None:
    """uid 1000 must be MEASURED, not assumed.

    A tmpfs created with the wrong uid yields a HOME the agent cannot write,
    failing every mutation agent. The constant is correct today on all five
    images; this is what keeps it correct.
    """
    image, _ = _PROBES[backend]
    reported = _run_in_image(image, "id -u node; id -g node")
    assert reported.split() == [str(NODE_UID), str(NODE_GID)], (
        f"{backend}: {image} reports node as {reported!r}, not {NODE_UID}:{NODE_GID}"
    )


@pytest.mark.parametrize("backend", sorted(_PROBES))
def test_declared_auth_dir_exists_or_is_absent_for_a_reason(backend: str) -> None:
    """The auth_dir path must be one the image's CLI would actually use.

    Weak by necessity -- a fresh image has no auth dir until first login -- so
    this asserts only that the declared path is under the image's real home,
    catching a typo'd or renamed auth_dir rather than proving occupancy.
    """
    image, _ = _PROBES[backend]
    home = _run_in_image(image, "getent passwd node | cut -d: -f6")
    assert BACKEND_LAYOUTS[backend].auth_dir.startswith(home + "/"), (
        f"{backend}: auth_dir is not under the image's home {home!r}"
    )


# F-7 NOTE -- why the bound is NOT asserted against the live volume here.
#
# The session-wide safety guard makes ``helix-auth-*`` an ABSOLUTE denial with
# no override, including for docker_integration tests. A test that reads the
# real auth volume is therefore forbidden by design, and correctly so.
#
# So ``stable_files`` is bounded against a RECORDED measurement instead: see
# ``measured_entries`` in helix/backend_layout.py, which carries the ``ls -A``
# output and its provenance. The bound is asserted in
# tests/unit/test_backend_layout_registry.py. Re-verifying that recording
# requires an operator running the documented command out of band -- the test
# suite structurally cannot, and pretending otherwise would be a worse answer
# than stating the limit.
