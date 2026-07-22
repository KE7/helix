"""Destination-scoped guard: NOTHING shared may land on the agent's HOME.

Every prior mount guard asked a NAME-scoped question -- "is the AUTH VOLUME on
HOME?" -- so any other name passed. The property is DESTINATION-scoped: a rogue
bind or a differently-named volume at ``/home/node`` reproduces the original
cross-candidate defect exactly while satisfying every auth-volume assertion.

This is the mount-side counterpart of ``_assert_env_is_granted``, and it exists
for the same reason: the original defects were introduced by a NEW CALL SITE
added after the convention was established, which only a check over the FINAL
artifact can catch.

Scope: AGENT containers only. ``helix sandbox login`` mounts the auth volume at
HOME legitimately and never enters ``_docker_args``, so no doc sentence of the
form "HELIX never mounts the auth volume over HOME" is true -- it must say "no
AGENT container mounts...".
"""

from __future__ import annotations

import pytest

from helix.exceptions import SharedHomeMountError
from helix.sandbox import (  # noqa: PLC2701 - the guard is the unit under test
    _assert_no_shared_home_mount,
    _mount_destinations,
)


_PRIVATE_HOME = ["--tmpfs", "/home/node:rw,uid=1000,gid=1000,mode=0755"]
_TRANSCRIPTS = ["-v", "/host/cand.helix-transcripts:/home/node/.claude/projects:rw"]


def _ok_argv() -> list[str]:
    return [
        "docker",
        "run",
        "-v",
        "/host/ws:/workspace:rw",
        *_PRIVATE_HOME,
        *_TRANSCRIPTS,
    ]


def test_the_shipped_env_mode_shape_passes() -> None:
    """Non-vacuity: the guard is not an unconditional refusal."""
    _assert_no_shared_home_mount(_ok_argv())


# --- the three mutations that were GREEN before this guard existed ----------


def test_S1_rogue_host_bind_over_home_is_rejected() -> None:
    """S1: ``-v /tmp/helix-shared-home:/home/node:rw`` beside the private tmpfs.

    Every candidate of every run would then share one host directory as HOME --
    the original defect, in ENV MODE, which is the mode all four demos run.
    Passed every existing assertion because the source is not named
    ``helix-auth-*``.
    """
    argv = [*_ok_argv(), "-v", "/tmp/helix-shared-home:/home/node:rw"]
    with pytest.raises(SharedHomeMountError):
        _assert_no_shared_home_mount(argv)


def test_S2_rogue_named_volume_over_home_is_rejected() -> None:
    """S2: the same outcome via ``--mount type=volume,dst=/home/node``."""
    argv = [*_ok_argv(), "--mount", "type=volume,src=helix-shared,dst=/home/node"]
    with pytest.raises(SharedHomeMountError):
        _assert_no_shared_home_mount(argv)


def test_S3_nested_shared_mount_inside_home_is_rejected() -> None:
    """S3: a second mount of the auth volume at ``/home/node/.cache``.

    Reintroduces cross-run sharing of exactly the path the original audit
    singled out as "neither auth nor isolated".
    """
    argv = [
        *_ok_argv(),
        "--mount",
        "type=volume,src=helix-auth-claude,dst=/home/node/.cache",
    ]
    with pytest.raises(SharedHomeMountError):
        _assert_no_shared_home_mount(argv, allowed_auth_dir="/home/node/.claude")


def test_ancestor_of_home_is_rejected() -> None:
    """A mount at ``/home`` or ``/`` shares HOME transitively."""
    for ancestor in ("/home", "/"):
        with pytest.raises(SharedHomeMountError):
            _assert_no_shared_home_mount([*_ok_argv(), "-v", f"/src:{ancestor}:rw"])


def test_missing_private_home_is_rejected() -> None:
    """Zero private tmpfs means the agent inherits the image's shared HOME."""
    with pytest.raises(SharedHomeMountError):
        _assert_no_shared_home_mount(["docker", "run", *_TRANSCRIPTS])


def test_duplicate_private_home_is_rejected() -> None:
    """Two mounts at HOME are ambiguous; one of them wins and it is unclear which."""
    with pytest.raises(SharedHomeMountError):
        _assert_no_shared_home_mount([*_ok_argv(), *_PRIVATE_HOME])


def test_declared_auth_dir_is_permitted_but_only_at_its_exact_path() -> None:
    """Volume mode's ONE shared mount is allowed, and nothing else inside HOME."""
    auth = ["--mount", "type=volume,src=helix-auth-codex,dst=/home/node/.codex"]
    _assert_no_shared_home_mount(
        [*_ok_argv(), *auth], allowed_auth_dir="/home/node/.codex"
    )
    # ...but not at a different path, even for the same volume
    with pytest.raises(SharedHomeMountError):
        _assert_no_shared_home_mount(
            [
                *_ok_argv(),
                "--mount",
                "type=volume,src=helix-auth-codex,dst=/home/node/.elsewhere",
            ],
            allowed_auth_dir="/home/node/.codex",
        )


# --- the parser must SEE tmpfs, which the inherited one did not -------------


def test_parser_sees_tmpfs_specs() -> None:
    """The inherited parser was BLIND to tmpfs and nobody noticed.

    ``_dest_is_home_or_above`` took its ``":" in spec`` branch only when
    ``"=" not in spec``, and every tmpfs spec contains ``uid=1000`` -- so the
    private HOME tmpfs was invisible, and that audit's "no mount lands on HOME"
    result was far narrower than it read.

    Catches: a parser regression that would silently make every assertion above
    vacuous.
    """
    found = dict(_mount_destinations(_ok_argv()))
    assert found["/home/node"] == "tmpfs"
    assert found["/home/node/.claude/projects"] == "bind"
    assert found["/workspace"] == "bind"


def test_parser_handles_all_four_syntaxes() -> None:
    """``-v``, ``--volume``, ``--mount`` and ``--tmpfs`` must all be parsed.

    A syntax the parser does not understand is a syntax the guard cannot check.
    """
    argv = [
        "-v",
        "/a:/dst-v:rw",
        "--volume",
        "/b:/dst-volume:ro",
        "--mount",
        "type=bind,src=/c,dst=/dst-mount",
        "--tmpfs",
        "/dst-tmpfs:rw,uid=1000",
    ]
    found = dict(_mount_destinations(argv))
    assert set(found) == {"/dst-v", "/dst-volume", "/dst-mount", "/dst-tmpfs"}
    assert found["/dst-tmpfs"] == "tmpfs"
