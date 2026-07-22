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


def test_parser_handles_space_separated_forms() -> None:
    """``-v``, ``--volume``, ``--mount`` and ``--tmpfs``, space-separated.

    NAME CORRECTED. This was called ``..._all_four_syntaxes``, which ASSERTED
    COMPLETENESS while enumerating only the space-separated spellings -- so a
    reviewer checking exhaustiveness found a green test saying yes, while the
    equals-attached forms were unparsed and unguarded. Completeness is now
    claimed only by the pair of tests, not by this one's name.
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


@pytest.mark.parametrize(
    "rogue",
    [
        "--volume=/tmp/helix-shared-home:/home/node:rw",
        "--mount=type=volume,src=helix-shared,dst=/home/node",
        "--tmpfs=/home/node",
    ],
)
def test_equals_attached_forms_are_parsed_and_guarded(rogue: str) -> None:
    """Docker honours ``--flag=value``; a parser blind to it cannot guard it.

    F-19: the first version of this parser matched EXACT TOKENS only, so
    ``--volume=...:/home/node:rw`` and ``--mount=...,dst=/home/node`` were
    skipped entirely and the full suite stayed green -- both in ENV MODE, the
    demo path.

    Recorded because the regression was mine and it went the opposite way to
    the one it fixed: the INHERITED parser DID handle these, via
    ``tok.startswith(("--mount=", "--tmpfs=", "--volume="))``. Verified against
    the audit branch. The rewrite closed a tmpfs blindness and opened an
    equals-form blindness the old parser did not have.
    """
    with pytest.raises(SharedHomeMountError):
        _assert_no_shared_home_mount([*_ok_argv(), rogue])


def test_equals_attached_parsing_is_not_merely_rejecting_everything() -> None:
    """Non-vacuity for the test above.

    A parser that raised on any token containing ``=`` would pass the rogue
    cases for the wrong reason -- and every legitimate tmpfs spec contains
    ``uid=1000``.
    """
    benign = [*_ok_argv(), "--tmpfs=/home/node/.cache/inner:rw,uid=1000,gid=1000"]
    _assert_no_shared_home_mount(benign)
    found = dict(_mount_destinations(["--volume=/a:/dst-eq:rw"]))
    assert found == {"/dst-eq": "bind"}


# ---------------------------------------------------------------------------
# F-20: the guard's CALL must be asserted, not just its behaviour
# ---------------------------------------------------------------------------


def test_docker_args_actually_invokes_the_guard() -> None:
    """Deleting the call from ``_docker_args`` must RED.

    Every test above calls ``_assert_no_shared_home_mount`` DIRECTLY, so all of
    them -- including S1 and S2 -- go green again the moment the production
    call is removed. The guard's own regression tests sit DOWNSTREAM of the
    thing that is unprotected, so EA's bar fails for the guard itself: the
    dangerous edit and the loud breakage were not the same edit.

    This drives the real argv builder and requires the error to PROPAGATE OUT
    OF ``_docker_args``. It is the same shape as
    ``test_preflight_calls_the_capability_check_before_touching_the_volume``.
    """
    from pathlib import Path

    from helix.config import SandboxConfig
    from helix.envpolicy import EnvGrant
    from helix.sandbox import _docker_args  # noqa: PLC2701 - argv builder under test
    import helix.sandbox as sandbox_mod

    grants = [
        EnvGrant(
            name="ANTHROPIC_API_KEY",
            value="SYNTHETIC-NOT-REAL",
            origin="auth_env_allow",
            scopes=frozenset({"agent"}),
        )
    ]
    config = SandboxConfig(
        enabled=True,
        image="i:latest",
        network="none",
        auth="env",
        auth_env_allow=["ANTHROPIC_API_KEY"],
    )

    def build() -> list[str]:
        return _docker_args(
            ["sh", "-c", "true"],
            {"ANTHROPIC_API_KEY": "SYNTHETIC-NOT-REAL"},
            Path("/tmp/ws-cand"),
            config,
            "agent",
            "i:latest",
            "claude",
            grants=grants,
        )

    # Non-vacuity: the unmodified builder produces a VALID argv.
    assert build(), "baseline argv build failed for an unrelated reason"

    # Inject a rogue whole-HOME bind into the emitted argv, exactly as a future
    # edit adding a mount would, and require the guard to catch it at the end
    # of _docker_args rather than in a direct call.
    original = sandbox_mod.private_home_tmpfs_arg

    def rogue() -> list[str]:
        return [*original(), "-v", "/tmp/helix-shared-home:/home/node:rw"]

    sandbox_mod.private_home_tmpfs_arg = rogue  # type: ignore[assignment]
    try:
        with pytest.raises(SharedHomeMountError):
            build()
    finally:
        sandbox_mod.private_home_tmpfs_arg = original  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# F-19: the full spelling matrix, each POSITIVELY VERIFIED as accepted by Docker
# ---------------------------------------------------------------------------
#
# All ten were probed against the real daemon (bind-mounting a temp dir and
# checking the destination existed inside the container). This corpus is the
# floor, NOT the source of correctness -- the parser FAILS CLOSED on anything
# mount-like it cannot fully resolve, which is what survives Docker adding an
# eleventh spelling.
_ROGUE_SPELLINGS = [
    "-v",
    "/tmp/shared:/home/node:rw",  # spaced short
    "-v/tmp/shared:/home/node:rw",  # ATTACHED, no sep
    "--volume",
    "/tmp/shared:/home/node:rw",  # spaced long
    "--volume=/tmp/shared:/home/node:rw",  # equals
    "--mount",
    "type=volume,src=s,dst=/home/node",  # spaced, dst=
    "--mount=type=volume,src=s,dst=/home/node",  # equals, dst=
    "--mount",
    "type=volume,src=s,destination=/home/node",  # destination=
    "--mount",
    "type=volume,src=s,target=/home/node",  # target=
]


@pytest.mark.parametrize(
    "rogue",
    [
        ["-v", "/tmp/shared:/home/node:rw"],
        ["-v/tmp/shared:/home/node:rw"],
        ["--volume", "/tmp/shared:/home/node:rw"],
        ["--volume=/tmp/shared:/home/node:rw"],
        ["--mount", "type=volume,src=s,dst=/home/node"],
        ["--mount=type=volume,src=s,dst=/home/node"],
        ["--mount", "type=volume,src=s,destination=/home/node"],
        ["--mount", "type=volume,src=s,target=/home/node"],
    ],
    ids=lambda r: r[0][:14],
)
def test_every_verified_docker_spelling_is_guarded(rogue: list[str]) -> None:
    """Each spelling was confirmed ACCEPTED by Docker, so each must be guarded.

    Catches: a parser that handles the spellings it was shown and silently
    ignores the rest -- which is how ``--volume=`` and ``--mount=`` bypassed
    the guard entirely while the suite stayed green, in ENV MODE.
    """
    with pytest.raises(SharedHomeMountError):
        _assert_no_shared_home_mount([*_ok_argv(), *rogue])


def test_unparseable_mount_like_token_fails_closed() -> None:
    """The design rule, not the corpus: refuse what you cannot parse.

    Enumerating parseable spellings fails OPEN by construction -- every
    unrecognised form is a silent bypass. Refusing unresolvable mount-like
    tokens fails CLOSED, which is the only version that survives a future
    Docker release adding a spelling nobody here anticipated.

    Catches: reverting to skip-what-you-do-not-understand.
    """
    from helix.sandbox import UnparseableMountError  # noqa: PLC2701

    with pytest.raises(UnparseableMountError):
        _mount_destinations(["--mount", "type=volume,src=s,no-destination-key=x"])
    with pytest.raises(UnparseableMountError):
        _mount_destinations(["--volume", "no-colon-so-no-destination"])
    with pytest.raises(UnparseableMountError):
        _mount_destinations(["--tmpfs"])  # flag with no value


def test_fail_closed_does_not_reject_legitimate_argv() -> None:
    """Non-vacuity: fail-closed must not mean fail-always.

    Without this, making ``_mount_destinations`` raise unconditionally would
    satisfy every test above.
    """
    assert _mount_destinations(_ok_argv())
    _assert_no_shared_home_mount(_ok_argv())


# ---------------------------------------------------------------------------
# F-21: the ARGV allowlist -- the class no spelling coverage can reach
# ---------------------------------------------------------------------------


def test_volumes_from_is_refused_though_it_has_no_destination_string() -> None:
    """``--volumes-from`` mounts another container's volumes at THEIR paths.

    It contains no ``dst=``, no ``src:dst`` pair, and no destination string at
    all -- so a destination parser cannot see it however many spellings it
    learns. Adding the four missing spellings closes X1/X2/X3/X6 and leaves
    this WIDE OPEN.

    Catches: reverting the argv allowlist to a mount-spelling denylist.
    """
    from helix.sandbox import (  # noqa: PLC2701
        UnknownAgentArgvFlagError,
        _assert_agent_argv_uses_only_known_flags,
    )

    for spelling in (
        ["--volumes-from", "other-container"],
        ["--volumes-from=other-container"],
    ):
        with pytest.raises(UnknownAgentArgvFlagError):
            _assert_agent_argv_uses_only_known_flags(["docker", "run", *spelling])


def test_argv_allowlist_refuses_an_unanticipated_flag() -> None:
    """The point is the INVERSION, not this particular flag.

    A flag nobody has thought of cannot be inspected for a destination, so it
    is refused outright rather than ignored. This is the same
    allowlist-not-denylist argument this change makes about HOME paths, applied
    one level up to the argv.
    """
    from helix.sandbox import (  # noqa: PLC2701
        UnknownAgentArgvFlagError,
        _assert_agent_argv_uses_only_known_flags,
    )

    with pytest.raises(UnknownAgentArgvFlagError):
        _assert_agent_argv_uses_only_known_flags(
            ["docker", "run", "--some-future-docker-flag=whatever"]
        )


def test_argv_allowlist_accepts_everything_helix_actually_emits() -> None:
    """Non-vacuity: fail-closed must not mean fail-always.

    Without this, refusing every flag would satisfy both tests above -- and
    would break every run.
    """
    from helix.sandbox import (  # noqa: PLC2701
        _assert_agent_argv_uses_only_known_flags,
    )

    _assert_agent_argv_uses_only_known_flags(
        [
            "docker",
            "run",
            "--rm",
            "--workdir",
            "/workspace",
            "--user",
            "node",
            "--network",
            "none",
            "--security-opt",
            "no-new-privileges",
            "--name",
            "c",
            "--pids-limit",
            "512",
            "--cpus",
            "2.0",
            "--memory",
            "2g",
            "--add-host",
            "h:1.2.3.4",
            "-v",
            "/a:/workspace:rw",
            "-v/b:/x:ro",
            "--tmpfs",
            "/home/node:rw,uid=1000",
            "--mount",
            "type=volume,src=s,dst=/d",
            "-e",
            "HOME=/home/node",
        ]
    )


def test_docker_args_actually_invokes_the_argv_allowlist() -> None:
    """The CALL, not the function -- F-20's lesson applied to the new guard.

    Catches: deleting ``_assert_agent_argv_uses_only_known_flags`` from
    ``_docker_args``, which would leave every test above green while the
    production argv went unchecked.
    """
    from pathlib import Path

    from helix.config import SandboxConfig
    from helix.envpolicy import EnvGrant
    from helix.sandbox import (  # noqa: PLC2701
        UnknownAgentArgvFlagError,
        _docker_args,
    )
    import helix.sandbox as sandbox_mod

    original = sandbox_mod.private_home_tmpfs_arg

    def rogue() -> list[str]:
        return [*original(), "--volumes-from", "other-container"]

    sandbox_mod.private_home_tmpfs_arg = rogue  # type: ignore[assignment]
    try:
        with pytest.raises(UnknownAgentArgvFlagError):
            _docker_args(
                ["sh", "-c", "true"],
                {"ANTHROPIC_API_KEY": "SYNTHETIC-NOT-REAL"},
                Path("/tmp/ws-cand"),
                SandboxConfig(
                    enabled=True,
                    image="i:latest",
                    network="none",
                    auth="env",
                    auth_env_allow=["ANTHROPIC_API_KEY"],
                ),
                "agent",
                "i:latest",
                "claude",
                grants=[
                    EnvGrant(
                        name="ANTHROPIC_API_KEY",
                        value="SYNTHETIC-NOT-REAL",
                        origin="auth_env_allow",
                        scopes=frozenset({"agent"}),
                    )
                ],
            )
    finally:
        sandbox_mod.private_home_tmpfs_arg = original  # type: ignore[assignment]
