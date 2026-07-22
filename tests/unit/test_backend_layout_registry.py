"""Registry guards for the per-backend HOME layout.

Two jobs:

1. Pin the layout against the exact runtimes it was measured on, so a backend
   upgrade cannot silently introduce a newly shared path.
2. Prove the registry is LOAD-BEARING -- that mutating or ignoring any class
   turns something red.  A registry nobody can show is load-bearing is the
   same failure as a guard that never fires.

The expected values here are stated INDEPENDENTLY of the production registry
(literal paths, not imports of the thing under test) wherever the assertion
would otherwise be self-consistent by construction.
"""

from __future__ import annotations

import pytest

from helix.backend_layout import (
    BACKEND_LAYOUTS,
    BackendHomeLayout,
    UnsupportedBackendLayoutError,
    assert_layout_is_isolatable,
    layout_for,
    unisolatable_files,
)
from helix.backends import BACKENDS
from helix.sandbox_home import NODE_GID, NODE_UID


# ---------------------------------------------------------------------------
# Pinned-runtime guards
# ---------------------------------------------------------------------------

# Measured from the runner images themselves.  If a base-image bump changes
# any of these, the layout below was measured against a different program and
# must be re-measured rather than assumed to still hold.
PINNED_CLI_VERSIONS = {
    # The DIGEST the demos run (@sha256:6be6fef…) is 2.1.138. The ``:latest``
    # tag is 2.1.120 and is only the login/status path. Measuring the layout
    # against ``:latest`` misses a live production channel: the
    # ``.last-cleanup`` literal is absent from 2.1.120 and present in 2.1.138.
    "claude": "2.1.138",
    "codex": "0.125.0",
    "cursor": "2026.04.17-787b533",
    "gemini": "0.39.1",
    "opencode": "1.14.24",
}


@pytest.mark.parametrize("backend", BACKENDS)
def test_every_backend_has_a_measured_layout(backend: str) -> None:
    """Catches: adding a backend without measuring its HOME layout."""
    assert backend in BACKEND_LAYOUTS


@pytest.mark.parametrize("backend", BACKENDS)
def test_layout_is_pinned_to_the_runtime_it_was_measured_on(backend: str) -> None:
    """Catches: a CLI upgrade landing without re-measuring the layout.

    The layout is a statement about a specific program's on-disk behaviour.
    Bumping the CLI without re-measuring silently converts a measurement into
    an assumption -- which is how a newly shared path gets introduced.
    """
    assert BACKEND_LAYOUTS[backend].pinned_cli_version == PINNED_CLI_VERSIONS[backend]


def test_node_uid_constant_is_pinned() -> None:
    """Catches: a base-image bump moving ``node`` off uid 1000.

    Measured as 1000:1000 on all five images.  A tmpfs created with the wrong
    uid yields a HOME the agent cannot write, failing EVERY mutation agent --
    so this must fail loudly at test time rather than at run time.
    """
    assert (NODE_UID, NODE_GID) == (1000, 1000)


@pytest.mark.parametrize("backend", BACKENDS)
def test_credential_lives_inside_the_auth_dir(backend: str) -> None:
    """The credential's parent MUST be the shared mount.

    This is not a style rule: OAuth rotation renames a temp file over the
    credential, and rename cannot cross filesystems.  If the credential were
    outside ``auth_dir`` the rotation would land on a per-run tmpfs and be
    destroyed at container exit while the server side had already rotated.
    """
    layout = BACKEND_LAYOUTS[backend]
    assert "/" not in layout.credential_file, (
        f"{backend}: credential must sit directly in auth_dir, not a subpath; "
        f"got {layout.credential_file!r}"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_auth_dir_is_never_the_whole_home(backend: str) -> None:
    """Catches: regressing to the whole-HOME mount, per backend."""
    layout = BACKEND_LAYOUTS[backend]
    assert layout.auth_dir != "/home/node"
    assert layout.auth_dir.startswith("/home/node/")


# Backends whose per-run state cannot be relocated on the pinned runtime, and
# which therefore REFUSE to run under volume mode.  Stated as data so the set
# is reviewable, and asserted in both directions below.
#
# gemini: ``state.json`` is a regular file beside the credential and gemini's
# only knob (GEMINI_CLI_HOME) moves the whole home including the credential,
# so no class-3 split is expressible.
#
# claude: an independent classification found ALL THREE of ``.last-cleanup``,
# ``mcp-needs-auth-cache.json`` and ``policy-limits.json`` CARRYING on the
# pinned digest (2.1.138), with no unknowns. Each is written and read by the
# CLI inside the shared HOME and none has a relocation knob.
# ``--strict-mcp-config`` closes only the MCP channel and is NOT a rescue.
#
# Running either backend under ``auth = "env"`` is unaffected -- there is no
# shared store in that mode.
FAIL_CLOSED_UNDER_VOLUME_MODE = {"gemini", "claude"}


@pytest.mark.parametrize("backend", BACKENDS)
def test_volume_mode_isolatability_matches_the_declared_set(backend: str) -> None:
    """Every backend either isolates cleanly or refuses -- never in between.

    Asserted in BOTH directions so the fail-closed set cannot quietly grow (a
    backend silently becoming unsupported) or quietly shrink (a backend
    becoming "supported" because someone deleted an inconvenient entry).
    """
    layout = BACKEND_LAYOUTS[backend]
    if backend in FAIL_CLOSED_UNDER_VOLUME_MODE:
        with pytest.raises(UnsupportedBackendLayoutError):
            assert_layout_is_isolatable(layout)
    else:
        assert_layout_is_isolatable(layout)


# ---------------------------------------------------------------------------
# Independent statements of measured ground truth
# ---------------------------------------------------------------------------


def test_codex_relocates_its_agent_memory_databases() -> None:
    """codex's memories/goals DBs are the reason class 3 exists.

    They are REGULAR FILES beside ``auth.json``; no directory overlay can mask
    a sibling file. Proven behaviourally: without CODEX_SQLITE_HOME 4 sqlite
    files land in ~/.codex; with it, 0 there and 6 in the redirect dir.

    Catches: dropping the redirect, which would restore a cross-run agent
    MEMORY channel -- the most directly benchmark-invalidating leak found.
    """
    layout = BACKEND_LAYOUTS["codex"]
    for db in ("memories_1.sqlite", "goals_1.sqlite", "state_5.sqlite"):
        assert db in layout.ephemeral_files, db
        assert layout.ephemeral_files[db] in layout.env_redirects
    assert layout.env_redirects["CODEX_SQLITE_HOME"].startswith("/home/node/")


def test_claude_transcripts_are_treated_as_ephemeral() -> None:
    """``projects/`` holds agent transcripts and is the primary channel."""
    assert "projects" in BACKEND_LAYOUTS["claude"].ephemeral_subdirs


def test_cursor_has_no_subdirectories_and_is_all_class3() -> None:
    """Independent record that the claude shape does not generalise.

    cursor's auth dir contains only regular files, so a design based on
    subdirectory overlays does literally nothing for it. If someone later
    "simplifies" the registry to subdirs-only, this fails.
    """
    layout = BACKEND_LAYOUTS["cursor"]
    assert layout.ephemeral_subdirs == ()
    assert layout.ephemeral_files, "cursor's per-run state is entirely class 3"


# ---------------------------------------------------------------------------
# The registry must be load-bearing (anti-vacuity)
# ---------------------------------------------------------------------------


def test_mutating_a_class3_knob_fails_closed() -> None:
    """Removing a relocation knob MUST raise, not degrade quietly.

    This is the anti-vacuity requirement for the registry itself: if dropping
    a knob changed nothing observable, the registry would be decoration.
    """
    broken = BackendHomeLayout(
        backend="codex",
        auth_dir="/home/node/.codex",
        volume_subpath=".codex",
        credential_file="auth.json",
        ephemeral_files={"memories_1.sqlite": "CODEX_SQLITE_HOME"},
        env_redirects={},  # knob removed
    )
    assert unisolatable_files(broken) == ("memories_1.sqlite",)
    with pytest.raises(UnsupportedBackendLayoutError) as exc:
        assert_layout_is_isolatable(broken)
    # the error must name the file and point at the remedy
    assert "memories_1.sqlite" in str(exc.value)
    assert 'auth = "env"' in str(exc.value)


def test_unknown_backend_fails_closed_rather_than_defaulting() -> None:
    """Catches: a permissive fallback layout for an unmeasured backend.

    Defaulting to the claude shape is exactly the error this project keeps
    producing, and here it would mount a persistent store over unmeasured
    state while reporting success.
    """
    with pytest.raises(UnsupportedBackendLayoutError) as exc:
        layout_for("some-new-backend")
    assert 'auth = "env"' in str(exc.value)


def test_at_least_one_backend_actually_isolates() -> None:
    """Non-vacuity control for the fail-closed tests above.

    If ``assert_layout_is_isolatable`` raised for everything, the mutation
    test and the fail-closed assertions would all pass for the wrong reason.
    """
    clean = [
        b for b, layout in BACKEND_LAYOUTS.items() if unisolatable_files(layout) == ()
    ]
    assert clean, "no backend isolates -- the isolatability check is vacuous"
    assert "codex" in clean, (
        "codex must isolate: CODEX_SQLITE_HOME was proven to relocate its "
        "agent memory/goals databases"
    )


def test_registry_does_not_claim_candidate_independence() -> None:
    """Wording guard, and it is load-bearing.

    Volume mode closes INCIDENTAL CLI-written cross-run state. It cannot stop
    an agent creating an unenumerated file in the writable auth dir -- proven
    by canary through the full layout. Describing it as candidate
    independence would be a false certification, so the claim must not appear
    in the module that would most naturally make it.
    """
    import helix.backend_layout as mod

    text = (mod.__doc__ or "").lower()
    assert "not a candidate-independence guarantee" in text
    assert "incidental" in text


# ---------------------------------------------------------------------------
# The argv must APPLY the registry, not merely declare it
# ---------------------------------------------------------------------------


def _volume_argv(backend: str) -> list[str]:
    from pathlib import Path

    from helix.config import SandboxConfig
    from helix.envpolicy import EnvGrant
    from helix.sandbox import _docker_args  # noqa: PLC2701 - argv is under test

    return _docker_args(
        ["sh", "-c", "true"],
        {"X": "1"},
        Path("/tmp/ws-cand"),
        SandboxConfig(enabled=True, image="i:latest", network="none", auth="volume"),
        "agent",
        "i:latest",
        backend,
        grants=[
            EnvGrant(
                name="X",
                value="1",
                origin="helix_internal",
                scopes=frozenset({"agent"}),
            )
        ],
    )


@pytest.mark.parametrize(
    "backend",
    [b for b in BACKENDS if b not in FAIL_CLOSED_UNDER_VOLUME_MODE],
)
def test_argv_applies_every_class2_overlay(backend: str) -> None:
    """Each class-2 subdir must be individually re-isolated in the FINAL argv.

    Catches: a registry entry that exists but is never emitted -- the shape of
    a control that reports success while the property is false.
    """
    layout = BACKEND_LAYOUTS[backend]
    joined = " ".join(_volume_argv(backend))
    for subdir in layout.ephemeral_subdirs:
        assert f"{layout.auth_dir}/{subdir}" in joined, subdir


@pytest.mark.parametrize(
    "backend",
    [b for b in BACKENDS if b not in FAIL_CLOSED_UNDER_VOLUME_MODE],
)
def test_argv_applies_every_class3_env_redirect(backend: str) -> None:
    """Class-3 knobs must reach the container ENVIRONMENT, not just the registry.

    This is the regression that motivated the test: the registry declared
    CODEX_SQLITE_HOME and ``_docker_args`` never set it, so codex was reported
    isolatable while its agent memory and goals databases were still being
    written into the shared auth directory.

    Also asserts the redirect TARGET is mounted: a knob pointing at a directory
    that does not exist can silently fall back to the shared location.
    """
    layout = BACKEND_LAYOUTS[backend]
    argv = _volume_argv(backend)
    joined = " ".join(argv)
    for knob, target in layout.env_redirects.items():
        assert f"{knob}={target}" in argv, f"{knob} not set in argv"
        assert f"{target}:rw" in joined, f"{target} not mounted"


def test_codex_memory_databases_are_redirected_in_the_final_argv() -> None:
    """The headline leak, asserted end to end on the argv.

    codex keeps memories/goals/state SQLite as REGULAR FILES beside auth.json;
    no overlay can mask a sibling file. If this fails, a candidate can read the
    previous candidate's agent memory.
    """
    argv = _volume_argv("codex")
    assert "CODEX_SQLITE_HOME=/home/node/.helix-run/codex-sqlite" in argv, argv
