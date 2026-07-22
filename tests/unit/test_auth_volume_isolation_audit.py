"""Cross-run HOME-isolation regression suite (audit/auth-volume-state).

These tests assert on the FINAL Docker argv produced by
``helix.sandbox._docker_args`` / ``sandbox_auth_docker_args``.  They encode the
post-fix invariant identified by the auth-volume-state audit:

    credentials persist and stay refresh-capable, BUT arbitrary prior-run
    HOME / session / cache / transcript files must not cross runs.

Ownership note
--------------
A separate agent owns the ``src/helix`` auth changes.  This file deliberately
touches no production module and adds no fixtures to the existing
``tests/unit/test_sandbox.py``, so it cannot collide with that work.

Status: the ``xfail(strict=True)`` markers have been FLIPPED
-----------------------------------------------------------
These assertions were authored as the *fail-before* half of a
fail-before/pass-after pair, marked ``strict`` so that a passing test would
turn the suite RED and force whoever landed the fix to reconcile with the
audit rather than diverge silently.  The structural fix has landed and the
markers are removed; the assertions now hold directly.

Two reconciliations were required and are recorded rather than applied
quietly:

1. The helper needed provenance grants (see below). Ported verbatim, every
   assertion died on a ``ValueError`` *before* an argv existed, so all 16
   strict xfails "passed" for a reason unrelated to mount layout -- and would
   have kept xfailing after the fix, destroying the very signal this suite
   provides.

2. ``claude`` and ``gemini`` now FAIL CLOSED under volume mode: their per-run
   state cannot be relocated off the shared store, so HELIX refuses to run
   them there rather than report an isolated run that is not isolated. For
   those backends the isolation questions are answered by *refusal*, which is
   a stronger outcome than a clean mount layout, and the tests assert the
   refusal.
"""

from __future__ import annotations

import pytest

from helix.backend_layout import UnsupportedBackendLayoutError
from helix.backends import BACKENDS
from helix.envpolicy import EnvGrant
from helix.sandbox import (
    SandboxConfig,
    sandbox_auth_docker_args,
    sandbox_auth_volume_name,
)
from helix.sandbox import _docker_args  # noqa: PLC2701 - argv is the unit under test


# Interface reconciliation with the 9f2bcaa credential fix.
#
# The audit authored this suite against 3bf6c80, where ``_docker_args`` accepted
# a bare env dict.  At 9f2bcaa, agent scope REQUIRES provenance grants and
# raises ValueError without them.
#
# This is load-bearing, not cosmetic: ported verbatim, every assertion below
# died on that ValueError *before* an argv was ever built, so all 16 strict
# xfails "passed" for a reason unrelated to the mount layout and would have gone
# on xfailing after the fix landed — silently destroying the fail-before/
# pass-after signal this suite exists to provide.  The grant below restores
# genuine argv construction so the assertions test what they claim to test.
_GRANTS = [
    EnvGrant(
        name="HELIX_DEBUG",
        value="1",
        origin="helix_internal",
        scopes=frozenset({"agent", "evaluator"}),
    )
]


# Backends that REFUSE to run under volume mode because their per-run state
# cannot be relocated off the shared store. For these the isolation questions
# below are answered by refusal, which is a stronger outcome than a clean mount
# layout: HELIX declines to run rather than report an isolated run that is not.
_VOLUME_FAIL_CLOSED = {"claude", "gemini"}


def _agent_argv(backend: str) -> list[str]:
    return _docker_args(
        ["claude", "-p", "prompt"],
        {"HELIX_DEBUG": "1"},
        "/tmp/ws",  # type: ignore[arg-type]
        SandboxConfig(enabled=True, image="helix-test:latest", network="none"),
        "agent",
        "helix-test:latest",
        backend,
        grants=_GRANTS,
    )


def _mount_targets(argv: list[str]) -> list[str]:
    """Every ``-v``/``--mount``/``--tmpfs`` spec in the final argv."""
    specs: list[str] = []
    for index, token in enumerate(argv):
        if token in {"-v", "--volume", "--mount", "--tmpfs"}:
            specs.append(argv[index + 1])
    return specs


# ---------------------------------------------------------------------------
# fail-before / pass-after: the isolation invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_agent_container_does_not_mount_auth_volume_over_whole_home(
    backend: str,
) -> None:
    """The shared auth volume must not BE the agent's HOME.

    Catches: reintroduction of
    ``-v helix-auth-<backend>:/home/node:rw`` (or ``:ro``, or any mount of the
    persistent auth volume whose destination is exactly ``/home/node``).

    FLIPPED from ``xfail(strict=True)``: the structural fix has landed, so this
    now asserts the invariant directly.

    Non-vacuity: ``test_guard_audit_helpers_see_the_real_mount`` below asserts
    that the parser still finds the auth mount at its NEW, narrowed
    destination -- so "no whole-HOME mount" cannot pass because the parser
    stopped seeing mounts.
    """
    if backend in _VOLUME_FAIL_CLOSED:
        with pytest.raises(UnsupportedBackendLayoutError):
            _agent_argv(backend)
        return
    volume = sandbox_auth_volume_name(backend)
    for spec in _mount_targets(_agent_argv(backend)):
        assert not spec.startswith(f"{volume}:/home/node"), spec


@pytest.mark.parametrize("backend", BACKENDS)
def test_agent_container_gets_a_private_per_run_home(backend: str) -> None:
    """Each mutation agent must get its own empty HOME.

    Catches: dropping the auth mount without provisioning a replacement HOME
    (which would leave the image's baked ``/home/node`` shared through the
    read-only image layer and silently break credential persistence).
    """
    if backend in _VOLUME_FAIL_CLOSED:
        with pytest.raises(UnsupportedBackendLayoutError):
            _agent_argv(backend)
        return
    specs = _mount_targets(_agent_argv(backend))
    # A per-run HOME is a mount whose DESTINATION is /home/node and whose
    # SOURCE is not the persistent auth volume: a tmpfs (``/home/node``),
    # an anonymous volume (``/home/node``), or a run-scoped host path.
    # ``helix-auth-<backend>:/home/node:rw`` must NOT satisfy this.
    volume = sandbox_auth_volume_name(backend)
    private_home = [
        spec
        for spec in specs
        if volume not in spec
        and (spec.split(":")[0] == "/home/node" or ":/home/node:" in spec)
    ]
    assert private_home, (
        "expected a per-run tmpfs or anonymous volume at /home/node; "
        f"got mounts: {specs}"
    )


# Per-backend class-2 paths: directories that live INSIDE the backend's auth
# directory and carry per-run agent state.  Enumerated from ``:ro`` metadata
# reads of the five real auth volumes on the pinned runtimes.
#
# Deliberately hardcoded here rather than imported from the production layout
# registry.  A regression test that derives its expectations from the code it
# checks is self-consistent by construction: a registry that silently drops a
# path would drop it from the assertion too, and the suite would stay green
# while the leak reopened.  These lists are an INDEPENDENT statement of ground
# truth and must be updated only against re-measured volume evidence.
_AUTH_DIR = {
    "claude": "/home/node/.claude",
    "codex": "/home/node/.codex",
    "cursor": "/home/node/.cursor",
    "gemini": "/home/node/.gemini",
    "opencode": "/home/node/.local/share/opencode",
}
_CLASS2_SUBDIRS = {
    "claude": (
        "projects",
        "sessions",
        "backups",
        "shell-snapshots",
        "session-env",
        "cache",
    ),
    "codex": (
        "sessions",
        "log",
        "shell_snapshots",
        "memories",
        "tmp",
        "cache",
    ),
    "cursor": (),
    "gemini": ("tmp", "history"),
    "opencode": (),
}


@pytest.mark.parametrize("backend", BACKENDS)
def test_shared_auth_state_is_narrower_than_home(backend: str) -> None:
    """Persistent sharing must be narrower than HOME *in effect*, not just in dst.

    The invariant is NOT "the mount destination isn't the auth dir" -- the
    approved four-class architecture mounts the persistent volume exactly at
    the auth dir, because the credential's parent directory must be on the
    persistent filesystem for OAuth's atomic rename to work at all (a
    single-file bind is EBUSY on both rename-over and unlink).

    The real invariant is:

      1. the persistent volume is still mounted (credentials persist), and
      2. it targets a PROPER SUBPATH of HOME, never HOME itself, and
      3. every class-2 subdirectory inside that auth dir is individually
         re-isolated by a per-run mount, so prior-run agent state cannot be
         read through the shared mount.

    Clause 3 is what the original form missed and what makes this test
    falsifiable: deleting any single overlay from the production mount layer
    turns this red for that backend.
    """
    if backend in _VOLUME_FAIL_CLOSED:
        with pytest.raises(UnsupportedBackendLayoutError):
            _agent_argv(backend)
        return
    volume = sandbox_auth_volume_name(backend)
    specs = _mount_targets(_agent_argv(backend))
    shared = [spec for spec in specs if volume in spec]

    # (1) credentials must still persist -- guards the D2/D3 over-correction.
    assert shared, f"credentials must still persist for {backend}"

    # (2) never HOME itself.
    for spec in shared:
        assert not spec.startswith(f"{volume}:/home/node:"), (
            f"auth volume must not BE the container HOME: {spec}"
        )
        assert "/home/node/" in spec, (
            f"auth mount must target a subpath of HOME, not HOME itself: {spec}"
        )

    # (3) every class-2 subdir is individually re-isolated.
    non_auth = [spec for spec in specs if volume not in spec]
    for leaky in _CLASS2_SUBDIRS[backend]:
        target = f"{_AUTH_DIR[backend]}/{leaky}"
        assert any(target in spec for spec in non_auth), (
            f"{backend}: {target} is inside the shared auth mount and carries "
            f"per-run agent state, but no per-run mount re-isolates it -- "
            f"prior-run state crosses. mounts: {specs}"
        )


def test_transcript_root_is_not_inside_shared_persistent_state() -> None:
    """``claude_transcript_root`` must not point into the shared volume.

    Catches: leaving ``sandbox.claude_transcript_root`` at
    ``/home/node/.claude/projects/-workspace``.  That default is the concrete
    cross-candidate channel: because every workspace is mounted at
    ``/workspace``, every candidate of every run writes into the SAME Claude
    project directory, and the audit observed 28 transcripts from four dates
    co-resident there.
    """
    default_root = SandboxConfig(enabled=True).claude_transcript_root
    assert default_root is None or not default_root.startswith(
        "/home/node/.claude/projects"
    ), default_root


# ---------------------------------------------------------------------------
# guards: must pass BEFORE and AFTER the fix
# ---------------------------------------------------------------------------


def test_evaluator_scope_never_mounts_any_auth_volume() -> None:
    """The candidate/evaluator scope must stay credential-free.

    This is the AlgoTune criterion-8 guard.  Catches: any future change that
    starts mounting the agent auth volume (or any ``helix-auth-*`` volume)
    into the evaluator sidecar, which would put a shared persistent HOME
    behind the private-empty-candidate-HOME claim.

    Non-vacuity: the same helper, called with ``scope="agent"``, does find an
    auth volume (asserted in ``test_guard_audit_helpers_see_the_real_mount``),
    so an empty result here is a real property of the evaluator scope and not
    a broken parser.
    """
    argv = _docker_args(
        ["python", "evaluate.py"],
        {},
        "/tmp/ws",  # type: ignore[arg-type]
        SandboxConfig(enabled=True, image="helix-test:latest", network="none"),
        "evaluator",
        "helix-test:latest",
        "claude",
    )
    joined = " ".join(argv)
    assert "helix-auth-" not in joined, joined


def test_credential_persistence_is_still_required_for_auth_commands() -> None:
    """``helix sandbox login`` must keep a WRITABLE credential store.

    Catches: an over-eager 'isolation' fix that makes the auth store
    read-only or ephemeral.  OAuth refresh rewrites the credential wholesale,
    so a read-only or per-run credential mount would rotate the server-side
    token and then discard the new one — silently destroying the login for
    every later run (audit tests T4/T7/T8/T9).
    """
    argv = sandbox_auth_docker_args(
        "claude", image="helix-claude:latest", action="login", interactive=True
    )
    joined = " ".join(argv)
    assert "helix-auth-claude" in joined
    assert ":ro" not in joined
    assert "--user node" in joined


def test_guard_audit_helpers_see_the_real_mount() -> None:
    """Non-vacuity control for the parser used by every test above.

    If ``_mount_targets`` ever stops parsing the argv (renamed flag, changed
    argv shape), this test fails loudly instead of letting the isolation
    assertions pass for the wrong reason.  At 3bf6c80 it documents the defect
    itself: the agent argv really does mount the shared volume over HOME.
    """
    # ``claude`` now FAILS CLOSED under volume mode, so it can no longer serve
    # as the non-vacuity subject -- there is no argv to parse. ``codex`` is the
    # backend proven isolatable (CODEX_SQLITE_HOME relocates its agent memory
    # and goals databases), so it carries the control.
    specs = _mount_targets(_agent_argv("codex"))
    assert specs, "argv parser found no mounts at all"

    # The parser must still find the persistent auth volume -- at its NEW,
    # narrowed destination. Without this, every "no whole-HOME mount"
    # assertion above could pass simply because the parser stopped seeing
    # mounts, or because the volume name changed, or because credentials
    # stopped being mounted at all (the D2/D3 over-correction).
    volume = sandbox_auth_volume_name("codex")
    auth_specs = [spec for spec in specs if volume in spec]
    assert auth_specs, f"parser found no {volume} mount; credentials must still persist"
    assert any("volume-subpath=" in spec for spec in auth_specs), auth_specs
    assert not any(spec.startswith(f"{volume}:/home/node") for spec in auth_specs), (
        "the whole-HOME mount is back"
    )

    # And a per-run HOME must be present, so the narrowed mount is not simply
    # sitting on top of the image's shared /home/node.
    assert any(spec.startswith("/home/node:") for spec in specs), specs
