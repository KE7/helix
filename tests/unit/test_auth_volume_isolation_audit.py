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

Why ``xfail(strict=True)``
--------------------------
The isolation assertions describe behaviour HELIX does **not** have at
3bf6c80 — they are the *fail-before* half of a fail-before/pass-after pair.
Marking them ``strict`` keeps this branch green today and turns the suite RED
(XPASS is a failure) the moment the structural fix lands, forcing whoever
lands it to flip the marker and reconcile with this audit rather than
silently diverging.  The two *guard* tests at the bottom are unmarked: they
must pass both before and after the fix.
"""

from __future__ import annotations

import pytest

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
@pytest.mark.xfail(
    strict=True,
    reason=(
        "3bf6c80 mounts the persistent auth volume over the whole container "
        "HOME for every mutation agent (sandbox.py:1197). Proven leaky by "
        "behavioural canary test A->B on disposable synthetic volumes."
    ),
)
def test_agent_container_does_not_mount_auth_volume_over_whole_home(
    backend: str,
) -> None:
    """The shared auth volume must not BE the agent's HOME.

    Catches: reintroduction of
    ``-v helix-auth-<backend>:/home/node:rw`` (or ``:ro``, or any mount of the
    persistent auth volume whose destination is exactly ``/home/node``).

    Non-vacuity: ``test_guard_audit_helpers_see_the_real_mount`` below asserts
    the very string this test forbids IS present today, so the parser and the
    volume-name spelling are both exercised.
    """
    volume = sandbox_auth_volume_name(backend)
    for spec in _mount_targets(_agent_argv(backend)):
        assert not spec.startswith(f"{volume}:/home/node"), spec


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.xfail(
    strict=True,
    reason="no per-run HOME is provisioned at 3bf6c80; HOME is the shared volume",
)
def test_agent_container_gets_a_private_per_run_home(backend: str) -> None:
    """Each mutation agent must get its own empty HOME.

    Catches: dropping the auth mount without provisioning a replacement HOME
    (which would leave the image's baked ``/home/node`` shared through the
    read-only image layer and silently break credential persistence).
    """
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
@pytest.mark.xfail(
    strict=True,
    reason=(
        "no narrowed credential mount exists at 3bf6c80/9f2bcaa. Re-expressed "
        "from the audit's original form, which asserted the auth mount must "
        "not target ~/.claude -- the approved architecture mounts EXACTLY "
        "there, and the original assertion was vacuous for a --mount spec "
        "(f-string with no placeholder, unused loop variable, and dst is not "
        "the end of a type=volume,...,volume-subpath=... spec)."
    ),
)
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


@pytest.mark.xfail(
    strict=True,
    reason="transcripts are read back out of the shared volume at 3bf6c80",
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
    assert not default_root.startswith("/home/node/.claude/projects"), default_root


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
    specs = _mount_targets(_agent_argv("claude"))
    assert specs, "argv parser found no mounts at all"
    assert any(spec.startswith("helix-auth-claude:/home/node") for spec in specs), (
        "expected the (defective) whole-HOME auth mount at this commit; if this "
        "fails the fix has landed — flip the xfail markers above and re-verify "
        "against the audit memo."
    )
