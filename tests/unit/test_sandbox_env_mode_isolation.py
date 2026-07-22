"""Env-mode isolation and candidate-MCP containment (final-argv assertions).

Env mode (``sandbox.auth = "env"``) is a first-class, fail-closed alternative
to volume mode, not a fallback.  Its isolation claim is structurally stronger:
because no persistent store is mounted at all, the cross-run channel does not
exist rather than being masked, so the claim does not depend on any denylist
being complete.

Every assertion here is on the FINAL Docker argv or the FINAL backend argv --
the thing that actually runs -- rather than on intermediate state.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from helix.backends import BACKENDS
from helix.config import AgentConfig, SandboxConfig
from helix.envpolicy import EnvGrant
from helix.mutator import _build_backend_args  # noqa: PLC2701 - argv is under test
from helix.sandbox import _docker_args  # noqa: PLC2701 - argv is under test
from helix.sandbox_home import NODE_GID, NODE_UID, transcript_host_dir


def _env_argv(backend: str, workspace: str = "/tmp/ws-cand-1") -> list[str]:
    grants = [
        EnvGrant(
            name="ANTHROPIC_API_KEY",
            value="SYNTHETIC-NOT-REAL",
            origin="auth_env_allow",
            scopes=frozenset({"agent"}),
        )
    ]
    return _docker_args(
        ["claude", "-p", "prompt"],
        {"ANTHROPIC_API_KEY": "SYNTHETIC-NOT-REAL"},
        Path(workspace),
        SandboxConfig(
            enabled=True,
            image="helix-test:latest",
            network="none",
            auth="env",
            auth_env_allow=["ANTHROPIC_API_KEY"],
        ),
        "agent",
        "helix-test:latest",
        backend,
        grants=grants,
    )


# ---------------------------------------------------------------------------
# Gate 1 -- env mode mounts no persistent auth volume
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_env_mode_argv_contains_no_auth_volume_reference(backend: str) -> None:
    """ZERO ``helix-auth-*`` references in env mode, at any mount mode.

    Catches: the pre-fix behaviour, which mounted the persistent volume ``:ro``
    over the whole HOME on the reasoning that env mode cannot refresh it.  A
    read-only whole-HOME mount still exposes every prior run's transcripts and
    sessions FOR READING, and reading is the entire cross-candidate channel --
    proven by canary: under the ``:ro`` argv a later container read a prior
    run's transcript verbatim; under this argv it reads nothing.

    Non-vacuity: ``test_volume_mode_still_mounts_the_auth_volume`` below shows
    the same helper DOES produce an auth mount in volume mode, so an empty
    result here is a property of env mode and not a broken query.
    """
    joined = " ".join(_env_argv(backend))
    assert "helix-auth-" not in joined, joined


# claude and gemini FAIL CLOSED under volume mode -- their per-run state cannot
# be relocated off the shared store -- so they have no volume-mode argv to
# inspect. The non-vacuity control below uses the backends that do.
_VOLUME_FAIL_CLOSED = {"claude", "gemini"}


@pytest.mark.parametrize(
    "backend", [b for b in BACKENDS if b not in _VOLUME_FAIL_CLOSED]
)
def test_volume_mode_still_mounts_the_auth_volume(backend: str) -> None:
    """Non-vacuity control for the assertion above."""
    grants = [
        EnvGrant(
            name="X",
            value="1",
            origin="helix_internal",
            scopes=frozenset({"agent"}),
        )
    ]
    argv = _docker_args(
        ["claude", "-p", "p"],
        {"X": "1"},
        Path("/tmp/ws"),
        SandboxConfig(enabled=True, image="i:latest", network="none", auth="volume"),
        "agent",
        "i:latest",
        backend,
        grants=grants,
    )
    assert f"helix-auth-{backend}" in " ".join(argv)


@pytest.mark.parametrize("backend", BACKENDS)
def test_env_mode_home_is_private_and_uid_correct(backend: str) -> None:
    """A private HOME is mandatory in env mode, and must be writable by ``node``.

    Catches BOTH failure modes:
      - dropping the auth mount without provisioning any replacement HOME,
        which would leave the image's baked ``/home/node`` in place; and
      - provisioning a bare ``--tmpfs /home/node``, which Docker creates as
        ``root:root`` so uid 1000 cannot write its own HOME and EVERY mutation
        agent fails.
    """
    argv = _env_argv(backend)
    specs = [
        argv[i + 1]
        for i, tok in enumerate(argv)
        if tok == "--tmpfs" and argv[i + 1].startswith("/home/node:")
    ]
    assert specs, f"env mode must provision a private per-run HOME: {argv}"
    assert f"uid={NODE_UID}" in specs[0], specs[0]
    assert f"gid={NODE_GID}" in specs[0], specs[0]


@pytest.mark.parametrize("backend", BACKENDS)
def test_env_mode_binds_a_candidate_keyed_transcript_dir(backend: str) -> None:
    """Transcript capture must survive env mode, keyed per candidate.

    ``preserve_backend_transcripts`` is coupled to the defective mount, so an
    isolation fix that ignores it silently breaks a shipped feature.
    """
    argv = _env_argv(backend)
    binds = [
        argv[i + 1]
        for i, tok in enumerate(argv)
        if tok == "-v" and argv[i + 1].endswith("/home/node/.claude/projects:rw")
    ]
    assert binds, f"env mode must bind a candidate transcript dir: {argv}"


def test_distinct_candidates_get_distinct_transcript_roots() -> None:
    """The guarantee must be HELIX's, not the backend's.

    Today the copy-out builds its path from a session id parsed out of backend
    stdout, so the candidate identity never enters the path: two candidates
    reporting the same session id collide into one file (reproduced as a single
    interleaved transcript).  Deriving the location from the per-candidate
    workspace makes distinctness structural.
    """
    a = transcript_host_dir(Path("/tmp/helix/cand-a"))
    b = transcript_host_dir(Path("/tmp/helix/cand-b"))
    assert a != b
    # never inside the workspace: anything there is synced back into the repo
    assert Path("/tmp/helix/cand-a") not in a.parents


# ---------------------------------------------------------------------------
# Gate 2 -- the candidate cannot select an MCP endpoint or server name
# ---------------------------------------------------------------------------


def _claude_argv() -> list[str]:
    return _build_backend_args(
        "/workspace",
        AgentConfig(backend="claude"),
        "prompt.md",
    )


def test_claude_argv_ignores_candidate_authored_mcp_config() -> None:
    """``--strict-mcp-config`` must be passed, with no ``--mcp-config``.

    The candidate worktree is the agent's cwd, so a candidate-authored
    ``.mcp.json`` is inside default discovery and the approval prompt is
    already suppressed by ``--dangerously-skip-permissions``.

    Positive control on the pinned 2.1.120 runtime: WITHOUT this flag a
    candidate-authored ``.mcp.json`` caused the agent to spawn the command it
    named; WITH the flag it did not.  So this closes candidate-controlled
    execution/egress AND the ``mcp-needs-auth-cache.json`` key channel.

    Catches: removing the flag, or adding a ``--mcp-config`` that would let
    servers back in.
    """
    argv = _claude_argv()
    assert "--strict-mcp-config" in argv, argv
    assert "--mcp-config" not in argv, (
        "HELIX has no MCP configuration surface in 0.3.0; any future one must "
        "be mounted outside the candidate worktree and omitted from the "
        "agent-visible tree"
    )


def test_mcp_guard_precedes_the_prompt_argument() -> None:
    """Non-vacuity: the flag must be a real option, not trailing prompt text.

    A flag appended after the positional prompt would be consumed as prompt
    content and silently do nothing, which is exactly the shape of failure this
    suite exists to catch.
    """
    argv = _claude_argv()
    assert argv.index("--strict-mcp-config") < len(argv) - 1
    assert argv[-1] != "--strict-mcp-config"
