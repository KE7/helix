"""Tests: per-candidate agent-state relocation away from the shared auth volume.

The invariant these tests defend is narrow and load-bearing: HELIX may move a
backend's *state* to a per-candidate location, but it must never move, copy,
name or shadow the *credential*, because the shared ``helix-auth-<backend>``
volume is what keeps token refresh and the CLIs' refresh locks working.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helix.agent_state import (
    AGENT_STATE_CONTAINER_ROOT,
    REJECTED_AGENT_STATE_KNOBS,
    STATE_RELOCATING_BACKENDS,
    UNRELOCATED_AGENT_STATE,
    agent_state_cli_args,
    agent_state_env,
    agent_state_subdirs,
    cursor_credential_hazard,
)
from helix.backends import BACKENDS
from helix.config import AgentConfig, SandboxConfig
from helix.mutator import _build_backend_args
from helix.sandbox import _docker_args, _prepare_agent_state_dir


# ---------------------------------------------------------------------------
# The state root must never sit inside the shared auth volume
# ---------------------------------------------------------------------------


def test_state_root_is_outside_the_auth_volume_mount() -> None:
    """The per-candidate mount must not be nested under ``/home/node``.

    Mounting inside the auth volume would create a new entry in it, which is
    exactly what the shared mount is not allowed to acquire.
    """
    assert not AGENT_STATE_CONTAINER_ROOT.startswith("/home/node")
    assert Path(AGENT_STATE_CONTAINER_ROOT).is_absolute()


# ---------------------------------------------------------------------------
# Per-backend knobs
# ---------------------------------------------------------------------------


def test_codex_relocates_state_databases_via_sqlite_home() -> None:
    args = agent_state_cli_args("codex", state_root=AGENT_STATE_CONTAINER_ROOT)
    assert args == ["-c", 'sqlite_home="/helix-state/codex"']


def test_opencode_relocates_only_the_database_file() -> None:
    env = agent_state_env("opencode", state_root=AGENT_STATE_CONTAINER_ROOT)
    assert env == {"OPENCODE_DB": "/helix-state/opencode/opencode.db"}


def test_cursor_relocates_state_via_config_dir() -> None:
    env = agent_state_env("cursor", state_root=AGENT_STATE_CONTAINER_ROOT)
    assert env == {"CURSOR_CONFIG_DIR": "/helix-state/cursor"}


@pytest.mark.parametrize("backend", ["claude", "gemini"])
def test_backends_without_a_safe_knob_get_nothing(backend: str) -> None:
    """claude and gemini have no knob that separates state from credential."""
    assert agent_state_env(backend, state_root=AGENT_STATE_CONTAINER_ROOT) == {}
    assert agent_state_cli_args(backend, state_root=AGENT_STATE_CONTAINER_ROOT) == []
    assert agent_state_subdirs(backend) == ()
    assert backend not in STATE_RELOCATING_BACKENDS


@pytest.mark.parametrize("backend", sorted(STATE_RELOCATING_BACKENDS))
def test_relocating_backends_emit_exactly_one_knob(backend: str) -> None:
    """Each backend uses one knob, so there is a single thing to re-verify."""
    knobs = list(agent_state_env(backend, state_root=AGENT_STATE_CONTAINER_ROOT))
    knobs += agent_state_cli_args(backend, state_root=AGENT_STATE_CONTAINER_ROOT)[:1]
    assert len(knobs) == 1, f"{backend} should relocate state with one knob"


# ---------------------------------------------------------------------------
# Credential-safety: the knobs we must never emit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", BACKENDS)
def test_no_backend_ever_receives_a_credential_moving_knob(backend: str) -> None:
    """Regression guard for the knobs recorded in REJECTED_AGENT_STATE_KNOBS.

    Every name below relocates the backend's credential file along with its
    state.  Emitting any of them would make an existing login invisible to the
    CLI, which is the failure this whole module exists to avoid.
    """
    forbidden = {
        "XDG_DATA_HOME",
        "XDG_CONFIG_HOME",
        "XDG_STATE_HOME",
        "HOME",
        "CODEX_HOME",
        "CLAUDE_CONFIG_DIR",
        "OPENCODE_CONFIG_DIR",
    }
    env = agent_state_env(backend, state_root=AGENT_STATE_CONTAINER_ROOT)
    assert forbidden.isdisjoint(env), (
        f"{backend} must not receive a credential-relocating env var"
    )
    rendered = " ".join(
        agent_state_cli_args(backend, state_root=AGENT_STATE_CONTAINER_ROOT)
    )
    assert "codex_home" not in rendered


def test_rejected_knobs_are_documented_with_a_reason() -> None:
    """Keep the 'do not re-try this' list honest and non-empty."""
    assert REJECTED_AGENT_STATE_KNOBS
    for name, reason in REJECTED_AGENT_STATE_KNOBS.items():
        assert ":" in name, f"{name} should read as 'backend:KNOB'"
        assert reason.strip()


def test_every_backend_has_a_leftover_state_entry() -> None:
    """Residue must be recorded for all backends, including the ones we fixed."""
    assert set(UNRELOCATED_AGENT_STATE) == set(BACKENDS)


# ---------------------------------------------------------------------------
# cursor's XDG_CONFIG_HOME hazard
# ---------------------------------------------------------------------------


def test_cursor_hazard_warns_when_xdg_config_home_is_present() -> None:
    warning = cursor_credential_hazard("cursor", {"XDG_CONFIG_HOME": "/somewhere"})
    assert warning is not None
    assert "XDG_CONFIG_HOME" in warning


def test_cursor_hazard_silent_when_absent_or_other_backend() -> None:
    assert cursor_credential_hazard("cursor", {"PATH": "/usr/bin"}) is None
    assert cursor_credential_hazard("codex", {"XDG_CONFIG_HOME": "/x"}) is None


# ---------------------------------------------------------------------------
# Sandbox wiring
# ---------------------------------------------------------------------------


def _agent_docker_args(backend: str, state_dir: Path | None) -> list[str]:
    return _docker_args(
        ["echo", "hi"],
        {},
        Path("/tmp/workspace"),
        SandboxConfig(enabled=True),
        "agent",
        "img:latest",
        backend,
        agent_state_dir=state_dir,
    )


def test_shared_auth_volume_mount_is_unchanged_by_relocation() -> None:
    """The credential mount must stay ``<volume>:/home/node:rw``, always."""
    for backend in BACKENDS:
        with_state = _agent_docker_args(backend, Path("/tmp/state"))
        without_state = _agent_docker_args(backend, None)
        expected = f"helix-auth-{backend}:/home/node:rw"
        assert expected in with_state
        assert expected in without_state
        # Relocation adds mounts; it never removes or rewrites the auth mount.
        assert with_state.count(expected) == without_state.count(expected) == 1


def test_state_dir_is_mounted_outside_the_auth_volume() -> None:
    args = _agent_docker_args("codex", Path("/tmp/state"))
    assert f"/tmp/state:{AGENT_STATE_CONTAINER_ROOT}:rw" in args
    # No mount target may be nested inside the shared volume.
    targets = [
        args[i + 1].split(":")[1] for i, a in enumerate(args) if a == "-v"
    ]
    nested = [t for t in targets if t.startswith("/home/node/")]
    assert not nested, f"mounts nested inside the auth volume: {nested}"


def test_relocation_env_reaches_the_container() -> None:
    args = _agent_docker_args("cursor", Path("/tmp/state"))
    assert f"CURSOR_CONFIG_DIR={AGENT_STATE_CONTAINER_ROOT}/cursor" in args
    args = _agent_docker_args("opencode", Path("/tmp/state"))
    assert f"OPENCODE_DB={AGENT_STATE_CONTAINER_ROOT}/opencode/opencode.db" in args


def test_evaluator_scope_gets_no_state_dir(tmp_path: Path) -> None:
    """Only agent commands touch the auth volume, so only they need relocation."""
    assert (
        _prepare_agent_state_dir(
            tmp_path, scope="evaluator", agent_backend="codex", image="img"
        )
        is None
    )


def test_no_state_dir_for_backends_without_a_knob(tmp_path: Path) -> None:
    assert (
        _prepare_agent_state_dir(
            tmp_path, scope="agent", agent_backend="claude", image="img"
        )
        is None
    )


def test_state_dir_lives_in_the_per_candidate_scratch_tree(
    tmp_path: Path, mocker
) -> None:
    """The directory must sit under the sandbox temp tree that is rmtree'd."""
    mocker.patch("helix.sandbox._docker_chown_workspace")
    state_dir = _prepare_agent_state_dir(
        tmp_path, scope="agent", agent_backend="codex", image="img"
    )
    assert state_dir is not None
    assert state_dir.is_relative_to(tmp_path)
    assert (state_dir / "codex").is_dir()


# ---------------------------------------------------------------------------
# Backend argv wiring
# ---------------------------------------------------------------------------


def test_codex_argv_carries_sqlite_home_when_sandboxed() -> None:
    args = _build_backend_args(
        "/workspace",
        AgentConfig(backend="codex"),
        "prompt.md",
        agent_state_root=AGENT_STATE_CONTAINER_ROOT,
    )
    assert "-c" in args
    assert 'sqlite_home="/helix-state/codex"' in args


def test_codex_argv_unchanged_without_a_sandbox() -> None:
    """Unsandboxed runs have no container state mount to point at."""
    args = _build_backend_args("/wt", AgentConfig(backend="codex"), "prompt.md")
    assert not any("sqlite_home" in a for a in args)


@pytest.mark.parametrize("backend", ["claude", "cursor", "gemini", "opencode"])
def test_non_codex_argv_never_carries_a_state_override(backend: str) -> None:
    args = _build_backend_args(
        "/workspace",
        AgentConfig(backend=backend),
        "prompt.md",
        agent_state_root=AGENT_STATE_CONTAINER_ROOT,
    )
    assert not any("sqlite_home" in a for a in args)


def test_local_opencode_db_stays_in_the_gitignored_state_dir(
    tmp_path: Path, mocker
) -> None:
    """Unsandboxed opencode keeps its database inside .helix_opencode_state/."""
    from helix.mutator import invoke_claude_code

    mock_run = mocker.patch("helix.mutator.subprocess.run")
    mock_run.return_value = MagicMock(
        stdout='{"type":"result","sessionID":"ses_abc"}\n', stderr="", returncode=0
    )
    invoke_claude_code(str(tmp_path), "prompt", AgentConfig(backend="opencode"))

    db_path = Path(mock_run.call_args[1]["env"]["OPENCODE_DB"])
    assert db_path.is_relative_to(tmp_path / ".helix_opencode_state")
    assert db_path.parent.is_dir(), "parent dir must exist before opencode starts"
