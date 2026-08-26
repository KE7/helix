"""Container proof that agent state relocates and the credential does not.

Each test asserts the same three things against a real backend container:

(a) the backend's state lands in the per-candidate directory;
(b) nothing new lands in the shared login volume;
(c) the CLI still reports itself authenticated.

Credentials are synthetic and live in throwaway volumes (see ``conftest``);
no test logs in, and none can reach a real ``helix-auth-*`` volume.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

from helix.agent_state import (
    AGENT_STATE_CONTAINER_ROOT,
    agent_state_cli_args,
    agent_state_env,
)


pytestmark = pytest.mark.docker_integration


CODEX_IMAGE = "ghcr.io/ke7/helix-evo-runner-codex:latest"
CURSOR_IMAGE = "ghcr.io/ke7/helix-evo-runner-cursor:latest"
OPENCODE_IMAGE = "ghcr.io/ke7/helix-evo-runner-opencode:latest"

# Deliberately malformed-but-well-shaped values.  They are never accepted by a
# real API; they only have to be present for a CLI to report a stored login.
SYNTHETIC_CODEX_AUTH = (
    'mkdir -p /home/node/.codex; printf "%s" '
    "'{\"OPENAI_API_KEY\":\"sk-SYNTHETIC-NOT-A-REAL-KEY\"}' "
    "> /home/node/.codex/auth.json"
)
SYNTHETIC_CURSOR_AUTH = (
    'mkdir -p /home/node/.config/cursor; printf "%s" '
    "'{\"accessToken\":\"SYNTHETIC\",\"refreshToken\":\"SYNTHETIC\"}' "
    "> /home/node/.config/cursor/auth.json"
)
SYNTHETIC_OPENCODE_AUTH = (
    'mkdir -p /home/node/.local/share/opencode; printf "%s" '
    "'{\"anthropic\":{\"type\":\"api\",\"key\":\"sk-ant-SYNTHETIC\"}}' "
    "> /home/node/.local/share/opencode/auth.json"
)


def _run_backend(
    *,
    image: str,
    volume: str,
    state_dir: Path,
    backend: str,
    shell_command: str,
    timeout: int = 120,
) -> str:
    """Run one backend container the way ``helix.sandbox`` would.

    The auth volume is mounted read-write at ``/home/node`` exactly as in
    production; the per-candidate state directory is a separate mount outside
    it, carrying the relocation env vars from ``helix.agent_state``.
    """
    (state_dir / backend).mkdir(parents=True, exist_ok=True)
    args = [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "node",
        "-v",
        f"{volume}:/home/node:rw",
        "-v",
        f"{state_dir}:{AGENT_STATE_CONTAINER_ROOT}:rw",
        "-e",
        "HOME=/home/node",
    ]
    for key, value in agent_state_env(
        backend, state_root=AGENT_STATE_CONTAINER_ROOT
    ).items():
        args.extend(["-e", f"{key}={value}"])
    args.extend([image, "sh", "-lc", shell_command])
    result = subprocess.run(
        args, capture_output=True, text=True, check=False, timeout=timeout
    )
    return result.stdout + result.stderr


def _relative_paths(state_dir: Path) -> set[str]:
    return {
        str(p.relative_to(state_dir))
        for p in state_dir.rglob("*")
        if p.is_file()
    }


# ---------------------------------------------------------------------------
# codex
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_codex_state_databases_relocate(
    tmp_path: Path, require_image, throwaway_volume, volume_listing
) -> None:
    image = require_image(CODEX_IMAGE)
    volume = throwaway_volume(image, SYNTHETIC_CODEX_AUTH)
    before = volume_listing(volume, image)

    state_dir = tmp_path / "agent-state"
    sqlite_args = " ".join(
        agent_state_cli_args("codex", state_root=AGENT_STATE_CONTAINER_ROOT)
    )
    _run_backend(
        image=image,
        volume=volume,
        state_dir=state_dir,
        backend="codex",
        # The state databases are opened during startup, well before any API
        # call, so a short timeout is enough and no model is ever reached
        # (the container has no network).
        shell_command=(
            "cd /tmp; timeout 20 codex exec --json "
            f"--dangerously-bypass-approvals-and-sandbox {sqlite_args} hi "
            ">/dev/null 2>&1 || true"
        ),
    )

    # (a) state landed per-candidate
    relocated = _relative_paths(state_dir)
    assert any(name.endswith("state_5.sqlite") for name in relocated), relocated
    assert any(name.endswith("logs_2.sqlite") for name in relocated), relocated

    # (b) no sqlite state landed in the shared volume
    after = volume_listing(volume, image)
    new_entries = after - before
    assert not [e for e in new_entries if ".sqlite" in e], new_entries

    # (c) the credential is untouched and still reported
    status = _run_backend(
        image=image,
        volume=volume,
        state_dir=state_dir,
        backend="codex",
        shell_command="codex login status",
    )
    assert "Logged in" in status, status


# ---------------------------------------------------------------------------
# cursor
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_cursor_state_relocates_and_login_survives(
    tmp_path: Path, require_image, throwaway_volume, volume_listing
) -> None:
    image = require_image(CURSOR_IMAGE)
    volume = throwaway_volume(image, SYNTHETIC_CURSOR_AUTH)
    before = volume_listing(volume, image)

    state_dir = tmp_path / "agent-state"
    status = _run_backend(
        image=image,
        volume=volume,
        state_dir=state_dir,
        backend="cursor",
        shell_command="timeout 60 cursor-agent status",
    )

    # (a) ~/.cursor state landed per-candidate
    assert "cursor/cli-config.json" in _relative_paths(state_dir)

    # (b) the shared volume is byte-for-byte unchanged
    assert volume_listing(volume, image) == before

    # (c) the shared credential is still found
    assert "Logged in" in status, status


def test_cursor_xdg_config_home_would_hide_the_credential(
    tmp_path: Path, require_image, throwaway_volume
) -> None:
    """Guard the rejected knob: XDG_CONFIG_HOME breaks cursor's login.

    This is why ``helix.agent_state`` uses CURSOR_CONFIG_DIR and why
    ``cursor_credential_hazard`` warns when a user routes XDG_CONFIG_HOME
    through ``passthrough_env``.
    """
    image = require_image(CURSOR_IMAGE)
    volume = throwaway_volume(image, SYNTHETIC_CURSOR_AUTH)
    state_dir = tmp_path / "agent-state"
    state_dir.mkdir()

    result = subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none", "--user", "node",
            "-v", f"{volume}:/home/node:rw",
            "-v", f"{state_dir}:{AGENT_STATE_CONTAINER_ROOT}:rw",
            "-e", "HOME=/home/node",
            "-e", f"XDG_CONFIG_HOME={AGENT_STATE_CONTAINER_ROOT}/cursor",
            image, "sh", "-lc", "timeout 60 cursor-agent status",
        ],
        capture_output=True, text=True, check=False, timeout=120,
    )
    assert "Not logged in" in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# opencode
# ---------------------------------------------------------------------------


@pytest.mark.timeout(300)
def test_opencode_database_relocates_and_credential_stays(
    tmp_path: Path, require_image, throwaway_volume, volume_listing
) -> None:
    image = require_image(OPENCODE_IMAGE)
    volume = throwaway_volume(image, SYNTHETIC_OPENCODE_AUTH)
    before = volume_listing(volume, image)

    state_dir = tmp_path / "agent-state"
    listing = _run_backend(
        image=image,
        volume=volume,
        state_dir=state_dir,
        backend="opencode",
        shell_command="cd /tmp; timeout 120 opencode auth list",
        timeout=200,
    )

    # (a) the database (which also carries token columns) landed per-candidate
    assert "opencode/opencode.db" in _relative_paths(state_dir)

    # (b) no database landed in the shared volume
    new_entries = volume_listing(volume, image) - before
    assert not [e for e in new_entries if "opencode.db" in e], new_entries

    # (c) the shared auth.json is still the credential source, and is seen
    assert "auth.json" in listing
    assert "0 credentials" not in listing, listing


@pytest.mark.timeout(300)
def test_opencode_xdg_data_home_would_hide_the_credential(
    tmp_path: Path, require_image, throwaway_volume
) -> None:
    """Guard the rejected knob: XDG_DATA_HOME moves auth.json with the database."""
    image = require_image(OPENCODE_IMAGE)
    volume = throwaway_volume(image, SYNTHETIC_OPENCODE_AUTH)
    state_dir = tmp_path / "agent-state"
    state_dir.mkdir()

    result = subprocess.run(
        [
            "docker", "run", "--rm", "--network", "none", "--user", "node",
            "-v", f"{volume}:/home/node:rw",
            "-v", f"{state_dir}:{AGENT_STATE_CONTAINER_ROOT}:rw",
            "-e", "HOME=/home/node",
            "-e", f"XDG_DATA_HOME={AGENT_STATE_CONTAINER_ROOT}",
            image, "sh", "-lc", "cd /tmp; timeout 120 opencode auth list",
        ],
        capture_output=True, text=True, check=False, timeout=200,
    )
    assert "0 credentials" in result.stdout + result.stderr


# ---------------------------------------------------------------------------
# The invariant that outranks all of the above
# ---------------------------------------------------------------------------


def test_real_auth_volumes_are_never_addressed() -> None:
    """No test in this suite may name a concrete login volume.

    Checked by inspecting the suite source rather than the daemon, so it holds
    even on a machine that has no login volumes at all.  Prose mentioning the
    volume family in the abstract is fine; a resolvable name is not.
    """
    concrete_name = re.compile(r"helix-auth-[a-z0-9]+")
    for path in (Path(__file__), Path(__file__).parent / "conftest.py"):
        for number, line in enumerate(path.read_text().splitlines(), start=1):
            if concrete_name.search(line):
                pytest.fail(
                    f"{path.name}:{number} names a real auth volume: {line.strip()}"
                )
