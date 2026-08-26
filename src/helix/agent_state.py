"""Per-candidate relocation of agent-CLI state, away from the shared auth volume.

Why this module exists
----------------------
HELIX mounts one login volume per backend (``helix-auth-<backend>``) at
``/home/node`` read-write in *every* candidate container.  That mount is what
keeps the CLIs' token refresh and their cross-process refresh locks working, so
it is deliberately shared and must stay exactly as it is.

The problem is what else rides along in that volume.  Each CLI also writes its
*agent state* under ``$HOME`` -- transcripts, session databases, memories,
to-do lists.  Because the volume is shared, candidate N starts life reading
candidate N-1's leftovers.  For an evolutionary optimizer whose candidates are
meant to be independent samples, that is contamination of the experiment.  A
second, smaller consequence is that some of this state doubles as a credential
store (opencode's ``opencode.db`` carries ``access_token`` / ``refresh_token``
columns), so it should not be lying around in a shared location either.

What this module does
---------------------
For the backends that expose a knob separating *state* from *credential*, it
returns the environment variables and CLI arguments that point state at a
per-candidate directory.  The credential file is never named, never moved and
never copied: it keeps living in the shared volume exactly where the CLI put
it.  See :data:`UNRELOCATED_AGENT_STATE` for what each backend leaves behind.

Choosing a knob is not obvious, and the wrong choice silently breaks login.
The knobs below were each verified against the real CLI in a container with a
synthetic credential; the rejected alternatives are recorded in
:data:`REJECTED_AGENT_STATE_KNOBS` so nobody re-tries them.
"""

from __future__ import annotations

import json


AGENT_STATE_CONTAINER_ROOT = "/helix-state"
"""Container path where the per-candidate state directory is mounted.

Deliberately *outside* ``/home/node``.  Mounting anywhere under ``/home/node``
would place a mountpoint inside the shared auth volume, which creates a new
entry there -- the one thing the shared mount is not allowed to acquire.
"""


STATE_RELOCATING_BACKENDS: frozenset[str] = frozenset(
    {"codex", "cursor", "opencode"}
)
"""Backends with a knob that moves state without moving the credential.

``claude`` and ``gemini`` are absent on purpose; see
:data:`UNRELOCATED_AGENT_STATE`.
"""


UNRELOCATED_AGENT_STATE: dict[str, tuple[str, ...]] = {
    # Paths are relative to ``$HOME`` (the shared auth volume mount point) and
    # still carry cross-candidate state after relocation.  Named here so the
    # residue is discoverable rather than forgotten.
    "codex": (
        ".codex/sessions/<year>/<month>/<day>/rollout-*.jsonl",  # full transcript
        ".codex/shell_snapshots/*.sh",
        ".codex/memories/",
        ".codex/config.toml",
        ".codex/installation_id",
    ),
    "cursor": (),
    "opencode": (
        ".local/share/opencode/log/*.log",
        ".local/share/opencode/storage/session_diff/ses_*.json",
        ".local/share/opencode/storage/migration",
        ".config/opencode/.gitignore",
    ),
    "claude": (
        ".claude/projects/<slug>/*.jsonl",  # full transcript
        ".claude/projects/<slug>/memory/",
        ".claude/sessions/",
        ".claude/telemetry/",
        ".claude/backups/",
        ".claude.json",
    ),
    "gemini": (".gemini/", ".config/google-gemini/"),
}


REJECTED_AGENT_STATE_KNOBS: dict[str, str] = {
    # Each of these looks like the obvious knob and each one breaks login.
    "opencode:XDG_DATA_HOME": (
        "moves opencode.db AND auth.json together; with it set, "
        "`opencode auth list` reports 0 credentials"
    ),
    "cursor:XDG_CONFIG_HOME": (
        "moves cli-config.json AND auth.json together; with it set, "
        "`cursor-agent status` reports 'Not logged in'"
    ),
    "cursor:CURSOR_DATA_DIR": "accepted but relocates nothing; cli-config.json stays in $HOME",
    "codex:CODEX_HOME": "moves the state databases AND auth.json together",
    "claude:CLAUDE_CONFIG_DIR": (
        "moves the transcripts AND .credentials.json together, and pulls "
        ".claude.json in as well"
    ),
}


def _backend_state_dir(backend: str, state_root: str) -> str:
    """Return the per-backend subdirectory of the per-candidate state root."""
    return f"{state_root.rstrip('/')}/{backend}"


def agent_state_subdirs(backend: str) -> tuple[str, ...]:
    """Return directories to create under the state root before the container runs.

    The CLIs are not uniformly willing to create a missing parent directory for
    a relocated database, so HELIX creates them itself and keeps the behaviour
    deterministic across backends.  Paths are relative to the state root.
    """
    if backend not in STATE_RELOCATING_BACKENDS:
        return ()
    return (backend,)


def agent_state_env(backend: str, *, state_root: str) -> dict[str, str]:
    """Environment variables that point *backend*'s state at a per-candidate dir.

    Returns an empty mapping for backends without a safe knob, so callers can
    apply the result unconditionally.
    """
    state_dir = _backend_state_dir(backend, state_root)
    if backend == "opencode":
        # Verified: relocates opencode.db and its -wal/-shm companions alone.
        # auth.json stays at $HOME/.local/share/opencode/auth.json and the
        # refresh lock stays at $HOME/.local/state/opencode/locks/.
        return {"OPENCODE_DB": f"{state_dir}/opencode.db"}
    if backend == "cursor":
        # Verified: relocates the whole ~/.cursor state tree (cli-config.json,
        # agent-cli-state.json, statsig-cache.json, projects/<slug>/mcp-auth.json).
        # The credential is read from ${XDG_CONFIG_HOME||~/.config}/cursor/auth.json,
        # which this knob does not affect.
        return {"CURSOR_CONFIG_DIR": state_dir}
    return {}


def agent_state_cli_args(backend: str, *, state_root: str) -> list[str]:
    """CLI arguments that point *backend*'s state at a per-candidate dir.

    Used for backends whose only knob is a config override rather than an
    environment variable.
    """
    if backend == "codex":
        # Verified: relocates state_5.sqlite and logs_2.sqlite (plus their
        # -wal/-shm companions).  auth.json stays at $HOME/.codex/auth.json.
        #
        # ``-c key=value`` requires a TOML literal on the right-hand side;
        # ``json.dumps`` emits a double-quoted string that is also valid TOML
        # basic-string syntax, matching how ``model_reasoning_effort`` is
        # passed in ``helix.mutator._build_backend_args``.
        state_dir = _backend_state_dir(backend, state_root)
        return ["-c", f"sqlite_home={json.dumps(state_dir)}"]
    return []


def cursor_credential_hazard(backend: str, env: dict[str, str]) -> str | None:
    """Return a warning when *env* would hide cursor's shared credential.

    ``cursor-agent`` resolves its credential to
    ``${XDG_CONFIG_HOME||~/.config}/cursor/auth.json``.  HELIX never sets
    ``XDG_CONFIG_HOME`` itself -- the env scrub in ``helix.executor`` is an
    allowlist -- but a user can route it through ``passthrough_env`` or the
    ``[env]`` table in ``helix.toml``.  If they do, cursor stops seeing the
    shared login volume entirely and reports "Not logged in", which is worth a
    warning rather than a silent authentication failure mid-run.
    """
    if backend != "cursor" or "XDG_CONFIG_HOME" not in env:
        return None
    return (
        "XDG_CONFIG_HOME is set for the cursor backend. Cursor reads its "
        "credential from ${XDG_CONFIG_HOME}/cursor/auth.json, so this hides the "
        "shared login volume and cursor will report 'Not logged in'. Remove "
        "XDG_CONFIG_HOME from passthrough_env / [env] in helix.toml."
    )
