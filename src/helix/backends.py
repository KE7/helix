"""Shared metadata for supported HELIX agent backends."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, TypeAlias


BackendName: TypeAlias = Literal["agy", "claude", "codex", "cursor", "opencode"]

BACKENDS: tuple[BackendName, ...] = ("agy", "claude", "codex", "cursor", "opencode")


# ---------------------------------------------------------------------------
# AgentConfig.effort metadata
# ---------------------------------------------------------------------------
#
# ``agent.effort`` is the user-facing knob for "reasoning level / thinking
# budget" on backends that expose one.  The string value is forwarded to a
# backend-native CLI flag in ``helix.mutator``:
#
#   - ``agy``:       ``--effort <value>``   (Antigravity CLI reasoning effort)
#   - ``claude``:    ``--effort <value>``   (Claude Code thinking budget)
#   - ``codex``:     ``-c model_reasoning_effort=<value>`` (Codex CLI config)
#   - ``opencode``:  ``--variant <value>``  (model variant selector)
#   - others:        silently ignored (no equivalent CLI surface)
#
# The two registries below let the config layer fail fast on bad combinations
# without hard-coding backend knowledge into ``HelixConfig``.

EFFORT_AWARE_BACKENDS: frozenset[BackendName] = frozenset(
    {"agy", "claude", "codex", "opencode"}
)
"""Backends that propagate ``agent.effort`` to their underlying CLI."""

# ``None`` here means "any string accepted" — we still let strange values
# through and let the backend CLI decide, but a known set of values lets
# HELIX warn early on obvious typos like ``effrot = "high"``.
#
# NOTE: this allowlist is best-effort and may lag the upstream CLI.  When
# a backend ships a new tier (e.g. Anthropic's ``"minimal"`` on some
# surfaces), users will see a non-fatal "not a recognized value" warning
# until this map is updated; the value still passes through to the CLI.
EFFORT_VALID_VALUES: dict[BackendName, frozenset[str] | None] = {
    "agy": frozenset({"low", "medium", "high"}),
    "claude": frozenset({"low", "medium", "high"}),
    "codex": frozenset({"minimal", "low", "medium", "high", "xhigh"}),
    "opencode": None,  # variant strings are model-specific; opencode validates them.
}

BACKEND_DISPLAY_NAMES: dict[str, str] = {
    "agy": "Antigravity CLI",
    "claude": "Claude Code",
    "codex": "Codex CLI",
    "cursor": "Cursor Agent",
    "opencode": "OpenCode",
}

DEFAULT_BACKEND_IMAGES: dict[str, str] = {
    "agy": "ghcr.io/ke7/helix-evo-runner-agy:latest",
    "claude": "ghcr.io/ke7/helix-evo-runner-claude:latest",
    "codex": "ghcr.io/ke7/helix-evo-runner-codex:latest",
    "cursor": "ghcr.io/ke7/helix-evo-runner-cursor:latest",
    "opencode": "ghcr.io/ke7/helix-evo-runner-opencode:latest",
}

# Ambient credential variable names forwarded to a backend when the operator
# has them set. A backend with no entry here (``agy``, ``codex``) authenticates
# only through its sandbox login volume; naming a variable for it would be
# claiming a credential path this project has not established.
BACKEND_AUTH_ENV: dict[str, tuple[str, ...]] = {
    "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "cursor": ("CURSOR_API_KEY",),
    "opencode": ("OPENCODE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
}

# Environment that stops the agent CLIs' own cross-session memory.
#
# Candidates share one login volume per backend (mounted at /home/node), so a
# CLI that reads memory from an earlier session would carry state from
# candidate N-1 into candidate N.  Probed 2026-09-10 against the real CLIs:
# plant a fact in one one-shot session, ask for it in a fresh one, same cwd,
# no tools.  That probe measures the spontaneous channel only -- what a CLI
# recalls on its own.  It is not a session-isolation guarantee: a candidate
# with shell access can still write the shared HOME's config files
# (``~/.claude/CLAUDE.md``, ``~/.claude/settings.json``, ``~/.codex/AGENTS.md``,
# ``~/.codex/config.toml``), which later sessions load.  That channel is
# deliberately left open, because the volume is shared by design so that
# transcripts and a refreshed login persist across candidates.
#
# Applies to sandboxed and unsandboxed runs alike (``invoke_claude_code``
# sets it on the backend environment either way).  An operator who names the
# same key in ``[env]`` wins: the value here is a default, not an override.
#   claude   recalled it -- auto-memory, keyed by repo root, so it spans
#            worktrees and the shared HOME; CLAUDE_CODE_DISABLE_AUTO_MEMORY=1
#            stops it and no memory directory is created.
#   codex    did not; its ``memories`` feature flag is off by default and is
#            pinned off in argv (``-c features.memories=false``, see
#            ``helix.mutator._build_backend_args``) because it is a config
#            override, not an environment variable.
#   agy, cursor, opencode   did not; they read nothing from a prior session,
#            so there is nothing to disable and no entry here.
# Transcripts and session databases are still written to the login volume so
# they can be read after a run.
BACKEND_FRESH_SESSION_ENV: dict[str, dict[str, str]] = {
    "claude": {"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1"},
}

# Every shell entry below is ``sh -c``, never ``sh -lc``: a login shell sources
# ``/etc/profile`` and ``$HOME/.profile`` from the shared login volume, which
# every candidate container mounts read-write, so ``-l`` would let a candidate
# plant code that runs in the next auth command.  PATH is pinned with ``-e`` by
# ``helix.sandbox.sandbox_auth_docker_args``; nothing here needs a profile.
BACKEND_AUTH_COMMANDS: dict[str, dict[str, list[str]]] = {
    "agy": {
        # No dedicated non-interactive login subcommand; the bare interactive
        # launch is the login flow, same pattern as ``opencode`` below.
        "login": ["agy"],
        # ``agy models`` returns exit 0 even when logged out, so it cannot be
        # used as the status signal. Probe the credential file directly
        # instead, mirroring claude's file-probe pattern below. The path was
        # derived by tracing the CLI's own file opens and has not yet been
        # confirmed against a completed sign-in.
        "status": [
            "sh",
            "-c",
            'set -eu; test -s "${HOME:-/home/node}/.gemini/antigravity-cli/antigravity-oauth-token"',
        ],
        # Surgical: only remove agy's own state directory. ``~/.gemini`` also
        # holds unrelated Google CLI state, so a blanket ``rm -rf ~/.gemini``
        # would destroy state this backend does not own.
        "logout": [
            "sh",
            "-c",
            'set -eu; rm -rf "${HOME:-/home/node}/.gemini/antigravity-cli"',
        ],
    },
    "claude": {
        "login": ["claude", "auth", "login", "--claudeai"],
        # ``claude auth status --text`` returns 0 even when there are no
        # credentials, so we additionally require the on-disk credential file
        # written by ``claude auth login`` to be non-empty. Using a file probe
        # avoids depending on the exact human-readable wording (which is
        # localised in some CLI versions).
        "status": [
            "sh",
            "-c",
            "set -eu; "
            "claude auth status --text 2>&1 || true; "
            'test -s "${HOME:-/home/node}/.claude/.credentials.json"',
        ],
        "logout": ["claude", "auth", "logout"],
    },
    "codex": {
        "login": ["codex", "login", "--device-auth"],
        "status": ["codex", "login", "status"],
        "logout": ["codex", "logout"],
    },
    "cursor": {
        "login": ["cursor-agent", "login"],
        "status": ["cursor-agent", "status"],
        "logout": ["cursor-agent", "logout"],
    },
    "opencode": {
        "login": ["opencode"],
        "status": ["opencode", "providers", "list"],
        "logout": ["opencode", "providers", "logout"],
    },
}


def backend_display_name(backend: str) -> str:
    return BACKEND_DISPLAY_NAMES.get(backend, backend)


# ---------------------------------------------------------------------------
# Native transcript stores
# ---------------------------------------------------------------------------
#
# Every backend CLI keeps its own durable record of a session somewhere under
# ``$HOME`` (``/home/node`` inside the sandbox auth volume).  The table below
# is the single place that knows where, keyed by backend, so that
# ``helix.sandbox`` (copy out of the ``helix-auth-<backend>`` volume) and
# ``helix.mutator`` (copy from the operator's ``$HOME`` when unsandboxed,
# then record the artifact) can share one locator.  Paths were established
# against codex 0.154, opencode 1.18, agy 1.1 and cursor-agent 2026.08; they
# are best-effort and a miss is recorded, never raised.
#
# ``claude`` is listed for completeness of ``id_keys`` only: its source path is
# ``SandboxConfig.claude_transcript_root`` (or ``HELIX_CLAUDE_TRANSCRIPT_ROOT``)
# and is handled by the pre-existing claude-specific code path.

# Session ids are interpolated into shell globs and file names, so only accept
# the shapes the CLIs actually emit (UUIDs, ``ses_…``, ``thr_…``).
TRANSCRIPT_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class BackendTranscriptSource:
    """Where a backend CLI persists one session's transcript.

    ``id_keys`` are the field names, in priority order, that carry the session
    id in the backend's structured stdout.  ``files`` maps a ``$HOME``-relative
    glob (``{session_id}`` is substituted; ``*`` is left for the shell/glob) to
    the file name written under ``<transcript_artifact_dir>/<backend>/``.
    ``sqlite_export`` names the staged raw sqlite copy in ``files`` and the
    ``(table, session_id_column)`` pairs whose rows for this session are
    exported to ``{session_id}.jsonl`` once the raw copy is on the host.
    """

    id_keys: tuple[str, ...]
    files: tuple[tuple[str, str], ...]
    sqlite_export: tuple[str, tuple[tuple[str, str], ...]] | None = None


BACKEND_TRANSCRIPT_SOURCES: dict[str, BackendTranscriptSource] = {
    "claude": BackendTranscriptSource(
        id_keys=("session_id", "sessionId", "sessionID"),
        files=(),
    ),
    # ``codex exec --json`` opens with ``thread.started {thread_id}``; the
    # rollout file is date-partitioned by local start time and its name ends
    # in that thread id.  ``session_meta.payload.id`` inside equals it.
    "codex": BackendTranscriptSource(
        id_keys=("thread_id", "session_id", "sessionId"),
        files=(
            (
                ".codex/sessions/*/*/*/rollout-*-{session_id}.jsonl",
                "{session_id}.jsonl",
            ),
        ),
    ),
    # ``cursor-agent --output-format stream-json`` reports the session id on
    # its ``system`` event.  Cursor keeps a binary blob store under
    # ``~/.cursor/chats/<cwd-hash>/<session>/store.db`` and a plain JSONL
    # transcript under ``~/.cursor/projects/<cwd-slug>/agent-transcripts``;
    # the JSONL is the one preserved.
    "cursor": BackendTranscriptSource(
        id_keys=("session_id", "sessionId", "sessionID"),
        files=(
            (
                ".cursor/projects/*/agent-transcripts/{session_id}/{session_id}.jsonl",
                "{session_id}.jsonl",
            ),
        ),
    ),
    # Antigravity keeps the readable transcript (and a ``_full`` variant with
    # tool payloads) per conversation under ``brain/``.  The sibling
    # ``conversations/<id>.db`` / ``.pb`` are the same conversation in binary
    # form and are not copied.
    "agy": BackendTranscriptSource(
        id_keys=("conversation_id",),
        files=(
            (
                ".gemini/antigravity-cli/brain/{session_id}/.system_generated/logs/transcript.jsonl",
                "{session_id}.jsonl",
            ),
            (
                ".gemini/antigravity-cli/brain/{session_id}/.system_generated/logs/transcript_full.jsonl",
                "{session_id}.full.jsonl",
            ),
        ),
    ),
    # OpenCode >= 1.x persists sessions only in one sqlite database shared by
    # every session (the older ``storage/`` JSON tree is no longer written).
    # The db and its WAL are staged next to the artifacts under a dot-prefixed
    # name, this session's ``session``/``message``/``part`` rows are exported
    # to ``{session_id}.jsonl``, and the staged copy is removed.
    "opencode": BackendTranscriptSource(
        id_keys=("sessionID", "session_id", "sessionId"),
        files=(
            (".local/share/opencode/opencode.db", ".{session_id}.opencode.db"),
            (".local/share/opencode/opencode.db-wal", ".{session_id}.opencode.db-wal"),
        ),
        sqlite_export=(
            ".{session_id}.opencode.db",
            (("session", "id"), ("message", "session_id"), ("part", "session_id")),
        ),
    ),
}


def transcript_session_id_keys(backend: str) -> tuple[str, ...]:
    """Structured-stdout field names that carry ``backend``'s session id."""
    source = BACKEND_TRANSCRIPT_SOURCES.get(backend)
    if source is None:
        return ("session_id", "sessionId", "sessionID")
    return source.id_keys


# ---------------------------------------------------------------------------
# Per-generation sandbox state reset
# ---------------------------------------------------------------------------
#
# Every sandboxed agent run leaves session state behind in the
# ``helix-auth-<backend>`` volume: transcripts, memories, shell snapshots,
# indexes.  Left alone it accumulates across a whole evolution run and leaks
# earlier candidates' work into later ones through the CLI's own history and
# memory features.  ``helix.sandbox.reset_sandbox_agent_state`` deletes the
# ``$HOME``-relative paths below at the start of every generation, after the
# previous generation's transcripts were copied into the run.  Credentials and
# operator configuration are deliberately absent from this table; the unit
# test ``test_sandbox_state_paths_never_match_credentials`` guards that.  The
# entries were verified against local installs of codex 0.154, opencode 1.18,
# agy 1.1, cursor-agent 2026.08 and claude code; unknown directories were left
# alone rather than guessed at.
BACKEND_STATE_PATHS: dict[str, tuple[str, ...]] = {
    "claude": (
        ".claude/projects",  # per-project transcripts, memory/, todo state
        ".claude/sessions",  # session index
        ".claude/telemetry",
        ".claude/backups",  # settings/conversation backups
        ".claude/history.jsonl",  # prompt history
        ".claude/shell-snapshots",
        ".claude/file-history",  # pre-edit file copies
        ".claude/session-env",
        ".claude/debug",  # per-session debug logs
        ".claude/paste-cache",
        # kept: .credentials.json, settings.json, plugins, skills, cache, ~/.claude.json
    ),
    "codex": (
        ".codex/sessions",  # rollout JSONL transcripts
        ".codex/archived_sessions",
        ".codex/session_index.jsonl",
        ".codex/history.jsonl",  # prompt history
        ".codex/memories",  # auto-memory text
        ".codex/memories_*.sqlite*",
        ".codex/shell_snapshots",
        ".codex/state_*.sqlite*",  # thread state (state_5.sqlite + wal/shm)
        ".codex/thread_history*.sqlite*",
        ".codex/logs_*.sqlite*",
        ".codex/models_cache.json",
        ".codex/log",
        # kept: auth.json, config.toml, AGENTS.md, installation_id, version.json,
        #       .codex-global-state.json, skills, rules, plugins
    ),
    "agy": (
        ".gemini/antigravity-cli/conversations",  # per-conversation .db/.pb
        ".gemini/antigravity-cli/brain",  # transcripts, steps, task logs
        ".gemini/antigravity-cli/cache",
        ".gemini/antigravity-cli/log",
        ".gemini/antigravity-cli/cli.log",
        ".gemini/antigravity-cli/crashes",
        ".gemini/antigravity-cli/presence",
        ".gemini/antigravity-cli/knowledge",  # auto-memory
        ".gemini/antigravity-cli/history.jsonl",
        ".gemini/antigravity-cli/conversation_summaries.db",
        # kept: antigravity-oauth-token, settings.json, installation_id,
        #       keybindings.json, mcp, plugins, bin, builtin, updater
    ),
    "cursor": (
        ".cursor/projects",  # agent-transcripts/, per-project state
        ".cursor/chats",  # per-session store.db + meta.json
        ".cursor/ai-tracking",
        ".cursor/browser-logs",
        ".cursor/prompt_history.json",
        ".cursor/snapshots",
        ".cursor/worktrees",
        # kept: cli-config.json, mcp.json, agent-cli-state.json, extensions,
        #       plugins, skills-cursor, ~/.config/cursor/auth.json
    ),
    "opencode": (
        ".local/share/opencode/storage",  # legacy per-session JSON tree
        ".local/share/opencode/snapshot",  # per-session git snapshots
        ".local/share/opencode/tool-output",
        ".local/share/opencode/log",
        ".local/state/opencode/prompt-history.jsonl",
        # opencode.db is NOT a path here: its ``account`` table holds the
        # opencode account's access/refresh tokens, so the file is kept and
        # only session rows are deleted (``OPENCODE_STATE_TABLES``).
        # kept: auth.json, account.json, bin, ~/.local/state/opencode/locks
    ),
}

# Tables in ``~/.local/share/opencode/opencode.db`` whose rows are per-session
# state.  ``session`` is the parent; the others reference it by ``session_id``
# (``part`` also by ``message_id``).  Listed children-first so the deletes do
# not depend on foreign-key enforcement being on.
OPENCODE_STATE_TABLES: tuple[str, ...] = ("part", "message", "session")
