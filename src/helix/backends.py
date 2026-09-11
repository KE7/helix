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
            "-lc",
            'set -eu; test -s "${HOME:-/home/node}/.gemini/antigravity-cli/antigravity-oauth-token"',
        ],
        # Surgical: only remove agy's own state directory. ``~/.gemini`` also
        # holds unrelated Google CLI state, so a blanket ``rm -rf ~/.gemini``
        # would destroy state this backend does not own.
        "logout": [
            "sh",
            "-lc",
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
            "-lc",
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
# Native transcript stores (``sandbox.preserve_backend_transcripts``)
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
