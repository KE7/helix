"""Shared metadata for supported HELIX agent backends."""

from __future__ import annotations

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
