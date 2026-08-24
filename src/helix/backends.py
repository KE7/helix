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

AGENT_LOGIN_IDENTITY_ENV: tuple[str, ...] = ("USER", "LOGNAME")
"""Non-secret identity variables preserved for host/unsandboxed agent CLIs.

Agent CLIs resolve stored interactive credentials on their own, and some need
to know who the host user is to do it: Claude Code on macOS, for example, uses
``USER`` to locate its Keychain entry even when ``HOME`` is set. ``LOGNAME`` is
the conventional Unix spelling of the same thing and costs nothing to include.

Host path only. A sandboxed agent gets neither name: its credential is a file
under a tmpfs ``HOME``, so there is no keychain for an identity to unlock.
Evaluator environments keep their existing strict allowlist.
"""

BACKEND_AUTH_COMMANDS: dict[str, dict[str, list[str]]] = {
    "agy": {
        # No dedicated non-interactive login subcommand; the bare interactive
        # launch is the login flow, same pattern as ``opencode`` below.
        "login": ["agy"],
        # ``agy models`` returns exit 0 even when logged out, so it cannot be
        # used as the status signal. Probe the credential file directly
        # instead, mirroring claude's file-probe pattern above.
        "status": [
            "sh",
            "-lc",
            'set -eu; test -s "${HOME:-/home/node}/.gemini/antigravity-cli/antigravity-oauth-token"',
        ],
        # Surgical: only remove agy's own state directory. ``~/.gemini`` is
        # shared with legacy gemini-cli state (``~/.gemini/config``), so a
        # blanket ``rm -rf ~/.gemini`` would destroy state this backend
        # doesn't own.
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
