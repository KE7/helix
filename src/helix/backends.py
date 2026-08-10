"""Shared metadata for supported HELIX agent backends."""

from __future__ import annotations

from typing import Literal, TypeAlias


BackendName: TypeAlias = Literal["claude", "codex", "cursor", "gemini", "opencode"]

BACKENDS: tuple[BackendName, ...] = ("claude", "codex", "cursor", "gemini", "opencode")


# ---------------------------------------------------------------------------
# AgentConfig.effort metadata
# ---------------------------------------------------------------------------
#
# ``agent.effort`` is the user-facing knob for "reasoning level / thinking
# budget" on backends that expose one.  The string value is forwarded to a
# backend-native CLI flag in ``helix.mutator``:
#
#   - ``claude``:    ``--effort <value>``   (Claude Code thinking budget)
#   - ``codex``:     ``-c model_reasoning_effort=<value>`` (Codex CLI config)
#   - ``opencode``:  ``--variant <value>``  (model variant selector)
#   - others:        silently ignored (no equivalent CLI surface)
#
# The two registries below let the config layer fail fast on bad combinations
# without hard-coding backend knowledge into ``HelixConfig``.

EFFORT_AWARE_BACKENDS: frozenset[BackendName] = frozenset(
    {"claude", "codex", "opencode"}
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
    "claude": frozenset({"low", "medium", "high"}),
    "codex": frozenset({"minimal", "low", "medium", "high", "xhigh"}),
    "opencode": None,  # variant strings are model-specific; opencode validates them.
}

BACKEND_DISPLAY_NAMES: dict[str, str] = {
    "claude": "Claude Code",
    "codex": "Codex CLI",
    "cursor": "Cursor Agent",
    "gemini": "Gemini CLI",
    "opencode": "OpenCode",
}

DEFAULT_BACKEND_IMAGES: dict[str, str] = {
    "claude": "ghcr.io/ke7/helix-evo-runner-claude:latest",
    "codex": "ghcr.io/ke7/helix-evo-runner-codex:latest",
    "cursor": "ghcr.io/ke7/helix-evo-runner-cursor:latest",
    "gemini": "ghcr.io/ke7/helix-evo-runner-gemini:latest",
    "opencode": "ghcr.io/ke7/helix-evo-runner-opencode:latest",
}

AGENT_LOGIN_IDENTITY_ENV: tuple[str, ...] = ("USER", "LOGNAME")
"""Non-secret identity variables preserved for every agent CLI.

Agent CLIs resolve stored interactive credentials independently.  Claude Code
on macOS, for example, needs ``USER`` to locate its Keychain entry even when
``HOME`` is present.  ``LOGNAME`` is harmless, conventional on Unix-like
systems, and provides compatibility for CLIs that use that spelling instead.
Keep this narrowly scoped to agent subprocesses: evaluator environments retain
their existing strict allowlist.
"""

ANTHROPIC_KEY_ENV: tuple[str, ...] = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")
"""Anthropic API-key credentials that HELIX never auto-forwards to an agent.

Anthropic backends authenticate through *login* — ``helix sandbox login
<backend>`` writes OAuth / subscription credentials into the persistent
``helix-auth-<backend>`` Docker volume that agent containers mount at
``/home/node``.  These env vars are the mutually exclusive alternative: when
one is present in an agent's environment the CLI bills against the API key and
the login credential sitting in the auth volume is ignored.

Auto-forwarding them therefore silently revokes the login the user just
performed.  HELIX resolves the ambiguity in favour of login and requires an
explicit opt-in (top-level ``passthrough_env`` or ``[env]`` in ``helix.toml``)
before an API key reaches an agent.  See ``helix.mutator._add_backend_auth_env``.
"""

ANTHROPIC_LOGIN_BACKENDS: frozenset[str] = frozenset({"claude", "opencode"})
"""Backends whose CLI reads :data:`ANTHROPIC_KEY_ENV` in preference to login."""

BACKEND_AUTH_ENV: dict[str, tuple[str, ...]] = {
    # NOTE: "claude" is deliberately absent, and "opencode" deliberately omits
    # ANTHROPIC_API_KEY — see ANTHROPIC_KEY_ENV above.  Only credentials with
    # no login-based alternative are auto-forwarded.
    "cursor": ("CURSOR_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "opencode": ("OPENCODE_API_KEY", "OPENAI_API_KEY"),
}

BACKEND_AUTH_COMMANDS: dict[str, dict[str, list[str]]] = {
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
    "gemini": {
        "login": ["gemini", "--skip-trust"],
        "status": ["gemini", "--version"],
        # The auth volume is mounted at /home/node and is shared across
        # backends, so logout must scrub only Gemini's state directory rather
        # than the whole home tree.
        "logout": [
            "sh",
            "-lc",
            'set -eu; rm -rf "/home/node/.gemini" "/home/node/.config/google-gemini"',
        ],
    },
    "opencode": {
        "login": ["opencode"],
        "status": ["opencode", "providers", "list"],
        "logout": ["opencode", "providers", "logout"],
    },
}


def backend_display_name(backend: str) -> str:
    return BACKEND_DISPLAY_NAMES.get(backend, backend)
