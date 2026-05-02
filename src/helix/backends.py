"""Shared metadata for supported HELIX agent backends."""

from __future__ import annotations

from typing import Literal, TypeAlias


BackendName: TypeAlias = Literal["claude", "codex", "cursor", "gemini", "opencode"]

BACKENDS: tuple[BackendName, ...] = ("claude", "codex", "cursor", "gemini", "opencode")

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

BACKEND_AUTH_ENV: dict[str, tuple[str, ...]] = {
    "claude": ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"),
    "cursor": ("CURSOR_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "opencode": ("OPENCODE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"),
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
