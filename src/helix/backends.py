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
        # The auth volume is mounted at /home/node, so logout must scrub only
        # Gemini's state directory rather than the whole home tree.
        #
        # NOTE: the volume is NOT shared across backends -- an earlier version
        # of this comment said so and was wrong.  ``sandbox_auth_volume_name``
        # derives a per-backend name (``helix-auth-<backend>``), and five
        # distinct volumes exist in practice.  What the volume IS shared
        # across is RUNS, which is the direction that causes cross-candidate
        # state to leak between candidates of different runs.  The narrow
        # scrub above is still correct; only its stated rationale was.
        #
        # Do not extend this scrub to "clean up" pre-existing root-owned
        # entries.  helix-auth-claude contains uid-0 files (two backups, two
        # transcripts) written out-of-band by an ad-hoc root container; no
        # production path in sandbox.py can produce them.
        #
        # Deletion is prohibited by POLICY -- they are incident evidence.  Be
        # explicit about the mechanics so this guard is not "corrected" away:
        # a ``--user node`` container CAN unlink and rename-over them, because
        # POSIX unlink needs write+execute on the PARENT directory (both
        # parents are 1000:1000 drwxr-xr-x, no sticky bit), not ownership of
        # the file.  An ``rm -rf`` here would SUCCEED and destroy the
        # evidence; that is the reason to keep the scrub narrow, not an
        # inability to delete.
        #
        # The operative technical hazard is READ, not delete: those files are
        # 0600 root, so ``--user node`` cannot read them.  Per-run isolation
        # MASKS them so no candidate inherits them.
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
