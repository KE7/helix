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

# Environment that makes every candidate start from a fresh agent session.
#
# Candidates share one login volume per backend (mounted at /home/node), so a
# CLI that reads memory from an earlier session would carry state from
# candidate N-1 into candidate N.  Probed 2026-09-10 against the real CLIs:
# plant a fact in one one-shot session, ask for it in a fresh one, same cwd,
# no tools.
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
        # Credential warm -- see CREDENTIAL_WARM_SKIP_REASONS below for why
        # codex is the only backend with one.
        #
        # NOT ``codex login status``.  Measured against codex-cli 0.130.0 with
        # a synthetic credential in a throwaway volume: ``codex login status``
        # prints "Logged in using ChatGPT" and exits 0 without issuing a single
        # request, whether the stored ``last_refresh`` is minutes or 30 days
        # old.  It reads auth.json; it never takes the refresh path, so warming
        # with it would be a placebo.
        #
        # ``codex debug models`` renders the CLI's model catalog.  It loads
        # auth through the refreshing path, so it performs the refresh this
        # warm exists to perform, and it is free:
        #   * with a fresh credential it completes with ``--network none`` and
        #     makes no request at all -- the only thing it may write to the
        #     login volume is its own ``~/.codex/models_cache.json`` (a
        #     catalog cache with a 5-minute TTL), never the credential;
        #   * with a stale credential its only request is the OAuth token
        #     exchange, which the refreshed credential is then written back
        #     from.  No model is invoked and no quota is consumed either way.
        # Its exit code says nothing about the refresh: a rejected exchange is
        # logged and swallowed and the command still exits 0.  That is why
        # ``helix.sandbox.warm_backend_credential`` reads ``last_refresh``
        # back from ``auth.json`` and only reports ``warmed`` when the
        # credential is verifiably inside codex's refresh interval.  There is
        # no flag to bypass the catalog cache (``--bundled`` does the
        # opposite: it skips the refresh), so a cache younger than 5 minutes
        # can short-circuit the refresh; the read-back catches that too.
        # stdout is discarded because the catalog is ~200 KB and the command is
        # run for its side effect on the credential, not for its output;
        # stderr is kept so a failure stays diagnosable.
        "warm": ["sh", "-c", "set -eu; codex debug models >/dev/null"],
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


# ---------------------------------------------------------------------------
# Per-generation credential warm
# ---------------------------------------------------------------------------
#
# Every candidate container mounts the shared login volume read-write, which is
# what lets a backend CLI refresh its own OAuth token and keep the refreshed
# credential for the next candidate.  The hazard is the *first* moment after a
# credential goes stale: several candidates start at once, each decides
# independently that a refresh is due, and each posts the same single-use
# refresh token.  One wins; the rest are told the token was already consumed.
#
# ``helix.sandbox.warm_backend_credential`` narrows that window by running the
# command below once, in one container, before a generation dispatches any
# candidate -- so a refresh that is due at that moment happens under a single
# writer.  The credential is then fresh at the *start* of the generation; a
# token that crosses its refresh threshold during the generation (parent
# evaluations run before each mutation, and queued slots start later still)
# can still be raced by the candidates in flight.  The warm is skipped when
# at most one candidate can write the shared login at a time, since a single
# writer cannot race itself.
#
# A backend is warmed only when a command exists here that (a) actually takes
# the CLI's refresh path and (b) costs nothing.  Both halves are load-bearing:
# a command that never refreshes buys no safety, and a command that bills the
# operator's account once per generation would be worse than the race it
# prevents.  Backends with no entry are listed in
# CREDENTIAL_WARM_SKIP_REASONS with the reason they need none.


CREDENTIAL_WARM_SKIP_REASONS: dict[str, str] = {
    "claude": (
        "Claude Code serialises its own refresh: it takes a real cross-process "
        "lock file, retries while another process holds it, and re-reads the "
        "credential afterwards, so concurrent candidates cannot consume the "
        "same refresh token. Warming would add a container per generation and "
        "remove no hazard."
    ),
    "cursor": (
        "Cursor Agent never spends its stored refresh token: it re-exchanges "
        "an API key instead, so there is no single-use grant for candidates to "
        "compete over."
    ),
    "agy": (
        "No free Antigravity CLI command is known to take the refresh path. "
        "The registered status probe is a file test "
        "(`test -s ~/.gemini/antigravity-cli/antigravity-oauth-token`), which "
        "touches no credential path, so warming with it would be a placebo. "
        "`agy models` exits 0 even when logged out, and whether it takes the "
        "refresh path has not been measured; no agy credential exists to "
        "measure one against. Left unwarmed deliberately rather than warmed "
        "on a guess."
    ),
    "opencode": (
        "OpenCode refreshes an `oauth`-type credential only from inside the "
        "fetch wrapper that issues a model request -- read from opencode-ai "
        "1.14.24, which refreshes when `expires` has passed and writes the new "
        "credential back unlocked. There is therefore no command that performs "
        "that refresh without also invoking a model, and a per-generation model "
        "call on the operator's account is a worse cost than the race. "
        "`opencode providers list` was measured to be free -- it completes with "
        "`--network none` against an expired oauth credential and leaves "
        "auth.json byte-identical -- but for exactly that reason it refreshes "
        "nothing. `api`-type credentials never refresh and are not at risk."
    ),
}
"""Why a backend has no ``warm`` entry in :data:`BACKEND_AUTH_COMMANDS`.

Skipping is a correctness statement, not an optimisation: each entry records
either that the backend cannot lose a refresh race, or that no free command
would win it.
"""


def backend_credential_warm_skip_reason(backend: str) -> str | None:
    """Return why *backend* is not credential-warmed, or ``None`` if it is."""
    if "warm" in BACKEND_AUTH_COMMANDS.get(backend, {}):
        return None
    return CREDENTIAL_WARM_SKIP_REASONS.get(
        backend, "no credential-warm command is registered for this backend"
    )


def backend_display_name(backend: str) -> str:
    return BACKEND_DISPLAY_NAMES.get(backend, backend)
