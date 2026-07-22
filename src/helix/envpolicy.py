"""Per-scope environment provenance for HELIX subprocesses and containers.

Why this module exists
----------------------
Environment construction used to be a sequence of mutations to a plain
``dict`` performed across four files, and *nothing recorded why a variable was
present*.  By the time the dict reached :func:`helix.sandbox._docker_args`, a
credential and ``PATH`` were indistinguishable.  That is what allowed a
credential to be absent at the scrubber — which every test asserted on — and
still be handed to the container, because ``_add_backend_auth_env`` re-added it
from ``os.environ`` *downstream* of the scrubber.

Three separate origins could place a variable in a mutation-agent container:

1. the per-backend table in :mod:`helix.backends` (``BACKEND_AUTH_ENV``),
2. ``passthrough_env`` / ``env`` configuration fields,
3. the ``HELIX_*`` prefix wildcard in the scrubber.

Because the backend-table re-add was guarded with ``key not in env``, origins
(2) and (3) *won* over (1) — so a policy gating only the backend table is
theatre.  This module replaces all three with grants carrying an explicit
origin and an explicit set of authorized scopes, and
:func:`helix.sandbox._docker_args` independently re-checks every grant it is
about to emit.

The scope authorization table below IS the policy.  Two properties fall out of
it *mechanically*, not by convention:

* **A backend rename cannot change agent credential flow**, because under
  ``sandbox.enabled`` the ``backend_auth_env`` origin grants nothing to agent
  scope.  Flipping ``backend = "claude"`` to ``"opencode"`` — a one-word edit
  that previously handed the solver credential to the mutation agent — is
  inert.
* **A sidecar secret cannot reach an agent by cross-field union**, because
  ``sidecar_passthrough`` has no agent scope at all and the union formerly
  performed in ``evolution.py`` no longer exists.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

from helix.backends import BACKEND_AUTH_ENV


Scope = Literal["agent", "evaluator", "sidecar"]

Origin = Literal[
    "helix_internal",
    "config_env",
    "config_passthrough",
    "agent_passthrough",
    "sidecar_passthrough",
    "auth_env_allow",
    "backend_auth_env",
]


# ---------------------------------------------------------------------------
# Credential-suppression policy
# ---------------------------------------------------------------------------

# Variables that turn OAuth mode OFF in the backend CLI, so that setting them
# prevents *container-side* token refresh — both the proactive-on-expiry path
# and the 401-triggered path.
#
# For claude this is established from the pinned runner's own predicate: the
# first two make the OAuth-mode predicate false, and CLAUDE_CODE_OAUTH_TOKEN
# makes the credential accessor return a synthetic record with a null refresh
# token, which disables refresh *permanently*.
#
# Backends whose refresh semantics are not established default to True
# (assume suppression, and say so).  Guessing "no suppression" would let a
# mounted auth volume rot silently, which is the failure this release removes.
OAUTH_SUPPRESSING_ENV: dict[str, frozenset[str]] = {
    "claude": frozenset(
        {
            "ANTHROPIC_API_KEY",
            "ANTHROPIC_AUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN",
            "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
            "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
        }
    ),
}

# Never permitted in ``sandbox.auth_env_allow`` under any backend (R6).
# CLAUDE_CODE_OAUTH_TOKEN is not an env-mode mechanism: it does not merely
# bypass refresh, it corrupts the credential record so refresh can never
# resume.  The FILE_DESCRIPTOR variants sit in the same selector in the pinned
# bundle and are the obvious workaround someone reaches for when the first is
# rejected.
FORBIDDEN_AUTH_ENV_NAMES: frozenset[str] = frozenset(
    {
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN_FILE_DESCRIPTOR",
        "CLAUDE_CODE_API_KEY_FILE_DESCRIPTOR",
    }
)


def env_var_suppresses_oauth(backend: str, name: str) -> bool:
    """Return True iff ``name`` disables container-side OAuth refresh.

    Consulted by config validation, by the auth-volume mount-mode decision,
    and by the env-mode startup disclosure.  Unknown backends default to
    ``True``: assuming suppression is the safe direction, because the failure
    mode of guessing wrong is a silently rotting credential volume.
    """
    known = OAUTH_SUPPRESSING_ENV.get(backend)
    if known is None:
        return True
    return name in known


# ---------------------------------------------------------------------------
# HELIX_* registry (R5)
# ---------------------------------------------------------------------------

# The ``HELIX_*`` prefix wildcard is a namespace, not a boundary: nothing stops
# an operator naming a credential ``HELIX_OPENAI_KEY``, and nothing in any
# config file would record that they had.  The wildcard is preserved for
# evaluator and sidecar scope (where it is actually used and the blast radius
# is HELIX's own trusted code) and replaced for SANDBOXED agent scope by this
# explicit registry plus ``sandbox.agent_passthrough_env``.
#
# These names are the legitimate uses enumerated across examples/, docs/ and
# src/: the ASI debug log channel, artifact-name overrides, and the transcript
# root.  Names HELIX sets itself are passed explicitly and do not rely on the
# wildcard.
AGENT_HELIX_ENV_REGISTRY: frozenset[str] = frozenset(
    {
        "HELIX_SPLIT",
        "HELIX_INSTANCE_IDS",
        "HELIX_EVALUATOR_ENDPOINT",
        "HELIX_ASI_LOG",
        "HELIX_ASI_LOG_ENV",
        "HELIX_ARTIFACT_NAMES",
        "HELIX_CLAUDE_TRANSCRIPT_ROOT",
        "HELIX_DIR",
        "HELIX_RESULT",
        "HELIX_TOML_TEMPLATE",
    }
)


# ---------------------------------------------------------------------------
# Grants
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvGrant:
    """One environment variable, with a recorded reason for being present.

    ``value`` is credential material in the general case.  ``__repr__`` is
    overridden to redact it so that a grant landing in a log line, a traceback
    or a pytest assertion diff cannot leak a secret.
    """

    name: str
    value: str
    origin: Origin
    scopes: frozenset[Scope]

    def __repr__(self) -> str:  # pragma: no cover - trivial, but load-bearing
        return (
            f"EnvGrant(name={self.name!r}, value=<redacted len="
            f"{len(self.value)}>, origin={self.origin!r}, "
            f"scopes={sorted(self.scopes)!r})"
        )

    def authorizes(self, scope: Scope) -> bool:
        return scope in self.scopes


# The scope authorization table, expressed as data.  ``agent`` membership is
# conditional for two origins and is therefore decided in ``resolve_env``
# rather than here; this map holds the unconditional part.
_ALL_SCOPES: frozenset[Scope] = frozenset({"agent", "evaluator", "sidecar"})
_EVAL_SIDECAR: frozenset[Scope] = frozenset({"evaluator", "sidecar"})
_AGENT_ONLY: frozenset[Scope] = frozenset({"agent"})

_STATIC_SCOPES: dict[Origin, frozenset[Scope]] = {
    "config_env": frozenset({"agent", "evaluator", "sidecar"}),
    "config_passthrough": frozenset({"evaluator", "sidecar"}),
    "agent_passthrough": frozenset({"agent"}),
    "sidecar_passthrough": frozenset({"sidecar"}),
}


def _grant(
    name: str, value: str, origin: Origin, scopes: frozenset[Scope]
) -> EnvGrant:
    return EnvGrant(name=name, value=value, origin=origin, scopes=scopes)


def resolve_env_grants(
    *,
    scope: Scope,
    backend: str,
    sandbox_enabled: bool,
    auth_mode: str | None,
    auth_env_allow: list[str] | None = None,
    agent_passthrough_env: list[str] | None = None,
    config_passthrough_env: list[str] | None = None,
    sidecar_passthrough_env: list[str] | None = None,
    config_env: dict[str, str] | None = None,
    split: str | None = None,
    instance_ids: list[str] | None = None,
    environ: dict[str, str] | None = None,
) -> list[EnvGrant]:
    """Resolve the full grant list for one scope.  The ONLY policy decision point.

    This is the single function permitted to decide what may enter a container
    or subprocess environment.  ``_docker_args`` re-checks the result
    independently; that second check is defence in depth against a future call
    site added downstream, which is *exactly* how the original bug arose.

    ``auth_mode`` is ``None`` when ``sandbox_enabled`` is False (R1: the auth
    mode is inert for non-sandboxed runs, whose behaviour is preserved
    exactly).
    """
    env = dict(os.environ if environ is None else environ)
    grants: list[EnvGrant] = []
    all_scopes = _ALL_SCOPES

    # --- helix_internal -------------------------------------------------
    for key in ("PATH", "HOME"):
        if key in env:
            grants.append(_grant(key, env[key], "helix_internal", all_scopes))
    if split is not None:
        grants.append(_grant("HELIX_SPLIT", split, "helix_internal", all_scopes))
    if instance_ids is not None:
        grants.append(
            _grant(
                "HELIX_INSTANCE_IDS",
                ",".join(str(i) for i in instance_ids),
                "helix_internal",
                all_scopes,
            )
        )

    # R5: the HELIX_* wildcard survives for evaluator/sidecar scope, but for a
    # SANDBOXED agent it is replaced by the explicit registry.  A non-sandboxed
    # agent keeps the wildcard so today's behaviour is preserved exactly (R1).
    restrict_agent = sandbox_enabled
    for key, value in env.items():
        if not key.startswith("HELIX_"):
            continue
        if key in ("HELIX_SPLIT", "HELIX_INSTANCE_IDS"):
            continue
        wildcard_scopes: frozenset[Scope] = (
            _EVAL_SIDECAR
            if restrict_agent and key not in AGENT_HELIX_ENV_REGISTRY
            else all_scopes
        )
        grants.append(_grant(key, value, "helix_internal", wildcard_scopes))

    # --- credential origins ---------------------------------------------
    # Emitted BEFORE the config origins because later grants win on a name
    # collision, and credential grants must have the LOWEST precedence. That
    # reproduces the previous ``key not in env`` guard exactly: an explicit
    # ``env = { ANTHROPIC_API_KEY = "..." }`` in helix.toml continues to
    # override the host value rather than being silently replaced by it.
    #
    # Exactly one of these two can grant to agent scope, and never both:
    #   * sandbox disabled -> backend table (today's behaviour, preserved)
    #   * sandbox enabled + auth == "env" -> auth_env_allow, as a REPLACEMENT
    #     for the backend table, not a union with it.
    #   * sandbox enabled + auth == "volume" -> neither. The volume is the
    #     only credential path; there is no fallback in either direction (R4).
    if not sandbox_enabled:
        for key in BACKEND_AUTH_ENV.get(backend, ()):
            if key in env:
                grants.append(_grant(key, env[key], "backend_auth_env", _AGENT_ONLY))
    elif auth_mode == "env":
        for key in auth_env_allow or []:
            if key in env:
                grants.append(_grant(key, env[key], "auth_env_allow", _AGENT_ONLY))

    # --- config_passthrough ---------------------------------------------
    # Top-level ``passthrough_env`` no longer grants to a SANDBOXED agent.
    # This is the channel that is live in a shipped lane config today, and it
    # is why a backend-table-only fix would not have fixed that lane.
    passthrough_scopes: frozenset[Scope] = (
        _EVAL_SIDECAR if sandbox_enabled else _ALL_SCOPES
    )
    for key in config_passthrough_env or []:
        if key in env:
            grants.append(
                _grant(key, env[key], "config_passthrough", passthrough_scopes)
            )

    # --- agent_passthrough (new) ----------------------------------------
    for key in agent_passthrough_env or []:
        if key in env:
            grants.append(_grant(key, env[key], "agent_passthrough", _AGENT_ONLY))

    # --- sidecar_passthrough --------------------------------------------
    for key in sidecar_passthrough_env or []:
        if key in env:
            grants.append(
                _grant(
                    key,
                    env[key],
                    "sidecar_passthrough",
                    _STATIC_SCOPES["sidecar_passthrough"],
                )
            )

    # --- config_env (literal values in-file) ----------------------------
    for key, value in (config_env or {}).items():
        grants.append(
            _grant(str(key), str(value), "config_env", _STATIC_SCOPES["config_env"])
        )

    return grants_for_scope(grants, scope)


def sidecar_passthrough_names(
    config_passthrough_env: list[str],
    sidecar_passthrough_env: list[str],
) -> list[str]:
    """Names the evaluator sidecar may receive, per the scope table.

    Both ``config_passthrough`` and ``sidecar_passthrough`` authorize sidecar
    scope, so the sidecar's own list is unchanged by this release.  What
    changed is the *other* direction: neither origin grants agent scope under
    a sandbox any more, so a name added to either field can no longer reach a
    mutation agent.  Expressing it here keeps the scope table the single
    source of truth rather than an ad-hoc union performed at the call site.

    Names are NOT filtered against ``os.environ`` — the sidecar launcher does
    that itself, and filtering here would silently drop a variable that is set
    later.
    """
    return list(dict.fromkeys([*config_passthrough_env, *sidecar_passthrough_env]))


def grants_for_scope(grants: list[EnvGrant], scope: Scope) -> list[EnvGrant]:
    """Filter grants down to those authorized for ``scope``."""
    return [g for g in grants if g.authorizes(scope)]


def env_dict(grants: list[EnvGrant], scope: Scope) -> dict[str, str]:
    """Materialize the environment dict for ``scope`` from its grants.

    Later grants win on a name collision, matching the previous construction
    order (``fixed_env`` overrode ``passthrough_env``, and so on).
    """
    return {g.name: g.value for g in grants_for_scope(grants, scope)}


def resolve_env(
    *,
    scope: Scope,
    backend: str,
    sandbox_enabled: bool,
    auth_mode: str | None,
    **kwargs: object,
) -> dict[str, str]:
    """Convenience wrapper returning the materialized env dict for ``scope``."""
    grants = resolve_env_grants(
        scope=scope,
        backend=backend,
        sandbox_enabled=sandbox_enabled,
        auth_mode=auth_mode,
        **kwargs,  # type: ignore[arg-type]
    )
    return env_dict(grants, scope)
