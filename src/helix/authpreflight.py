"""Once-per-run backend authentication preflight.

Runs BEFORE the first mutation dispatch, with zero proposal, budget, ledger or
population side effects.  A clean pre-dispatch refusal is enormously better
than a mid-run 401 that has already spent budget and written state — most
acutely for a lane with a single timed window.

Why a real authenticated request is the only sound signal
---------------------------------------------------------
Each cheaper candidate was ruled out by measurement, not by taste:

* **file presence** (``test -s .credentials.json``) passed against a token
  that had expired 23.5 hours earlier, and passes against a file containing
  no real token material at all;
* **the backend's own status text** returns 0 with no credentials, and the
  shell probe discarded its exit code with ``|| true`` anyway;
* **``refreshTokenExpiresAt``** has zero occurrences in the pinned runner's
  CLI — reasoning about it describes a CLI that is not running;
* **``docker volume inspect`` succeeding** establishes existence only, which
  is necessary and nowhere near sufficient.

Two rules here are counterintuitive, and violating either produces a
FALSE GREEN — a probe that passes while measuring nothing:

1. **The probe uses the REAL volume at ``:rw``, never a copy.**  A successful
   refresh *rotates* the server-side refresh token.  Probing a copy would
   persist the rotation to the copy while the real volume kept the now-dead
   token — HELIX would inflict on itself precisely the bug it is fixing,
   while reporting success.  ``:ro`` is equally wrong: refresh needs an
   exclusive lockfile in the credential directory, and on a read-only mount
   that fails *silently*, so a ``:ro`` probe under-reports validity and
   discards the repair it just performed.
2. **The probe environment carries no credentials, across all three origins.**
   If the probe inherited ``ANTHROPIC_API_KEY`` the way real runs used to,
   OAuth mode goes off, the volume's token is never consulted, and the probe
   PASSES ON THE ENV KEY while reporting the volume valid.

Rule 1 is enforced structurally rather than by comment-and-test: the probe's
docker argv is built by the SAME ``_docker_args(scope="agent")`` call the real
mutation uses, so there is no separate probe mount string to "improve".  Any
edit that isolates the probe necessarily isolates the production agent too,
which breaks the run loudly instead of greening the probe silently.  Rule 2 is
enforced by constructing the probe environment through the same
``resolve_env_grants`` the production path uses, under ``auth="volume"`` —
credential-free by construction rather than by remembering.
"""

from __future__ import annotations

import logging
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from helix.config import HelixConfig
from helix.envpolicy import env_dict, resolve_env_grants
from helix.backend_layout import layout_for
from helix.subpath_bootstrap import (
    assert_volume_subpath_supported,
    missing_subpath_error,
)
from helix.exceptions import SandboxAuthConcurrencyError, SandboxAuthPreflightError
from helix.sandbox import (
    AuthVolumeManifest,
    DockerRunner,
    _docker_args,
    docker_volume_exists,
    read_auth_manifest,
    resolve_sandbox_image,
    sandbox_auth_volume_name,
)


logger = logging.getLogger(__name__)


PreflightOutcome = Literal["ok", "skipped_env_mode", "skipped_no_sandbox"]


@dataclass(frozen=True)
class PreflightResult:
    outcome: PreflightOutcome
    volume: str | None = None
    image: str | None = None
    manifest: AuthVolumeManifest | None = None
    provenance: Literal["unknown", "matched", "skew", "wrong_backend"] = "unknown"
    billable_calls: int = 0


# In-process singleflight.  One verdict per HELIX run, keyed on volume name.
#
# Caching is IN-PROCESS ONLY and nothing is persisted.  A persisted verdict
# would be a new file-presence-style sufficiency signal with a timestamp on it
# — the exact class this release removes — and correct invalidation would have
# to cover volume identity, runtime image digest, backend, credential mutation
# by any other process, expiry and revocation, and cross-process
# synchronisation.  A design addressing four of those six is worse than none.
_verdicts: dict[str, PreflightResult] = {}
_verdict_lock = threading.Lock()


def reset_preflight_cache() -> None:
    """Clear the in-process verdict cache (tests, and repeated runs in-process)."""
    with _verdict_lock:
        _verdicts.clear()


# The probe command per backend: the cheapest possible real authenticated
# operation.  Refresh is proactive-on-expiry, so the probe does NOT need to
# provoke a 401 — any real authenticated call exercises the refresh path.
BACKEND_PROBE_COMMANDS: dict[str, list[str]] = {
    "claude": ["claude", "--print", "--output-format", "json", "say ok"],
    "codex": ["codex", "exec", "--json", "say ok"],
    "cursor": ["cursor-agent", "-p", "say ok"],
    "gemini": ["gemini", "-p", "say ok"],
    "opencode": ["opencode", "run", "say ok"],
}


class _VolumeLock:
    """Cross-process advisory lock keyed on the auth volume name.

    The backend CLI serialises its own refresh with a lockfile that goes stale
    after ~10s, retries 5 times, and then gives up SILENTLY.  With several
    containers starting at once — which parallel proposals across lanes make
    routine — that silent give-up is reachable, and silence is the pathology
    being removed.  HELIX therefore adds its own coordination and FAILS LOUD
    rather than proceeding unverified.

    Residual, and not closeable from inside HELIX: this lock serialises HELIX
    runs against each other.  It cannot coordinate with a host ``claude``
    process, an editor integration, or any other CLI touching the same account
    or volume — and there is direct evidence a host CLI has written to this
    credential record.  The preflight's verdict is therefore scoped to "valid
    at the moment we checked"; the runtime handler is the backstop.
    """

    def __init__(self, volume: str, *, timeout: float = 120.0) -> None:
        self.path = Path(tempfile.gettempdir()) / f"helix-auth-{volume}.lock"
        self.volume = volume
        self.timeout = timeout
        self._fh: object | None = None

    def __enter__(self) -> _VolumeLock:
        import fcntl
        import time

        fh = open(self.path, "w")
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._fh = fh
                return self
            except OSError:
                if time.monotonic() >= deadline:
                    fh.close()
                    raise SandboxAuthConcurrencyError(
                        f"another HELIX run is verifying auth volume "
                        f"{self.volume!r}; retry when it completes.",
                        operation="auth preflight lock",
                        suggestion=(
                            "Wait for the other run to finish its preflight, or "
                            "stagger run start times."
                        ),
                    ) from None
                time.sleep(0.25)

    def __exit__(self, *exc: object) -> None:
        import fcntl

        if self._fh is not None:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)  # type: ignore[attr-defined]
            self._fh.close()  # type: ignore[attr-defined]
            self._fh = None


def _classify_probe_failure(stdout: str, stderr: str) -> tuple[str, str]:
    """Distinguish a failed token refresh from a failed inference.

    These have DIFFERENT remedies and collapsing them sends users to a remedy
    that does not help: a failed token POST means `helix sandbox login`, while
    a failed inference *after a successful refresh* means quota, model access
    or network.  Where the signal is ambiguous this says so explicitly rather
    than guessing — naming which two states could not be distinguished is more
    useful than a confident wrong answer.

    Returns ``(kind, remedy)`` where kind is one of ``refresh``, ``inference``,
    ``transport``, ``ambiguous``.
    """
    blob = f"{stdout}\n{stderr}".lower()

    transport_markers = (
        "could not resolve host",
        "connection refused",
        "network is unreachable",
        "temporary failure in name resolution",
        "timed out",
        "econnreset",
    )
    if any(m in blob for m in transport_markers):
        return (
            "transport",
            "This looks like a network/transport failure rather than an auth "
            "failure. Check connectivity and retry; credentials may be fine.",
        )

    refresh_markers = (
        "oauth/token",
        "invalid_grant",
        "refresh",
        "invalid bearer token",
        "authentication_error",
        "401",
    )
    inference_markers = (
        "rate limit",
        "quota",
        "overloaded",
        "429",
        "model not found",
        "permission",
    )
    saw_refresh = any(m in blob for m in refresh_markers)
    saw_inference = any(m in blob for m in inference_markers)

    if saw_refresh and not saw_inference:
        return ("refresh", "helix sandbox login {backend}")
    if saw_inference and not saw_refresh:
        return (
            "inference",
            "The credential refreshed successfully but the request itself "
            "failed (quota, model access, or rate limit). Re-authenticating "
            "will NOT help.",
        )
    return (
        "ambiguous",
        "HELIX could not distinguish a failed token refresh (remedy: "
        "`helix sandbox login`) from a failed request after a successful "
        "refresh (remedy: quota / model access / network). Both are "
        "consistent with the output.",
    )


def preflight_auth(
    config: HelixConfig,
    *,
    runner: DockerRunner,
    probe: bool = True,
) -> PreflightResult:
    """Verify the backend auth volume once, before any mutation is dispatched.

    ``runner`` is a REQUIRED dependency with no default.  That is deliberate
    and is a safety control, not an ergonomic choice: this function performs a
    real authenticated operation against the real shared auth volume at
    ``:rw``, and a successful refresh ROTATES the stored token.  With a
    defaulted runner, any caller that merely constructs a config — including a
    unit test — silently acquires the ability to do that.  Requiring the
    dependency means a non-production caller cannot reach Docker by omission;
    it has to pass something, and what it passes is visible at the call site.

    Production callers use :func:`helix.sandbox.production_docker_runner`.

    Raises :class:`SandboxAuthPreflightError` on missing / empty /
    wrong-backend / expired / real-auth-failing volumes, with a redacted,
    actionable message and zero side effects.
    """
    sandbox = config.sandbox
    backend = config.agent.backend

    if not sandbox.enabled:
        return PreflightResult(outcome="skipped_no_sandbox")

    mode = sandbox.resolved_auth()
    if mode == "env":
        # Env mode never consults the volume, so there is nothing to verify.
        # It is also not a fallback: it was chosen explicitly in a config file.
        return PreflightResult(outcome="skipped_env_mode")

    volume = sandbox_auth_volume_name(backend)

    with _verdict_lock:
        cached = _verdicts.get(volume)
    if cached is not None:
        return cached

    run = runner
    image = resolve_sandbox_image(sandbox, backend)

    # --- Stage -1: daemon capability, BEFORE touching the volume ------
    #
    # Volume mode mounts the store with ``volume-subpath``. On a daemon that
    # cannot do that, the run dies inside Docker with an opaque error and
    # there is no safe fallback -- the only other way to mount the store is
    # over the whole container HOME, which is the defect this release removes.
    #
    # Checked here rather than at container start so the operator learns before
    # a proposal is created or budget charged.
    version = run(
        [
            "docker",
            "version",
            "--format",
            "{{.Server.Version}}|{{.Server.APIVersion}}",
        ],
        check=False,
    )
    if version.returncode == 0:
        server, _, api = (version.stdout or "").strip().partition("|")
        assert_volume_subpath_supported(server_version=server, api_version=api)

    with _VolumeLock(volume):
        with _verdict_lock:
            cached = _verdicts.get(volume)
        if cached is not None:
            return cached

        # --- Stage 0: existence, side-effect free -----------------------
        # `docker volume inspect`, never `docker run -v` — the latter CREATES.
        if not docker_volume_exists(volume, runner=run):
            raise SandboxAuthPreflightError(
                f"sandbox auth volume {volume!r} does not exist.\n"
                f'  sandbox.enabled = true and sandbox.auth = "volume", so the '
                f"mutation agent has no credential path. HELIX will not fall "
                f"back to environment variables.\n"
                f"  No proposal was created and no budget was charged.",
                remedy=f"helix sandbox login {backend}",
                operation="auth preflight (stage 0: existence)",
                suggestion=f"helix sandbox login {backend}",
            )

        # --- Stage 0b: the auth SUBPATH must exist ---------------------
        #
        # ``volume-subpath`` requires the directory to exist before the
        # container starts. On a volume that exists but was never authenticated
        # for this backend, the agent run otherwise dies with
        # ``cannot access path ...: no such file or directory`` -- an auth
        # problem wearing the costume of an internal Docker error.
        #
        # Read-only probe: it inspects, and never creates. Creating the subpath
        # is ``login``'s job.
        layout = layout_for(backend)
        subpath_probe = run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--user",
                "node",
                "-v",
                f"{volume}:/helix-auth-root:ro",
                image,
                "test",
                "-d",
                f"/helix-auth-root/{layout.volume_subpath}",
            ],
            check=False,
        )
        if subpath_probe.returncode != 0:
            raise SandboxAuthPreflightError(
                missing_subpath_error(
                    backend=backend,
                    volume=volume,
                    subpath=layout.volume_subpath,
                ),
                remedy=f"helix sandbox login {backend}",
                operation="auth preflight (stage 0b: auth subpath)",
                suggestion=f"helix sandbox login {backend}",
            )

        # --- Stage 1: provenance stamp ---------------------------------
        manifest = read_auth_manifest(backend, image=image, runner=run)
        provenance: Literal["unknown", "matched", "skew", "wrong_backend"]
        if manifest is None:
            # Absent on every volume provisioned before stamps existed.
            # Unknown is never valid — but it must not hard-fail, or this
            # change bricks every existing volume on first upgrade.
            provenance = "unknown"
            logger.warning(
                "auth volume %s has no HELIX provenance stamp; provenance is "
                "unknown and skew cannot be checked. Re-run `helix sandbox "
                "login %s` to record it.",
                volume,
                backend,
            )
        elif manifest.backend and manifest.backend != backend:
            raise SandboxAuthPreflightError(
                f"auth volume {volume!r} was provisioned for backend "
                f"{manifest.backend!r} (per its HELIX auth manifest) but this "
                f"run's agent.backend is {backend!r}.",
                remedy=f"helix sandbox login {backend}",
                operation="auth preflight (stage 1: provenance)",
                suggestion=f"helix sandbox login {backend}",
            )
        elif manifest.image and manifest.image != image:
            provenance = "skew"
            message = (
                f"auth volume {volume!r} was written by {manifest.image} "
                f"(CLI {manifest.cli_version or 'unknown'}), but this run's "
                f"runner image is {image}. Credential-file fields written by a "
                f"different CLI build may be ignored by this one."
            )
            if sandbox.require_cli_match:
                raise SandboxAuthPreflightError(
                    message,
                    remedy=(
                        f"re-run `helix sandbox login {backend}` with the "
                        f"configured runner image, or set "
                        f"sandbox.require_cli_match = false to accept this skew"
                    ),
                    operation="auth preflight (stage 1: provenance)",
                )
            logger.warning("%s", message)
        else:
            provenance = "matched"

        if not probe:
            result = PreflightResult(
                outcome="ok",
                volume=volume,
                image=image,
                manifest=manifest,
                provenance=provenance,
                billable_calls=0,
            )
            with _verdict_lock:
                _verdicts[volume] = result
            return result

        # --- Stage 2: a real authenticated operation --------------------
        command = BACKEND_PROBE_COMMANDS.get(backend)
        if command is None:
            raise SandboxAuthPreflightError(
                f"no authenticated probe is defined for backend {backend!r}, "
                "so HELIX cannot verify its credentials before dispatch.",
                remedy=f"set sandbox.auth explicitly for backend {backend!r}",
                operation="auth preflight (stage 2: sufficiency)",
            )

        # Credential-free by CONSTRUCTION: the same resolver the production
        # path uses, under auth="volume", which grants no credential to agent
        # scope from any of the three origins.  If this ever regressed, the
        # probe would authenticate on the env key and report the volume valid
        # while measuring nothing — a false green worse than no preflight.
        grants = resolve_env_grants(
            scope="agent",
            backend=backend,
            sandbox_enabled=True,
            auth_mode="volume",
            agent_passthrough_env=sandbox.agent_passthrough_env,
            config_passthrough_env=config.passthrough_env,
            config_env=config.env,
        )
        env = env_dict(grants, "agent")

        with tempfile.TemporaryDirectory(prefix="helix-authprobe-") as tmp:
            # The probe argv is built by the SAME function, with the SAME
            # scope, as the real mutation container.  That is what makes the
            # no-copy / :rw rule structural: there is no probe-specific mount
            # to isolate.  Anyone who "improves" this by pointing it at a copy
            # must change the production agent mount at the same time, which
            # fails the run loudly instead of greening the probe silently.
            args = _docker_args(
                list(command),
                env,
                Path(tmp),
                sandbox,
                "agent",
                image,
                backend,
                grants=grants,
            )
            result_proc = run(args, check=False)

        if result_proc.returncode != 0:
            kind, remedy = _classify_probe_failure(
                result_proc.stdout or "", result_proc.stderr or ""
            )
            remedy = remedy.replace("{backend}", backend)
            raise SandboxAuthPreflightError(
                f"backend authentication failed for {backend!r} using auth "
                f"volume {volume!r}.\n"
                f"  A non-empty credentials file is NOT evidence of valid "
                f"credentials; this check performed a real authenticated "
                f"request.\n"
                f"  Diagnosis: {kind}\n"
                f"  HELIX will not fall back to environment-variable "
                f"authentication. To use environment credentials deliberately, "
                f'set sandbox.auth = "env" and list the variables in '
                f"sandbox.auth_env_allow — see the exposure disclosure in the "
                f"docs before doing so.\n"
                f"  No proposal was created and no budget was charged.",
                remedy=remedy,
                operation="auth preflight (stage 2: sufficiency)",
                suggestion=remedy,
            )

        result = PreflightResult(
            outcome="ok",
            volume=volume,
            image=image,
            manifest=manifest,
            provenance=provenance,
            billable_calls=1,
        )
        with _verdict_lock:
            _verdicts[volume] = result
        return result


def env_mode_disclosure(config: HelixConfig) -> str:
    """Non-suppressible startup disclosure for ``sandbox.auth = "env"``.

    Deliberately not reassuring.  Env mode is a tradeoff, not an equivalent
    alternative: the variables it injects turn OAuth mode off in the backend
    CLI, so the run performs no container-side refresh at all and WILL let a
    mounted auth volume's token go stale.  Names only, never values.
    """
    names = "\n".join(f"    {n}" for n in config.sandbox.auth_env_allow)
    return (
        'HELIX: sandbox.auth = "env" — EXPLICIT ENVIRONMENT CREDENTIAL MODE.\n'
        "  The following variable NAMES will be placed in the mutation "
        "agent's container:\n"
        f"{names}\n"
        "  The mutation agent runs with --dangerously-skip-permissions on a "
        f"'{config.sandbox.network}' network, so code it executes can read "
        "these values and can reach the network.\n"
        "\n"
        "  This mode DISABLES OAuth token refresh inside the container. "
        "Setting these variables turns OAuth mode off in the backend CLI, so "
        "no proactive or 401-triggered refresh is attempted. If an auth volume "
        "exists for this backend, THIS RUN WILL NOT REFRESH IT and its stored "
        "token will go stale. This is a tradeoff, not an equivalent "
        'alternative to sandbox.auth = "volume".'
    )


__all__ = [
    "PreflightResult",
    "preflight_auth",
    "reset_preflight_cache",
    "env_mode_disclosure",
]
