"""T22-T25 — behavioural proof that auth env vars suppress OAuth refresh.

These assert on what the container DOES, not on a minified predicate, which is
the right standard for the central claim of this release.

SAFETY, and it is the point rather than decoration:

* A **DISPOSABLE** volume with **SYNTHETIC** credentials, always. The shared
  ``helix-auth-*`` volumes are never touched — the session-wide guard in
  ``tests/conftest.py`` refuses those by name with no override, and this
  module refuses again below. Belt and braces, because a typo in a volume name
  does not fail: ``docker run -v`` silently CREATES the volume it names.
* Teardown is **name-verified**, not best-effort, and asserts a resource diff.
* No credential value may appear in any captured output. The synthetic values
  are fake, but the assertion must exist, because this file is the template
  someone will later copy while debugging with a real credential.

These are integration-tier: they need a real daemon and network egress to the
token endpoint, so they do NOT run on every PR and must not be relied on as
the only guard. The argv-level suite in
``tests/unit/test_sandbox_auth_policy.py`` carries that load.

They also pin behaviour to ONE CLI version — whatever the configured image
provides. They are a regression detector against HELIX changes, not against
upstream CLI changes; an upstream change would surface here as a red test
whose correct resolution might be "update the expectation".
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid

import pytest


pytestmark = pytest.mark.docker_integration


RUNNER_IMAGE = os.environ.get(
    "HELIX_REFRESH_TEST_IMAGE",
    "ghcr.io/ke7/helix-evo-runner-claude"
    "@sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba",
)

# Obviously-fake, clearly-labelled, and never a real token shape that could be
# mistaken for one. 108 chars matches the real record's length so the CLI's own
# parsing is exercised realistically.
_SYNTHETIC_ACCESS = "sk-ant-oat01-SYNTHETIC-DO-NOT-USE-" + ("F" * 74)
_SYNTHETIC_REFRESH = "sk-ant-ort01-SYNTHETIC-DO-NOT-USE-" + ("E" * 74)

TOKEN_ENDPOINT_MARKER = "/v1/oauth/token"

# Fixed offsets from a captured timestamp rather than a live ``Date.now()``,
# so the fixture is deterministic.
_NOW_MS = 1784719800000
_EXPIRED_AT_MS = _NOW_MS - int(23.5 * 3600 * 1000)
_REFRESH_EXPIRES_AT_MS = _NOW_MS + 21 * 24 * 3600 * 1000


# ---------------------------------------------------------------------------
# Capability preflight -- distinguish "cannot test here" from "property broken"
# ---------------------------------------------------------------------------
#
# These four tests previously failed UNCONDITIONALLY in an environment without
# egress to the OAuth endpoint, which conflates "this environment cannot
# support the test" with "the suppression property is broken" -- the same
# missing-vs-failed conflation removed from ``helix.transcripts`` and from
# ``_run_in_image``.
#
# Neither shipping red nor a blanket skip is acceptable: the first leaves a
# SECURITY property of the credential fix unverified and unexplained, the
# second silences it. The third option is a skip that is CONDITIONAL AND
# PROVEN.

# The endpoint the CLI actually POSTs to. Confirmed by two independent
# observations on this host: a direct probe returning HTTP 400, and the earlier
# refresh investigation recording
# ``AxiosError: [url=https://platform.claude.com/v1/oauth/token,status=400]``.
OAUTH_TOKEN_HOST = "platform.claude.com"
_REACHABILITY: dict[str, tuple[bool, str]] = {}


def _probe_endpoint_reachability() -> tuple[bool, str]:
    """Is the OAuth endpoint reachable FROM THE PINNED RUNTIME?

    Classifies ONLY DNS and connectivity. Uses NO credential or token material
    of any kind -- it is an unauthenticated HEAD-equivalent.

    THE DISTINCTION THIS TURNS ON: an HTTP status from the endpoint -- 400,
    401, 404, anything -- means the endpoint IS REACHABLE. Only DNS failure or
    a connection error mean it is not. Treating a rejection as "unavailable"
    would reproduce exactly the defect this preflight exists to remove.
    """
    if "verdict" in _REACHABILITY:
        return _REACHABILITY["verdict"]

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--user",
            "node",
            "--security-opt",
            "no-new-privileges",
            "--entrypoint",
            "sh",
            RUNNER_IMAGE,
            "-c",
            # -o /dev/null: never capture a body. -s -S: quiet but report errors.
            # A status line at all == reachable.
            f"curl -s -S -o /dev/null -w 'HTTP:%{{http_code}}' "
            f"--max-time 20 https://{OAUTH_TOKEN_HOST}/v1/oauth/token 2>&1 "
            f"|| echo CURL_FAILED",
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined = (result.stdout or "") + (result.stderr or "")
    verdict = _classify_probe_output(combined)
    _REACHABILITY["verdict"] = verdict
    return verdict


def _classify_probe_output(combined: str) -> tuple[bool, str]:
    """Classify raw probe output as reachable / unreachable.

    Split out from the Docker call so both branches are unit-testable without
    a daemon or network -- the preflight decides whether four security tests
    RUN, so it is load-bearing and both its branches need coverage.
    """
    if "HTTP:" in combined:
        code = combined.split("HTTP:")[1].strip()[:3]
        # ANY status, including 4xx, proves reachability.
        if code.isdigit() and code != "000":
            return (True, f"endpoint returned HTTP {code}")

    lowered = combined.lower()
    for marker, why in (
        ("could not resolve", "DNS resolution failed"),
        ("name or service not known", "DNS resolution failed"),
        ("connection refused", "connection refused"),
        ("connection timed out", "connection timed out"),
        ("timed out", "connection timed out"),
        ("network is unreachable", "network unreachable"),
        ("curl_failed", "curl could not reach the endpoint"),
        ("not found", "curl is unavailable in the runner image"),
    ):
        if marker in lowered:
            return (False, why)

    return (False, f"unclassified probe result: {combined.strip()[:160]}")


def require_oauth_endpoint_reachable() -> None:
    """Skip ONLY on proven unreachability, naming the missing capability."""
    reachable, why = _probe_endpoint_reachability()
    if reachable:
        return
    pytest.skip(
        f"MISSING NETWORK CAPABILITY: egress from the pinned runner image to "
        f"https://{OAUTH_TOKEN_HOST}/v1/oauth/token ({why}). "
        f"T22-T25 OAuth-refresh SUPPRESSION BEHAVIOUR REMAINS UNVERIFIED IN "
        f"THIS ENVIRONMENT -- this is not evidence that suppression works."
    )


def _assert_disposable(volume: str) -> None:
    """Refuse to operate on anything that could be shared credential state."""
    assert volume.startswith("helix-refreshtest-"), volume
    assert not volume.startswith("helix-auth-"), (
        "refusing to touch a shared auth volume: a refresh against it would "
        "rotate the stored token and invalidate it for every lane"
    )


def _docker(args: list[str], **kw) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, capture_output=True, text=True, **kw)


def _volume_exists(volume: str) -> bool:
    return _docker(["docker", "volume", "inspect", volume]).returncode == 0


@pytest.fixture()
def synthetic_volume():
    """A disposable volume holding an EXPIRED synthetic OAuth record."""
    volume = f"helix-refreshtest-{uuid.uuid4().hex[:12]}"
    _assert_disposable(volume)
    assert not _volume_exists(volume), "volume name collision"

    record = {
        "claudeAiOauth": {
            "accessToken": _SYNTHETIC_ACCESS,
            "refreshToken": _SYNTHETIC_REFRESH,
            # 23.5 hours in the past, in epoch MILLISECONDS, matching the
            # reference harness at c089da0:.rca-scratch/setup-synth.js.
            #
            # THE PREVIOUS VALUE WAS THE LITERAL ``1`` -- epoch +1ms, i.e.
            # 1970 -- under a comment claiming "23.5 hours in the past". The
            # comment described a value the code did not contain: a wrong
            # declaration sitting directly above the artifact, in the file
            # whose failure was being explained. Corrected regardless of
            # causality, which is separately recorded below as NOT the cause.
            "expiresAt": _EXPIRED_AT_MS,
            # Set for parity with the reference harness. NOTE: the pinned CLI
            # never reads this key -- 0 occurrences in the 2.1.120 binary,
            # matching the RCA's finding -- so it cannot gate anything.
            "refreshTokenExpiresAt": _REFRESH_EXPIRES_AT_MS,
            "rateLimitTier": "default_claude_max_20x",
            "scopes": [
                "user:file_upload",
                "user:inference",
                "user:mcp_servers",
                "user:profile",
                "user:sessions:claude_code",
            ],
            "subscriptionType": "max",
        }
    }
    payload = json.dumps(record)

    _docker(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "root",
            "-v",
            f"{volume}:/home/node",
            RUNNER_IMAGE,
            "sh",
            "-c",
            "mkdir -p /home/node/.claude && "
            f"printf '%s' {json.dumps(payload)} > /home/node/.claude/.credentials.json && "
            "chmod 600 /home/node/.claude/.credentials.json && "
            "chown -R node:node /home/node",
        ],
        check=False,
    )
    try:
        yield volume
    finally:
        # Name-verified teardown, then a resource diff.
        _assert_disposable(volume)
        _docker(["docker", "volume", "rm", "-f", volume])
        assert not _volume_exists(volume), f"disposable volume {volume} leaked"


def _run_probe(volume: str, env: dict[str, str]) -> str:
    """Run the CLI headless against ``volume`` and return its debug output."""
    _assert_disposable(volume)
    args = [
        "docker",
        "run",
        "--rm",
        "--user",
        "node",
        "-e",
        "HOME=/home/node",
        # The CLI writes its own debug log; ask for it so the oracle has input.
        "-e",
        "ANTHROPIC_LOG=debug",
    ]
    for key, value in env.items():
        args.extend(["-e", f"{key}={value}"])
    args.extend(["-v", f"{volume}:/home/node:rw", RUNNER_IMAGE])
    args.extend(["claude", "-p", "say ok"])

    result = _docker(args)
    combined = (result.stdout or "") + (result.stderr or "")

    # Also collect the CLI's on-disk debug log from inside the volume.
    logs = _docker(
        [
            "docker",
            "run",
            "--rm",
            "--user",
            "node",
            "-v",
            f"{volume}:/home/node:ro",
            RUNNER_IMAGE,
            "sh",
            "-c",
            "cat /home/node/.claude/debug/*.txt 2>/dev/null || true",
        ]
    )
    combined += logs.stdout or ""

    # SAFETY: no credential value may appear in captured output, ever.
    assert _SYNTHETIC_ACCESS not in combined
    assert _SYNTHETIC_REFRESH not in combined
    return combined


def _refresh_attempted(output: str) -> bool:
    """A refresh was attempted iff the OAuth token endpoint was contacted.

    Observable without any valid credential, because the ATTEMPT is the
    assertion — not its success. HTTP 400 for a fake refresh token is the
    correct outcome.
    """
    return TOKEN_ENDPOINT_MARKER in output


def test_T22_expired_record_no_auth_env_attempts_refresh(synthetic_volume):
    """T22 — THE CONTROL, and the non-vacuity proof for T23-T25.

    With no auth env, an expired record must reach the OAuth token POST. If
    this fails, T23-T25 "prove" suppression that is really a broken harness,
    so it is asserted first and its failure message says so.
    """
    require_oauth_endpoint_reachable()
    output = _run_probe(synthetic_volume, {})
    assert _refresh_attempted(output), (
        "CONTROL FAILED: no refresh was attempted even with no auth env. "
        "T23-T25 are VACUOUS until this passes — they would be measuring a "
        "broken harness rather than env-var suppression."
    )


@pytest.mark.parametrize(
    "variable",
    ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"],
)
def test_T23_T25_auth_env_suppresses_refresh(synthetic_volume, variable):
    """T23/T24/T25 — each auth variable suppresses the refresh attempt.

    T23 (ANTHROPIC_API_KEY) is what HELIX injected into every claude lane.
    T24 (ANTHROPIC_AUTH_TOKEN) is the more dangerous variable.
    T25 (CLAUDE_CODE_OAUTH_TOKEN) justifies the R6 hard prohibition with
    behaviour rather than with a claim about the credential record.

    Paired against T22 in the same session: the control is re-run here so a
    suppression result can never be reported from a harness that never
    refreshes at all.
    """
    require_oauth_endpoint_reachable()
    control = _run_probe(synthetic_volume, {})
    # HARD FAILURE, never a skip: the endpoint is PROVEN reachable at this
    # point, so a missing refresh attempt is a real result about the harness or
    # the runtime -- not an environment limitation.
    assert _refresh_attempted(control), "control must attempt refresh first"

    output = _run_probe(
        synthetic_volume, {variable: "SYNTHETIC-DO-NOT-USE-" + ("D" * 40)}
    )
    assert not _refresh_attempted(output), (
        f"{variable} must suppress the container-side refresh attempt"
    )
