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
            # 23.5 hours in the past, in epoch MILLISECONDS, so the CLI's
            # proactive-on-expiry gate (5-minute margin) is tripped.
            "expiresAt": 1,
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": "max",
        }
    }
    payload = json.dumps(record)

    _docker(
        [
            "docker", "run", "--rm", "--user", "root",
            "-v", f"{volume}:/home/node",
            RUNNER_IMAGE, "sh", "-c",
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
        "docker", "run", "--rm", "--user", "node",
        "-e", "HOME=/home/node",
        # The CLI writes its own debug log; ask for it so the oracle has input.
        "-e", "ANTHROPIC_LOG=debug",
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
            "docker", "run", "--rm", "--user", "node",
            "-v", f"{volume}:/home/node:ro", RUNNER_IMAGE,
            "sh", "-c",
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
    control = _run_probe(synthetic_volume, {})
    assert _refresh_attempted(control), "control must attempt refresh first"

    output = _run_probe(
        synthetic_volume, {variable: "SYNTHETIC-DO-NOT-USE-" + ("D" * 40)}
    )
    assert not _refresh_attempted(output), (
        f"{variable} must suppress the container-side refresh attempt"
    )
