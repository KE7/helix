"""Which CREDENTIAL SOURCE does the CLI select? (mode-selection oracle)

WHAT THIS PROVES, AND WHAT IT DOES NOT.

PROVEN: mode selection and credential-source precedence -- that an injected auth
variable causes the CLI to send THAT VARIABLE'S credential and NOT the OAuth
credential stored in the volume.

NOT PROVEN: that a token *refresh* is suppressed. Refresh execution has never
been observed on this host, in any arm, on either CLI version, with either
fixture (see the design doc's §8e falsification table). Mode selection and
refresh execution are different properties and only the first is demonstrated
here. The test names say so deliberately, because names are what a future
reader trusts.

WHY THIS IS STRONGER THAN WATCHING THE NETWORK: the property we care about is
governed by the CLI's mode determination. Observing an outbound refresh tests a
downstream CONSEQUENCE that depends on egress, server behaviour and emulation --
all of which have already produced misleading results here. Observing WHICH
CREDENTIAL IS SENT tests the mechanism, offline and deterministically.

THE DESIGN-CRITICAL DETAIL: ``ANTHROPIC_AUTH_TOKEN`` and
``CLAUDE_CODE_OAUTH_TOKEN`` REPLACE THE BEARER VALUE rather than switching
header type, so they present exactly like the OAuth path. An oracle asserting
"no bearer header" would pass for ``ANTHROPIC_API_KEY`` and FAIL for the other
two -- measuring the HEADER TYPE (adjacent) instead of the CREDENTIAL SOURCE
(the property). Every source therefore gets a DISTINCT synthetic canary, and
the assertion is which canary ARRIVED.

No real credential material anywhere. Canaries are compared internally and
NEVER emitted in failure output.
"""

from __future__ import annotations

import http.server
import json
import subprocess
import threading
import uuid
from pathlib import Path

import pytest


pytestmark = pytest.mark.docker_integration

RUNNER_IMAGE = (
    "ghcr.io/ke7/helix-evo-runner-claude"
    "@sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba"
)

# One DISTINCT canary per credential source. Never printed.
_CANARY = {
    "volume": "sk-ant-oat01-SYNTHETIC-VOLUMESRC-" + ("V" * 74),
    "ANTHROPIC_API_KEY": "SYNTHETIC-APIKEYSRC-" + ("K" * 40),
    "ANTHROPIC_AUTH_TOKEN": "SYNTHETIC-AUTHTOKENSRC-" + ("T" * 40),
    "CLAUDE_CODE_OAUTH_TOKEN": "SYNTHETIC-OAUTHTOKENSRC-" + ("O" * 40),
}
_NOW_MS = 1784719800000


class _Capture(http.server.BaseHTTPRequestHandler):
    seen: list[dict[str, str]] = []

    def do_POST(self) -> None:  # noqa: N802
        type(self).seen.append({k.lower(): v for k, v in self.headers.items()})
        body = json.dumps({"type": "error", "error": {"type": "invalid_request"}})
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body.encode())

    do_GET = do_POST  # noqa: N815

    def log_message(self, *args: object) -> None:
        return  # keep test output clean


def _which_source(headers: list[dict[str, str]]) -> set[str]:
    """Which canaries arrived. Returns SOURCE NAMES only -- never values."""
    blob = " ".join(v for h in headers for v in h.values())
    return {name for name, canary in _CANARY.items() if canary in blob}


@pytest.fixture
def capture_server():
    _Capture.seen = []
    server = http.server.HTTPServer(("0.0.0.0", 0), _Capture)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def oauth_volume():
    volume = f"helix-oracle-{uuid.uuid4().hex[:10]}"
    assert volume.startswith("helix-oracle-")
    assert not volume.startswith("helix-auth-")
    subprocess.run(["docker", "volume", "create", volume], capture_output=True)
    record = {
        "claudeAiOauth": {
            "accessToken": _CANARY["volume"],
            "refreshToken": "sk-ant-ort01-SYNTHETIC-" + ("R" * 74),
            "expiresAt": _NOW_MS + 8 * 3600 * 1000,  # VALID, so OAuth is used
            "refreshTokenExpiresAt": _NOW_MS + 21 * 24 * 3600 * 1000,
            "scopes": ["user:inference", "user:profile"],
            "subscriptionType": "max",
        }
    }
    subprocess.run(
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
            f"printf '%s' {json.dumps(json.dumps(record))} "
            "> /home/node/.claude/.credentials.json && "
            "chmod 600 /home/node/.claude/.credentials.json && "
            "chown -R node:node /home/node",
        ],
        capture_output=True,
        timeout=180,
    )
    try:
        yield volume
    finally:
        subprocess.run(["docker", "volume", "rm", "-f", volume], capture_output=True)


def _run_cli(volume: str, port: int, env: dict[str, str]) -> None:
    args = [
        "docker",
        "run",
        "--rm",
        "--pull=never",
        "--user",
        "node",
        "--add-host",
        "host.docker.internal:host-gateway",
        "-e",
        "HOME=/home/node",
        "-e",
        f"ANTHROPIC_BASE_URL=http://host.docker.internal:{port}",
    ]
    for key, value in env.items():
        args.extend(["-e", f"{key}={value}"])
    args.extend(["-v", f"{volume}:/home/node:rw", RUNNER_IMAGE])
    args.extend(["claude", "-p", "say ok"])
    subprocess.run(args, capture_output=True, text=True, timeout=180)


def test_oracle_captures_requests_in_the_clean_arm(capture_server, oauth_volume):
    """NON-VACUITY, and it must come first.

    An oracle that captures nothing in either arm proves nothing -- the exact
    shape of the probes discarded earlier in this investigation. Every
    assertion below is worthless unless this passes.
    """
    _run_cli(oauth_volume, capture_server, {})
    assert _Capture.seen, (
        "the loopback capture server observed NO requests; every "
        "credential-source assertion below would be vacuous"
    )


def test_no_env_selects_the_VOLUME_oauth_credential(capture_server, oauth_volume):
    """Baseline: with no auth env, the CLI sends the VOLUME's credential.

    Catches: an oracle that cannot see the OAuth path at all, which would make
    every suppression assertion below unfalsifiable.
    """
    _run_cli(oauth_volume, capture_server, {})
    sources = _which_source(_Capture.seen)
    assert sources == {"volume"}, f"expected only the volume source, got {sources}"


@pytest.mark.parametrize(
    "variable",
    ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN"],
)
def test_injected_variable_replaces_the_volume_credential(
    capture_server, oauth_volume, variable
):
    """Each injected variable SUPPRESSES the volume OAuth credential.

    Asserts on WHICH CANARY ARRIVED, not which header appeared.
    ``ANTHROPIC_AUTH_TOKEN`` and ``CLAUDE_CODE_OAUTH_TOKEN`` replace the BEARER
    VALUE rather than switching header type, so a "no bearer header" assertion
    would pass for ANTHROPIC_API_KEY and fail for these two -- measuring the
    header type instead of the credential source.

    Catches: a precedence change that lets the volume credential win, or reach
    the wire alongside the injected one.
    """
    _run_cli(oauth_volume, capture_server, {variable: _CANARY[variable]})
    sources = _which_source(_Capture.seen)

    assert _Capture.seen, "no requests captured; this assertion would be vacuous"
    assert variable in sources, (
        f"{variable} was set but its credential never reached the wire; "
        f"sources seen: {sorted(sources)}"
    )
    assert "volume" not in sources, (
        f"{variable} did NOT suppress the volume OAuth credential -- the volume "
        f"credential still reached the wire. sources seen: {sorted(sources)}"
    )


def test_no_assertion_message_can_interpolate_a_canary() -> None:
    """Assertion messages name SOURCES, never credential-shaped strings.

    The habit is what matters even when every value is synthetic: a test that
    prints a canary on failure prints a credential-shaped string on failure.

    Checked by AST over the assert MESSAGES specifically -- an earlier version
    grepped the raw file for ``_CANARY[`` and fired on the legitimate dict
    literal in ``_run_cli``, i.e. it matched a construction site rather than a
    message. Measuring the wrong thing while looking correct is the class this
    branch exists to remove, so it is measured precisely here.
    """
    import ast

    tree = ast.parse(Path(__file__).read_text())
    offenders: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert) or node.msg is None:
            continue
        for inner in ast.walk(node.msg):
            if isinstance(inner, ast.Name) and inner.id == "_CANARY":
                offenders.append(node.lineno)
            if isinstance(inner, ast.Subscript):
                value = inner.value
                if isinstance(value, ast.Name) and value.id == "_CANARY":
                    offenders.append(node.lineno)
    assert not offenders, (
        f"assertion messages at line(s) {offenders} interpolate a canary value; "
        f"emit the SOURCE NAME instead"
    )


def test_which_source_returns_names_only() -> None:
    """The comparison helper must never hand a caller a credential value."""
    result = _which_source([{"authorization": f"Bearer {_CANARY['volume']}"}])
    assert result == {"volume"}
    assert all(isinstance(name, str) for name in result)
    assert not any(_CANARY[name] in name for name in result)


# ---------------------------------------------------------------------------
# LB3 -- distinctness IS the mechanism
# ---------------------------------------------------------------------------


def test_the_per_source_canaries_are_all_distinct() -> None:
    """Catches: making the canaries identical, which is LB2 by the back door.

    Source-based assertion only works if the sources are distinguishable. If a
    future edit made two canaries equal, every "which canary arrived" assertion
    would silently degrade into "some canary arrived" -- i.e. back to measuring
    the header type, which is the adjacency defect this oracle exists to avoid.
    """
    values = list(_CANARY.values())
    assert len(set(values)) == len(values), "per-source canaries must be distinct"
    # and no canary may be a substring of another, or containment matching lies
    for name, canary in _CANARY.items():
        others = [v for k, v in _CANARY.items() if k != name]
        assert not any(canary in other for other in others), name


# ---------------------------------------------------------------------------
# LB8 -- PRECEDENCE, asserted from measurement rather than assumed
# ---------------------------------------------------------------------------


def test_both_variables_set_still_suppresses_the_volume_credential(
    capture_server, oauth_volume
):
    """The BOTH-SET case. Precedence is half the claim, so it gets an arm.

    MEASURED RESULT, and it is NOT "one variable wins": with both
    ``ANTHROPIC_API_KEY`` and ``ANTHROPIC_AUTH_TOKEN`` set, BOTH canaries reach
    the wire -- the CLI sends them in DIFFERENT headers (``x-api-key`` and the
    bearer) rather than selecting between them.

    So this asserts what was observed rather than a tidier story:
      - the VOLUME credential is still suppressed (the invariant that matters);
      - and both injected credentials are transmitted.

    That second half is worth an operator's attention: setting both EXPOSES
    BOTH. It is asserted here so the behaviour cannot change silently, and so
    nobody later documents a precedence that does not exist.
    """
    _run_cli(
        oauth_volume,
        capture_server,
        {
            "ANTHROPIC_API_KEY": _CANARY["ANTHROPIC_API_KEY"],
            "ANTHROPIC_AUTH_TOKEN": _CANARY["ANTHROPIC_AUTH_TOKEN"],
        },
    )
    assert _Capture.seen, "no requests captured; this assertion would be vacuous"
    sources = _which_source(_Capture.seen)

    assert "volume" not in sources, (
        f"the volume OAuth credential survived with both variables set; "
        f"sources seen: {sorted(sources)}"
    )
    assert sources == {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"}, (
        f"expected BOTH injected credentials on the wire (measured behaviour: "
        f"the CLI does not select between them); sources seen: {sorted(sources)}"
    )
