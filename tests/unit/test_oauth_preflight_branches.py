"""Meta-tests for the OAuth-endpoint capability preflight.

The preflight decides whether T22-T25 RUN or SKIP, so it is itself a
load-bearing control -- and its dangerous failure mode is the quiet one:
*reporting UNREACHABLE when the endpoint is actually reachable* silently
converts a real failure into a permanent skip, and a skip reads as green in the
tier output.

Both branches are therefore tested. A skip-branch test alone would read as
"covered" while the fail branch rots -- the X8/X8b lesson applied here.

These are unit-tier: they exercise the CLASSIFIER over captured probe output
and never touch Docker or the network.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def _module():
    """Load the integration module WITHOUT its docker_integration marker."""
    path = (
        Path(__file__).resolve().parents[1]
        / "integration"
        / "test_oauth_refresh_suppression.py"
    )
    spec = importlib.util.spec_from_file_location("_oauth_refresh_probe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_any_http_status_counts_as_REACHABLE() -> None:
    """A 400/401/405 from the endpoint means it IS reachable.

    THE DISTINCTION THE WHOLE FIX TURNS ON. Treating a rejection as
    "unavailable" would reproduce the missing-vs-failed conflation removed from
    ``transcripts.py`` and ``_run_in_image`` -- inside the fix for it.

    Catches: classifying a fake-token rejection or a CLI exit code as absent
    egress, which would convert a REAL failure into a permanent skip.
    """
    module = _module()
    for status in ("400", "401", "404", "405", "200"):
        module._REACHABILITY.clear()
        module._probe_endpoint_reachability.__wrapped__ if False else None
        reachable, why = module._classify_probe_output(f"HTTP:{status}")
        assert reachable, f"HTTP {status} must count as reachable, got {why!r}"


@pytest.mark.parametrize(
    ("output", "expected_reason"),
    [
        ("curl: (6) Could not resolve host", "DNS"),
        ("curl: (7) Connection refused", "refused"),
        ("curl: (28) Connection timed out", "timed out"),
        ("CURL_FAILED", "reach"),
    ],
)
def test_only_connectivity_failures_count_as_UNREACHABLE(
    output: str, expected_reason: str
) -> None:
    """Skip ONLY on proven unreachability, and name the capability."""
    module = _module()
    reachable, why = module._classify_probe_output(output)
    assert not reachable
    assert expected_reason.lower() in why.lower(), why


def test_unclassified_output_is_treated_as_unreachable_but_says_so() -> None:
    """Fail closed on an output the classifier does not understand.

    A skip is the safe direction here ONLY because the reason string carries
    the raw output, so an unexplained skip is visibly unexplained rather than
    silently routine.
    """
    module = _module()
    reachable, why = module._classify_probe_output("something nobody anticipated")
    assert not reachable
    assert "unclassified" in why.lower()
    assert "something nobody anticipated" in why


def test_skip_message_names_the_capability_and_the_unverified_property() -> None:
    """Catches: a generic skip reason.

    "image unavailable" would hide which capability is missing and would let a
    reader conclude the suppression property was checked.
    """
    module = _module()
    module._REACHABILITY.clear()
    module._REACHABILITY["verdict"] = (False, "DNS resolution failed")
    with pytest.raises(Exception) as exc:
        module.require_oauth_endpoint_reachable()
    message = str(exc.value)
    assert "MISSING NETWORK CAPABILITY" in message
    assert "oauth/token" in message
    assert "UNVERIFIED" in message
    module._REACHABILITY.clear()


def test_reachable_verdict_does_not_skip() -> None:
    """Non-vacuity: the gate is not an unconditional skip.

    Without this, a preflight that always skipped would satisfy every test
    above -- and would silently retire four security tests.
    """
    module = _module()
    module._REACHABILITY.clear()
    module._REACHABILITY["verdict"] = (True, "endpoint returned HTTP 405")
    module.require_oauth_endpoint_reachable()  # must not raise
    module._REACHABILITY.clear()
