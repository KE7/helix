"""Tests for the shared diagnostic rendering redactor."""

from __future__ import annotations

import json

from helix.redaction import REDACTED_VALUE, DiagnosticRedactor, redact_diagnostics


def test_redacts_substrings_and_json_encoded_values_recursively():
    secret = 'sk-secret-47-"quoted"'
    encoded_secret = json.dumps(secret)[1:-1]
    redactor = DiagnosticRedactor.from_values([secret])

    value = {
        "credential": f"prefix {secret} suffix",
        "nested": [f'{{"token": "{encoded_secret}"}}'],
    }

    redacted = redactor.redact(value)

    assert redacted == {
        "credential": f"prefix {REDACTED_VALUE} suffix",
        "nested": [f'{{"token": "{REDACTED_VALUE}"}}'],
    }
    assert "credential" in redacted
    assert secret not in str(redacted)


def test_short_and_empty_values_do_not_over_redact_unrelated_output():
    output = "low battery; yellow light; unremarkable diagnostics"

    assert redact_diagnostics(output, ["", "low", "yellow"]) == output


def test_non_string_values_are_preserved():
    value = {"count": 3, "enabled": True, "nothing": None}

    assert redact_diagnostics(value, ["sk-secret-47"]) == value
