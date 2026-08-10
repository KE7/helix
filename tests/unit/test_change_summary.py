"""Tests for self-reported mutation change summaries."""

from __future__ import annotations

import json

from helix.change_summary import (
    CHANGE_SUMMARY_ARTIFACT_NAME,
    MAX_CHANGE_SUMMARY_BYTES,
    append_rejected_attempt,
    capture_change_summary,
    render_failure_history,
)
from helix.population import EvalResult


def _summary() -> dict[str, str]:
    return {
        "intent": "Fix the parser's off-by-one error.",
        "approach": "Adjust the final-token boundary check.",
        "expected_effect": "The last token is accepted without weakening validation.",
    }


def _evaluation() -> EvalResult:
    return EvalResult(
        candidate_id="g1-s0",
        scores={"quality": 0.4},
        instance_scores={"example-1": 0.4},
        asi={"stdout": "expected 2, got 1"},
    )


def test_good_summary_is_captured_attached_and_rendered_with_evaluator_output(tmp_path):
    (tmp_path / CHANGE_SUMMARY_ARTIFACT_NAME).write_text(json.dumps(_summary()))

    summary = capture_change_summary(tmp_path)
    history = append_rejected_attempt({}, "g0-s0", summary, _evaluation(), limit=3)
    rendered = render_failure_history(history["g0-s0"])

    assert summary == _summary()
    assert "Fix the parser" in rendered
    assert "expected 2, got 1" in rendered
    assert rendered.index("Fix the parser") < rendered.index("expected 2, got 1")


def test_missing_summary_is_recorded_but_not_rendered(tmp_path):
    history = append_rejected_attempt({}, "g0-s0", capture_change_summary(tmp_path), _evaluation())

    assert history["g0-s0"][0]["summary"] is None
    assert render_failure_history(history["g0-s0"]) == ""


def test_oversized_or_malformed_summary_is_absent(tmp_path):
    path = tmp_path / CHANGE_SUMMARY_ARTIFACT_NAME
    path.write_text("x" * (MAX_CHANGE_SUMMARY_BYTES + 1))
    assert capture_change_summary(tmp_path) is None

    path.write_text(json.dumps({"intent": "only one field"}))
    assert capture_change_summary(tmp_path) is None


def test_history_cap_evicts_oldest_attempt_first():
    history: dict[str, list[dict[str, object]]] = {}
    for index in range(4):
        summary = _summary() | {"intent": f"attempt {index}"}
        history = append_rejected_attempt(history, "g0-s0", summary, _evaluation(), limit=3)

    assert [item["summary"]["intent"] for item in history["g0-s0"]] == [
        "attempt 1",
        "attempt 2",
        "attempt 3",
    ]


def test_invalid_persisted_entry_is_omitted_instead_of_rendered():
    invalid = [{"summary": _summary(), "evaluator_output": "missing score"}]
    assert render_failure_history(invalid) == ""


def test_feedback_redacts_configured_secrets_before_persistence_or_rendering():
    secret = "very-secret-token"
    summary = _summary() | {"intent": f"use {secret}"}
    evaluation = EvalResult(
        candidate_id="g1-s0",
        scores={},
        instance_scores={"example-1": 0.4},
        asi={"stdout": json.dumps({"token": secret})},
    )

    history = append_rejected_attempt(
        {}, "g0-s0", summary, evaluation, secret_values=[secret]
    )
    rendered = render_failure_history(history["g0-s0"], [secret])

    assert secret not in json.dumps(history)
    assert secret not in rendered
    assert "<redacted>" in rendered
