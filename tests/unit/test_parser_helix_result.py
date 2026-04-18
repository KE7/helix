"""Unit tests for the ``helix_result`` score parser.

The parser is the companion to the ``HELIX_RESULT=`` override in
``helix.executor`` and must tolerate missing / malformed evaluator output
while still populating ``instance_scores`` from ``side_info["scores"]``
when the payload is well-formed.
"""

from __future__ import annotations

import json

from helix.parsers import get_parser
from helix.parsers.helix_result import parse as helix_result_parse


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHelixResultWellFormed:
    def test_scores_and_instance_scores_populated(self):
        side_info = {
            "accuracy": 0.42,
            "scores": {
                "cube_lifting__success": 1.0,
                "cube_stack__success": 0.0,
                "nut_assembly__success": 0.5,
            },
        }
        line = f"HELIX_RESULT={json.dumps([0.42, side_info])}"
        stdout = f"preamble\n{line}\ntrailing log\n"

        scores, instance_scores = helix_result_parse(0, stdout, "")

        assert scores["success"] == 0.42
        assert scores["accuracy"] == 0.42
        assert instance_scores == {
            "cube_lifting__success": 1.0,
            "cube_stack__success": 0.0,
            "nut_assembly__success": 0.5,
        }

    def test_last_helix_result_line_wins(self):
        # Mirrors executor's reverse-scan behaviour: the last line wins.
        line1 = f"HELIX_RESULT={json.dumps([0.1, {'scores': {'a__m': 0.1}}])}"
        line2 = f"HELIX_RESULT={json.dumps([0.9, {'scores': {'b__m': 0.9}}])}"
        stdout = f"{line1}\n{line2}\n"

        scores, instance_scores = helix_result_parse(0, stdout, "")

        assert scores["success"] == 0.9
        assert instance_scores == {"b__m": 0.9}

    def test_non_numeric_instance_values_are_dropped(self):
        side_info = {"scores": {"good__m": 0.75, "bad__m": "nope", "none__m": None}}
        line = f"HELIX_RESULT={json.dumps([0.75, side_info])}"
        stdout = f"{line}\n"

        _scores, instance_scores = helix_result_parse(0, stdout, "")

        assert instance_scores == {"good__m": 0.75}

    def test_accuracy_omitted_when_missing(self):
        side_info = {"scores": {"a__m": 1.0}}
        line = f"HELIX_RESULT={json.dumps([0.5, side_info])}"

        scores, _ = helix_result_parse(0, line + "\n", "")

        assert "accuracy" not in scores
        assert scores["success"] == 0.5

    def test_bool_top_level_score_true(self):
        # bool is a subclass of int; float(True) == 1.0.
        side_info = {"scores": {"a__m": 1.0}}
        line = f"HELIX_RESULT={json.dumps([True, side_info])}"

        scores, instance_scores = helix_result_parse(0, line + "\n", "")

        assert scores["success"] == 1.0
        assert instance_scores == {"a__m": 1.0}

    def test_bool_top_level_score_false(self):
        side_info = {"scores": {"a__m": 0.0}}
        line = f"HELIX_RESULT={json.dumps([False, side_info])}"

        scores, instance_scores = helix_result_parse(0, line + "\n", "")

        assert scores["success"] == 0.0
        assert instance_scores == {"a__m": 0.0}

    def test_int_top_level_score(self):
        side_info = {"scores": {"a__m": 1.0}}
        line = f"HELIX_RESULT={json.dumps([1, side_info])}"

        scores, instance_scores = helix_result_parse(0, line + "\n", "")

        assert scores["success"] == 1.0
        assert instance_scores == {"a__m": 1.0}

    def test_empty_scores_dict_yields_empty_instance_scores(self):
        side_info = {"scores": {}}
        line = f"HELIX_RESULT={json.dumps([0.5, side_info])}"

        _scores, instance_scores = helix_result_parse(0, line + "\n", "")

        assert instance_scores == {}

    def test_non_dict_scores_value_yields_empty_instance_scores(self):
        # side_info["scores"] is a string rather than a dict — must not crash.
        side_info = {"scores": "not-a-dict"}
        line = f"HELIX_RESULT={json.dumps([0.5, side_info])}"

        scores, instance_scores = helix_result_parse(0, line + "\n", "")

        assert scores["success"] == 0.5
        assert instance_scores == {}


# ---------------------------------------------------------------------------
# Fallback paths (missing / malformed / shape errors)
# ---------------------------------------------------------------------------


class TestHelixResultFallbacks:
    def test_missing_line_falls_back_to_exitcode_success(self):
        scores, instance_scores = helix_result_parse(0, "arbitrary log\n", "")
        assert scores == {"success": 1.0}
        assert instance_scores == {"success": 1.0}

    def test_missing_line_falls_back_to_exitcode_failure(self):
        scores, instance_scores = helix_result_parse(1, "crashed\n", "some err")
        assert scores == {"success": 0.0}
        assert instance_scores == {"success": 0.0}

    def test_malformed_json_falls_back(self):
        stdout = "HELIX_RESULT=not-valid-json\ntrailing\n"
        scores, instance_scores = helix_result_parse(0, stdout, "")
        assert scores == {"success": 1.0}
        assert instance_scores == {"success": 1.0}

    def test_wrong_shape_falls_back(self):
        # Not a 2-element list.
        stdout = f"HELIX_RESULT={json.dumps({'score': 0.5})}\n"
        scores, instance_scores = helix_result_parse(0, stdout, "")
        assert scores == {"success": 1.0}
        assert instance_scores == {"success": 1.0}

    def test_non_dict_side_info_yields_empty_instance_scores(self):
        # Score is still honoured, but instance_scores is empty.
        stdout = f"HELIX_RESULT={json.dumps([0.8, 'just a string'])}\n"
        scores, instance_scores = helix_result_parse(0, stdout, "")
        assert scores == {"success": 0.8}
        assert instance_scores == {}

    def test_nan_and_inf_are_rejected(self):
        # Sanity-check: json.loads accepts NaN/Infinity by default, so the
        # parser actually sees non-finite floats post-decode.
        decoded = json.loads(
            '[NaN, {"scores": {"a": 1.0, "b": NaN, "c": Infinity, "d": -Infinity}}]'
        )
        top_score, side_info = decoded
        import math as _math

        assert _math.isnan(top_score)
        assert _math.isnan(side_info["scores"]["b"])
        assert _math.isinf(side_info["scores"]["c"])
        assert _math.isinf(side_info["scores"]["d"])

        # NaN top-level score -> full fallback (both dicts == {'success': 1.0}).
        stdout = (
            'HELIX_RESULT=[NaN, {"scores": {"a": 1.0, "b": NaN, '
            '"c": Infinity, "d": -Infinity}}]\n'
        )
        scores, instance_scores = helix_result_parse(0, stdout, "")
        assert scores == {"success": 1.0}
        assert instance_scores == {"success": 1.0}

    def test_finite_score_drops_non_finite_instance_values(self):
        # When the top-level score is finite, the parser keeps the good
        # instance scores and silently drops NaN/+Inf/-Inf entries, matching
        # how non-numeric values are dropped today.
        stdout = (
            'HELIX_RESULT=[0.5, {"scores": {"a": 1.0, "b": NaN, '
            '"c": Infinity, "d": -Infinity}}]\n'
        )
        scores, instance_scores = helix_result_parse(0, stdout, "")
        assert scores["success"] == 0.5
        assert instance_scores == {"a": 1.0}


# ---------------------------------------------------------------------------
# Returncode semantics
# ---------------------------------------------------------------------------


class TestHelixResultReturncode:
    def test_nonzero_returncode_zeros_success(self):
        side_info = {"scores": {"t__m": 1.0}}
        line = f"HELIX_RESULT={json.dumps([0.9, side_info])}"
        scores, instance_scores = helix_result_parse(1, line + "\n", "")

        assert scores["success"] == 0.0
        # instance_scores still reflect evaluator's per-instance reports.
        assert instance_scores == {"t__m": 1.0}


# ---------------------------------------------------------------------------
# Registry wiring
# ---------------------------------------------------------------------------


class TestRegistryWiring:
    def test_get_parser_returns_helix_result(self):
        parser = get_parser("helix_result")
        assert parser is helix_result_parse
