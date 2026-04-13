"""Unit tests for HELIX score parsers."""

from __future__ import annotations

import pytest

from helix.parsers import get_parser
from helix.parsers.pytest import parse as pytest_parse
from helix.parsers.exitcode import parse as exitcode_parse


# ---------------------------------------------------------------------------
# Pytest parser fixtures
# ---------------------------------------------------------------------------

MIXED_OUTPUT = """\
tests/test_auth.py::test_login PASSED
tests/test_auth.py::test_logout PASSED
FAILED tests/test_auth.py::test_rate - AssertionError: rate limit not enforced
FAILED tests/test_auth.py::test_signup - ValueError: invalid email
5 passed, 2 failed in 3.42s
"""

ALL_PASSING_OUTPUT = """\
tests/test_math.py::test_add PASSED
tests/test_math.py::test_sub PASSED
tests/test_math.py::test_mul PASSED
3 passed in 1.23s
"""

ALL_FAILING_OUTPUT = """\
FAILED tests/test_math.py::test_add - AssertionError
FAILED tests/test_math.py::test_sub - AssertionError
2 failed in 0.55s
"""

EMPTY_OUTPUT = ""


# ---------------------------------------------------------------------------
# Pytest parser tests
# ---------------------------------------------------------------------------


class TestPytestParser:
    def test_mixed_pass_fail(self):
        scores, instance_scores = pytest_parse(MIXED_OUTPUT, "")
        assert "pass_rate" in scores
        assert "duration" in scores
        # 5 passed / (5+2) total
        assert abs(scores["pass_rate"] - 5 / 7) < 1e-6
        assert scores["duration"] == pytest.approx(3.42)

    def test_mixed_instance_scores_keys(self):
        scores, instance_scores = pytest_parse(MIXED_OUTPUT, "")
        assert "tests/test_auth.py::test_login" in instance_scores
        assert "tests/test_auth.py::test_logout" in instance_scores
        assert "tests/test_auth.py::test_rate" in instance_scores
        assert "tests/test_auth.py::test_signup" in instance_scores

    def test_mixed_instance_scores_values(self):
        scores, instance_scores = pytest_parse(MIXED_OUTPUT, "")
        assert instance_scores["tests/test_auth.py::test_login"] == 1.0
        assert instance_scores["tests/test_auth.py::test_logout"] == 1.0
        assert instance_scores["tests/test_auth.py::test_rate"] == 0.0
        assert instance_scores["tests/test_auth.py::test_signup"] == 0.0

    def test_all_passing(self):
        scores, instance_scores = pytest_parse(ALL_PASSING_OUTPUT, "")
        assert scores["pass_rate"] == pytest.approx(1.0)
        assert scores["duration"] == pytest.approx(1.23)
        assert len(instance_scores) == 3
        assert all(v == 1.0 for v in instance_scores.values())

    def test_all_passing_instance_scores(self):
        scores, instance_scores = pytest_parse(ALL_PASSING_OUTPUT, "")
        assert instance_scores["tests/test_math.py::test_add"] == 1.0
        assert instance_scores["tests/test_math.py::test_sub"] == 1.0
        assert instance_scores["tests/test_math.py::test_mul"] == 1.0

    def test_all_failing(self):
        scores, instance_scores = pytest_parse(ALL_FAILING_OUTPUT, "")
        assert scores["pass_rate"] == pytest.approx(0.0)
        assert all(v == 0.0 for v in instance_scores.values())

    def test_all_failing_instance_scores(self):
        scores, instance_scores = pytest_parse(ALL_FAILING_OUTPUT, "")
        assert instance_scores["tests/test_math.py::test_add"] == 0.0
        assert instance_scores["tests/test_math.py::test_sub"] == 0.0

    def test_empty_output(self):
        scores, instance_scores = pytest_parse(EMPTY_OUTPUT, "")
        assert scores["pass_rate"] == 0.0
        assert instance_scores == {}

    def test_empty_output_duration(self):
        scores, instance_scores = pytest_parse(EMPTY_OUTPUT, "")
        assert scores["duration"] == 0.0


# ---------------------------------------------------------------------------
# Exitcode parser tests
# ---------------------------------------------------------------------------


class TestExitcodeParser:
    def test_success_returncode_zero(self):
        scores, instance_scores = exitcode_parse(0, "", "")
        assert scores["success"] == 1.0
        assert instance_scores == {"success": 1.0}

    def test_failure_returncode_nonzero(self):
        scores, instance_scores = exitcode_parse(1, "", "")
        assert scores["success"] == 0.0
        assert instance_scores == {"success": 0.0}

    def test_failure_returncode_two(self):
        scores, instance_scores = exitcode_parse(2, "some output", "some error")
        assert scores["success"] == 0.0

    def test_failure_returncode_minus_one(self):
        scores, instance_scores = exitcode_parse(-1, "", "")
        assert scores["success"] == 0.0


# ---------------------------------------------------------------------------
# get_parser registry tests
# ---------------------------------------------------------------------------


class TestGetParser:
    def test_get_pytest_parser(self):
        parser = get_parser("pytest")
        assert callable(parser)
        scores, instance_scores = parser("", "")
        assert "pass_rate" in scores

    def test_get_exitcode_parser(self):
        parser = get_parser("exitcode")
        assert callable(parser)
        scores, instance_scores = parser(0, "", "")
        assert "success" in scores

    def test_unknown_parser_raises(self):
        with pytest.raises(KeyError, match="Unknown parser"):
            get_parser("nonexistent")
