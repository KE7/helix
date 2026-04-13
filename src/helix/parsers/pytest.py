"""Parse pytest -q output into scores and instance_scores."""

from __future__ import annotations

import re


def parse(stdout: str, stderr: str) -> tuple[dict[str, float], dict[str, float]]:
    """Parse pytest -q output.

    Returns:
        scores: dict with "pass_rate" and "duration"
        instance_scores: dict mapping test id to 1.0 (pass) or 0.0 (fail/error)
    """
    instance_scores: dict[str, float] = {}

    # Parse PASSED lines: "tests/test_auth.py::test_login PASSED"
    for match in re.finditer(r"^(\S+::\S+)\s+PASSED", stdout, re.MULTILINE):
        test_id = match.group(1)
        instance_scores[test_id] = 1.0

    # Parse FAILED lines: "FAILED tests/test_auth.py::test_rate - AssertionError"
    for match in re.finditer(r"^FAILED\s+(\S+::\S+)", stdout, re.MULTILINE):
        test_id = match.group(1)
        instance_scores[test_id] = 0.0

    # Parse ERROR lines: "ERROR tests/test_auth.py::test_foo"
    for match in re.finditer(r"^ERROR\s+(\S+::\S+)", stdout, re.MULTILINE):
        test_id = match.group(1)
        if test_id not in instance_scores:
            instance_scores[test_id] = 0.0

    # Parse summary line: "5 passed, 2 failed, 1 error in 3.42s"
    passed = 0
    failed = 0
    errors = 0
    duration = 0.0

    summary_match = re.search(
        r"(\d+)\s+passed(?:,\s+(\d+)\s+failed)?(?:,\s+(\d+)\s+error(?:s)?)?\s+in\s+([\d.]+)s",
        stdout,
    )
    if summary_match is None:
        # Try alternate: only failed or only passed
        summary_match = re.search(
            r"(?:(\d+)\s+passed)?(?:,?\s*(\d+)\s+failed)?(?:,?\s*(\d+)\s+error(?:s)?)?\s+in\s+([\d.]+)s",
            stdout,
        )

    if summary_match:
        passed = int(summary_match.group(1) or 0)
        failed = int(summary_match.group(2) or 0)
        errors = int(summary_match.group(3) or 0)
        duration = float(summary_match.group(4))

    total = passed + failed + errors
    pass_rate = (passed / total) if total > 0 else 0.0

    scores: dict[str, float] = {
        "pass_rate": pass_rate,
        "duration": duration,
    }

    return scores, instance_scores
