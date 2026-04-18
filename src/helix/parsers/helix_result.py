"""Parse HELIX_RESULT= lines emitted by evaluators.

This is the companion parser to the ``HELIX_RESULT=`` override handled
inside :mod:`helix.executor` (see ``run_evaluator``). Evaluators that want
to hand HELIX a scalar score *and* a diagnostics dictionary on the GEPA
``(accuracy, side_info)`` contract emit exactly one line of the form::

    HELIX_RESULT=[<score>, <side_info_dict>]

as their last meaningful stdout line. The executor scans for that line,
overrides ``scores["success"]`` with ``<score>``, and stashes
``side_info`` on the :class:`EvalResult`.  This parser exists so the
config's ``score_parser = "helix_result"`` is a valid selector and so the
per-instance scores in ``side_info["scores"]`` (e.g. ``task__metric`` →
float) flow into HELIX's minibatch-gate via ``instance_scores``.

Fallbacks mirror ``exitcode``/``json_score`` semantics so a missing or
malformed ``HELIX_RESULT=`` line does not crash the evolve loop.
"""

from __future__ import annotations

import json
import math
from typing import Any


def parse(
    returncode: int, stdout: str, stderr: str
) -> tuple[dict[str, float], dict[str, float]]:
    """Parse ``HELIX_RESULT=[score, side_info]`` from evaluator stdout.

    Behaviour:
        * Scan stdout lines in reverse for the last line starting with
          ``HELIX_RESULT=``. Strip the prefix, ``json.loads`` the payload.
        * When the payload is well-formed ``[float, dict]``:
              - ``scores = {"success": float(payload[0])}``; if
                ``side_info`` carries a numeric ``"accuracy"`` field, also
                set ``scores["accuracy"]`` for readability.
              - ``instance_scores`` is populated from
                ``side_info.get("scores", {})`` when that value is a
                ``dict`` of ``str`` → number; otherwise empty.
              - If ``returncode != 0``, coerce ``scores["success"] = 0.0``
                to match ``json_score``'s failure semantics.
        * When the line is missing or malformed, fall back to
          ``exitcode`` semantics (``success = 1.0 if returncode == 0 else
          0.0``) so the evolve loop still runs.

    Note:
        The executor independently scans for ``HELIX_RESULT=`` and
        overrides ``scores["success"]`` from the same payload; this
        parser duplicates the scan so it remains useful on its own
        (tests, non-executor code paths) and so it can populate
        ``instance_scores`` from ``side_info["scores"]``.
    """
    fallback_success = 1.0 if returncode == 0 else 0.0

    # Find the last HELIX_RESULT= line (matches executor's reverse-scan order).
    result_line: str | None = None
    for line in reversed(stdout.splitlines()):
        if line.startswith("HELIX_RESULT="):
            result_line = line
            break

    if result_line is None:
        return (
            {"success": fallback_success},
            {"success": fallback_success},
        )

    try:
        payload: Any = json.loads(result_line[len("HELIX_RESULT="):])
    except (json.JSONDecodeError, ValueError, TypeError):
        return (
            {"success": fallback_success},
            {"success": fallback_success},
        )

    if not (isinstance(payload, list) and len(payload) == 2):
        return (
            {"success": fallback_success},
            {"success": fallback_success},
        )

    raw_score, raw_side_info = payload

    try:
        score_value = float(raw_score)
    except (TypeError, ValueError):
        return (
            {"success": fallback_success},
            {"success": fallback_success},
        )

    if not math.isfinite(score_value):
        return (
            {"success": fallback_success},
            {"success": fallback_success},
        )

    if returncode != 0:
        score_value = 0.0

    scores: dict[str, float] = {"success": score_value}

    side_info: dict[str, Any] = raw_side_info if isinstance(raw_side_info, dict) else {}

    # Cosmetic: surface accuracy alongside success when the evaluator reports it.
    raw_accuracy = side_info.get("accuracy")
    if isinstance(raw_accuracy, (int, float)) and not isinstance(raw_accuracy, bool):
        scores["accuracy"] = float(raw_accuracy)

    instance_scores: dict[str, float] = {}
    raw_instances = side_info.get("scores", {})
    if isinstance(raw_instances, dict):
        for k, v in raw_instances.items():
            if isinstance(v, bool):
                # bool is a subclass of int; treat as numeric 0/1.
                instance_scores[str(k)] = float(v)
                continue
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                instance_scores[str(k)] = float(v)

    return scores, instance_scores
