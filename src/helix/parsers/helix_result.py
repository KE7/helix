"""Parse ``HELIX_RESULT=<per-example [score, side_info] pairs>``.

GEPA ``optimize_anything`` evaluator-contract parity.  GEPA's O.A.
evaluator returns ``(score: float, side_info: dict)`` **per example**
(``src/gepa/optimize_anything.py:387-438``); the O.A. adapter then wraps
that per-example stream into ``EvaluationBatch(scores=list[float],
trajectories=list[SideInfo], ...)``
(``optimize_anything_adapter.py:218-292``).  This parser hands HELIX
the same per-example stream.

**BREAKING (pre-1.0):** ``helix_result`` previously accepted
``[score: float, side_info: dict]`` with an id-keyed
``side_info["scores"]`` sub-dict.  That contract silently failed the
minibatch gate whenever the evaluator keyed its dict by aggregate
metric names (``task__metric``) instead of the per-example ids HELIX
wrote to ``helix_batch.json`` (``task__trialN``): 113 generations of
evolution were once wasted on exactly that zero-vs-zero footgun.
See ``/tmp/gepa_audit_report.md`` for the full audit.

Contract
--------
The evaluator (run with ``cwd=candidate.worktree_path``) emits exactly
one line on stdout of the form::

    HELIX_RESULT=[[s_0, side_info_0], [s_1, side_info_1], ..., [s_{N-1}, side_info_{N-1}]]

where element ``i`` corresponds to the id at position ``i`` in the
``helix_batch.json`` HELIX wrote pre-invocation (see
:func:`helix.evolution._write_helix_batch`).  Each inner pair is
``[score_i: float, side_info_i: dict]`` — matching GEPA O.A.'s
per-example return.

``len(payload) == len(ids)`` is required.

Per-example side_info is a freeform dict (evaluator's own
observability).  The executor stores the full list on
:class:`helix.population.EvalResult.per_example_side_info` for the
reflection prompt (actual wiring into the mutation prompt is a
follow-up PR).

Reserved key: ``side_info_i["scores"]``
---------------------------------------
Mirrors GEPA's ``OptimizeAnythingAdapter._process_side_info``
(``optimize_anything_adapter.py:260-272``).  When a per-example
``side_info_i["scores"]`` is a dict of ``{objective_name: float}``,
HELIX harvests it into the corresponding slot of
:attr:`helix.population.EvalResult.objective_scores`
(``list[dict[str, float]]`` positional to the ids).  That axis feeds
the multi-axis Pareto frontier when ``evolution.frontier_type`` is
``"objective"``, ``"hybrid"``, or ``"cartesian"`` (wired in a later
commit).  No consumer in this commit; the harvest is strictly pass-
through on :class:`EvalResult`.

Aggregate
---------
* ``instance_scores = dict(zip(ids, [p[0] for p in payload]))`` —
  feeds the minibatch gate at ``evolution.py:1920-1939`` as before.
* ``scores["success"] = mean([p[0] for p in payload])`` — or 0.0 when
  ``returncode != 0`` or the payload is empty.

Strictness
----------
The parser raises :class:`helix.exceptions.EvaluatorError` on any
deviation from the contract — missing ``helix_batch.json``, missing /
malformed ``HELIX_RESULT=`` line, wrong outer shape, per-example
element not a 2-tuple, non-numeric / non-finite score, or length
mismatch.  Silent fallback would reintroduce the exact footgun this
rewrite exists to close.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from helix.exceptions import EvaluatorError


def _read_helix_batch(worktree_path: str | Path) -> list[str]:
    """Read the id list HELIX wrote to ``{worktree}/helix_batch.json``.

    Raises
    ------
    EvaluatorError
        If the file is missing or not a JSON list of strings.
    """
    batch_file = Path(worktree_path) / "helix_batch.json"
    if not batch_file.exists():
        raise EvaluatorError(
            "helix_result parser requires {worktree}/helix_batch.json "
            "to be present (HELIX writes it before every evaluator "
            "invocation).  Make sure the evaluator runs with "
            "cwd=candidate.worktree_path.",
            operation="parse_helix_result",
            cwd=str(worktree_path),
        )
    try:
        raw = batch_file.read_text()
        ids_any: Any = json.loads(raw)
    except (OSError, json.JSONDecodeError) as e:
        raise EvaluatorError(
            f"Failed to read helix_batch.json: {e}",
            operation="parse_helix_result",
            cwd=str(worktree_path),
        ) from e
    if not isinstance(ids_any, list) or not all(isinstance(x, str) for x in ids_any):
        raise EvaluatorError(
            "helix_batch.json must contain a JSON list[str] of example ids "
            f"(got {type(ids_any).__name__})",
            operation="parse_helix_result",
            cwd=str(worktree_path),
        )
    return [str(x) for x in ids_any]


def _extract_helix_result_line(stdout: str) -> str | None:
    """Return the last ``HELIX_RESULT=``-prefixed line, or ``None``.

    Matches :func:`helix.executor.run_evaluator`'s reverse-scan order
    so the two agree on "which line wins" when (buggy) evaluators emit
    more than one.
    """
    for line in reversed(stdout.splitlines()):
        if line.startswith("HELIX_RESULT="):
            return line
    return None


def _harvest_objective_scores(side_info: dict[str, Any]) -> dict[str, float]:
    """Extract ``side_info["scores"]`` into a ``dict[str, float]``.

    Mirrors GEPA's ``OptimizeAnythingAdapter._process_side_info``
    (``optimize_anything_adapter.py:260-272``): the reserved
    ``"scores"`` key carries a dict of ``{objective_name: float}`` that
    becomes the per-example ``objective_scores`` slot feeding the
    multi-axis Pareto frontier.  Non-dict values and non-numeric /
    non-finite entries are silently dropped so evaluators can stuff
    arbitrary diagnostics under ``"scores"`` without the parser
    rejecting well-formed payloads elsewhere (the primary per-example
    score is what the minibatch gate depends on; the objective axis
    is best-effort).
    """
    raw = side_info.get("scores")
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        if not isinstance(k, str):
            continue
        if isinstance(v, bool):
            out[k] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)) and math.isfinite(float(v)):
            out[k] = float(v)
    return out


def _coerce_score(
    raw_val: Any,
    eid: str,
    ctx_stdout: str,
    ctx_stderr: str,
    returncode: int,
) -> float:
    """Coerce a per-example raw score to a finite float or raise."""
    if isinstance(raw_val, bool):
        val = 1.0 if raw_val else 0.0
    elif isinstance(raw_val, (int, float)):
        val = float(raw_val)
    else:
        raise EvaluatorError(
            f"helix_result score for id {eid!r} is not numeric "
            f"(got {type(raw_val).__name__}: {raw_val!r}).",
            operation="parse_helix_result",
            stdout=ctx_stdout,
            stderr=ctx_stderr,
            exit_code=returncode,
        )
    if not math.isfinite(val):
        raise EvaluatorError(
            f"helix_result score for id {eid!r} is non-finite ({val!r}).",
            operation="parse_helix_result",
            stdout=ctx_stdout,
            stderr=ctx_stderr,
            exit_code=returncode,
        )
    return val


def parse(
    returncode: int,
    stdout: str,
    stderr: str,
    worktree_path: str | Path,
) -> tuple[
    dict[str, float],
    dict[str, float],
    list[dict[str, Any]],
    list[dict[str, float]],
]:
    """Parse per-example ``[score, side_info]`` pairs from ``HELIX_RESULT=``.

    Parameters
    ----------
    returncode:
        Evaluator subprocess exit code.  Non-zero zeroes the aggregate
        ``success`` score but leaves per-example data intact so
        reflection / diagnostics remain informative.
    stdout, stderr:
        Captured evaluator output.  ``stderr`` is unused by the parser
        but kept in the signature for registry-contract uniformity.
    worktree_path:
        Evaluator cwd — the directory HELIX wrote ``helix_batch.json``
        to before invoking the evaluator.

    Returns
    -------
    tuple
        Four elements, positionally aligned to the id list from
        ``helix_batch.json``:

        * ``scores: dict[str, float]`` — top-level aggregate
          (``scores["success"] = mean(per-example scores)``,
          0.0 on non-zero exit or empty list).
        * ``instance_scores: dict[str, float]`` —
          ``dict(zip(ids, per_example_scores))``.  Feeds the minibatch
          gate at :func:`helix.evolution._minibatch_gate_accept`.
        * ``per_example_side_info: list[dict[str, Any]]`` — the raw
          freeform side_info dict for each example, in id order.
          Stored on :class:`helix.population.EvalResult` for the
          reflection prompt.
        * ``objective_scores: list[dict[str, float]]`` — the
          ``side_info_i["scores"]`` harvest for each example, in id
          order.  GEPA analogue:
          :attr:`gepa.core.adapter.EvaluationBatch.objective_scores`
          (``src/gepa/core/adapter.py:26``).  Feeds the multi-axis
          Pareto frontier; stored on
          :attr:`helix.population.EvalResult.objective_scores`.

    Raises
    ------
    EvaluatorError
        On any contract violation (see module docstring).
    """
    ids = _read_helix_batch(worktree_path)

    result_line = _extract_helix_result_line(stdout)
    if result_line is None:
        raise EvaluatorError(
            "helix_result parser found no HELIX_RESULT= line on stdout. "
            "Emit exactly one line of shape "
            "HELIX_RESULT=[[score_0, side_info_0], [score_1, side_info_1], ...] "
            "with one [score, side_info] pair per id in helix_batch.json.",
            operation="parse_helix_result",
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
        )

    try:
        payload: Any = json.loads(result_line.removeprefix("HELIX_RESULT="))
    except (json.JSONDecodeError, ValueError) as e:
        raise EvaluatorError(
            f"Failed to JSON-decode HELIX_RESULT= payload: {e}",
            operation="parse_helix_result",
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
        ) from e

    if not isinstance(payload, list):
        raise EvaluatorError(
            "HELIX_RESULT= payload must be a list of per-example "
            "[score, side_info] pairs; "
            f"got {type(payload).__name__}.  See the "
            "helix.parsers.helix_result module docstring for the "
            "BREAKING change from the previous [score, side_info_dict] "
            "shape.",
            operation="parse_helix_result",
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
        )

    # Guard the two legacy scalar-plus-dict shapes with explicit error
    # messages so migrators get a pointer instead of a cryptic
    # "element is not a list" further down.
    if (
        len(payload) == 2
        and isinstance(payload[0], (int, float))
        and not isinstance(payload[0], bool)
        and isinstance(payload[1], dict)
    ):
        raise EvaluatorError(
            "HELIX_RESULT= looks like the removed legacy shape "
            "[score: float, side_info: dict].  The new contract is a "
            "list of per-example [score, side_info] pairs, one per id "
            "in helix_batch.json.  See helix.parsers.helix_result for "
            "migration guidance.",
            operation="parse_helix_result",
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
        )
    if (
        len(payload) == 2
        and isinstance(payload[0], list)
        and isinstance(payload[1], dict)
        and all(
            isinstance(x, (int, float)) and not isinstance(x, bool)
            for x in payload[0]
        )
    ):
        raise EvaluatorError(
            "HELIX_RESULT= looks like the intermediate shape "
            "[scores_list, side_info_dict] (one batch-level side_info). "
            "The contract is now a list of per-example [score, side_info] "
            "pairs — one side_info per example, not one per batch.  "
            "Zip your scores with per-example side_info dicts before "
            "emitting.",
            operation="parse_helix_result",
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
        )

    if len(payload) != len(ids):
        raise EvaluatorError(
            f"helix_result length mismatch: payload has {len(payload)} "
            f"per-example entries but helix_batch.json has {len(ids)} ids. "
            "The evaluator must emit exactly one [score, side_info] pair "
            "per id, in the same order.",
            operation="parse_helix_result",
            stdout=stdout,
            stderr=stderr,
            exit_code=returncode,
        )

    instance_scores: dict[str, float] = {}
    per_example_side_info: list[dict[str, Any]] = []
    objective_scores: list[dict[str, float]] = []

    for eid, entry in zip(ids, payload):
        if not isinstance(entry, list) or len(entry) != 2:
            raise EvaluatorError(
                f"helix_result per-example entry for id {eid!r} must be "
                f"a 2-element [score, side_info] list; got "
                f"{type(entry).__name__}"
                + (f" (len={len(entry)})" if isinstance(entry, list) else "")
                + ".",
                operation="parse_helix_result",
                stdout=stdout,
                stderr=stderr,
                exit_code=returncode,
            )
        raw_score, raw_side_info = entry
        score_val = _coerce_score(raw_score, eid, stdout, stderr, returncode)
        if raw_side_info is None:
            side_info: dict[str, Any] = {}
        elif isinstance(raw_side_info, dict):
            side_info = raw_side_info
        else:
            raise EvaluatorError(
                f"helix_result side_info for id {eid!r} must be a dict "
                f"(or null); got {type(raw_side_info).__name__}.",
                operation="parse_helix_result",
                stdout=stdout,
                stderr=stderr,
                exit_code=returncode,
            )

        instance_scores[eid] = score_val
        per_example_side_info.append(side_info)
        objective_scores.append(_harvest_objective_scores(side_info))

    if returncode != 0 or not instance_scores:
        success = 0.0
    else:
        success = sum(instance_scores.values()) / len(instance_scores)

    scores: dict[str, float] = {"success": success}
    return scores, instance_scores, per_example_side_info, objective_scores
