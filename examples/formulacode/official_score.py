"""FormulaCode-compatible performance scoring for the pinned smoke subset.

The formulas in this module follow fc-eval's official FormulaCode scorer at
commit c08f665e7bf3b4de225b72dc02ce9b15b7aaba2b.  The implementation is kept
dependency-free so the evaluator can run in a small, pinned Python environment.

FormulaCode/fc-eval is BSD-3-Clause licensed.  See LICENSES.md next to this
file for attribution and source links.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
import statistics
from typing import Any


class ScoreInputError(ValueError):
    """Raised when measurement payloads cannot produce a faithful score."""


def geometric_mean(values: Sequence[float]) -> float:
    """Return the geometric mean of finite positive values."""

    if not values:
        raise ScoreInputError("geometric mean requires at least one value")
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ScoreInputError("geometric mean inputs must be finite and positive")
    return math.exp(math.fsum(math.log(value) for value in values) / len(values))


def _median(samples: Sequence[float], *, label: str) -> float:
    if not samples:
        raise ScoreInputError(f"{label} has no timing samples")
    value = float(statistics.median(samples))
    if not math.isfinite(value) or value <= 0:
        raise ScoreInputError(f"{label} median must be finite and positive")
    return value


def _deconstruct_benchmark(
    benchmark: str,
) -> tuple[str | None, str | None, str | None]:
    """Port fc-eval ``FormulaCodeParser._deconstruct_benchmark`` exactly."""

    name = benchmark.strip()
    if "(" in name and name.endswith(")"):
        name = name[: name.find("(")].strip()
    parts = [part for part in name.split(".") if part]
    if not parts:
        return (None, None, None)
    if len(parts) == 1:
        return (None, None, parts[0])
    if len(parts) >= 3 and parts[-2][:1].isupper():
        module = ".".join(parts[:-2]) or None
        return (module, parts[-2], parts[-1])
    module = ".".join(parts[:-1]) or None
    return (module, None, parts[-1])


def score_measurements(
    *,
    ordered_workloads: Sequence[str],
    nop_samples: Mapping[str, Sequence[float]],
    oracle_samples: Mapping[str, Sequence[float]],
    agent_samples: Mapping[str, Sequence[float]],
    correctness_passed: bool,
) -> dict[str, Any]:
    """Compute FormulaCode task speedup and advantage.

    Per workload, speedup is ``median(nop) / median(agent)`` and the human
    oracle speedup is ``median(nop) / median(oracle)``.  Task speedup is the
    geometric mean of agent speedups.  Primary advantage is level 4:
    ``gmean(agent speedups) - gmean(oracle speedups)``.

    On a correctness failure, fc-eval's revert-to-baseline rule is applied:
    every effective agent speedup becomes exactly 1.0 while the oracle stays
    unchanged.  ``ordered_workloads`` is deliberately a sequence rather than a
    set so repeated/padded HELIX inputs retain their official multiplicity.
    """

    if not ordered_workloads:
        raise ScoreInputError("at least one workload is required")

    per_benchmark: list[dict[str, Any]] = []
    agent_speedups: list[float] = []
    oracle_speedups: list[float] = []

    level_groups: dict[str, dict[tuple[str | None, ...], dict[str, list[float]]]] = (
        defaultdict(lambda: defaultdict(lambda: {"agent": [], "oracle": []}))
    )

    for name in ordered_workloads:
        if name not in nop_samples or name not in oracle_samples:
            raise ScoreInputError(f"missing pinned baseline for workload {name!r}")
        nop_median = _median(nop_samples[name], label=f"nop:{name}")
        oracle_median = _median(oracle_samples[name], label=f"oracle:{name}")
        oracle_speedup = nop_median / oracle_median

        if correctness_passed:
            if name not in agent_samples:
                raise ScoreInputError(f"missing agent measurement for {name!r}")
            agent_median = _median(agent_samples[name], label=f"agent:{name}")
            agent_speedup = nop_median / agent_median
        else:
            agent_median = None
            agent_speedup = 1.0

        module, class_name, function = _deconstruct_benchmark(name)
        groups = {
            "level1": (module,),
            "level2": (module, class_name),
            "level3": (module, class_name, function),
            "level4": (),
        }
        for level, group in groups.items():
            level_groups[level][group]["agent"].append(agent_speedup)
            level_groups[level][group]["oracle"].append(oracle_speedup)

        agent_speedups.append(agent_speedup)
        oracle_speedups.append(oracle_speedup)
        per_benchmark.append(
            {
                "workload": name,
                "nop_median_seconds": nop_median,
                "oracle_median_seconds": oracle_median,
                "agent_median_seconds": agent_median,
                "agent/nop": agent_speedup,
                "oracle/nop": oracle_speedup,
                "advantage": agent_speedup - oracle_speedup,
                "fallback_to_baseline": not correctness_passed,
            }
        )

    def level_advantage(level: str) -> float:
        values: list[float] = []
        for group in level_groups[level].values():
            values.append(
                geometric_mean(group["agent"]) - geometric_mean(group["oracle"])
            )
        return math.fsum(values) / len(values)

    task_speedup = geometric_mean(agent_speedups)
    oracle_task_speedup = geometric_mean(oracle_speedups)
    level1 = level_advantage("level1")
    level2 = level_advantage("level2")
    level3 = level_advantage("level3")
    level4 = level_advantage("level4")

    return {
        "success": correctness_passed,
        "fallback_to_baseline": not correctness_passed,
        "task_speedup": task_speedup,
        "oracle_task_speedup": oracle_task_speedup,
        "agent_advantage": level4,
        "agent_advantage_level1": level1,
        "agent_advantage_level2": level2,
        "agent_advantage_level3": level3,
        "agent_advantage_level4": level4,
        "num_valid_benchmarks": len(per_benchmark),
        "total_benchmarks": len(per_benchmark),
        "per_benchmark_speedups": per_benchmark,
    }


def helix_result_pairs(
    metrics: Mapping[str, Any],
    ordered_ids: Sequence[str],
    id_to_workload: Mapping[str, str],
    *,
    failure_kind: str | None = None,
) -> list[list[Any]]:
    """Translate task metrics to HELIX's positional per-example contract."""

    per_benchmark = metrics["per_benchmark_speedups"]
    if len(per_benchmark) != len(ordered_ids):
        raise ScoreInputError("metric cardinality does not match requested ids")

    score = float(metrics["agent_advantage"])
    pairs: list[list[Any]] = []
    for instance_id, detail in zip(ordered_ids, per_benchmark, strict=True):
        if instance_id not in id_to_workload:
            raise ScoreInputError(f"unknown instance id {instance_id!r}")
        side_info = {
            "instance_id": instance_id,
            "workload": id_to_workload[instance_id],
            "success": bool(metrics["success"]),
            "fallback_to_baseline": bool(metrics["fallback_to_baseline"]),
            "failure_kind": failure_kind,
            "per_benchmark": detail,
            "task_speedup": float(metrics["task_speedup"]),
            "oracle_task_speedup": float(metrics["oracle_task_speedup"]),
            "agent_advantage": score,
            "scores": {
                "advantage": score,
                "speedup": float(metrics["task_speedup"]),
                "correctness": 1.0 if metrics["success"] else 0.0,
            },
        }
        pairs.append([score, side_info])
    return pairs
