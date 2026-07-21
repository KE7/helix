"""Validate and emit a machine-readable HELIX state/ledger/accounting audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EXPECTED_P = 2
EXPECTED_N = 2
EXPECTED_TASKS = EXPECTED_P * EXPECTED_N
TERMINAL_STATUSES = {
    "skipped",
    "failed",
    "tampered",
    "rejected",
    "applied",
    "interrupted",
}
TERMINAL_CLEANUP = {"not_required", "removed", "missing"}
SELECTION_RESULTS = {"not_applicable", "not_selected", "selected"}
OBSOLETE_TASK_KEYS = {
    "reserved_child_id",
    "parent_slot",
    "mutation_slot",
    "evaluation_delta",
}


class InspectionError(ValueError):
    """Raised when durable run evidence does not satisfy this demo's contract."""


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InspectionError(f"{label} must be an object")
    return value


def _required(value: Mapping[str, Any], key: str, label: str) -> Any:
    if key not in value:
        raise InspectionError(f"{label} is missing required key {key!r}")
    return value[key]


def _integer(
    value: Mapping[str, Any], key: str, label: str, *, minimum: int = 0
) -> int:
    raw = _required(value, key, label)
    if isinstance(raw, bool) or not isinstance(raw, int) or raw < minimum:
        raise InspectionError(f"{label}.{key} must be an integer >= {minimum}")
    return int(raw)


def _string(value: Mapping[str, Any], key: str, label: str) -> str:
    raw = _required(value, key, label)
    if not isinstance(raw, str) or not raw:
        raise InspectionError(f"{label}.{key} must be a non-empty string")
    return raw


def _validate_task(
    raw_task: object,
    *,
    batch_id: str,
    task_index: int,
    candidate_ids: set[str],
) -> tuple[dict[str, Any], int]:
    label = f"batch {batch_id!r} task {task_index}"
    task = _mapping(raw_task, label)
    obsolete = sorted(OBSOLETE_TASK_KEYS.intersection(task))
    if obsolete:
        raise InspectionError(f"{label} uses obsolete key(s): {', '.join(obsolete)}")

    if _string(task, "batch_id", label) != batch_id:
        raise InspectionError(f"{label}.batch_id does not match its batch")
    if _integer(task, "p", label, minimum=1) != EXPECTED_P:
        raise InspectionError(f"{label}.p must equal {EXPECTED_P}")
    if _integer(task, "n", label, minimum=1) != EXPECTED_N:
        raise InspectionError(f"{label}.n must equal {EXPECTED_N}")
    if _integer(task, "task_index", label) != task_index:
        raise InspectionError(f"{label} is not in contiguous task-index order")

    expected_parent, expected_mutation = divmod(task_index, EXPECTED_N)
    parent_group = _integer(task, "parent_group", label)
    mutation_index = _integer(task, "mutation_index", label)
    if (parent_group, mutation_index) != (expected_parent, expected_mutation):
        raise InspectionError(
            f"{label} is not parent-major; expected "
            f"({expected_parent}, {expected_mutation})"
        )
    _string(task, "parent_id", label)

    child_id = _string(task, "child_id", label)
    if child_id in candidate_ids:
        raise InspectionError(f"duplicate child_id across proposal batches: {child_id}")
    candidate_ids.add(child_id)

    status = _string(task, "status", label)
    if status not in TERMINAL_STATUSES:
        raise InspectionError(f"{label}.status is not terminal: {status!r}")
    selection = _string(task, "selection", label)
    if selection not in SELECTION_RESULTS:
        raise InspectionError(
            f"{label}.selection is pending or invalid for a complete batch: "
            f"{selection!r}"
        )
    cleanup = _string(task, "cleanup", label)
    if cleanup == "failed":
        raise InspectionError(f"{label}.cleanup failed in a complete batch")
    if cleanup not in TERMINAL_CLEANUP:
        raise InspectionError(f"{label}.cleanup is not terminal: {cleanup!r}")
    if _required(task, "budget_accounted", label) is not True:
        raise InspectionError(f"{label}.budget_accounted must be true")
    if not isinstance(_required(task, "applied", label), bool):
        raise InspectionError(f"{label}.applied must be a boolean")

    charge = _mapping(_required(task, "budget_charge", label), f"{label}.budget_charge")
    evaluations = _integer(charge, "evaluations", f"{label}.budget_charge")
    return (
        {
            "task_index": task_index,
            "parent_group": parent_group,
            "mutation_index": mutation_index,
            "parent_id": task["parent_id"],
            "child_id": child_id,
            "status": status,
            "selection": selection,
            "cleanup": cleanup,
            "applied": task["applied"],
            "budget_accounted": True,
            "budget_charge": dict(charge),
        },
        evaluations,
    )


def summarize(project: Path) -> dict[str, Any]:
    state_path = project / ".helix" / "state.json"
    state_bytes = state_path.read_bytes()
    state = _mapping(json.loads(state_bytes), "state")

    budget = _mapping(_required(state, "budget", "state"), "state.budget")
    global_evaluations = _integer(budget, "evaluations", "state.budget")
    if _integer(state, "generation", "state") != 1:
        raise InspectionError("state.generation must equal 1")
    raw_batches = _required(state, "proposal_batches", "state")
    if not isinstance(raw_batches, list) or len(raw_batches) != 1:
        raise InspectionError("state.proposal_batches must contain exactly one batch")

    candidate_ids: set[str] = set()
    ordered_ids: list[str] = []
    proposal_evaluations = 0
    batches: list[dict[str, Any]] = []
    for batch_index, raw_batch in enumerate(raw_batches):
        label = f"proposal_batches[{batch_index}]"
        batch = _mapping(raw_batch, label)
        batch_id = _string(batch, "batch_id", label)
        if _integer(batch, "p", label, minimum=1) != EXPECTED_P:
            raise InspectionError(f"{label}.p must equal {EXPECTED_P}")
        if _integer(batch, "n", label, minimum=1) != EXPECTED_N:
            raise InspectionError(f"{label}.n must equal {EXPECTED_N}")
        if _required(batch, "phase", label) != "complete":
            raise InspectionError(f"{label}.phase must equal 'complete'")

        before = _integer(batch, "budget_before_dispatch", label)
        after = _integer(batch, "budget_after_apply", label)
        if after < before:
            raise InspectionError(f"{label} budget decreased across dispatch")
        raw_tasks = _required(batch, "tasks", label)
        if not isinstance(raw_tasks, list) or len(raw_tasks) != EXPECTED_TASKS:
            raise InspectionError(
                f"{label}.tasks must contain exactly {EXPECTED_TASKS} entries"
            )

        tasks: list[dict[str, Any]] = []
        batch_charge = 0
        for task_index, raw_task in enumerate(raw_tasks):
            task_summary, charge = _validate_task(
                raw_task,
                batch_id=batch_id,
                task_index=task_index,
                candidate_ids=candidate_ids,
            )
            tasks.append(task_summary)
            ordered_ids.append(task_summary["child_id"])
            batch_charge += charge

        budget_delta = after - before
        if batch_charge != budget_delta:
            raise InspectionError(
                f"{label} task evaluation charge {batch_charge} does not equal "
                f"budget delta {budget_delta}"
            )
        proposal_evaluations += batch_charge
        batches.append(
            {
                "batch_id": batch_id,
                "p": EXPECTED_P,
                "n": EXPECTED_N,
                "phase": "complete",
                "budget_before_dispatch": before,
                "budget_after_apply": after,
                "evaluation_charge": batch_charge,
                "tasks": tasks,
            }
        )

    if proposal_evaluations > global_evaluations:
        raise InspectionError(
            "proposal evaluation charges exceed the global evaluation budget"
        )
    nonproposal_evaluations = global_evaluations - proposal_evaluations
    return {
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "schema_version": state.get("schema_version"),
        "generation": state.get("generation"),
        "frontier": state.get("frontier", []),
        "active_frontier": state.get("active_frontier", {}),
        "budget": dict(budget),
        "mutation_counter": state.get("mutation_counter"),
        "proposal_batches": batches,
        "candidate_ids": ordered_ids,
        "candidate_ids_distinct": True,
        "candidate_ids_parent_major": True,
        "accounting": {
            "global_evaluations": global_evaluations,
            "proposal_evaluations": proposal_evaluations,
            "nonproposal_evaluations": nonproposal_evaluations,
        },
        "evaluation_artifacts": len(
            list((project / "artifacts" / "evaluations").glob("*.json"))
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(summarize(args.dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
