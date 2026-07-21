"""Audit durable HELIX proposal, budget, and score state without benchmark data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TERMINAL_STATUSES = {
    "skipped",
    "failed",
    "tampered",
    "rejected",
    "applied",
    "interrupted",
}
TERMINAL_CLEANUP = {"not_required", "removed", "missing", "failed"}


def audit_state(data: dict[str, Any], *, require_terminal: bool = False) -> dict[str, Any]:
    """Validate P-by-N identities and conservation in a decoded state file."""
    budget = data.get("budget")
    batches = data.get("proposal_batches")
    if not isinstance(budget, dict) or not isinstance(batches, list):
        raise ValueError("state must contain budget and proposal_batches")

    child_ids: list[str] = []
    ledger_evaluations = 0
    batch_summaries = []
    for batch in batches:
        if not isinstance(batch, dict) or not isinstance(batch.get("tasks"), list):
            raise ValueError("each proposal batch must contain a task list")
        p, n = int(batch["p"]), int(batch["n"])
        tasks = batch["tasks"]
        if len(tasks) != p * n:
            raise ValueError("proposal batch does not have exactly P*N tasks")
        batch_charge = 0
        terminal = batch.get("phase") in {"complete", "interrupted"}
        for index, task in enumerate(tasks):
            expected_group, expected_mutation = divmod(index, n)
            if (
                task.get("task_index") != index
                or task.get("parent_group") != expected_group
                or task.get("mutation_index") != expected_mutation
                or task.get("p") != p
                or task.get("n") != n
                or task.get("batch_id") != batch.get("batch_id")
            ):
                raise ValueError("proposal task is not in parent-major P-by-N order")
            child_id = str(task.get("child_id", ""))
            if not child_id:
                raise ValueError("proposal task has an empty child ID")
            child_ids.append(child_id)
            charge = task.get("budget_charge")
            if not isinstance(charge, dict):
                raise ValueError("proposal task is missing its budget charge")
            batch_charge += int(charge.get("evaluations", 0))
            task_terminal = (
                task.get("status") in TERMINAL_STATUSES
                and task.get("cleanup") in TERMINAL_CLEANUP
                and task.get("budget_accounted") is True
            )
            terminal = terminal and task_terminal
        after = batch.get("budget_after_apply")
        before = int(batch.get("budget_before_dispatch", 0))
        if batch.get("phase") == "complete":
            if after is None or int(after) - before != batch_charge:
                raise ValueError("completed batch budget does not conserve charges")
        if require_terminal and not terminal:
            raise ValueError("proposal ledger is not terminal and fully accounted")
        ledger_evaluations += batch_charge
        batch_summaries.append(
            {
                "batch_id": batch.get("batch_id"),
                "p": p,
                "n": n,
                "phase": batch.get("phase"),
                "candidate_ids": [task["child_id"] for task in tasks],
                "evaluations": batch_charge,
                "terminal_and_accounted": terminal,
            }
        )

    if len(child_ids) != len(set(child_ids)):
        raise ValueError("candidate IDs are not globally distinct")
    evaluations = int(budget.get("evaluations", 0))
    if ledger_evaluations > evaluations:
        raise ValueError("proposal ledger charges exceed the global budget")
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return {
        "state_sha256": hashlib.sha256(encoded).hexdigest(),
        "schema_version": data.get("schema_version"),
        "generation": data.get("generation"),
        "frontier": data.get("frontier", []),
        "active_frontier": data.get("active_frontier", {}),
        "candidate_ids": child_ids,
        "candidate_count": len(child_ids),
        "instance_scores": data.get("instance_scores", {}),
        "budget": budget,
        "ledger_evaluations": ledger_evaluations,
        "nonproposal_evaluations": evaluations - ledger_evaluations,
        "batches": batch_summaries,
        "scheduler_state": data.get("scheduler_state", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("state", nargs="?", type=Path, default=Path(".helix/state.json"))
    parser.add_argument("--require-terminal", action="store_true")
    args = parser.parse_args()
    data = json.loads(args.state.read_text())
    print(json.dumps(audit_state(data, require_terminal=args.require_terminal), indent=2))


if __name__ == "__main__":
    main()
