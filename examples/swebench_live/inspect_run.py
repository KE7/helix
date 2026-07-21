"""Emit a compact, machine-readable HELIX state/ledger/accounting audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def summarize(project: Path) -> dict[str, Any]:
    state_path = project / ".helix" / "state.json"
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    batches = state.get("proposal_batches", [])
    tasks = [task for batch in batches for task in batch.get("tasks", [])]
    candidate_ids = [str(task.get("reserved_child_id")) for task in tasks]
    return {
        "state_sha256": hashlib.sha256(state_bytes).hexdigest(),
        "schema_version": state.get("schema_version"),
        "generation": state.get("generation"),
        "frontier": state.get("frontier", []),
        "active_frontier": state.get("active_frontier", {}),
        "budget": state.get("budget", {}),
        "mutation_counter": state.get("mutation_counter"),
        "proposal_batches": [
            {
                "batch_id": batch.get("batch_id"),
                "p": batch.get("p"),
                "n": batch.get("n"),
                "phase": batch.get("phase"),
                "budget_before_dispatch": batch.get("budget_before_dispatch"),
                "budget_after_apply": batch.get("budget_after_apply"),
                "tasks": [
                    {
                        "task_index": task.get("task_index"),
                        "parent_slot": task.get("parent_slot"),
                        "mutation_slot": task.get("mutation_slot"),
                        "candidate_id": task.get("reserved_child_id"),
                        "status": task.get("status"),
                        "selection": task.get("selection"),
                        "cleanup": task.get("cleanup"),
                        "applied": task.get("applied"),
                        "evaluation_delta": task.get("evaluation_delta"),
                    }
                    for task in batch.get("tasks", [])
                ],
            }
            for batch in batches
        ],
        "candidate_ids": candidate_ids,
        "candidate_ids_distinct": len(candidate_ids) == len(set(candidate_ids)),
        "candidate_ids_parent_major": candidate_ids == sorted(
            candidate_ids,
            key=lambda value: tuple(
                int(part[1:]) for part in value.split("-") if part[:1] in {"g", "s"}
            ),
        ),
        "evaluation_artifacts": len(list((project / "artifacts" / "evaluations").glob("*.json"))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(summarize(args.dir.resolve()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
