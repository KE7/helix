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
SUCCESSFUL_CLEANUP = {"not_required", "removed", "missing"}
TERMINAL_SELECTION = {"not_applicable", "not_selected", "selected"}


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def audit_state(
    data: dict[str, Any], *, require_terminal: bool = False
) -> dict[str, Any]:
    """Validate durable identities and conservation in a decoded state file.

    ``require_terminal`` is deliberately the fixed release gate for this demo's
    P=2,N=2 run. The default mode remains shape-agnostic so it can summarize the
    separate P=1,N=1 comparison without weakening the release assertion.
    """
    budget = data.get("budget")
    batches = data.get("proposal_batches")
    if not isinstance(budget, dict) or not isinstance(batches, list):
        raise ValueError("state must contain budget and proposal_batches")
    if require_terminal:
        if data.get("generation") != 1:
            raise ValueError("release gate requires state generation exactly 1")
        if len(batches) != 1:
            raise ValueError("release gate requires exactly one proposal batch")

    child_ids: list[str] = []
    ledger_evaluations = 0
    batch_summaries = []
    for batch in batches:
        if not isinstance(batch, dict) or not isinstance(batch.get("tasks"), list):
            raise ValueError("each proposal batch must contain a task list")
        p, n = int(batch["p"]), int(batch["n"])
        tasks = batch["tasks"]
        if require_terminal and (p, n) != (2, 2):
            raise ValueError("release gate requires proposal shape P=2,N=2")
        if len(tasks) != p * n:
            raise ValueError("proposal batch does not have exactly P*N tasks")
        batch_charge = 0
        terminal = batch.get("phase") == "complete"
        if require_terminal and not terminal:
            raise ValueError("release gate requires proposal phase exactly complete")
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
                and task.get("cleanup") in SUCCESSFUL_CLEANUP
                and task.get("budget_accounted") is True
                and task.get("selection") in TERMINAL_SELECTION
            )
            terminal = terminal and task_terminal
        after = batch.get("budget_after_apply")
        before = int(batch.get("budget_before_dispatch", 0))
        if batch.get("phase") == "complete":
            if (
                after is None
                or int(after) < before
                or int(after) - before != batch_charge
            ):
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
    final_budget_after = batches[-1].get("budget_after_apply") if batches else None
    budget_conserved = (
        final_budget_after is not None
        and int(final_budget_after) == evaluations
        and ledger_evaluations <= evaluations
    )
    if require_terminal and not budget_conserved:
        raise ValueError("global budget contains unexplained post-batch spend")
    encoded = json.dumps(data, sort_keys=True, separators=(",", ":")).encode()
    return {
        "state_sha256": _sha256(encoded),
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
        "budget_conserved": budget_conserved,
        "batches": batch_summaries,
        "scheduler_phase": data.get("scheduler_state", {}).get("phase"),
    }


def build_resume_manifest(state_path: Path) -> dict[str, Any]:
    """Return a deterministic, non-vacuous terminal-resume fingerprint.

    HELIX appends to ``helix.log`` during an otherwise idempotent terminal
    resume, and cleaned candidate worktrees are intentionally ephemeral.  The
    durable artifact view therefore excludes that log and the ``worktrees/``
    subtree, while independently hashing the exact state-file bytes, a stable
    semantic projection, and the remaining durable artifact inventory.

    No timestamps or absolute paths enter the result, so two manifests can be
    compared byte-for-byte across consecutive terminal resumes.
    """
    state_path = state_path.resolve()
    state_dir = state_path.parent
    data = json.loads(state_path.read_text())
    audit = audit_state(data, require_terminal=True)

    durable_files: list[dict[str, Any]] = []
    for path in sorted(state_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(state_dir)
        if relative == Path("helix.log"):
            continue
        if relative.parts and relative.parts[0] == "worktrees":
            continue
        payload = path.read_bytes()
        durable_files.append(
            {
                "path": relative.as_posix(),
                "sha256": _sha256(payload),
                "size": len(payload),
            }
        )

    stable_projection = {
        "schema_version": audit["schema_version"],
        "generation": audit["generation"],
        "frontier": audit["frontier"],
        "active_frontier": audit["active_frontier"],
        "candidate_ids": audit["candidate_ids"],
        "instance_scores": audit["instance_scores"],
        "budget": audit["budget"],
        "ledger_evaluations": audit["ledger_evaluations"],
        "nonproposal_evaluations": audit["nonproposal_evaluations"],
        "budget_conserved": audit["budget_conserved"],
        "batches": audit["batches"],
        "scheduler_phase": audit["scheduler_phase"],
    }
    stable_bytes = json.dumps(
        stable_projection, sort_keys=True, separators=(",", ":")
    ).encode()
    artifact_bytes = json.dumps(
        durable_files, sort_keys=True, separators=(",", ":")
    ).encode()
    tasks = data["proposal_batches"][0]["tasks"]

    manifest: dict[str, Any] = {
        "schema_version": audit["schema_version"],
        "generation": audit["generation"],
        "candidate_ids": audit["candidate_ids"],
        "candidate_count": audit["candidate_count"],
        "distinct_candidate_count": len(set(audit["candidate_ids"])),
        "frontier": audit["frontier"],
        "active_frontier": audit["active_frontier"],
        "budget": audit["budget"],
        "ledger_evaluations": audit["ledger_evaluations"],
        "nonproposal_evaluations": audit["nonproposal_evaluations"],
        "budget_conserved": audit["budget_conserved"],
        "batch_count": len(audit["batches"]),
        "batch_phases": [batch["phase"] for batch in audit["batches"]],
        "task_statuses": [task["status"] for task in tasks],
        "task_selections": [task["selection"] for task in tasks],
        "task_cleanups": [task["cleanup"] for task in tasks],
        "scheduler_phase": audit["scheduler_phase"],
        "durable_file_count": len(durable_files),
        "durable_total_bytes": sum(item["size"] for item in durable_files),
        "state_file_sha256": _sha256(state_path.read_bytes()),
        "stable_projection_sha256": _sha256(stable_bytes),
        "durable_artifact_manifest_sha256": _sha256(artifact_bytes),
    }
    manifest["substantive_key_count"] = len(manifest) + 1
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "state", nargs="?", type=Path, default=Path(".helix/state.json")
    )
    parser.add_argument("--require-terminal", action="store_true")
    parser.add_argument(
        "--resume-manifest",
        action="store_true",
        help="emit a deterministic terminal-state manifest for byte comparison",
    )
    args = parser.parse_args()
    if args.resume_manifest:
        result = build_resume_manifest(args.state)
    else:
        data = json.loads(args.state.read_text())
        result = audit_state(data, require_terminal=args.require_terminal)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
