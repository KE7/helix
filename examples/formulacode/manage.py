"""Set up, inspect, fingerprint, and clean the pinned FormulaCode demo."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Any, cast

from workloads import BASE_COMMIT, ORACLE_COMMIT, REPOSITORY, TASK_ID, WORKLOADS


DEMO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = DEMO_ROOT.parents[1]
WORK_ROOT = DEMO_ROOT / ".work"
PROJECT_ROOT = WORK_ROOT / TASK_ID
ORACLE_ROOT = WORK_ROOT / "oracle-reference"
VENV_ROOT = WORK_ROOT / "venv"
RESOURCE_RECORD = WORK_ROOT / "resources-before.json"
RUNNER_IMAGE = (
    "ghcr.io/ke7/helix-evo-runner-claude@"
    "sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba"
)
COPY_TO_PRIVATE = (
    "evaluator.py",
    "official_score.py",
    "workloads.py",
    "measurement.json",
    "pins.json",
    "LICENSES.md",
)


class DemoError(RuntimeError):
    """Actionable setup or inspection failure."""


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 300,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
        input=input_text,
    )
    if check and result.returncode != 0:
        stderr = result.stderr.strip().splitlines()
        tail = stderr[-1] if stderr else "no stderr"
        raise DemoError(f"command failed ({result.returncode}): {args[0]}: {tail}")
    return result


def resource_snapshot() -> dict[str, Any]:
    containers = run(
        ["docker", "ps", "--format", "{{.ID}}\t{{.Names}}\t{{.Image}}"],
        timeout=20,
    ).stdout.splitlines()
    benchmark_containers = [
        line
        for line in containers
        if any(
            marker in line.lower()
            for marker in (
                "helix-cmd",
                "helix-evo-runner",
                "formulacode",
                "benchmark",
                "networkx",
                "sweb",
            )
        )
    ]
    task_owned_containers = [
        line
        for line in containers
        if any(
            marker in line.lower()
            for marker in ("helix-cmd", "formulacode", "networkx", "runner-claude")
        )
    ]
    info = (
        run(
            [
                "docker",
                "info",
                "--format",
                "{{.NCPU}} {{.MemTotal}} {{.Architecture}}",
            ],
            timeout=20,
        )
        .stdout.strip()
        .split()
    )
    disk = shutil.disk_usage(REPO_ROOT)
    runner_present = (
        run(
            ["docker", "image", "inspect", RUNNER_IMAGE], timeout=20, check=False
        ).returncode
        == 0
    )
    return {
        "benchmark_containers": benchmark_containers,
        "all_container_ids": sorted(line.split("\t", 1)[0] for line in containers),
        "task_owned_container_ids": sorted(
            line.split("\t", 1)[0] for line in task_owned_containers
        ),
        "docker_cpus": int(info[0]),
        "docker_memory_bytes": int(info[1]),
        "docker_architecture": info[2],
        "disk_free_bytes": disk.free,
        "runner_image_present": runner_present,
    }


def assert_timing_window_clear() -> dict[str, Any]:
    snapshot = resource_snapshot()
    if snapshot["benchmark_containers"]:
        raise DemoError(
            "official timing window is busy: "
            + ", ".join(snapshot["benchmark_containers"])
        )
    if snapshot["disk_free_bytes"] < 8 * 1024**3:
        raise DemoError("fewer than 8 GiB remain; refusing benchmark setup")
    return snapshot


def _python() -> Path:
    suffix = "Scripts/python.exe" if os.name == "nt" else "bin/python"
    return VENV_ROOT / suffix


def _calibrate(repo: Path) -> dict[str, Any]:
    result = run(
        [
            str(_python()),
            str(DEMO_ROOT / "workloads.py"),
            "--repo",
            str(repo),
            "--measurement",
            str(DEMO_ROOT / "measurement.json"),
        ],
        cwd=REPO_ROOT,
        timeout=180,
    )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise DemoError("calibration output must be a JSON object")
    return cast(dict[str, Any], payload)


def _correctness(repo: Path) -> dict[str, Any]:
    config = json.loads((DEMO_ROOT / "measurement.json").read_text())
    result = run(
        [
            str(_python()),
            "-m",
            "pytest",
            "-q",
            *config["correctness_tests"],
        ],
        cwd=repo,
        timeout=int(config["correctness_timeout_seconds"]),
        check=False,
    )
    return {
        "returncode": result.returncode,
        "summary": result.stdout.strip().splitlines()[-1]
        if result.stdout.strip()
        else "",
    }


def _append_local_ignores(path: Path) -> None:
    additions = (
        "\n# HELIX FormulaCode local state (never commit/push)\n"
        ".helix/\nhelix_batch.json\n.helix_*\n.agent_*\n"
    )
    content = path.read_text() if path.exists() else ""
    if ".helix/" not in content.splitlines():
        path.write_text(content.rstrip() + additions)


def setup() -> dict[str, Any]:
    if (PROJECT_ROOT / "helix.toml").is_file():
        return {
            "status": "already_setup",
            "project": str(PROJECT_ROOT),
            **inspect(require_terminal=False),
        }
    if WORK_ROOT.exists():
        raise DemoError(
            f"partial setup exists at {WORK_ROOT}; inspect it or run cleanup first"
        )

    snapshot = assert_timing_window_clear()
    WORK_ROOT.mkdir(parents=True)
    RESOURCE_RECORD.write_text(json.dumps(snapshot, indent=2, sort_keys=True) + "\n")
    try:
        run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                REPOSITORY,
                str(PROJECT_ROOT),
            ],
            cwd=WORK_ROOT,
            timeout=180,
        )
        run(["git", "checkout", "--detach", BASE_COMMIT], cwd=PROJECT_ROOT)
        run(
            ["git", "worktree", "add", "--detach", str(ORACLE_ROOT), ORACLE_COMMIT],
            cwd=PROJECT_ROOT,
        )
        run(["uv", "venv", "--python", "3.12", str(VENV_ROOT)], cwd=REPO_ROOT)
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(_python()),
                "pytest==8.4.1",
            ],
            cwd=REPO_ROOT,
            timeout=180,
        )

        # Recheck immediately before timing; this is a single read, never a poll.
        timing_snapshot = assert_timing_window_clear()
        base_correctness = _correctness(PROJECT_ROOT)
        oracle_correctness = _correctness(ORACLE_ROOT)
        if base_correctness["returncode"] or oracle_correctness["returncode"]:
            raise DemoError(
                "pinned upstream correctness failed: "
                f"base={base_correctness}, oracle={oracle_correctness}"
            )
        nop = _calibrate(PROJECT_ROOT)
        oracle = _calibrate(ORACLE_ROOT)
        if nop["commit"] != BASE_COMMIT or oracle["commit"] != ORACLE_COMMIT:
            raise DemoError("calibration checkout did not match pinned commits")

        run(["git", "worktree", "remove", str(ORACLE_ROOT)], cwd=PROJECT_ROOT)
        private = PROJECT_ROOT / ".formulacode"
        private.mkdir()
        for name in COPY_TO_PRIVATE:
            shutil.copy2(DEMO_ROOT / name, private / name)
        baselines = {
            "schema_version": 1,
            "task_id": TASK_ID,
            "base_commit": BASE_COMMIT,
            "oracle_commit": ORACLE_COMMIT,
            "measurement_lock": str((WORK_ROOT / "measurement.lock").resolve()),
            "machine": {
                "system": platform.system(),
                "machine": platform.machine(),
                "python": nop["python"],
                **timing_snapshot,
            },
            "correctness": {"base": base_correctness, "oracle": oracle_correctness},
            "nop": nop["samples"],
            "oracle": oracle["samples"],
            "workloads": {
                split: {key: asdict(value) for key, value in items.items()}
                for split, items in WORKLOADS.items()
            },
        }
        (private / "baselines.json").write_text(
            json.dumps(baselines, indent=2, sort_keys=True) + "\n"
        )
        template = (DEMO_ROOT / "helix.toml.template").read_text()
        (PROJECT_ROOT / "helix.toml").write_text(
            template.replace("__PYTHON__", str(_python().resolve()))
        )
        _append_local_ignores(PROJECT_ROOT / ".gitignore")
        run(["git", "switch", "-c", "helix-formulacode-seed"], cwd=PROJECT_ROOT)
        run(
            [
                "git",
                "add",
                ".formulacode",
                "helix.toml",
                ".gitignore",
            ],
            cwd=PROJECT_ROOT,
        )
        run(
            [
                "git",
                "-c",
                "user.name=HELIX FormulaCode Demo",
                "-c",
                "user.email=helix-demo@example.invalid",
                "commit",
                "-m",
                "chore: configure pinned FormulaCode smoke task",
            ],
            cwd=PROJECT_ROOT,
        )
    except BaseException:
        # Preserve partial setup for diagnosis.  cleanup validates the exact
        # owned path before removing it.
        raise
    return {
        "status": "setup",
        "project": str(PROJECT_ROOT),
        **inspect(require_terminal=False),
    }


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    if state.get("generation") != 1:
        raise DemoError("state is not terminal at the pinned generation 1")
    raw_batches = state.get("proposal_batches")
    if not isinstance(raw_batches, list) or len(raw_batches) != 1:
        raise DemoError("expected exactly one terminal P=2,N=2 proposal batch")
    batch = raw_batches[0]
    if not isinstance(batch, dict):
        raise DemoError("proposal batch must be a JSON object")
    if batch.get("phase") != "complete" or (batch.get("p"), batch.get("n")) != (
        2,
        2,
    ):
        raise DemoError("proposal batch is not complete P=2,N=2")
    raw_tasks = batch.get("tasks")
    if not isinstance(raw_tasks, list) or len(raw_tasks) != 4:
        raise DemoError("terminal P=2,N=2 batch must contain exactly four tasks")

    tasks: list[dict[str, Any]] = []
    child_ids: list[str] = []
    task_charge_evaluations = 0
    terminal_statuses = {
        "skipped",
        "failed",
        "tampered",
        "rejected",
        "applied",
        "interrupted",
    }
    terminal_selections = {"not_applicable", "not_selected", "selected"}
    terminal_cleanups = {"not_required", "removed", "missing"}
    for index, raw_task in enumerate(raw_tasks):
        if not isinstance(raw_task, dict):
            raise DemoError(f"proposal task {index} must be a JSON object")
        task = cast(dict[str, Any], raw_task)
        if (
            task.get("batch_id") != batch.get("batch_id")
            or (task.get("p"), task.get("n")) != (2, 2)
            or task.get("task_index") != index
            or task.get("parent_group") != index // 2
            or task.get("mutation_index") != index % 2
        ):
            raise DemoError(f"proposal task {index} violates parent-major P=2,N=2")
        child_id = task.get("child_id")
        if not isinstance(child_id, str) or not child_id:
            raise DemoError(f"proposal task {index} has no reserved child ID")
        child_ids.append(child_id)
        if task.get("status") not in terminal_statuses:
            raise DemoError(f"proposal task {child_id} is not terminal")
        if task.get("selection") not in terminal_selections:
            raise DemoError(f"proposal task {child_id} has nonterminal selection")
        if task.get("cleanup") not in terminal_cleanups:
            raise DemoError(f"proposal task {child_id} has nonterminal cleanup")
        if task.get("budget_accounted") is not True:
            raise DemoError(f"proposal task {child_id} is not budget-accounted")
        raw_charge = task.get("budget_charge")
        if not isinstance(raw_charge, dict):
            raise DemoError(f"proposal task {child_id} has no budget charge")
        charge = raw_charge.get("evaluations")
        if not isinstance(charge, int) or isinstance(charge, bool) or charge < 0:
            raise DemoError(f"proposal task {child_id} has invalid evaluation charge")
        task_charge_evaluations += charge
        tasks.append(task)

    if len(set(child_ids)) != 4:
        raise DemoError("expected exactly four globally distinct proposal child IDs")

    budget_before = batch.get("budget_before_dispatch")
    budget_after = batch.get("budget_after_apply")
    if (
        not isinstance(budget_before, int)
        or isinstance(budget_before, bool)
        or budget_before < 0
        or not isinstance(budget_after, int)
        or isinstance(budget_after, bool)
        or budget_after < budget_before
    ):
        raise DemoError("proposal batch has invalid evaluation budget boundaries")
    batch_delta = budget_after - budget_before
    if task_charge_evaluations != batch_delta:
        raise DemoError(
            "proposal task charges do not conserve the complete batch budget delta"
        )

    budget = state.get("budget")
    if not isinstance(budget, dict):
        raise DemoError("global budget must be a JSON object")
    global_evaluations = budget.get("evaluations")
    if (
        not isinstance(global_evaluations, int)
        or isinstance(global_evaluations, bool)
        or global_evaluations < task_charge_evaluations
    ):
        raise DemoError("global evaluation budget is smaller than proposal charges")
    if budget_after != global_evaluations:
        raise DemoError("terminal batch budget does not match the global budget")
    nonproposal_evaluations = global_evaluations - task_charge_evaluations

    return {
        "generation": state.get("generation"),
        "frontier": state.get("frontier", []),
        "budget": budget,
        "mutation_counter": state.get("mutation_counter"),
        "batch_count": 1,
        "batches": [
            {
                "batch_id": batch.get("batch_id"),
                "phase": batch.get("phase"),
                "p": batch.get("p"),
                "n": batch.get("n"),
                "budget_before_dispatch": budget_before,
                "budget_after_apply": budget_after,
                "budget_delta_evaluations": batch_delta,
                "task_charge_evaluations": task_charge_evaluations,
                "budget_conserved": True,
            }
        ],
        "tasks": [
            {
                key: task.get(key)
                for key in (
                    "batch_id",
                    "p",
                    "n",
                    "task_index",
                    "child_id",
                    "parent_id",
                    "parent_group",
                    "mutation_index",
                    "status",
                    "selection",
                    "cleanup",
                    "budget_charge",
                    "budget_accounted",
                    "applied",
                    "score_delta",
                )
            }
            for task in tasks
        ],
        "accounting": {
            "global_evaluations": global_evaluations,
            "proposal_evaluations": task_charge_evaluations,
            "nonproposal_evaluations": nonproposal_evaluations,
            "complete_batch_delta_evaluations": batch_delta,
            "budget_conserved": True,
        },
        "distinct_child_ids": True,
        "parent_major_order": True,
        "terminal_p2n2": True,
    }


def fingerprint() -> str:
    helix_dir = PROJECT_ROOT / ".helix"
    if not helix_dir.is_dir():
        raise DemoError("no .helix state to fingerprint")
    digest = hashlib.sha256()
    for path in sorted(helix_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".log"):
            continue
        digest.update(path.relative_to(helix_dir).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def inspect(*, require_terminal: bool = True) -> dict[str, Any]:
    result: dict[str, Any] = {
        "project_exists": PROJECT_ROOT.is_dir(),
        "task_id": TASK_ID,
        "project": str(PROJECT_ROOT),
    }
    if not PROJECT_ROOT.is_dir():
        if require_terminal:
            raise DemoError("FormulaCode project is not set up")
        return result
    head = run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).stdout.strip()
    worktrees = run(["git", "worktree", "list", "--porcelain"], cwd=PROJECT_ROOT)
    result.update(
        {
            "seed_head": head,
            "worktree_paths": [
                line.removeprefix("worktree ")
                for line in worktrees.stdout.splitlines()
                if line.startswith("worktree ")
            ],
            "evaluation_files": sorted(
                path.name
                for path in (PROJECT_ROOT / ".helix" / "evaluations").glob("*.json")
            )
            if (PROJECT_ROOT / ".helix" / "evaluations").is_dir()
            else [],
        }
    )
    state_path = PROJECT_ROOT / ".helix" / "state.json"
    if state_path.is_file():
        state = json.loads(state_path.read_text())
        result["state"] = _state_summary(state)
        result["fingerprint"] = fingerprint()

        secrets = [
            os.environ[key]
            for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")
            if os.environ.get(key)
        ]
        matches = 0
        for path in (PROJECT_ROOT / ".helix").rglob("*"):
            if path.is_file():
                data = path.read_bytes()
                matches += sum(secret.encode() in data for secret in secrets)
        result["secret_scan"] = {"values_checked": len(secrets), "matches": matches}
    elif require_terminal:
        raise DemoError("no terminal HELIX state is available to inspect")
    return result


def cleanup(*, remove_runner_image: bool) -> dict[str, Any]:
    if WORK_ROOT.resolve() != (DEMO_ROOT / ".work").resolve():
        raise DemoError("refusing cleanup: owned work root changed")
    if not WORK_ROOT.exists():
        return {"status": "already_clean", **verify_clean()}
    before = (
        json.loads(RESOURCE_RECORD.read_text()) if RESOURCE_RECORD.is_file() else None
    )
    helix_dir = PROJECT_ROOT / ".helix"
    if helix_dir.exists():
        helix_cli = shutil.which("helix")
        if helix_cli is None:
            raise DemoError("helix executable is not available for cleanup")
        run(
            [helix_cli, "clean", "--dir", str(PROJECT_ROOT)],
            cwd=REPO_ROOT,
            timeout=180,
            input_text="y\n",
        )
    if PROJECT_ROOT.is_dir():
        worktree_text = run(
            ["git", "worktree", "list", "--porcelain"], cwd=PROJECT_ROOT
        ).stdout
        paths = [
            Path(line.removeprefix("worktree ")).resolve()
            for line in worktree_text.splitlines()
            if line.startswith("worktree ")
        ]
        unexpected = [path for path in paths if path != PROJECT_ROOT.resolve()]
        if unexpected:
            raise DemoError(f"HELIX worktrees remain: {unexpected}")
        branches = run(
            ["git", "branch", "--list", "helix/*"], cwd=PROJECT_ROOT
        ).stdout.strip()
        if branches:
            raise DemoError(f"HELIX branches remain: {branches}")
    after = resource_snapshot()
    if before is not None:
        new_containers = sorted(
            set(after["task_owned_container_ids"])
            - set(before.get("task_owned_container_ids", []))
        )
        if new_containers:
            raise DemoError(f"task-created containers remain: {new_containers}")
    if (
        remove_runner_image
        and before is not None
        and not before["runner_image_present"]
    ):
        run(["docker", "image", "rm", RUNNER_IMAGE], timeout=180, check=False)
    shutil.rmtree(WORK_ROOT)
    return {
        "status": "cleaned",
        "task_created_containers": 0,
        "runner_image_removed": bool(
            remove_runner_image
            and before is not None
            and not before["runner_image_present"]
        ),
        **verify_clean(),
    }


def verify_clean() -> dict[str, Any]:
    return {
        "work_root_absent": not WORK_ROOT.exists(),
        "project_absent": not PROJECT_ROOT.exists(),
        "task_files_tracked_only": run(
            [
                "git",
                "status",
                "--short",
                "--",
                "examples/formulacode",
                "tests/examples/test_formulacode.py",
            ],
            cwd=REPO_ROOT,
        ).stdout.splitlines(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("preflight")
    sub.add_parser("setup")
    sub.add_parser("inspect")
    sub.add_parser("fingerprint")
    cleanup_parser = sub.add_parser("cleanup")
    cleanup_parser.add_argument("--remove-runner-image", action="store_true")
    sub.add_parser("verify-clean")
    args = parser.parse_args()
    try:
        if args.command == "preflight":
            output: Any = assert_timing_window_clear()
        elif args.command == "setup":
            output = setup()
        elif args.command == "inspect":
            output = inspect()
        elif args.command == "fingerprint":
            output = {"fingerprint": fingerprint()}
        elif args.command == "cleanup":
            output = cleanup(remove_runner_image=args.remove_runner_image)
        else:
            output = verify_clean()
    except (DemoError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}))
        return 2
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
