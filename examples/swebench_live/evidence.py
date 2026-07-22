"""Capture reproducible, secret-safe evidence for the pinned SWE demo.

This helper intentionally performs no cleanup.  It records global snapshots
for attribution, but treats a resource as run-owned only when a container mount
points into the disposable run or its name contains a reserved proposal ID.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

from pins import CONTAINER_PREFIX, OFFICIAL_IMAGE


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER_IMAGE = (
    "ghcr.io/ke7/helix-evo-runner-claude@"
    "sha256:6be6fef217bd083c462abbe2388c6a33a896a34812522de15516b59837293cba"
)
SECRET_NAMES = (
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
)


class EvidenceError(RuntimeError):
    """Raised when requested evidence cannot be captured unambiguously."""


def run(
    args: list[str],
    *,
    cwd: Path | None = None,
    timeout: int = 120,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        args,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and result.returncode != 0:
        detail = result.stderr.strip().splitlines()
        raise EvidenceError(
            f"command failed ({result.returncode}): {args[0]}: "
            f"{detail[-1] if detail else 'no stderr'}"
        )
    return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def json_lines(args: list[str]) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in run(args).stdout.splitlines()
        if line.strip()
    ]


def image_record(reference: str) -> dict[str, Any]:
    result = run(
        ["docker", "image", "inspect", reference, "--format", "{{json .}}"],
        check=False,
    )
    if result.returncode != 0:
        return {"reference": reference, "present": False}
    raw = json.loads(result.stdout)
    return {
        "reference": reference,
        "present": True,
        "id": raw.get("Id"),
        "repo_digests": raw.get("RepoDigests") or [],
        "architecture": raw.get("Architecture"),
        "os": raw.get("Os"),
    }


def resource_snapshot() -> dict[str, Any]:
    containers = json_lines(
        ["docker", "ps", "-a", "--no-trunc", "--format", "{{json .}}"]
    )
    volumes = json_lines(["docker", "volume", "ls", "--format", "{{json .}}"])
    images = json_lines(
        ["docker", "image", "ls", "-a", "--no-trunc", "--format", "{{json .}}"]
    )
    networks = json_lines(["docker", "network", "ls", "--format", "{{json .}}"])
    worktrees = [
        line.removeprefix("worktree ")
        for line in run(
            ["git", "worktree", "list", "--porcelain"], cwd=REPO_ROOT
        ).stdout.splitlines()
        if line.startswith("worktree ")
    ]
    return {
        "captured_at_ns": time.time_ns(),
        "counts": {
            "containers": len(containers),
            "volumes": len(volumes),
            "images": len(images),
            "networks": len(networks),
            "worktrees": len(worktrees),
        },
        "containers": containers,
        "volumes": volumes,
        "images": images,
        "networks": networks,
        "worktrees": worktrees,
        "stash": run(["git", "stash", "list"], cwd=REPO_ROOT).stdout.splitlines(),
        "pinned_images": {
            "runner": image_record(RUNNER_IMAGE),
            "task": image_record(OFFICIAL_IMAGE),
        },
    }


def baseline(output: Path) -> dict[str, Any]:
    import helix

    head = run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).stdout.strip()
    status = run(["git", "status", "--short", "--branch"], cwd=REPO_ROOT)
    helix_path = shutil.which("helix")
    if not helix_path:
        raise EvidenceError("helix console script is not available")
    version = run([helix_path, "--version"]).stdout.strip()
    payload = {
        "repository": {
            "root": str(REPO_ROOT),
            "head": head,
            "accepted_central_is_ancestor": run(
                [
                    "git",
                    "merge-base",
                    "--is-ancestor",
                    "f971e7aa74fbbaf0b6962d569d1264f0418ef433",
                    "HEAD",
                ],
                cwd=REPO_ROOT,
                check=False,
            ).returncode
            == 0,
            "status": status.stdout.splitlines(),
        },
        "runtime": {
            "helix_realpath": str(Path(helix.__file__).resolve()),
            "helix_version": helix.__version__,
            "metadata_version": importlib.metadata.version("helix-evo"),
            "console_script": str(Path(helix_path).resolve()),
            "console_version": version,
        },
        "credential_lengths": {
            name: {"present": bool(os.environ.get(name)), "length": len(os.environ.get(name, ""))}
            for name in SECRET_NAMES
        },
        "resources": resource_snapshot(),
    }
    write_json(output, payload)
    return payload


def candidate_ids(run_dir: Path) -> list[str]:
    state = run_dir / ".helix" / "state.json"
    if not state.is_file():
        return []
    raw = json.loads(state.read_text(encoding="utf-8"))
    ids: list[str] = []
    for batch in raw.get("proposal_batches", []):
        for task in batch.get("tasks", []):
            child_id = task.get("child_id")
            if isinstance(child_id, str) and child_id not in ids:
                ids.append(child_id)
    return ids


def container_records() -> list[dict[str, Any]]:
    ids = run(["docker", "ps", "-aq", "--no-trunc"]).stdout.splitlines()
    if not ids:
        return []
    raw = json.loads(run(["docker", "inspect", *ids]).stdout)
    return [
        {
            "id": item.get("Id"),
            "name": str(item.get("Name") or "").removeprefix("/"),
            "status": (item.get("State") or {}).get("Status"),
            "image_id": item.get("Image"),
            "image_reference": (item.get("Config") or {}).get("Image"),
            "labels": (item.get("Config") or {}).get("Labels") or {},
            "mounts": [
                {
                    "type": mount.get("Type"),
                    "name": mount.get("Name"),
                    "source": mount.get("Source"),
                    "destination": mount.get("Destination"),
                    "rw": mount.get("RW"),
                }
                for mount in item.get("Mounts") or []
            ],
        }
        for item in raw
    ]


def _path_is_within(raw: object, parent: Path) -> bool:
    if not isinstance(raw, str) or not raw:
        return False
    try:
        Path(raw).resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def live_capture(run_dir: Path, output: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    ids = candidate_ids(run_dir)
    worktree_paths: list[str] = []
    worktree_result = run(
        ["git", "worktree", "list", "--porcelain"], cwd=run_dir, check=False
    )
    if worktree_result.returncode == 0:
        worktree_paths = [
            line.removeprefix("worktree ")
            for line in worktree_result.stdout.splitlines()
            if line.startswith("worktree ")
        ]

    all_containers = container_records()
    owned: list[dict[str, Any]] = []
    for container in all_containers:
        name = str(container["name"])
        mounted_from_run = any(
            _path_is_within(mount.get("source"), run_dir)
            for mount in container["mounts"]
        )
        candidate_named = name.startswith(CONTAINER_PREFIX) and any(
            child_id.lower() in name.lower() for child_id in ids
        )
        if mounted_from_run or candidate_named:
            owned.append(container)

    snapshot_candidates: dict[str, Any] = {}
    for child_id in ids:
        expected = run_dir / ".helix" / "worktrees" / child_id
        observed = [
            path
            for path in worktree_paths
            if Path(path).name == child_id or Path(path).resolve() == expected.resolve()
        ]
        paths = observed or ([str(expected)] if expected.exists() else [])
        transcript_roots = [
            str(Path(path).parent / f"{Path(path).name}.helix-transcripts")
            for path in paths
        ]
        matching_containers = [
            container
            for container in owned
            if child_id.lower() in str(container["name"]).lower()
            or any(
                _path_is_within(mount.get("source"), expected)
                for mount in container["mounts"]
            )
        ]
        snapshot_candidates[child_id] = {
            "worktrees": paths,
            "worktree_exists": any(Path(path).is_dir() for path in paths),
            "transcript_roots": transcript_roots,
            "transcript_root_exists": any(
                Path(path).is_dir() for path in transcript_roots
            ),
            "containers": matching_containers,
        }

    snapshot = {
        "captured_at_ns": time.time_ns(),
        "candidate_ids": ids,
        "git_worktrees": worktree_paths,
        "candidates": snapshot_candidates,
        "owned_containers": owned,
        "foreign_containers_observed": [
            {key: container[key] for key in ("id", "name", "status", "image_reference")}
            for container in all_containers
            if container not in owned
        ],
    }
    payload: dict[str, Any] = {"run_dir": str(run_dir), "snapshots": [], "candidates": {}}
    if output.is_file():
        payload = json.loads(output.read_text(encoding="utf-8"))
    payload.setdefault("snapshots", []).append(snapshot)
    aggregate = payload.setdefault("candidates", {})
    for child_id, record in snapshot_candidates.items():
        current = aggregate.setdefault(
            child_id,
            {"worktrees": [], "transcript_roots": [], "containers": []},
        )
        for key in ("worktrees", "transcript_roots"):
            current[key] = sorted(set(current[key]) | set(record[key]))
        by_name = {item["name"]: item for item in current["containers"]}
        by_name.update({item["name"]: item for item in record["containers"]})
        current["containers"] = [by_name[name] for name in sorted(by_name)]
    write_json(output, payload)
    return payload


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode()


def _tree_digest(files: dict[str, dict[str, Any]], prefix: str) -> str:
    selected = {
        path: record
        for path, record in files.items()
        if path == prefix or path.startswith(f"{prefix}/")
    }
    return _sha256(_canonical(selected))


def manifest(run_dir: Path, output: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    files: dict[str, dict[str, Any]] = {}
    for root_name in (".helix", "artifacts"):
        root = run_dir / root_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix in {".log", ".lock", ".tmp"}:
                continue
            data = path.read_bytes()
            files[path.relative_to(run_dir).as_posix()] = {
                "bytes": len(data),
                "sha256": _sha256(data),
            }
    state_path = run_dir / ".helix" / "state.json"
    if not state_path.is_file():
        raise EvidenceError("state.json is missing")
    state_bytes = state_path.read_bytes()
    state = json.loads(state_bytes)
    digests = {
        "state_bytes_sha256": _sha256(state_bytes),
        "proposal_ledger_sha256": _sha256(_canonical(state.get("proposal_batches"))),
        "budget_sha256": _sha256(_canonical(state.get("budget"))),
        "frontier_sha256": _sha256(_canonical(state.get("frontier"))),
        "active_frontier_sha256": _sha256(_canonical(state.get("active_frontier"))),
        "attempts_tree_sha256": _tree_digest(files, ".helix/attempts"),
        "helix_evaluations_tree_sha256": _tree_digest(files, ".helix/evaluations"),
        "artifact_evaluations_tree_sha256": _tree_digest(
            files, "artifacts/evaluations"
        ),
        "lineage_tree_sha256": _tree_digest(files, ".helix/lineage.json"),
        "trajectory_tree_sha256": _tree_digest(files, ".helix/trajectories"),
        "substantive_tree_sha256": _sha256(_canonical(files)),
    }
    substantive_keys = len(files) + len(digests)
    if substantive_keys < 13:
        raise EvidenceError(
            f"manifest is vacuous: only {substantive_keys} substantive keys"
        )
    payload = {
        "schema_version": 1,
        "files": files,
        "digests": digests,
        "file_count": len(files),
        "independent_digest_count": len(digests),
        "substantive_key_count": substantive_keys,
        "non_vacuous": True,
    }
    write_json(output, payload)
    return payload


def run_helix(
    run_dir: Path,
    mode: str,
    status_file: Path,
    transcript: Path,
) -> int:
    executable = shutil.which("helix")
    if not executable:
        raise EvidenceError("helix console script is not available")
    executable = str(Path(executable).resolve())
    command = [executable, mode, "--dir", str(run_dir.resolve())]
    status_file.parent.mkdir(parents=True, exist_ok=True)
    transcript.parent.mkdir(parents=True, exist_ok=True)
    started_at_ns = time.time_ns()
    with transcript.open("wb") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        assert process.stdout is not None
        while chunk := process.stdout.read(65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            handle.write(chunk)
            handle.flush()
        returncode = process.wait()
    write_json(
        status_file,
        {
            "command": command,
            "console_script_realpath": executable,
            "started_at_ns": started_at_ns,
            "finished_at_ns": time.time_ns(),
            "exit_status": returncode,
        },
    )
    return returncode


def export_run(run_dir: Path, destination: Path) -> dict[str, Any]:
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for name in (".helix", "artifacts"):
        source = run_dir / name
        target = destination / name
        if target.exists():
            raise EvidenceError(f"refusing to overwrite exported evidence: {target}")
        if source.is_dir():
            shutil.copytree(source, target, symlinks=True)
            copied.append(name)
    payload = {"run_dir": str(run_dir.resolve()), "destination": str(destination.resolve()), "copied": copied}
    write_json(destination / "export.json", payload)
    return payload


def secret_scan(roots: list[Path], output: Path) -> dict[str, Any]:
    values = [(name, os.environ.get(name, "")) for name in SECRET_NAMES]
    values = [(name, value) for name, value in values if value]
    reports: list[dict[str, Any]] = []
    for name, value in values:
        needle = value.encode()
        hits = 0
        files_with_hits = 0
        files_scanned = 0
        for root in roots:
            paths = [root] if root.is_file() else root.rglob("*")
            for path in paths:
                if not path.is_file() or path.resolve() == output.resolve():
                    continue
                files_scanned += 1
                data = path.read_bytes()
                count = data.count(needle)
                hits += count
                files_with_hits += int(count > 0)
        reports.append(
            {
                "credential_name": name,
                "credential_length": len(value),
                "files_scanned": files_scanned,
                "files_with_hits": files_with_hits,
                "hit_count": hits,
            }
        )
    payload = {
        "roots": [str(root.resolve()) for root in roots],
        "credential_values_checked": len(values),
        "reports": reports,
        "total_hits": sum(report["hit_count"] for report in reports),
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    baseline_parser = sub.add_parser("baseline")
    baseline_parser.add_argument("--output", type=Path, required=True)
    live_parser = sub.add_parser("live")
    live_parser.add_argument("--run-dir", type=Path, required=True)
    live_parser.add_argument("--output", type=Path, required=True)
    manifest_parser = sub.add_parser("manifest")
    manifest_parser.add_argument("--run-dir", type=Path, required=True)
    manifest_parser.add_argument("--output", type=Path, required=True)
    run_parser = sub.add_parser("run-helix")
    run_parser.add_argument("mode", choices=("evolve", "resume"))
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--status-file", type=Path, required=True)
    run_parser.add_argument("--transcript", type=Path, required=True)
    export_parser = sub.add_parser("export")
    export_parser.add_argument("--run-dir", type=Path, required=True)
    export_parser.add_argument("--destination", type=Path, required=True)
    scan_parser = sub.add_parser("secret-scan")
    scan_parser.add_argument("--root", type=Path, action="append", required=True)
    scan_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "baseline":
            result = baseline(args.output)
        elif args.command == "live":
            result = live_capture(args.run_dir, args.output)
        elif args.command == "manifest":
            result = manifest(args.run_dir, args.output)
        elif args.command == "run-helix":
            return run_helix(
                args.run_dir, args.mode, args.status_file, args.transcript
            )
        elif args.command == "export":
            result = export_run(args.run_dir, args.destination)
        else:
            result = secret_scan(args.root, args.output)
    except (EvidenceError, OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": type(exc).__name__, "detail": str(exc)}))
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
