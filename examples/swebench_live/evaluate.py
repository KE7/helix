"""Host adapter launching one isolated pinned official task container."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from pins import (
    CONTAINER_PREFIX,
    OFFICIAL_IMAGE,
    OFFICIAL_IMAGE_PLATFORM,
    PRIVATE_VOLUME,
    RESOURCE_LABEL,
    TASK_ID,
)


def redact_diagnostic(value: str) -> str:
    value = re.sub(r"(?i)(api[_-]?key|token|secret|password)=\S+", r"\1=[REDACTED]", value)
    value = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", value)
    return value[:500]


def candidate_id(cwd: Path) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", cwd.name)[:48] or "candidate"


def original_project_root(cwd: Path) -> Path:
    result = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return cwd
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = (cwd / common).resolve()
    root = common.parent if common.name == ".git" else cwd
    # In this repository the example is a subdirectory of HELIX's outer Git
    # checkout.  In a disposable demo run the project itself owns the common
    # .git directory.  Only the latter is the durable artifact root.
    return root if (root / "helix.toml").is_file() else cwd


def _docker(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["docker", *args],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def run_container(cwd: Path, gold_smoke: bool = False) -> dict[str, Any]:
    _docker("image", "inspect", OFFICIAL_IMAGE)
    _docker("volume", "inspect", PRIVATE_VOLUME)
    cid = candidate_id(cwd)
    name = f"{CONTAINER_PREFIX}{cid}-{uuid.uuid4().hex[:8]}".lower()
    args = [
        "create",
        "--platform",
        OFFICIAL_IMAGE_PLATFORM,
        "--network",
        "none",
        "--cpus",
        "4",
        "--memory",
        "4g",
        "--name",
        name,
        "--label",
        RESOURCE_LABEL,
        "--mount",
        f"type=volume,src={PRIVATE_VOLUME},dst=/private,readonly",
        OFFICIAL_IMAGE,
        "python3",
        "/tmp/official_runner.py",
    ]
    if gold_smoke:
        args.append("--gold-smoke")
    created = False
    try:
        _docker(*args)
        created = True
        _docker("cp", str(cwd / "official_runner.py"), f"{name}:/tmp/official_runner.py")
        _docker("cp", str(cwd / "coding_agent.py"), f"{name}:/tmp/coding_agent.py")
        started = time.monotonic()
        attached = _docker("start", "--attach", name, timeout=1_900)
        elapsed = round(time.monotonic() - started, 3)
        inspected = json.loads(_docker("inspect", name).stdout)[0]
        exit_code = int(inspected["State"]["ExitCode"])
        if exit_code != 0:
            raise RuntimeError(f"task container exited with code {exit_code}")
        raw_payload = json.loads(attached.stdout.strip().splitlines()[-1])
        if not isinstance(raw_payload, dict):
            raise ValueError("task runner did not return a JSON object")
        payload: dict[str, Any] = raw_payload
        payload["container"] = {
            "name": name,
            "platform": OFFICIAL_IMAGE_PLATFORM,
            "elapsed_seconds": elapsed,
        }
        return payload
    finally:
        if created:
            subprocess.run(
                ["docker", "rm", "--force", name],
                capture_output=True,
                text=True,
                timeout=30,
            )


def evaluation_result(cwd: Path, gold_smoke: bool = False) -> dict[str, Any]:
    cid = candidate_id(cwd)
    try:
        payload = run_container(cwd, gold_smoke=gold_smoke)
        official = payload["official_report"]
        resolved = bool(official.get("resolved", False))
        result = {
            "accuracy": 1.0 if resolved else 0.0,
            "instance_scores": {TASK_ID: 1.0 if resolved else 0.0},
            "candidate_id": cid,
            "score_source": "official_swebench_live_resolution",
            "official_report": official,
            "auxiliary_diagnostics": payload.get("diagnostics", {}),
            "container": payload.get("container", {}),
            "gold_smoke_only": gold_smoke,
        }
    except Exception as exc:
        result = {
            "accuracy": 0.0,
            "instance_scores": {TASK_ID: 0.0},
            "candidate_id": cid,
            "score_source": "official_swebench_live_resolution",
            "official_report": {"resolved": False, "error": type(exc).__name__},
            "auxiliary_diagnostics": {"error": redact_diagnostic(str(exc))},
            "gold_smoke_only": gold_smoke,
        }
    root = original_project_root(cwd)
    artifacts = root / "artifacts" / "evaluations"
    artifacts.mkdir(parents=True, exist_ok=True)
    artifact = artifacts / f"{cid}-{time.time_ns()}.json"
    artifact.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-smoke", action="store_true")
    args = parser.parse_args()
    print(json.dumps(evaluation_result(Path.cwd(), args.gold_smoke), sort_keys=True))


if __name__ == "__main__":
    main()
