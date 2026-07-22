"""Root-only task-container runner preserving SWE-bench-Live resolution rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


REPO = Path("/testbed")
PRIVATE_TASK = Path("/private/task.json")
CANDIDATE_SCRIPT = Path("/tmp/coding_agent.py")
CANDIDATE_REPO = Path("/tmp/candidate-repo")
ISSUE_FILE = Path("/tmp/problem_statement.txt")
CANDIDATE_UID = 65534
CANDIDATE_GID = 65534


def _command(command: str, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-lc", command],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _git(*args: str, timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        timeout=timeout,
    )


def _apply_patch(patch: str) -> bool:
    # The candidate runs as an unprivileged user with write access to /tmp and
    # can pre-create a fixed patch path as a symlink before it exits.  Writing
    # through it would follow the link and clobber an arbitrary file as root.
    # A fresh root-owned 0700 directory is unpredictable and untraversable, so
    # the candidate cannot plant a link inside it.
    staging = Path(tempfile.mkdtemp(prefix="helix-apply-"))
    try:
        os.chmod(staging, 0o700)
        patch_file = staging / "solution.patch"
        fd = os.open(
            patch_file,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
        )
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(patch)
        result = _git("apply", "--reject", "--whitespace=nowarn", str(patch_file))
        return result.returncode == 0
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _drop_privileges() -> None:
    os.setsid()
    os.setgid(CANDIDATE_GID)
    os.setuid(CANDIDATE_UID)


def _run_candidate(problem_statement: str, timeout: int) -> tuple[str, dict[str, Any]]:
    shutil.rmtree(CANDIDATE_REPO, ignore_errors=True)
    # A full clone of /testbed copies every ref in the official task image.
    # That image carries the upstream history *past* base_commit, including the
    # commit that fixes this very instance, so a full clone hands the candidate
    # the gold patch (`git log --all --grep=<issue>`).  A depth-1, tag-less
    # clone carries only base_commit, which is all the candidate legitimately
    # needs and still supports `git diff HEAD` for patch extraction.
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            "--depth",
            "1",
            "--no-tags",
            f"file://{REPO}",
            str(CANDIDATE_REPO),
        ],
        check=True,
        timeout=120,
    )
    # Fail closed if the candidate repository can still reach anything beyond
    # base_commit; scoring a run on a leaking repository is worse than failing.
    base_commit = _git("rev-parse", "HEAD").stdout.strip()
    cloned_head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=CANDIDATE_REPO,
        text=True,
        capture_output=True,
        timeout=60,
    ).stdout.strip()
    if cloned_head != base_commit:
        raise ValueError("candidate repository HEAD does not match base_commit")
    extra = subprocess.run(
        ["git", "log", "--all", "--oneline", "--not", "HEAD"],
        cwd=CANDIDATE_REPO,
        text=True,
        capture_output=True,
        timeout=60,
    ).stdout.strip()
    if extra:
        raise ValueError("candidate repository exposes commits beyond base_commit")
    ISSUE_FILE.write_text(problem_statement, encoding="utf-8")
    ISSUE_FILE.chmod(0o444)
    CANDIDATE_SCRIPT.chmod(0o555)
    subprocess.run(
        ["chown", "-R", f"{CANDIDATE_UID}:{CANDIDATE_GID}", str(CANDIDATE_REPO)],
        check=True,
    )
    env = {
        "HOME": "/tmp",
        "LANG": "C.UTF-8",
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
    }
    started = time.monotonic()
    proc = subprocess.Popen(
        [
            "python3",
            str(CANDIDATE_SCRIPT),
            "--repo",
            str(CANDIDATE_REPO),
            "--issue-file",
            str(ISSUE_FILE),
        ],
        cwd=CANDIDATE_REPO,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=_drop_privileges,
    )
    timed_out = False
    try:
        returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(proc.pid, signal.SIGKILL)
        returncode = proc.wait(timeout=10)
    subprocess.run(["pkill", "-KILL", "-u", str(CANDIDATE_UID)], check=False)
    patch_result = subprocess.run(
        ["git", "diff", "--binary", "--text", "HEAD"],
        cwd=CANDIDATE_REPO,
        text=True,
        capture_output=True,
        timeout=60,
        env=env,
        user=CANDIDATE_UID,
        group=CANDIDATE_GID,
    )
    patch = patch_result.stdout if patch_result.returncode == 0 else ""
    shutil.rmtree(CANDIDATE_REPO, ignore_errors=True)
    return patch, {
        "candidate_exit_code": returncode,
        "candidate_timed_out": timed_out,
        "candidate_seconds": round(time.monotonic() - started, 3),
    }


def official_resolution(status: dict[str, str], task: dict[str, Any]) -> dict[str, Any]:
    """Apply the upstream evaluation.py PASS_TO_PASS/FAIL_TO_PASS rule."""

    passed = {name for name, value in status.items() if "pass" in value.lower()}
    failed = {name for name, value in status.items() if "fail" in value.lower()}
    p2p = set(task["PASS_TO_PASS"])
    f2p = set(task["FAIL_TO_PASS"])
    report = {
        "instance_id": task["instance_id"],
        "resolved": False,
        "PASS_TO_PASS": {
            "success": sorted(passed & p2p),
            "failure": sorted(failed & p2p),
        },
        "FAIL_TO_PASS": {
            "success": sorted(passed & f2p),
            "failure": sorted(failed & f2p),
        },
    }
    all_f2p = f2p.issubset(set(report["FAIL_TO_PASS"]["success"])) or (
        len(report["FAIL_TO_PASS"]["success"]) == len(f2p)
    )
    report["resolved"] = (
        not report["PASS_TO_PASS"]["failure"]
        and not report["FAIL_TO_PASS"]["failure"]
        and all_f2p
    )
    return report


def _parse_status(parser_source: str, log: str) -> dict[str, str]:
    namespace: dict[str, Any] = {}
    exec(parser_source, namespace)
    parser = namespace.get("parser")
    if not callable(parser):
        raise ValueError("official row parser is missing parser(log)")
    status = parser(log)
    if not isinstance(status, dict):
        raise ValueError("official row parser did not return a dict")
    return {str(key): str(value) for key, value in status.items()}


def run(gold_smoke: bool, agent_timeout: int) -> dict[str, Any]:
    envelope = json.loads(PRIVATE_TASK.read_text(encoding="utf-8"))
    task = envelope["task"]
    if _git("rev-parse", "HEAD").stdout.strip() != task["base_commit"]:
        raise ValueError("official image base commit mismatch")
    diagnostics: dict[str, Any] = {"gold_smoke": gold_smoke}
    if gold_smoke:
        solution_patch = task["patch"]
    else:
        solution_patch, candidate_info = _run_candidate(
            task["problem_statement"], agent_timeout
        )
        diagnostics.update(candidate_info)
    diagnostics["patch_bytes"] = len(solution_patch.encode())
    diagnostics["patch_sha256"] = hashlib.sha256(solution_patch.encode()).hexdigest()
    if not solution_patch.strip():
        return {
            "official_report": {
                "instance_id": task["instance_id"],
                "resolved": False,
                "empty_patch": True,
            },
            "diagnostics": diagnostics,
        }
    diagnostics["test_patch_applied"] = _apply_patch(task["test_patch"])
    diagnostics["solution_patch_applied"] = _apply_patch(solution_patch)
    rebuild = _command(" ; ".join(task.get("rebuild_cmds", [])), timeout=900)
    diagnostics["rebuild_exit_code"] = rebuild.returncode
    test_result = _command(" ; ".join(task.get("test_cmds", [])), timeout=900)
    diagnostics["test_exit_code"] = test_result.returncode
    printed = _command(" ; ".join(task.get("print_cmds", [])), timeout=60)
    log = printed.stdout
    diagnostics["log_bytes"] = len(log.encode())
    diagnostics["log_sha256"] = hashlib.sha256(log.encode()).hexdigest()
    status = _parse_status(task["log_parser"], log)
    diagnostics["parsed_tests"] = len(status)
    return {
        "official_report": official_resolution(status, task),
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-smoke", action="store_true")
    parser.add_argument("--agent-timeout", type=int, default=120)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        result = run(args.gold_smoke, args.agent_timeout)
    except subprocess.TimeoutExpired as exc:
        result = {
            "official_report": {"resolved": False, "error": "command_timeout"},
            "diagnostics": {"timeout_seconds": exc.timeout, "gold_smoke": args.gold_smoke},
        }
    except Exception as exc:
        result = {
            "official_report": {"resolved": False, "error": type(exc).__name__},
            "diagnostics": {"gold_smoke": args.gold_smoke},
        }
    result["diagnostics"]["total_seconds"] = round(time.monotonic() - started, 3)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
