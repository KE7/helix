"""Private circle-packing evaluator service (the HELIX evaluator sidecar).

Runs as root inside the sidecar container, on HELIX's internal evaluator
network.  Mutation agents run on a different network and cannot reach it.

For every ``POST /evaluate`` from an evaluator-runner container it:

1. rejects requests that originate inside this container (so candidate code
   it is executing cannot request extra evaluations) and requests that arrive
   while another evaluation is running;
2. unpacks the candidate tree the runner sent into a fresh directory owned by
   the unprivileged ``candidate`` user;
3. runs ``solve.pack_circles(26)`` through ``candidate_harness.py`` as that
   user, with a wall-clock limit, then kills every process the user owns;
4. validates and scores the returned packing with ``packing_rules`` (strict
   geometry, certified score; invalid packings, exceptions and timeouts
   score 0);
5. appends one line to the private ledger ``evaluations.jsonl`` and keeps the
   candidate tree it scored under ``candidate_archives/``: every scoring
   execution is recorded, in order, whether or not it succeeded;
6. returns ``[score, side_info]`` for the runner to print as ``HELIX_RESULT``.

``CP_MAX_EXECUTIONS`` (0 = unlimited) is a backstop on the number of scoring
executions: once the ledger holds that many, further requests are refused
with HTTP 429 and logged as rejections, so no run can execute the scoring
function more often than that, whatever the optimizer's own accounting does.

The ledger sits below a root-only directory; the candidate user can read
neither the ledger nor this service's code.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import pwd
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import threading
import time
import uuid
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import packing_rules  # noqa: E402

PROTOCOL = 1
LISTEN_PORT = int(os.environ.get("CP_EVAL_PORT", "8080"))
# The ledger is mounted below /srv/cpeval, a root-only (0700) directory of the
# image: bind mounts on Docker Desktop do not enforce Linux permissions, so
# the parent directory is what keeps candidate code out.
LEDGER_DIR = Path(os.environ.get("CP_LEDGER_DIR", "/srv/cpeval/ledger"))
WORK_ROOT = Path(os.environ.get("CP_WORK_ROOT", "/work"))
HARNESS = os.environ.get("CP_HARNESS", "/opt/cpeval/harness/candidate_harness.py")
CANDIDATE_USER = os.environ.get("CP_CANDIDATE_USER", "candidate")
CANDIDATE_TIMEOUT_SECONDS = float(os.environ.get("CP_CANDIDATE_TIMEOUT", "600"))
MAX_EXECUTIONS = int(os.environ.get("CP_MAX_EXECUTIONS", "0"))

MAX_REQUEST_BYTES = 48 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_UNPACKED_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 5000
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
STREAM_TAIL = 4000
OUTPUT_NAME = ".cp_eval_output.json"

_evaluation_lock = threading.Lock()
_ledger_lock = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    """Replace non-finite floats so every record is strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class Ledger:
    """Append-only JSONL records; ``seq`` counts scoring executions."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        self.evaluations = directory / "evaluations.jsonl"
        self.archives = directory / "candidate_archives"
        self.archives.mkdir(mode=0o700, exist_ok=True)
        self.rejections = directory / "rejected_requests.jsonl"
        self.service = directory / "service_events.jsonl"
        self.seq = 0
        if self.evaluations.exists():
            with self.evaluations.open("r", encoding="utf-8") as handle:
                self.seq = sum(1 for line in handle if line.strip())

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(_json_safe(record), sort_keys=True, allow_nan=False)
        with _ledger_lock:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)

    def next_seq(self) -> int:
        with _ledger_lock:
            self.seq += 1
            return self.seq

    def record_evaluation(self, record: dict[str, Any]) -> None:
        self._append(self.evaluations, record)

    def keep_archive(self, seq: int, archive: bytes) -> str:
        """Store the scored candidate tree; return its path relative to the ledger."""
        name = f"{seq:05d}-{hashlib.sha256(archive).hexdigest()[:16]}.tar.gz"
        fd = os.open(self.archives / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(archive)
            handle.flush()
            os.fsync(handle.fileno())
        return f"candidate_archives/{name}"

    def exhausted(self) -> bool:
        with _ledger_lock:
            return MAX_EXECUTIONS > 0 and self.seq >= MAX_EXECUTIONS

    def record_rejection(self, record: dict[str, Any]) -> None:
        self._append(self.rejections, record)

    def record_service_event(self, record: dict[str, Any]) -> None:
        self._append(self.service, record)


# ---------------------------------------------------------------------------
# Peer checks
# ---------------------------------------------------------------------------


def local_addresses() -> set[str]:
    """Addresses that belong to this container's own network namespace."""
    addresses = {"127.0.0.1", "::1", "0.0.0.0", "::"}
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None):
            addresses.add(str(info[4][0]))
    except OSError:
        pass
    try:
        lines = Path("/proc/net/fib_trie").read_text().splitlines()
        for index, line in enumerate(lines):
            if "/32 host LOCAL" in line and index > 0:
                candidate = lines[index - 1].strip().split()[-1]
                addresses.add(candidate)
    except OSError:
        pass
    return addresses


def is_local_peer(peer: str, local: set[str]) -> bool:
    if peer.startswith("::ffff:"):
        peer = peer[len("::ffff:") :]
    return peer in local or peer.startswith("127.")


# ---------------------------------------------------------------------------
# Candidate tree handling
# ---------------------------------------------------------------------------


class ArchiveError(ValueError):
    pass


def safe_extract(archive: bytes, destination: Path) -> list[dict[str, Any]]:
    """Extract regular files and directories only; refuse anything else.

    Returns the list of extracted files with sizes.  Absolute paths, ``..``
    components, links, devices and oversized content raise ``ArchiveError``
    or are skipped (links and special files), never followed.
    """
    files: list[dict[str, Any]] = []
    total = 0
    try:
        tar = tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise ArchiveError(f"unreadable archive: {exc}") from None
    with tar:
        members = tar.getmembers()
        if len(members) > MAX_MEMBERS:
            raise ArchiveError(f"archive has {len(members)} members")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.name in ("", "."):
                raise ArchiveError(f"unsafe path in archive: {member.name!r}")
            target = destination.joinpath(*path.parts)
            if member.isdir():
                target.mkdir(mode=0o700, parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue
            total += member.size
            if total > MAX_UNPACKED_BYTES:
                raise ArchiveError("unpacked candidate tree is too large")
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            source = tar.extractfile(member)
            if source is None:
                continue
            fd = os.open(
                target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
            )
            with os.fdopen(fd, "wb") as handle:
                shutil.copyfileobj(source, handle)
            files.append({"path": str(path), "bytes": member.size})
    return files


def chown_tree(root: Path, uid: int, gid: int) -> None:
    os.lchown(root, uid, gid)
    for directory, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            os.lchown(os.path.join(directory, name), uid, gid)


def reap_children() -> None:
    """Collect exited children (this service is PID 1, so orphans end up here)."""
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return


def kill_user_processes(uid: int, rounds: int = 50) -> int:
    """SIGKILL every process whose real uid is ``uid``; return how many."""
    killed = 0
    for _ in range(rounds):
        found = False
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                status = (entry / "status").read_text()
            except OSError:
                continue
            for line in status.splitlines():
                if line.startswith("Uid:"):
                    if int(line.split()[1]) == uid:
                        found = True
                        try:
                            os.kill(int(entry.name), signal.SIGKILL)
                            killed += 1
                        except ProcessLookupError:
                            pass
                    break
        if not found:
            return killed
        time.sleep(0.05)
    return killed


def _tail(path: Path) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - STREAM_TAIL))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _read_candidate_output(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        return None, "the candidate produced no output (crashed, killed or timed out)"
    with os.fdopen(fd, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            return None, "candidate output is not a regular file"
        data = handle.read(MAX_OUTPUT_BYTES + 1)
    if len(data) > MAX_OUTPUT_BYTES:
        return None, "candidate output is too large"

    def _constant(name: str) -> float:
        return {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}[name]

    try:
        payload = json.loads(data.decode("utf-8"), parse_constant=_constant)
    except (UnicodeDecodeError, ValueError, RecursionError):
        return None, "candidate output is not valid JSON"
    if not isinstance(payload, dict):
        return None, "candidate output has the wrong shape"
    return payload, None


def run_candidate(workdir: Path, uid: int, gid: int, logs: Path) -> dict[str, Any]:
    """Run the harness as the candidate user and collect its result."""
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "MPLBACKEND": "Agg",
    }
    stdout_path = logs / "stdout.txt"
    stderr_path = logs / "stderr.txt"
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        proc = subprocess.Popen(
            [sys.executable, HARNESS, str(workdir / OUTPUT_NAME)],
            cwd=workdir,
            env=env,
            user=uid,
            group=gid,
            extra_groups=[],
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=err,
            start_new_session=True,
            umask=0o077,
        )
        try:
            exit_code: int | None = proc.wait(timeout=CANDIDATE_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:
            timed_out = True
            exit_code = None
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            killed = kill_user_processes(uid)
            proc.wait()
            reap_children()
    wall = time.monotonic() - started
    payload, output_error = (None, "timed out") if timed_out else _read_candidate_output(
        workdir / OUTPUT_NAME
    )
    return {
        "timed_out": timed_out,
        "exit_code": exit_code,
        "wall_seconds": wall,
        "processes_killed_after_run": killed,
        "payload": payload,
        "output_error": output_error,
        "stdout_tail": _tail(stdout_path),
        "stderr_tail": _tail(stderr_path),
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _rules() -> dict[str, Any]:
    return {
        "n": packing_rules.N_CIRCLES,
        "tolerance": packing_rules.TOLERANCE,
        "score": "sum of radii after subtracting the largest residual violation",
        "time_limit_seconds": CANDIDATE_TIMEOUT_SECONDS,
        "max_executions": MAX_EXECUTIONS,
    }


def build_side_info(
    verdict: dict[str, Any], run: dict[str, Any], circles: Any
) -> dict[str, Any]:
    score = float(verdict["score"])
    if verdict["valid"]:
        shrink = float(verdict["details"].get("certification_shrink", 0.0))
        feedback = "Valid packing of 26 circles."
        if shrink > 0.0:
            feedback += (
                f" Its largest residual violation, {shrink:.3e}, was subtracted "
                "from every radius before summing."
            )
    else:
        feedback = "Invalid packing, score 0: " + "; ".join(verdict["errors"])
    side_info: dict[str, Any] = {
        "scores": {"sum_radii": score},
        "valid": bool(verdict["valid"]),
        "feedback": feedback,
        "validation": verdict["details"],
        "candidate_run": {
            "timed_out": run.get("timed_out", False),
            "exit_code": run.get("exit_code"),
            "wall_seconds": round(float(run.get("wall_seconds", 0.0)), 3),
            "time_limit_seconds": CANDIDATE_TIMEOUT_SECONDS,
        },
    }
    if isinstance(circles, list) and len(circles) == packing_rules.N_CIRCLES:
        # Full-precision coordinates, as optimize_anything's side info carries
        # the circles; a JSON string keeps the prompt rendering compact.
        side_info["circles_json"] = json.dumps(_json_safe(circles), allow_nan=False)
    payload = run.get("payload") or {}
    if payload.get("status") == "error":
        side_info["error"] = payload.get("error")
        side_info["traceback_tail"] = payload.get("traceback_tail")
    if run.get("output_error"):
        side_info["error"] = side_info.get("error") or run["output_error"]
    if run.get("stdout_tail"):
        side_info["stdout_tail"] = run["stdout_tail"]
    if run.get("stderr_tail"):
        side_info["stderr_tail"] = run["stderr_tail"]
    return _json_safe(side_info)


def evaluate_request(body: dict[str, Any], ledger: Ledger, peer: str) -> dict[str, Any]:
    received = _utc_now()
    evaluation_id = uuid.uuid4().hex
    ids = body.get("ids")
    if (
        body.get("protocol") != PROTOCOL
        or not isinstance(ids, list)
        or len(ids) != 1
        or not all(isinstance(i, str) for i in ids)
    ):
        raise ValueError("request must use protocol 1 and carry exactly one example id")
    split = str(body.get("split", ""))

    seq = ledger.next_seq()
    record: dict[str, Any] = {
        "seq": seq,
        "evaluation_id": evaluation_id,
        "received_utc": received,
        "peer": peer,
        "split": split,
        "ids": ids,
        "archive_sha256": body.get("archive_sha256"),
        "files_sha256": body.get("files_sha256"),
    }
    candidate = pwd.getpwnam(CANDIDATE_USER)
    workdir = WORK_ROOT / evaluation_id
    logs = LEDGER_DIR / "candidate_logs" / f"{seq:05d}-{evaluation_id}"
    logs.mkdir(parents=True, exist_ok=True)
    run: dict[str, Any] = {}
    circles: Any = None
    try:
        archive_b64 = body.get("archive_b64")
        if body.get("client_error"):
            raise ArchiveError(str(body["client_error"])[:500])
        if not isinstance(archive_b64, str):
            raise ArchiveError("request carries no candidate archive")
        archive = base64.b64decode(archive_b64, validate=True)
        if len(archive) > MAX_ARCHIVE_BYTES:
            raise ArchiveError("candidate archive is too large")
        if hashlib.sha256(archive).hexdigest() != body.get("archive_sha256"):
            raise ArchiveError("candidate archive checksum mismatch")
        record["archive"] = ledger.keep_archive(seq, archive)
        workdir.mkdir(mode=0o700)
        record["files"] = safe_extract(archive, workdir)
        # A stale output file shipped in the tree must never stand in for the
        # packing this run returns.
        stale = workdir / OUTPUT_NAME
        if stale.is_symlink() or stale.exists():
            stale.unlink()
        chown_tree(workdir, candidate.pw_uid, candidate.pw_gid)
        run = run_candidate(workdir, candidate.pw_uid, candidate.pw_gid, logs)
        payload = run.get("payload")
        if isinstance(payload, dict) and payload.get("status") == "ok":
            circles = payload.get("circles")
            verdict = packing_rules.evaluate_packing(circles)
        else:
            verdict = packing_rules.evaluate_packing(None)
            verdict["errors"] = [
                run.get("output_error")
                or (payload or {}).get("error")
                or "pack_circles did not return a packing"
            ]
    except (ArchiveError, ValueError) as exc:
        verdict = {
            "valid": False,
            "score": 0.0,
            "errors": [f"candidate tree rejected: {exc}"],
            "details": {"expected_circles": packing_rules.N_CIRCLES},
        }
    except Exception as exc:  # noqa: BLE001 - every started execution is recorded
        verdict = {
            "valid": False,
            "score": 0.0,
            "errors": [f"evaluator internal error: {type(exc).__name__}"],
            "details": {"expected_circles": packing_rules.N_CIRCLES},
        }
        record["internal_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        kill_user_processes(candidate.pw_uid)
        reap_children()
        shutil.rmtree(workdir, ignore_errors=True)

    side_info = build_side_info(verdict, run, circles)
    record.update(
        {
            "finished_utc": _utc_now(),
            "valid": verdict["valid"],
            "score": verdict["score"],
            "errors": verdict["errors"],
            "details": verdict["details"],
            "circles": circles if isinstance(circles, list) and len(circles) <= 64 else None,
            "timed_out": run.get("timed_out"),
            "exit_code": run.get("exit_code"),
            "wall_seconds": run.get("wall_seconds"),
            "processes_killed_after_run": run.get("processes_killed_after_run"),
            "rules": _rules(),
        }
    )
    ledger.record_evaluation(record)
    return {"evaluation_id": evaluation_id, "results": [[verdict["score"], side_info]]}


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    server_version = "cp-evaluator/1"
    ledger: Ledger
    local: set[str]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        sys.stderr.write("%s %s\n" % (self.address_string(), format % args))

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        data = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _reject(self, status: int, reason: str) -> None:
        self.ledger.record_rejection(
            {
                "utc": _utc_now(),
                "peer": self.client_address[0],
                "method": self.command,
                "path": self.path,
                "status": status,
                "reason": reason,
            }
        )
        self._send(status, {"error": reason})

    def do_GET(self) -> None:  # noqa: N802
        self._send(HTTPStatus.METHOD_NOT_ALLOWED, {"error": "use POST /evaluate"})

    def do_POST(self) -> None:  # noqa: N802
        peer = self.client_address[0]
        if is_local_peer(peer, self.local):
            self._reject(HTTPStatus.FORBIDDEN, "requests from inside the evaluator are refused")
            return
        if self.path.split("?", 1)[0] != "/evaluate":
            self._reject(HTTPStatus.NOT_FOUND, "unknown path")
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
        except ValueError:
            length = -1
        if length < 0 or length > MAX_REQUEST_BYTES:
            self._reject(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "missing or oversized body")
            return
        if not _evaluation_lock.acquire(blocking=False):
            self._reject(HTTPStatus.CONFLICT, "another evaluation is running")
            return
        try:
            if self.ledger.exhausted():
                self._reject(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    f"execution cap of {MAX_EXECUTIONS} scoring executions reached",
                )
                return
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
            except (ValueError, UnicodeDecodeError) as exc:
                self._reject(HTTPStatus.BAD_REQUEST, f"malformed request: {exc}")
                return
            try:
                response = evaluate_request(body, self.ledger, peer)
            except ValueError as exc:
                self._reject(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send(HTTPStatus.OK, response)
        finally:
            _evaluation_lock.release()


def main() -> int:
    if os.geteuid() != 0:
        print("evaluator_server must run as root to drop privileges", file=sys.stderr)
        return 2
    ledger = Ledger(LEDGER_DIR)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(WORK_ROOT, 0o711)
    Handler.ledger = ledger
    Handler.local = local_addresses()
    ledger.record_service_event(
        {
            "utc": _utc_now(),
            "event": "start",
            "existing_evaluations": ledger.seq,
            "rules": _rules(),
            "local_addresses": sorted(Handler.local),
            "packing_rules_sha256": hashlib.sha256(
                Path(packing_rules.__file__).read_bytes()
            ).hexdigest(),
        }
    )
    server = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    server.daemon_threads = True
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
