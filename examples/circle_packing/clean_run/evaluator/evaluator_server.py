"""Private circle-packing evaluator service (the HELIX evaluator sidecar).

Runs as root inside the sidecar container, on HELIX's internal evaluator
network.  Mutation agents run on a different network and cannot reach it.

For every ``POST /evaluate`` from an evaluator-runner container it:

1. rejects requests that originate inside this container (so candidate code
   it is executing cannot request extra evaluations) and requests that arrive
   while another evaluation is running;
2. answers a request for a candidate tree it has already executed from its
   cache, without executing anything (``CP_CACHE_IDENTICAL_TREES=1``), as
   optimize_anything's circle-packing run never re-executed unchanged code;
3. otherwise allocates the next execution number, records that the
   execution started, unpacks the candidate tree into a fresh directory owned
   by the unprivileged ``candidate`` user and runs ``solve.pack_circles(26)``
   through ``candidate_harness.py`` as that user, with a wall-clock limit;
4. kills every process the user owns and deletes everything the user left
   in world-writable locations (``/tmp``, ``/var/tmp``, ``/dev/shm``, ...) and
   its System V IPC objects, so no state carries over to the next execution;
5. validates and scores the returned packing with ``packing_rules`` (strict
   geometry, certified score; invalid packings, exceptions and timeouts
   score 0);
6. appends one line to the private ledger ``evaluations.jsonl`` for every
   started execution, whatever happened, and keeps the candidate tree it
   scored under ``candidate_archives/``;
7. returns ``[score, side_info]`` for the runner to print as ``HELIX_RESULT``.

Every answered request (execution or cache hit) is also listed in
``requests.jsonl`` with the HELIX candidate id the runner sent.

``CP_MAX_EXECUTIONS`` (0 = unlimited) is the execution budget: once that
many executions have started, ``budget_exhausted.json`` is written and
any request that would need a new execution, or that starts a new proposal
round (a ``train``-split request), is refused with HTTP 429.  A validation
request for a tree already executed is still answered from the cache, so the
last candidate executed within budget can finish its evaluation.

A monitor thread aborts the service (after recording the execution) when an
evaluation exceeds its time limit plus ``CP_POST_RUN_GRACE`` seconds, so a
candidate can never hang the service.

The ledger sits below a root-only directory; the candidate user can read
neither the ledger nor this service's code.
"""

from __future__ import annotations

import base64
import ctypes
import ctypes.util
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
# Full candidate stdout/stderr go here (inside the container, root only, not
# bind-mounted); the ledger keeps only a capped tail of each.
STREAM_ROOT = Path(os.environ.get("CP_STREAM_ROOT", "/srv/cpeval/streams"))
WORK_ROOT = Path(os.environ.get("CP_WORK_ROOT", "/work"))
HARNESS = os.environ.get("CP_HARNESS", "/opt/cpeval/harness/candidate_harness.py")
CANDIDATE_USER = os.environ.get("CP_CANDIDATE_USER", "candidate")
CANDIDATE_TIMEOUT_SECONDS = float(os.environ.get("CP_CANDIDATE_TIMEOUT", "600"))
MAX_EXECUTIONS = int(os.environ.get("CP_MAX_EXECUTIONS", "0"))
CACHE_IDENTICAL_TREES = os.environ.get("CP_CACHE_IDENTICAL_TREES", "1") == "1"
POST_RUN_GRACE_SECONDS = float(os.environ.get("CP_POST_RUN_GRACE", "180"))

MAX_REQUEST_BYTES = 48 * 1024 * 1024
MAX_ARCHIVE_BYTES = 32 * 1024 * 1024
MAX_UNPACKED_BYTES = 64 * 1024 * 1024
MAX_MEMBERS = 5000
MAX_OUTPUT_BYTES = 2 * 1024 * 1024
STREAM_TAIL = 4000
STORED_STREAM_BYTES = 256 * 1024
OUTPUT_NAME = ".cp_eval_output.json"
BUDGET_MARKER = "budget_exhausted.json"

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


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class Ledger:
    """Append-only JSONL records; ``seq`` numbers scoring executions.

    ``execution_starts.jsonl`` gets a line before an execution begins and
    ``evaluations.jsonl`` a line when it ends, so an execution that never
    finished shows up as a start without a record, and a restarted service
    never reuses its number.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)
        self.evaluations = directory / "evaluations.jsonl"
        self.starts = directory / "execution_starts.jsonl"
        self.requests = directory / "requests.jsonl"
        self.archives = directory / "candidate_archives"
        self.archives.mkdir(mode=0o700, exist_ok=True)
        self.rejections = directory / "rejected_requests.jsonl"
        self.service = directory / "service_events.jsonl"
        self.marker = directory / BUDGET_MARKER
        records = _read_jsonl(self.evaluations)
        started = _read_jsonl(self.starts)
        self.seq = max(
            [0]
            + [int(r["seq"]) for r in records if isinstance(r.get("seq"), int)]
            + [int(r["seq"]) for r in started if isinstance(r.get("seq"), int)]
        )
        self.executions = sum(1 for r in records if isinstance(r.get("seq"), int))
        self.request_no = len(_read_jsonl(self.requests))
        self._recorded: set[int] = {
            int(r["seq"]) for r in records if isinstance(r.get("seq"), int)
        }
        # archive_sha256 -> (seq, results) of the first execution of a tree.
        self.cache: dict[str, tuple[int, list[Any]]] = {}
        for record in records:
            self._remember(record)

    def _append(self, path: Path, record: dict[str, Any]) -> None:
        line = json.dumps(_json_safe(record), sort_keys=True, allow_nan=False)
        with _ledger_lock:
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
                os.fsync(fd)
            finally:
                os.close(fd)

    def _remember(self, record: dict[str, Any]) -> None:
        key = record.get("archive_sha256")
        results = record.get("response_results")
        if (
            CACHE_IDENTICAL_TREES
            and isinstance(key, str)
            and record.get("archive")
            and isinstance(results, list)
            and not record.get("internal_error")
            and key not in self.cache
        ):
            self.cache[key] = (int(record["seq"]), results)

    def start_execution(self, record: dict[str, Any]) -> int:
        with _ledger_lock:
            self.seq += 1
            seq = self.seq
        self._append(self.starts, {**record, "seq": seq, "started_utc": _utc_now()})
        return seq

    def record_evaluation(self, record: dict[str, Any]) -> bool:
        """Append the record for ``record['seq']`` once; return False if already there."""
        seq = int(record["seq"])
        with _ledger_lock:
            if seq in self._recorded:
                return False
            self._recorded.add(seq)
            self.executions += 1
        self._append(self.evaluations, record)
        self._remember(record)
        self.write_marker_if_exhausted()
        return True

    def write_marker_if_exhausted(self) -> None:
        if not self.exhausted() or self.marker.exists():
            return
        fd = os.open(self.marker, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump(
                {"utc": _utc_now(), "executions_started": self.seq,
                 "executions_recorded": self.executions,
                 "max_executions": MAX_EXECUTIONS},
                handle,
            )

    def record_request(self, record: dict[str, Any]) -> None:
        with _ledger_lock:
            self.request_no += 1
            number = self.request_no
        self._append(self.requests, {**record, "request_no": number})

    def cached(self, archive_sha256: Any) -> tuple[int, list[Any]] | None:
        if not CACHE_IDENTICAL_TREES or not isinstance(archive_sha256, str):
            return None
        return self.cache.get(archive_sha256)

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
        """Every started execution counts, including one that never finished."""
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


# ---------------------------------------------------------------------------
# State the candidate could leave behind
# ---------------------------------------------------------------------------

_EXCLUDED_ROOTS = ("/proc", "/sys")


def candidate_writable_dirs(
    uid: int, gid: int, exclude: list[Path], root: str = "/"
) -> list[str]:
    """Directories outside ``exclude`` in which ``uid`` can create entries.

    Scanned once at start-up (the whole filesystem except ``/proc``, ``/sys``
    and ``exclude``).  The candidate cannot make a directory it does not own
    writable, and every directory it creates is owned by it, so deleting the
    candidate's entries in these directories after every execution removes
    everything it wrote outside its (deleted) working directory.
    """
    excluded = tuple(str(p) for p in exclude) + _EXCLUDED_ROOTS
    found: list[str] = []
    for directory, dirnames, _ in os.walk(root, followlinks=False):
        dirnames[:] = [
            d for d in dirnames
            if not os.path.join(directory, d).startswith(excluded)
        ]
        try:
            info = os.lstat(directory)
        except OSError:
            continue
        mode = info.st_mode
        if (
            mode & stat.S_IWOTH
            or (info.st_uid == uid and mode & stat.S_IWUSR)
            or (info.st_gid == gid and mode & stat.S_IWGRP)
        ):
            found.append(directory)
    return sorted(found)


SCRUB_TIMEOUT_SECONDS = 120


def owned_entries(directories: list[str], uid: int) -> list[str]:
    """Entries directly inside ``directories`` whose owner is ``uid`` (links not followed)."""
    found: list[str] = []
    for directory in directories:
        try:
            entries = list(os.scandir(directory))
        except OSError:
            continue
        for entry in entries:
            try:
                if entry.stat(follow_symlinks=False).st_uid == uid:
                    found.append(entry.path)
            except OSError:
                continue
    return sorted(found)


def remove_owned_entries(directories: list[str], uid: int) -> tuple[list[str], list[str]]:
    """Delete every entry owned by ``uid`` in ``directories``.

    Entries are removed whole with ``rm -rf`` (iterative, so arbitrarily deep
    trees cannot exhaust Python's recursion limit; links are never
    followed).  Returns ``(removed, remaining)``: an entry that is still
    there afterwards is reported, never silently kept.
    """
    entries = owned_entries(directories, uid)
    if entries:
        for start in range(0, len(entries), 500):
            try:
                subprocess.run(
                    ["rm", "-rf", "--", *entries[start : start + 500]],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=SCRUB_TIMEOUT_SECONDS,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                pass
    remaining = [path for path in entries if os.path.lexists(path)]
    removed = [path for path in entries if path not in remaining]
    return removed, remaining


def remove_owned_ipc(uid: int) -> list[str]:
    """Delete System V shared memory, semaphores and queues owned by ``uid``."""
    owned: list[tuple[str, int]] = []
    for table, id_column in (("shm", "shmid"), ("sem", "semid"), ("msg", "msqid")):
        try:
            lines = Path(f"/proc/sysvipc/{table}").read_text().splitlines()
        except OSError:
            continue
        if not lines:
            continue
        header = lines[0].split()
        for line in lines[1:]:
            fields = dict(zip(header, line.split()))
            try:
                if int(fields.get("uid", "-1")) == uid:
                    owned.append((table, int(fields[id_column])))
            except (KeyError, ValueError):
                continue
    libc_name = ctypes.util.find_library("c") if owned else None
    if libc_name is None:
        return []
    libc = ctypes.CDLL(libc_name, use_errno=True)
    ipc_rmid = 0
    removers = {
        "shm": lambda ident: libc.shmctl(ident, ipc_rmid, None),
        "sem": lambda ident: libc.semctl(ident, 0, ipc_rmid),
        "msg": lambda ident: libc.msgctl(ident, ipc_rmid, None),
    }
    return [f"{table}:{ident}" for table, ident in owned if removers[table](ident) == 0]


# ---------------------------------------------------------------------------
# Running a candidate
# ---------------------------------------------------------------------------


def _tail(path: Path, size: int = STREAM_TAIL) -> str:
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            handle.seek(max(0, end - size))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _store_stream(full: Path, stored: Path) -> int:
    """Copy at most the last ``STORED_STREAM_BYTES`` of ``full``; return its size."""
    try:
        size = full.stat().st_size
        with full.open("rb") as source:
            source.seek(max(0, size - STORED_STREAM_BYTES))
            data = source.read(STORED_STREAM_BYTES)
    except OSError:
        return 0
    fd = os.open(stored, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(data)
    return size


def _read_candidate_output(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read the harness output without ever blocking on it.

    Candidate code controls this path: it may be missing, a symlink, a FIFO, a
    directory or a device.  Only a regular file is read, and it is opened
    non-blocking and without following links, after an ``lstat``.
    """
    try:
        info = os.lstat(path)
    except OSError:
        return None, "the candidate produced no output (crashed, killed or timed out)"
    if not stat.S_ISREG(info.st_mode):
        return None, "candidate output is not a regular file"
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None, "candidate output could not be opened"
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
    streams = STREAM_ROOT / workdir.name
    streams.mkdir(mode=0o700, parents=True, exist_ok=True)
    stdout_path = streams / "stdout.txt"
    stderr_path = streams / "stderr.txt"
    started = time.monotonic()
    timed_out = False
    try:
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
            "stdout_bytes": _store_stream(stdout_path, logs / "stdout.tail.txt"),
            "stderr_bytes": _store_stream(stderr_path, logs / "stderr.tail.txt"),
        }
    finally:
        shutil.rmtree(streams, ignore_errors=True)


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
        "cache_identical_trees": CACHE_IDENTICAL_TREES,
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


class BudgetExhausted(Exception):
    """The request needs a new execution or starts a round after the budget."""


class InFlight:
    """The execution in progress, for the deadline monitor."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.record: dict[str, Any] | None = None
        self.started = 0.0

    def begin(self, record: dict[str, Any]) -> None:
        with self.lock:
            self.record = record
            self.started = time.monotonic()

    def end(self) -> None:
        with self.lock:
            self.record = None

    def overdue(self, limit: float) -> dict[str, Any] | None:
        with self.lock:
            if self.record is not None and time.monotonic() - self.started > limit:
                return self.record
            return None


IN_FLIGHT = InFlight()


def _short_str(value: Any, limit: int = 200) -> str | None:
    return value[:limit] if isinstance(value, str) else None


def _request_fields(body: dict[str, Any], peer: str) -> dict[str, Any]:
    return {
        "peer": peer,
        "split": str(body.get("split", "")),
        "ids": body.get("ids"),
        "candidate_id": _short_str(body.get("candidate_id")),
        "phase": _short_str(body.get("phase")),
        "archive_sha256": _short_str(body.get("archive_sha256")),
        "files_sha256": _short_str(body.get("files_sha256")),
    }


def _decode_archive(body: dict[str, Any]) -> tuple[bytes | None, str | None]:
    """Return the candidate archive the request carries, or why it has none."""
    if body.get("client_error"):
        return None, str(body["client_error"])[:500]
    archive_b64 = body.get("archive_b64")
    if not isinstance(archive_b64, str):
        return None, "request carries no candidate archive"
    try:
        archive = base64.b64decode(archive_b64, validate=True)
    except ValueError:
        return None, "candidate archive is not valid base64"
    if len(archive) > MAX_ARCHIVE_BYTES:
        return None, "candidate archive is too large"
    if hashlib.sha256(archive).hexdigest() != body.get("archive_sha256"):
        return None, "candidate archive checksum mismatch"
    return archive, None


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
    fields = _request_fields(body, peer)
    archive, archive_error = _decode_archive(body)

    # Only a verified archive can be answered from the cache.
    cached = ledger.cached(fields["archive_sha256"]) if archive is not None else None
    if ledger.exhausted() and (cached is None or fields["split"] != "val"):
        raise BudgetExhausted()
    if cached is not None:
        served_seq, results = cached
        ledger.record_request(
            {**fields, "evaluation_id": evaluation_id, "received_utc": received,
             "finished_utc": _utc_now(), "kind": "cache_hit", "seq": served_seq,
             "score": results[0][0] if results and isinstance(results[0], list) else None}
        )
        return {"evaluation_id": evaluation_id, "results": results, "cached_from_seq": served_seq}

    seq = ledger.start_execution({**fields, "evaluation_id": evaluation_id})
    record: dict[str, Any] = {
        **fields,
        "seq": seq,
        "evaluation_id": evaluation_id,
        "received_utc": received,
    }
    candidate = pwd.getpwnam(CANDIDATE_USER)
    workdir = WORK_ROOT / evaluation_id
    logs = LEDGER_DIR / "candidate_logs" / f"{seq:05d}-{evaluation_id}"
    run: dict[str, Any] = {}
    circles: Any = None
    verdict: dict[str, Any] = {
        "valid": False,
        "score": 0.0,
        "errors": ["evaluator internal error: evaluation did not complete"],
        "details": {"expected_circles": packing_rules.N_CIRCLES},
    }
    side_info: dict[str, Any] = {}
    IN_FLIGHT.begin(record)
    try:
        logs.mkdir(parents=True, exist_ok=True)
        try:
            if archive is None:
                raise ArchiveError(archive_error or "request carries no candidate archive")
            record["archive"] = ledger.keep_archive(seq, archive)
            workdir.mkdir(mode=0o700)
            record["files"] = safe_extract(archive, workdir)
            # A stale output file shipped in the tree must never stand in for
            # the packing this run returns.
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
            removed, remaining = remove_owned_entries(
                [str(WORK_ROOT), *SCRUB_DIRS], candidate.pw_uid
            )
            if os.path.lexists(workdir):
                subprocess.run(["rm", "-rf", "--", str(workdir)],
                               timeout=SCRUB_TIMEOUT_SECONDS, check=False)
                if os.path.lexists(workdir):
                    remaining.append(str(workdir))
            record["candidate_state_removed"] = removed + remove_owned_ipc(candidate.pw_uid)
            if remaining:
                record["candidate_state_remaining"] = remaining
        side_info = build_side_info(verdict, run, circles)
    except Exception as exc:  # noqa: BLE001 - the record below is written regardless
        record.setdefault("internal_error", f"{type(exc).__name__}: {exc}")
        verdict = {
            "valid": False,
            "score": 0.0,
            "errors": [f"evaluator internal error: {type(exc).__name__}"],
            "details": {"expected_circles": packing_rules.N_CIRCLES},
        }
        side_info = {"scores": {"sum_radii": 0.0}, "valid": False,
                     "feedback": "Invalid packing, score 0: evaluator internal error"}
    finally:
        IN_FLIGHT.end()
        results = [[verdict["score"], side_info]]
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
                "stdout_bytes": run.get("stdout_bytes"),
                "stderr_bytes": run.get("stderr_bytes"),
                "rules": _rules(),
                "response_results": _json_safe(results),
            }
        )
        if ledger.record_evaluation(record):
            ledger.record_request(
                {**fields, "evaluation_id": evaluation_id, "received_utc": received,
                 "finished_utc": record["finished_utc"], "kind": "execution",
                 "seq": seq, "score": verdict["score"]}
            )
    if record.get("candidate_state_remaining"):
        ledger.record_service_event({"event": "evaluation_aborted", "utc": _utc_now(),
                                     "seq": seq, "reason": "candidate state cleanup failed"})
        os._exit(70)
    return {"evaluation_id": evaluation_id, "results": results}


def deadline_monitor(ledger: Ledger, limit: float, poll: float = 5.0) -> None:
    """Abort the service when an evaluation overruns ``limit`` seconds.

    The overdue execution is recorded first (score 0), so its number is never
    lost; the container then exits and HELIX's runner sees a failed request.
    """
    while True:
        time.sleep(poll)
        record = IN_FLIGHT.overdue(limit)
        if record is None:
            continue
        aborted = {
            **record,
            "finished_utc": _utc_now(),
            "valid": False,
            "score": 0.0,
            "errors": ["evaluator internal error: evaluation exceeded its deadline"],
            "details": {"expected_circles": packing_rules.N_CIRCLES},
            "internal_error": f"evaluation exceeded {limit:.0f} s; service aborted",
            "rules": _rules(),
        }
        ledger.record_evaluation(aborted)
        ledger.record_service_event(
            {"utc": _utc_now(), "event": "evaluation_aborted", "seq": record.get("seq"),
             "limit_seconds": limit}
        )
        os._exit(70)


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

    def _reject(self, status: int, reason: str, extra: dict[str, Any] | None = None) -> None:
        self.ledger.record_rejection(
            {
                "utc": _utc_now(),
                "peer": self.client_address[0],
                "local_peer": is_local_peer(self.client_address[0], self.local),
                "method": self.command,
                "path": self.path,
                "status": status,
                "reason": reason,
                **(extra or {}),
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
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
            except (ValueError, UnicodeDecodeError) as exc:
                self._reject(HTTPStatus.BAD_REQUEST, f"malformed request: {exc}")
                return
            try:
                response = evaluate_request(body, self.ledger, peer)
            except BudgetExhausted:
                self._reject(
                    HTTPStatus.TOO_MANY_REQUESTS,
                    f"execution budget of {MAX_EXECUTIONS} scoring executions is spent",
                    {"split": body.get("split"), "candidate_id": body.get("candidate_id")},
                )
                return
            except ValueError as exc:
                self._reject(HTTPStatus.BAD_REQUEST, str(exc))
                return
            self._send(HTTPStatus.OK, response)
        finally:
            _evaluation_lock.release()


SCRUB_DIRS: list[str] = []


def main() -> int:
    if os.geteuid() != 0:
        print("evaluator_server must run as root to drop privileges", file=sys.stderr)
        return 2
    ledger = Ledger(LEDGER_DIR)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    os.chmod(WORK_ROOT, 0o711)
    STREAM_ROOT.mkdir(mode=0o700, parents=True, exist_ok=True)
    candidate = pwd.getpwnam(CANDIDATE_USER)
    SCRUB_DIRS[:] = candidate_writable_dirs(
        candidate.pw_uid, candidate.pw_gid,
        exclude=[WORK_ROOT, LEDGER_DIR.parent, STREAM_ROOT],
    )
    ledger.write_marker_if_exhausted()
    Handler.ledger = ledger
    Handler.local = local_addresses()
    ledger.record_service_event(
        {
            "utc": _utc_now(),
            "event": "start",
            "existing_executions": ledger.executions,
            "next_seq": ledger.seq + 1,
            "unfinished_seqs": sorted(
                {int(r["seq"]) for r in _read_jsonl(ledger.starts) if isinstance(r.get("seq"), int)}
                - ledger._recorded
            ),
            "cached_trees": len(ledger.cache),
            "rules": _rules(),
            "scrubbed_directories": SCRUB_DIRS,
            "local_addresses": sorted(Handler.local),
            "packing_rules_sha256": hashlib.sha256(
                Path(packing_rules.__file__).read_bytes()
            ).hexdigest(),
        }
    )
    threading.Thread(
        target=deadline_monitor,
        args=(ledger, CANDIDATE_TIMEOUT_SECONDS + POST_RUN_GRACE_SECONDS),
        daemon=True,
    ).start()
    server = ThreadingHTTPServer(("0.0.0.0", LISTEN_PORT), Handler)
    server.daemon_threads = True
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
