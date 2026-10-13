"""Evaluator for the maniskill_pick_place benchmark.

Reads the instance ids in ``helix_batch.json`` and prints one
``HELIX_RESULT=[[score, side_info], ...]`` line. Run it as
``python3 -I evaluate.py`` so nothing in the working directory can shadow a
module it imports.

Three kinds of process take part:

- This parent process, which never imports ``sim`` or ``solver``.
- An env process, which loads ``sim``, owns the ManiSkill envs and scores
  every trial. If it dies or stops answering, that trial scores 0 and the
  rest of the batch runs in a new env process.
- A solver per trial, which runs in a throwaway Docker container built from
  ``solver.Dockerfile``: no network, a read-only filesystem, and only a copy
  of ``solver/`` and this file mounted. It loads ``solver`` and gets a
  ``ctx`` whose four primitives are JSON lines to the env process over the
  container's stdin and stdout, so a candidate can change its score only by
  moving the robot. A solver that does not finish within SOLVER_WALL_SECONDS
  is killed, and its trial is scored as it stands.

``--unsafe-local`` runs each solver as a plain local process instead, as
your own user and with no isolation, for machines without Docker.

If the simulator or Docker cannot start, the evaluator exits with that error
instead of printing a result, so HELIX does not score zeros.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from types import ModuleType
from typing import IO, Any, NoReturn

# Loading ManiSkill and building the envs; not charged to any trial.
STARTUP_WALL_SECONDS = 600.0
# A solver normally finishes in a few seconds. This only trips on solvers that
# loop without calling a primitive, or that swallow BudgetExhausted.
SOLVER_WALL_SECONDS = 60.0
# How long the env process gets beyond the solver's limit to settle and score
# a trial before the parent gives up on it.
SCORING_WALL_SECONDS = 60.0
# The largest request a solver may send.
MAX_REQUEST_BYTES = 1 << 16

ROOT = Path(__file__).resolve().parent
PRIMITIVES = ("get_observation", "move_to", "open_gripper", "close_gripper")
SOLVER_DOCKERFILE = ROOT / "solver.Dockerfile"
# Every solver container of one evaluation carries this label with the same
# value, so the parent can remove any that outlive their env process.
CONTAINER_LABEL = "helix-maniskill-evaluation"


def _read_batch_ids() -> list[str]:
    """Read the instance ids HELIX wrote to helix_batch.json."""
    ids = json.loads((Path.cwd() / "helix_batch.json").read_text())
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError("helix_batch.json must contain a JSON list[str]")
    return ids


def _load_package(name: str, directory: Path) -> ModuleType:
    """Import the package in *directory* by path, not through sys.path."""
    spec = importlib.util.spec_from_file_location(
        name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {directory}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _silence_stdio() -> None:
    # Nothing written by this process or its children, including native
    # libraries and docker, can then reach the pipes that carry HELIX_RESULT
    # and stderr.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)
    os.dup2(devnull, 2)


def _traceback() -> str:
    # The innermost frames, where the error was raised.
    return traceback.format_exc(limit=-8)


def _benchmark_error() -> dict[str, Any]:
    return {
        "reward": 0.0,
        "success": False,
        "stage": "benchmark_error",
        "benchmark_error": _traceback(),
    }


# --- solver (inside the container) -----------------------------------------


class BudgetExhausted(BaseException):
    """Raised in the solver once the trial's step budget is spent.

    It derives from BaseException so that a solver's ``except Exception``
    cannot swallow it.
    """


def _send_line(stream: IO[bytes], message: dict[str, Any]) -> None:
    stream.write(json.dumps(message).encode() + b"\n")
    stream.flush()


class _Ctx:
    """The ``ctx`` a solver sees: task metadata and the four primitives."""

    def __init__(
        self,
        requests: IO[bytes],
        replies: IO[bytes],
        task_family: str,
        instruction: str,
    ) -> None:
        self._out = requests
        self._in = replies
        self.task_family = task_family
        self.instruction = instruction

    def _call(self, name: str, *args: float) -> dict[str, Any]:
        _send_line(self._out, {"call": name, "args": list(args)})
        reply = json.loads(self._in.readline())
        if reply.get("end"):
            raise BudgetExhausted
        if "error" in reply:
            raise ValueError(reply["error"])
        return reply

    def get_observation(self) -> dict[str, Any]:
        return self._call("get_observation")

    def move_to(self, x: float, y: float, z: float) -> dict[str, Any]:
        return self._call("move_to", x, y, z)

    def open_gripper(self) -> dict[str, Any]:
        return self._call("open_gripper")

    def close_gripper(self) -> dict[str, Any]:
        return self._call("close_gripper")


def _solver_client() -> None:
    """Run solver.run(ctx) once, talking to the env process over stdin/stdout."""
    # Keep the protocol on private copies of fds 0 and 1, and give the solver
    # an empty stdin and a stdout that goes to stderr, which the env process
    # discards. Nothing the solver reads or prints can then garble a message.
    replies = os.fdopen(os.dup(0), "rb")
    requests = os.fdopen(os.dup(1), "wb")
    os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
    os.dup2(2, 1)
    task = json.loads(replies.readline())
    error = ""
    try:
        solver = _load_package("solver", ROOT / "solver")
        solver.run(_Ctx(requests, replies, task["task_family"], task["instruction"]))
    except BudgetExhausted:
        pass
    except (Exception, SystemExit):  # solver bugs become feedback
        error = "solver.run(ctx) raised:\n" + _traceback()
    try:
        _send_line(requests, {"done": True, "error": error})
    except (OSError, ValueError):
        pass


# --- env process ------------------------------------------------------------


def _serve(
    trial: Any, request: Any, budget_exhausted: type
) -> tuple[dict[str, Any], bool]:
    """Answer one solver request. Returns the reply and whether the trial ends."""
    if not isinstance(request, dict) or request.get("call") not in PRIMITIVES:
        return {"error": f"unknown request {request!r:.200}"}, False
    name, args = request["call"], request.get("args", [])
    if name == "move_to":
        if not (
            isinstance(args, list)
            and len(args) == 3
            and all(
                isinstance(a, (int, float))
                and not isinstance(a, bool)
                and math.isfinite(a)
                for a in args
            )
        ):
            return {"error": "move_to needs three finite numbers"}, False
    elif args:
        return {"error": f"{name} takes no arguments"}, False
    try:
        return getattr(trial, name)(*args), False
    except budget_exhausted:
        return {"status": "budget_exhausted", "end": True}, True


def _solver_command(
    workspace: str, image: str | None, name: str, label: str
) -> list[str]:
    """The command that runs one solver against the copy in *workspace*."""
    if image is None:  # --unsafe-local
        return [
            sys.executable,
            "-I",
            str(Path(workspace) / "evaluate.py"),
            "--solver-client",
        ]
    return [
        "docker", "run", "--rm", "--interactive",
        "--name", name,
        "--label", f"{CONTAINER_LABEL}={label}",
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--pids-limit", "256",
        "--memory", "1g",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m",
        "--volume", f"{workspace}:/workspace:ro",
        "--workdir", "/workspace",
        "--env", "PYTHONDONTWRITEBYTECODE=1",
        image,
        "python3", "-I", "evaluate.py", "--solver-client",
    ]  # fmt: skip


def _read_lines(stream: IO[bytes], lines: queue.Queue[bytes | None]) -> None:
    # On a thread, so a solver that writes half a line and stalls cannot block
    # the env process past the solver's deadline.
    while line := stream.readline(MAX_REQUEST_BYTES + 1):
        lines.put(line)
    lines.put(None)


def _stop_solver(proc: subprocess.Popen[bytes], name: str, image: str | None) -> None:
    if proc.poll() is None:
        if image is not None:
            subprocess.run(["docker", "kill", name], capture_output=True)
        proc.kill()
    proc.wait()


def _run_solver(
    trial: Any,
    solver_seconds: float,
    budget_exhausted: type,
    workspace: str,
    image: str | None,
    label: str,
) -> str:
    """Let one solver drive *trial*. Returns feedback on how it ended."""
    name = f"helix-solver-{uuid.uuid4().hex[:12]}"
    proc = subprocess.Popen(
        _solver_command(workspace, image, name, label),
        cwd=workspace,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    assert proc.stdin is not None and proc.stdout is not None
    lines: queue.Queue[bytes | None] = queue.Queue()
    threading.Thread(target=_read_lines, args=(proc.stdout, lines), daemon=True).start()
    deadline = time.monotonic() + solver_seconds
    finished = False
    try:
        _send_line(
            proc.stdin,
            {
                "task_family": trial.instance.family,
                "instruction": trial.instance.instruction,
            },
        )
        while True:
            try:
                line = lines.get(timeout=max(deadline - time.monotonic(), 0.0))
            except queue.Empty:
                return (
                    f"The solver did not finish within {solver_seconds:.0f} s and "
                    "was stopped; the trial was scored as it stands. Once the step "
                    "budget is spent, move_to and the gripper calls raise "
                    "BudgetExhausted to end solver.run, so the solver most likely "
                    "looped without calling them, or caught that exception with a "
                    "bare except."
                )
            if line is None:
                return (
                    f"The solver process exited with code {proc.wait()} before "
                    "solver.run returned, for example because solver code called "
                    "os._exit(); the trial was scored as it stands."
                )
            try:
                if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
                    raise ValueError("request too long")
                request = json.loads(line)
            except ValueError:
                return (
                    "The solver sent a request the benchmark could not read; the "
                    "trial was scored as it stands."
                )
            if isinstance(request, dict) and request.get("done"):
                finished = True
                return str(request.get("error", ""))
            reply, end = _serve(trial, request, budget_exhausted)
            _send_line(proc.stdin, reply)
            if end:
                finished = True
                return ""
    except BrokenPipeError:
        return (
            f"The solver process exited with code {proc.wait()} before solver.run "
            "returned; the trial was scored as it stands."
        )
    finally:
        if finished:
            try:
                proc.wait(timeout=5.0)  # it is exiting on its own
            except subprocess.TimeoutExpired:
                pass
        _stop_solver(proc, name, image)


def _env_main(
    root: str,
    instance_ids: list[str],
    conn: Connection,
    solver_seconds: float,
    image: str | None,
    label: str,
) -> None:
    """Env process: run each trial and send its payload to the parent."""
    _silence_stdio()
    # The solvers get a private copy of solver/ and this file, nothing else.
    workspace = tempfile.mkdtemp(prefix="helix-solver-")
    os.chmod(workspace, 0o755)  # readable by the container's own user
    shutil.copytree(
        Path(root) / "solver",
        Path(workspace) / "solver",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copy(Path(root) / "evaluate.py", workspace)
    try:
        _run_trials(root, instance_ids, conn, solver_seconds, image, label, workspace)
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _run_trials(
    root: str,
    instance_ids: list[str],
    conn: Connection,
    solver_seconds: float,
    image: str | None,
    label: str,
    workspace: str,
) -> None:
    try:
        _load_package("sim", Path(root) / "sim")
        from sim.env import BudgetExhausted as SimBudgetExhausted
        from sim.env import ManiSkillTrial, make_env
        from sim.tasks import make_instance

        env_ids = set()
        for instance_id in instance_ids:
            try:
                env_ids.add(make_instance(instance_id).env_id)
            except ValueError:
                pass  # reported with its trial below
        for env_id in sorted(env_ids):
            make_env(env_id)
    except Exception:
        conn.send({"fatal": traceback.format_exc()})
        return
    conn.send({"ready": True})

    for instance_id in instance_ids:
        payload: dict[str, Any] = {"instance_id": instance_id}
        try:
            make_instance(instance_id)
        except ValueError:
            payload.update(_benchmark_error())
            conn.send(payload)
            continue
        try:
            trial = ManiSkillTrial(instance_id)
        except Exception:
            # The simulator cannot build this env, so no score would mean anything.
            conn.send({"fatal": traceback.format_exc()})
            return
        try:
            note = _run_solver(
                trial, solver_seconds, SimBudgetExhausted, workspace, image, label
            )
            if note:
                payload["solver_note"] = note
            result = trial.finish()
        except Exception:
            payload.update(_benchmark_error())
        else:
            payload.update(
                reward=result.reward,
                success=result.success,
                stage=result.stage,
                hint=result.hint,
                diagnostics=result.diagnostics,
            )
        conn.send(payload)
    conn.close()


# --- parent process ---------------------------------------------------------


def _score_and_side_info(payload: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Turn one env payload into a HELIX_RESULT pair."""
    reward = payload["reward"]
    parts = [
        f"[{payload['instance_id']}] stage={payload['stage']}, reward={reward:.3f}."
    ]
    if "hint" in payload:
        parts.append(payload["hint"])
    if "solver_note" in payload:
        parts.append(payload["solver_note"])
    if "benchmark_error" in payload:
        parts.append("Benchmark-side error:\n" + payload["benchmark_error"])
    side_info = {
        "scores": {"reward": reward, "success": 1.0 if payload["success"] else 0.0},
        "feedback": " ".join(parts),
        **payload.get("diagnostics", {}),
    }
    return reward, side_info


def _unscored(feedback: str) -> tuple[float, dict[str, Any]]:
    return 0.0, {"scores": {"reward": 0.0, "success": 0.0}, "feedback": feedback}


def _stop(proc: BaseProcess) -> None:
    """Stop *proc*, escalating from SIGTERM to SIGKILL."""
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=5.0)
    if proc.is_alive():
        proc.kill()
        proc.join()


def _remove_containers(label: str) -> None:
    """Kill solver containers left behind by an env process that died."""
    listed = subprocess.run(
        ["docker", "ps", "--quiet", "--filter", f"label={CONTAINER_LABEL}={label}"],
        capture_output=True,
        text=True,
    )
    if ids := listed.stdout.split():
        subprocess.run(["docker", "kill", *ids], capture_output=True)


def _no_docker(detail: str) -> NoReturn:
    sys.exit(
        "Solvers run in a Docker container and Docker is not available, so no "
        "trial was scored. Start Docker, or pass --unsafe-local to run solvers "
        "as your own user with no isolation.\n" + detail
    )


def _solver_image() -> str:
    """Build the solver image from solver.Dockerfile unless it already exists."""
    dockerfile = SOLVER_DOCKERFILE.read_bytes()
    tag = f"helix-maniskill-solver:{hashlib.sha256(dockerfile).hexdigest()[:12]}"
    try:
        found = subprocess.run(["docker", "image", "inspect", tag], capture_output=True)
    except FileNotFoundError:
        _no_docker("The docker command was not found.")
    if found.returncode != 0:
        built = subprocess.run(
            ["docker", "build", "--quiet", "--tag", tag, "-"],
            input=dockerfile,
            capture_output=True,
        )
        if built.returncode != 0:
            _no_docker(built.stderr.decode(errors="replace")[-2000:])
    return tag


def _receive(conn: Connection, timeout: float) -> tuple[str, dict[str, Any]]:
    """Wait for the env process's next message.

    Returns ("ok", message), ("timeout", {}) or ("exited", {}).
    """
    if not conn.poll(timeout):
        return "timeout", {}
    try:
        return "ok", conn.recv()
    except EOFError:
        return "exited", {}


def _startup_failed(detail: str) -> NoReturn:
    sys.exit(
        "The simulator failed to start, so no trial was scored. See the "
        "Requirements section of README.md.\n" + detail
    )


def _run_env(
    batch: list[str], solver_seconds: float, image: str | None, label: str
) -> tuple[list[dict[str, Any]], str]:
    """Run *batch* in one env process.

    Returns the payloads it sent, in batch order, and the feedback for the
    trial it stopped at, or "" if it finished the batch.
    """
    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    proc = mp_ctx.Process(
        target=_env_main,
        args=(str(ROOT), batch, child_conn, solver_seconds, image, label),
    )
    proc.start()
    child_conn.close()
    trial_seconds = solver_seconds + SCORING_WALL_SECONDS

    payloads: list[dict[str, Any]] = []
    stop_reason = ""
    try:
        status, message = _receive(parent_conn, STARTUP_WALL_SECONDS)
        if status == "timeout":
            _startup_failed(
                f"The simulator did not load within {STARTUP_WALL_SECONDS:.0f} s."
            )
        if status == "exited":
            _stop(proc)
            _startup_failed(f"The simulator process exited with code {proc.exitcode}.")
        if "fatal" in message:
            _startup_failed(message["fatal"])

        for instance_id in batch:
            status, message = _receive(parent_conn, trial_seconds)
            if status == "timeout":
                stop_reason = (
                    f"Trial {instance_id} was not scored within "
                    f"{trial_seconds:.0f} s, so the evaluator stopped it."
                )
                break
            if status == "exited":
                _stop(proc)
                stop_reason = (
                    f"The evaluator's env process exited with code {proc.exitcode} "
                    f"during trial {instance_id}, for example because solver code "
                    "killed it."
                )
                break
            if "fatal" in message:
                _startup_failed(message["fatal"])
            payloads.append(message)
    finally:
        _stop(proc)
        if image is not None:
            _remove_containers(label)
    return payloads, stop_reason


def evaluate(
    solver_seconds: float = SOLVER_WALL_SECONDS, unsafe_local: bool = False
) -> None:
    ids = _read_batch_ids()
    if unsafe_local:
        print(
            "WARNING: --unsafe-local runs solver code as your own user with no "
            "isolation; it can change any file you can, including the scorer.",
            file=sys.stderr,
        )
        image = None
    else:
        image = _solver_image()
    label = uuid.uuid4().hex
    results: list[tuple[float, dict[str, Any]]] = []
    while len(results) < len(ids):
        payloads, stop_reason = _run_env(
            ids[len(results) :], solver_seconds, image, label
        )
        results.extend(_score_and_side_info(p) for p in payloads)
        if stop_reason:
            # Score the trial that stopped the env process; the rest run in a new one.
            results.append(_unscored(stop_reason))
    print("HELIX_RESULT=" + json.dumps(results))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--solver-seconds",
        type=float,
        default=SOLVER_WALL_SECONDS,
        help="wall time each trial's solver gets (default: %(default)s)",
    )
    parser.add_argument(
        "--unsafe-local",
        action="store_true",
        help="run solvers as local processes with no isolation (no Docker needed)",
    )
    parser.add_argument("--solver-client", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.solver_client:
        _solver_client()
    else:
        evaluate(args.solver_seconds, args.unsafe_local)
