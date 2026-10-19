"""Evaluator for the maniskill_pick_place benchmark.

Reads the instance ids in ``helix_batch.json`` and runs the trials in one
spawned child process, since ManiSkill is slow to import. The child sends one
payload per trial over a pipe. If the next payload does not arrive within
PER_TRIAL_WALL_SECONDS, or the child dies, the remaining trials score 0.
Prints one ``HELIX_RESULT=[[score, side_info], ...]`` line. If the simulator
raises an error while starting (for example a missing system library), it
exits with that error instead, so HELIX does not score zeros.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import traceback
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess
from pathlib import Path
from typing import Any

# Only trips on solvers that loop without calling move_to or the gripper; once
# the step budget is spent those calls end the trial.
PER_TRIAL_WALL_SECONDS = 150.0

ROOT = Path(__file__).resolve().parent


def _read_batch_ids() -> list[str]:
    """Read the instance ids HELIX wrote to helix_batch.json."""
    ids = json.loads((Path.cwd() / "helix_batch.json").read_text())
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        raise ValueError("helix_batch.json must contain a JSON list[str]")
    return ids


def _benchmark_error() -> dict[str, Any]:
    return {
        "reward": 0.0,
        "success": False,
        "stage": "benchmark_error",
        "benchmark_error": traceback.format_exc(limit=8),
    }


def _run_trials(root: str, instance_ids: list[str], conn: Connection) -> None:
    """Child process: run each trial and send its payload to the parent."""
    # Point fd 1 at /dev/null so nothing written here, by solver code or by
    # native libraries, can reach the stdout that carries HELIX_RESULT.
    os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    sys.path.insert(0, root)
    try:
        from sim import Ctx, ManiSkillTrial
        from sim.env import BudgetExhausted
        from sim.tasks import make_instance
    except Exception:
        conn.send({"fatal": traceback.format_exc()})
        return

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
            import solver

            solver.run(Ctx(trial))
        except BudgetExhausted:
            pass
        except (Exception, SystemExit):  # solver bugs become feedback
            payload["solver_error"] = traceback.format_exc(limit=8)
        try:
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


def _score_and_side_info(payload: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Turn one child payload into a HELIX_RESULT pair."""
    reward = payload["reward"]
    parts = [
        f"[{payload['instance_id']}] stage={payload['stage']}, reward={reward:.3f}."
    ]
    if "hint" in payload:
        parts.append(payload["hint"])
    if "solver_error" in payload:
        parts.append("solver.run(ctx) raised:\n" + payload["solver_error"])
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
    proc.join(timeout=5.0)
    if proc.is_alive():
        proc.terminate()
        proc.join()


def evaluate() -> None:
    ids = _read_batch_ids()
    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe(duplex=False)
    proc = mp_ctx.Process(target=_run_trials, args=(str(ROOT), ids, child_conn))
    proc.start()
    child_conn.close()

    received: dict[str, dict[str, Any]] = {}
    stuck_id, timed_out, stuck_reason = None, False, ""
    for instance_id in ids:
        if not parent_conn.poll(PER_TRIAL_WALL_SECONDS):
            stuck_id, timed_out = instance_id, True
            break
        try:
            payload = parent_conn.recv()
        except EOFError:
            stuck_id = instance_id
            break
        if "fatal" in payload:
            _stop(proc)
            sys.exit(
                "The simulator failed to start, so no trial was scored. See the "
                "Requirements section of README.md.\n" + payload["fatal"]
            )
        received[payload["instance_id"]] = payload
    _stop(proc)

    if timed_out:
        stuck_reason = (
            f"Trial {stuck_id} did not finish within {PER_TRIAL_WALL_SECONDS:.0f} s "
            "and was killed. Once the step budget is spent, move_to and the "
            "gripper calls end the trial, so the solver most likely looped "
            "without calling them."
        )
    elif stuck_id is not None:
        stuck_reason = (
            f"The evaluator's child process exited with code {proc.exitcode} "
            f"during trial {stuck_id}, for example because solver code called "
            "os._exit() or crashed the interpreter."
        )
    results = []
    for instance_id in ids:
        if instance_id in received:
            results.append(_score_and_side_info(received[instance_id]))
        elif instance_id == stuck_id:
            results.append(_unscored(stuck_reason))
        else:
            results.append(
                _unscored(f"Not run: the evaluator stopped at trial {stuck_id}.")
            )
    print("HELIX_RESULT=" + json.dumps(results))


if __name__ == "__main__":
    evaluate()
