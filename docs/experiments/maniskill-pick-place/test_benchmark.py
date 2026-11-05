"""Benchmark tests for examples/maniskill_pick_place.

They live outside the example because they import the reference policy, which
the mutation agent must not see. Run from the repository root:

    uv run --no-project --python 3.12 \
        --with-requirements examples/maniskill_pick_place/requirements.txt \
        --with pytest pytest docs/experiments/maniskill-pick-place -q
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "examples" / "maniskill_pick_place"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import check_server  # noqa: E402
import solver  # noqa: E402
from reference_policy import solve as reference_solve  # noqa: E402
from sim import Ctx, ManiSkillTrial  # noqa: E402
from sim.env import MAX_ENV_STEPS, SETTLE_STEPS, BudgetExhausted  # noqa: E402
from sim.tasks import FAMILIES, make_instance  # noqa: E402


def _split_ids(split: str) -> list[str]:
    return sorted(p.stem for p in (ROOT / "splits" / split).glob("*.json"))


TRAIN_IDS, VAL_IDS = _split_ids("train"), _split_ids("val")
ALL_IDS = TRAIN_IDS + VAL_IDS
# Drawn once with random.Random(20261027).sample(range(1_000_000, 10_000_000), 30)
# and kept out of the example, so an agent cannot target the held-out trials.
HELDOUT_TRIALS = [
    1431072, 1464500, 1684658, 1837208, 2315639, 2696748, 3171132, 3177901,
    3566995, 3901034, 4147175, 4198787, 4365715, 4473652, 4813802, 4907322,
    5291303, 5355432, 6348011, 6478242, 7179989, 7452036, 7840177, 7851017,
    8045647, 8070176, 8265920, 9141871, 9261966, 9300182,
]  # fmt: skip
HELDOUT_IDS = [f"{family}__{n}" for family in FAMILIES for n in HELDOUT_TRIALS]
# Trials the solver in evolved/ failed when it was recorded.
RECORDED_FAILURES: set[str] = set()


def test_instance_seeds_are_stable_and_unique() -> None:
    assert make_instance("pickcube__0").seed == 580467845  # crc32, not hash()
    seeds = {make_instance(i).seed for i in ALL_IDS}
    assert len(seeds) == len(ALL_IDS)


def test_splits_are_disjoint() -> None:
    assert len(TRAIN_IDS) == 33 and len(VAL_IDS) == 18
    assert len(set(TRAIN_IDS) | set(VAL_IDS) | set(HELDOUT_IDS)) == 33 + 18 + 90


def test_bad_instance_id_rejected() -> None:
    with pytest.raises(ValueError):
        make_instance("flycube__0")
    with pytest.raises(ValueError):
        make_instance("pickcube")
    for bad_trial in ["x", "\u00b2", "03", "-1", ""]:
        with pytest.raises(ValueError):
            make_instance(f"pickcube__{bad_trial}")


def test_reset_is_deterministic() -> None:
    a = ManiSkillTrial("pickcube__0").get_observation()
    b = ManiSkillTrial("pickcube__0").get_observation()
    assert a["objects"] == b["objects"]
    assert a["goal"] == b["goal"]


def test_move_to_converges() -> None:
    trial = ManiSkillTrial("pickcube__0")
    result = trial.move_to(-0.2, 0.1, 0.2)
    assert result["status"] == "converged", result
    assert result["position"] == pytest.approx([-0.2, 0.1, 0.2], abs=0.015)


def test_move_to_stops_at_the_edge_of_reach() -> None:
    trial = ManiSkillTrial("pushcube__4")
    trial.close_gripper()
    trial.move_to(0.0, -0.07, 0.1)
    trial.move_to(0.0, -0.07, 0.02)
    steps_before = trial.steps_used
    result = trial.move_to(0.25, -0.07, 0.02)
    # Full-length steps leave the arm frozen at about x = 0.13.
    assert result["position"][0] > 0.17, result
    assert result["status"] == "stalled", result
    assert trial.steps_used - steps_before < 60


def test_move_to_speeds_up_again_after_a_stall() -> None:
    trial = ManiSkillTrial("pickcube__0")
    real_step = trial._step
    stalls = iter(range(4))

    def step(pos_delta):  # the first 4 steps do not move the arm
        if next(stalls, None) is None:
            real_step(pos_delta)
        else:
            trial.steps_used += 1

    trial._step = step
    result = trial.move_to(-0.2, 0.1, 0.2)
    # Without growing back, the step cap would stay near 6 mm for the 20 cm move.
    assert result["status"] == "converged", result
    assert trial.steps_used < 20


def test_grasp_mechanics() -> None:
    trial = ManiSkillTrial("pickcube__0")
    cube = trial.get_observation()["objects"]["cube"]["position"]
    trial.open_gripper()
    trial.move_to(cube[0], cube[1], cube[2] + 0.08)
    trial.move_to(cube[0], cube[1], cube[2])
    closed = trial.close_gripper()
    assert closed["grasped"], "descend-and-close must grasp the cube"
    trial.move_to(cube[0], cube[1], cube[2] + 0.1)
    obs = trial.get_observation()
    assert obs["gripper"]["grasped"]
    assert obs["objects"]["cube"]["position"][2] > cube[2] + 0.05


def test_trial_result_does_not_depend_on_earlier_trials() -> None:
    def final_position(instance_id: str) -> list[float]:
        trial = ManiSkillTrial(instance_id)
        try:
            reference_solve(Ctx(trial))
        except BudgetExhausted:
            pass
        return trial.finish().diagnostics["final_object_position"]

    first = final_position("pushcube__10")
    final_position("pushcube__4")
    assert final_position("pushcube__10") == first


@pytest.mark.parametrize("instance_id", TRAIN_IDS)
def test_reference_policy_solves_every_train_instance(instance_id: str) -> None:
    trial = ManiSkillTrial(instance_id)
    reference_solve(Ctx(trial))
    result = trial.finish()
    assert result.success, (
        f"{instance_id}: stage={result.stage}, reward={result.reward}, "
        f"diagnostics={result.diagnostics}"
    )


@pytest.mark.parametrize("family", list(FAMILIES))
def test_seed_solver_does_not_succeed(family: str) -> None:
    trial = ManiSkillTrial(f"{family}__0")
    solver.run(Ctx(trial))
    result = trial.finish()
    assert not result.success, "the naive seed must leave headroom for evolution"


def test_success_is_judged_after_the_full_settle() -> None:
    trial = ManiSkillTrial("pickcube__0")
    reference_solve(Ctx(trial))
    steps_before = trial.steps_used
    result = trial.finish()
    assert result.success
    assert trial.steps_used == steps_before + SETTLE_STEPS


def test_a_push_through_the_goal_is_scored_where_the_cube_stops() -> None:
    trial = ManiSkillTrial("pushcube__12")
    obs = trial.get_observation()
    cube, goal = obs["objects"]["cube"]["position"], obs["goal"]["position"]
    trial.close_gripper()
    trial.move_to(cube[0] - 0.06, cube[1], 0.1)
    trial.move_to(cube[0] - 0.06, cube[1], cube[2])
    trial.move_to(goal[0] + 0.1, goal[1], cube[2])  # one hard push past the goal
    result = trial.finish()
    assert not result.success
    assert result.stage == "overshot_goal", result.hint
    assert result.diagnostics["closest_object_to_goal_m"] < 0.1
    assert result.reward < result.diagnostics["best_reward_seen"]


def test_primitives_end_the_solver_once_the_budget_is_spent() -> None:
    trial = ManiSkillTrial("pickcube__0")
    with pytest.raises(BudgetExhausted):
        while True:
            trial.move_to(-0.2, 0.1, 0.2)
            trial.move_to(-0.2, -0.1, 0.2)
    assert trial.steps_used == MAX_ENV_STEPS


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


needs_docker = pytest.mark.skipif(
    not _docker_available(), reason="the evaluator runs solvers in Docker"
)


def _workspace(tmp_path: Path, solver_dir: Path | None = None) -> Path:
    """Copy the example to tmp_path, optionally with a different solver/."""
    workspace = tmp_path / "workspace"
    skip = ["__pycache__", "helix_batch.json"] + (["solver"] if solver_dir else [])
    shutil.copytree(ROOT, workspace, ignore=shutil.ignore_patterns(*skip))
    if solver_dir:
        shutil.copytree(
            solver_dir,
            workspace / "solver",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    return workspace


def _solver(tmp_path: Path, source: str) -> Path:
    """Write a one-file solver package with *source* as its __init__.py."""
    solver_dir = tmp_path / "solver"
    solver_dir.mkdir()
    (solver_dir / "__init__.py").write_text(textwrap.dedent(source))
    return solver_dir


def _run_evaluator(
    workspace: Path,
    ids: list[str],
    *flags: str,
    timeout: float = 900,
) -> subprocess.CompletedProcess:
    """Run evaluate.py on *ids* the way helix.toml does, with python3 -I."""
    (workspace / "helix_batch.json").write_text(json.dumps(ids))
    return subprocess.run(
        [sys.executable, "-I", "evaluate.py", *flags],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _evaluate(
    workspace: Path, ids: list[str], *flags: str, timeout: float = 900
) -> list[list[Any]]:
    """Run evaluate.py on *ids*; return the parsed HELIX_RESULT payload."""
    proc = _run_evaluator(workspace, ids, *flags, timeout=timeout)
    assert proc.returncode == 0, proc.stderr
    result_lines = [
        line for line in proc.stdout.splitlines() if line.startswith("HELIX_RESULT=")
    ]
    assert len(result_lines) == 1, proc.stdout
    return json.loads(result_lines[0].removeprefix("HELIX_RESULT="))


def _solver_containers() -> list[str]:
    listed = subprocess.run(
        [
            "docker",
            "ps",
            "--all",
            "--quiet",
            "--filter",
            "label=helix-maniskill-evaluation",
        ],
        capture_output=True,
        text=True,
    )
    return listed.stdout.split()


@needs_docker
def test_evaluator_protocol(tmp_path: Path) -> None:
    ids = ["pushcube__0", "pickcube__0"]
    payload = _evaluate(_workspace(tmp_path), ids)
    assert len(payload) == len(ids)
    for score, side_info in payload:
        assert 0.0 <= score <= 1.0
        assert set(side_info["scores"]) == {"reward", "success"}
        assert side_info["feedback"]


def test_unsafe_local_mode_runs_without_docker(tmp_path: Path) -> None:
    proc = _run_evaluator(_workspace(tmp_path), ["pickcube__0"], "--unsafe-local")
    assert proc.returncode == 0, proc.stderr
    assert "WARNING: --unsafe-local" in proc.stderr
    payload = json.loads(proc.stdout.strip().removeprefix("HELIX_RESULT="))
    assert "[pickcube__0] stage=never_reached" in payload[0][1]["feedback"]


@needs_docker
@pytest.mark.timeout(600)  # 141 trials; the repo's default limit is 60 s
def test_recorded_evolved_solver_reproduces(tmp_path: Path) -> None:
    """The solver in evolved/ reproduces its recorded results."""
    workspace = _workspace(tmp_path, HERE / "evolved" / "solver")
    ids = ALL_IDS + HELDOUT_IDS
    payload = _evaluate(workspace, ids)
    failed = {
        i
        for i, (_score, info) in zip(ids, payload, strict=True)
        if info["scores"]["success"] < 1
    }
    assert failed == RECORDED_FAILURES, f"recorded solver failed on {sorted(failed)}"


@needs_docker
def test_solver_output_cannot_reach_the_result_line(tmp_path: Path) -> None:
    solver_dir = _solver(
        tmp_path,
        """\
        import os
        import sys

        FAKE = "HELIX_RESULT=[[1.0, {}]]\\n"


        def run(ctx):
            print(FAKE)
            os.write(1, FAKE.encode())
            os.write(2, FAKE.encode())
            sys.__stdout__.write(FAKE)
            sys.__stdout__.flush()
        """,
    )
    proc = _run_evaluator(_workspace(tmp_path, solver_dir), ["pickcube__0"])
    assert proc.stdout.count("HELIX_RESULT=") == 1
    assert "HELIX_RESULT" not in proc.stderr
    payload = json.loads(proc.stdout.strip().removeprefix("HELIX_RESULT="))
    assert payload[0][0] < 1.0


@needs_docker
def test_solver_container_is_isolated(tmp_path: Path) -> None:
    solver_dir = _solver(
        tmp_path,
        """\
        import importlib
        import pathlib
        import socket

        import numpy


        def must_fail(attempt, *args):
            try:
                attempt(*args)
            except Exception:
                return
            raise AssertionError(f"{attempt.__name__}{args} succeeded")


        def run(ctx):
            assert not hasattr(ctx, "_trial")
            must_fail(importlib.import_module, "sim")
            here = pathlib.Path(__file__).resolve().parents[1]
            assert not (here / "sim").exists()
            # The copy of the example, the image and the network are all off limits.
            must_fail(pathlib.Path.write_text, here / "evaluate.py", "")
            must_fail(pathlib.Path.write_text, here / "solver" / "x.py", "")
            must_fail(pathlib.Path.write_text, pathlib.Path(numpy.__file__), "")
            must_fail(socket.create_connection, ("1.1.1.1", 53), 2)
            ctx.get_observation()
        """,
    )
    payload = _evaluate(_workspace(tmp_path, solver_dir), ["pickcube__0"])
    score, info = payload[0]
    assert score < 1.0
    assert "stage=never_reached" in info["feedback"], info["feedback"]
    assert "raised" not in info["feedback"], info["feedback"]


def test_a_top_level_module_cannot_shadow_the_evaluators(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "json.py").write_text(
        "def dumps(*args, **kwargs):\n"
        '    return \'[[1.0, {"scores": {"reward": 1.0, "success": 1.0}}]]\'\n'
    )
    payload = _evaluate(workspace, ["pickcube__0"], "--unsafe-local")
    assert payload[0][0] < 1.0
    assert "stage=never_reached" in payload[0][1]["feedback"]


@needs_docker
def test_requests_are_not_unpickled(tmp_path: Path) -> None:
    marker = tmp_path / "unpickled"
    solver_dir = _solver(
        tmp_path,
        f"""\
        import os
        import pickle


        class Payload:
            def __reduce__(self):
                return (os.system, ("touch {marker}",))


        def run(ctx):
            ctx._out.write(pickle.dumps(Payload()) + b"\\n")
            ctx._out.flush()
            ctx.get_observation()
        """,
    )
    payload = _evaluate(_workspace(tmp_path, solver_dir), ["pickcube__0"])
    assert not marker.exists()
    assert "could not read" in payload[0][1]["feedback"]


@needs_docker
def test_move_to_rejects_non_finite_targets(tmp_path: Path) -> None:
    solver_dir = _solver(
        tmp_path,
        """\
        def run(ctx):
            ctx.move_to(float("nan"), 0.0, 0.2)
        """,
    )
    payload = _evaluate(_workspace(tmp_path, solver_dir), ["pickcube__0"])
    assert "move_to needs three finite numbers" in payload[0][1]["feedback"]


@needs_docker
def test_a_solver_cannot_leave_anything_running(tmp_path: Path) -> None:
    solver_dir = _solver(
        tmp_path,
        """\
        import os
        import signal
        import subprocess


        def run(ctx):
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            subprocess.Popen(["sleep", "987654"], start_new_session=True)
            if os.fork() == 0:
                os.setsid()
                os.execvp("sleep", ["sleep", "987655"])
            while True:
                pass
        """,
    )
    start = time.monotonic()
    payload = _evaluate(
        _workspace(tmp_path, solver_dir),
        ["pickcube__0"],
        "--solver-seconds",
        "5",
        timeout=120,
    )
    assert time.monotonic() - start < 60
    assert "did not finish within 5 s" in payload[0][1]["feedback"]
    assert _solver_containers() == []


def test_evaluator_exits_when_the_simulator_cannot_start(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "sim" / "env.py").write_text(
        "raise ImportError('simulated missing system library')\n"
    )
    proc = _run_evaluator(workspace, ["pickcube__0"], "--unsafe-local")
    assert proc.returncode != 0
    assert "HELIX_RESULT=" not in proc.stdout
    assert "simulated missing system library" in proc.stderr


@needs_docker
def test_a_stuck_or_exiting_solver_is_scored_as_it_stands(tmp_path: Path) -> None:
    solver_dir = _solver(
        tmp_path,
        """\
        import os


        def run(ctx):
            if ctx.task_family == "pickcube":
                while True:
                    try:
                        ctx.move_to(0.0, 0.0, 0.2)
                    except:  # also swallows BudgetExhausted
                        pass
            if ctx.task_family == "pushcube":
                os._exit(3)
        """,
    )
    ids = ["pickcube__0", "pushcube__0", "pushcube__x", "stackcube__0"]
    payload = _evaluate(_workspace(tmp_path, solver_dir), ids, "--solver-seconds", "10")
    feedback = [info["feedback"] for _, info in payload]
    assert "[pickcube__0] stage=" in feedback[0]
    assert "did not finish within 10 s" in feedback[0]
    assert "[pushcube__0] stage=" in feedback[1]
    assert "exited with code 3" in feedback[1]
    assert "stage=benchmark_error" in feedback[2]
    assert "[stackcube__0] stage=never_reached" in feedback[3]
    assert payload[3][0] > 0.0
    assert _solver_containers() == []


def test_check_service_accepts_only_training_ids() -> None:
    files = {"solver/__init__.py": "def run(ctx):\n    pass\n"}
    for ids in (["pushcube__26"], [HELDOUT_IDS[0]], ["pushcube__0", "pickcube__27"]):
        with pytest.raises(ValueError, match="only training ids"):
            check_server._check(ids, files, unsafe_local=True)
    # Every training id at once is allowed; this fails on the file, not the ids.
    with pytest.raises(ValueError, match="not a solver source file"):
        check_server._check(TRAIN_IDS, {"sim/env.py": ""}, unsafe_local=True)
    for bad in ("solver/../evaluate.py", "/etc/passwd", "sim/env.py", "solver/x.txt"):
        with pytest.raises(ValueError, match="not a solver source file"):
            check_server._check(["pushcube__0"], {bad: ""}, unsafe_local=True)
    refused = subprocess.run(
        [sys.executable, "check.py", "pushcube__26"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert refused.returncode == 2
    assert "refused" in refused.stderr


@needs_docker
def test_check_service_scores_training_trials() -> None:
    files = {
        str(path.relative_to(ROOT)): path.read_text()
        for path in (ROOT / "solver").rglob("*.py")
    }
    body = check_server._check(["pickcube__0"], files, unsafe_local=False)
    [result] = body["results"]
    assert result["id"] == "pickcube__0"
    assert "stage=never_reached" in result["side_info"]["feedback"]
