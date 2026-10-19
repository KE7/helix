"""Benchmark tests for examples/maniskill_pick_place.

They live outside the example because they import the reference policy, which
the mutation agent must not see. Run from the repository root:

    uv run --no-project --python 3.12 --with mani_skill==3.0.1 --with pin \
        --with pytest pytest docs/experiments/maniskill-pick-place -q
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2] / "examples" / "maniskill_pick_place"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

import solver  # noqa: E402
from reference_policy import solve as reference_solve  # noqa: E402
from sim import Ctx, ManiSkillTrial  # noqa: E402
from sim.env import MAX_ENV_STEPS, BudgetExhausted  # noqa: E402
from sim.tasks import FAMILIES, make_instance  # noqa: E402


def _split_ids(split: str) -> list[str]:
    return sorted(p.stem for p in (ROOT / "splits" / split).glob("*.json"))


TRAIN_IDS, VAL_IDS = _split_ids("train"), _split_ids("val")
ALL_IDS = TRAIN_IDS + VAL_IDS
# Trials 6-25 were never seen during the recorded run.
HELDOUT_IDS = [f"{family}__{trial}" for family in FAMILIES for trial in range(6, 26)]


def test_instance_seeds_are_stable_and_unique() -> None:
    assert make_instance("pickcube__0").seed == 580467845  # crc32, not hash()
    seeds = {make_instance(i).seed for i in ALL_IDS}
    assert len(seeds) == len(ALL_IDS)


def test_splits_are_disjoint() -> None:
    assert len(TRAIN_IDS) == len(VAL_IDS) == 9
    assert not set(TRAIN_IDS) & set(VAL_IDS)


def test_bad_instance_id_rejected() -> None:
    with pytest.raises(ValueError):
        make_instance("flycube__0")
    with pytest.raises(ValueError):
        make_instance("pickcube")
    with pytest.raises(ValueError):
        make_instance("pickcube__x")


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


@pytest.mark.parametrize("instance_id", ALL_IDS)
def test_reference_policy_solves_every_instance(instance_id: str) -> None:
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


def test_primitives_end_the_solver_once_the_budget_is_spent() -> None:
    trial = ManiSkillTrial("pickcube__0")
    with pytest.raises(BudgetExhausted):
        while True:
            trial.move_to(-0.2, 0.1, 0.2)
            trial.move_to(-0.2, -0.1, 0.2)
    assert trial.steps_used == MAX_ENV_STEPS


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


def _run_evaluator(workspace: Path, ids: list[str]) -> subprocess.CompletedProcess:
    (workspace / "helix_batch.json").write_text(json.dumps(ids))
    return subprocess.run(
        [sys.executable, "evaluate.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
        timeout=900,
    )


def _evaluate(workspace: Path, ids: list[str]) -> list[list[Any]]:
    """Run evaluate.py on *ids*; return the parsed HELIX_RESULT payload."""
    proc = _run_evaluator(workspace, ids)
    assert proc.returncode == 0, proc.stderr
    result_lines = [
        line for line in proc.stdout.splitlines() if line.startswith("HELIX_RESULT=")
    ]
    assert len(result_lines) == 1, proc.stdout
    return json.loads(result_lines[0].removeprefix("HELIX_RESULT="))


def test_evaluator_protocol(tmp_path: Path) -> None:
    ids = ["pushcube__0", "pickcube__0"]
    payload = _evaluate(_workspace(tmp_path), ids)
    assert len(payload) == len(ids)
    for score, side_info in payload:
        assert 0.0 <= score <= 1.0
        assert set(side_info["scores"]) == {"reward", "success"}
        assert side_info["feedback"]


def test_recorded_evolved_solver_reproduces(tmp_path: Path) -> None:
    """The g1-s1 solver in evolved/ reproduces its recorded results."""
    workspace = _workspace(tmp_path, HERE / "evolved" / "solver")
    ids = ALL_IDS + HELDOUT_IDS
    payload = _evaluate(workspace, ids)
    failed = {
        i
        for i, (_score, info) in zip(ids, payload, strict=True)
        if info["scores"]["success"] < 1
    }
    assert not failed, f"recorded solver failed on {sorted(failed)}"


def test_solver_output_cannot_reach_the_result_line(tmp_path: Path) -> None:
    solver_dir = tmp_path / "solver"
    solver_dir.mkdir()
    (solver_dir / "__init__.py").write_text(
        textwrap.dedent(
            """\
            import os
            import sys

            FAKE = "HELIX_RESULT=[[1.0, {}]]\\n"


            def run(ctx):
                print(FAKE)
                os.write(1, FAKE.encode())
                sys.__stdout__.write(FAKE)
                sys.__stdout__.flush()
            """
        )
    )
    payload = _evaluate(_workspace(tmp_path, solver_dir), ["pickcube__0"])
    assert payload[0][0] < 1.0


def test_evaluator_exits_when_the_simulator_cannot_start(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path)
    (workspace / "sim" / "env.py").write_text(
        "raise ImportError('simulated missing system library')\n"
    )
    proc = _run_evaluator(workspace, ["pickcube__0"])
    assert proc.returncode != 0
    assert "HELIX_RESULT=" not in proc.stdout
    assert "simulated missing system library" in proc.stderr
