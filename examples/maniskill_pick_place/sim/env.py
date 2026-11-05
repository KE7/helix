"""ManiSkill trial adapter: solver primitives and scoring.

The primitives drive ManiSkill's ``pd_ee_delta_pose`` controller. A trial is
scored from the env's normalized dense reward and success flag, which the
solver never sees.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from sim.tasks import make_instance

MAX_ENV_STEPS = 400  # control-step budget per trial
MOVE_TOLERANCE = 0.01  # m
MOVE_MAX_STEPS = 120  # per move_to call
GRIPPER_STEPS = 12  # per open/close call
SETTLE_STEPS = 40  # zero-motion steps in finish() before judging
ACTION_POS_SCALE = 0.1  # position error (m) that saturates the delta action
# A step that moves the TCP less than this (m) while far from the target means
# the controller found no IK solution for it; move_to then shortens its steps.
STALL_DISTANCE = 0.0005
MIN_STEP = 0.005  # m, the shortest step move_to falls back to
STALL_STEPS = 3  # stalled steps at MIN_STEP before move_to gives up

CUBE_HALF_SIZE = 0.02  # m
# The env success distances (PushCube goal_radius, PickCube goal_thresh), shown to
# solvers as goal.threshold. StackCube checks offsets from cube B instead.
GOAL_THRESH = {"pushcube": 0.1, "pickcube": 0.025}
# Workspace the solver may request (Panda tabletop reach, m).
WORKSPACE_X = (-0.6, 0.25)
WORKSPACE_Y = (-0.45, 0.45)
WORKSPACE_Z = (0.01, 0.5)

MAX_PARTIAL_REWARD = 0.95  # failed trials score strictly below success
# Feedback stages only; these do not affect the score.
REACH_RADIUS = 0.05  # TCP-to-cube distance that counts as reached
PROGRESS_RATIO = 0.9  # object must close 10% of its start distance to the goal


@functools.cache
def make_env(env_id: str) -> Any:
    """Headless CPU env, built once per process."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401  (registers env ids)

    return gym.make(
        env_id,
        obs_mode="state_dict",
        control_mode="pd_ee_delta_pose",
        reward_mode="normalized_dense",
        render_mode=None,
        render_backend="none",  # nothing is rendered; see sim/__init__.py
        max_episode_steps=MAX_ENV_STEPS + SETTLE_STEPS,  # the longest a trial runs
    )


def _rounded(xyz: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in xyz]


def _flag(info: dict[str, Any], key: str) -> bool:
    v = info.get(key, False)
    return bool(v.reshape(-1)[0].item()) if isinstance(v, torch.Tensor) else bool(v)


class BudgetExhausted(BaseException):
    """Raised by a primitive called after the step budget is spent.

    It derives from BaseException so that a solver's ``except Exception``
    cannot swallow it; evaluate.py catches it and scores the trial as it stands.
    """


@dataclass
class TrialResult:
    """Outcome of one scored rollout."""

    reward: float
    success: bool
    stage: str
    hint: str
    diagnostics: dict[str, Any]


class ManiSkillTrial:
    """One deterministic episode of a ManiSkill tabletop task."""

    def __init__(self, instance_id: str) -> None:
        self.instance = make_instance(instance_id)
        self.env = make_env(self.instance.env_id)
        # Rebuild the scene on every reset; a plain reset keeps physics state
        # from the previous trial, which changes how this one plays out.
        self._obs, self._info = self.env.reset(
            seed=self.instance.seed, options={"reconfigure": True}
        )
        self.steps_used = 0
        self.primitive_calls = 0
        self._gripper_cmd = 1.0  # open
        self._best_reward = 0.0
        self._last_reward = 0.0
        self._ever_grasped = False
        self._closest_tcp_obj = float("inf")
        self._closest_obj_goal = float("inf")
        self._start_obj_goal = self._obj_goal_dist()
        self._track()

    def _pos(self, key: str) -> np.ndarray:
        return np.array(self._obs["extra"][key].reshape(-1)[:3].tolist())

    def _tcp_pos(self) -> np.ndarray:
        return self._pos("tcp_pose")

    def _obj_pos(self) -> np.ndarray:
        return self._pos(
            "cubeA_pose" if self.instance.family == "stackcube" else "obj_pose"
        )

    def _goal_pos(self) -> np.ndarray:
        if self.instance.family == "stackcube":
            x, y, z = self._pos("cubeB_pose")
            return np.array([x, y, z + 2 * CUBE_HALF_SIZE])  # on top of cube B
        return self._pos("goal_pos")

    def _obj_goal_dist(self) -> float:
        if self.instance.family == "pushcube":
            return float(np.linalg.norm((self._obj_pos() - self._goal_pos())[:2]))
        return float(np.linalg.norm(self._obj_pos() - self._goal_pos()))

    def _grasped(self) -> bool:
        if self.instance.family == "pickcube":
            return _flag(self._obs["extra"], "is_grasped")
        if self.instance.family == "stackcube":
            return _flag(self._info, "is_cubeA_grasped")
        return False

    def _track(self) -> None:
        self._ever_grasped |= self._grasped()
        tcp_obj = float(np.linalg.norm(self._tcp_pos() - self._obj_pos()))
        self._closest_tcp_obj = min(self._closest_tcp_obj, tcp_obj)
        self._closest_obj_goal = min(self._closest_obj_goal, self._obj_goal_dist())

    def _step(self, pos_delta: np.ndarray) -> None:
        action = np.zeros(7, dtype=np.float32)
        action[:3] = np.clip(pos_delta / ACTION_POS_SCALE, -1.0, 1.0)
        action[6] = self._gripper_cmd
        self._obs, reward, _, _, self._info = self.env.step(action)
        self.steps_used += 1
        self._last_reward = float(reward)
        self._best_reward = max(self._best_reward, self._last_reward)
        self._track()

    def _budget_left(self) -> int:
        return MAX_ENV_STEPS - self.steps_used

    def _start_primitive(self) -> None:
        if self._budget_left() <= 0:
            raise BudgetExhausted
        self.primitive_calls += 1

    def _exhausted_reply(self) -> dict[str, Any]:
        return {"status": "budget_exhausted", "position": _rounded(self._tcp_pos())}

    def _set_gripper(self, cmd: float) -> bool:
        """Hold position for GRIPPER_STEPS under a new gripper command.

        Returns False if the step budget ran out.
        """
        self._start_primitive()
        self._gripper_cmd = cmd
        for _ in range(GRIPPER_STEPS):
            if self._budget_left() <= 0:
                return False
            self._step(np.zeros(3))
        return True

    def get_observation(self) -> dict[str, Any]:
        """Current world state as plain Python data."""
        obs: dict[str, Any] = {
            "tcp": {"position": _rounded(self._tcp_pos())},
            "gripper": {
                "command": "open" if self._gripper_cmd > 0 else "closed",
                "grasped": self._grasped(),
            },
            "workspace": {
                "x_range": list(WORKSPACE_X),
                "y_range": list(WORKSPACE_Y),
                "z_range": list(WORKSPACE_Z),
            },
            "steps_left": self._budget_left(),
        }
        if self.instance.family == "stackcube":
            obs["objects"] = {
                "cube_a": {
                    "position": _rounded(self._pos("cubeA_pose")),
                    "half_size": CUBE_HALF_SIZE,
                },
                "cube_b": {
                    "position": _rounded(self._pos("cubeB_pose")),
                    "half_size": CUBE_HALF_SIZE,
                },
            }
            obs["goal"] = {
                "description": "cube A resting on top of cube B, gripper released",
            }
        else:
            obs["objects"] = {
                "cube": {
                    "position": _rounded(self._obj_pos()),
                    "half_size": CUBE_HALF_SIZE,
                }
            }
            obs["goal"] = {
                "position": _rounded(self._pos("goal_pos")),
                "threshold": GOAL_THRESH[self.instance.family],
            }
        return obs

    def move_to(self, x: float, y: float, z: float) -> dict[str, Any]:
        """Drive the tool-center-point to (x, y, z) with a closed-loop move."""
        self._start_primitive()
        target = np.array(
            [
                float(np.clip(x, *WORKSPACE_X)),
                float(np.clip(y, *WORKSPACE_Y)),
                float(np.clip(z, *WORKSPACE_Z)),
            ]
        )
        # Near the edge of the arm's reach the controller cannot solve IK for
        # a long step and leaves the arm where it is, even though a shorter
        # step would still move it. So after a step that does not move the
        # TCP, cap the step length and halve the cap on each further stall,
        # then double it again after each step that moves, so one brief stall
        # (contact, brushing the table) does not slow the rest of the move.
        # If even MIN_STEP steps do not move it, the arm is as close as it
        # can get (or is blocked), and the move ends.
        step_cap: float | None = None
        stalls_at_min = 0
        for _ in range(MOVE_MAX_STEPS):
            if self._budget_left() <= 0:
                return self._exhausted_reply()
            start = self._tcp_pos()
            err = target - start
            dist = float(np.linalg.norm(err))
            if dist < MOVE_TOLERANCE:
                break
            if step_cap is not None and dist > step_cap:
                err = err * (step_cap / dist)
            self._step(err)
            if float(np.linalg.norm(self._tcp_pos() - start)) >= STALL_DISTANCE:
                stalls_at_min = 0
                if step_cap is not None:
                    step_cap = step_cap * 2
                    if step_cap >= ACTION_POS_SCALE:
                        step_cap = None
            elif step_cap == MIN_STEP:
                stalls_at_min += 1
                if stalls_at_min >= STALL_STEPS:
                    break
            else:
                cap = min(dist, ACTION_POS_SCALE) if step_cap is None else step_cap
                step_cap = max(cap / 2, MIN_STEP)
        err_norm = float(np.linalg.norm(target - self._tcp_pos()))
        if err_norm < MOVE_TOLERANCE:
            status = "converged"
        elif stalls_at_min >= STALL_STEPS:
            status = "stalled"
        else:
            status = "timeout"
        return {
            "status": status,
            "position": _rounded(self._tcp_pos()),
            "requested": _rounded(target),
        }

    def close_gripper(self) -> dict[str, Any]:
        """Close the gripper and report whether the target object is grasped."""
        if not self._set_gripper(-1.0):
            return self._exhausted_reply()
        return {"status": "closed", "grasped": self._grasped()}

    def open_gripper(self) -> dict[str, Any]:
        """Open the gripper fully."""
        if not self._set_gripper(1.0):
            return self._exhausted_reply()
        return {"status": "open"}

    def finish(self) -> TrialResult:
        """Hold the arm still for SETTLE_STEPS, then score the final state.

        Both success and partial credit come from the last step, so a cube
        that slides through the goal and out again, or a stack that topples
        while settling, is judged where it ends up.
        """
        for _ in range(SETTLE_STEPS):
            self._step(np.zeros(3))
        success = _flag(self._info, "success")
        reward = (
            1.0 if success else min(round(self._last_reward, 4), MAX_PARTIAL_REWARD)
        )
        stage = "success" if success else self._stage()
        diagnostics = {
            "env_info_flags": {
                k: _flag(self._info, k)
                for k, v in self._info.items()
                if k != "elapsed_steps" and isinstance(v, torch.Tensor)
            },
            "ever_grasped": self._ever_grasped,
            "closest_tcp_to_object_m": round(self._closest_tcp_obj, 4),
            "closest_object_to_goal_m": round(self._closest_obj_goal, 4),
            "final_object_to_goal_m": round(self._obj_goal_dist(), 4),
            "best_reward_seen": round(self._best_reward, 4),
            "final_object_position": _rounded(self._obj_pos()),
            "steps_used": self.steps_used,
            "primitive_calls": self.primitive_calls,
        }
        return TrialResult(reward, success, stage, self._hint(stage), diagnostics)

    def _stage(self) -> str:
        moved = self._closest_obj_goal <= PROGRESS_RATIO * self._start_obj_goal
        if self.instance.family == "pushcube":
            if self._closest_obj_goal < GOAL_THRESH["pushcube"]:
                return "overshot_goal"
            return "pushed_partway" if moved else "never_moved_object"
        if not self._ever_grasped:
            if self._closest_tcp_obj < REACH_RADIUS:
                return "reached_not_grasped"
            return "never_reached"
        return "transported_not_finished" if moved else "grasped_no_transport"

    def _hint(self, stage: str) -> str:
        if stage == "success":
            return "Task solved."
        if stage == "never_moved_object":
            return (
                "The cube never moved toward the goal. Close the gripper to make "
                "a pusher, position the TCP just behind the cube (opposite side "
                "from the goal) at cube height, then move through it toward the "
                "goal in small segments, re-observing between pushes."
            )
        if stage == "overshot_goal":
            return (
                f"The cube got within {self._closest_obj_goal:.3f} m of the goal "
                f"but ended {self._obj_goal_dist():.3f} m away (needs under "
                f"{GOAL_THRESH['pushcube']} m): it passed through the goal region "
                "and slid past it. Use shorter final pushes and stop once it is "
                "inside the threshold."
            )
        if stage == "pushed_partway":
            return (
                f"The cube got within {self._closest_obj_goal:.3f} m of the goal "
                f"(needs under {GOAL_THRESH['pushcube']} m). Keep pushing in shorter "
                "segments and re-observe after each push; approach from the side "
                "opposite the goal."
            )
        if stage == "never_reached":
            return (
                f"The TCP never came within {REACH_RADIUS} m of the cube (closest "
                f"{self._closest_tcp_obj:.3f} m). Move above the cube's xy at a safe "
                "height, open the gripper, then descend to the cube's z before "
                "closing."
            )
        if stage == "reached_not_grasped":
            return (
                "The TCP reached the cube but never grasped it. Open before "
                "descending, descend so the fingers straddle the cube (TCP z "
                "roughly at the cube center), close, and confirm 'grasped' in "
                "the observation before lifting."
            )
        if stage == "grasped_no_transport":
            return (
                "The cube was grasped but never carried toward the goal. While "
                "holding it, move_to the goal position; for stacking, hover "
                "above cube B, descend until cube A rests on it, then open."
            )
        if self.instance.family == "pickcube":
            return (
                f"The cube got within {self._closest_obj_goal:.3f} m of the goal "
                f"(needs {GOAL_THRESH['pickcube']} m) but the final state does not "
                "pass. Hold the cube still at the goal point."
            )
        return (
            f"Cube A got within {self._closest_obj_goal:.3f} m of its place on top "
            "of cube B, but the stack check failed. It needs cube A centered on "
            "cube B within about 3.3 cm, resting on it within 5 mm, released, and "
            "still. Release just above cube B and retreat upward."
        )


class Ctx:
    """The solver-facing context: task metadata plus the four primitives."""

    def __init__(self, trial: ManiSkillTrial) -> None:
        self._trial = trial
        self.task_family = trial.instance.family
        self.instruction = trial.instance.instruction

    def get_observation(self) -> dict[str, Any]:
        return self._trial.get_observation()

    def move_to(self, x: float, y: float, z: float) -> dict[str, Any]:
        return self._trial.move_to(x, y, z)

    def open_gripper(self) -> dict[str, Any]:
        return self._trial.open_gripper()

    def close_gripper(self) -> dict[str, Any]:
        return self._trial.close_gripper()
