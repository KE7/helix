"""Record videos of a maniskill_pick_place solver on chosen trials.

The trial runs exactly as in the evaluator's local mode (same env, seed, controller
and primitives); the only change is that the env is built with a render camera, and
a frame is grabbed after every control step. Rendering does not touch the physics,
and the script prints each trial's reward so it can be checked against
recorded_run.json.

Run it from the example's directory with the example's pinned packages:

    cd examples/maniskill_pick_place
    uv run --no-project --python 3.12 --with-requirements requirements.txt \
        --with imageio==2.37.0 --with imageio-ffmpeg==0.6.0 \
        python ../../docs/experiments/maniskill-pick-place/render_rollouts.py \
        --solver solver --out /tmp/seed pushcube__26 pickcube__26 stackcube__26

--solver is the directory that holds the solver package (the seed is ``solver``;
the recorded run's evolved package is
``../../docs/experiments/maniskill-pick-place/evolved/solver``). Each solver runs in
its own process, since both packages are named ``solver``.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[3] / "examples" / "maniskill_pick_place"


# One caption per task naming what each colour is. PickCube's green sphere is a
# visual marker with no collision; success means the cube's centre ends within
# 2.5 cm of it. In StackCube the green cube is the base, not a marker.
CAPTIONS = {
    "pushcube": "Push the blue cube onto the red and white target",
    "pickcube": "Lift the red cube to the green sphere, which marks the goal",
    "stackcube": "Stack the red cube on the green cube",
}


def add_wrist_camera(env, size: int):
    """A camera on the Panda's hand, where ManiSkill's panda_wristcam puts its RealSense.

    The example's robot (panda_v2) has no camera link, so the camera is mounted on
    panda_hand at panda_v3's realsense_joint + camera_link_joint offset. A camera
    has no physics.
    """
    import numpy as np
    import sapien
    from mani_skill.sensors.camera import Camera, CameraConfig
    from transforms3d.euler import euler2quat

    mount = sapien.Pose([0.035, 0, 0.036], euler2quat(0, -1.5707, 3.1415, "sxyz"))
    pose = mount * sapien.Pose([0, 0.02, 0.0115])
    cfg = CameraConfig(uid="wrist", pose=pose, width=size, height=size, fov=np.pi / 2,
                       near=0.01, far=100, mount=env.agent.robot.links_map["panda_hand"])
    return Camera(cfg, env.scene)


def caption(frame, text: str):
    """Write *text* on a dark band along the bottom of *frame*."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h = img.height
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", h // 26)
    except OSError:
        font = ImageFont.load_default(size=h // 26)
    band = h // 14
    draw.rectangle([0, h - band, img.width, h], fill=(20, 20, 20))
    draw.text((img.width // 2, h - band // 2), text, fill=(240, 240, 240), font=font, anchor="mm")
    return np.asarray(img)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pip", type=float, default=0.34, help="wrist inset size as a fraction of the frame")
    ap.add_argument("--solver", required=True, help="directory of the solver package")
    ap.add_argument("--out", required=True, help="output directory for MP4s")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("trials", nargs="+")
    args = ap.parse_args()

    solver_dir = Path(args.solver).resolve()
    # The solver's parent goes first: the example directory has its own (seed) solver/.
    sys.path.insert(0, str(EXAMPLE))
    sys.path.insert(0, str(solver_dir.parent))

    import imageio.v2 as imageio
    import numpy as np

    import sim  # noqa: F401  (sets up MoltenVK on macOS)
    import sim.env as E
    import solver

    @functools.cache
    def make_env(env_id: str):
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401

        return gym.make(
            env_id,
            obs_mode="state_dict",
            control_mode="pd_ee_delta_pose",
            reward_mode="normalized_dense",
            render_mode="rgb_array",
            human_render_camera_configs=dict(width=args.width, height=args.height),
            max_episode_steps=E.MAX_ENV_STEPS + E.SETTLE_STEPS,
        )

    E.make_env = make_env

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for trial_id in args.trials:
        trial = E.ManiSkillTrial(trial_id)
        env = trial.env.unwrapped
        inset = int(args.height * args.pip) // 2 * 2
        wrist = add_wrist_camera(env, inset)
        text = CAPTIONS.get(trial.instance.family)
        frames = []

        def grab() -> None:
            frame = np.asarray(trial.env.render()).reshape(args.height, args.width, 3).copy()
            wrist.capture()
            w = wrist.get_obs(rgb=True, depth=False, position=False, segmentation=False)["rgb"]
            w = np.asarray(w).reshape(inset, inset, 3)
            b = max(3, args.height // 180)  # white border, top-left corner
            frame[: inset + 2 * b, : inset + 2 * b] = 255
            frame[b : b + inset, b : b + inset] = w
            frames.append(caption(frame, text) if text else frame)

        step = trial._step

        def recorded_step(pos_delta, _step=step):
            _step(pos_delta)
            grab()

        trial._step = recorded_step
        grab()
        try:
            solver.run(E.Ctx(trial))
        except E.BudgetExhausted:
            pass
        result = trial.finish()
        path = out / f"{trial_id}.mp4"
        imageio.mimwrite(path, frames, fps=20, macro_block_size=8)
        print(json.dumps({"trial": trial_id, "reward": round(result.reward, 4),
                          "success": result.success, "steps": len(frames) - 1, "video": str(path)}))


if __name__ == "__main__":
    main()
