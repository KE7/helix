"""Task instances, with ids of the form ``<family>__<trial>``.

The family picks the ManiSkill env, and the CRC32 of the whole id seeds
``env.reset``, so an id always gives the same initial state.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass

# family -> (ManiSkill env id, instruction shown to the solver)
FAMILIES = {
    "pushcube": (
        "PushCube-v1",
        "Push the cube so it ends up inside the goal region on the table.",
    ),
    "pickcube": (
        "PickCube-v1",
        "Pick up the cube and hold it still at the 3-D goal position.",
    ),
    "stackcube": (
        "StackCube-v1",
        "Pick up cube A and stack it on top of cube B, then release it.",
    ),
}


@dataclass(frozen=True)
class TaskInstance:
    """One deterministic trial."""

    instance_id: str
    family: str
    env_id: str
    seed: int
    instruction: str


def make_instance(instance_id: str) -> TaskInstance:
    """Parse an instance id into its environment and reset seed.

    Raises:
        ValueError: if the id is not ``<family>__<trial>`` with a known family
            and a non-negative integer trial.
    """
    family, sep, trial = instance_id.partition("__")
    if not sep or family not in FAMILIES or not trial.isdigit():
        raise ValueError(
            f"instance id {instance_id!r} must look like '<family>__<trial>' "
            f"with family in {tuple(FAMILIES)}"
        )
    env_id, instruction = FAMILIES[family]
    return TaskInstance(
        instance_id=instance_id,
        family=family,
        env_id=env_id,
        seed=zlib.crc32(instance_id.encode("utf-8")) % (2**31),
        instruction=instruction,
    )
