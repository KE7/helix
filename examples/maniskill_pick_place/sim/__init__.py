"""ManiSkill adapter for the maniskill_pick_place evaluator."""

import os
import platform
from pathlib import Path

_MOLTENVK_ICDS = (
    "/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json",
    "/usr/local/etc/vulkan/icd.d/MoltenVK_icd.json",
)


def _use_homebrew_moltenvk() -> None:
    """Point the Vulkan loader at Homebrew's MoltenVK on macOS.

    ManiSkill 3.0.1 needs a Vulkan device even when nothing is rendered, and
    SAPIEN's bundled loader does not search Homebrew's install prefix.
    """
    if platform.system() != "Darwin":
        return
    if os.environ.get("VK_ICD_FILENAMES") or os.environ.get("VK_DRIVER_FILES"):
        return
    for icd in _MOLTENVK_ICDS:
        if Path(icd).exists():
            # Older loaders read VK_ICD_FILENAMES, newer ones VK_DRIVER_FILES.
            os.environ["VK_ICD_FILENAMES"] = os.environ["VK_DRIVER_FILES"] = icd
            return
    raise RuntimeError("ManiSkill on macOS needs MoltenVK: brew install molten-vk")


_use_homebrew_moltenvk()

from sim.env import Ctx, ManiSkillTrial  # noqa: E402

__all__ = ["Ctx", "ManiSkillTrial"]
