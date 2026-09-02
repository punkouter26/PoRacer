"""Chase-specific observation terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def target_pos_b(env: "ManagerBasedEnv", command_name: str = "target") -> torch.Tensor:
    """The chase target in the base frame, norm-clipped (3 floats). See TargetPositionCommand."""
    return env.command_manager.get_command(command_name)
