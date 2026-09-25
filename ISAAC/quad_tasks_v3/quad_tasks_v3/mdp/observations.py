"""Quad-only observation terms (QUAD_SPEC.md "Observation"); the torso-frame terms are the worm's."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def joint_pos_rel_ordered(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions minus the rest pose [rad], in the spec's action order. (8)

    The quad's rest pose is 0 on every joint (quad_rig.json restPose), so this equals q; the scale
    1 / 0.785398 is applied by the ObsTerm.
    """
    d = env.scene[asset_cfg.name].data
    return d.joint_pos.torch[:, asset_cfg.joint_ids] - d.default_joint_pos.torch[:, asset_cfg.joint_ids]


def target_speed_obs(env: "ManagerBasedEnv", value: float) -> torch.Tensor:
    """The observed target speed: 1.49 / 2 = 0.745, constant. (1)"""
    return torch.full((env.num_envs, 1), value, device=env.device)
