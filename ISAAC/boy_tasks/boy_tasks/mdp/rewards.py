"""Chase-specific reward terms.

All of them read the live TargetPositionCommand term for the world-space goal and the
robot root state for velocity, so they are agnostic to the observation clipping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _t(x):
    return x.torch if hasattr(x, "torch") else x


def _dir_and_speed(env, command_name, asset_cfg):
    asset = env.scene[asset_cfg.name]
    term = env.command_manager.get_term(command_name)
    delta = (term.target_pos_w - _t(asset.data.root_pos_w))[:, :2]
    dist = torch.norm(delta, dim=-1)
    direction = delta / dist.clamp_min(1e-6).unsqueeze(-1)
    v_along = torch.sum(_t(asset.data.root_lin_vel_w)[:, :2] * direction, dim=-1)
    active = dist > term.cfg.reach_radius
    return direction, v_along, active, dist


def target_speed_exp(
    env: "ManagerBasedRLEnv",
    command_name: str = "target",
    target_speed: float = 1.0,
    std: float = 0.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """exp(-((v_along - target_speed)/std)^2): closing on the target at a steady pace."""
    _, v_along, active, _ = _dir_and_speed(env, command_name, asset_cfg)
    err = (v_along - target_speed) / std
    return torch.exp(-err * err) * active.float()


def target_progress(
    env: "ManagerBasedRLEnv",
    command_name: str = "target",
    max_speed: float = 1.5,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Velocity along the target direction, clipped to [-max_speed, max_speed] and normalised."""
    _, v_along, active, _ = _dir_and_speed(env, command_name, asset_cfg)
    return torch.clamp(v_along, -max_speed, max_speed) / max_speed * active.float()


def heading_to_target(
    env: "ManagerBasedRLEnv",
    command_name: str = "target",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """cos(angle between the base forward axis and the target direction), in [-1, 1]."""
    asset = env.scene[asset_cfg.name]
    direction, _, active, _ = _dir_and_speed(env, command_name, asset_cfg)
    heading = _t(asset.data.heading_w)
    fwd = torch.stack([torch.cos(heading), torch.sin(heading)], dim=-1)
    return torch.sum(fwd * direction, dim=-1) * active.float()


def targets_reached(env: "ManagerBasedRLEnv", command_name: str = "target") -> torch.Tensor:
    """Sparse +1 on the step a target is first reached."""
    term = env.command_manager.get_term(command_name)
    # reached is set in _update_command for the step it happens; time_left == 0 marks it
    return (term.reached & (term.time_left <= 0.0)).float()
