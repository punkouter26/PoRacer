"""Reward terms for Worm5 (WORM_SPEC.md "Task and reward").

Each function returns the spec's raw per-step formula. The env cfg gives each term the spec's
per-second weight; Isaac Lab's RewardManager then multiplies by weight * step_dt (0.02 s), which
is the spec's "reward per policy step, multiplied by dt = 0.02 s".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ref(env, asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    body = asset_cfg.body_ids[0] if not isinstance(asset_cfg.body_ids, slice) else 0
    return asset, body


def progress(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, clip: tuple[float, float]) -> torch.Tensor:
    """seg2 world velocity . goal direction (world +x), clipped to [-1, 2] m/s."""
    asset, body = _ref(env, asset_cfg)
    return torch.clamp(asset.data.body_link_lin_vel_w[:, body, 0], clip[0], clip[1])


def heading(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """cos(angle between seg2's horizontal forward axis and the goal).

    Forward axis = seg2 body +x (toward the head), projected onto the world xy plane and
    normalised; the goal is world +x, so the cosine is f_x / |f_xy|.
    """
    asset, body = _ref(env, asset_cfg)
    q = asset.data.body_link_quat_w[:, body]
    fwd = torch.zeros(q.shape[0], 3, device=q.device)
    fwd[:, 0] = 1.0
    f = quat_apply(q, fwd)[:, :2]
    return f[:, 0] / torch.linalg.norm(f, dim=-1).clamp_min(1e-6)


def action_rate_mean(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Mean over joints of (a_t - a_{t-1})^2, on the (clipped) actions the env received."""
    am = env.action_manager
    return torch.mean(torch.square(am.action - am.prev_action), dim=1)


def effort_mean(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, force_limit: float) -> torch.Tensor:
    """Mean over joints of (applied torque / 12)^2.

    ``applied_torque`` is the implicit actuator's clip(kp * (target - q) + kd * (0 - qdot), +/-12)
    for the last physics substep, with the per-episode randomised kp. With the default
    WORM_DAMPING_MODE=joint the drive damping kd is 0, so this is exactly MuJoCo's
    ``actuator_force`` of a <position> servo (the passive joint damping is not actuator effort).
    """
    tau = env.scene[asset_cfg.name].data.applied_torque[:, asset_cfg.joint_ids]
    return torch.mean(torch.square(tau / force_limit), dim=1)


def lateral_drift(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """abs(seg2 world velocity . world y)."""
    asset, body = _ref(env, asset_cfg)
    return torch.abs(asset.data.body_link_lin_vel_w[:, body, 1])
