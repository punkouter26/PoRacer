"""Observation terms for Worm5 (WORM_SPEC.md "Observation"), Isaac Lab 3.

Same formulas as the 2.3 task. Isaac Lab 3 data properties are ProxyArrays (``.torch`` gives the
zero-copy tensor) and quaternions are XYZW; ``isaaclab.utils.math`` uses XYZW too, so the maths is
unchanged. The reference frame B is the body frame of segment 2 (``seg2``); velocities are those of
the body frame's origin, which for seg2 is also its centre of mass (MJCF ipos = 0).
Scales are applied by ``ObsTerm(scale=...)`` in the env cfg, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _ref(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg):
    asset = env.scene[asset_cfg.name]
    body = asset_cfg.body_ids[0] if not isinstance(asset_cfg.body_ids, slice) else 0
    return asset, body


def ref_gravity_b(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Unit gravity direction (0, 0, -1) expressed in B. (3)"""
    asset, body = _ref(env, asset_cfg)
    q = asset.data.body_link_quat_w.torch[:, body]
    g = torch.zeros(q.shape[0], 3, device=q.device)
    g[:, 2] = -1.0
    return quat_apply_inverse(q, g)


def ref_lin_vel_b(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Segment-2 linear velocity in B [m/s]. (3)"""
    asset, body = _ref(env, asset_cfg)
    d = asset.data
    return quat_apply_inverse(d.body_link_quat_w.torch[:, body], d.body_link_lin_vel_w.torch[:, body])


def ref_ang_vel_b(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Segment-2 angular velocity in B [rad/s]. (3)"""
    asset, body = _ref(env, asset_cfg)
    d = asset.data
    return quat_apply_inverse(d.body_link_quat_w.torch[:, body], d.body_link_ang_vel_w.torch[:, body])


def joint_pos_ordered(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions [rad] in the spec's action order (straight worm = 0). (8)"""
    return env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]


def joint_vel_ordered(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint velocities [rad/s] in the spec's action order. (8)"""
    return env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]


def goal_dir_b(env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Goal direction (world +x) expressed in B, keep (x, y), renormalised to unit length. (2)

    WORM_SPEC.md item 1: rotate world +x into B with B's FULL orientation, drop z, normalise.
    """
    asset, body = _ref(env, asset_cfg)
    q = asset.data.body_link_quat_w.torch[:, body]
    goal = torch.zeros(q.shape[0], 3, device=q.device)
    goal[:, 0] = 1.0
    g_b = quat_apply_inverse(q, goal)[:, :2]
    return g_b / torch.linalg.norm(g_b, dim=-1, keepdim=True).clamp_min(1e-6)
