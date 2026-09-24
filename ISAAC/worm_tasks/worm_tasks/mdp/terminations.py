"""Terminations for Worm5: time-out (bootstrapped) and the simulation health guard.

WORM_SPEC.md item 12: a world whose state is non-finite, or with any joint or body speed above
500 (rad/s or m/s), ends as TERMINAL - no value bootstrap, zero reward for that step - and
resets. Item 13's order holds in ManagerBasedRLEnv.step: substeps -> kinematics -> episode
counter += 1 -> terminations -> rewards (post-step state) -> reset -> observations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def diverged_mask(env: "ManagerBasedRLEnv", asset_name: str = "robot", max_speed: float = 500.0) -> torch.Tensor:
    """True where the articulation state is non-finite or any speed exceeds ``max_speed``."""
    d = env.scene[asset_name].data
    parts = (d.joint_pos, d.joint_vel, d.body_link_pos_w.flatten(1), d.body_link_quat_w.flatten(1),
             d.body_link_lin_vel_w.flatten(1), d.body_link_ang_vel_w.flatten(1))
    bad = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for p in parts:
        bad |= ~torch.isfinite(p).all(dim=1)
    bad |= torch.nan_to_num(d.joint_vel, nan=0.0).abs().amax(dim=1) > max_speed
    bad |= torch.nan_to_num(torch.linalg.norm(d.body_link_lin_vel_w, dim=-1), nan=0.0).amax(dim=1) > max_speed
    bad |= torch.nan_to_num(torch.linalg.norm(d.body_link_ang_vel_w, dim=-1), nan=0.0).amax(dim=1) > max_speed
    return bad


def sim_diverged(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, max_speed: float) -> torch.Tensor:
    """Health guard termination (terminal, time_out=False)."""
    return diverged_mask(env, asset_cfg.name, max_speed)


def time_out_healthy(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, max_speed: float) -> torch.Tensor:
    """Episode time-out, except for worlds that diverged on the same step (those get no bootstrap)."""
    return (env.episode_length_buf >= env.max_episode_length) & ~diverged_mask(env, asset_cfg.name, max_speed)
