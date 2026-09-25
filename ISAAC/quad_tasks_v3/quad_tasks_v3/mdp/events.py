"""Quad-only events (QUAD_SPEC.md "Episodes"). The per-reset friction / mass / kp randomisers are the
worm port's lean per-world writers (worm_tasks_v3.mdp.events), reused unchanged."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def push_horizontal(env: "ManagerBasedEnv", env_ids: torch.Tensor, speed: float,
                    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """A 0.5 m/s push in a random horizontal direction.

    Adds speed * (cos t, sin t, 0), t ~ U(0, 2 pi), to the root's (torso's) world linear velocity - the
    free joint's linear qvel - and leaves the angular and joint velocities alone, so every body gains the
    same 0.5 m/s. Run as an Isaac Lab ``interval`` event with a per-env timer U(10, 15) s that is redrawn
    at every reset and after every push (so a 20 s episode gets exactly one push, 10-15 s in).
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    vel = asset.data.root_vel_w.torch[env_ids].clone()
    theta = torch.rand(len(env_ids), device=env.device) * (2.0 * math.pi)
    vel[:, 0] += speed * torch.cos(theta)
    vel[:, 1] += speed * torch.sin(theta)
    asset.write_root_velocity_to_sim_index(root_velocity=vel, env_ids=env_ids)
