"""Quad-only events (QUAD_SPEC.md "Episodes"). The per-reset friction / mass / kp randomisers are the
worm port's lean per-world writers (worm_tasks_v3.mdp.events), reused unchanged."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.managers import EventTermCfg
    from isaaclab.envs import ManagerBasedEnv


class randomize_floor_friction(ManagerTermBase):
    """Per world, floor sliding friction = base * s, s ~ U(lo, hi), drawn at every reset (floor-priority regime).

    Newton has ONE ground shape shared by all worlds, but MuJoCo-Warp's geom_friction is batched per world, so the
    value is written straight into ``mjw_model.geom_friction[world, floor, 0]`` of the reset worlds (the lean-writer
    pattern of the worm's mass / kp randomisers). No SHAPE_PROPERTIES notification is sent - Newton would answer it
    by rewriting every world's floor friction from the one shape. The floor has priority 1, so every floor contact
    (feet included) uses exactly this friction. Bound lazily on the first call (the solver does not exist when the
    event manager is built).
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._bound = False
        self.scale = torch.ones(env.num_envs, device=env.device)

    def _bind(self, env):
        import isaaclab_newton.physics.newton_manager as nm  # noqa: PLC0415
        import warp as wp  # noqa: PLC0415

        solver = nm.NewtonManager._solver
        mm = solver.mj_model
        floor = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == 0]
        if len(floor) != 1:
            raise RuntimeError(f"expected one floor geom, found {floor}")
        self.floor = floor[0]
        self.gf = wp.to_torch(solver.mjw_model.geom_friction)  # (nworld, ngeom, 3)
        if self.gf.dim() != 3 or self.gf.shape[0] != env.num_envs:
            raise RuntimeError("MuJoCo-Warp geom_friction is not batched per world")
        self._bound = True

    def __call__(self, env: "ManagerBasedEnv", env_ids: torch.Tensor | None, base: float,
                 scale_range: tuple[float, float]):
        if not self._bound:
            self._bind(env)
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device).long()
        s = torch.empty(len(ids), device=env.device).uniform_(*scale_range)
        self.scale[ids] = s
        self.gf[ids, self.floor, 0] = (base * s).to(self.gf.dtype)


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
