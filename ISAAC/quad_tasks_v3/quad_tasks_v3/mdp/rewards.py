"""Quad-only reward terms (QUAD_SPEC.md "Task and reward"). Heading, lateral drift, effort and action
rate are the worm port's functions (identical formulas with "segment 2" = torso).

Each function returns the raw per-step formula; the env cfg gives it the spec's per-second weight and
the RewardManager multiplies by weight * step_dt (0.02 s). Worlds ended by the health guard this step get
zero (worm ``_guard``, WORM_SPEC item 12).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply

from worm_tasks_v3.mdp.rewards import _guard

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg


def _body(asset_cfg: SceneEntityCfg) -> int:
    return asset_cfg.body_ids[0] if not isinstance(asset_cfg.body_ids, slice) else 0


def speed_tracking(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, target: float, sigma: float) -> torch.Tensor:
    """exp(-((v_x - 1.49) / 0.5)^2), two-sided; v_x = torso world linear velocity along +x (the goal)."""
    vx = env.scene[asset_cfg.name].data.body_link_lin_vel_w.torch[:, _body(asset_cfg), 0]
    return _guard(env, torch.exp(-torch.square((vx - target) / sigma)))


def alive(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, up_min: float, z_min: float) -> torch.Tensor:
    """1 on every policy step that did not end in a fall (round 6); 0 on the falling step."""
    from .terminations import fallen_mask  # noqa: PLC0415

    return _guard(env, (~fallen_mask(env, asset_cfg, up_min, z_min)).float())


def progress(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, target: float) -> torch.Tensor:
    """clip(v_x, 0, 1.49) / 1.49 (QUAD_SPEC round 4); v_x = torso origin world velocity along +x."""
    vx = env.scene[asset_cfg.name].data.body_link_lin_vel_w.torch[:, _body(asset_cfg), 0]
    return _guard(env, torch.clamp(vx, 0.0, target) / target)


class feet_air_time(ManagerTermBase):
    """Feet air time (QUAD_SPEC rounds 5-6): at each foot's DEBOUNCED touchdown add min(t_air, 0.5 s) - 0.25 s;
    the policy step's sum pays only if the POST-STEP torso v_x > 0.3 m/s (the speed term's v_x), gated once
    per policy step.

    PER FOOTFALL, not per second. Contact state flips only after 3 consecutive substeps (15 ms) in the new
    state; everything runs at SUBSTEP resolution (5 ms) in the articulation's FootContactTracker
    (contact_tracker.py). This term processes the step's last substep, takes the sum of this policy step's
    touchdown terms, and returns it / step_dt so that the RewardManager's weight * step_dt scaling gives exactly
    weight x sum. Reset: t_air 0, debounced state = contact.
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.tracker = env.scene[cfg.params["asset_cfg"].name].contact_tracker
        if self.tracker is None:
            raise RuntimeError("feet_air_time needs the QuadArticulation contact tracker")

    def reset(self, env_ids=None):
        self.tracker.reset(slice(None) if env_ids is None else env_ids)

    def __call__(self, env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, torso_cfg: SceneEntityCfg,
                 vx_min: float) -> torch.Tensor:
        self.tracker.finish_step(env.common_step_counter)
        vx = env.scene[torso_cfg.name].data.body_link_lin_vel_w.torch[:, _body(torso_cfg), 0]
        return _guard(env, self.tracker.take_reward() * (vx > vx_min).float() / env.step_dt)


def flight(env: "ManagerBasedRLEnv", decimation: int) -> torch.Tensor:
    """Fraction of the policy step's physics steps with no foot down in the DEBOUNCED contact state (round 7).

    Per second: the manager multiplies weight (-1.0) x step_dt, so a step fully in flight costs 1.0 x 0.05.
    """
    tracker = env.scene["robot"].contact_tracker
    tracker.finish_step(env.common_step_counter)
    return _guard(env, tracker.flight_step / float(decimation))


def vertical_bounce(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """(torso world v_z)^2 [(m/s)^2] (QUAD_SPEC round 3: stops the bounding gait)."""
    vz = env.scene[asset_cfg.name].data.body_link_lin_vel_w.torch[:, _body(asset_cfg), 2]
    return _guard(env, torch.square(vz))


def upright(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """torso up (body z axis) . world up."""
    q = env.scene[asset_cfg.name].data.body_link_quat_w.torch[:, _body(asset_cfg)]
    z = torch.zeros(q.shape[0], 3, device=q.device)
    z[:, 2] = 1.0
    return _guard(env, quat_apply(q, z)[:, 2])


class foot_slip(ManagerTermBase):
    """Sum over lower legs in floor contact of the foot point's horizontal speed [m/s].

    quad_rig.json "footSlip" (the contract both trainers implement): the foot point is the centre of
    the lower capsule's bottom end-cap (footPointsLocal, lower-leg body frame);
    v_foot = v_body + w x (R p), horizontal = world (x, y). "In contact" = MuJoCo lists a floor contact
    for that lower-leg geom with dist < 0. Read here straight from MuJoCo-Warp's contact buffer (the
    last substep's collision pass, exactly what CPU MuJoCo's ``d.contact`` holds after ``mj_step``),
    fully on the GPU - no host sync, so it also works inside Newton's graphed decimation.

    ``self.contact`` (num_envs, 4) keeps the last contact flags for logging / evaluation.
    """

    def __init__(self, cfg: "RewardTermCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self._bound = False
        self.contact = torch.zeros(env.num_envs, 4, dtype=torch.bool, device=env.device)

    def _bind(self, env, asset_cfg: SceneEntityCfg, foot_points):
        import isaaclab_newton.physics.newton_manager as newton_manager_module  # noqa: PLC0415

        solver = newton_manager_module.NewtonManager._solver
        if solver is None or not hasattr(solver, "mjw_data"):
            raise RuntimeError("foot_slip needs Newton's SolverMuJoCo (MuJoCo-Warp) with use_mujoco_contacts")
        self.solver = solver
        mm = solver.mj_model
        ids = asset_cfg.body_ids
        self.body_ids = torch.tensor(list(range(self.asset.num_bodies))[ids] if isinstance(ids, slice) else [int(i) for i in ids],
                                     device=env.device, dtype=torch.long)
        lut = torch.full((mm.ngeom,), -1, dtype=torch.long)
        floor = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == 0]
        if len(floor) != 1:
            raise RuntimeError(f"expected one world (floor) geom in the MuJoCo model, found {floor}")
        self.floor = floor[0]
        for slot, bi in enumerate(self.body_ids.tolist()):
            n = self.asset.body_names[bi]
            geoms = [g for g in range(mm.ngeom) if mm.body(int(mm.geom_bodyid[g])).name.endswith("_" + n)]
            if len(geoms) != 1:
                raise RuntimeError(f"cannot map lower leg {n!r} to one MuJoCo geom: {geoms}")
            lut[geoms[0]] = slot
        self.lut = lut.to(env.device)
        self.p_local = torch.tensor(foot_points, device=env.device, dtype=torch.float32)  # (4, 3)
        self._bound = True

    def contact_flags(self, env) -> torch.Tensor:
        """(num_envs, n_legs) bool: lower-leg geom has a floor contact with dist < 0 (last substep)."""
        c = self.solver.mjw_data.contact
        geom = wp.to_torch(c.geom).long()       # (naconmax, 2)
        world = wp.to_torch(c.worldid).long()   # (naconmax,)
        dist = wp.to_torch(c.dist)              # (naconmax,)
        nacon = wp.to_torch(self.solver.mjw_data.nacon)[:1].long()
        valid = torch.arange(geom.shape[0], device=geom.device) < nacon
        g0, g1 = geom[:, 0], geom[:, 1]
        other = torch.where(g0 == self.floor, g1, g0)
        is_floor = (g0 == self.floor) | (g1 == self.floor)
        slot = self.lut[other.clamp(0, self.lut.numel() - 1)]
        hit = valid & is_floor & (slot >= 0) & (dist < 0.0) & (world >= 0) & (world < env.num_envs)
        n_leg = self.p_local.shape[0]
        idx = (world.clamp(0, env.num_envs - 1) * n_leg + slot.clamp(0, n_leg - 1))
        flags = torch.zeros(env.num_envs * n_leg, device=geom.device)
        flags.index_add_(0, idx, hit.float())
        return flags.view(env.num_envs, n_leg) > 0

    def foot_velocity_w(self) -> torch.Tensor:
        """(num_envs, n_legs, 3) world velocity of each foot point."""
        d = self.asset.data
        q = d.body_link_quat_w.torch[:, self.body_ids]            # (N, 4, 4) xyzw
        v = d.body_link_lin_vel_w.torch[:, self.body_ids]         # (N, 4, 3) at the body frame origin
        w = d.body_link_ang_vel_w.torch[:, self.body_ids]
        r = quat_apply(q.reshape(-1, 4), self.p_local.expand(q.shape[0], -1, -1).reshape(-1, 3)).view_as(v)
        return v + torch.cross(w, r, dim=-1)

    def __call__(self, env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, foot_points) -> torch.Tensor:
        if not self._bound:
            self._bind(env, asset_cfg, foot_points)
        self.contact = self.contact_flags(env)
        speed_xy = torch.linalg.norm(self.foot_velocity_w()[..., :2], dim=-1)
        return _guard(env, (self.contact.float() * speed_xy).sum(dim=1))
