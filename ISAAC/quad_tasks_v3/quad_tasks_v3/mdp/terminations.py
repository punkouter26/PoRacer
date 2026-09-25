"""Quad terminations (QUAD_SPEC.md "Episodes", stage 1): a fall ends the episode as TERMINAL; the
20 s time-out is bootstrapped; the simulation health guard (WORM_SPEC item 12) is the worm's.

A fall (round 5) is any of, on the post-step state:
  * torso up . world up < 0.5
  * torso origin height < 0.45 m
  * the torso or ANY upper leg touches the floor: a MuJoCo floor contact with dist < 0 for that body's geom at
    ANY of the policy step's 10 substeps (raw contact, no debounce; the substep FootContactTracker records it).
    Catches kneeling on the thighs and sitting, which round 4 learned at 0.55 m.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

from worm_tasks_v3.mdp.terminations import diverged_mask

from .. import spec

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _touch_lut(env, bodies: list[str]):
    """(solver, floor geom id, lut geom -> 1 for the given bodies' geoms), cached on the env."""
    cache = getattr(env, "_quad_touch_lut", None)
    if cache is not None:
        return cache
    import isaaclab_newton.physics.newton_manager as nm  # noqa: PLC0415

    solver = nm.NewtonManager._solver
    mm = solver.mj_model
    floor = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == 0]
    if len(floor) != 1:
        raise RuntimeError(f"expected one floor geom, found {floor}")
    lut = torch.zeros(mm.ngeom, dtype=torch.bool)
    for n in bodies:
        bs = [b for b in range(mm.nbody) if mm.body(b).name.endswith("_" + n)]
        if len(bs) != 1:
            raise RuntimeError(f"cannot map body {n!r}: {bs}")
        gs = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == bs[0]]
        if not gs:
            raise RuntimeError(f"body {n!r} has no geom")
        lut[gs] = True
    env._quad_touch_lut = (solver, floor[0], lut.to(env.device))
    return env._quad_touch_lut


def floor_touch_mask(env: "ManagerBasedRLEnv", bodies: list[str] | None = None) -> torch.Tensor:
    """(num_envs,) bool: the torso or an upper leg has a floor contact with dist < 0 in the CURRENT contact
    buffer (the last substep). Diagnostics only; the fall rule uses touch_this_step (all 10 substeps)."""
    solver, floor, lut = _touch_lut(env, spec.FALL_TOUCH_BODIES if bodies is None else bodies)
    c = solver.mjw_data.contact
    geom = wp.to_torch(c.geom).long()
    world = wp.to_torch(c.worldid).long()
    dist = wp.to_torch(c.dist)
    nacon = wp.to_torch(solver.mjw_data.nacon)[:1].long()
    valid = torch.arange(geom.shape[0], device=geom.device) < nacon
    other = torch.where(geom[:, 0] == floor, geom[:, 1], torch.where(geom[:, 1] == floor, geom[:, 0], -1))
    hit = valid & (other >= 0) & lut[other.clamp_min(0)] & (dist < 0.0) & (world >= 0) & (world < env.num_envs)
    out = torch.zeros(env.num_envs + 1, dtype=torch.float32, device=geom.device)
    out.index_add_(0, torch.where(hit, world, torch.full_like(world, env.num_envs)), hit.float())
    return out[:-1] > 0


def fallen_mask(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, up_min: float, z_min: float) -> torch.Tensor:
    """up.z < up_min (0.5), torso origin height < z_min (0.45 m), or torso / upper-leg floor contact."""
    asset = env.scene[asset_cfg.name]
    body = asset_cfg.body_ids[0] if not isinstance(asset_cfg.body_ids, slice) else 0
    q = asset.data.body_link_quat_w.torch[:, body]
    z = torch.zeros(q.shape[0], 3, device=q.device)
    z[:, 2] = 1.0
    up = quat_apply(q, z)[:, 2]
    h = asset.data.body_link_pos_w.torch[:, body, 2] - env.scene.env_origins[:, 2]
    return (up < up_min) | (h < z_min) | touch_this_step(env)


def touch_this_step(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """(num_envs,) bool: torso / upper-leg floor contact at any substep of the current policy step."""
    tracker = env.scene["robot"].contact_tracker
    tracker.finish_step(env.common_step_counter)
    return tracker.touch_step


def fallen(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, up_min: float, z_min: float) -> torch.Tensor:
    """Fall termination (terminal, time_out=False: no value bootstrap)."""
    return fallen_mask(env, asset_cfg, up_min, z_min)


def time_out_healthy_standing(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, up_min: float, z_min: float,
                              max_speed: float) -> torch.Tensor:
    """Episode time-out, except for worlds that fell or diverged on that same step (those are terminal)."""
    return ((env.episode_length_buf >= env.max_episode_length)
            & ~fallen_mask(env, asset_cfg, up_min, z_min) & ~diverged_mask(env, asset_cfg.name, max_speed))
