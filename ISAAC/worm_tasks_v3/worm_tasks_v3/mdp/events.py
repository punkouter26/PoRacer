"""Per-episode randomisation for Worm5 on Newton / MuJoCo-Warp (WORM_SPEC.md items 7, 14, 15).

* friction: the spec scales ONE coefficient per episode (MuJoCo has a single sliding friction per
  contact), while Isaac Lab's ``randomize_rigid_body_material`` samples every shape independently.
* segment mass and kp: Isaac Lab's ``randomize_rigid_body_mass`` / ``randomize_actuator_gains`` work on
  Newton, but every call notifies BODY_INERTIAL_PROPERTIES / JOINT_DOF_PROPERTIES, and Newton's
  SolverMuJoCo answers each with MuJoCo-Warp ``set_const`` over ALL worlds (~19 ms per call at 4096
  worlds). Resets are staggered, so that ran on every policy step and halved the throughput. The lean
  terms below write the new values for the RESET worlds only, into both the Newton model (so Isaac Lab
  data and any later Newton notification agree) and MuJoCo-Warp's per-world arrays - exactly what the
  MuJoCo trainer does (training/worm/mujoco/worm_env.py A6: body_mass, body_inertia, actuator gain/bias
  per world; invweight0 left at nominal, as CPU MuJoCo does without mj_setConst).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import warp as wp

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


class randomize_friction_scale(ManagerTermBase):
    """Set every worm shape's sliding friction (Newton ``shape_material_mu``) to ``base * s``.

    ``s ~ U(lo, hi)`` is drawn once per env per reset (continuous: Newton has no unique-material cap).
    SolverMuJoCo copies ``shape_material_mu`` into MuJoCo-Warp's per-world ``geom_friction[:, 0]``.
    MuJoCo combines a contact's two geoms with max(); the floor is kept below the smallest worm value
    (spec.GROUND_FRICTION), so worm-floor pairs get exactly ``base * s`` and worm-worm pairs (same value
    on both shapes) too. Torsional/rolling entries stay at worm.xml's values; with condim 3 they are
    inactive, as in MuJoCo.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        import isaaclab_newton.physics.newton_manager as newton_manager_module  # noqa: PLC0415
        from newton.solvers import SolverNotifyFlags  # noqa: PLC0415

        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self._manager = newton_manager_module.NewtonManager
        self._flag = SolverNotifyFlags.SHAPE_PROPERTIES
        model = self._manager.get_model()
        # (num_envs, num_shapes_of_this_articulation) view into model.shape_material_mu
        self._mu = self.asset._root_view.get_attribute("shape_material_mu", model)[:, 0]
        self.scale = torch.ones(env.scene.num_envs, device=env.device)

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        base: float,
        scale_range: tuple[float, float],
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        else:
            env_ids = env_ids.to(env.device).long()
        lo, hi = scale_range
        s = torch.empty(len(env_ids), device=env.device).uniform_(lo, hi)
        self.scale[env_ids] = s
        mu = wp.to_torch(self._mu)
        mu[env_ids] = (base * s).unsqueeze(-1).to(mu.dtype)
        self._manager.add_model_change(self._flag)


def _ids(ids, count: int) -> list[int]:
    """SceneEntityCfg resolves a selection covering every element to slice(None)."""
    return list(range(count))[ids] if isinstance(ids, slice) else [int(i) for i in ids]


def _solver_and_names():
    import isaaclab_newton.physics.newton_manager as newton_manager_module  # noqa: PLC0415

    solver = newton_manager_module.NewtonManager._solver
    if solver is None or not hasattr(solver, "mjw_model"):
        raise RuntimeError("the lean worm randomisers need Newton's SolverMuJoCo (MuJoCo-Warp)")
    return solver


class randomize_segment_mass_lean(ManagerTermBase):
    """Scale each segment's mass by an independent U(lo, hi) per reset; inertia scaled to match (item 7).

    Links are not randomised (the ``asset_cfg`` body list is the 5 segments). Writes the reset worlds'
    Newton ``body_mass`` / ``body_inertia`` (Isaac Lab data views) and MuJoCo-Warp ``body_mass`` /
    ``body_inertia`` (principal moments; a uniform scale keeps the principal axes). No ``set_const``.

    Isaac Lab builds class terms on PHYSICS_READY, before Newton creates the MuJoCo solver, so the
    solver arrays (and the nominal values) are bound on the first call - the initial reset, when every
    world is still nominal.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self._bound = False

    def _bind(self, env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg):
        mm = (solver := _solver_and_names()).mj_model
        body_ids = _ids(asset_cfg.body_ids, self.asset.num_bodies)
        self.body_ids = torch.tensor(body_ids, device=env.device, dtype=torch.long)
        mjc = []
        for n in (self.asset.body_names[i] for i in body_ids):
            hits = [j for j in range(mm.nbody) if mm.body(j).name.endswith("_" + n)]
            if len(hits) != 1:
                raise RuntimeError(f"cannot map body {n!r} to one MuJoCo body: {hits}")
            mjc.append(hits[0])
        self.mjc_ids = torch.tensor(mjc, device=env.device, dtype=torch.long)
        self.mjw_mass = wp.to_torch(solver.mjw_model.body_mass)  # (nworld, nbody)
        self.mjw_inertia = wp.to_torch(solver.mjw_model.body_inertia)  # (nworld, nbody, 3)
        if self.mjw_mass.shape[0] != env.num_envs or self.mjw_inertia.shape[0] != env.num_envs:
            raise RuntimeError("MuJoCo-Warp body_mass/body_inertia are not batched per world")
        d = self.asset.data
        self.default_mass = d.body_mass.torch[:, self.body_ids].clone()
        self.default_inertia = d.body_inertia.torch[:, self.body_ids].clone()
        self.default_mjw_mass = self.mjw_mass[:, self.mjc_ids].clone()
        self.default_mjw_inertia = self.mjw_inertia[:, self.mjc_ids].clone()
        if not torch.allclose(self.default_mjw_mass, self.default_mass):
            raise RuntimeError("MuJoCo-Warp body masses do not match Isaac Lab body masses")
        self.scale = torch.ones(env.num_envs, len(mjc), device=env.device)
        self._bound = True

    def __call__(self, env: "ManagerBasedEnv", env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg,
                 scale_range: tuple[float, float]):
        if not self._bound:
            self._bind(env, asset_cfg)
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        e = env_ids.to(env.device).long()
        lo, hi = scale_range
        s = torch.empty(len(e), len(self.body_ids), device=env.device).uniform_(lo, hi)
        self.scale[e] = s
        ee, bb, mj = e[:, None], self.body_ids[None, :], self.mjc_ids[None, :]
        d = self.asset.data
        d.body_mass.torch[ee, bb] = self.default_mass[e] * s
        d.body_inertia.torch[ee, bb] = self.default_inertia[e] * s.unsqueeze(-1)
        self.mjw_mass[ee, mj] = self.default_mjw_mass[e] * s
        self.mjw_inertia[ee, mj] = self.default_mjw_inertia[e] * s.unsqueeze(-1)


class randomize_kp_lean(ManagerTermBase):
    """Scale each joint's servo stiffness kp by an independent U(lo, hi) per reset (items 7, 15).

    The force limit and the joint damping stay fixed. Writes the reset worlds' kp into Newton's
    ``joint_target_ke`` (Isaac Lab ``data.joint_stiffness`` view, read by the graphed torque telemetry),
    into the Isaac Lab implicit actuator's ``stiffness`` (read by the per-substep path), and into
    MuJoCo-Warp's per-world position actuator: gainprm[0] = kp, biasprm[1] = -kp. Bound lazily, like
    :class:`randomize_segment_mass_lean`.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self._bound = False

    def _bind(self, env: "ManagerBasedEnv", asset_cfg: SceneEntityCfg):
        mm = (solver := _solver_and_names()).mj_model
        joint_ids = _ids(asset_cfg.joint_ids, self.asset.num_joints)
        self.joint_ids = torch.tensor(joint_ids, device=env.device, dtype=torch.long)
        acts = []
        for n in (self.asset.joint_names[i] for i in joint_ids):
            hits = [a for a in range(mm.nu) if mm.joint(int(mm.actuator_trnid[a][0])).name.endswith("_" + n)]
            if len(hits) != 1:
                raise RuntimeError(f"cannot map joint {n!r} to one MuJoCo actuator: {hits}")
            acts.append(hits[0])
        self.act_ids = torch.tensor(acts, device=env.device, dtype=torch.long)
        self.gain = wp.to_torch(solver.mjw_model.actuator_gainprm)  # (nworld, nu, 10)
        self.bias = wp.to_torch(solver.mjw_model.actuator_biasprm)  # (nworld, nu, 10)
        if self.gain.shape[0] != env.num_envs or self.bias.shape[0] != env.num_envs:
            raise RuntimeError("MuJoCo-Warp actuator gains are not batched per world")
        self.default_kp = self.asset.data.joint_stiffness.torch[:, self.joint_ids].clone()
        if not torch.allclose(self.gain[:, self.act_ids, 0], self.default_kp):
            raise RuntimeError("MuJoCo-Warp actuator gains do not match Isaac Lab joint stiffness")
        # per actuator group: (columns of actuator.stiffness, matching Lab joint ids)
        self.groups = []
        for act in self.asset.actuators.values():
            cols = act.joint_indices
            cols = list(range(self.asset.num_joints))[cols] if isinstance(cols, slice) else [int(c) for c in cols]
            pairs = [(k, joint_ids.index(j)) for k, j in enumerate(cols) if j in joint_ids]
            if pairs:
                self.groups.append((act, torch.tensor([p[0] for p in pairs], device=env.device),
                                    torch.tensor([p[1] for p in pairs], device=env.device)))
        self.kp = self.default_kp.clone()
        self._bound = True

    def __call__(self, env: "ManagerBasedEnv", env_ids: torch.Tensor | None, asset_cfg: SceneEntityCfg,
                 scale_range: tuple[float, float]):
        if not self._bound:
            self._bind(env, asset_cfg)
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device=env.device)
        e = env_ids.to(env.device).long()
        lo, hi = scale_range
        kp = self.default_kp[e] * torch.empty(len(e), len(self.joint_ids), device=env.device).uniform_(lo, hi)
        self.kp[e] = kp
        ee, jj, aa = e[:, None], self.joint_ids[None, :], self.act_ids[None, :]
        self.asset.data.joint_stiffness.torch[ee, jj] = kp
        for act, act_cols, sel in self.groups:  # Lab implicit actuator (per-substep path's torque estimate)
            act.stiffness[ee, act_cols[None, :]] = kp[:, sel]
        self.gain[ee, aa, 0] = kp
        self.bias[ee, aa, 1] = -kp
