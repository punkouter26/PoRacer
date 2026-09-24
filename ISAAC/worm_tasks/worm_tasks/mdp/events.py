"""Per-episode friction randomisation for Worm5 (WORM_SPEC.md "Resets and randomisation").

Mass and kp use Isaac Lab's own ``randomize_rigid_body_mass`` / ``randomize_actuator_gains``
in ``reset`` mode; friction needs its own term because the spec scales ONE coefficient per
episode (MuJoCo has a single sliding friction per contact), while Isaac Lab's
``randomize_rigid_body_material`` draws static and dynamic friction independently per shape.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from isaaclab.managers import EventTermCfg


class randomize_friction_scale(ManagerTermBase):
    """Set every worm shape's static AND dynamic friction to ``base * s``, restitution 0.

    ``s ~ U(lo, hi)`` is drawn once per env per reset and snapped to ``step`` (PhysX caps the
    number of unique materials; the tensor API makes one per distinct value). The ground uses
    friction 1.0 with combine mode "multiply", so the worm-ground pair friction is exactly
    ``base * s``; worm-worm pairs (same value on both shapes) are ``base * s`` under any mode.
    """

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params["asset_cfg"].name]
        self.scale = torch.ones(env.scene.num_envs, device="cpu")

    def __call__(
        self,
        env: "ManagerBasedEnv",
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        base: float,
        scale_range: tuple[float, float],
        step: float,
    ):
        if env_ids is None:
            env_ids = torch.arange(env.scene.num_envs, device="cpu")
        else:
            env_ids = env_ids.cpu()
        lo, hi = scale_range
        s = torch.empty(len(env_ids)).uniform_(lo, hi)
        s = torch.clamp(torch.round(s / step) * step, lo, hi)
        self.scale[env_ids] = s
        mu = (base * s).unsqueeze(-1)
        mats = self.asset.root_physx_view.get_material_properties()
        mats[env_ids, :, 0] = mu
        mats[env_ids, :, 1] = mu
        mats[env_ids, :, 2] = 0.0
        self.asset.root_physx_view.set_material_properties(mats, env_ids)
