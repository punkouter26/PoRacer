"""Reference trot (QUAD_SPEC rounds 8-9), mirroring the MuJoCo trainer (training/quad/mujoco/quad_env.py Q16 and
training/creature/env.py ``reference_gait``):

* clock: per env, phi ~ U(0, 2 pi) at every reset; after the physics of each policy step phi += 2 pi f dt
  (f 1.5 Hz, dt 0.05 s), wrapped to [0, 2 pi). Rewards see the advanced (post-step) clock; worlds that reset get a
  fresh phi, and the observation shows it (obs[36:38] = sin phi, cos phi).
* reference pose, per leg with hip phase ph (quad_rig order RL, FL, RR, FR -> ph = 0, pi, pi, 0 from
  training/bugs/Quad_v01_rig.json gaitPhases), psi = phi + ph:
      hip  = rest + 0.785398 * 0.3 * sin(psi)
      knee = rest + 0.785398 * 0.5 * s * max(0, -cos(psi)),  s = +1
* stance expected for a foot while -cos(psi_hip) < 0.
* gait_ref      = exp(-mean_8 (q - q_ref)^2 / sigma^2)
* contact_phase = mean over the 4 feet of [debounced contact state == stance expected]

The clock lives on the env (``env._quad_gait_clock``) and advances exactly once per policy step, whichever
term asks first (keyed by ``env.common_step_counter``), so it works whatever terms are enabled.
"""

from __future__ import annotations

import json
import math
import os
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg

from .. import spec
from .rewards import _guard

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from isaaclab.managers import EventTermCfg

_BUG_RIG = os.path.join(spec.REPO, "training", "bugs", "Quad_v01_rig.json")
with open(_BUG_RIG, "r", encoding="utf-8") as _f:
    _GAIT_PHASES = [float(v) for v in json.load(_f)["agent"]["gaitPhases"]]
HIP_IDX = [0, 2, 4, 6]  # action order: hip, knee per leg (RL, FL, RR, FR)
HIP_PHASES = [_GAIT_PHASES[i] for i in HIP_IDX]  # 0, pi, pi, 0


class GaitClock:
    def __init__(self, env: "ManagerBasedEnv"):
        dev = env.device
        self.phase = torch.zeros(env.num_envs, device=dev)
        self.omega_dt = 2.0 * math.pi * spec.GAIT_FREQ * env.step_dt
        self.last_step = int(env.common_step_counter)
        self.offsets = torch.tensor([ph for ph in HIP_PHASES for _ in (0, 1)], device=dev)  # knee = its hip's phase
        self.amps = torch.tensor([spec.GAIT_AMP_HIP, spec.GAIT_AMP_KNEE] * 4, device=dev)
        self.lift = torch.tensor([False, True] * 4, device=dev)
        self.knee_sign = spec.GAIT_KNEE_SIGN
        self.foot_offsets = torch.tensor(HIP_PHASES, device=dev)
        self.rest = torch.tensor(spec.REST_POSE, device=dev)

    def advance(self, env) -> torch.Tensor:
        step = int(env.common_step_counter)
        if step != self.last_step:
            self.last_step = step
            self.phase = torch.remainder(self.phase + self.omega_dt, 2.0 * math.pi)
        return self.phase

    def reference_action(self, phase: torch.Tensor | None = None) -> torch.Tensor:
        """(N, 8) reference in action units (q_ref = rest + 0.785398 * this)."""
        psi = (self.phase if phase is None else phase).unsqueeze(1) + self.offsets
        shape = torch.where(self.lift, self.knee_sign * torch.clamp(-torch.cos(psi), min=0.0), torch.sin(psi))
        return self.amps * shape

    def reference_pose(self) -> torch.Tensor:
        return self.rest + spec.ACTION_SCALE * self.reference_action()

    def stance_expected(self) -> torch.Tensor:
        """(N, 4) bool in quad_rig lowerLegs order (RL, FL, RR, FR)."""
        return -torch.cos(self.phase.unsqueeze(1) + self.foot_offsets) < 0.0


def get_clock(env) -> GaitClock:
    clock = getattr(env, "_quad_gait_clock", None)
    if clock is None:
        clock = GaitClock(env)
        env._quad_gait_clock = clock
    return clock


class reset_gait_clock(ManagerTermBase):
    """Reset event: phi ~ U(0, 2 pi) for the reset envs (torch RNG, seeded with the env)."""

    def __init__(self, cfg: "EventTermCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)

    def __call__(self, env: "ManagerBasedEnv", env_ids: torch.Tensor | None):
        clock = get_clock(env)
        ids = torch.arange(env.num_envs, device=env.device) if env_ids is None else env_ids.to(env.device).long()
        clock.phase[ids] = torch.rand(len(ids), device=env.device) * (2.0 * math.pi)


def gait_clock_obs(env: "ManagerBasedEnv") -> torch.Tensor:
    """(sin phi, cos phi) of the current clock (post-step, post-reset). (2)"""
    phase = get_clock(env).advance(env)
    return torch.stack((torch.sin(phase), torch.cos(phase)), dim=1)


def gait_ref(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg, sigma: float) -> torch.Tensor:
    """exp(-mean over the 8 joints (q - q_ref)^2 / sigma^2) on the post-step state and clock."""
    clock = get_clock(env)
    clock.advance(env)
    q = env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]
    return _guard(env, torch.exp(-torch.mean(torch.square(q - clock.reference_pose()), dim=1) / (sigma * sigma)))


def contact_phase(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Share of the 4 feet whose DEBOUNCED contact state equals "stance expected" (post-step)."""
    clock = get_clock(env)
    clock.advance(env)
    tracker = env.scene["robot"].contact_tracker
    tracker.finish_step(env.common_step_counter)
    down = tracker.t_db.bool()
    return _guard(env, (down == clock.stance_expected()).float().mean(dim=1))
