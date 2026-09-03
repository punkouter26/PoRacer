"""Batched MuJoCo Warp environment for MojucuBoy, the 21-DOF humanoid racer.

Everything lives on the GPU: MJWarp integrates `NUM_WORLDS` copies of the model,
and the observation, reward and reset logic run as torch ops over zero-copy views
of MJWarp's own arrays (wp.to_torch). Nothing is read back to the host inside the
rollout loop -- a single .cpu() per step would cost more than the physics.

The observation is the contract in mojucuboy_obs.py / MojucuBoyObservation.cs. This module
reimplements it in batched torch rather than importing it, because that one is
scalar numpy over a single mjData; gate4_contract.py asserts the two agree
element-for-element so the duplication cannot silently drift.

Timing: 0.005 s physics x decimation 4 = 0.02 s policy, which is exactly Unity's
Time.fixedDeltaTime x 4. The policy therefore runs at the same rate in training
and in the game.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "mojucuboy_roundtrip.xml"
RIG_PATH = HERE / "mojucuboy_rig.json"

JOINT_COUNT = 21
OBS_SIZE = 75
ACTION_SIZE = 21
FORWARD_AXIS = 1          # body-local +Y, see build_mjcf.py

DECIMATION = 4
EPISODE_STEPS = 1000      # 20 s at 0.02 s

# Contact budget. Sized against a MEASURED worst case, not a guess: 12 tumbling
# rollouts under random torques peaked at ncon=11 / nefc=46, so these give ~2.9x
# headroom. An overflow silently drops contacts -- the exact failure this repo
# already recorded once -- so the trainer asserts on it rather than trusting it.
NCONMAX = 32
NJMAX = 128

# Reward weights. Two rules, both learned the hard way:
#
#  1. The survival terms must stay small relative to the tracking term.
#     training/fido/creature_env.py records a run that converged to 0.04 m/s
#     because standing still out-paid creeping forward.
#
#  2. Every penalty is SATURATING (tanh of a normalised quantity), so no single
#     term can swamp the reward however extreme the state. The first cut used raw
#     quadratics and the impact term measured -205 against a tracking term of
#     +0.13 -- the policy's best move would have been to stop moving. Bounded
#     penalties make that impossible by construction rather than by tuning.
W_TRACK = 2.0             # forward speed along the commanded heading
W_HEADING = 0.4           # facing the commanded heading
W_UPRIGHT = 0.05
W_ALIVE = 0.10
W_DRIFT = 0.15            # uncommanded lateral + yaw motion
W_CTRL = 0.005
W_ACCEL = 0.03            # excessive joint accelerations
W_IMPACT = 0.05           # impact forces beyond simply carrying body weight
W_ACTION_RATE = 0.01

# Normalising scales for the saturating penalties, measured from rollouts under
# random actions: drift ~12, joint accel ~1.8e6, constraint force ~1.4e6.
SCALE_DRIFT = 8.0
SCALE_ACCEL = 2.0e6
SPEED_SIGMA = 1.0         # width of the tracking kernel, m/s

TARGET_SPEED = 1.5

# How far, as a fraction of each joint's half-range, a saturated action moves the
# target away from the stance. Below 1.0 so the policy cannot slam a joint into its
# limit in a single step from the stance.
ACTION_SCALE = 0.6
MIN_HEIGHT = 0.45         # torso below this = fallen, terminate
MIN_UPRIGHT = 0.30

# Domain randomisation, per the brief.
GAIN_RANGE = 0.30         # actuator gains +/-30%
MASS_RANGE = 0.10         # link masses +/-10%
FRICTION_RANGE = (0.6, 1.4)
PUSH_INTERVAL = 150       # policy steps between external pushes
PUSH_VELOCITY = 1.1       # m/s impulse applied to the root


class MojucuBoyEnv:
    def __init__(self, num_worlds: int, device: str = "cuda:0", seed: int = 0):
        self.num_worlds = num_worlds
        self.device = torch.device(device)
        self.rig = json.loads(RIG_PATH.read_text())

        self.mjm = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.dt = float(self.mjm.opt.timestep) * DECIMATION

        mjd = mujoco.MjData(self.mjm)
        stance = np.array(self.rig["stance_qpos"])
        mjd.qpos[:] = stance
        mjd.ctrl[:] = stance[7:]
        mujoco.mj_forward(self.mjm, mjd)

        # Model parameters default to ONE shared copy across all worlds, so a naive
        # domain randomisation writes into a broadcast array and silently does
        # nothing. batch_sizes gives these four fields a per-world leading
        # dimension, which is what makes the randomisation real.
        self.wm = mjw.put_model(self.mjm, batch_sizes={
            "actuator_gainprm": num_worlds,
            "actuator_biasprm": num_worlds,
            "body_mass": num_worlds,
            "geom_friction": num_worlds,
        })
        # The overflow warning is a GPU printf per world per step; it produced
        # megabytes of output and dominated the measured step time. The trainer
        # checks nacon against naconmax explicitly instead.
        self.wm.opt.warn_overflow = 0
        self.wd = mjw.put_data(self.mjm, mjd, nworld=num_worlds,
                               nconmax=NCONMAX, njmax=NJMAX)

        self.root_body = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_BODY, "hips")
        order = self.rig["actuator_order"]
        qpos_addr, dof_addr = [], []
        for name in order:
            jid = mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_addr.append(int(self.mjm.jnt_qposadr[jid]))
            dof_addr.append(int(self.mjm.jnt_dofadr[jid]))
        self.qpos_addr = torch.tensor(qpos_addr, device=self.device, dtype=torch.long)
        self.dof_addr = torch.tensor(dof_addr, device=self.device, dtype=torch.long)

        lo = np.array([j["range_rad"][0] for j in self.rig["joints"]], dtype=np.float32)
        hi = np.array([j["range_rad"][1] for j in self.rig["joints"]], dtype=np.float32)
        self.joint_lo = torch.tensor(lo, device=self.device)
        self.joint_hi = torch.tensor(hi, device=self.device)
        self.joint_mid = 0.5 * (self.joint_lo + self.joint_hi)
        self.joint_half = 0.5 * (self.joint_hi - self.joint_lo)

        self.stance_qpos = torch.tensor(stance, device=self.device, dtype=torch.float32)
        self.stance_joints = self.stance_qpos[7:]

        # Pristine copies of the randomised model parameters, so each reset
        # perturbs the nominal values rather than compounding.
        self.nominal_gain = wp.to_torch(self.wm.actuator_gainprm).clone()
        self.nominal_bias = wp.to_torch(self.wm.actuator_biasprm).clone()
        self.nominal_mass = wp.to_torch(self.wm.body_mass).clone()
        self.nominal_friction = wp.to_torch(self.wm.geom_friction).clone()

        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.last_action = torch.zeros(num_worlds, ACTION_SIZE, device=self.device)
        self.prev_action = torch.zeros_like(self.last_action)
        self.prev_qvel = torch.zeros(num_worlds, JOINT_COUNT, device=self.device)
        self.episode_step = torch.zeros(num_worlds, device=self.device, dtype=torch.long)
        self.command_heading = torch.zeros(num_worlds, device=self.device)
        self.command_speed = torch.full((num_worlds,), TARGET_SPEED, device=self.device)

        self.reset(torch.arange(num_worlds, device=self.device))

    # ---- views onto MJWarp state (zero copy, stay on GPU) ------------------
    @property
    def qpos(self) -> torch.Tensor:
        return wp.to_torch(self.wd.qpos)

    @property
    def qvel(self) -> torch.Tensor:
        return wp.to_torch(self.wd.qvel)

    @property
    def xmat(self) -> torch.Tensor:
        return wp.to_torch(self.wd.ximat if hasattr(self.wd, "ximat") else self.wd.xmat)

    def root_rotation(self) -> torch.Tensor:
        """(N, 3, 3) body->world rotation of the root, from the free-joint quaternion.

        Read from qpos rather than xmat so it is valid even before a forward pass,
        and so the layout does not depend on MJWarp's body ordering.
        """
        q = self.qpos[:, 3:7]
        q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-9)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        rot = torch.empty(self.num_worlds, 3, 3, device=self.device)
        rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
        rot[:, 0, 1] = 2 * (x * y - z * w)
        rot[:, 0, 2] = 2 * (x * z + y * w)
        rot[:, 1, 0] = 2 * (x * y + z * w)
        rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
        rot[:, 1, 2] = 2 * (y * z - x * w)
        rot[:, 2, 0] = 2 * (x * z - y * w)
        rot[:, 2, 1] = 2 * (y * z + x * w)
        rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
        return rot

    # ---- observation -------------------------------------------------------
    def observation(self) -> torch.Tensor:
        rot = self.root_rotation()
        qpos, qvel = self.qpos, self.qvel

        gravity_local = -rot[:, 2, :]                                   # R^T @ (0,0,-1)
        linvel_local = torch.bmm(rot.transpose(1, 2), qvel[:, 0:3].unsqueeze(2)).squeeze(2)
        angvel_local = qvel[:, 3:6]

        forward = rot[:, :, FORWARD_AXIS]
        heading = torch.atan2(forward[:, 1], forward[:, 0])
        error = self.command_heading - heading
        error = (error + torch.pi) % (2 * torch.pi) - torch.pi

        obs = torch.empty(self.num_worlds, OBS_SIZE, device=self.device)
        obs[:, 0:3] = gravity_local
        obs[:, 3:6] = linvel_local
        obs[:, 6:9] = angvel_local
        obs[:, 9] = torch.cos(error)
        obs[:, 10] = torch.sin(error)
        obs[:, 11] = self.command_speed
        obs[:, 12:12 + JOINT_COUNT] = qpos[:, self.qpos_addr]
        obs[:, 12 + JOINT_COUNT:12 + 2 * JOINT_COUNT] = qvel[:, self.dof_addr]
        obs[:, 12 + 2 * JOINT_COUNT:] = self.last_action
        return obs

    # ---- reset -------------------------------------------------------------
    def reset(self, index: torch.Tensor) -> None:
        if index.numel() == 0:
            return
        n = index.numel()
        qpos, qvel = self.qpos, self.qvel

        qpos[index] = self.stance_qpos.unsqueeze(0).expand(n, -1).to(qpos.dtype)
        noise = (torch.rand(n, JOINT_COUNT, generator=self.generator,
                            device=self.device) * 2 - 1) * 0.10
        joints = torch.clamp(self.stance_joints + noise, self.joint_lo, self.joint_hi)
        qpos[index[:, None], self.qpos_addr[None, :]] = joints.to(qpos.dtype)

        # Small random yaw so the policy cannot memorise a single world heading.
        yaw = (torch.rand(n, generator=self.generator, device=self.device) * 2 - 1) * torch.pi
        qpos[index, 3] = torch.cos(yaw / 2).to(qpos.dtype)
        qpos[index, 4] = 0
        qpos[index, 5] = 0
        qpos[index, 6] = torch.sin(yaw / 2).to(qpos.dtype)

        qvel[index] = 0
        self.last_action[index] = 0
        self.prev_action[index] = 0
        self.prev_qvel[index] = 0
        self.episode_step[index] = 0

        # Command a heading near the racer's own, so the task starts tractable and
        # the policy still has to steer.
        self.command_heading[index] = yaw + (
            torch.rand(n, generator=self.generator, device=self.device) * 2 - 1) * 0.6
        self.command_speed[index] = TARGET_SPEED

        self._randomise(index)

    def _randomise(self, index: torch.Tensor) -> None:
        """Domain randomisation: actuator gains, link masses, ground friction.

        MJWarp keeps these as per-world arrays when the model was put with
        nworld > 1; if a build ever exposes them as shared, this degrades to a
        single global perturbation rather than failing, so the check is explicit.
        """
        n = index.numel()

        def spread(size, amount):
            return 1.0 + (torch.rand(size, generator=self.generator,
                                     device=self.device) * 2 - 1) * amount

        gain = wp.to_torch(self.wm.actuator_gainprm)
        bias = wp.to_torch(self.wm.actuator_biasprm)
        if gain.shape[0] == self.num_worlds:
            scale = spread((n, gain.shape[1]), GAIN_RANGE).unsqueeze(-1)
            gain[index] = self.nominal_gain[index] * scale
            bias[index] = self.nominal_bias[index] * scale

        mass = wp.to_torch(self.wm.body_mass)
        if mass.shape[0] == self.num_worlds:
            mass[index] = self.nominal_mass[index] * spread((n, mass.shape[1]), MASS_RANGE)

        friction = wp.to_torch(self.wm.geom_friction)
        if friction.shape[0] == self.num_worlds:
            lo, hi = FRICTION_RANGE
            factor = lo + torch.rand(n, generator=self.generator, device=self.device) * (hi - lo)
            friction[index] = self.nominal_friction[index] * factor.view(n, 1, 1)

    # ---- step --------------------------------------------------------------
    def step(self, action: torch.Tensor):
        action = torch.clamp(action, -1.0, 1.0)
        self.prev_action = self.last_action
        self.last_action = action

        # Actions are joint-position targets in normalised [-1, 1], expressed as an
        # offset from the STANDING STANCE, not from the middle of each joint's range.
        # Centring on the range midpoint makes action=0 an arbitrary splayed pose:
        # measured, every world fell within 50 steps under a zero action. Centring on
        # the stance makes action=0 mean "hold the pose you start in", which is a far
        # better prior and is what the stance was calibrated for in Gate 1.
        target = torch.clamp(self.stance_joints + ACTION_SCALE * self.joint_half * action,
                             self.joint_lo, self.joint_hi)
        ctrl = wp.to_torch(self.wd.ctrl)
        ctrl[:] = target.to(ctrl.dtype)

        prev_qvel = self.qvel[:, self.dof_addr].clone()
        for _ in range(DECIMATION):
            mjw.step(self.wm, self.wd)

        obs = self.observation()
        reward, terms = self._reward(prev_qvel)

        self.episode_step += 1
        height = self.qpos[:, 2]
        upright = -self.observation()[:, 2]   # gravity_local z, 1 when upright
        fallen = (height < MIN_HEIGHT) | (upright < MIN_UPRIGHT)
        timeout = self.episode_step >= EPISODE_STEPS
        done = fallen | timeout

        # Scheduled external pushes, per the brief's domain randomisation.
        pushing = (self.episode_step % PUSH_INTERVAL == 0) & ~done
        if pushing.any():
            idx = pushing.nonzero(as_tuple=True)[0]
            qvel = self.qvel
            kick = (torch.rand(idx.numel(), 2, generator=self.generator,
                               device=self.device) * 2 - 1) * PUSH_VELOCITY
            qvel[idx, 0:2] += kick.to(qvel.dtype)

        terms["fallen"] = fallen.float()
        terms["timeout"] = timeout.float()
        return obs, reward, done, terms

    def _reward(self, prev_joint_qvel: torch.Tensor):
        rot = self.root_rotation()
        qvel = self.qvel
        obs_gravity_z = -rot[:, 2, 2]

        # Velocity along the COMMANDED heading, not along +y: the racer is steered.
        heading = self.command_heading
        cmd_dir = torch.stack([torch.cos(heading), torch.sin(heading)], dim=1)
        planar = qvel[:, 0:2].float()
        along = (planar * cmd_dir).sum(dim=1)
        lateral = (planar * torch.stack([-cmd_dir[:, 1], cmd_dir[:, 0]], dim=1)).sum(dim=1)

        # Saturating tracking term: exceeding the target speed earns nothing extra,
        # which stops the policy trading stability for a sprint it cannot hold.
        shortfall = (along - self.command_speed).clamp(max=0.0)
        track = torch.exp(-(shortfall / SPEED_SIGMA) ** 2)

        forward = rot[:, :, FORWARD_AXIS]
        facing = (forward[:, 0] * cmd_dir[:, 0] + forward[:, 1] * cmd_dir[:, 1])

        joint_qvel = qvel[:, self.dof_addr].float()
        accel = ((joint_qvel - prev_joint_qvel.float()) / self.dt).pow(2).sum(dim=1)
        action_rate = (self.last_action - self.prev_action).pow(2).mean(dim=1)
        ctrl_cost = self.last_action.pow(2).mean(dim=1)
        drift = lateral.pow(2) + qvel[:, 5].float().pow(2)

        # Impact, expressed as constraint force in units of body weight. The raw
        # qfrc_constraint includes the ground reaction that simply holds the racer
        # up, so penalising it directly penalises standing; only the EXCESS over
        # carrying its own weight is an impact worth discouraging.
        force = wp.to_torch(self.wd.qfrc_constraint).float().norm(dim=1)
        weight = float(self.mjm.body_mass.sum() * abs(self.mjm.opt.gravity[2]))
        impact = (force / weight - 1.5).clamp(min=0.0)

        reward = (
            W_TRACK * track
            + W_HEADING * facing.clamp(min=0.0)
            + W_UPRIGHT * obs_gravity_z.clamp(min=0.0)
            + W_ALIVE
            - W_DRIFT * torch.tanh(drift / SCALE_DRIFT)
            - W_CTRL * ctrl_cost
            - W_ACCEL * torch.tanh(accel / SCALE_ACCEL)
            - W_IMPACT * torch.tanh(impact)
            - W_ACTION_RATE * action_rate
        )
        terms = {
            "track": track, "facing": facing, "speed_along": along,
            "drift": drift, "accel": accel, "impact": impact,
        }
        return reward, terms

    def contact_overflow(self) -> int:
        """Peak contacts this step. The trainer aborts if this reaches naconmax:
        an overflow silently drops contacts and the creature sinks."""
        return int(wp.to_torch(self.wd.nacon).max().item())

    @property
    def naconmax(self) -> int:
        return int(self.wd.naconmax)
