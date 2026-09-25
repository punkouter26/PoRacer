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
import math
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
# Paid continuously for being on its feet. This is what makes standing back up
# worth the effort when a fall no longer ends the episode.
W_GETUP = 0.60
STANDING_HEIGHT = 0.77          # hips height of the calibrated stance, metres

# Fraction of resets that START the racer on the ground, in a random sprawl. A
# policy that only ever begins upright almost never sees a recoverable fallen
# state, so it never learns the recovery -- it has to be trained on it directly.
RESET_FALLEN_FRACTION = 0.30

# Normalising scales for the saturating penalties, measured from rollouts under
# random actions: drift ~12, joint accel ~1.8e6, constraint force ~1.4e6.
SCALE_DRIFT = 8.0
SCALE_ACCEL = 2.0e6
SPEED_SIGMA = 1.0         # width of the tracking kernel, m/s

TARGET_SPEED = 1.5

# Lane following (--lane-follow). The race gives every racer a lane: a line from its
# spawn point toward its own point on the finish line, and the game steers him back
# onto it by aiming LANE_LOOKAHEAD metres ahead on the line, with the correction
# capped at LANE_MAX_TURN from where he faces (MojucuBoyController / Agent_MojucuBoy).
# Training with the same steering, and pricing net distance off the line rather
# than every sideways step, is what the K5 dead end in rl_optimization_log.md
# (M35-M37: per-step lateral velocity is what walking is made of) recommended.
LANE_LOOKAHEAD = 3.0      # metres ahead on the lane the heading command aims at
LANE_MAX_TURN = math.radians(30.0)
W_XTRACK = 0.5            # distance off the lane, saturating
SCALE_XTRACK = 1.0        # metres; tanh(1) of the weight at 1 m off

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
    def __init__(self, num_worlds: int, device: str = "cuda:0", seed: int = 0,
                 two_sided_speed: bool = False,
                 upright_weight: float = W_UPRIGHT,
                 reset_fallen_fraction: float = RESET_FALLEN_FRACTION,
                 terminate_on_fall: bool = False,
                 sprawl_max_tilt: float = math.pi,
                 command_speed_range: tuple = None,
                 scale_drift: float = SCALE_DRIFT,
                 ctrl_weight: float = W_CTRL,
                 drift_yaw_weight: float = 1.0,
                 heading_weight: float = W_HEADING,
                 drift_weight: float = W_DRIFT,
                 lane_follow: bool = False,
                 xtrack_weight: float = W_XTRACK):
        self.num_worlds = num_worlds
        self.device = torch.device(device)
        # W_UPRIGHT is the only positive term NOT gated on `standing`, so while
        # the racer is down it is the entire gradient back towards getting up.
        # At the shipped 0.05 it is half of the unconditional W_ALIVE (0.10),
        # which was survivable only because the old sign let `standing` be
        # maximised by lying inverted (M9). With that exploit closed the racer
        # has to learn real balance, and 0.05 measurably is not enough to climb:
        # a 1500-iteration run sat flat at uptime 0.08 for 49 M steps.
        self.upright_weight = upright_weight
        self.reset_fallen_fraction = reset_fallen_fraction
        self.terminate_on_fall = terminate_on_fall
        # Largest tilt off vertical a sprawl start may use, radians. pi = the shipped
        # fully-random orientation; smaller values are the difficulty curriculum the
        # trainer ramps. Settable per-iteration, so one run can walk it up.
        self.sprawl_max_tilt = float(sprawl_max_tilt)
        # (low, high) m/s to sample the commanded speed from per episode, or None
        # for the shipped fixed TARGET_SPEED. See the note in reset() -- a constant
        # command is an observation with zero variance, which is an observation the
        # policy cannot learn to use (M11).
        self.command_speed_range = command_speed_range
        # SCALE_DRIFT and W_CTRL, overridable. Both were measured in M14 to sit far
        # below the range where they influence anything for a TRAINED policy: the
        # drift tanh operates at 2-5 %% of its range (contributing 0.004-0.008 against
        # W_TRACK 1.8), and the un-saturated ctrl cost contributes 0.0004-0.004. The
        # scales were calibrated against random-action rollouts, which drift ~30x more
        # than a competent policy does.
        self.scale_drift = float(scale_drift)
        self.ctrl_weight = float(ctrl_weight)
        # How much of the drift penalty is YAW RATE, as opposed to lateral velocity.
        # 1.0 reproduces the shipped `drift = lateral^2 + yaw_rate^2`.
        #
        # These two quantities were conflated into one penalty, and they are not the
        # same thing: lateral velocity is pure waste, but YAW RATE IS HOW A RACER
        # CORRECTS ITS HEADING. Measured, going from scale_drift 8.0 to 0.5 (which is
        # the M16 fix for K5):
        #
        #     K5 lateral   0.365 -> 0.225  (gate < 0.25)   PASSES
        #     K4 heading   13.5  -> 16.7 deg (gate < 15)   now FAILS
        #                          22.3 deg at command 2.0
        #
        # So strengthening the penalty bought K5 by suppressing the yaw corrections
        # K4 depends on. Splitting the term lets lateral drift be priced without
        # taxing heading control.
        self.drift_yaw = float(drift_yaw_weight)
        # W_HEADING, overridable. Never varied in this project's history, and it is the
        # only term that pays for heading accuracy DIRECTLY. It matters because K4 and
        # K5 were measured to trade along a frontier that misses the gate corner:
        #
        #   scale_drift 8.0 -> K4 13.5 pass / K5 0.391 fail
        #   scale_drift 1.5 -> K4 14.2 pass / K5 0.311 fail
        #   scale_drift 1.0 -> K4 15.4 fail / K5 0.260 fail   <- closest approach
        #
        # so no setting of scale_drift satisfies K4 < 15 AND K5 < 0.25. Raising the
        # heading term buys K4 without relaxing the drift pressure K5 needs.
        self.heading_weight = float(heading_weight)
        # W_DRIFT, overridable. Never varied in this project's history -- only the
        # NORMALISER inside the tanh (`scale_drift`) was ever swept, which changes where
        # the penalty saturates, not what it is worth. At the shipped 0.15 the drift
        # penalty is at most 0.15 per step against a ~2.2 positive budget: under 7% even
        # fully saturated, so K5 was being chased with a term that cannot pay for itself.
        #
        # This is the same oversight as W_HEADING, which sat at 0.4 untouched while K4
        # was chased through the drift term -- and raising W_HEADING to 1.6 solved K4
        # immediately. Raising the weight also avoids the failure mode of lowering
        # scale_drift: that saturates the tanh early and flattens the gradient, which is
        # how f21a/m25a collapsed. Scaling the weight keeps the gradient's shape.
        self.drift_weight = float(drift_weight)
        # See LANE_LOOKAHEAD. Off reproduces every earlier run exactly.
        self.lane_follow = bool(lane_follow)
        self.xtrack_weight = float(xtrack_weight)
        # See _reward. The shipped brain runs at 2.04 m/s against a 1.5 m/s
        # command because the tracking kernel clamps positive error away, so
        # overshoot is free. Setting this makes the kernel symmetric, which is
        # what "track the commanded speed to +/-10%" actually requires.
        self.two_sided_speed = two_sided_speed
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
        # The lane: a start point and a direction per world, and the signed distance
        # off it, refreshed every step. Unused unless lane_follow.
        self.lane_origin = torch.zeros(num_worlds, 2, device=self.device)
        self.lane_heading = torch.zeros(num_worlds, device=self.device)
        self.cross_track = torch.zeros(num_worlds, device=self.device)

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
        # One diverged world must not poison the batch. A NaN here reaches the
        # policy's mean and torch.distributions rejects the whole tensor, ending a
        # multi-hour run over a single bad contact.
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

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

        # Start a share of the worlds already on the ground, in a random sprawl, so
        # the policy is trained on recovery rather than only on staying up. Without
        # this it would almost never encounter a fallen state it could still act
        # from, and "get back up" would stay unlearned however long it trained.
        fallen = (torch.rand(n, generator=self.generator, device=self.device)
                  < self.reset_fallen_fraction)
        if fallen.any():
            picked = index[fallen]
            m = picked.numel()
            if self.sprawl_max_tilt >= math.pi - 1e-6:
                # A random orientation, dropped just clear of the floor: face down, on
                # its back and everything between. This is the shipped behaviour and
                # what E11-E13 all trained against.
                quat = torch.randn(m, 4, generator=self.generator, device=self.device)
                quat = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-6)
            else:
                # DIFFICULTY-LIMITED sprawl: tilt off vertical by at most
                # `sprawl_max_tilt`, about a random horizontal axis.
                #
                # E11, E12 and E13 escalated the FREQUENCY of sprawl starts
                # (0.0 -> 0.3 -> 0.6) and never once varied their DIFFICULTY: every
                # one was a fully random quaternion. Measured by K10, all three
                # learned nothing -- 0 % recovery, and `ever_stood_again` also 0 %,
                # so the policy never even transiently reached standing from the
                # floor in 35 episodes x ~950 steps.
                #
                # That is an exploration failure, not a reward failure: the payoff
                # for standing is large and dense (W_GETUP 0.6 + W_TRACK 2.0 both
                # scale with `standing`), but from supine the motor sequence that
                # reaches it is long and specific, and random exploration never
                # finds it. A tilt bound makes the first rung of the ladder short
                # enough to stumble onto, and the ramp keeps it that way.
                axis_angle = (torch.rand(m, generator=self.generator, device=self.device)
                              * 2 * math.pi)
                tilt = (torch.rand(m, generator=self.generator, device=self.device)
                        * self.sprawl_max_tilt)
                half = tilt * 0.5
                sin_half = torch.sin(half)
                quat = torch.stack([
                    torch.cos(half),
                    sin_half * torch.cos(axis_angle),
                    sin_half * torch.sin(axis_angle),
                    torch.zeros_like(half),
                ], dim=1)
                quat = quat / quat.norm(dim=1, keepdim=True).clamp_min(1e-6)
            qpos[picked[:, None], torch.arange(3, 7, device=self.device)[None, :]] = quat.to(qpos.dtype)
            # Drop from clear air, do NOT teleport to floor height. At an arbitrary
            # orientation the body reaches ~0.5 m from the hips, so placing the root
            # at 0.25 m buried half the racer in the ground; the resulting contact
            # forces went infinite and put NaN straight into the policy input,
            # killing the run on its first iteration. From 0.6-0.8 m he simply falls
            # and lands in a sprawl within half a second, which is the state wanted.
            qpos[picked, 2] = (0.60 + torch.rand(m, generator=self.generator,
                                                 device=self.device) * 0.20).to(qpos.dtype)

        qvel[index] = 0
        self.last_action[index] = 0
        self.prev_action[index] = 0
        self.prev_qvel[index] = 0
        self.episode_step[index] = 0

        # Command ANY heading, not one within 0.6 rad of the spawn yaw. The narrow
        # band trained a racer that could only hold a lane: measured in the race
        # scene, a 90 degree correction put him on the floor in under 5 seconds.
        self.command_heading[index] = yaw + (
            torch.rand(n, generator=self.generator, device=self.device) * 2 - 1) * torch.pi
        # The lane starts where he stands and runs along the sampled heading, which
        # may be anywhere up to 180 degrees from his facing: the capped steering then
        # turns him onto it a stride at a time, as the race would.
        self.lane_heading[index] = self.command_heading[index]
        self.lane_origin[index] = self.qpos[index, 0:2].float()
        self.cross_track[index] = 0
        # Commanded speed, sampled per episode when a band is configured.
        #
        # It was a hardcoded TARGET_SPEED on every reset, which made observation
        # index 11 CONSTANT across all training -- zero variance, so RunningNorm
        # held var ~ 0 for it and any other value normalised to a saturated +/-10.
        # The policy therefore never learned what the input means: E12 tracks 1.5
        # m/s beautifully (-1.1 %) and collapses to 0.325 m/s when commanded 2.0
        # (M11). The observation was decoration, and one Unity setter away from
        # crippling every MuJoCo racer.
        #
        # Sampling it gives the input real variance, which is what makes it
        # learnable -- and a policy that genuinely tracks a command is worth more
        # than either current brain, because the game can then choose the pace.
        if self.command_speed_range is None:
            self.command_speed[index] = TARGET_SPEED
        else:
            low, high = self.command_speed_range
            self.command_speed[index] = (
                low + torch.rand(n, generator=self.generator, device=self.device) * (high - low)
            ).to(self.command_speed.dtype)

        self._randomise(index)
        if self.lane_follow:
            # So the very first observation after a reset is already capped.
            self._steer_to_lane()

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

        if self.lane_follow:
            self._steer_to_lane()
        obs = self.observation()
        reward, terms = self._reward(prev_qvel, torch.zeros(self.num_worlds, device=self.device))

        self.episode_step += 1
        height = self.qpos[:, 2]
        # Measured: as a racer collapses under zero action (height 0.754 ->
        # 0.153) obs[:, 2] runs -0.684 -> -0.057, so uprightness is -obs[:, 2],
        # positive when upright. Reuse the obs computed above rather than
        # calling observation() a second time.
        upright = -obs[:, 2]
        fallen = (height < MIN_HEIGHT) | (upright < MIN_UPRIGHT)

        # A fall does NOT end the episode. The racer is never picked up, in
        # training or in the race, so it has to learn to get itself back on its
        # feet -- and it can only learn that by living through the consequences of
        # going down. Terminating on a fall teaches the opposite: that the floor is
        # an absorbing state, which is exactly the policy that then lies there.
        timeout = self.episode_step >= EPISODE_STEPS
        # Optional early termination. The comment above is right that terminating
        # on a fall teaches the floor is absorbing and prevents get-up ever being
        # learned -- but that only bites once the racer can stand at all. With the
        # M9 inversion exploit closed, two full-length runs converged instead to
        # lying down (torso 0.10-0.14 m against a 0.77 m stance), because nothing
        # ends an episode and 1000 steps of lying still is a comfortable local
        # optimum. Terminating on a fall is what every standard humanoid
        # locomotion benchmark does, and it is stage one of a curriculum: learn
        # balance with it on, then relearn get-up with it off.
        #
        # It also un-degenerates the metrics of M4: with a real terminal
        # condition, episode length and survival rate measure something again.
        done = (timeout | fallen) if self.terminate_on_fall else timeout

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

    def _steer_to_lane(self) -> None:
        """Pure pursuit onto the lane, exactly as the game steers him: aim
        LANE_LOOKAHEAD metres ahead on the line, then cap the heading error handed
        to the policy at LANE_MAX_TURN from his current facing."""
        lane_dir = torch.stack([torch.cos(self.lane_heading), torch.sin(self.lane_heading)], dim=1)
        offset = self.qpos[:, 0:2].float() - self.lane_origin
        # Signed distance off the line, positive to the LEFT of the lane direction.
        self.cross_track = lane_dir[:, 0] * offset[:, 1] - lane_dir[:, 1] * offset[:, 0]
        aim = self.lane_heading + torch.atan2(-self.cross_track,
                                              torch.full_like(self.cross_track, LANE_LOOKAHEAD))
        forward = self.root_rotation()[:, :, FORWARD_AXIS]
        facing = torch.atan2(forward[:, 1], forward[:, 0])
        error = (aim - facing + torch.pi) % (2 * torch.pi) - torch.pi
        self.command_heading = facing + error.clamp(-LANE_MAX_TURN, LANE_MAX_TURN)

    def _reward(self, prev_joint_qvel: torch.Tensor, fallen_now: torch.Tensor):
        rot = self.root_rotation()
        qvel = self.qvel
        # Uprightness is +rot[2,2], NOT -rot[2,2].
        #
        # reset() builds the standing orientation as a pure yaw quaternion
        # (qpos[4] = qpos[5] = 0), so rot[2,2] = 1 - 2(x^2 + y^2) = +1 for an
        # upright racer. The old sign made obs_gravity_z = -1 while standing,
        # which clamp(min=0) then floored to zero: W_UPRIGHT paid nothing for
        # standing, `standing` was 0 upright and 1 inverted, and since W_TRACK,
        # W_HEADING and W_GETUP are all gated on `standing`, the only way to earn
        # any of them was to turn upside down. The policy obliged -- measured on
        # the 1500-iteration run: uprightness -0.873, torso 0.585 m, travelling
        # 1.245 m/s inverted, with `standing` reporting a healthy 0.682.
        obs_gravity_z = rot[:, 2, 2]

        # Velocity along the COMMANDED heading, not along +y: the racer is steered.
        heading = self.command_heading
        cmd_dir = torch.stack([torch.cos(heading), torch.sin(heading)], dim=1)
        planar = qvel[:, 0:2].float()
        along = (planar * cmd_dir).sum(dim=1)
        lateral = (planar * torch.stack([-cmd_dir[:, 1], cmd_dir[:, 0]], dim=1)).sum(dim=1)

        # Saturating tracking term: exceeding the target speed earns nothing extra,
        # which stops the policy trading stability for a sprint it cannot hold.
        error = along - self.command_speed
        # One-sided by default, which is what shipped: clamping the positive
        # side away means running fast earns nothing extra but also costs
        # nothing, so nothing pulls an overshooting policy back to the command.
        # Two-sided makes the kernel symmetric about the commanded speed.
        shortfall = error if self.two_sided_speed else error.clamp(max=0.0)
        track = torch.exp(-(shortfall / SPEED_SIGMA) ** 2)

        forward = rot[:, :, FORWARD_AXIS]
        facing = (forward[:, 0] * cmd_dir[:, 0] + forward[:, 1] * cmd_dir[:, 1])

        joint_qvel = qvel[:, self.dof_addr].float()
        accel = ((joint_qvel - prev_joint_qvel.float()) / self.dt).pow(2).sum(dim=1)
        action_rate = (self.last_action - self.prev_action).pow(2).mean(dim=1)
        ctrl_cost = self.last_action.pow(2).mean(dim=1)

        # APPLIED TORQUE, which is what "control effort" is supposed to mean.
        #
        # `ctrl_cost` above is the mean squared ACTION, and the actuators are position
        # servos -- `<general biastype="affine" gainprm="1200 0 0"
        # biasprm="0 -1200 -78.43" ctrlrange="+-45deg" forcerange="+-200">`, so
        # force = 1200*(ctrl - qpos) - 78.43*qvel clamped to +-200 N.m. `ctrl` is a
        # commanded JOINT ANGLE, not a torque, and the servo saturates at a position
        # error of only 200/1200 = 0.167 rad = 9.5 deg. So `ctrl_cost` measures how
        # extreme a POSE the policy asks for, which is not the same quantity as how
        # hard the actuators work, and beyond 9.5 deg of error it is not even
        # monotonically related to it.
        #
        # CLAUDE.md states this exact principle as a project rule: "Read load directly
        # from applied torque, not the action vector (isometric bracing produces
        # near-zero action at near-maximum torque)." K6 has been measured against the
        # action vector for this project's whole history, in violation of it.
        #
        # Reported, never charged: adding it to `reward` would change what every
        # policy optimises and break comparability with every run already recorded.
        torque = wp.to_torch(self.wd.qfrc_actuator).float()[:, self.dof_addr]
        torque_cost = torque.pow(2).mean(dim=1)
        drift = lateral.pow(2) + self.drift_yaw * qvel[:, 5].float().pow(2)

        # Impact, expressed as constraint force in units of body weight. The raw
        # qfrc_constraint includes the ground reaction that simply holds the racer
        # up, so penalising it directly penalises standing; only the EXCESS over
        # carrying its own weight is an impact worth discouraging.
        force = wp.to_torch(self.wd.qfrc_constraint).float().norm(dim=1)
        weight = float(self.mjm.body_mass.sum() * abs(self.mjm.opt.gravity[2]))
        impact = (force / weight - 1.5).clamp(min=0.0)

        # How much of a standing racer this is right now: 1 upright at full height,
        # 0 flat on the floor. Speed and heading rewards are GATED on it, so a
        # racer cannot farm the tracking term by sliding along on its face, and
        # getting back up is the only route to the large rewards.
        height = self.qpos[:, 2].float()
        standing = (obs_gravity_z.clamp(min=0.0)
                    * (height / STANDING_HEIGHT).clamp(0.0, 1.0))

        reward = (
            W_TRACK * track * standing
            + self.heading_weight * facing.clamp(min=0.0) * standing
            + self.upright_weight * obs_gravity_z.clamp(min=0.0)
            + W_GETUP * standing
            + W_ALIVE
            - self.drift_weight * torch.tanh(drift / self.scale_drift)
            - self.ctrl_weight * ctrl_cost
            - W_ACCEL * torch.tanh(accel / SCALE_ACCEL)
            - W_IMPACT * torch.tanh(impact)
            - W_ACTION_RATE * action_rate
        )
        xtrack = self.cross_track.abs()
        if self.lane_follow:
            reward = reward - self.xtrack_weight * torch.tanh(xtrack / SCALE_XTRACK)
        # Everything the reward actually charges for is reported, so the ten
        # weights above can be tuned against evidence instead of against the
        # single scalar return. ctrl/action_rate/lateral/upright/height were
        # charged but not reported until now.
        terms = {
            "track": track, "facing": facing, "speed_along": along,
            "drift": drift, "accel": accel, "impact": impact,
            "standing": standing, "fallen": fallen_now,
            "ctrl": ctrl_cost, "action_rate": action_rate,
            # The honest control-effort measure; see the note beside torque_cost.
            "torque": torque_cost, "torque_abs": torque.abs().mean(dim=1),
            "lateral": lateral.abs(), "upright": obs_gravity_z.clamp(min=0.0),
            "height": height, "xtrack": xtrack,
        }
        return reward, terms

    def contact_overflow(self) -> int:
        """Peak contacts this step. The trainer aborts if this reaches naconmax:
        an overflow silently drops contacts and the creature sinks."""
        return int(wp.to_torch(self.wd.nacon).max().item())

    @property
    def naconmax(self) -> int:
        return int(self.wd.naconmax)
