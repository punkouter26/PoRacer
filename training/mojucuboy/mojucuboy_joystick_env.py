"""MojucuBoy on a joystick: velocity commands and a gait clock, after MuJoCo Playground.

  .venv-mujoco\\Scripts\\python.exe training/mojucuboy/train_mojucuboy.py --task joystick

Why this exists. The heading task (mojucuboy_env.py) tells him "face this way, walk at
1.5 m/s" and taxes sideways motion as a side effect. Nine runs of tuning that tax
(rl_optimization_log M29-M37) never made him hold a lane: in the race he drifted 4-6 m
sideways within 15 m. MuJoCo Playground's Unitree G1 joystick task
(mujoco_playground/_src/locomotion/g1/joystick.py) solves the same problem another way,
and this module ports its recipe onto MojucuBoy's body:

  * The command is a body-frame VELOCITY: forward vx, sideways vy, yaw rate wz. Sideways
    speed is something he is PAID to match - including zero - instead of a penalty that
    fights the weight shifts walking is made of. The race can also sidestep him back
    into his lane instead of turning him.
  * A gait clock (phase) in the observation, and rewards for feet swinging on that clock
    and landing with real air time, without sliding. An asymmetric stride is the usual
    reason a biped veers; the heading task gives him no rhythm at all.
  * A fall ENDS the episode (Playground and every standard locomotion benchmark). So this
    policy learns balance, not recovery: getting up is a separate curriculum stage.

The per-step reward is clipped at zero, as Playground's is, so an episode that ends
early simply forfeits the reward it would have earned; there is no reward for dying.

Observation (77), the Unity contract (MojucuBoyObservation.cs, joystick layout):
   0..2   gravity in the body frame
   3..5   linear velocity, body frame
   6..8   angular velocity, body frame
   9..11  command: vx (m/s, forward = body +Y), vy (m/s, left = body -X), wz (rad/s)
  12..13  gait clock: cos(phase), sin(phase) of the LEFT foot; the right foot is phase + pi
  14..34  joint angles (actuator order)
  35..55  joint velocities
  56..76  previous action
"""

from __future__ import annotations

import math

import mujoco
import torch
import warp as wp

import mojucuboy_env
from mojucuboy_env import (ACTION_SCALE, ACTION_SIZE, DECIMATION, EPISODE_STEPS, FORWARD_AXIS,
                           JOINT_COUNT, MIN_HEIGHT, MIN_UPRIGHT, MojucuBoyEnv)

OBS_SIZE = 77

# Command ranges. Playground's G1 uses vx [-1, 1]; MojucuBoy is human-sized and races at
# 1.5 m/s, so forward runs to 2.0 and backward is dropped. Sideways and turning follow G1.
COMMAND_VX = (0.0, 2.0)
COMMAND_VY = (-0.5, 0.5)
COMMAND_WZ = (-1.0, 1.0)
ZERO_COMMAND_PROBABILITY = 0.1     # Playground: 10 % of commands are "stand still"
COMMAND_RESAMPLE_STEPS = (150, 400)  # mid-episode changes, so he learns transitions
STILL_COMMAND = 0.1                # below this command norm he is meant to stand

# Gait clock: full cycles per second, sampled per episode. A 1.7 m adult walks at about
# 0.9-1.0 strides/s at 1.5 m/s; the band leaves room either side.
GAIT_FREQUENCY = (0.9, 1.4)
SWING_HEIGHT = 0.10                # peak sole height in swing, metres

# Reward weights, from Playground's G1 joystick config. Their per-step rewards are scaled
# by dt; these are not, but every term here is, so the RATIOS are Playground's.
W_TRACK_LIN = 1.0
W_TRACK_ANG = 0.75
TRACKING_SIGMA = 0.25
W_ORIENTATION = -2.0
W_ANG_VEL_XY = -0.15
W_FEET_AIR_TIME = 2.0
W_FEET_PHASE = 1.0
W_FEET_SLIP = -0.25
W_STAND_STILL = -1.0
W_HIP_DEVIATION = -0.25
W_KNEE_DEVIATION = -0.1
W_ACTION_RATE = -0.01              # not in G1's config; kept from ours against twitching
AIR_TIME_MIN = 0.2                 # seconds of swing that start earning
AIR_TIME_MAX = 0.5

# Soles: box geoms 0.0428 m half-height. In contact below this sole height.
FOOT_GEOMS = ("foot_L", "foot_R")
FOOT_HALF_HEIGHT = 0.0428
CONTACT_HEIGHT = 0.02

# Pushes, Playground-style: every 5-10 s, up to 1.5 m/s on the base.
PUSH_STEPS = (250, 500)
PUSH_MAX = 1.5

HIP_DEVIATION_JOINTS = ("hip_z_L", "hip_y_L", "hip_z_R", "hip_y_R")
KNEE_JOINTS = ("knee_L", "knee_R")


class MojucuBoyJoystickEnv(MojucuBoyEnv):
    obs_size = OBS_SIZE

    def __init__(self, num_worlds: int, device: str = "cuda:0", seed: int = 0):
        self._joystick_ready = False
        # Never start fallen: this task terminates on a fall, so a sprawl start is an
        # episode that ends on its first step.
        super().__init__(num_worlds, device=device, seed=seed, reset_fallen_fraction=0.0,
                         terminate_on_fall=True)
        order = self.rig["actuator_order"]
        self.hip_index = torch.tensor([order.index(n) for n in HIP_DEVIATION_JOINTS], device=self.device)
        self.knee_index = torch.tensor([order.index(n) for n in KNEE_JOINTS], device=self.device)
        self.foot_geoms = torch.tensor(
            [mujoco.mj_name2id(self.mjm, mujoco.mjtObj.mjOBJ_GEOM, n) for n in FOOT_GEOMS],
            device=self.device, dtype=torch.long)

    # ---- per-world joystick state -------------------------------------------
    def _ensure_state(self) -> None:
        if self._joystick_ready:
            return
        n, dev = self.num_worlds, self.device
        self.command = torch.zeros(n, 3, device=dev)
        self.command_timer = torch.zeros(n, device=dev, dtype=torch.long)
        self.phase = torch.zeros(n, device=dev)
        self.gait_frequency = torch.ones(n, device=dev)
        self.air_time = torch.zeros(n, 2, device=dev)
        self.last_contact = torch.zeros(n, 2, device=dev, dtype=torch.bool)
        self.last_foot_xy = torch.zeros(n, 2, 2, device=dev)
        self.push_timer = torch.zeros(n, device=dev, dtype=torch.long)
        self._joystick_ready = True

    def _uniform(self, n: int, low: float, high: float) -> torch.Tensor:
        return low + torch.rand(n, generator=self.generator, device=self.device) * (high - low)

    def _randint(self, n: int, bounds) -> torch.Tensor:
        return torch.randint(bounds[0], bounds[1] + 1, (n,), generator=self.generator,
                             device=self.device)

    def _sample_command(self, index: torch.Tensor) -> None:
        n = index.numel()
        if n == 0:
            return
        command = torch.stack([self._uniform(n, *COMMAND_VX),
                               self._uniform(n, *COMMAND_VY),
                               self._uniform(n, *COMMAND_WZ)], dim=1)
        still = torch.rand(n, generator=self.generator, device=self.device) < ZERO_COMMAND_PROBABILITY
        command[still] = 0.0
        self.command[index] = command
        self.command_timer[index] = self._randint(n, COMMAND_RESAMPLE_STEPS)

    def reset(self, index: torch.Tensor) -> None:
        super().reset(index)
        if index.numel() == 0:
            return
        if not hasattr(self, "_joystick_ready"):
            return
        self._ensure_state()
        n = index.numel()
        self._sample_command(index)
        self.phase[index] = self._uniform(n, 0.0, 2 * math.pi)
        self.gait_frequency[index] = self._uniform(n, *GAIT_FREQUENCY)
        self.air_time[index] = 0
        self.last_contact[index] = True
        self.push_timer[index] = self._randint(n, PUSH_STEPS)
        if hasattr(self, "foot_geoms"):
            self.last_foot_xy[index] = self._foot_positions()[index, :, 0:2]

    # ---- helpers --------------------------------------------------------------
    def _foot_positions(self) -> torch.Tensor:
        """(N, 2, 3) sole-centre positions: geom centre dropped by the box half-height."""
        xpos = wp.to_torch(self.wd.geom_xpos).float()[:, self.foot_geoms]
        sole = xpos.clone()
        sole[:, :, 2] -= FOOT_HALF_HEIGHT
        return sole

    # ---- observation -----------------------------------------------------------
    def observation(self) -> torch.Tensor:
        self._ensure_state()
        rot = self.root_rotation()
        qpos, qvel = self.qpos, self.qvel
        obs = torch.empty(self.num_worlds, OBS_SIZE, device=self.device)
        obs[:, 0:3] = -rot[:, 2, :]
        obs[:, 3:6] = torch.bmm(rot.transpose(1, 2), qvel[:, 0:3].unsqueeze(2)).squeeze(2)
        obs[:, 6:9] = qvel[:, 3:6]
        obs[:, 9:12] = self.command
        obs[:, 12] = torch.cos(self.phase)
        obs[:, 13] = torch.sin(self.phase)
        obs[:, 14:14 + JOINT_COUNT] = qpos[:, self.qpos_addr]
        obs[:, 14 + JOINT_COUNT:14 + 2 * JOINT_COUNT] = qvel[:, self.dof_addr]
        obs[:, 14 + 2 * JOINT_COUNT:] = self.last_action
        return torch.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    # ---- step -------------------------------------------------------------------
    def step(self, action: torch.Tensor):
        self._ensure_state()
        action = torch.clamp(action, -1.0, 1.0)
        self.prev_action = self.last_action
        self.last_action = action
        target = torch.clamp(self.stance_joints + ACTION_SCALE * self.joint_half * action,
                             self.joint_lo, self.joint_hi)
        ctrl = wp.to_torch(self.wd.ctrl)
        ctrl[:] = target.to(ctrl.dtype)
        for _ in range(DECIMATION):
            mujoco_warp_step(self)

        self.phase = torch.remainder(self.phase + 2 * math.pi * self.gait_frequency * self.dt,
                                     2 * math.pi)
        obs = self.observation()
        reward, terms = self._joystick_reward(obs)

        self.episode_step += 1
        height = self.qpos[:, 2]
        upright = -obs[:, 2]
        fallen = (height < MIN_HEIGHT) | (upright < MIN_UPRIGHT)
        timeout = self.episode_step >= EPISODE_STEPS
        done = timeout | fallen

        # Mid-episode command changes, and Playground-style pushes.
        self.command_timer -= 1
        change = (self.command_timer <= 0) & ~done
        if change.any():
            self._sample_command(change.nonzero(as_tuple=True)[0])
        self.push_timer -= 1
        push = (self.push_timer <= 0) & ~done
        if push.any():
            idx = push.nonzero(as_tuple=True)[0]
            angle = self._uniform(idx.numel(), 0.0, 2 * math.pi)
            magnitude = self._uniform(idx.numel(), 0.1, PUSH_MAX)
            qvel = self.qvel
            qvel[idx, 0] += (magnitude * torch.cos(angle)).to(qvel.dtype)
            qvel[idx, 1] += (magnitude * torch.sin(angle)).to(qvel.dtype)
            self.push_timer[idx] = self._randint(idx.numel(), PUSH_STEPS)

        terms["fallen"] = fallen.float()
        terms["timeout"] = timeout.float()
        return obs, reward, done, terms

    def _joystick_reward(self, obs: torch.Tensor):
        linvel = obs[:, 3:6]
        angvel = obs[:, 6:9]
        gravity = obs[:, 0:3]
        # Body frame: forward is +Y (FORWARD_AXIS 1), left is -X. The command's vx is
        # forward and vy is left, so the matching body velocities are (y, -x).
        vel_forward = linvel[:, FORWARD_AXIS]
        vel_left = -linvel[:, 0]
        wz = angvel[:, 2]
        cmd = self.command

        lin_error = (cmd[:, 0] - vel_forward).pow(2) + (cmd[:, 1] - vel_left).pow(2)
        track_lin = torch.exp(-lin_error / TRACKING_SIGMA)
        track_ang = torch.exp(-(cmd[:, 2] - wz).pow(2) / TRACKING_SIGMA)
        orientation = gravity[:, 0].pow(2) + gravity[:, 1].pow(2)
        ang_vel_xy = (angvel[:, 0].pow(2) + angvel[:, 1].pow(2)).clamp(max=25.0)

        feet = self._foot_positions()
        sole_height = feet[:, :, 2]
        contact = sole_height < CONTACT_HEIGHT
        foot_velocity = (feet[:, :, 0:2] - self.last_foot_xy) / self.dt
        self.last_foot_xy = feet[:, :, 0:2].clone()
        slip = (foot_velocity.pow(2).sum(-1) * contact.float()).sum(-1).clamp(max=25.0)

        moving = cmd.norm(dim=1) > STILL_COMMAND
        # Playground's feet_air_time: paid once per touchdown, for the swing just ended.
        first_contact = contact & ~self.last_contact
        swing = self.air_time + self.dt
        air_reward = ((swing - AIR_TIME_MIN).clamp(max=AIR_TIME_MAX - AIR_TIME_MIN)
                      * first_contact.float()).sum(-1) * moving.float()
        self.air_time = torch.where(contact, torch.zeros_like(swing), swing)
        self.last_contact = contact

        # Swing target on the clock: left foot on phase, right half a cycle later.
        phases = torch.stack([self.phase, self.phase + math.pi], dim=1)
        target_height = SWING_HEIGHT * torch.sin(phases).clamp(min=0.0)
        phase_error = (sole_height.clamp(min=0.0) - target_height).pow(2).sum(-1)
        feet_phase = torch.exp(-phase_error / 0.01) * moving.float()

        deviation = (self.qpos[:, self.qpos_addr].float() - self.stance_joints)
        stand_still = deviation.abs().sum(-1) * (~moving).float()
        hip_deviation = deviation[:, self.hip_index].abs().sum(-1)
        knee_deviation = deviation[:, self.knee_index].abs().sum(-1)
        action_rate = (self.last_action - self.prev_action).pow(2).mean(dim=1)

        reward = (W_TRACK_LIN * track_lin
                  + W_TRACK_ANG * track_ang
                  + W_ORIENTATION * orientation
                  + W_ANG_VEL_XY * ang_vel_xy
                  + W_FEET_AIR_TIME * air_reward
                  + W_FEET_PHASE * feet_phase
                  + W_FEET_SLIP * slip
                  + W_STAND_STILL * stand_still
                  + W_HIP_DEVIATION * hip_deviation
                  + W_KNEE_DEVIATION * knee_deviation
                  + W_ACTION_RATE * action_rate)
        # Playground clips the step reward at zero: dying forfeits future reward rather
        # than escaping accumulated penalties.
        reward = reward.clamp(min=0.0)

        terms = {
            "track_lin": track_lin, "track_ang": track_ang,
            "orientation": orientation, "ang_vel_xy": ang_vel_xy,
            "air_time": air_reward, "feet_phase": feet_phase, "slip": slip,
            "stand_still": stand_still, "hip_dev": hip_deviation,
            "vel_forward_err": (cmd[:, 0] - vel_forward).abs(),
            "vel_left_err": (cmd[:, 1] - vel_left).abs(),
            "yaw_rate_err": (cmd[:, 2] - wz).abs(),
            "speed_along": vel_forward,
            "upright": (-gravity[:, 2]).clamp(min=0.0),
            "height": self.qpos[:, 2].float(),
        }
        return reward, terms


def mujoco_warp_step(env: MojucuBoyEnv) -> None:
    mojucuboy_env.mjw.step(env.wm, env.wd)
