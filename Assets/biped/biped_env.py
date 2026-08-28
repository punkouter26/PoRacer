"""A two-legged robot that learns to walk to a target."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, sample_uniform, yaw_quat

if TYPE_CHECKING:
    from .biped_env_cfg import BipedEnvCfg

# action order: left leg then right leg, proximal to distal
JOINT_NAMES = [f"{s}_{j}" for s in ("L", "R") for j in ("hip_yaw", "hip_roll", "hip_pitch", "knee", "ankle")]


class BipedEnv(DirectRLEnv):
    cfg: BipedEnvCfg

    def __init__(self, cfg: BipedEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids, joint_names = self.robot.find_joints(JOINT_NAMES, preserve_order=True)
        print(f"[BipedEnv] controlling {len(joint_names)} joints: {joint_names}")

        # the policy acts around the nominal standing pose, not around zero
        self.default_q = self.robot.data.default_joint_pos.torch[:, self._joint_ids].clone()

        self._foot_ids, foot_names = self.robot.find_bodies(".*_foot", preserve_order=True)
        print(f"[BipedEnv] feet: {foot_names}")

        self.targets_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.actions = torch.zeros_like(self.prev_actions)
        self._reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.foot_air_time = torch.zeros(self.num_envs, len(self._foot_ids), device=self.device)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.target_markers = VisualizationMarkers(self.cfg.target_marker_cfg)

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        target = self.default_q + self.cfg.action_scale * self.actions
        self.robot.set_joint_position_target_index(target=target, joint_ids=self._joint_ids)

    # ----------------------------------------------------------------- helpers
    def _target_vec_b(self) -> torch.Tensor:
        """Target position relative to the torso, in its yaw-only frame (x forward)."""
        root_pos = self.robot.data.root_pos_w.torch
        root_quat = self.robot.data.root_quat_w.torch
        return quat_apply_inverse(yaw_quat(root_quat), self.targets_w - root_pos)

    def _distance(self) -> torch.Tensor:
        d = self.targets_w[:, :2] - self.robot.data.root_pos_w.torch[:, :2]
        return torch.norm(d, dim=-1)

    def _sample_targets(self, env_ids: torch.Tensor):
        n = len(env_ids)
        r = sample_uniform(self.cfg.target_radius_range[0], self.cfg.target_radius_range[1], (n,), self.device)
        ang = sample_uniform(-math.pi, math.pi, (n,), self.device)
        self.targets_w[env_ids, 0] = self.scene.env_origins[env_ids, 0] + r * torch.cos(ang)
        self.targets_w[env_ids, 1] = self.scene.env_origins[env_ids, 1] + r * torch.sin(ang)
        self.targets_w[env_ids, 2] = 0.15

    # ------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        d = self.robot.data
        tv = self._target_vec_b()
        dist = torch.norm(tv[:, :2], dim=-1, keepdim=True)
        obs = torch.cat(
            (
                d.joint_pos.torch[:, self._joint_ids] - self.default_q,  # 10
                d.joint_vel.torch[:, self._joint_ids] * 0.1,  # 10
                d.root_lin_vel_b.torch,  # 3
                d.root_ang_vel_b.torch * 0.25,  # 3
                d.projected_gravity_b.torch,  # 3
                tv[:, :2] / (dist + 1e-6),  # 2  unit heading to the target
                (dist / 5.0).clamp(max=1.0),  # 1  how far away it is
                self.actions,  # 10 last action
            ),
            dim=-1,
        )
        self.target_markers.visualize(translations=self.targets_w)
        return {"policy": obs}

    # ----------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        d = self.robot.data
        dist = self._distance()
        progress = self.prev_dist - dist
        self.prev_dist = dist

        tv = self._target_vec_b()
        heading_cos = tv[:, 0] / (torch.norm(tv[:, :2], dim=-1) + 1e-6)

        self._reached = dist < self.cfg.reach_threshold
        upright = -d.projected_gravity_b.torch[:, 2]

        # stay tall: only the shortfall below the nominal hip height is penalised
        shortfall = (self.cfg.nominal_height - d.root_pos_w.torch[:, 2]).clamp(min=0.0)

        # reward real steps: on flat ground a foot is planted when its centre is low enough,
        # so swing duration can be tracked from kinematics without a contact sensor
        foot_z = d.body_pos_w.torch[:, self._foot_ids, 2]
        planted = foot_z < self.cfg.foot_contact_height
        landed = planted & (self.foot_air_time > 0.0)  # airborne last step, down now
        air_time = torch.sum((self.foot_air_time - self.cfg.air_time_target) * landed.float(), dim=1)
        air_time *= (dist > self.cfg.reach_threshold).float()  # no need to march once it has arrived
        self.foot_air_time = torch.where(planted, 0.0, self.foot_air_time + self.step_dt)

        ang_vel_xy = torch.sum(torch.square(d.root_ang_vel_b.torch[:, :2]), dim=-1)
        action_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
        joint_vel = torch.sum(torch.square(d.joint_vel.torch[:, self._joint_ids]), dim=-1)
        torque = torch.sum(torch.square(d.applied_torque.torch[:, self._joint_ids]), dim=-1)

        reward = (
            self.cfg.rew_progress * progress
            + self.cfg.rew_heading * heading_cos
            + self.cfg.rew_reach * self._reached.float()
            + self.cfg.rew_alive * (~self._fallen).float() * self.step_dt
            + self.cfg.rew_upright * upright * self.step_dt
            + self.cfg.rew_height * torch.square(shortfall)
            + self.cfg.rew_air_time * air_time
            + self.cfg.rew_ang_vel_xy * ang_vel_xy * self.step_dt
            + self.cfg.rew_action_rate * action_rate
            + self.cfg.rew_joint_vel * joint_vel * self.step_dt
            + self.cfg.rew_torque * torque * self.step_dt
            + self.cfg.rew_fall * self._fallen.float()
        )

        # target reached -> hand out a new one, the episode keeps going
        reached_ids = self._reached.nonzero(as_tuple=False).squeeze(-1)
        if len(reached_ids) > 0:
            self._sample_targets(reached_ids)
            self.prev_dist[reached_ids] = self._distance()[reached_ids]

        log = self.extras.setdefault("log", {})
        log["Metrics/dist_to_target"] = dist.mean().item()
        log["Metrics/reached_frac"] = self._reached.float().mean().item()
        log["Metrics/torso_height"] = d.root_pos_w.torch[:, 2].mean().item()
        log["Metrics/speed"] = torch.norm(d.root_lin_vel_b.torch[:, :2], dim=-1).mean().item()
        return reward

    # ------------------------------------------------------------------- dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        d = self.robot.data
        collapsed = d.root_pos_w.torch[:, 2] < self.cfg.min_height
        tipped = d.projected_gravity_b.torch[:, 2] > self.cfg.max_tilt
        self._fallen = collapsed | tipped
        return self._fallen, time_out

    # ------------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device)
        super()._reset_idx(env_ids)
        n = len(env_ids)

        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_pos += sample_uniform(-0.05, 0.05, joint_pos.shape, self.device)
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()

        root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        # random facing so the policy has to learn to turn towards the target
        yaw = sample_uniform(-math.pi, math.pi, (n,), self.device)
        zeros = torch.zeros_like(yaw)
        root_pose[:, 3:7] = quat_from_euler_xyz(zeros, zeros, yaw)
        root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()

        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self._sample_targets(env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._fallen[env_ids] = False
        self.foot_air_time[env_ids] = 0.0
        self.prev_dist[env_ids] = torch.norm(self.targets_w[env_ids, :2] - root_pose[:, :2], dim=-1)
