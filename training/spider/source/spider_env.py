"""8-legged spider that learns to walk to a target."""

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
from isaaclab.utils.math import quat_apply_inverse, sample_uniform, yaw_quat

if TYPE_CHECKING:
    from .spider_env_cfg import SpiderEnvCfg

LEGS = ["L1", "L2", "L3", "L4", "R1", "R2", "R3", "R4"]
JOINT_NAMES = [f"{leg}_{j}" for leg in LEGS for j in ("hip", "knee")]


class SpiderEnv(DirectRLEnv):
    cfg: SpiderEnvCfg

    def __init__(self, cfg: SpiderEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._joint_ids, joint_names = self.spider.find_joints(JOINT_NAMES, preserve_order=True)
        print(f"[SpiderEnv] controlling {len(joint_names)} joints: {joint_names}")

        self.targets_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_dist = torch.zeros(self.num_envs, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.actions = torch.zeros_like(self.prev_actions)
        self._reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._fallen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    # ------------------------------------------------------------------ scene
    def _setup_scene(self):
        self.spider = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["spider"] = self.spider
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.target_markers = VisualizationMarkers(self.cfg.target_marker_cfg)

    # ---------------------------------------------------------------- actions
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions = self.actions.clone()
        self.actions = actions.clone().clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        self.spider.set_joint_position_target_index(target=self.cfg.action_scale * self.actions, joint_ids=self._joint_ids)

    # ----------------------------------------------------------------- helpers
    def _target_vec_b(self) -> torch.Tensor:
        """Target position relative to the body, expressed in the body's yaw-only frame (x forward)."""
        root_pos = self.spider.data.root_pos_w.torch
        root_quat = self.spider.data.root_quat_w.torch
        return quat_apply_inverse(yaw_quat(root_quat), self.targets_w - root_pos)

    def _distance(self) -> torch.Tensor:
        d = self.targets_w[:, :2] - self.spider.data.root_pos_w.torch[:, :2]
        return torch.norm(d, dim=-1)

    def _sample_targets(self, env_ids: torch.Tensor):
        n = len(env_ids)
        r = sample_uniform(self.cfg.target_radius_range[0], self.cfg.target_radius_range[1], (n,), self.device)
        ang = sample_uniform(-math.pi, math.pi, (n,), self.device)
        self.targets_w[env_ids, 0] = self.scene.env_origins[env_ids, 0] + r * torch.cos(ang)
        self.targets_w[env_ids, 1] = self.scene.env_origins[env_ids, 1] + r * torch.sin(ang)
        self.targets_w[env_ids, 2] = 0.12

    # ------------------------------------------------------------ observations
    def _get_observations(self) -> dict:
        d = self.spider.data
        tv = self._target_vec_b()
        obs = torch.cat(
            (
                d.joint_pos.torch[:, self._joint_ids],  # 16
                d.joint_vel.torch[:, self._joint_ids] * 0.1,  # 16
                d.root_lin_vel_b.torch,  # 3
                d.root_ang_vel_b.torch * 0.2,  # 3
                d.projected_gravity_b.torch,  # 3
                tv[:, :2],  # 2  where is the target (yaw frame)
                self.actions,  # 16 last action
            ),
            dim=-1,
        )
        self.target_markers.visualize(translations=self.targets_w)
        return {"policy": obs}

    # ----------------------------------------------------------------- rewards
    def _get_rewards(self) -> torch.Tensor:
        dist = self._distance()
        progress = self.prev_dist - dist
        self.prev_dist = dist

        tv = self._target_vec_b()
        heading_cos = tv[:, 0] / (torch.norm(tv[:, :2], dim=-1) + 1e-6)

        self._reached = dist < self.cfg.reach_threshold
        upright = -self.spider.data.projected_gravity_b.torch[:, 2]
        action_rate = torch.sum(torch.square(self.actions - self.prev_actions), dim=-1)
        joint_vel = torch.sum(torch.square(self.spider.data.joint_vel.torch[:, self._joint_ids]), dim=-1)

        reward = (
            self.cfg.rew_progress * progress
            + self.cfg.rew_heading * heading_cos
            + self.cfg.rew_reach * self._reached.float()
            + self.cfg.rew_upright * upright
            + self.cfg.rew_action_rate * action_rate
            + self.cfg.rew_joint_vel * joint_vel
            + self.cfg.rew_fall * self._fallen.float()
        )

        # target reached -> give the spider a new one (same episode keeps going)
        reached_ids = self._reached.nonzero(as_tuple=False).squeeze(-1)
        if len(reached_ids) > 0:
            self._sample_targets(reached_ids)
            self.prev_dist[reached_ids] = self._distance()[reached_ids]

        log = self.extras.setdefault("log", {})
        log["Metrics/dist_to_target"] = dist.mean().item()
        log["Metrics/reached_frac"] = self._reached.float().mean().item()
        return reward

    # ------------------------------------------------------------------- dones
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        d = self.spider.data
        flipped = d.projected_gravity_b.torch[:, 2] > 0.0  # belly up
        collapsed = d.root_pos_w.torch[:, 2] < self.cfg.min_body_height
        flew = d.root_pos_w.torch[:, 2] > 1.0
        self._fallen = flipped | collapsed | flew
        return self._fallen, time_out

    # ------------------------------------------------------------------- reset
    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.spider._ALL_INDICES
        env_ids = torch.as_tensor(env_ids, device=self.device)
        super()._reset_idx(env_ids)

        joint_pos = self.spider.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.spider.data.default_joint_vel.torch[env_ids].clone()
        root_pose = self.spider.data.default_root_pose.torch[env_ids].clone()
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_vel = self.spider.data.default_root_vel.torch[env_ids].clone()

        self.spider.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.spider.write_root_velocity_to_sim_index(root_velocity=root_vel, env_ids=env_ids)
        self.spider.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.spider.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        self._sample_targets(env_ids)
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self._fallen[env_ids] = False
        self.prev_dist[env_ids] = torch.norm(self.targets_w[env_ids, :2] - root_pose[:, :2], dim=-1)
