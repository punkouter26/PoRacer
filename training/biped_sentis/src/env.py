"""A 12-DoF biped that must walk to a goal marker.

Control runs at 40 Hz (0.005 s physics step, frame_skip=5). Observations are
expressed in the robot's own frame, so the policy learns "walk toward the
direction I sense" instead of memorising world-frame headings. That is what
makes the goal-reaching behaviour generalise to any target placement.
"""

from __future__ import annotations

import os

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "biped.xml")

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": 2,
    "distance": 5.0,
    "lookat": np.array([0.0, 0.0, 0.8]),
    "elevation": -15.0,
}

OBS_DIM = 49


class BipedTargetEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 40,
    }

    def __init__(
        self,
        frame_skip: int = 5,
        forward_reward_weight: float = 4.0,
        heading_reward_weight: float = 0.1,
        healthy_reward: float = 0.5,
        fall_penalty: float = 25.0,
        ctrl_cost_weight: float = 0.1,
        reach_bonus: float = 40.0,
        reach_radius: float = 0.6,
        healthy_z_range: tuple = (0.55, 1.10),
        min_uprightness: float = 0.4,
        target_distance_range: tuple = (3.0, 6.0),
        target_angle_range: float = 2.4,
        max_speed_reward: float = 2.0,
        reset_noise_scale: float = 0.02,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            frame_skip,
            forward_reward_weight,
            heading_reward_weight,
            healthy_reward,
            fall_penalty,
            ctrl_cost_weight,
            reach_bonus,
            reach_radius,
            healthy_z_range,
            min_uprightness,
            target_distance_range,
            target_angle_range,
            max_speed_reward,
            reset_noise_scale,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._heading_reward_weight = heading_reward_weight
        self._healthy_reward = healthy_reward
        self._fall_penalty = fall_penalty
        self._ctrl_cost_weight = ctrl_cost_weight
        self._reach_bonus = reach_bonus
        self._reach_radius = reach_radius
        self._healthy_z_range = healthy_z_range
        self._min_uprightness = min_uprightness
        self._target_distance_range = target_distance_range
        self._target_angle_range = target_angle_range
        self._max_speed_reward = max_speed_reward
        self._reset_noise_scale = reset_noise_scale

        # 1 height + 3 gravity + 3 linvel + 3 angvel + 12 qpos + 12 qvel
        # + 2 target direction + 1 target distance + 12 last action = 49
        observation_space = Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float64)

        MujocoEnv.__init__(
            self,
            XML_PATH,
            frame_skip,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

        self._last_action = np.zeros(self.model.nu)
        self._prev_distance = 0.0
        self._targets_reached = 0

    # ------------------------------------------------------------------ utils

    @property
    def _torso_xy(self) -> np.ndarray:
        return self.data.qpos[0:2].copy()

    @property
    def _target_xy(self) -> np.ndarray:
        return self.data.mocap_pos[0][:2].copy()

    def _rotation_matrix(self) -> np.ndarray:
        """Torso->world rotation, from the free joint quaternion (w, x, y, z)."""
        w, x, y, z = self.data.qpos[3:7]
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
            ]
        )

    def _heading(self) -> float:
        rot = self._rotation_matrix()
        return float(np.arctan2(rot[1, 0], rot[0, 0]))

    def _sample_target(self, origin: np.ndarray, heading: float) -> None:
        """Place the goal marker relative to where the robot is and faces."""
        distance = self.np_random.uniform(*self._target_distance_range)
        angle = heading + self.np_random.uniform(
            -self._target_angle_range, self._target_angle_range
        )
        xy = origin + distance * np.array([np.cos(angle), np.sin(angle)])
        self.data.mocap_pos[0] = np.array([xy[0], xy[1], 0.02])

    # ------------------------------------------------------------------- core

    def _get_obs(self) -> np.ndarray:
        rot = self._rotation_matrix()
        qpos, qvel = self.data.qpos, self.data.qvel

        # Gravity in the torso frame encodes roll/pitch without leaking heading.
        projected_gravity = rot.T @ np.array([0.0, 0.0, -1.0])
        local_linvel = rot.T @ qvel[0:3]
        local_angvel = rot.T @ qvel[3:6]

        to_target = self._target_xy - self._torso_xy
        distance = float(np.linalg.norm(to_target))
        world_dir = to_target / max(distance, 1e-6)
        heading = self._heading()
        cos_h, sin_h = np.cos(-heading), np.sin(-heading)
        local_dir = np.array(
            [
                cos_h * world_dir[0] - sin_h * world_dir[1],
                sin_h * world_dir[0] + cos_h * world_dir[1],
            ]
        )

        return np.concatenate(
            [
                [qpos[2]],
                projected_gravity,
                np.clip(local_linvel, -10.0, 10.0),
                np.clip(local_angvel, -10.0, 10.0),
                qpos[7:],
                np.clip(qvel[6:], -20.0, 20.0),
                local_dir,
                [min(distance, 10.0)],
                self._last_action,
            ]
        ).astype(np.float64)

    @property
    def _is_healthy(self) -> bool:
        z_min, z_max = self._healthy_z_range
        upright = self._rotation_matrix()[2, 2]
        return bool(
            np.isfinite(self.state_vector()).all()
            and z_min < self.data.qpos[2] < z_max
            and upright > self._min_uprightness
        )

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)

        to_target = self._target_xy - self._torso_xy
        distance = float(np.linalg.norm(to_target))

        # Closing speed toward the goal, in m/s.
        progress = (self._prev_distance - distance) / self.dt
        progress = float(np.clip(progress, -self._max_speed_reward, self._max_speed_reward))
        self._prev_distance = distance

        # Rewarding "face the goal" is what makes the robot learn to turn.
        world_dir = to_target / max(distance, 1e-6)
        heading = self._heading()
        facing = float(world_dir[0] * np.cos(heading) + world_dir[1] * np.sin(heading))

        ctrl_cost = self._ctrl_cost_weight * float(np.mean(np.square(action)))
        healthy = self._healthy_reward if self._is_healthy else 0.0

        reward = (
            self._forward_reward_weight * progress
            + self._heading_reward_weight * facing
            + healthy
            - ctrl_cost
        )

        reached = distance < self._reach_radius
        if reached:
            reward += self._reach_bonus
            self._targets_reached += 1
            self._sample_target(self._torso_xy, heading)
            # Re-anchor so the jump to a new goal is not scored as lost progress.
            self._prev_distance = float(np.linalg.norm(self._target_xy - self._torso_xy))

        self._last_action = action.copy()
        terminated = not self._is_healthy
        if terminated:
            # Falling must cost more than the burst of speed a dive buys,
            # or lunging at the goal beats walking to it.
            reward -= self._fall_penalty
        obs = self._get_obs()

        info = {
            "distance_to_target": distance,
            "targets_reached": self._targets_reached,
            "closing_speed": progress,
            "facing": facing,
            "reward_healthy": healthy,
            "reward_ctrl": -ctrl_cost,
            "reached_target": float(reached),
            "torso_height": float(self.data.qpos[2]),
        }

        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info

    def reset_model(self):
        noise = self._reset_noise_scale
        qpos = self.init_qpos + self.np_random.uniform(-noise, noise, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(-noise, noise, size=self.model.nv)
        # Start facing a random world direction so heading cannot be memorised.
        yaw = float(self.np_random.uniform(-np.pi, np.pi))
        qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.set_state(qpos, qvel)

        self._last_action = np.zeros(self.model.nu)
        self._targets_reached = 0
        self._sample_target(self._torso_xy, yaw)
        self._prev_distance = float(np.linalg.norm(self._target_xy - self._torso_xy))
        return self._get_obs()
