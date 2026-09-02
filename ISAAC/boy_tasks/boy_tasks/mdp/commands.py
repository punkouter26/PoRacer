"""Target-position command: the point the Boy is chasing.

A goal is sampled on a ring around the robot's current position (radius in
``radius_range``, heading uniform on the full circle). It is resampled when the robot gets
within ``reach_radius`` of it, or after ``resampling_time_range`` seconds, whichever comes
first. The exposed command is the goal expressed in the robot's BASE frame and norm-clipped
to ``max_obs_distance`` - that is exactly the 3-float term the policy observes, so Unity only
has to reproduce this one function (see CONTRACT.md section 2).
"""

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _t(x):
    """Torch view of an Isaac Lab data buffer (Newton-backed buffers expose ``.torch``)."""
    return x.torch if hasattr(x, "torch") else x


def _quat_apply_inverse(q, v):
    fn = getattr(math_utils, "quat_apply_inverse", None) or getattr(math_utils, "quat_rotate_inverse")
    return fn(q, v)


class TargetPositionCommand(CommandTerm):
    """Chase-target command term. ``command`` is ``target_pos_b`` (3), clipped."""

    cfg: "TargetPositionCommandCfg"

    def __init__(self, cfg: "TargetPositionCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        n = self.num_envs
        self.target_pos_w = torch.zeros(n, 3, device=self.device)
        self.target_pos_b = torch.zeros(n, 3, device=self.device)
        self.distance = torch.zeros(n, device=self.device)
        self.reached = torch.zeros(n, dtype=torch.bool, device=self.device)
        self.reached_count = torch.zeros(n, device=self.device)
        self.metrics["distance_to_target"] = torch.zeros(n, device=self.device)
        self.metrics["targets_reached"] = torch.zeros(n, device=self.device)

    def __str__(self) -> str:
        return (
            f"TargetPositionCommand: radius {self.cfg.radius_range} m, reach {self.cfg.reach_radius} m, "
            f"resample {self.cfg.resampling_time_range} s, obs clip {self.cfg.max_obs_distance} m"
        )

    # ------------------------------------------------------------------ properties --
    @property
    def command(self) -> torch.Tensor:
        return self.target_pos_b

    # ---------------------------------------------------------------- implementation --
    def _update_metrics(self):
        self.metrics["distance_to_target"] = self.distance.clone()
        self.metrics["targets_reached"] = self.reached_count.clone()

    def _resample_command(self, env_ids):
        n = len(env_ids)
        root = _t(self.robot.data.root_pos_w)[env_ids]
        radius = torch.empty(n, device=self.device).uniform_(*self.cfg.radius_range)
        angle = torch.empty(n, device=self.device).uniform_(-torch.pi, torch.pi)
        goal = root.clone()
        goal[:, 0] += radius * torch.cos(angle)
        goal[:, 1] += radius * torch.sin(angle)
        goal[:, 2] = root[:, 2]  # flat ground: keep the target at hip height
        self.target_pos_w[env_ids] = goal
        self.reached[env_ids] = False

    def _update_command(self):
        root_pos = _t(self.robot.data.root_pos_w)
        root_quat = _t(self.robot.data.root_quat_w)
        delta = self.target_pos_w - root_pos
        self.distance = torch.norm(delta[:, :2], dim=-1)
        in_base = _quat_apply_inverse(root_quat, delta)
        norm = torch.norm(in_base, dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(self.cfg.max_obs_distance / norm, max=1.0)
        self.target_pos_b = in_base * scale
        # reached: count it once, then force a resample on the next compute()
        newly = (self.distance < self.cfg.reach_radius) & ~self.reached
        self.reached_count += newly.float()
        self.reached |= newly
        self.time_left[newly] = 0.0

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self.reached_count[env_ids] = 0.0
        return super().reset(env_ids)

    # ------------------------------------------------------------------ debug vis --
    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_visualizer"):
                from isaaclab.markers import VisualizationMarkers
                from isaaclab.markers.config import CUBOID_MARKER_CFG

                cfg = CUBOID_MARKER_CFG.copy()
                cfg.prim_path = "/Visuals/Command/target"
                cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                self.goal_visualizer = VisualizationMarkers(cfg)
            self.goal_visualizer.set_visibility(True)
        elif hasattr(self, "goal_visualizer"):
            self.goal_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if hasattr(self, "goal_visualizer"):
            self.goal_visualizer.visualize(self.target_pos_w)


@configclass
class TargetPositionCommandCfg(CommandTermCfg):
    class_type: type = TargetPositionCommand
    asset_name: str = MISSING
    radius_range: tuple[float, float] = (3.0, 10.0)
    reach_radius: float = 0.5
    max_obs_distance: float = 5.0
