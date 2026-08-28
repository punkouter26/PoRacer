"""Configuration for the 8-legged spider "walk to the target" task."""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

SPIDER_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "spider_usd", "spider", "spider.usda")

SPIDER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SPIDER_USD,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=10.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    # feet reach ~0.13 m below the body centre -> spawn slightly above standing height
    init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, 0.18), joint_pos={".*": 0.0}),
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip", ".*_knee"],
            effort_limit_sim=15.0,
            velocity_limit_sim=12.0,
            stiffness=25.0,
            damping=1.0,
        ),
    },
)

TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Targets",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.12,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
        )
    },
)


@configclass
class SpiderEnvCfg(DirectRLEnvCfg):
    # timing: physics 120 Hz, policy 30 Hz
    decimation = 4
    episode_length_s = 15.0

    # spaces
    action_space = 16  # 8 legs x (hip swing, knee)
    observation_space = 59  # 16 q + 16 qd + 3 v + 3 w + 3 g + 2 target + 16 last action
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=6.0, replicate_physics=True)
    robot_cfg: ArticulationCfg = SPIDER_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    target_marker_cfg: VisualizationMarkersCfg = TARGET_MARKER_CFG

    # task
    action_scale = 0.8  # rad: actions in [-1, 1] map to joint targets in [-0.8, 0.8]
    target_radius_range = (1.5, 3.5)  # target sampled on this ring around the env origin [m]
    reach_threshold = 0.3  # [m]
    min_body_height = 0.06  # [m] terminate if the body drops below this (collapsed)

    # reward scales
    rew_progress = 20.0  # per metre of distance closed, per step
    rew_heading = 0.1  # cos(angle to target) shaping
    rew_reach = 10.0  # bonus when target reached
    rew_upright = 0.05  # -projected_gravity_z (1 when flat)
    rew_action_rate = -0.002
    rew_joint_vel = -0.00002
    rew_fall = -5.0  # once, on termination by falling
