"""Configuration for the two-legged "walk to the target" task."""

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

BIPED_USD = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "biped_usd", "biped", "biped.usda")

# nominal standing crouch; keep in sync with biped/make_urdf.py
Q_HIP_PITCH, Q_KNEE, Q_ANKLE = -0.25, 0.50, -0.25
STANDING_HIP_HEIGHT = 0.66

BIPED_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=BIPED_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_linear_velocity=100.0,
            max_angular_velocity=100.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, STANDING_HIP_HEIGHT + 0.02),
        joint_pos={
            ".*_hip_yaw": 0.0,
            ".*_hip_roll": 0.0,
            ".*_hip_pitch": Q_HIP_PITCH,
            ".*_knee": Q_KNEE,
            ".*_ankle": Q_ANKLE,
        },
    ),
    actuators={
        "hips": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll"],
            effort_limit_sim=100.0,
            velocity_limit_sim=12.0,
            stiffness=100.0,
            damping=4.0,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_pitch", ".*_knee"],
            effort_limit_sim=150.0,
            velocity_limit_sim=15.0,
            stiffness=150.0,
            damping=5.0,
        ),
        "ankles": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle"],
            effort_limit_sim=80.0,
            velocity_limit_sim=15.0,
            stiffness=80.0,
            damping=3.0,
        ),
    },
)

TARGET_MARKER_CFG = VisualizationMarkersCfg(
    prim_path="/Visuals/Targets",
    markers={
        "target": sim_utils.SphereCfg(
            radius=0.15,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.9, 0.2)),
        )
    },
)


@configclass
class BipedEnvCfg(DirectRLEnvCfg):
    # timing: physics 200 Hz, policy 50 Hz
    decimation = 4
    episode_length_s = 20.0

    # spaces
    action_space = 10  # 2 legs x (hip yaw, hip roll, hip pitch, knee, ankle)
    observation_space = 42  # 10 q + 10 qd + 3 v + 3 w + 3 g + 3 target + 10 last action
    state_space = 0

    sim: SimulationCfg = SimulationCfg(dt=1 / 200, render_interval=decimation)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=6.0, replicate_physics=True)
    robot_cfg: ArticulationCfg = BIPED_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    target_marker_cfg: VisualizationMarkersCfg = TARGET_MARKER_CFG

    # task
    action_scale = 0.5  # rad, added on top of the nominal standing pose
    target_radius_range = (2.0, 4.5)  # target sampled on this ring around the env origin [m]
    reach_threshold = 0.4  # [m]
    nominal_height = STANDING_HIP_HEIGHT
    min_height = 0.40  # [m] torso lower than this counts as a fall
    max_tilt = -0.4  # fall once projected gravity z rises above this (~66 deg of lean)
    air_time_target = 0.35  # [s] rewarded swing duration per step
    foot_contact_height = 0.05  # [m] foot centre below this counts as planted (sole sits at 0.02)

    # reward scales
    rew_progress = 15.0  # per metre of distance closed, per step
    rew_heading = 0.5  # cos(angle to target)
    rew_reach = 20.0  # bonus when a target is reached
    rew_alive = 1.0  # per step spent upright -- the key term for a biped
    rew_upright = 0.5  # -projected_gravity_z (1 when the torso is vertical)
    rew_height = -10.0  # squared shortfall below the nominal hip height
    rew_air_time = 0.6  # rewards real steps instead of shuffling
    rew_ang_vel_xy = -0.05  # damp roll/pitch rates
    rew_action_rate = -0.005
    rew_joint_vel = -0.0001
    rew_torque = -0.00002
    rew_fall = -20.0  # once, on termination by falling
