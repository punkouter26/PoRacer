"""Flat-ground target-chasing environment for the Boy.

Observation (75 floats, in this order - CONTRACT.md section 2 is generated from it):
    base_lin_vel(3) base_ang_vel(3) projected_gravity(3) target_pos_b(3)
    joint_pos_rel(21) joint_vel_rel(21) last_action(21)

Action: 21 joint position targets, target = default + 0.5 * action, implicit PD drives.
Physics 200 Hz, decimation 4 -> 50 Hz policy. Episode 20 s.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from . import mdp
from .boy_cfg import BOY_CFG, BOY_RIG

# ---------------------------------------------------------------- compat shims --
# Isaac Lab 2026 split the physics back-ends into isaaclab_physx / isaaclab_newton and
# selects them through PresetCfg; earlier 2.x versions took a single ContactSensorCfg
# and SimulationCfg(physx=...). Support both so the task runs on whichever checkout
# install.ps1 pulled.
try:
    from isaaclab_physx.sensors import ContactSensorCfg
except ImportError:  # pragma: no cover
    from isaaclab.sensors import ContactSensorCfg


def _make_sim_cfg() -> SimulationCfg:
    try:
        from isaaclab_physx.physics import PhysxCfg

        return SimulationCfg(physics=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))
    except (ImportError, TypeError):  # pragma: no cover
        from isaaclab.sim import PhysxCfg

        return SimulationCfg(physx=PhysxCfg(gpu_max_rigid_patch_count=10 * 2**15))


CHASE = BOY_RIG["chase"]
TIMING = BOY_RIG["timing"]


# ------------------------------------------------------------------------ scene --
@configclass
class BoySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=float(BOY_RIG["physics"]["groundStaticFriction"]),
            dynamic_friction=float(BOY_RIG["physics"]["groundDynamicFriction"]),
            restitution=float(BOY_RIG["physics"]["groundRestitution"]),
        ),
        debug_vis=False,
    )
    robot = BOY_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0),
    )


# -------------------------------------------------------------------------- mdp --
@configclass
class CommandsCfg:
    target = mdp.TargetPositionCommandCfg(
        asset_name="robot",
        resampling_time_range=tuple(CHASE["resampleRangeS"]),
        radius_range=tuple(CHASE["targetRadiusRange"]),
        reach_radius=float(CHASE["reachRadius"]),
        max_obs_distance=float(CHASE["targetObsClip"]),
        debug_vis=False,
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=[".*"], scale=float(BOY_RIG["actionScale"]), use_default_offset=True
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ORDER MATTERS: this is the ONNX input layout Unity reproduces index by index.
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        target_pos_b = ObsTerm(func=mdp.target_pos_b, params={"command_name": "target"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # startup: the robot's own shapes get a random friction; the ground stays 1.0/1.0 and
    # PhysX multiplies, so the effective pair friction is whatever is drawn here.
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.6, 1.0),
            "dynamic_friction_range": (0.4, 0.8),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="spine"),
            "mass_distribution_params": (1 / 1.25, 1.25),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="spine"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (-0.3, 0.3), "y": (-0.3, 0.3), "z": (-0.2, 0.2),
                "roll": (-0.3, 0.3), "pitch": (-0.3, 0.3), "yaw": (-0.3, 0.3),
            },
        },
    )
    # the default pose is a precise standing pose: do not scale it randomly
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (1.0, 1.0), "velocity_range": (0.0, 0.0)},
    )
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    # -- task: close on the target at ~1 m/s, facing it
    target_speed_exp = RewTerm(
        func=mdp.target_speed_exp,
        weight=1.5,
        params={"command_name": "target", "target_speed": float(CHASE["targetSpeed"]), "std": 0.5},
    )
    target_progress = RewTerm(func=mdp.target_progress, weight=0.5, params={"command_name": "target", "max_speed": 1.5})
    heading_to_target = RewTerm(func=mdp.heading_to_target, weight=0.3, params={"command_name": "target"})
    targets_reached = RewTerm(func=mdp.targets_reached, weight=5.0, params={"command_name": "target"})
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-200.0)
    # -- gait shaping (H1 values)
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.0,
        params={
            "command_name": "target",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "threshold": 0.6,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="foot_.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names="foot_.*"),
        },
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["thigh_.*", "shin_.*", "upper_arm_.*", "forearm_.*"]),
            "threshold": 1.0,
        },
    )
    # -- regularisation
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0)
    dof_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    dof_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-7)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot", joint_names="ankle_.*")}
    )
    joint_deviation_hip = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["hip_yaw_.*", "hip_roll_.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["shoulder_.*", "elbow_.*"])},
    )
    joint_deviation_torso = RewTerm(
        func=mdp.joint_deviation_l1, weight=-0.1, params={"asset_cfg": SceneEntityCfg("robot", joint_names="spine_pitch")}
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["hips", "spine"]), "threshold": 1.0},
    )


# --------------------------------------------------------------------------- env --
@configclass
class BoyChaseFlatEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = _make_sim_cfg()
    scene: BoySceneCfg = BoySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = int(TIMING["isaacDecimation"])
        self.episode_length_s = float(TIMING["episodeLengthS"])
        self.sim.dt = float(TIMING["isaacPhysicsDt"])
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt


@configclass
class BoyChaseFlatEnvCfg_PLAY(BoyChaseFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 64
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # no noise, no pushes: the reference recording must be deterministic
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.physics_material.params["static_friction_range"] = (0.8, 0.8)
        self.events.physics_material.params["dynamic_friction_range"] = (0.6, 0.6)
        self.events.add_base_mass = None
