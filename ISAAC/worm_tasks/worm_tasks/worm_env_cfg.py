"""Isaac Lab manager-based env for Worm5, built to training/worm/WORM_SPEC.md.

Observation (35 floats, this order - it is the ONNX input layout):
    gravity_b(3) lin_vel_b(3)*0.5 ang_vel_b(3)*0.25 joint_pos(8)/0.785398 joint_vel(8)*0.1
    last_action(8) goal_dir_b(2)
Action: 8 joint targets in ACTION_ORDER, target = a * 0.785398 rad, a clipped to [-1, 1] by the
RSL-RL wrapper (agent cfg clip_actions=1.0). Physics 0.005 s, decimation 4 -> 50 Hz.
Episode 20 s (1000 policy steps), time-out (bootstrapped) + simulation health guard (terminal),
no fall termination.
"""

from __future__ import annotations

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
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from . import mdp, spec
from .worm_cfg import WORM_CFG

REF = SceneEntityCfg("robot", body_names=[spec.REF_BODY])
JOINTS = SceneEntityCfg("robot", joint_names=spec.ACTION_ORDER, preserve_order=True)
SEGMENTS = SceneEntityCfg("robot", body_names=spec.SEGMENT_NAMES, preserve_order=True)

# The worm's own shapes use the scene's default material (0.9 / 0.9 / bounce 0). The ground is
# a neutral 1.0 with "multiply", which outranks the worm's "average" in PhysX, so every
# worm-ground contact has friction 0.9 * s exactly, and worm-worm contacts 0.9 * s as well
# (equal values on both sides). MuJoCo's condim=3 contact is sliding friction only, so the
# torsional/rolling entries of worm.xml's friction="0.9 0.005 0.0001" are inactive there too.
WORM_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=spec.FRICTION,
    dynamic_friction=spec.FRICTION,
    restitution=0.0,
    friction_combine_mode="average",
    restitution_combine_mode="average",
)
GROUND_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=1.0,
    dynamic_friction=1.0,
    restitution=0.0,
    friction_combine_mode="multiply",
    restitution_combine_mode="multiply",
)


@configclass
class WormSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=GROUND_MATERIAL,
        debug_vis=False,
    )
    robot = WORM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    sky_light = AssetBaseCfg(prim_path="/World/skyLight", spawn=sim_utils.DomeLightCfg(intensity=750.0))


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=spec.ACTION_ORDER,
        preserve_order=True,
        scale=spec.ACTION_SCALE,
        use_default_offset=True,  # default joint pos = 0 (straight worm)
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ORDER MATTERS: this is WORM_SPEC.md's observation table, index for index.
        gravity_b = ObsTerm(func=mdp.ref_gravity_b, params={"asset_cfg": REF})
        lin_vel_b = ObsTerm(func=mdp.ref_lin_vel_b, params={"asset_cfg": REF}, scale=spec.OBS_LIN_VEL_SCALE)
        ang_vel_b = ObsTerm(func=mdp.ref_ang_vel_b, params={"asset_cfg": REF}, scale=spec.OBS_ANG_VEL_SCALE)
        joint_pos = ObsTerm(func=mdp.joint_pos_ordered, params={"asset_cfg": JOINTS}, scale=spec.OBS_JOINT_POS_SCALE)
        joint_vel = ObsTerm(func=mdp.joint_vel_ordered, params={"asset_cfg": JOINTS}, scale=spec.OBS_JOINT_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        goal_dir_b = ObsTerm(func=mdp.goal_dir_b, params={"asset_cfg": REF})

        def __post_init__(self):
            self.enable_corruption = False  # no observation noise
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # -- reset: straight worm, head at the env origin, z 0.05, yaw U(+/-45 deg) about the head
    #    (the articulation root, like the MJCF free joint), zero velocities
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"yaw": (-spec.RESET_YAW, spec.RESET_YAW)}, "velocity_range": {}},
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": JOINTS,
            "position_range": (-spec.RESET_JOINT_NOISE, spec.RESET_JOINT_NOISE),
            "velocity_range": (0.0, 0.0),
        },
    )
    # -- per-episode randomisation (applied at every reset, i.e. once per episode per env)
    friction = EventTerm(
        func=mdp.randomize_friction_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "base": spec.FRICTION,
            "scale_range": spec.FRICTION_SCALE,
            "step": spec.FRICTION_SCALE_STEP,
        },
    )
    segment_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SEGMENTS,
            "mass_distribution_params": spec.MASS_SCALE,
            "operation": "scale",
            "distribution": "uniform",
            "recompute_inertia": True,
        },
    )
    kp = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": spec.KP_SCALE,
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class RewardsCfg:
    # weights are the spec's per-second weights; the manager multiplies by step_dt = 0.02 s
    progress = RewTerm(func=mdp.progress, weight=spec.W_PROGRESS, params={"asset_cfg": REF, "clip": spec.PROGRESS_CLIP})
    heading = RewTerm(func=mdp.heading, weight=spec.W_HEADING, params={"asset_cfg": REF})
    action_rate = RewTerm(func=mdp.action_rate_mean, weight=spec.W_ACTION_RATE)
    effort = RewTerm(
        func=mdp.effort_mean, weight=spec.W_EFFORT, params={"asset_cfg": JOINTS, "force_limit": spec.FORCE_LIMIT}
    )
    roll_rate = RewTerm(func=mdp.roll_rate, weight=spec.W_ROLL_RATE, params={"asset_cfg": REF})
    belly_down = RewTerm(func=mdp.belly_down, weight=spec.W_BELLY_DOWN, params={"asset_cfg": REF})
    lateral_drift = RewTerm(func=mdp.lateral_drift, weight=spec.W_LATERAL, params={"asset_cfg": REF})


@configclass
class TerminationsCfg:
    # a time-out is not a failure: time_out=True puts it in extras["time_outs"], which RSL-RL
    # PPO uses to bootstrap the value. The worm cannot fall: no fall termination.
    # Health guard (WORM_SPEC.md item 12): terminal, no bootstrap, zero reward that step (the
    # reward terms read this term). A world that diverges on its time-out step is terminal too.
    diverged = DoneTerm(
        func=mdp.sim_diverged,
        time_out=False,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_speed": spec.HEALTH_MAX_SPEED},
    )
    time_out = DoneTerm(
        func=mdp.time_out_healthy,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot"), "max_speed": spec.HEALTH_MAX_SPEED},
    )


@configclass
class WormFlatEnvCfg(ManagerBasedRLEnvCfg):
    sim: SimulationCfg = SimulationCfg(physx=PhysxCfg())
    scene: WormSceneCfg = WormSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = spec.DECIMATION
        self.episode_length_s = spec.EPISODE_LENGTH_S
        self.sim.dt = spec.PHYSICS_DT
        self.sim.gravity = (0.0, 0.0, -9.81)
        self.sim.render_interval = self.decimation
        self.sim.physics_material = WORM_MATERIAL  # default material = the worm's shapes


@configclass
class WormFlatEnvCfg_PLAY(WormFlatEnvCfg):
    """Evaluation/export: the spec reset, NO randomisation (nominal friction, masses, kp)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.events.friction = None
        self.events.segment_mass = None
        self.events.kp = None
