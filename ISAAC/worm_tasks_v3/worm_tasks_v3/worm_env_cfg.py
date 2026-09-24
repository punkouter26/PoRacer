"""Isaac Lab 3 manager-based env for Worm5 on Newton + MuJoCo-Warp, built to training/worm/WORM_SPEC.md.

Observation (35 floats, this order - it is the ONNX input layout):
    gravity_b(3) lin_vel_b(3)*0.5 ang_vel_b(3)*0.25 joint_pos(8)/0.785398 joint_vel(8)*0.1
    last_action(8) goal_dir_b(2)
Action: 8 joint targets in ACTION_ORDER, target = a * 0.785398 rad, a clipped to [-1, 1] by the
RSL-RL wrapper (agent cfg clip_actions=1.0). Physics 0.005 s, decimation 4 -> 50 Hz.
Episode 20 s (1000 policy steps), time-out (bootstrapped) + simulation health guard (terminal),
no fall termination.

Differences from the 2.3 / PhysX task are physics-backend plumbing only (see worm_cfg.py and
spec.py): the simulator is Newton with the MuJoCo-Warp solver configured like worm.xml's <option>,
joint damping is MuJoCo's own passive dof_damping, contacts use worm.xml's solref (0.01, 1), and the
friction randomisation is continuous (no PhysX material cap).
"""

from __future__ import annotations

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonShapeCfg

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass

from . import mdp, spec
from .worm_cfg import WORM_CFG

REF = SceneEntityCfg("robot", body_names=[spec.REF_BODY])
JOINTS = SceneEntityCfg("robot", joint_names=spec.ACTION_ORDER, preserve_order=True)
SEGMENTS = SceneEntityCfg("robot", body_names=spec.SEGMENT_NAMES, preserve_order=True)


@configclass
class WormShapeCfg(NewtonShapeCfg):
    """Newton per-shape defaults for every shape WITHOUT authored values (i.e. the ground plane).

    ke/kd give MuJoCo geom solref (0.01, 1) - worm.xml's default, floor included (the worm's capsules
    carry the same values authored in the USD). Margin 0 like worm.xml; gap is unused by MuJoCo-Warp's
    own collision detection.
    """

    ke: float = spec.CONTACT_KE
    kd: float = spec.CONTACT_KD
    margin: float = 0.0


WORM_NEWTON_CFG = NewtonCfg(
    solver_cfg=MJWarpSolverCfg(
        solver=spec.MJ_SOLVER,
        iterations=spec.MJ_ITERATIONS,
        ls_iterations=spec.MJ_LS_ITERATIONS,
        integrator=spec.MJ_INTEGRATOR,
        cone=spec.MJ_CONE,
        impratio=spec.MJ_IMPRATIO,
        tolerance=spec.MJ_TOLERANCE,
        njmax=spec.MJ_NJMAX,
        nconmax=spec.MJ_NCONMAX,
        use_mujoco_contacts=True,  # MuJoCo's own collision detection, as the MuJoCo trainer
    ),
    num_substeps=1,  # sim.dt IS the 0.005 s physics step; decimation 4 happens in the env
    debug_mode=False,
    default_shape_cfg=WormShapeCfg(),
)


@configclass
class WormSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        # below the smallest worm friction (0.765): MuJoCo's max() then picks the worm's 0.9 * s
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=spec.GROUND_FRICTION, dynamic_friction=spec.GROUND_FRICTION, restitution=0.0
        ),
        debug_vis=False,
    )
    robot = WORM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


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
    #    (the floating base, like the MJCF free joint), zero velocities
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
        params={"asset_cfg": SceneEntityCfg("robot"), "base": spec.FRICTION, "scale_range": spec.FRICTION_SCALE},
    )
    # segment masses (links not randomised) with inertia scaled to match, and kp per joint. These are the
    # lean per-world writers (mdp/events.py), not Isaac Lab's randomize_rigid_body_mass /
    # randomize_actuator_gains: same distributions, but no MuJoCo-Warp set_const over every world on
    # every reset (which halved throughput) and the MuJoCo trainer's semantics (invweight0 at nominal).
    segment_mass = EventTerm(
        func=mdp.randomize_segment_mass_lean,
        mode="reset",
        params={"asset_cfg": SEGMENTS, "scale_range": spec.MASS_SCALE},
    )
    kp = EventTerm(
        func=mdp.randomize_kp_lean,
        mode="reset",
        params={"asset_cfg": JOINTS, "scale_range": spec.KP_SCALE},
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
    # time_out=True -> extras["time_outs"] -> RSL-RL bootstraps the value. No fall termination.
    # Health guard (item 12): terminal, no bootstrap, zero reward that step.
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
class WormFlatNewtonEnvCfg(ManagerBasedRLEnvCfg):
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
        self.sim.physics = WORM_NEWTON_CFG
        # Newton's actuator fast path. With only implicit actuators this changes no physics (the solver's
        # joint drive does the PD either way; targets are held for the 4 substeps either way); it lets
        # NewtonManager run the WHOLE decimation loop (4 substeps + torque telemetry) as one CUDA graph
        # instead of 4 Python round-trips per policy step. Checked against the per-substep loop: identical
        # trajectories to float rounding (then contact chaos); applied_torque identical.
        # check_worm_physics_v3.py turns it off because it steps the simulation substep by substep.
        self.sim.use_newton_actuators = True


@configclass
class WormFlatNewtonEnvCfg_PLAY(WormFlatNewtonEnvCfg):
    """Evaluation/export: the spec reset, NO randomisation (nominal friction, masses, kp)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.events.friction = None
        self.events.segment_mass = None
        self.events.kp = None
