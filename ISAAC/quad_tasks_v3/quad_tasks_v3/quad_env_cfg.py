"""Isaac Lab 3 manager-based env for the quad on Newton + MuJoCo-Warp, built to training/quad/QUAD_SPEC.md.

Observation (36 floats, + 2 from round 8, this order - it is the ONNX input layout):
    gravity_b(3) lin_vel_b(3)*0.5 ang_vel_b(3)*0.25 joint_pos(8)/0.785398 joint_vel(8)*0.1
    last_action(8) goal_dir_b(2) target_speed/2(1) [sin phi, cos phi of the reference-gait clock (2)]
    (all in the torso frame; goal = world +x rotated into the torso frame, (x, y) renormalised)
Action: 8 joint targets in quad_rig.json actuator order, target = rest(0) + a * 0.785398 rad, a clipped to
[-1, 1] by the RSL-RL wrapper (agent cfg clip_actions=1.0). Physics 0.005 s, decimation 10 -> 20 Hz (QUAD_SPEC
round 3; reward dt = step_dt = 0.05 s). Episode 20 s (400 policy steps). Stage 1: a fall (torso up.z < 0.5 or torso z < 0.45 m) is TERMINAL; the
time-out bootstraps; the simulation health guard is terminal with zero reward.
Reset: rest pose at 0.92 m, yaw U(+/-45 deg), joint noise U(+/-0.05 rad), zero velocity. Per reset:
friction x U(0.85, 1.15) (one per env), body masses x U(0.9, 1.1) (each body, inertia scaled),
kp x U(0.8, 1.2) (each joint). A 0.5 m/s push in a random horizontal direction every U(10, 15) s.
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
from .quad_cfg import QUAD_CFG

TORSO = SceneEntityCfg("robot", body_names=[spec.TORSO])
JOINTS = SceneEntityCfg("robot", joint_names=spec.ACTION_ORDER, preserve_order=True)
BODIES = SceneEntityCfg("robot", body_names=spec.BODY_NAMES, preserve_order=True)
LOWER_LEGS = SceneEntityCfg("robot", body_names=spec.LOWER_LEGS, preserve_order=True)


@configclass
class QuadShapeCfg(NewtonShapeCfg):
    """Newton per-shape defaults for shapes WITHOUT authored values (the ground plane): MuJoCo geom solref
    (0.01, 1), or the round-9 floor-pair solref (0.03, 1), as ke/kd; margin 0. See spec.py "Ground contact"."""

    ke: float = spec.CONTACT_KE
    kd: float = spec.CONTACT_KD
    margin: float = 0.0


QUAD_NEWTON_CFG = NewtonCfg(
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
    default_shape_cfg=QuadShapeCfg(),
)


@configclass
class QuadSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        # spec.GROUND_FRICTION: 0.5 (below every body, max() -> body 0.9*s) or the floor-pair friction (priority 1)
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=spec.GROUND_FRICTION, dynamic_friction=spec.GROUND_FRICTION, restitution=0.0
        ),
        debug_vis=False,
    )
    robot = QUAD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=spec.ACTION_ORDER,
        preserve_order=True,
        scale=spec.ACTION_SCALE,
        use_default_offset=True,  # default joint pos = the rest pose (0)
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ORDER MATTERS: QUAD_SPEC.md "Observation", index for index.
        gravity_b = ObsTerm(func=mdp.ref_gravity_b, params={"asset_cfg": TORSO})
        lin_vel_b = ObsTerm(func=mdp.ref_lin_vel_b, params={"asset_cfg": TORSO}, scale=spec.OBS_LIN_VEL_SCALE)
        ang_vel_b = ObsTerm(func=mdp.ref_ang_vel_b, params={"asset_cfg": TORSO}, scale=spec.OBS_ANG_VEL_SCALE)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel_ordered, params={"asset_cfg": JOINTS}, scale=spec.OBS_JOINT_POS_SCALE)
        joint_vel = ObsTerm(func=mdp.joint_vel_ordered, params={"asset_cfg": JOINTS}, scale=spec.OBS_JOINT_VEL_SCALE)
        last_action = ObsTerm(func=mdp.last_action)
        goal_dir_b = ObsTerm(func=mdp.goal_dir_b, params={"asset_cfg": TORSO})
        target_speed = ObsTerm(func=mdp.target_speed_obs, params={"value": spec.OBS_TARGET_SPEED})
        gait_clock = ObsTerm(func=mdp.gait_clock_obs) if spec.GAIT else None  # rounds 8-9: obs[36:38] = sin, cos phi

        def __post_init__(self):
            self.enable_corruption = False  # no observation noise
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventsCfg:
    # -- reset: rest pose, torso at env origin + 0.92 m, yaw U(+/-45 deg) about the torso (the floating base,
    #    like the MJCF free joint), joint noise U(+/-0.05 rad), zero velocities
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
    # -- per-episode randomisation (every reset): lean per-world writers (see worm_tasks_v3/mdp/events.py)
    # floor-priority regime (round 9 final): the FLOOR's friction is randomised per world, the body stays 0.9;
    # otherwise the body shapes carry 0.9 * s and the ground sits below them (see spec.py "Ground contact")
    friction = (
        EventTerm(func=mdp.randomize_floor_friction, mode="reset",
                  params={"base": spec.GROUND_FRICTION, "scale_range": spec.FRICTION_SCALE})
        if spec.FLOOR_PRIORITY else
        EventTerm(func=mdp.randomize_friction_scale, mode="reset",
                  params={"asset_cfg": SceneEntityCfg("robot"), "base": spec.FRICTION, "scale_range": spec.FRICTION_SCALE})
    )
    body_mass = EventTerm(
        func=mdp.randomize_body_mass_lean,
        mode="reset",
        params={"asset_cfg": BODIES, "scale_range": spec.MASS_SCALE},
    )
    kp = EventTerm(
        func=mdp.randomize_kp_lean,
        mode="reset",
        params={"asset_cfg": JOINTS, "scale_range": spec.KP_SCALE},
    )
    # -- rounds 8-9: the reference-gait clock, phi ~ U(0, 2 pi) at every reset (training and evaluation)
    gait_clock = EventTerm(func=mdp.reset_gait_clock, mode="reset") if spec.GAIT else None
    # -- a 0.5 m/s horizontal push, per-env timer U(10, 15) s redrawn at every reset and after each push
    push = EventTerm(
        func=mdp.push_horizontal,
        mode="interval",
        interval_range_s=spec.PUSH_INTERVAL_S,
        is_global_time=False,
        params={"speed": spec.PUSH_SPEED, "asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsCfg:
    # per-second weights; the manager multiplies by step_dt = 0.05 s
    alive = RewTerm(func=mdp.alive, weight=spec.W_ALIVE,
                    params={"asset_cfg": TORSO, "up_min": spec.FALL_UP_DOT, "z_min": spec.FALL_HEIGHT})
    speed = RewTerm(func=mdp.speed_tracking, weight=spec.W_SPEED,
                    params={"asset_cfg": TORSO, "target": spec.TARGET_SPEED, "sigma": spec.SPEED_SIGMA})
    progress = RewTerm(func=mdp.progress, weight=spec.W_PROGRESS, params={"asset_cfg": TORSO, "target": spec.TARGET_SPEED})
    heading = RewTerm(func=mdp.heading, weight=spec.W_HEADING, params={"asset_cfg": TORSO})
    upright = RewTerm(func=mdp.upright, weight=spec.W_UPRIGHT, params={"asset_cfg": TORSO})
    lateral_drift = RewTerm(func=mdp.lateral_drift, weight=spec.W_LATERAL, params={"asset_cfg": TORSO})
    effort = RewTerm(func=mdp.effort_mean, weight=spec.W_EFFORT,
                     params={"asset_cfg": JOINTS, "force_limit": spec.FORCE_LIMIT})
    action_rate = RewTerm(func=mdp.action_rate_mean, weight=spec.W_ACTION_RATE)
    feet_air_time = RewTerm(func=mdp.feet_air_time, weight=spec.W_AIR_TIME,
                            params={"asset_cfg": SceneEntityCfg("robot"), "torso_cfg": TORSO, "vx_min": spec.AIR_VX_MIN})
    flight = RewTerm(func=mdp.flight, weight=spec.W_FLIGHT, params={"decimation": spec.DECIMATION})
    gait_ref = (RewTerm(func=mdp.gait_ref, weight=spec.W_GAIT_REF, params={"asset_cfg": JOINTS, "sigma": spec.GAIT_REF_SIGMA})
                if spec.GAIT else None)
    contact_phase = RewTerm(func=mdp.contact_phase, weight=spec.W_CONTACT_PHASE) if spec.GAIT else None
    vertical_bounce = RewTerm(func=mdp.vertical_bounce, weight=spec.W_BOUNCE, params={"asset_cfg": TORSO})
    foot_slip = RewTerm(func=mdp.foot_slip, weight=spec.W_FOOT_SLIP,
                        params={"asset_cfg": LOWER_LEGS, "foot_points": spec.FOOT_POINTS})


@configclass
class TerminationsCfg:
    # health guard (WORM_SPEC item 12): terminal, no bootstrap, zero reward that step
    diverged = DoneTerm(func=mdp.sim_diverged, time_out=False,
                        params={"asset_cfg": SceneEntityCfg("robot"), "max_speed": spec.HEALTH_MAX_SPEED})
    # stage 1: a fall ends the episode as terminal (no bootstrap)
    fall = DoneTerm(func=mdp.fallen, time_out=False,
                    params={"asset_cfg": TORSO, "up_min": spec.FALL_UP_DOT, "z_min": spec.FALL_HEIGHT})
    # 20 s: time_out=True -> extras["time_outs"] -> RSL-RL bootstraps the value
    time_out = DoneTerm(func=mdp.time_out_healthy_standing, time_out=True,
                        params={"asset_cfg": TORSO, "up_min": spec.FALL_UP_DOT, "z_min": spec.FALL_HEIGHT,
                                "max_speed": spec.HEALTH_MAX_SPEED})


@configclass
class QuadFlatNewtonEnvCfg(ManagerBasedRLEnvCfg):
    scene: QuadSceneCfg = QuadSceneCfg(num_envs=4096, env_spacing=3.0)
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
        self.sim.physics = QUAD_NEWTON_CFG
        # Newton's graphed decimation (4 substeps + torque telemetry as one CUDA graph); identical physics
        # with implicit actuators (checked on the worm). check_quad_physics_v3.py turns it off.
        self.sim.use_newton_actuators = True


@configclass
class QuadFlatNewtonEnvCfg_PLAY(QuadFlatNewtonEnvCfg):
    """Evaluation/export: the spec reset, NO randomisation (nominal friction, masses, kp), no pushes."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 100
        self.events.friction = None
        self.events.body_mass = None
        self.events.kp = None
        self.events.push = None
