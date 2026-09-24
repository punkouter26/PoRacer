"""Articulation configuration for Worm5 (training/worm/worm.xml converted to worm.usda).

Physics parameters are those of WORM_SPEC.md "Body":

All values are read from training/worm/worm_rig.json via :mod:`worm_tasks.spec`:

* position servo  kp 30 N*m/rad, force limit +/-6 N*m           -> PhysX implicit drive
                                                                     (stiffness 30, maxForce 6)
* joint damping   2.0 N*m*s/rad (MuJoCo: passive, on the joint)  -> see DAMPING_MODE below
* armature        0.01                                           -> PhysX joint armature
* joint limits    +/-45 deg (from the USD, i.e. from worm.xml)

DAMPING_MODE (env var ``WORM_DAMPING_MODE``):

``joint`` (default)
    PhysX 5 (Isaac Sim >= 5.0) articulation joints have a *viscous friction* coefficient:
    tau = -c_v * qdot, applied by the solver on the joint, independent of the drive and NOT
    limited by the drive's maxForce. That is exactly MuJoCo's ``<joint damping>``. The drive
    then has damping 0, so drive force = clip(kp * (target - q), +/-6), which is exactly
    MuJoCo's ``<position kp forcerange>`` actuator force. This is the spec-identical setting.

``drive``
    Drive damping (2.0) and no joint viscous friction: drive force =
    clip(kp * (target - q) - 2.0 * qdot, +/-6). Same damping coefficient, but the damping
    torque counts against the 6 N*m limit (MuJoCo's does not). Kept as a fallback.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from . import spec

DAMPING_MODE = os.environ.get("WORM_DAMPING_MODE", "joint").strip().lower()
if DAMPING_MODE not in ("joint", "drive"):
    raise ValueError(f"WORM_DAMPING_MODE must be 'joint' or 'drive', got {DAMPING_MODE!r}")

if not os.path.isfile(spec.WORM_USD):
    raise FileNotFoundError(
        f"{spec.WORM_USD} not found. Run ISAAC/scripts/convert_worm.py with the Isaac Lab venv python first."
    )

WORM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=spec.WORM_USD,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            # MuJoCo has no per-body velocity damping; PhysX articulation links default to 0.05
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            # the reset's +/-0.05 rad pitch noise can put the tail a few mm into the floor;
            # push it out gently instead of launching it (PhysX-only knob, no MuJoCo analogue)
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            # everything collides except the adjacent-segment pairs, which the USD filters
            # with UsdPhysics.FilteredPairsAPI (converted from worm.xml's <contact><exclude>)
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # the articulation root is seg0 (the head); worm.xml puts it at (0, 0, 0.05), the rest
        # of the body trailing along -x
        pos=(0.0, 0.0, spec.SPAWN_Z),
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    # no soft limit shrink: the reset noise clamp and the obs scale both use the full +/-45 deg
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=spec.ACTION_ORDER,
            stiffness=spec.KP,
            damping=spec.JOINT_DAMPING if DAMPING_MODE == "drive" else 0.0,
            effort_limit_sim=spec.FORCE_LIMIT,
            armature=spec.ARMATURE,
            friction=0.0,
            dynamic_friction=0.0,
            viscous_friction=spec.JOINT_DAMPING if DAMPING_MODE == "joint" else 0.0,
        )
    },
)
"""Worm5: 5 capsule segments + 4 pitch links, 8 revolute joints (+/-45 deg), 7.08 kg."""
