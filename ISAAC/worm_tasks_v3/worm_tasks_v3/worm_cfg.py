"""Articulation configuration for Worm5 on Isaac Lab 3 / Newton (MuJoCo-Warp solver).

The asset is ISAAC/worm_tasks_v3/assets/worm_v3.usda, written kit-less from training/worm/worm.xml by
ISAAC/scripts/convert_worm_v3.py (masses/inertias taken from the compiled MJCF).

How WORM_SPEC.md "Body" maps onto Newton -> MuJoCo-Warp (Newton's SolverMuJoCo writes an mjModel):

* position servo  kp 30, force limit +/-6  -> ImplicitActuatorCfg: stiffness 30, damping 0,
                                              effort_limit_sim 6. Newton turns joint_target_ke/kd into a
                                              MuJoCo position actuator (gain kp, bias -kp*q) and the effort
                                              limit into jnt_actfrcrange +/-6: actuator force =
                                              clip(kp*(target - q), +/-6), exactly worm.xml's <position>.
* joint damping   2.0 (passive)            -> USD ``mjc:damping`` -> MuJoCo ``dof_damping`` 2.0. This is
                                              the SAME mechanism as worm.xml's <joint damping> (not capped
                                              by the force limit, not part of actuator_force). No PhysX
                                              "viscous friction" stand-in is needed on this backend.
* armature        0.01                     -> ImplicitActuatorCfg armature (-> dof_armature)
* joint limits    +/-45 deg                 -> from the USD; limit solref left at MuJoCo's default (0.02, 1)
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from . import spec

if not os.path.isfile(spec.WORM_USD):
    raise FileNotFoundError(
        f"{spec.WORM_USD} not found. Run ISAAC/scripts/convert_worm_v3.py with the Isaac Lab 3 venv python first."
    )

WORM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(usd_path=spec.WORM_USD),
    init_state=ArticulationCfg.InitialStateCfg(
        # the floating base is seg0 (the head); worm.xml puts it at (0, 0, 0.05), the rest of the body
        # trailing along -x. Quaternions are XYZW in Isaac Lab 3.
        pos=(0.0, 0.0, spec.SPAWN_Z),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    # no soft limit shrink: the reset noise clamp and the obs scale both use the full +/-45 deg
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=spec.ACTION_ORDER,
            stiffness=spec.KP,
            damping=0.0,  # the 2.0 is PASSIVE joint damping (mjc:damping in the USD), not drive damping
            effort_limit_sim=spec.FORCE_LIMIT,
            armature=spec.ARMATURE,
            friction=0.0,
        )
    },
)
"""Worm5: 5 capsule segments + 4 pitch links, 8 revolute joints (+/-45 deg), 7.08 kg."""
