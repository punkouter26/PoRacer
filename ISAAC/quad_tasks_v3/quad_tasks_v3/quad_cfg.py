"""Articulation configuration for the quad on Isaac Lab 3 / Newton (MuJoCo-Warp solver).

The asset is ISAAC/quad_tasks_v3/assets/quad_v3.usda, written kit-less from training/quad/quad.xml by
ISAAC/scripts/convert_mjcf_v3.py (the shared MJCF converter; masses/inertias taken from the compiled MJCF).

How QUAD_SPEC.md "Body" maps onto Newton -> MuJoCo-Warp (Newton's SolverMuJoCo writes an mjModel):

* position servo  kp 1500, force limit +/-300 -> ImplicitActuatorCfg: stiffness 1500, damping 0,
                                               effort_limit_sim 300. Newton turns joint_target_ke/kd into a
                                               MuJoCo position actuator (gain kp, bias -kp*q) and the effort
                                               limit into jnt_actfrcrange +/-300: actuator force =
                                               clip(kp*(target - q), +/-300), exactly quad.xml's <position>.
* joint damping   100 (passive)              -> USD ``mjc:damping`` -> MuJoCo ``dof_damping`` 100 (not capped by
                                               the force limit, not part of actuator_force), as quad.xml.
* armature        0                          -> ImplicitActuatorCfg armature (-> dof_armature)
* joint limits    hip +/-45, knee +/-60 deg  -> from the USD; limit solref (0.01, 1) pinned by the converter
                                               (Newton limit gains 1e4 / 200 -> MuJoCo solref (-1e4, -200),
                                               the same K and B as quad.xml's solreflimit).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from . import spec
from .quad_articulation import QuadArticulation

if not os.path.isfile(spec.QUAD_USD):
    raise FileNotFoundError(
        f"{spec.QUAD_USD} not found. Run ISAAC/scripts/convert_mjcf_v3.py --mjcf training/quad/quad.xml "
        "--out ISAAC/quad_tasks_v3/assets/quad_v3.usda --root quad with the Isaac Lab 3 venv python first."
    )

QUAD_CFG = ArticulationCfg(
    class_type=QuadArticulation,  # stock Newton articulation + substep foot-contact tracker
    spawn=sim_utils.UsdFileCfg(usd_path=spec.QUAD_USD),
    init_state=ArticulationCfg.InitialStateCfg(
        # the floating base is the torso; spec reset = rest pose at 0.90 m + 2 cm. Quaternions are XYZW.
        pos=(0.0, 0.0, spec.SPAWN_Z),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={n: r for n, r in zip(spec.ACTION_ORDER, spec.REST_POSE)},
        joint_vel={".*": 0.0},
    ),
    # no soft limit shrink: the reset noise clamp and the obs scale both use the full ranges
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=spec.ACTION_ORDER,
            stiffness=spec.KP,
            damping=0.0,  # the 100 is PASSIVE joint damping (mjc:damping in the USD), not drive damping
            effort_limit_sim=spec.FORCE_LIMIT,
            armature=spec.ARMATURE,
            friction=0.0,
        )
    },
)
"""Quad: 60 kg box torso + 4 x (4 kg upper, 3.5 kg lower) capsule legs, 8 pitch hinges, 90 kg."""
