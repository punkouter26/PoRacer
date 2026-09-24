"""Every number from training/worm/WORM_SPEC.md that the Isaac Lab 3 / Newton side uses, in one place.

Same values as ISAAC/worm_tasks/worm_tasks/spec.py (the 2.3 / PhysX task). WORM_SPEC.md is the
contract: change it there first, then on every trainer. Body numbers are read from worm_rig.json
(written by build_worm.py alongside worm.xml) so the Newton asset and the MJCF cannot drift apart.
"""

from __future__ import annotations

import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
WORM_DIR = os.path.join(REPO, "training", "worm")
WORM_USD = os.environ.get("WORM_V3_USD", os.path.normpath(os.path.join(_HERE, "..", "assets", "worm_v3.usda")))

with open(os.path.join(WORM_DIR, "worm_rig.json"), "r", encoding="utf-8") as _f:
    RIG = json.load(_f)

# ------------------------------------------------------------------------------ body --
KP = float(RIG["kp"])                      # 30 N*m/rad
FORCE_LIMIT = float(RIG["forceLimit"])     # 6 N*m
JOINT_DAMPING = float(RIG["jointDamping"])  # 2.0 N*m*s/rad, PASSIVE (MuJoCo dof_damping via USD mjc:damping)
ARMATURE = float(RIG["armature"])          # 0.01 kg*m^2
FRICTION = float(RIG["friction"])          # 0.9
SPAWN_Z = float(RIG["spawnHeight"])        # 0.05 m
JOINT_RANGE = float(RIG["jointRangeRad"])  # 0.785398 rad = 45 deg
PHYSICS_DT = float(RIG["physicsDt"])       # 0.005 s
DECIMATION = int(RIG["decimation"])        # 4 -> 50 Hz policy
STEP_DT = PHYSICS_DT * DECIMATION          # 0.02 s

SEGMENT_NAMES = [f"seg{i}" for i in range(5)]
REF_BODY = "seg2"  # the observation / reward reference frame "B"

# ------------------------------------------------------------------- action / joints --
ACTION_ORDER = ["j0_pitch", "j0_yaw", "j1_pitch", "j1_yaw", "j2_pitch", "j2_yaw", "j3_pitch", "j3_yaw"]
ACTION_SCALE = 0.785398  # target = a * 0.785398 around a straight worm (spec literal, not math.pi/4)
NUM_ACTIONS = 8

# ------------------------------------------------------------------------ observation --
OBS_LIN_VEL_SCALE = 0.5
OBS_ANG_VEL_SCALE = 0.25
OBS_JOINT_POS_SCALE = 1.0 / 0.785398
OBS_JOINT_VEL_SCALE = 0.1
NUM_OBS = 35
GOAL_DIR_W = (1.0, 0.0)  # world +x, straight down the race lane

# ----------------------------------------------------------------------------- reward --
# Weights are PER SECOND; Isaac Lab's RewardManager multiplies every term by weight * step_dt (0.02 s).
W_PROGRESS = 1.0
W_HEADING = 0.1
W_ACTION_RATE = -0.02
W_EFFORT = -0.01  # (applied torque / FORCE_LIMIT)^2
W_ROLL_RATE = -0.1  # abs(seg2 angular velocity about its own long axis, B-frame x) [rad/s]
W_BELLY_DOWN = 0.2  # seg2 z axis . world z
W_LATERAL = -0.5  # abs(seg2 world velocity . world y)
PROGRESS_CLIP = (-1.0, 2.0)  # m/s

# ------------------------------------------------------------- simulation health guard --
HEALTH_MAX_SPEED = 500.0  # WORM_SPEC.md item 12

# ---------------------------------------------------------------------- episode/reset --
EPISODE_LENGTH_S = 20.0
RESET_YAW = math.radians(45.0)
RESET_JOINT_NOISE = 0.05
EVAL_SEED = 12345  # WORM_SPEC.md item 15

# ------------------------------------------------------------------ randomisation --
FRICTION_SCALE = (0.85, 1.15)
MASS_SCALE = (0.9, 1.1)
KP_SCALE = (0.8, 1.2)
# Newton has no PhysX-style unique-material cap, so (unlike the 2.3 task, which snapped s to a
# 0.005 grid) the friction scale is drawn continuously from U(0.85, 1.15), exactly as the spec says.

# ------------------------------------------------------- MuJoCo-Warp solver (worm.xml <option>) --
# Newton's SolverMuJoCo builds an mjModel from the Newton model; these mirror worm.xml's <option>
# and the MuJoCo trainer's put_data budget (training/worm/mujoco/worm_env.py NCONMAX / NJMAX).
MJ_SOLVER = "newton"
MJ_ITERATIONS = 10
MJ_LS_ITERATIONS = 8
MJ_INTEGRATOR = "implicitfast"
MJ_CONE = "pyramidal"
MJ_IMPRATIO = 1.0
MJ_TOLERANCE = 1e-8  # MuJoCo's default (worm.xml does not set it); Isaac Lab's cfg default is 1e-6
MJ_NCONMAX = 32
MJ_NJMAX = 128
# geom solref (0.01, 1) of worm.xml's <default><geom solref>, as Newton contact ke/kd
# (SolverMuJoCo.convert_solref: timeconst = 2/kd, dampratio = kd/2*sqrt(1/ke))
CONTACT_KE = 1.0e4
CONTACT_KD = 200.0
# Floor friction. MuJoCo-Warp combines two geoms' sliding friction with max(); the ground is ONE
# shape shared by all worlds, so it cannot be scaled per world like the MuJoCo trainer's floor.
# Keeping it below the smallest worm value (0.9 * 0.85 = 0.765) makes every worm-floor contact use
# exactly the worm's 0.9 * s - the same pair friction the spec asks for (item 14).
GROUND_FRICTION = 0.5
