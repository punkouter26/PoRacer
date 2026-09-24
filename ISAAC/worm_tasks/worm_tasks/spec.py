"""Every number from training/worm/WORM_SPEC.md that the Isaac side uses, in one place.

WORM_SPEC.md is a contract shared with the MuJoCo trainer (method A). Change it there first,
then here, then on the MuJoCo side. Body numbers are read from worm_rig.json (written by
build_worm.py alongside worm.xml) so the Isaac asset and the MJCF cannot drift apart.
"""

from __future__ import annotations

import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
WORM_DIR = os.path.join(REPO, "training", "worm")
WORM_USD = os.environ.get("WORM_USD", os.path.normpath(os.path.join(_HERE, "..", "assets", "worm.usda")))

with open(os.path.join(WORM_DIR, "worm_rig.json"), "r", encoding="utf-8") as _f:
    RIG = json.load(_f)

# ------------------------------------------------------------------------------ body --
KP = float(RIG["kp"])                      # 30 N*m/rad
FORCE_LIMIT = float(RIG["forceLimit"])     # 12 N*m
JOINT_DAMPING = float(RIG["jointDamping"])  # 1.0 N*m*s/rad (MuJoCo: joint damping; PhysX: drive damping)
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
# Weights are PER SECOND. Isaac Lab's RewardManager multiplies every term by weight * step_dt
# (0.02 s), which is exactly the spec's "reward per policy step, multiplied by dt = 0.02 s".
W_PROGRESS = 1.0
W_HEADING = 0.1
W_ACTION_RATE = -0.02
W_EFFORT = -0.01
# Was -0.1: the Isaac smoke run learned a diagonal sidewinder that drifted 6 m sideways in
# 20 s for 5 m forward, and the race lanes are 2 m apart. WORM_SPEC.md changed on both sides.
W_LATERAL = -0.5
PROGRESS_CLIP = (-1.0, 2.0)  # m/s

# ---------------------------------------------------------------------- episode/reset --
EPISODE_LENGTH_S = 20.0
RESET_YAW = math.radians(45.0)
RESET_JOINT_NOISE = 0.05

# ------------------------------------------------------------------ randomisation --
FRICTION_SCALE = (0.85, 1.15)
MASS_SCALE = (0.9, 1.1)
KP_SCALE = (0.8, 1.2)
# PhysX caps the number of unique materials (64k), and the tensor API creates one per distinct
# value, so the per-episode friction scale is drawn from U(0.85, 1.15) and snapped to this grid
# (61 levels). The distribution is otherwise uniform.
FRICTION_SCALE_STEP = 0.005
