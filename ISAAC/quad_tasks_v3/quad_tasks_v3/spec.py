"""Every number from training/quad/QUAD_SPEC.md that the Isaac Lab 3 / Newton side uses, in one place.

QUAD_SPEC.md is the contract (where it is silent, training/worm/WORM_SPEC.md "Details resolved" 1-16
apply, "segment 2" read as "torso"). Body numbers come from training/quad/quad_rig.json (written by
build_quad.py next to quad.xml) so the Newton asset and the MJCF cannot drift apart; USD prim names come
from the converter's sidecar (quad_v3.usda.names.json: MJCF "Upper_-1_1" -> USD "Upper_m1_1").
"""

from __future__ import annotations

import json
import math
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(_HERE, "..", "..", ".."))
QUAD_DIR = os.path.join(REPO, "training", "quad")
QUAD_XML = os.path.join(QUAD_DIR, "quad.xml")
QUAD_USD = os.environ.get("QUAD_V3_USD", os.path.normpath(os.path.join(_HERE, "..", "assets", "quad_v3.usda")))

with open(os.path.join(QUAD_DIR, "quad_rig.json"), "r", encoding="utf-8") as _f:
    RIG = json.load(_f)
if os.path.isfile(QUAD_USD + ".names.json"):
    with open(QUAD_USD + ".names.json", "r", encoding="utf-8") as _f:
        NAMES = json.load(_f)
else:  # before the first conversion (the cfg module raises a clear error in that case)
    NAMES = {"bodies": {}, "joints": {}, "geoms": {}}


def usd_body(mjcf_name: str) -> str:
    return NAMES["bodies"].get(mjcf_name, mjcf_name.replace("-", "m"))


def usd_joint(mjcf_name: str) -> str:
    return NAMES["joints"].get(mjcf_name, mjcf_name.replace("-", "m"))


# ------------------------------------------------------------------------------ body --
_ACT = {a["name"]: a for a in RIG["actuators"]}
_JNT = {j["name"]: j for j in RIG["joints"]}
ACTION_ORDER_MJCF = list(RIG["actionOrder"])                      # quad_rig.json actuator order
ACTION_ORDER = [usd_joint(_ACT[n]["joint"]) for n in ACTION_ORDER_MJCF]  # Isaac Lab / USD joint names
NUM_ACTIONS = len(ACTION_ORDER)                                   # 8
KP = float(_ACT[ACTION_ORDER_MJCF[0]]["kp"])                      # 1500 N*m/rad (was 4500)
FORCE_LIMIT = float(RIG["forceLimit"])                            # 300 N*m
JOINT_DAMPING = float(_JNT[ACTION_ORDER_MJCF[0]]["damping"])      # 100 N*m*s/rad (was 300), PASSIVE (mjc:damping)
ARMATURE = float(_JNT[ACTION_ORDER_MJCF[0]]["armature"])          # 0
assert all(float(_ACT[n]["kp"]) == KP for n in ACTION_ORDER_MJCF), "per-joint kp not supported here"
assert all(float(_JNT[n]["damping"]) == JOINT_DAMPING and float(_JNT[n]["armature"]) == ARMATURE for n in ACTION_ORDER_MJCF)
REST_POSE = [float(RIG["restPose"][_ACT[n]["joint"]]) for n in ACTION_ORDER_MJCF]  # all 0
FRICTION = 0.9
TOTAL_MASS = float(RIG["totalMass"])                              # 90 kg
REST_Z = float(RIG["restRootHeight"])                             # 0.90 m
SPAWN_Z = float(RIG["spawnRootHeight"])                           # 0.92 m (rest + 2 cm)
PHYSICS_DT = float(RIG["physics"]["timestep"])                    # 0.005 s
DECIMATION = int(RIG["physics"]["decimation"])                    # 10 -> 20 Hz (QUAD_SPEC round 3)
assert DECIMATION == 10 and abs(float(RIG["physics"]["policyDt"]) - 0.05) < 1e-12, "quad_rig.json != 20 Hz"
STEP_DT = PHYSICS_DT * DECIMATION                                 # 0.05 s
EPISODE_STEPS = 400                                               # 20 s at 20 Hz

TORSO_MJCF = RIG["torso"]                                         # "Quad_v01"
TORSO = usd_body(TORSO_MJCF)
BODY_NAMES = [usd_body(b["name"]) for b in RIG["bodies"]]         # every body (mass randomisation)
LOWER_LEGS = [usd_body(n) for n in RIG["lowerLegs"]]              # foot-slip bodies
UPPER_LEGS = [usd_body(b["name"]) for b in RIG["bodies"] if b["name"].startswith("Upper_")]
FALL_TOUCH_BODIES = [TORSO] + UPPER_LEGS                          # round 5: floor contact here = a fall
FOOT_POINTS = [tuple(float(v) for v in RIG["footPointsLocal"][n]) for n in RIG["lowerLegs"]]  # lower-leg frame

# ------------------------------------------------------------------- action / obs --
ACTION_SCALE = float(RIG["actionScaleRad"])  # 0.785398: target = rest + a * 0.785398
OBS_LIN_VEL_SCALE = 0.5
OBS_ANG_VEL_SCALE = 0.25
OBS_JOINT_POS_SCALE = 1.0 / ACTION_SCALE
OBS_JOINT_VEL_SCALE = 0.1
# Contract round. 7 = the 36-float task of rounds 5-7; 8 / 9 add the reference trot (clock obs -> 38 floats,
# gait_ref, contact_phase; see mdp/gait.py). Default 9 (prepared, not yet smoked); override with QUAD_V3_ROUND.
ROUND = int(os.environ.get("QUAD_V3_ROUND", "9"))
assert ROUND in (7, 8, 9), ROUND
GAIT = ROUND >= 8
NUM_OBS = 38 if GAIT else 36
GOAL_DIR_W = (1.0, 0.0)  # world +x
TARGET_SPEED = float(RIG["task"]["targetSpeed"])  # 1.49 m/s (Froude 0.25), observed as / 2
OBS_TARGET_SPEED = TARGET_SPEED / 2.0

# ----------------------------------------------------------------------------- reward --
# per-second weights; Isaac Lab's RewardManager multiplies every term by weight * step_dt (0.05 s at 20 Hz).
# Effort / action rate / foot slip were raised 10x / 25x / 10x in QUAD_SPEC.md after the first MuJoCo smoke
# run learned a 50 Hz buzz-hop (actions flipping +-1 every step, airborne 57 % of the time).
W_ALIVE = 1.0           # every policy step that did not end in a fall; round 6 (falls were free in round 5)
W_SPEED = 2.0
W_PROGRESS = 1.0        # clip(v_x, 0, 1.49) / 1.49; round 4 (0.5), round 6 (1.0)
SPEED_SIGMA = 1.0        # exp(-((v_x - 1.49) / 1.0)^2), two-sided; round 6 (was 0.5)
W_HEADING = 0.1         # (was 0.3, round 4)
W_UPRIGHT = 0.2         # (was 0.5, round 4)
W_LATERAL = -0.5
W_EFFORT = -0.2          # mean over joints of (actuator force / 300)^2   (was -0.02)
W_ACTION_RATE = -0.2     # per policy step (was -0.02, then -0.5; round 4)
W_FOOT_SLIP = -1.0       # sum over lower legs in floor contact of the foot point's horizontal speed (was -0.1)
W_BOUNCE = -2.0          # (torso world v_z)^2; round 3, stops the bounding gait
W_FLIGHT = -1.0          # per second x fraction of the step's substeps with all 4 debounced feet airborne; round 7
W_AIR_TIME = 0.5         # PER FOOTFALL (not per second): debounced touchdown, min(swing, 0.5) - 0.25, only if v_x > 0.3; round 5 (was 1.0)
AIR_TIME_OFFSET = 0.25   # s
# reference trot (rounds 8-9, training/quad/mujoco/quad_env.py Q16)
GAIT_FREQ = 1.5          # Hz
GAIT_AMP_HIP = 0.3       # action units (x 0.785398 rad)
GAIT_AMP_KNEE = 0.5
GAIT_KNEE_SIGN = 1.0     # knee = s * A_k * max(0, -cos psi), s = +1
W_GAIT_REF = 2.0         # per second x exp(-mean (q - q_ref)^2 / sigma^2)
GAIT_REF_SIGMA = 0.2 if ROUND >= 9 else 0.3   # rad (round 9: 0.2)
W_CONTACT_PHASE = 2.0 if ROUND >= 9 else 0.5  # per second (round 9: 2.0)
AIR_VX_MIN = 0.3         # m/s: a step's touchdown sum pays only if the post-step torso v_x > 0.3 (round 6)

# ------------------------------------------------------------------------ episodes --
EPISODE_LENGTH_S = 20.0
FALL_UP_DOT = 0.5        # torso up . world up < 0.5 -> fall (terminal)
FALL_HEIGHT = 0.45       # torso z < 0.45 m -> fall (terminal)
HEALTH_MAX_SPEED = 500.0  # WORM_SPEC item 12
RESET_YAW = math.radians(45.0)
RESET_JOINT_NOISE = 0.05
EVAL_SEED = 12345

# ------------------------------------------------------------------ randomisation --
FRICTION_SCALE = (0.85, 1.15)
MASS_SCALE = (0.9, 1.1)   # every body, independently, inertia scaled to match
KP_SCALE = (0.8, 1.2)     # every joint, independently; force limit and damping fixed
PUSH_SPEED = 0.5          # m/s, random horizontal direction, added to the root (whole body)
PUSH_INTERVAL_S = (10.0, 15.0)

# ------------------------------------------------ MuJoCo-Warp solver (quad.xml <option>) --
MJ_SOLVER = "newton"
MJ_ITERATIONS = int(RIG["physics"]["iterations"])       # 10
MJ_LS_ITERATIONS = int(RIG["physics"]["lsIterations"])  # 8
MJ_INTEGRATOR = "implicitfast"
MJ_CONE = "pyramidal"
MJ_IMPRATIO = 1.0
MJ_TOLERANCE = float(RIG["physics"]["tolerance"])       # 1e-8 (MuJoCo default; Isaac Lab's cfg default is 1e-6)
# Contact buffer per world: box-plane up to 4, capsule-plane 2 each (16), plus leg-leg / leg-torso contacts
# when fallen or tangled. 64 contacts x 4 pyramid rows + 8 limits fits in 320 constraint rows.
MJ_NCONMAX = 64
MJ_NJMAX = 320
# Ground contact. Two regimes, chosen by quad.xml:
#
# * no floor <pair>s (rounds 1-8): the ground has MuJoCo's default geom solref (0.01, 1) as Newton gains and
#   priority 0. It is one shape shared by all worlds, so it cannot be scaled per world; MuJoCo combines two
#   geoms' friction with max(), so keeping the ground below the smallest body value (0.9 * 0.85 = 0.765) makes
#   every body-floor contact use exactly the body's 0.9 * s (WORM_SPEC item 14).
# * floor <pair>s (round 9 soft feet: floor <-> each lower leg, solref 0.03 1, friction 0.9, and an <exclude> of
#   the automatic world/lower-leg contact). USD cannot carry <pair>, so the GROUND takes the pair parameters
#   (solref -> Newton ke/kd, friction, solimp default) and MuJoCo priority 1 (QuadArticulation writes it into the
#   Newton model before the solver is built). Then every floor contact uses the pair's parameters: the foot-floor
#   contacts exactly as quad.xml; the torso / upper-leg floor contacts too (in quad.xml those keep solref 0.01 and
#   friction 0.9 * s), but such a contact ends the episode as a fall on that step. Body-body contacts are unchanged.
#
# * floor priority (round 9 final): quad.xml's FLOOR geom itself is soft (solref 0.03, 1), priority 1, friction 0.9;
#   every body geom stays 0.01 / priority 0 / friction 0.9. The floor's parameters then win in every floor contact.
#   The ground takes exactly those parameters, and the per-episode friction randomisation moves to the floor:
#   per world, floor friction = 0.9 * U(0.85, 1.15), written straight into MuJoCo-Warp's per-world geom_friction
#   (randomize_floor_friction); body friction fixed 0.9. Newton has one ground shape for all worlds, so nothing
#   may notify SHAPE_PROPERTIES afterwards (it would rewrite every world's floor friction from that one shape).
FLOOR_CONTACT = NAMES.get("floorContact")  # from the converter sidecar; None without floor pairs
FLOOR_GEOM = NAMES.get("floorGeom")        # quad.xml's floor geom parameters
FLOOR_PRIORITY = bool(FLOOR_GEOM and FLOOR_GEOM["priority"] > 0 and not FLOOR_CONTACT)


def _gains(solref):
    tc, dr = solref
    return 1.0 / (tc * tc * dr * dr), 2.0 / tc   # Newton ke/kd that SolverMuJoCo.convert_solref maps back


if FLOOR_PRIORITY:
    CONTACT_KE, CONTACT_KD = _gains(FLOOR_GEOM["solref"])      # (0.03, 1) -> 1111.1, 66.67
    GROUND_FRICTION = float(FLOOR_GEOM["friction"][0])         # 0.9 nominal; x U(0.85, 1.15) per world per reset
    GROUND_PRIORITY = int(FLOOR_GEOM["priority"])
    assert FLOOR_GEOM["condim"] == 3 and FLOOR_GEOM["solimp"] == [0.9, 0.95, 0.001, 0.5, 2.0]
    assert FLOOR_GEOM["margin"] == 0.0 and FLOOR_GEOM["gap"] == 0.0 and FLOOR_GEOM["solmix"] == 1.0
    assert abs(FLOOR_GEOM["friction"][1] - 0.005) < 1e-9 and abs(FLOOR_GEOM["friction"][2] - 0.0001) < 1e-9
elif FLOOR_CONTACT:
    CONTACT_KE, CONTACT_KD = _gains(FLOOR_CONTACT["solref"])   # (0.03, 1) -> 1111.1, 66.67
    GROUND_FRICTION = float(FLOOR_CONTACT["friction"][0])        # 0.9, fixed (quad.xml's pair friction)
    GROUND_PRIORITY = 1
    assert FLOOR_CONTACT["condim"] == 3 and FLOOR_CONTACT["solimp"] == [0.9, 0.95, 0.001, 0.5, 2.0]
    assert abs(FLOOR_CONTACT["friction"][2] - 0.005) < 1e-9 and abs(FLOOR_CONTACT["friction"][3] - 0.0001) < 1e-9
else:
    CONTACT_KE, CONTACT_KD = 1.0e4, 200.0   # geom solref (0.01, 1)
    GROUND_FRICTION = 0.5
    GROUND_PRIORITY = 0
