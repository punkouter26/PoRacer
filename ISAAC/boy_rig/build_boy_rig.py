#!/usr/bin/env python3
"""
build_boy_rig.py - turn the authored Boy_Character rig into a physics articulation.

Reads the skeletal hierarchy out of Boy_Character.glb (the glTF JSON chunk, no FBX SDK
needed) and emits, in ONE frame (Isaac: right-handed, Z-up, X-forward, metres, radians):

  out/boy.usda                      the headless articulation Isaac Lab simulates:
                                    UsdPhysics rigid links, primitive collision shapes
                                    (capsules for limbs, boxes/spheres for feet, pelvis,
                                    torso, head), revolute joints, drive + armature.
                                    No visual meshes. Authored as text - usd-core is
                                    only used to VALIDATE it when available.
  out/boy_rig.json                  everything Unity needs: bodies, joints, shapes,
                                    masses, inertias, gains, limits, timing, spawn.
                                    Also copied to Assets/unity_export/Boy/.
  out/kinematics_reference.json     link positions in the hips frame for 3 poses, from
                                    an independent forward-kinematics pass. The Unity
                                    play-mode test compares its rig against this.

Frame map (glTF -> Isaac). glTF is right-handed Y-up and this character faces +Z with
its left side at +X. Isaac is right-handed Z-up X-forward with left at +Y, so

    isaac = (z_gltf, x_gltf, y_gltf)         (a cyclic permutation, det +1)

The Unity side then applies M: isaac (x,y,z) -> unity (-y, z, x) exactly as the IsaacH1
port does; see Assets/unity_export/Shared/IsaacFrameMap.cs.

Design decisions (all of them are also written into boy_rig.json so the Unity builder
and the Isaac config cannot drift apart):

  * The articulation ZERO pose is the authored T-pose. Every link frame is world-aligned
    at zero, and every joint axis is one of X, Y or Z of that frame. The DEFAULT pose
    (arms hanging, slight knee bend) is a joint-angle offset on top of it, exactly like
    Isaac Lab's init_state.joint_pos.
  * Multi-DoF joints are chains of single-axis revolute joints through small massive
    intermediate links, the way the H1 hips are built. PhysX 5 (Isaac) and PhysX 4.1
    (Unity) then see the SAME kinematic tree - a spherical ArticulationBody in Unity
    would use a twist/swing parameterisation that does not match Isaac's Euler chain.
  * Hip = yaw(Z) -> roll(X) -> pitch(Y).  Knee = pitch(Y).  Ankle = pitch(Y) -> roll(X).
    Spine = pitch(Y).  Shoulder = roll(X) -> pitch(Z) -> yaw(Y).  Elbow = Z.
    The shoulder/elbow axes look odd because they are written in the T-pose frame: after
    the roll joint hangs the arm down (-90 deg about X) the child frame's Z becomes the
    world lateral axis, so "pitch about Z" is a forward/back swing and "elbow about Z" is
    a forward bend. Section 5 of CONTRACT.md walks through it.
  * Mass is 45 kg total, split by Winter's anthropometric fractions. Intermediate links
    get 0.2 kg / 2e-4 kg.m2 so neither solver sees a near-massless link.

Usage:
    python build_boy_rig.py [--glb Boy_Character.glb] [--out out]
                            [--unity-dir ../../Assets/unity_export/Boy] [--total-mass 45]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_GLB = os.path.join(HERE, "Boy_Character.glb")
DEFAULT_OUT = os.path.join(HERE, "out")
DEFAULT_UNITY = os.path.normpath(os.path.join(HERE, "..", "..", "Assets", "unity_export", "Boy"))

CREATURE = "Boy"
TASK = "Isaac-Chase-Flat-Boy-v0"
PLAY_TASK = "Isaac-Chase-Flat-Boy-Play-v0"

# ---------------------------------------------------------------- timing / task --
PHYSICS_DT = 0.005      # 200 Hz - the PoRacer project's locked Time.fixedDeltaTime
DECIMATION = 4          # 50 Hz policy, identical to the IsaacH1 port
POLICY_DT = PHYSICS_DT * DECIMATION
EPISODE_LENGTH_S = 20.0
ACTION_SCALE = 0.5
TARGET_OBS_CLIP_M = 5.0          # target_pos_b is norm-clipped to this radius
TARGET_RADIUS_RANGE = (3.0, 10.0)
TARGET_REACH_RADIUS = 0.5
TARGET_RESAMPLE_RANGE_S = (8.0, 12.0)
TARGET_SPEED_M_S = 1.0           # the approach speed the reward tracks

# ------------------------------------------------------------------ small maths --
def v_add(a, b): return [a[i] + b[i] for i in range(3)]
def v_sub(a, b): return [a[i] - b[i] for i in range(3)]
def v_scale(a, s): return [a[i] * s for i in range(3)]
def v_norm(a): return math.sqrt(sum(x * x for x in a))


def quat_to_mat(q):
    x, y, z, w = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def axis_angle_mat(axis, angle):
    """Rotation matrix about a unit axis ('X'|'Y'|'Z' or a 3-vector) by angle [rad]."""
    if isinstance(axis, str):
        axis = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[axis]
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return [
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ]


def r(v, n=6):
    if isinstance(v, (list, tuple)):
        return [r(x, n) for x in v]
    return round(float(v), n)


# ------------------------------------------------------------------- GLB parse --
def load_gltf_json(path):
    with open(path, "rb") as f:
        magic, version, length = struct.unpack("<III", f.read(12))
        if magic != 0x46546C67:
            raise ValueError(f"{path} is not a binary glTF file")
        chunk_len, chunk_type = struct.unpack("<II", f.read(8))
        if chunk_type != 0x4E4F534A:
            raise ValueError("first GLB chunk is not JSON")
        return json.loads(f.read(chunk_len))


def node_local(n):
    if "matrix" in n:
        m = n["matrix"]  # column-major
        rot = [[m[0], m[4], m[8]], [m[1], m[5], m[9]], [m[2], m[6], m[10]]]
        return rot, [m[12], m[13], m[14]]
    rot = quat_to_mat(n.get("rotation", [0, 0, 0, 1]))
    s = n.get("scale", [1, 1, 1])
    rot = [[rot[i][j] * s[j] for j in range(3)] for i in range(3)]
    return rot, list(n.get("translation", [0, 0, 0]))


def bone_world_positions_gltf(g):
    """name -> world position (glTF Y-up) for every non-mesh node, from the scene root."""
    nodes = g["nodes"]
    out = OrderedDict()

    def walk(i, prot, ppos):
        n = nodes[i]
        lrot, lpos = node_local(n)
        wpos = v_add(ppos, mat_vec(prot, lpos))
        wrot = mat_mul(prot, lrot)
        if "mesh" not in n:
            out[n.get("name", f"node{i}")] = wpos
        for c in n.get("children", []):
            walk(c, wrot, wpos)

    ident = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    for root in g["scenes"][g.get("scene", 0)]["nodes"]:
        walk(root, ident, [0, 0, 0])
    return out


def gltf_to_isaac(p):
    """glTF (RH, Y-up, faces +Z, left = +X) -> Isaac (RH, Z-up, faces +X, left = +Y)."""
    return [p[2], p[0], p[1]]


# ------------------------------------------------------------- shape utilities --
def shape_volume(s):
    k = s["kind"]
    if k == "box":
        a, b, c = s["size"]
        return a * b * c
    if k == "sphere":
        return 4.0 / 3.0 * math.pi * s["radius"] ** 3
    if k == "capsule":
        rr, h = s["radius"], s["height"]
        return math.pi * rr * rr * h + 4.0 / 3.0 * math.pi * rr ** 3
    raise ValueError(k)


def shape_inertia_diag(s, mass):
    """Diagonal inertia about the shape's own centroid, in the link (world-aligned) frame."""
    k = s["kind"]
    if k == "box":
        a, b, c = s["size"]
        return [mass / 12 * (b * b + c * c), mass / 12 * (a * a + c * c), mass / 12 * (a * a + b * b)]
    if k == "sphere":
        i = 0.4 * mass * s["radius"] ** 2
        return [i, i, i]
    if k == "capsule":
        rr, h = s["radius"], s["height"]
        vc = math.pi * rr * rr * h
        vs = 4.0 / 3.0 * math.pi * rr ** 3
        mc = mass * vc / (vc + vs)
        mh = 0.5 * mass * vs / (vc + vs)  # each hemisphere
        i_axis = 0.5 * mc * rr * rr + 2 * (0.4 * mh * rr * rr)
        i_perp = mc * (h * h / 12 + rr * rr / 4) + 2 * mh * (0.4 * rr * rr + h * h / 4 + 3 * h * rr / 8)
        ax = s["axis"]
        return {"X": [i_axis, i_perp, i_perp], "Y": [i_perp, i_axis, i_perp], "Z": [i_perp, i_perp, i_axis]}[ax]
    raise ValueError(k)


def link_mass_properties(shapes, mass):
    """Split mass by volume over the link's shapes; return (com, diagonal inertia about com)."""
    vols = [shape_volume(s) for s in shapes]
    vtot = sum(vols)
    masses = [mass * v / vtot for v in vols]
    com = [0.0, 0.0, 0.0]
    for s, m in zip(shapes, masses):
        com = v_add(com, v_scale(s["center"], m))
    com = v_scale(com, 1.0 / mass)
    inertia = [0.0, 0.0, 0.0]
    for s, m in zip(shapes, masses):
        d = v_sub(s["center"], com)
        own = shape_inertia_diag(s, m)
        # parallel axis, diagonal terms only: all centroids sit on one link axis by
        # construction, so the products of inertia are zero (asserted below)
        inertia[0] += own[0] + m * (d[1] ** 2 + d[2] ** 2)
        inertia[1] += own[1] + m * (d[0] ** 2 + d[2] ** 2)
        inertia[2] += own[2] + m * (d[0] ** 2 + d[1] ** 2)
    # products of inertia must vanish for the diagonal to be the whole story
    for s, m in zip(shapes, masses):
        d = v_sub(s["center"], com)
        offaxis = sum(1 for x in d if abs(x) > 1e-9)
        if offaxis > 1:
            raise ValueError(f"shape {s} is off two link axes at once; tensor would not be diagonal")
    return com, inertia


# ------------------------------------------------------------------- the rig --
# Anthropometric mass fractions (Winter, Biomechanics and Motor Control of Human
# Movement). hand is merged into forearm, head+neck+clavicles into the spine link.
FRACTIONS = {
    "hips": 0.142,
    "spine": 0.355 + 0.081,          # trunk above the pelvis + head/neck
    "thigh": 0.100,
    "shin": 0.0465,
    "foot": 0.0145,
    "upper_arm": 0.028,
    "forearm": 0.016 + 0.006,        # forearm + hand
}
INTERMEDIATE_MASS = 0.2
INTERMEDIATE_INERTIA = 2.0e-4

# PD gains [N.m/rad], [N.m.s/rad], effort limit [N.m] per joint family.
GAINS = {
    "hip_yaw":        (80.0, 3.0, 150.0),
    "hip_roll":       (100.0, 4.0, 150.0),
    "hip_pitch":      (150.0, 5.0, 150.0),
    "knee":           (150.0, 5.0, 150.0),
    "ankle_pitch":    (30.0, 3.0, 50.0),
    "ankle_roll":     (20.0, 2.0, 50.0),
    "spine_pitch":    (400.0, 15.0, 200.0),   # 100 sagged 28 deg under the 19 kg torso (rung 1)
    "shoulder_roll":  (30.0, 3.0, 40.0),
    "shoulder_pitch": (30.0, 3.0, 40.0),
    "shoulder_yaw":   (30.0, 3.0, 40.0),
    "elbow":          (20.0, 2.0, 40.0),
}


def side_sign(side):
    return 1.0 if side == "L" else -1.0


def build_spec(bp):
    """bp: bone name -> world position in the ISAAC frame. Returns the ordered body list."""
    bodies = []

    def add(name, parent, pos, joint=None, shapes=None, frac=None, bone=None):
        bodies.append({
            "name": name, "parent": parent, "worldPos": pos, "joint": joint,
            "shapes": shapes or [], "frac": frac, "bone": bone,
        })

    def J(name, family, axis, lo, hi, default=0.0):
        kp, kd, eff = GAINS[family]
        return {"name": name, "family": family, "axis": axis, "lowerRad": lo, "upperRad": hi,
                "defaultPosRad": default, "stiffness": kp, "damping": kd, "effortLimit": eff,
                "armature": 0.0}

    hips = bp["hips"]
    spine = bp["spine"]
    neck = bp["neck"]
    head = bp["head"]

    # -- pelvis (root). Box centred on the hips origin.
    add("hips", None, hips, shapes=[{"kind": "box", "center": [0, 0, 0], "size": [0.18, 0.26, 0.16]}],
        frac="hips", bone="hips")

    # -- torso: spine link carries chest, neck, head and both clavicles.
    torso_top = neck[2] - spine[2]
    head_center_z = (head[2] + 0.14) - spine[2]
    add("spine", "hips", spine,
        joint=J("spine_pitch", "spine_pitch", "Y", -0.5, 0.5),
        shapes=[
            {"kind": "box", "center": [0, 0, torso_top * 0.5], "size": [0.18, 0.30, torso_top]},
            {"kind": "sphere", "center": [0, 0, head_center_z], "radius": 0.13},
        ],
        frac="spine", bone="spine")

    for side in ("L", "R"):
        sg = side_sign(side)
        thigh = bp[f"thigh.{side}"]
        shin = bp[f"shin.{side}"]
        foot = bp[f"foot.{side}"]
        thigh_len = v_norm(v_sub(shin, thigh))
        shin_len = v_norm(v_sub(foot, shin))

        add(f"hip_yaw_link_{side}", "hips", thigh, joint=J(f"hip_yaw_{side}", "hip_yaw", "Z", -0.6, 0.6))
        add(f"hip_roll_link_{side}", f"hip_yaw_link_{side}", thigh,
            joint=J(f"hip_roll_{side}", "hip_roll", "X", -0.6, 0.6))
        add(f"thigh_{side}", f"hip_roll_link_{side}", thigh,
            joint=J(f"hip_pitch_{side}", "hip_pitch", "Y", -2.0, 0.8, default=-0.2),
            shapes=[{"kind": "capsule", "axis": "Z", "radius": 0.075, "height": thigh_len - 0.15,
                     "center": [0, 0, -thigh_len * 0.5]}],
            frac="thigh", bone=f"thigh.{side}")
        add(f"shin_{side}", f"thigh_{side}", shin,
            joint=J(f"knee_{side}", "knee", "Y", 0.0, 2.4, default=0.4),
            shapes=[{"kind": "capsule", "axis": "Z", "radius": 0.055, "height": shin_len - 0.11,
                     "center": [0, 0, -shin_len * 0.5]}],
            frac="shin", bone=f"shin.{side}")
        add(f"ankle_pitch_link_{side}", f"shin_{side}", foot,
            joint=J(f"ankle_pitch_{side}", "ankle_pitch", "Y", -0.9, 0.9, default=-0.2))
        # foot box: heel 0.08 m behind the ankle, toes 0.18 m ahead, sole at z = 0
        foot_len, foot_w, foot_h = 0.26, 0.10, foot[2]
        add(f"foot_{side}", f"ankle_pitch_link_{side}", foot,
            joint=J(f"ankle_roll_{side}", "ankle_roll", "X", -0.5, 0.5),
            shapes=[{"kind": "box", "center": [0.06, 0, -foot_h * 0.5], "size": [foot_len, foot_w, foot_h]}],
            frac="foot", bone=f"foot.{side}")

        upper = bp[f"upper_arm.{side}"]
        fore = bp[f"forearm.{side}"]
        hand = bp[f"hand.{side}"]
        upper_len = v_norm(v_sub(fore, upper))
        fore_len = v_norm(v_sub(hand, fore)) + 0.16  # forearm + the hand it carries

        # shoulder: roll hangs the arm (left -90 deg, right +90 deg is the default pose)
        add(f"shoulder_roll_link_{side}", "spine", upper,
            joint=J(f"shoulder_roll_{side}", "shoulder_roll", "X",
                    -1.75 if side == "L" else -0.6, 0.6 if side == "L" else 1.75,
                    default=-1.35 * sg))
        add(f"shoulder_pitch_link_{side}", f"shoulder_roll_link_{side}", upper,
            joint=J(f"shoulder_pitch_{side}", "shoulder_pitch", "Z",
                    -2.5 if side == "L" else -1.0, 1.0 if side == "L" else 2.5))
        add(f"upper_arm_{side}", f"shoulder_pitch_link_{side}", upper,
            joint=J(f"shoulder_yaw_{side}", "shoulder_yaw", "Y", -1.0, 1.0),
            shapes=[{"kind": "capsule", "axis": "Y", "radius": 0.045, "height": upper_len - 0.09,
                     "center": [0, sg * upper_len * 0.5, 0]}],
            frac="upper_arm", bone=f"upper_arm.{side}")
        add(f"forearm_{side}", f"upper_arm_{side}", fore,
            joint=J(f"elbow_{side}", "elbow", "Z",
                    -2.3 if side == "L" else 0.0, 0.0 if side == "L" else 2.3,
                    default=-0.4 * sg),
            shapes=[{"kind": "capsule", "axis": "Y", "radius": 0.04, "height": fore_len - 0.08,
                     "center": [0, sg * fore_len * 0.5, 0]}],
            frac="forearm", bone=f"forearm.{side}")

    return bodies


def breadth_first(bodies):
    by_parent = {}
    for b in bodies:
        by_parent.setdefault(b["parent"], []).append(b)
    order, queue = [], list(by_parent[None])
    while queue:
        b = queue.pop(0)
        order.append(b)
        queue.extend(by_parent.get(b["name"], []))
    return order


def assign_masses(bodies, total_mass):
    real = [b for b in bodies if b["frac"] is not None]
    inter = [b for b in bodies if b["frac"] is None]
    budget = total_mass - INTERMEDIATE_MASS * len(inter)
    frac_sum = sum(FRACTIONS[b["frac"]] for b in real)
    for b in real:
        b["mass"] = budget * FRACTIONS[b["frac"]] / frac_sum
        b["com"], b["inertia"] = link_mass_properties(b["shapes"], b["mass"])
    for b in inter:
        b["mass"] = INTERMEDIATE_MASS
        b["com"] = [0.0, 0.0, 0.0]
        b["inertia"] = [INTERMEDIATE_INERTIA] * 3
    got = sum(b["mass"] for b in bodies)
    assert abs(got - total_mass) < 1e-6, got


# ---------------------------------------------------------- forward kinematics --
def forward_kinematics(bodies, q):
    """World pose of every link for joint angles q (dict joint name -> rad). Zero pose = T-pose."""
    by_name = {b["name"]: b for b in bodies}
    pose = {}
    for b in bodies:  # BFS order guarantees the parent is done
        if b["parent"] is None:
            pose[b["name"]] = ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], list(b["worldPos"]))
            continue
        prot, ppos = pose[b["parent"]]
        parent = by_name[b["parent"]]
        local = v_sub(b["worldPos"], parent["worldPos"])
        pos = v_add(ppos, mat_vec(prot, local))
        rot = mat_mul(prot, axis_angle_mat(b["joint"]["axis"], q.get(b["joint"]["name"], 0.0)))
        pose[b["name"]] = (rot, pos)
    return pose


def lowest_point(bodies, pose):
    """z of the lowest collision-shape extent, for the spawn height."""
    zmin = float("inf")
    for b in bodies:
        rot, pos = pose[b["name"]]
        for s in b["shapes"]:
            c = v_add(pos, mat_vec(rot, s["center"]))
            if s["kind"] == "box":
                hx, hy, hz = [x * 0.5 for x in s["size"]]
                for sx in (-1, 1):
                    for sy in (-1, 1):
                        for sz in (-1, 1):
                            corner = v_add(c, mat_vec(rot, [sx * hx, sy * hy, sz * hz]))
                            zmin = min(zmin, corner[2])
            elif s["kind"] == "sphere":
                zmin = min(zmin, c[2] - s["radius"])
            else:
                ax = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[s["axis"]]
                half = mat_vec(rot, v_scale(ax, s["height"] * 0.5))
                zmin = min(zmin, c[2] - abs(half[2]) - s["radius"])
    return zmin


# ---------------------------------------------------------------- USD authoring --
def usd_quat(wxyz):
    return "({}, {}, {}, {})".format(*[r(x) for x in wxyz])


def write_usda(bodies, joints, path):
    lines = [
        "#usda 1.0",
        "(",
        f'    doc = "Boy articulation for Isaac Lab, generated by build_boy_rig.py. No visuals."',
        f'    defaultPrim = "{CREATURE}"',
        "    metersPerUnit = 1",
        '    upAxis = "Z"',
        ")",
        "",
        f'def Xform "{CREATURE}" (',
        '    prepend apiSchemas = ["PhysicsArticulationRootAPI", "PhysxArticulationAPI"]',
        ")",
        "{",
        "    bool physxArticulation:enabledSelfCollisions = 0",
        "    int physxArticulation:solverPositionIterationCount = 4",
        "    int physxArticulation:solverVelocityIterationCount = 4",
        "",
    ]
    for b in bodies:
        lines += [
            f'    def Xform "{b["name"]}" (',
            '        prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysxRigidBodyAPI"]',
            "    )",
            "    {",
            f"        float physics:mass = {r(b['mass'])}",
            f"        point3f physics:centerOfMass = ({r(b['com'][0])}, {r(b['com'][1])}, {r(b['com'][2])})",
            f"        float3 physics:diagonalInertia = ({r(b['inertia'][0], 8)}, {r(b['inertia'][1], 8)}, {r(b['inertia'][2], 8)})",
            "        quatf physics:principalAxes = (1, 0, 0, 0)",
            "        float physxRigidBody:maxDepenetrationVelocity = 1",
            "        float physxRigidBody:linearDamping = 0",
            "        float physxRigidBody:angularDamping = 0",
            f"        double3 xformOp:translate = ({r(b['worldPos'][0])}, {r(b['worldPos'][1])}, {r(b['worldPos'][2])})",
            "        quatf xformOp:orient = (1, 0, 0, 0)",
            '        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:orient"]',
        ]
        for k, s in enumerate(b["shapes"]):
            c = s["center"]
            if s["kind"] == "box":
                lines += [
                    "",
                    f'        def Cube "collision_{k}" (',
                    '            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]',
                    "        )",
                    "        {",
                    "            double size = 1",
                    f"            double3 xformOp:translate = ({r(c[0])}, {r(c[1])}, {r(c[2])})",
                    f"            float3 xformOp:scale = ({r(s['size'][0])}, {r(s['size'][1])}, {r(s['size'][2])})",
                    '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]',
                    "        }",
                ]
            elif s["kind"] == "sphere":
                lines += [
                    "",
                    f'        def Sphere "collision_{k}" (',
                    '            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]',
                    "        )",
                    "        {",
                    f"            double radius = {r(s['radius'])}",
                    f"            double3 xformOp:translate = ({r(c[0])}, {r(c[1])}, {r(c[2])})",
                    '            uniform token[] xformOpOrder = ["xformOp:translate"]',
                    "        }",
                ]
            else:
                lines += [
                    "",
                    f'        def Capsule "collision_{k}" (',
                    '            prepend apiSchemas = ["PhysicsCollisionAPI", "PhysxCollisionAPI"]',
                    "        )",
                    "        {",
                    f'            uniform token axis = "{s["axis"]}"',
                    f"            double height = {r(s['height'])}",
                    f"            double radius = {r(s['radius'])}",
                    f"            double3 xformOp:translate = ({r(c[0])}, {r(c[1])}, {r(c[2])})",
                    '            uniform token[] xformOpOrder = ["xformOp:translate"]',
                    "        }",
                ]
        lines += ["    }", ""]

    lines += ['    def Scope "joints"', "    {"]
    for j in joints:
        lines += [
            f'        def PhysicsRevoluteJoint "{j["name"]}" (',
            '            prepend apiSchemas = ["PhysicsDriveAPI:angular", "PhysxJointAPI"]',
            "        )",
            "        {",
            f"            rel physics:body0 = </{CREATURE}/{j['parent']}>",
            f"            rel physics:body1 = </{CREATURE}/{j['child']}>",
            f"            point3f physics:localPos0 = ({r(j['localPos0'][0])}, {r(j['localPos0'][1])}, {r(j['localPos0'][2])})",
            "            point3f physics:localPos1 = (0, 0, 0)",
            "            quatf physics:localRot0 = (1, 0, 0, 0)",
            "            quatf physics:localRot1 = (1, 0, 0, 0)",
            f'            uniform token physics:axis = "{j["axis"]}"',
            f"            float physics:lowerLimit = {r(math.degrees(j['lowerRad']))}",
            f"            float physics:upperLimit = {r(math.degrees(j['upperRad']))}",
            '            uniform token drive:angular:physics:type = "force"',
            f"            float drive:angular:physics:stiffness = {r(j['stiffness'])}",
            f"            float drive:angular:physics:damping = {r(j['damping'])}",
            f"            float drive:angular:physics:maxForce = {r(j['effortLimit'])}",
            f"            float drive:angular:physics:targetPosition = 0",
            f"            float physxJoint:armature = {r(j['armature'])}",
            "            float physxJoint:jointFriction = 0",
            "        }",
        ]
    lines += ["    }", "}", ""]
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))


def validate_usda(path):
    try:
        from pxr import Usd, UsdPhysics
    except ImportError:
        print("[rig] usd-core not installed; skipping USD validation (pip install usd-core)")
        return None
    stage = Usd.Stage.Open(path)
    if stage is None:
        raise RuntimeError(f"pxr could not open {path}")
    links = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    shapes = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    for j in joints:
        rj = UsdPhysics.RevoluteJoint(j)
        for rel in (rj.GetBody0Rel(), rj.GetBody1Rel()):
            for t in rel.GetTargets():
                if not stage.GetPrimAtPath(t).IsValid():
                    raise RuntimeError(f"{j.GetPath()} targets missing prim {t}")
    return {"links": len(links), "joints": len(joints), "collisionShapes": len(shapes)}


# ------------------------------------------------------------------------ main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glb", default=DEFAULT_GLB)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--unity-dir", default=DEFAULT_UNITY, help="Assets/unity_export/Boy (rig json + FK reference)")
    ap.add_argument("--total-mass", type=float, default=45.0)
    args = ap.parse_args()

    g = load_gltf_json(args.glb)
    gltf_pos = bone_world_positions_gltf(g)
    skin = g["skins"][0]
    skin_bones = [g["nodes"][j].get("name") for j in skin["joints"]]
    bp = {k: gltf_to_isaac(v) for k, v in gltf_pos.items()}

    needed = ["hips", "spine", "chest", "neck", "head", "thigh.L", "shin.L", "foot.L", "thigh.R", "shin.R",
              "foot.R", "upper_arm.L", "forearm.L", "hand.L", "upper_arm.R", "forearm.R", "hand.R"]
    missing = [n for n in needed if n not in bp]
    if missing:
        sys.exit(f"[rig] bones missing from {args.glb}: {missing}")

    bodies = breadth_first(build_spec(bp))
    assign_masses(bodies, args.total_mass)
    by_name = {b["name"]: b for b in bodies}

    joints = []
    for b in bodies:
        if b["joint"] is None:
            continue
        j = dict(b["joint"])
        j["parent"] = b["parent"]
        j["child"] = b["name"]
        j["localPos0"] = v_sub(b["worldPos"], by_name[b["parent"]]["worldPos"])
        j["index"] = len(joints)
        joints.append(j)
        b["joint"] = j
    joint_order = [j["name"] for j in joints]
    default_q = {j["name"]: j["defaultPosRad"] for j in joints}

    # spawn height: hips origin such that the lowest shape sits 2 cm above the ground
    pose_default = forward_kinematics(bodies, default_q)
    z_low = lowest_point(bodies, pose_default)
    spawn_z = bp["hips"][2] - z_low + 0.02
    pose_zero = forward_kinematics(bodies, {})
    rest_hips_z_zero = bp["hips"][2] - lowest_point(bodies, pose_zero)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.unity_dir, exist_ok=True)

    # ---- USD
    usd_path = os.path.join(args.out, "boy.usda")
    write_usda(bodies, joints, usd_path)
    usd_stats = validate_usda(usd_path)

    # ---- rig JSON (Isaac frame throughout)
    axis_vec = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}
    rig = OrderedDict()
    rig["_comment"] = ("Generated by ISAAC/boy_rig/build_boy_rig.py. ISAAC FRAME (right-handed, Z-up, "
                       "X-forward, m, rad). Zero pose = the authored T-pose. Do not hand-edit; re-run "
                       "the script. export_bundle.py rewrites jointOrder/index from the live sim.")
    rig["creature"] = CREATURE
    rig["sourceTask"] = PLAY_TASK
    rig["trainTask"] = TASK
    rig["sourceModel"] = os.path.basename(args.glb)
    rig["jointOrder"] = joint_order
    rig["bodyOrder"] = [b["name"] for b in bodies]
    rig["skinBones"] = skin_bones
    rig["actionScale"] = ACTION_SCALE
    rig["useDefaultOffset"] = True
    rig["obsDim"] = 12 + 3 * len(joints)
    rig["actDim"] = len(joints)
    rig["obsLayout"] = [
        {"term": "base_lin_vel", "start": 0, "end": 3},
        {"term": "base_ang_vel", "start": 3, "end": 6},
        {"term": "projected_gravity", "start": 6, "end": 9},
        {"term": "target_pos_b", "start": 9, "end": 12},
        {"term": "joint_pos", "start": 12, "end": 12 + len(joints)},
        {"term": "joint_vel", "start": 12 + len(joints), "end": 12 + 2 * len(joints)},
        {"term": "actions", "start": 12 + 2 * len(joints), "end": 12 + 3 * len(joints)},
    ]
    rig["timing"] = {"policyDt": POLICY_DT, "isaacPhysicsDt": PHYSICS_DT, "isaacDecimation": DECIMATION,
                     "episodeLengthS": EPISODE_LENGTH_S}
    rig["spawn"] = {"posIsaac": [0.0, 0.0, r(spawn_z, 4)], "rotXyzw": [0.0, 0.0, 0.0, 1.0],
                    "hipsHeightAtZeroPoseRest": r(rest_hips_z_zero, 4),
                    "hipsHeightAtDefaultPoseRest": r(spawn_z - 0.02, 4)}
    rig["chase"] = {"targetObsClip": TARGET_OBS_CLIP_M, "targetRadiusRange": list(TARGET_RADIUS_RANGE),
                    "reachRadius": TARGET_REACH_RADIUS, "resampleRangeS": list(TARGET_RESAMPLE_RANGE_S),
                    "targetSpeed": TARGET_SPEED_M_S}
    rig["physics"] = {
        "gravity": [0.0, 0.0, -9.81],
        "groundStaticFriction": 1.0, "groundDynamicFriction": 1.0, "groundRestitution": 0.0,
        "robotStaticFriction": 0.8, "robotDynamicFriction": 0.6, "robotRestitution": 0.0,
        "frictionCombineMode": "multiply",
        "maxLinearVelocity": 1000.0, "maxAngularVelocity": 1000.0, "maxDepenetrationVelocity": 1.0,
        "linearDamping": 0.0, "angularDamping": 0.0, "jointFriction": 0.0,
        "solverPositionIterations": 4, "solverVelocityIterations": 4,
        "enabledSelfCollisions": False, "contactOffset": 0.02, "restOffset": 0.0,
        "isaacSolverType": "TGS",
    }
    rig["eval"] = {"_note": "filled in by export_bundle.py after training",
                   "meanSpeed": 0.0, "meanTargetSpeedError": 0.0, "fallsPerRobotPerMinute": 0.0,
                   "targetsReachedPerMinute": 0.0}
    rig["totalMass"] = args.total_mass
    out_bodies = []
    for b in bodies:
        d = OrderedDict()
        d["name"] = b["name"]
        d["parent"] = b["parent"]
        d["isRoot"] = b["parent"] is None
        d["boneName"] = b["bone"] or ""
        d["mass"] = r(b["mass"])
        d["com"] = r(b["com"])
        d["inertiaDiag"] = r(b["inertia"], 8)
        d["worldPos"] = r(b["worldPos"])
        d["localPos"] = r(v_sub(b["worldPos"], by_name[b["parent"]]["worldPos"])) if b["parent"] else [0.0, 0.0, 0.0]
        d["localRotWxyz"] = [1.0, 0.0, 0.0, 0.0]
        if b["joint"]:
            j = b["joint"]
            d["joint"] = OrderedDict([
                ("name", j["name"]), ("index", j["index"]), ("family", j["family"]),
                ("axisInChild", axis_vec[j["axis"]]), ("lowerRad", j["lowerRad"]), ("upperRad", j["upperRad"]),
                ("stiffness", j["stiffness"]), ("damping", j["damping"]), ("effortLimit", j["effortLimit"]),
                ("defaultPosRad", j["defaultPosRad"]), ("armature", j["armature"]),
            ])
        else:
            d["joint"] = None
        cols = []
        for s in b["shapes"]:
            c = OrderedDict([("kind", s["kind"]), ("center", r(s["center"]))])
            if s["kind"] == "box":
                c["size"] = r(s["size"])
            elif s["kind"] == "sphere":
                c["radius"] = r(s["radius"])
            else:
                c["radius"] = r(s["radius"])
                c["height"] = r(s["height"])
                c["axis"] = s["axis"]
            cols.append(c)
        d["colliders"] = cols
        out_bodies.append(d)
    rig["bodies"] = out_bodies

    rig_path = os.path.join(args.out, "boy_rig.json")
    with open(rig_path, "w", encoding="utf-8") as f:
        json.dump(rig, f, indent=1)
    with open(os.path.join(args.unity_dir, "boy_rig.json"), "w", encoding="utf-8") as f:
        json.dump(rig, f, indent=1)

    # ---- kinematics reference: link origins in the hips frame under 3 poses
    bent = {j["name"]: 0.5 * (j["lowerRad"] + j["upperRad"]) * 0.6 for j in joints}
    poses = []
    for name, q in (("zero", {}), ("default", default_q), ("bent", bent)):
        pose = forward_kinematics(bodies, q)
        hrot, hpos = pose["hips"]
        rel = []
        for b in bodies:
            rot, pos = pose[b["name"]]
            d = v_sub(pos, hpos)
            rel.append(r(mat_vec([[hrot[j][i] for j in range(3)] for i in range(3)], d)))
        poses.append({"name": name, "jointPosRad": [q.get(n, 0.0) for n in joint_order], "linkPosInHipsIsaac": rel})
    kin = {"_comment": "Independent Python forward kinematics of boy_rig.json. Isaac frame, hips-relative.",
           "bodyOrder": rig["bodyOrder"], "jointOrder": joint_order, "poses": poses}
    for path in (os.path.join(args.out, "kinematics_reference.json"),
                 os.path.join(args.unity_dir, "kinematics_reference.json")):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(kin, f, indent=1)

    # ---- summary
    print(f"[rig] {len(bodies)} links, {len(joints)} joints, "
          f"{sum(len(b['shapes']) for b in bodies)} collision shapes, {args.total_mass:.1f} kg")
    print(f"[rig] hips origin at rest: zero pose {rest_hips_z_zero:.4f} m, default pose {spawn_z - 0.02:.4f} m; "
          f"spawn z = {spawn_z:.4f} m")
    print("[rig] joint order:")
    for j in joints:
        print(f"   {j['index']:2d} {j['name']:18s} axis {j['axis']}  [{j['lowerRad']:+.2f}, {j['upperRad']:+.2f}]"
              f"  default {j['defaultPosRad']:+.2f}  kp {j['stiffness']:.0f} kd {j['damping']:.0f} "
              f"eff {j['effortLimit']:.0f}")
    print("[rig] masses:")
    for b in bodies:
        if b["frac"]:
            print(f"   {b['name']:16s} {b['mass']:6.3f} kg  I=({b['inertia'][0]:.5f}, {b['inertia'][1]:.5f}, "
                  f"{b['inertia'][2]:.5f})  com z {b['com'][2]:+.3f}")
    print(f"[rig] wrote {usd_path}" + (f"  (validated: {usd_stats})" if usd_stats else ""))
    print(f"[rig] wrote {rig_path} and {os.path.join(args.unity_dir, 'boy_rig.json')}")


if __name__ == "__main__":
    main()
