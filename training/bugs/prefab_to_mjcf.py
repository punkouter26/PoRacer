"""Unity ArticulationBody prefab (YAML) -> MuJoCo MJCF + rig JSON.

Builds MuJoCo twins of the ML-Agents "bug" creatures (Quad_v01, Hexapod_v01,
Crab_v01) straight from their Unity prefab files, so that method C (MuJoCo /
mujoco_warp training) trains the same animal that races in Unity.

The prefab is only READ. Nothing under Assets/ is written.

Usage (from the repo root):
    .venv-mjwarp\\Scripts\\python.exe training\\bugs\\prefab_to_mjcf.py            # all three
    .venv-mjwarp\\Scripts\\python.exe training\\bugs\\prefab_to_mjcf.py Quad_v01   # one

Outputs, next to this file:
    <name>.xml        MJCF model
    <name>_rig.json   single source of truth for later twin checks

-------------------------------------------------------------------------------
FRAME CONVENTION (identical to Assets/unity_export/MujocoBiped/CONTRACT.md)
-------------------------------------------------------------------------------
Unity : left-handed,  Y-up, +X right, +Z forward.
MuJoCo: right-handed, Z-up, +X forward, +Y left.

The physical identity (not a mirror) between the two frames is the linear map

    P : (x, y, z)_unity -> (z, -x, y)_mujoco          det(P) = -1

  * true vectors (positions, offsets, velocities, forces)   transform by  P
  * pseudovectors (rotation axes, angular velocity, torque)  transform by -P
        axis (x, y, z)_unity -> (-z, x, -y)_mujoco
  * quaternions: a rotation by angle t about unit axis a in Unity is the same
    physical rotation as angle t about -P a in MuJoCo, so
        (x, y, z, w)_unity -> (w, -z, x, -y)_mujoco   (MuJoCo stores w first)
  * sizes / box half-extents only permute: (sx, sy, sz)_unity -> (sz, sx, sy)

Because hinge axes use -P, a positive Unity jointPosition / xDrive.target is a
positive MuJoCo qpos / ctrl for the same physical motion: no sign flips are
needed anywhere when replaying Unity actions in MuJoCo.

Side check (done at run time in main()): a link whose name ends in _-1_* sits
at Unity x < 0 (the creature's left, since Unity +X is right) and must land at
MuJoCo y > 0 (MuJoCo +Y is left).

-------------------------------------------------------------------------------
UNITS
-------------------------------------------------------------------------------
Unity ArticulationDrive (Scripting API, 6000.x):
    target      degrees (angular)                   -> converted to radians
    lower/upperLimit degrees                         -> radians
    stiffness   N*m/rad (angular)                    -> kp as-is
    damping     N*m*s/rad (angular)                  -> joint damping as-is (see below)
    forceLimit  N*m (angular)                        -> forcerange as-is
Drive law (Force drive type): tau = clamp(k*(target - q) + d*(targetVel - qd),
+-forceLimit), targetVel = 0 (Agent_Creature only ever writes xDrive.target).

DAMPING PLACEMENT (the one deliberate deviation, measured in README.md):
the literal twin is <position kp kv forcerange>, i.e. damping inside the clamp.
At the required 5 ms step that form is numerically unstable here (kp 3500-4500
N*m/rad on 2-4 kg links, kv 233-300): joint speeds chatter to 60-240 rad/s,
joints overshoot their limits by up to 50 deg, and it is still 5-7 deg RMS off
its converged (0.5 ms) result at 1 ms. PhysX solves articulation drives implicitly and does not have this
problem. The default (damping_mode="joint") therefore puts Unity's damping on
the joint (<joint damping>, integrated implicitly by MuJoCo) and keeps
kp + forcerange on the actuator: stable at 5 ms and within ~2 deg joint RMS of
the converged kv-inside-clamp reference over 1 s of the coded gait. The cost:
damping torque is not included in the forceLimit clamp, so a saturated joint
moves somewhat slower than in Unity. damping_mode="actuator" emits the literal
form for comparison.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
ASSETS = REPO / "Assets"
CATALOG = ASSETS / "Settings" / "CreatureCatalog.asset"

CREATURES = {
    "Quad_v01": ASSETS / "Prefabs" / "Quad_v01.prefab",
    "Hexapod_v01": ASSETS / "Prefabs" / "Hexapod_v01.prefab",
    "Crab_v01": ASSETS / "Prefabs" / "Crab_v01.prefab",
}

# Unity class IDs used below.
CID_GAMEOBJECT = 1
CID_TRANSFORM = 4
CID_BOXCOLLIDER = 65
CID_SPHERECOLLIDER = 135
CID_CAPSULECOLLIDER = 136
CID_MONOBEHAVIOUR = 114
CID_ARTICULATIONBODY = 171741748

JOINT_TYPES = {0: "fixed", 1: "prismatic", 2: "revolute", 3: "spherical"}
DOF_LOCK = {0: "locked", 1: "limited", 2: "free"}
DRIVE_TYPES = {0: "force", 1: "acceleration", 2: "target", 3: "velocity"}
COMBINE = {0: "average", 1: "minimum", 2: "multiply", 3: "maximum"}

# Unity's built-in default PhysicsMaterial (used when a collider's m_Material is
# {fileID: 0} and DynamicsManager.m_DefaultMaterial is also {fileID: 0}), and
# what `new PhysicsMaterial("TrainingGround")` gives the training floor
# (Systems_TrainingArea.cs line 61).
UNITY_DEFAULT_MATERIAL = {
    "name": "<Unity default>",
    "dynamicFriction": 0.6,
    "staticFriction": 0.6,
    "bounciness": 0.0,
    "frictionCombine": "average",
    "bounceCombine": "average",
    "source": "Unity built-in default (collider m_Material fileID 0, DynamicsManager m_DefaultMaterial fileID 0)",
}

SPAWN_OFFSET = 0.05  # Systems_Spawn adds +0.05 m to CreatureCatalog.spawnHeight


# ----------------------------------------------------------------------------
# Unity YAML
# ----------------------------------------------------------------------------
_DOC_HEADER = re.compile(r"^--- !u!(\d+) &(-?\d+)( stripped)?\s*$")


def parse_unity_yaml(path: Path) -> dict[int, dict]:
    """Returns {fileID: {"cid": classID, "type": TypeName, "data": {...}}}."""
    docs: dict[int, dict] = {}
    header = None
    lines: list[str] = []

    def flush():
        if header is None:
            return
        cid, fid, stripped = header
        body = yaml.safe_load("\n".join(lines)) or {}
        if len(body) == 1:
            type_name, data = next(iter(body.items()))
        else:
            type_name, data = "?", body
        docs[fid] = {"cid": cid, "type": type_name, "data": data or {}, "stripped": stripped}

    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("%"):
            continue
        m = _DOC_HEADER.match(raw)
        if m:
            flush()
            header = (int(m.group(1)), int(m.group(2)), bool(m.group(3)))
            lines = []
        else:
            lines.append(raw)
    flush()
    return docs


def find_asset_by_guid(guid: str) -> Path | None:
    for meta in ASSETS.rglob("*.meta"):
        try:
            with meta.open("r", encoding="utf-8", errors="ignore") as fh:
                for _ in range(4):
                    line = fh.readline()
                    if line.startswith("guid:"):
                        if line.split(":", 1)[1].strip() == guid:
                            return meta.with_suffix("")
                        break
        except OSError:
            continue
    return None


def read_physics_material(ref: dict) -> dict:
    if not ref or int(ref.get("fileID", 0)) == 0:
        return dict(UNITY_DEFAULT_MATERIAL)
    guid = ref.get("guid")
    if not guid:
        return {**UNITY_DEFAULT_MATERIAL, "source": f"UNRESOLVED local material ref {ref}; Unity default assumed"}
    asset = find_asset_by_guid(guid)
    if asset is None or not asset.exists():
        return {**UNITY_DEFAULT_MATERIAL, "source": f"NOT FOUND: material guid {guid}; Unity default assumed"}
    doc = next(iter(parse_unity_yaml(asset).values()))["data"]
    return {
        "name": doc.get("m_Name"),
        "dynamicFriction": float(doc.get("m_DynamicFriction", 0.6)),
        "staticFriction": float(doc.get("m_StaticFriction", 0.6)),
        "bounciness": float(doc.get("m_Bounciness", 0.0)),
        "frictionCombine": COMBINE.get(int(doc.get("m_FrictionCombine", 0)), "?"),
        "bounceCombine": COMBINE.get(int(doc.get("m_BounceCombine", 0)), "?"),
        "source": asset.relative_to(REPO).as_posix(),
    }


def read_catalog_spawn_heights() -> dict[str, float]:
    heights: dict[str, float] = {}
    doc = next(d for d in parse_unity_yaml(CATALOG).values() if d["cid"] == CID_MONOBEHAVIOUR)
    for entry in doc["data"].get("_entries", []):
        heights[entry["id"]] = float(entry["spawnHeight"])
    return heights


# ----------------------------------------------------------------------------
# Math (Unity quaternions are (x, y, z, w); Hamilton product, same as PhysX)
# ----------------------------------------------------------------------------
def v3(d) -> np.ndarray:
    return np.array([float(d["x"]), float(d["y"]), float(d["z"])])


def q4(d) -> np.ndarray:
    q = np.array([float(d["x"]), float(d["y"]), float(d["z"]), float(d["w"])])
    return q / np.linalg.norm(q)


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def qconj(q: np.ndarray) -> np.ndarray:
    return np.array([-q[0], -q[1], -q[2], q[3]])


def qmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def pos_u2m(v) -> np.ndarray:
    return np.array([v[2], -v[0], v[1]])


def axis_u2m(v) -> np.ndarray:
    return np.array([-v[2], v[0], -v[1]])


def quat_u2m(q) -> np.ndarray:  # returns (w, x, y, z)
    return np.array([q[3], -q[2], q[0], -q[1]])


def size_u2m(v) -> np.ndarray:
    return np.array([abs(v[2]), abs(v[0]), abs(v[1])])


def fmt(values, digits: int = 6) -> str:
    out = []
    for value in np.atleast_1d(values):
        value = 0.0 if abs(value) < 10 ** (-digits - 1) else float(value)
        out.append(f"{value:.{digits}g}")
    return " ".join(out)


# ----------------------------------------------------------------------------
# Prefab -> intermediate rig
# ----------------------------------------------------------------------------
class Node:
    def __init__(self, tid: int, tdata: dict):
        self.tid = tid
        self.go = int(tdata["m_GameObject"]["fileID"])
        self.father = int(tdata["m_Father"]["fileID"])
        self.children = [int(c["fileID"]) for c in tdata.get("m_Children") or []]
        self.lpos = v3(tdata["m_LocalPosition"])
        self.lrot = q4(tdata["m_LocalRotation"])
        self.lscale = v3(tdata["m_LocalScale"])
        self.name = "?"
        self.components: list[int] = []
        self.wpos = np.zeros(3)
        self.wrot = np.array([0.0, 0.0, 0.0, 1.0])
        self.wlin = np.eye(3)  # world linear part (rotation * scale, incl. shear)
        self.lossy = np.ones(3)
        self.shear = 0.0


def build_rig(name: str, prefab: Path, spawn_height: float) -> dict:
    docs = parse_unity_yaml(prefab)
    notes: list[str] = []

    nodes = {fid: Node(fid, d["data"]) for fid, d in docs.items() if d["cid"] == CID_TRANSFORM}
    by_go: dict[int, Node] = {}
    for node in nodes.values():
        go = docs[node.go]["data"]
        node.name = go["m_Name"]
        node.components = [int(c["component"]["fileID"]) for c in go.get("m_Component", [])]
        by_go[node.go] = node
        if int(go.get("m_IsActive", 1)) != 1:
            notes.append(f"{node.name}: GameObject inactive in prefab")

    roots = [n for n in nodes.values() if n.father == 0]
    assert len(roots) == 1, f"expected one root transform, got {len(roots)}"
    root = roots[0]

    # World transforms in prefab space. Unity: world = parent * T * R * S.
    def walk(node: Node, parent: Node | None):
        if parent is None:
            node.wpos = node.lpos.copy()
            node.wrot = node.lrot.copy()
            node.wlin = qmat(node.lrot) @ np.diag(node.lscale)
        else:
            node.wpos = parent.wpos + parent.wlin @ node.lpos
            node.wrot = qmul(parent.wrot, node.lrot)
            node.wlin = parent.wlin @ qmat(node.lrot) @ np.diag(node.lscale)
        local = qmat(node.wrot).T @ node.wlin  # = diag(lossyScale) when there is no shear
        node.lossy = np.diag(local).copy()
        node.shear = float(np.max(np.abs(local - np.diag(node.lossy))))
        if node.shear > 1e-6:
            notes.append(f"{node.name}: transform has shear {node.shear:.2e} (non-uniform parent "
                         "scale under a rotated child); lossyScale is an approximation here")
        for child in node.children:
            walk(nodes[child], node)

    walk(root, None)

    def comp(node: Node, cid: int) -> list[dict]:
        return [docs[c]["data"] for c in node.components if c in docs and docs[c]["cid"] == cid]

    def comp_id(node: Node, cid: int) -> int | None:
        ids = [c for c in node.components if c in docs and docs[c]["cid"] == cid]
        return ids[0] if ids else None

    # Articulation bodies, in transform DFS order (= MuJoCo body order).
    order: list[Node] = []

    def dfs(node: Node):
        if comp(node, CID_ARTICULATIONBODY):
            order.append(node)
        for child in node.children:
            dfs(nodes[child])

    dfs(root)
    ab_to_node = {comp_id(n, CID_ARTICULATIONBODY): n for n in order}

    def body_parent(node: Node) -> Node | None:
        cur = node.father
        while cur:
            parent = nodes[cur]
            if comp(parent, CID_ARTICULATIONBODY):
                return parent
            cur = parent.father
        return None

    materials_seen: dict[str, dict] = {}
    bodies = []
    for node in order:
        ab = comp(node, CID_ARTICULATIONBODY)[0]
        parent = body_parent(node)
        R = qmat(node.wrot)

        geoms = []
        for kind, cid in (("capsule", CID_CAPSULECOLLIDER), ("box", CID_BOXCOLLIDER), ("sphere", CID_SPHERECOLLIDER)):
            for col in comp(node, cid):
                if int(col.get("m_Enabled", 1)) != 1 or int(col.get("m_IsTrigger", 0)) == 1:
                    notes.append(f"{node.name}: disabled/trigger {kind} collider skipped")
                    continue
                mat = read_physics_material(col.get("m_Material"))
                materials_seen[mat["source"]] = mat
                s = np.abs(node.lossy)
                center_u = node.lossy * v3(col["m_Center"])  # body-local, scaled
                g = {"unity_collider": kind, "material": mat["source"], "center_unity": center_u.tolist()}
                if kind == "capsule":
                    d = int(col["m_Direction"])
                    others = [i for i in range(3) if i != d]
                    r = float(col["m_Radius"]) * max(s[others[0]], s[others[1]])
                    half = max(float(col["m_Height"]) * s[d] * 0.5, r)
                    g.update(radius=r, half_height_total=half, direction=d,
                             unity_raw={"radius": float(col["m_Radius"]), "height": float(col["m_Height"]),
                                        "direction": d, "lossyScale": node.lossy.tolist()})
                    if half - r < 1e-4:
                        g["type"] = "sphere"
                        notes.append(f"{node.name}: capsule height*scale ({float(col['m_Height']) * s[d]:.4f}) <= 2*radius "
                                     f"({2 * r:.4f}); PhysX treats it as a sphere of r={r:.4f}, so does the MJCF")
                    else:
                        g["type"] = "capsule"
                        axis_u = np.zeros(3)
                        axis_u[d] = half - r
                        g["fromto_unity"] = [(center_u - axis_u).tolist(), (center_u + axis_u).tolist()]
                elif kind == "box":
                    size = v3(col["m_Size"]) * s
                    g.update(type="box", half_extents_unity=(size * 0.5).tolist())
                else:
                    r = float(col["m_Radius"]) * float(np.max(s))
                    g.update(type="sphere", radius=r)
                geoms.append(g)

        jt = JOINT_TYPES[int(ab["m_ArticulationJointType"])]
        joint = None
        if parent is not None:
            # Anchor points are in each body's local SCALED frame (TransformPoint
            # semantics). Check both sides give the same world point.
            anchor_local = v3(ab["m_AnchorPosition"])
            anchor_world_child = node.wpos + node.wlin @ anchor_local
            anchor_world_parent = parent.wpos + parent.wlin @ v3(ab["m_ParentAnchorPosition"])
            anchor_err = float(np.linalg.norm(anchor_world_child - anchor_world_parent))
            frame_child = qmul(node.wrot, q4(ab["m_AnchorRotation"]))
            frame_parent = qmul(parent.wrot, q4(ab["m_ParentAnchorRotation"]))
            rel = qmul(qconj(frame_parent), frame_child)  # child joint frame in parent joint frame
            if rel[3] < 0:
                rel = -rel
            twist0 = 2.0 * math.atan2(rel[0], rel[3])
            swing0 = 2.0 * math.degrees(math.asin(min(1.0, math.hypot(rel[1], rel[2]))))
            axis_body_u = qmat(q4(ab["m_AnchorRotation"])) @ np.array([1.0, 0.0, 0.0])
            xd = ab["m_XDrive"]
            joint = {
                "type": jt,
                "twist": DOF_LOCK[int(ab["m_Twist"])],
                "swingY": DOF_LOCK[int(ab["m_SwingY"])],
                "swingZ": DOF_LOCK[int(ab["m_SwingZ"])],
                "anchor_body_unity": (node.lossy * anchor_local).tolist(),
                "axis_body_unity": axis_body_u.tolist(),
                "axis_world_unity": (R @ axis_body_u).tolist(),
                "anchor_mismatch_m": anchor_err,
                "rest_angle_rad": twist0,
                "rest_swing_deg": swing0,
                "lower_deg": float(xd["lowerLimit"]),
                "upper_deg": float(xd["upperLimit"]),
                "stiffness_Nm_per_rad": float(xd["stiffness"]),
                "damping_Nms_per_rad": float(xd["damping"]),
                "force_limit_Nm": float(xd["forceLimit"]),
                "drive_target_deg": float(xd["target"]),
                "drive_target_velocity": float(xd["targetVelocity"]),
                "drive_type": DRIVE_TYPES.get(int(xd.get("driveType", 0)), "?"),
                "joint_friction_coef": float(ab["m_JointFriction"]),
            }
            if anchor_err > 1e-4:
                notes.append(f"{node.name}: parent/child anchors disagree by {anchor_err * 1000:.2f} mm "
                             "(child anchor used)")
            if abs(twist0) > 1e-4 or swing0 > 1e-3:
                notes.append(f"{node.name}: joint not at zero in the prefab pose (twist {math.degrees(twist0):.3f} deg, "
                             f"swing {swing0:.3f} deg); MJCF uses ref= so Unity angle == MuJoCo qpos")
            for other in ("m_YDrive", "m_ZDrive"):
                if float(ab[other]["stiffness"]) or float(ab[other]["damping"]):
                    notes.append(f"{node.name}: {other} has gains but its DoF is {jt}; ignored")
            if jt not in ("revolute", "fixed"):
                raise NotImplementedError(f"{node.name}: joint type {jt} not supported by this converter")

        bodies.append({
            "name": node.name,
            "parent": parent.name if parent else None,
            "articulation_body_file_id": comp_id(node, CID_ARTICULATIONBODY),
            "mass": float(ab["m_Mass"]),
            "implicit_com": int(ab["m_ImplicitCom"]) == 1,
            "implicit_tensor": int(ab["m_ImplicitTensor"]) == 1,
            "com_unity": v3(ab["m_CenterOfMass"]).tolist(),
            "inertia_tensor_unity": v3(ab["m_InertiaTensor"]).tolist(),
            "inertia_rotation_unity": q4(ab["m_InertiaRotation"]).tolist(),
            "linear_damping": float(ab["m_LinearDamping"]),
            "angular_damping": float(ab["m_AngularDamping"]),
            "immovable": int(ab.get("m_Immovable", 0)) == 1,
            "use_gravity": int(ab.get("m_UseGravity", 1)) == 1,
            "world_pos_unity": node.wpos.tolist(),
            "world_rot_unity": node.wrot.tolist(),
            "lossy_scale_unity": node.lossy.tolist(),
            "geoms": geoms,
            "joint": joint,
            "_node": node,
            "_parent_node": parent,
        })
        if not bodies[-1]["implicit_com"] or not bodies[-1]["implicit_tensor"]:
            notes.append(f"{node.name}: explicit COM/inertia in the prefab; written as <inertial>")
        if not bodies[-1]["use_gravity"]:
            notes.append(f"{node.name}: useGravity is off in Unity; MuJoCo applies gravity to all bodies")

    # Agent MonoBehaviour: action order, drive scale, torque cap, gait tables.
    agent = None
    decision_period = None
    for fid, d in docs.items():
        if d["cid"] != CID_MONOBEHAVIOUR:
            continue
        ident = d["data"].get("m_EditorClassIdentifier", "")
        if ident.endswith("DecisionRequester"):
            decision_period = int(d["data"].get("DecisionPeriod", 0))
        if "_joints" in d["data"] and "_jointDriveScale" in d["data"]:
            a = d["data"]
            agent = {
                "class": ident,
                "root": ab_to_node[int(a["_root"]["fileID"])].name,
                "action_order": [ab_to_node[int(j["fileID"])].name for j in a["_joints"]],
                "jointDriveScale_deg": float(a["_jointDriveScale"]),
                "maxJointTorque_Nm": float(a["_maxJointTorque"]),
                "gaitFrequency_hz": float(a.get("_gaitFrequency", 0.0)),
                "gaitPhases": [float(x) for x in (a.get("_gaitPhases") or [])],
                "gaitAmplitudes": [float(x) for x in (a.get("_gaitAmplitudes") or [])],
                "gaitOffsets": [float(x) for x in (a.get("_gaitOffsets") or [])],
            }
    if agent is None:
        raise RuntimeError(f"{prefab.name}: no agent MonoBehaviour with _joints/_jointDriveScale")
    agent["decisionPeriod"] = decision_period

    return {"name": name, "prefab": prefab.relative_to(REPO).as_posix(), "bodies": bodies, "agent": agent,
            "materials": list(materials_seen.values()), "notes": notes, "catalog_spawn_height": spawn_height,
            "prefab_root_world_unity": root.wpos.tolist()}


# ----------------------------------------------------------------------------
# Rig -> MJCF
# ----------------------------------------------------------------------------
def emit_mjcf(rig: dict, friction: float, root_height: float, keyframes: str = "",
              damping_mode: str = "joint") -> str:
    bodies = rig["bodies"]
    children: dict[str | None, list[dict]] = {}
    for body in bodies:
        children.setdefault(body["parent"], []).append(body)

    out: list[str] = []
    w = out.append
    w(f"<!-- {rig['name']}: generated by training/bugs/prefab_to_mjcf.py from {rig['prefab']} - do not hand-edit.")
    w("     Frames: Unity (x,y,z) left-handed Y-up Z-forward -> MuJoCo (z,-x,y) right-handed Z-up X-forward.")
    w("     Positions map by P=(z,-x,y); hinge axes (pseudovectors) by -P=(-z,x,-y); quats (x,y,z,w)->(w,-z,x,-y).")
    w("     Hence a positive Unity jointPosition / xDrive.target == a positive MuJoCo qpos / ctrl.")
    w("     Units: SI. Angles in radians. kp = Unity stiffness (N*m/rad); Unity damping (N*m*s/rad) is")
    w(f"     {'joint damping (outside the force clamp; stable at 5 ms, see README)' if damping_mode == 'joint' else 'actuator kv (inside the force clamp; literal twin, unstable at 5 ms)'};")
    w("     forcerange = +-Unity forceLimit (N*m). ctrl = Unity xDrive.target converted deg -> rad.")
    w(f"     Friction {friction:g} everywhere (Unity default PhysicsMaterial 0.6/0.6 on creature and training floor). -->")
    w(f'<mujoco model="{rig["name"]}">')
    w('  <compiler angle="radian" autolimits="true" inertiafromgeom="auto"/>')
    w('  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" cone="pyramidal"/>')
    w('  <size memory="16M"/>')
    w("  <visual><global offwidth=\"1280\" offheight=\"720\"/></visual>")
    w("  <default>")
    # solref timeconst 0.01 s = 2 x timestep, the stiffest MuJoCo recommends; PhysX
    # contacts are rigid, and the default 0.02 let the hexapod sink 10 mm on landing.
    w(f'    <geom condim="3" friction="{friction:g} 0.005 0.0001" solref="0.01 1" contype="1" conaffinity="1" rgba="0.75 0.75 0.72 1"/>')
    # PhysX articulation limits are hard; 0.01 s is MuJoCo's stiffest stable setting at 5 ms.
    w('    <joint type="hinge" limited="true" armature="0" frictionloss="0" solreflimit="0.01 1"/>')
    w("  </default>")
    w("  <asset>")
    w('    <texture name="grid" type="2d" builtin="checker" rgb1="0.32 0.34 0.36" rgb2="0.26 0.28 0.3" width="256" height="256"/>')
    w('    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0"/>')
    w("  </asset>")
    w("  <worldbody>")
    w('    <light pos="0 0 6" dir="0 0 -1" directional="true"/>')
    w(f'    <geom name="floor" type="plane" size="50 50 0.1" material="grid" friction="{friction:g} 0.005 0.0001" solref="0.01 1"/>')

    def emit_body(body: dict, depth: int):
        ind = "  " * depth
        node = body["_node"]
        parent = body["_parent_node"]
        if parent is None:
            pos_m = np.array([0.0, 0.0, root_height])
            quat_m = quat_u2m(node.wrot)
        else:
            rel_u = qmat(parent.wrot).T @ (node.wpos - parent.wpos)
            pos_m = pos_u2m(rel_u)
            quat_m = quat_u2m(qmul(qconj(parent.wrot), node.wrot))
        quat_attr = "" if np.allclose(quat_m, [1, 0, 0, 0], atol=1e-9) else f' quat="{fmt(quat_m)}"'
        w(f'{ind}<body name="{body["name"]}" pos="{fmt(pos_m)}"{quat_attr}>')
        if parent is None:
            w(f'{ind}  <freejoint name="root"/>')
            w(f'{ind}  <site name="root_site" size="0.02"/>')
            w(f'{ind}  <camera name="track" mode="trackcom" pos="-3 0 1.5" xyaxes="0 -1 0 0.45 0 0.9"/>')
        else:
            j = body["joint"]
            if j["type"] == "revolute" and j["twist"] != "locked":
                rng = ""
                if j["twist"] == "limited":
                    rng = f' range="{fmt(math.radians(j["lower_deg"]))} {fmt(math.radians(j["upper_deg"]))}"'
                else:
                    rng = ' limited="false"'
                ref = "" if abs(j["rest_angle_rad"]) < 1e-6 else f' ref="{fmt(j["rest_angle_rad"])}"'
                damp = f' damping="{fmt(j["damping_Nms_per_rad"])}"' if damping_mode == "joint" else ""
                w(f'{ind}  <joint name="{body["name"]}" pos="{fmt(pos_u2m(j["anchor_body_unity"]))}" '
                  f'axis="{fmt(axis_u2m(j["axis_body_unity"]))}"{rng}{ref}{damp}/>')
        if not (body["implicit_com"] and body["implicit_tensor"]):
            com = pos_u2m(body["com_unity"])
            diag = size_u2m(body["inertia_tensor_unity"])
            iq = quat_u2m(body["inertia_rotation_unity"])
            w(f'{ind}  <inertial pos="{fmt(com)}" quat="{fmt(iq)}" mass="{fmt(body["mass"])}" diaginertia="{fmt(diag)}"/>')
        total_vol = 0.0
        vols = []
        for g in body["geoms"]:
            if g["type"] == "sphere":
                vol = 4.0 / 3.0 * math.pi * g["radius"] ** 3
            elif g["type"] == "capsule":
                r = g["radius"]
                vol = 4.0 / 3.0 * math.pi * r ** 3 + math.pi * r * r * 2 * (g["half_height_total"] - r)
            else:
                vol = 8.0 * float(np.prod(g["half_extents_unity"]))
            vols.append(vol)
            total_vol += vol
        for index, g in enumerate(body["geoms"]):
            # PhysX distributes an ArticulationBody's mass over its colliders by volume.
            mass = body["mass"] * vols[index] / total_vol if (body["implicit_com"] and body["implicit_tensor"]) else 0.0
            mass_attr = f' mass="{fmt(mass)}"' if mass > 0 else ' mass="0"'
            gname = f'{body["name"]}_geom{index if index else ""}'
            if g["type"] == "sphere":
                w(f'{ind}  <geom name="{gname}" type="sphere" size="{fmt(g["radius"])}" '
                  f'pos="{fmt(pos_u2m(g["center_unity"]))}"{mass_attr}/>')
            elif g["type"] == "capsule":
                a, b = (pos_u2m(p) for p in g["fromto_unity"])
                w(f'{ind}  <geom name="{gname}" type="capsule" size="{fmt(g["radius"])}" '
                  f'fromto="{fmt(np.concatenate([a, b]))}"{mass_attr}/>')
            else:
                w(f'{ind}  <geom name="{gname}" type="box" size="{fmt(size_u2m(g["half_extents_unity"]))}" '
                  f'pos="{fmt(pos_u2m(g["center_unity"]))}"{mass_attr}/>')
        for child in children.get(body["name"], []):
            emit_body(child, depth + 1)
        w(f"{ind}</body>")

    for top in children.get(None, []):
        emit_body(top, 2)
    w("  </worldbody>")

    # Actuators in the agent's ACTION order, so ctrl[i] is Unity action i.
    w("  <actuator>")
    by_name = {b["name"]: b for b in bodies}
    for jname in rig["agent"]["action_order"]:
        j = by_name[jname]["joint"]
        # Unity writes xDrive.target = clamp(action,-1,1) * _jointDriveScale and never
        # clamps it to the joint limits (a target past a limit just presses into it),
        # so the ctrl range is the action range, not the joint range.
        scale = math.radians(rig["agent"]["jointDriveScale_deg"])
        lo, hi = -scale, scale
        kv = f'kv="{fmt(j["damping_Nms_per_rad"])}" ' if damping_mode == "actuator" else ""
        w(f'    <position name="{jname}" joint="{jname}" kp="{fmt(j["stiffness_Nm_per_rad"])}" '
          f'{kv}forcerange="{fmt(-j["force_limit_Nm"])} {fmt(j["force_limit_Nm"])}" '
          f'ctrlrange="{fmt(lo)} {fmt(hi)}"/>')
    w("  </actuator>")
    if keyframes:
        w(keyframes)
    w("</mujoco>")
    return "\n".join(out) + "\n"


def lowest_point(model, data) -> float:
    """Lowest world z over all non-floor geoms (exact for sphere/capsule/box)."""
    import mujoco
    lowest = math.inf
    for gid in range(model.ngeom):
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        pos = data.geom_xpos[gid]
        mat = data.geom_xmat[gid].reshape(3, 3)
        size = model.geom_size[gid]
        t = model.geom_type[gid]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            z = pos[2] - size[0]
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            z = pos[2] - abs(mat[2, 2]) * size[1] - size[0]
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            z = pos[2] - float(np.sum(np.abs(mat[2, :]) * size))
        else:
            z = pos[2] - model.geom_rbound[gid]
        lowest = min(lowest, z)
    return lowest


def build(name: str, friction: float | None = None) -> dict:
    import mujoco

    heights = read_catalog_spawn_heights()
    rig = build_rig(name, CREATURES[name], heights[name])
    unity_mu = [m["dynamicFriction"] for m in rig["materials"]]
    if friction is None:
        # Creature colliders and the Unity training floor both use the default
        # material, so the Unity contact mu is 0.6 (static = dynamic).
        friction = unity_mu[0] if len(set(unity_mu)) == 1 else float(np.mean(unity_mu))

    spawn_z = rig["catalog_spawn_height"] + SPAWN_OFFSET

    # Pass 1: compile once to measure the rest height (feet exactly on the floor).
    xml = emit_mjcf(rig, friction, spawn_z)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[2] = 0.0
    mujoco.mj_forward(model, data)
    rest_z = -lowest_point(model, data)

    nq = model.nq
    zeros = " ".join(["0"] * (nq - 7))
    quat = fmt(model.qpos0[3:7])
    keys = [
        "  <keyframe>",
        f'    <key name="spawn" qpos="0 0 {fmt(spawn_z)} {quat} {zeros}" ctrl="{" ".join(["0"] * model.nu)}"/>',
        f'    <key name="rest" qpos="0 0 {fmt(rest_z)} {quat} {zeros}" ctrl="{" ".join(["0"] * model.nu)}"/>',
        "  </keyframe>",
    ]
    xml = emit_mjcf(rig, friction, spawn_z, "\n".join(keys), damping_mode="joint")
    model = mujoco.MjModel.from_xml_string(xml)  # compile check
    (HERE / f"{name}.xml").write_text(xml, encoding="utf-8")
    # Literal twin (damping inside the force clamp) for comparisons only - unstable at 5 ms.
    literal = emit_mjcf(rig, friction, spawn_z, "\n".join(keys), damping_mode="actuator")
    mujoco.MjModel.from_xml_string(literal)
    (HERE / f"{name}_literal.xml").write_text(literal, encoding="utf-8")

    # Side check: Unity x<0 (left) must be MuJoCo y>0 (left).
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    side_checks = []
    for body in rig["bodies"]:
        ux = body["world_pos_unity"][0]
        if abs(ux) < 1e-6:
            continue
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body["name"])
        my = float(data.xpos[bid][1])
        side_checks.append({"body": body["name"], "unity_x": ux, "mujoco_y": my, "ok": (ux < 0) == (my > 0)})

    joint_qpos_order = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    total_mass = float(sum(b["mass"] for b in rig["bodies"]))
    rig_json = {
        "name": name,
        "prefab": rig["prefab"],
        "mjcf": f"training/bugs/{name}.xml",
        "mjcf_literal": f"training/bugs/{name}_literal.xml",
        "damping_mode": "Unity damping as MuJoCo joint damping (outside the forceLimit clamp); "
                        "the literal actuator-kv form is in mjcf_literal (unstable at 5 ms)",
        "generator": "training/bugs/prefab_to_mjcf.py",
        "frames": {
            "unity": "left-handed, Y-up, +X right, +Z forward",
            "mujoco": "right-handed, Z-up, +X forward, +Y left",
            "position_map": "(x,y,z)_unity -> (z,-x,y)_mujoco",
            "axis_map": "(x,y,z)_unity -> (-z,x,-y)_mujoco",
            "quat_map": "(x,y,z,w)_unity -> (w,-z,x,-y)_mujoco",
            "joint_angle_sign": "Unity jointPosition == MuJoCo qpos (same sign, radians)",
        },
        "units": {
            "stiffness": "N*m/rad (Unity ArticulationDrive.stiffness docs) == MuJoCo kp",
            "damping": "N*m*s/rad (Unity ArticulationDrive.damping docs) == MuJoCo kv",
            "target": "degrees in Unity xDrive.target; radians in MuJoCo ctrl",
            "limits": "degrees in Unity; radians in MJCF",
        },
        "catalog_spawn_height": rig["catalog_spawn_height"],
        "unity_spawn_root_height": spawn_z,
        "rest_root_height": rest_z,
        "prefab_root_height": rig["prefab_root_world_unity"][1],
        "total_mass": total_mass,
        "friction_mujoco": friction,
        "unity_materials": rig["materials"],
        "agent": rig["agent"],
        "action_scale_rad": math.radians(rig["agent"]["jointDriveScale_deg"]),
        "action_to_ctrl": "ctrl[i] = clip(action[i],-1,1) * action_scale_rad (actuator i == Unity _joints[i])",
        "actuator_order": rig["agent"]["action_order"],
        "joint_qpos_order": joint_qpos_order,
        "bodies": [
            {k: v for k, v in b.items() if not k.startswith("_")} for b in rig["bodies"]
        ],
        "side_checks": side_checks,
        "notes": rig["notes"],
    }
    for body in rig_json["bodies"]:
        if body["joint"] is not None:
            j = body["joint"]
            j["axis_body_mujoco"] = axis_u2m(j["axis_body_unity"]).tolist()
            j["range_rad"] = [math.radians(j["lower_deg"]), math.radians(j["upper_deg"])]
    (HERE / f"{name}_rig.json").write_text(json.dumps(rig_json, indent=2), encoding="utf-8")
    return rig_json


def main(argv: list[str]) -> int:
    names = argv or list(CREATURES)
    ok = True
    for name in names:
        rig = build(name)
        bad = [c for c in rig["side_checks"] if not c["ok"]]
        ok &= not bad
        print(f"{name}: {len(rig['bodies'])} bodies, {len(rig['joint_qpos_order'])} hinges, "
              f"mass {rig['total_mass']:.2f} kg, rest root z {rig['rest_root_height']:.4f} m, "
              f"catalog {rig['catalog_spawn_height']:.3f} (+{SPAWN_OFFSET}) m, side check "
              f"{'OK' if not bad else 'FAILED ' + str(bad)}")
        for note in rig["notes"]:
            print(f"   note: {note}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
