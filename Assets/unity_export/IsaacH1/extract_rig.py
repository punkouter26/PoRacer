#!/usr/bin/env python3
"""
extract_rig.py - build IsaacH1_rig.json from the Isaac Lab export.

Source precedence (the Isaac cfg wins over the vendor URDF everywhere they differ):

  1. checkpoint/params/env.yaml   - actuator gains, effort limits, solver counts,
                                    per-body velocity caps, friction events.
  2. robot/usd/h1_minimal.usd     - the EXACT rigid bodies Isaac simulated:
                                    mass / CoM / diagonal inertia / armature,
                                    joint anchors + axes + limits, and the fact
                                    that only 3 collision shapes exist.
  3. export_report.json           - joint order, defaults, action scale, timing.
  4. robot/h1.urdf                - visual proxy sizes ONLY. Its masses, inertias
                                    and joint limits are WRONG for this policy.

Everything is emitted in the ISAAC frame (right-handed, Z-up, metres, radians).
The Unity frame map lives in C# (IsaacH1FrameMap) so it is visible and testable.

Usage:  python extract_rig.py [--export-dir ../../h1] [--out IsaacH1_rig.json]

Requires: usd-core (pip install usd-core), pyyaml.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXPORT = os.path.normpath(os.path.join(HERE, "..", "..", "h1"))


# --------------------------------------------------------------------------- #
# small maths helpers (kept dependency-free so the JSON is reproducible)
# --------------------------------------------------------------------------- #
def quat_to_mat(w, x, y, z):
    """USD quatf storage order is (w, x, y, z). Returns a row-major 3x3."""
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n == 0.0:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    w, x, y, z = w / n, x / n, y / n, z / n
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def mat_to_quat(m):
    """Row-major 3x3 -> (w, x, y, z)."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    n = math.sqrt(w * w + x * x + y * y + z * z)
    return [w / n, x / n, y / n, z / n]


def mat_T(m):
    return [[m[j][i] for j in range(3)] for i in range(3)]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def vsub(a, b):
    return [a[i] - b[i] for i in range(3)]


def rpy_to_mat(r, p, y):
    """URDF fixed-axis roll-pitch-yaw -> Rz(y) Ry(p) Rx(r)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def r6(v):
    """Round for stable, diff-able JSON without losing physical precision."""
    if isinstance(v, list):
        return [r6(x) for x in v]
    return round(float(v), 9)


# --------------------------------------------------------------------------- #
# USD -> bodies + joints (the authoritative source)
# --------------------------------------------------------------------------- #
def read_usd(usd_path):
    try:
        from pxr import Usd, UsdGeom  # noqa: F401
    except ImportError:
        sys.exit("ERROR: usd-core is required.  pip install usd-core")
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)
    if UsdGeom.GetStageUpAxis(stage) != "Z":
        sys.exit("ERROR: expected a Z-up stage (Isaac convention)")
    if abs(UsdGeom.GetStageMetersPerUnit(stage) - 1.0) > 1e-9:
        sys.exit("ERROR: expected metersPerUnit == 1.0; rig would be mis-scaled")

    bodies, joints, colliders = {}, {}, {}

    for prim in stage.Traverse():
        name = prim.GetName()
        if prim.GetTypeName() == "Xform" and "PhysicsRigidBodyAPI" in prim.GetAppliedSchemas():
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            if len(ops) != 1 or ops[0].GetOpName() != "xformOp:transform":
                sys.exit(f"ERROR: unexpected xform ops on {prim.GetPath()}")
            m = ops[0].Get()
            # USD Gf.Matrix4d is ROW-vector convention (v' = v * M): translation
            # lives in row 3 and the basis vectors are the ROWS. Everything below
            # uses column-vector convention (v' = R v), so transpose the 3x3.
            rot = [[m[c][r] for c in range(3)] for r in range(3)]
            pos = [m[3][0], m[3][1], m[3][2]]
            g = lambda a: prim.GetAttribute(a).Get()  # noqa: E731
            pa = g("physics:principalAxes")
            # (0,0,0,0) is USD's "unset" -> inertia is diagonal in the body frame.
            pa_is_identity = pa is None or (
                abs(pa.GetReal()) < 1e-12
                and all(abs(c) < 1e-12 for c in pa.GetImaginary())
            )
            if not pa_is_identity:
                sys.exit(f"ERROR: {name} has a non-identity principalAxes; unsupported")
            di = g("physics:diagonalInertia")
            com = g("physics:centerOfMass")
            bodies[name] = {
                "name": name,
                "mass": float(g("physics:mass")),
                "com": [com[0], com[1], com[2]],
                "inertiaDiag": [di[0], di[1], di[2]],
                "worldPos": pos,
                "worldRot": rot,
                "isArticulationRoot": "PhysicsArticulationRootAPI" in prim.GetAppliedSchemas(),
            }

        if prim.GetTypeName() == "PhysicsRevoluteJoint":
            g = lambda a: prim.GetAttribute(a).Get()  # noqa: E731
            b0 = prim.GetRelationship("physics:body0").GetTargets()
            b1 = prim.GetRelationship("physics:body1").GetTargets()
            lp0, lp1 = g("physics:localPos0"), g("physics:localPos1")
            lr0, lr1 = g("physics:localRot0"), g("physics:localRot1")
            axis_letter = g("physics:axis")
            joints[name] = {
                "name": name,
                "parent": os.path.basename(str(b0[0])),
                "child": os.path.basename(str(b1[0])),
                "localPos0": [lp0[0], lp0[1], lp0[2]],
                "localPos1": [lp1[0], lp1[1], lp1[2]],
                "localRot0": [lr0.GetReal(), *lr0.GetImaginary()],
                "localRot1": [lr1.GetReal(), *lr1.GetImaginary()],
                "usdAxisLetter": axis_letter,
                "lowerLimitDeg": float(g("physics:lowerLimit")),
                "upperLimitDeg": float(g("physics:upperLimit")),
                "armature": float(g("physxJoint:armature") or 0.0),
            }

    # collision shapes, resolved through the instanceable_meshes reference
    from pxr import Usd as _Usd
    for prim in stage.Traverse(_Usd.TraverseInstanceProxies(_Usd.PrimAllPrimsPredicate)):
        if "PhysicsCollisionAPI" not in prim.GetAppliedSchemas():
            continue
        path = str(prim.GetPath())
        if not path.startswith("/h1/"):
            continue  # GroundPlane
        link = path.split("/")[2]
        pts = prim.GetAttribute("points").Get()
        if pts is None:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        lo = [min(xs), min(ys), min(zs)]
        hi = [max(xs), max(ys), max(zs)]
        colliders.setdefault(link, []).append(
            {
                "approximation": str(prim.GetAttribute("physics:approximation").Get()),
                "vertexCount": len(pts),
                # convex hull reduced to its axis-aligned box in the link frame
                "boxCenter": [(lo[i] + hi[i]) / 2.0 for i in range(3)],
                "boxSize": [hi[i] - lo[i] for i in range(3)],
                "boxMin": lo,
                "boxMax": hi,
            }
        )
    return bodies, joints, colliders


def urdf_visual_proxies(urdf_path):
    """URDF collision primitives, used ONLY to draw non-colliding visual proxies."""
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall("link"):
        shapes = []
        for col in link.findall("collision"):
            geom = col.find("geometry")
            if geom is None or len(geom) == 0:
                continue
            g = list(geom)[0]
            o = col.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0").split())] if o is not None else [0, 0, 0]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0").split())] if o is not None else [0, 0, 0]
            s = {"origin": xyz, "rpy": rpy, "kind": g.tag}
            if g.tag == "box":
                s["size"] = [float(v) for v in g.get("size").split()]
            elif g.tag == "cylinder":
                s["radius"] = float(g.get("radius"))
                s["length"] = float(g.get("length"))
            elif g.tag == "sphere":
                s["radius"] = float(g.get("radius"))
            else:
                continue
            shapes.append(s)
        if shapes:
            out[link.get("name")] = shapes
    return out


def urdf_masses(urdf_path):
    root = ET.parse(urdf_path).getroot()
    out = {}
    for link in root.findall("link"):
        i = link.find("inertial")
        if i is None:
            continue
        I = i.find("inertia")
        o = i.find("origin")
        out[link.get("name")] = {
            "mass": float(i.find("mass").get("value")),
            "com": [float(v) for v in (o.get("xyz", "0 0 0").split())] if o is not None else [0, 0, 0],
            "inertiaDiag": [float(I.get("ixx")), float(I.get("iyy")), float(I.get("izz"))],
        }
    return out


def urdf_joint_velocity_limits(urdf_path):
    root = ET.parse(urdf_path).getroot()
    out = {}
    for j in root.findall("joint"):
        lim = j.find("limit")
        if lim is None:
            continue
        # Isaac drops the "_joint" suffix
        nm = j.get("name")
        nm = nm[:-6] if nm.endswith("_joint") else nm
        out[nm] = {"velocity": float(lim.get("velocity")), "effort": float(lim.get("effort"))}
    return out


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default=DEFAULT_EXPORT)
    ap.add_argument("--out", default=os.path.join(HERE, "IsaacH1_rig.json"))
    args = ap.parse_args()

    ex = args.export_dir
    report = json.load(open(os.path.join(ex, "export_report.json")))
    bodies, joints, colliders = read_usd(os.path.join(ex, "robot", "usd", "h1_minimal.usd"))
    proxies = urdf_visual_proxies(os.path.join(ex, "robot", "h1.urdf"))
    umass = urdf_masses(os.path.join(ex, "robot", "h1.urdf"))
    uvel = urdf_joint_velocity_limits(os.path.join(ex, "robot", "h1.urdf"))

    joint_order = report["joints"]["order"]
    body_order = report["bodies"]["names"]
    act = report["actuators"]["resolved_per_joint"]

    # ---- cross-checks: fail loudly rather than ship a silently wrong rig ----
    problems = []
    if set(joint_order) != set(joints):
        problems.append(f"joint name mismatch: report={sorted(joint_order)} usd={sorted(joints)}")
    if set(body_order) != set(bodies):
        problems.append(f"body name mismatch: report={sorted(body_order)} usd={sorted(bodies)}")
    for i, jn in enumerate(joint_order):
        lo_rad = math.radians(joints[jn]["lowerLimitDeg"])
        hi_rad = math.radians(joints[jn]["upperLimitDeg"])
        rlo, rhi = report["joints"]["pos_limits_rad"][i]
        if abs(lo_rad - rlo) > 2e-4 or abs(hi_rad - rhi) > 2e-4:
            problems.append(f"{jn}: USD limits {lo_rad:.5f}/{hi_rad:.5f} != report {rlo:.5f}/{rhi:.5f}")
    if problems:
        sys.exit("RIG CROSS-CHECK FAILED:\n  " + "\n  ".join(problems))

    # ---- parent / child topology, in Isaac's own body order ----
    child_joint = {j["child"]: j for j in joints.values()}
    root_name = next(n for n, b in bodies.items() if b["isArticulationRoot"])

    out_bodies = []
    for bn in body_order:
        b = bodies[bn]
        j = child_joint.get(bn)
        parent = j["parent"] if j else None

        entry = {
            "name": bn,
            "parent": parent,
            "isRoot": bn == root_name,
            "mass": r6(b["mass"]),
            "com": r6(b["com"]),
            "inertiaDiag": r6(b["inertiaDiag"]),
            "urdfMass": r6(umass[bn]["mass"]) if bn in umass else None,
            "urdfInertiaDiag": r6(umass[bn]["inertiaDiag"]) if bn in umass else None,
        }

        # pose of this link relative to its parent at the ZERO-joint pose
        if parent is None:
            entry["localPos"] = [0.0, 0.0, 0.0]
            entry["localRotWxyz"] = [1.0, 0.0, 0.0, 0.0]
        else:
            p = bodies[parent]
            Rp_T = mat_T(p["worldRot"])
            entry["localPos"] = r6(mat_vec(Rp_T, vsub(b["worldPos"], p["worldPos"])))
            entry["localRotWxyz"] = r6(mat_to_quat(mat_mul(Rp_T, b["worldRot"])))
            # the joint anchor must coincide with the child link origin
            d = max(abs(entry["localPos"][i] - j["localPos0"][i]) for i in range(3))
            if d > 1e-5:
                sys.exit(f"ERROR: {bn} localPos {entry['localPos']} != joint localPos0 {j['localPos0']} (d={d})")
            if max(abs(v) for v in j["localPos1"]) > 1e-9:
                sys.exit(f"ERROR: {bn} joint localPos1 is non-zero; builder assumes anchor at child origin")

        if j:
            ji = joint_order.index(j["name"])
            # USD authors every revolute about the joint frame's X; localRot1 carries
            # that frame into the CHILD link frame, which is what Unity's anchor needs.
            Rc = quat_to_mat(*j["localRot1"])
            axis_letter = {"X": 0, "Y": 1, "Z": 2}[j["usdAxisLetter"]]
            axis_child = [Rc[r][axis_letter] for r in range(3)]
            n = math.sqrt(sum(v * v for v in axis_child))
            axis_child = [v / n for v in axis_child]

            entry["joint"] = {
                "name": j["name"],
                "index": ji,
                # unit axis expressed in the CHILD link frame (Isaac, right-handed)
                "axisInChild": r6(axis_child),
                "lowerRad": r6(math.radians(j["lowerLimitDeg"])),
                "upperRad": r6(math.radians(j["upperLimitDeg"])),
                "stiffness": r6(act["stiffness"][ji]),
                "damping": r6(act["damping"][ji]),
                "effortLimit": r6(act["effort_limit"][ji]),
                "defaultPosRad": r6(report["joints"]["default_pos_rad"][ji]),
                # PhysX articulation rotor inertia. Unity has no equivalent field;
                # IsaacH1RigBuilder folds it into the child inertia about this axis.
                "armature": r6(j["armature"]),
                "urdfVelocityLimit": r6(uvel[j["name"]]["velocity"]) if j["name"] in uvel else None,
                "urdfEffortLimit": r6(uvel[j["name"]]["effort"]) if j["name"] in uvel else None,
            }

        # collision: EXACTLY the shapes Isaac simulated (torso + 2 feet), nothing else
        entry["colliders"] = [
            {
                "type": "box",
                "center": r6(c["boxCenter"]),
                "size": r6(c["boxSize"]),
                "sourceApproximation": c["approximation"],
                "sourceVertexCount": c["vertexCount"],
            }
            for c in colliders.get(bn, [])
        ]
        entry["visualProxies"] = [
            {
                "kind": s["kind"],
                "origin": r6(s["origin"]),
                "rpy": r6(s["rpy"]),
                **({"size": r6(s["size"])} if "size" in s else {}),
                **({"radius": r6(s["radius"])} if "radius" in s else {}),
                **({"length": r6(s["length"])} if "length" in s else {}),
            }
            for s in proxies.get(bn, [])
        ]
        out_bodies.append(entry)

    ev = json.load(open(os.path.join(ex, "export_report.json")))  # alias for clarity
    rig = {
        "_comment": "Generated by extract_rig.py. ISAAC FRAME (right-handed, Z-up, m, rad). "
                    "Do not hand-edit; re-run the script.",
        "creature": "IsaacH1",
        "sourceTask": report["task"],
        "jointOrder": joint_order,
        "bodyOrder": body_order,
        "actionScale": r6(report["actions"]["scale"]),
        "useDefaultOffset": bool(report["actions"]["use_default_offset"]),
        "obsDim": report["observations"]["dim"],
        "actDim": report["actions"]["dim"],
        "obsLayout": report["observations"]["layout"],
        "timing": {
            "policyDt": r6(report["timing"]["policy_dt_s"]),
            "isaacPhysicsDt": r6(report["timing"]["physics_dt_s"]),
            "isaacDecimation": report["timing"]["decimation"],
        },
        "spawn": {
            "posIsaac": r6(report["spawn"]["pos_isaac_xyz_m"]),
            "rotXyzw": r6(report["spawn"]["rot_wxyz"]),  # see CONTRACT.md: field is XYZW
        },
        "physics": {
            "gravity": r6(report["physics"]["gravity_m_s2"]),
            # sim.physics_material - the GROUND
            "groundStaticFriction": 1.0,
            "groundDynamicFriction": 1.0,
            "groundRestitution": 0.0,
            # events.physics_material - the ROBOT's own shapes (startup event,
            # degenerate range => a fixed value, not a random draw)
            "robotStaticFriction": 0.8,
            "robotDynamicFriction": 0.6,
            "robotRestitution": 0.0,
            "frictionCombineMode": "multiply",
            # scene.robot.spawn.rigid_props / articulation_props
            "maxLinearVelocity": 1000.0,
            "maxAngularVelocity": 1000.0,
            "maxDepenetrationVelocity": 1.0,
            "linearDamping": 0.0,
            "angularDamping": 0.0,
            "jointFriction": 0.0,
            "solverPositionIterations": 4,
            "solverVelocityIterations": 4,
            "enabledSelfCollisions": False,
            # PhysX defaults Isaac left untouched (env.yaml overrides neither)
            "contactOffset": 0.02,
            "restOffset": 0.0,
            "isaacSolverType": "TGS",
        },
        "eval": {
            "meanSpeed": r6(report["evaluation"]["mean_speed_m_s"]),
            "meanLinVelTrackingError": r6(report["evaluation"]["mean_lin_vel_tracking_error_m_s"]),
            "fallsPerRobotPerMinute": r6(report["evaluation"]["falls_per_robot_per_minute"]),
            "referenceCommand": r6(report["task_parameters"]["reference_command_used"]),
        },
        "deviationsFromUrdf": {
            "note": "The vendor URDF disagrees with the simulated USD. Isaac wins.",
            "totalMassUsd": r6(sum(b["mass"] for b in bodies.values())),
            "totalMassUrdf": r6(sum(v["mass"] for v in umass.values())),
            "torsoMassNominalUsd": r6(bodies["torso_link"]["mass"]),
            "torsoMassInReferenceRecording": r6(report["bodies"]["masses_kg"][body_order.index("torso_link")]),
            "torsoMassRandomisation": [0.8, 1.25],
        },
        "bodies": out_bodies,
    }

    with open(args.out, "w", newline="\n") as f:
        json.dump(rig, f, indent=2)
        f.write("\n")

    n_col = sum(len(b["colliders"]) for b in out_bodies)
    print(f"wrote {args.out}")
    print(f"  bodies={len(out_bodies)} joints={len(joint_order)} collisionShapes={n_col}")
    print(f"  total mass USD={rig['deviationsFromUrdf']['totalMassUsd']} kg "
          f"(URDF says {rig['deviationsFromUrdf']['totalMassUrdf']} kg)")
    print(f"  armature={out_bodies[1]['joint']['armature']} kg.m^2 on all joints "
          f"(Unity has no equivalent field - folded into inertia by the builder)")


if __name__ == "__main__":
    main()
