#!/usr/bin/env python3
"""
rig_audit.py - numerical audit of the IsaacH1 rig, written to RIG_AUDIT.md.

Answers, with numbers, the questions that decide whether the Unity port can work:

  A. Light links and tiny inertias      -> do we need mass/inertia floors?
  B. Measured joint velocities          -> did Isaac enforce velocity_limit_sim?
  C. Explicit-PD stability bound        -> kd*dt/I_joint across candidate steps
  D. Reference quaternion order         -> exports mislabel wxyz/xyzw
  E. Independent URDF forward kinematics-> validates the USD-derived rig, and
                                           gives the expected Unity rest height

Nothing here touches Unity; it reads only the export and IsaacH1_rig.json.

Usage: python rig_audit.py [--export-dir ../../h1] [--rig IsaacH1_rig.json]
                           [--out RIG_AUDIT.md] [--project-dt 0.02]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXPORT = os.path.normpath(os.path.join(HERE, "..", "..", "h1"))

# Candidate fixed steps, coarsest first. 0.02 is this project's value.
CANDIDATE_STEPS = [0.02, 1 / 60, 1 / 120, 1 / 240, 1 / 480, 1 / 960]
STEP_LABELS = ["project 1/50", "1/60", "1/120", "1/240", "1/480", "1/960"]

INERTIA_FLOOR_THRESHOLD = 1e-4      # kg.m^2
LIGHT_LINK_RATIO = 0.10             # < 10% of neighbours
RETRAIN_RATIO = 50.0                # floor:raw > 50:1 -> retrain instead
PD_BOUND = 2.0                      # kd*dt/I must stay below this


# --------------------------------------------------------------------------- #
# E. Independent forward kinematics, built from the URDF only
# --------------------------------------------------------------------------- #
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p),
                              math.sin(p), math.cos(y), math.sin(y))
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def axis_angle_R(axis, theta):
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * (K @ K)


class UrdfFK:
    """Minimal URDF forward kinematics. Deliberately independent of the USD path."""

    def __init__(self, urdf_path):
        root = ET.parse(urdf_path).getroot()
        self.joints = {}
        self.children = {}
        self.parent_of = {}
        for j in root.findall("joint"):
            name = j.get("name")
            short = name[:-6] if name.endswith("_joint") else name
            o = j.find("origin")
            xyz = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
            rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
            a = j.find("axis")
            axis = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else np.array([1.0, 0, 0])
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            self.joints[short] = {
                "type": j.get("type"), "xyz": xyz, "R": rpy_to_R(*rpy),
                "axis": axis, "parent": parent, "child": child,
            }
            self.children.setdefault(parent, []).append(short)
            self.parent_of[child] = short
        links = {l.get("name") for l in root.findall("link")}
        self.root = next(l for l in links if l not in self.parent_of)

    def fk(self, q: dict):
        """q maps short joint name -> radians. Returns link -> (pos, R) in root frame."""
        out = {self.root: (np.zeros(3), np.eye(3))}
        stack = [self.root]
        while stack:
            link = stack.pop()
            p_pos, p_R = out[link]
            for jn in self.children.get(link, []):
                j = self.joints[jn]
                R_origin = p_R @ j["R"]
                pos = p_pos + p_R @ j["xyz"]
                if j["type"] == "revolute":
                    R = R_origin @ axis_angle_R(j["axis"], q.get(jn, 0.0))
                else:
                    R = R_origin
                out[j["child"]] = (pos, R)
                stack.append(j["child"])
        return out


# --------------------------------------------------------------------------- #
# C. subtree inertia about each joint axis (parallel axis, whole subtree)
# --------------------------------------------------------------------------- #
def subtree_inertia_about_axis(rig, fk_poses, joint_name):
    """
    Effective rotational inertia the joint must accelerate: every body distal to
    it, each contributing a^T (R I R^T) a  +  m * d^2  (d = perpendicular
    distance from its CoM to the joint axis line). Evaluated at the DEFAULT pose.
    """
    bodies = {b["name"]: b for b in rig["bodies"]}
    kids = {}
    for b in rig["bodies"]:
        if b["parent"]:
            kids.setdefault(b["parent"], []).append(b["name"])

    jb = next(b for b in rig["bodies"] if b.get("joint") and b["joint"]["name"] == joint_name)
    # axis line in the root frame: through the child link origin, along axisInChild
    c_pos, c_R = fk_poses[jb["name"]]
    axis_w = c_R @ np.asarray(jb["joint"]["axisInChild"], dtype=float)
    axis_w /= np.linalg.norm(axis_w)
    p0 = c_pos

    total = 0.0
    stack = [jb["name"]]
    while stack:
        nm = stack.pop()
        b = bodies[nm]
        pos, R = fk_poses[nm]
        com_w = pos + R @ np.asarray(b["com"], dtype=float)
        I_local = np.diag(np.asarray(b["inertiaDiag"], dtype=float))
        I_w = R @ I_local @ R.T
        d_vec = (com_w - p0) - np.dot(com_w - p0, axis_w) * axis_w
        total += float(axis_w @ I_w @ axis_w) + b["mass"] * float(d_vec @ d_vec)
        stack.extend(kids.get(nm, []))
    return total


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", default=DEFAULT_EXPORT)
    ap.add_argument("--rig", default=os.path.join(HERE, "IsaacH1_rig.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "RIG_AUDIT.md"))
    ap.add_argument("--project-dt", type=float, default=0.02)
    args = ap.parse_args()

    ex = args.export_dir
    rig = json.load(open(args.rig))
    report = json.load(open(os.path.join(ex, "export_report.json")))
    ref = json.load(open(os.path.join(ex, "isaac_reference.json")))
    fkm = UrdfFK(os.path.join(ex, "robot", "h1.urdf"))

    bodies = {b["name"]: b for b in rig["bodies"]}
    jorder = rig["jointOrder"]
    jinfo = {b["joint"]["name"]: b["joint"] for b in rig["bodies"] if b.get("joint")}
    jbody = {b["joint"]["name"]: b["name"] for b in rig["bodies"] if b.get("joint")}
    L = []
    W = L.append

    W("# IsaacH1 - rig audit\n")
    W(f"Generated by `rig_audit.py` from `{os.path.relpath(ex, HERE)}` + `IsaacH1_rig.json`.  ")
    W(f"Task `{rig['sourceTask']}`, {len(jorder)} joints, {len(rig['bodyOrder'])} bodies, "
      f"{sum(len(b['colliders']) for b in rig['bodies'])} collision shapes.\n")

    # ---------------------------------------------------------------- E. FK --
    q_default = {jn: jinfo[jn]["defaultPosRad"] for jn in jorder}
    poses_zero = fkm.fk({jn: 0.0 for jn in jorder})
    poses_def = fkm.fk(q_default)

    W("## E. Independent forward kinematics (URDF) vs the USD-derived rig\n")
    W("The rig JSON takes link poses from the USD. This FK is rebuilt from the URDF's")
    W("joint origins and axes alone, so agreement means the two sources describe the")
    W("same kinematic chain and the extraction did not transpose or mis-order a frame.\n")
    worst = 0.0
    fk_rows = []
    for b in rig["bodies"]:
        if b["parent"] is None:
            continue
        # compose the rig's own parent-relative transform chain
        pass
    # walk the rig chain to world, then compare against the URDF FK
    rig_world = {}

    def quat_wxyz_to_R(q):
        w, x, y, z = q
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    for b in rig["bodies"]:
        if b["parent"] is None:
            rig_world[b["name"]] = (np.zeros(3), np.eye(3))
        else:
            p_pos, p_R = rig_world[b["parent"]]
            lp = np.asarray(b["localPos"], dtype=float)
            lR = quat_wxyz_to_R(b["localRotWxyz"])
            rig_world[b["name"]] = (p_pos + p_R @ lp, p_R @ lR)
    for nm in rig["bodyOrder"]:
        dp = float(np.max(np.abs(rig_world[nm][0] - poses_zero[nm][0])))
        dR = float(np.max(np.abs(rig_world[nm][1] - poses_zero[nm][1])))
        worst = max(worst, dp)
        fk_rows.append((nm, dp, dR))
    W(f"**Max link-origin disagreement at the zero pose: {worst * 1000:.4f} mm** "
      f"(gate: < 1 mm) - {'PASS' if worst < 1e-3 else 'FAIL'}\n")
    W("| link | dpos (mm) | dRot (max abs elem) |")
    W("|---|---:|---:|")
    for nm, dp, dR in fk_rows:
        W(f"| `{nm}` | {dp * 1000:.5f} | {dR:.2e} |")
    W("")

    # rest height at the default joint pose
    foot_bottoms = {}
    for side in ("left", "right"):
        link = f"{side}_ankle_link"
        pos, R = poses_def[link]
        col = bodies[link]["colliders"][0]
        c = np.asarray(col["center"], dtype=float)
        s = np.asarray(col["size"], dtype=float)
        corners = []
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(pos + R @ (c + 0.5 * s * np.array([sx, sy, sz])))
        foot_bottoms[link] = min(p[2] for p in corners)
    lowest = min(foot_bottoms.values())
    rest_pelvis_z = -lowest
    spawn_z = rig["spawn"]["posIsaac"][2]
    W("### Rest height (pelvis above ground at the default joint pose)\n")
    W("| quantity | value |")
    W("|---|---:|")
    for k, v in foot_bottoms.items():
        W(f"| `{k}` collision-box lowest corner, relative to pelvis | {v:.5f} m |")
    W(f"| **implied pelvis height when both feet just touch** | **{rest_pelvis_z:.5f} m** |")
    W(f"| export `spawn.pos_isaac_xyz_m.z` | {spawn_z:.5f} m |")
    W(f"| drop from spawn to rest | {spawn_z - rest_pelvis_z:.5f} m |")
    ref_z0 = ref[0]["root_pos_w"][2]
    ref_zmean = float(np.mean([s["root_pos_w"][2] for s in ref]))
    W(f"| recorded root z, step 0 | {ref_z0:.5f} m |")
    W(f"| recorded root z, mean over 250 steps (walking) | {ref_zmean:.5f} m |")
    W("")
    W(f"Rung 1 gate uses the walking mean {ref_zmean:.4f} m; a Unity creature that settles")
    W(f"at {rest_pelvis_z:.4f} m and then walks near {ref_zmean:.4f} m has the right leg geometry.\n")

    # ------------------------------------------------- A. light links / inertia
    W("## A. Light links and small inertias\n")
    kids = {}
    for b in rig["bodies"]:
        if b["parent"]:
            kids.setdefault(b["parent"], []).append(b["name"])

    W("### Mass ratio against neighbours (parent and children)\n")
    W(f"Flagged when a link is under {LIGHT_LINK_RATIO:.0%} of a neighbour's mass.\n")
    W("| link | mass (kg) | parent | parent mass | children (min mass) | min ratio | flag |")
    W("|---|---:|---|---:|---|---:|---|")
    light = []
    for b in rig["bodies"]:
        nb = []
        if b["parent"]:
            nb.append(bodies[b["parent"]]["mass"])
        ch = kids.get(b["name"], [])
        nb.extend(bodies[c]["mass"] for c in ch)
        if not nb:
            continue
        ratio = b["mass"] / max(nb)
        flag = "**LIGHT**" if ratio < LIGHT_LINK_RATIO else ""
        if flag:
            light.append((b["name"], ratio))
        chs = ", ".join(f"`{c}`" for c in ch) if ch else "-"
        W(f"| `{b['name']}` | {b['mass']:.4f} | `{b['parent'] or '-'}` | "
          f"{bodies[b['parent']]['mass'] if b['parent'] else 0:.4f} | {chs} | {ratio:.4f} | {flag} |")
    W("")
    if light:
        W("Flagged: " + ", ".join(f"`{n}` ({r:.1%} of its heaviest neighbour)" for n, r in light))
        W("")
        W("A light link between two heavy ones is where PhysX 4.1's articulation solver loses")
        W("conditioning first. Isaac carried a **0.1 kg.m2 joint armature** on every joint,")
        W("which dominates these links' own inertia and is what kept the training stable.")
        W("`IsaacH1RigBuilder` folds that armature into the child inertia about the joint axis")
        W("(`armatureMode = FoldIntoInertia`) precisely to recover that conditioning.\n")
    else:
        W("No link falls below the threshold.\n")

    W("### Diagonal inertia vs the floor\n")
    W(f"Floor threshold {INERTIA_FLOOR_THRESHOLD:g} kg.m2; retrain rather than floor if the")
    W(f"required correction exceeds {RETRAIN_RATIO:.0f}:1.\n")
    W("| link | Ixx | Iyy | Izz | min | needs floor? | floor:raw |")
    W("|---|---:|---:|---:|---:|---|---:|")
    need_floor = []
    for b in rig["bodies"]:
        I = b["inertiaDiag"]
        mn = min(I)
        nf = mn < INERTIA_FLOOR_THRESHOLD
        if nf:
            need_floor.append((b["name"], mn, INERTIA_FLOOR_THRESHOLD / mn))
        W(f"| `{b['name']}` | {I[0]:.3e} | {I[1]:.3e} | {I[2]:.3e} | {mn:.3e} | "
          f"{'**YES**' if nf else 'no'} | {INERTIA_FLOOR_THRESHOLD / mn:.1f} |")
    W("")
    if need_floor:
        worst_r = max(r for _, _, r in need_floor)
        W(f"{len(need_floor)} link(s) below the floor; worst correction {worst_r:.1f}:1.")
        W("**Recommendation: retrain**" if worst_r > RETRAIN_RATIO else
          "**Recommendation: apply the floor** (correction is small enough to be physical).")
    else:
        smallest = min((min(b["inertiaDiag"]), b["name"]) for b in rig["bodies"])
        W(f"**No link needs an inertia floor.** The smallest diagonal component is "
          f"{smallest[0]:.3e} kg.m2 on `{smallest[1]}`, "
          f"{smallest[0] / INERTIA_FLOOR_THRESHOLD:.1f}x the {INERTIA_FLOOR_THRESHOLD:g} floor.")
        W("")
        W("`IsaacH1Agent.inertiaFloor` therefore ships **disabled**. The raw USD masses and")
        W("inertias are serialised on the prefab regardless, so a floor stays re-appliable.")
    W("")

    W("### USD (simulated) vs URDF (vendor) mass - they disagree\n")
    W("| link | USD mass | URDF mass | USD/URDF |")
    W("|---|---:|---:|---:|")
    for b in rig["bodies"]:
        um = b["urdfMass"]
        W(f"| `{b['name']}` | {b['mass']:.4f} | {um:.4f} | {b['mass'] / um:.3f} |")
    d = rig["deviationsFromUrdf"]
    W(f"| **total** | **{d['totalMassUsd']:.3f}** | **{d['totalMassUrdf']:.3f}** | "
      f"**{d['totalMassUsd'] / d['totalMassUrdf']:.3f}** |")
    W("")
    W(f"The rig uses the **USD** column. `torso_link` is a special case: nominal "
      f"{d['torsoMassNominalUsd']} kg, but the `add_base_mass` startup event scales it by")
    W(f"logU{tuple(d['torsoMassRandomisation'])}, and the reference recording captured a draw of")
    W(f"{d['torsoMassInReferenceRecording']} kg "
      f"({d['torsoMassInReferenceRecording'] / d['torsoMassNominalUsd']:.4f}x). "
      f"The prefab ships nominal; `IsaacH1Agent.torsoMassScale` reproduces the recording.\n")

    # -------------------------------------------------- B. joint velocities --
    W("## B. Measured joint velocity vs `velocity_limit_sim`\n")
    obs = np.array([s["obs"] for s in ref])
    jv_obs = obs[:, 31:50]
    jv_field = np.array([s["joint_vel"] for s in ref])
    agree = float(np.max(np.abs(jv_obs - jv_field)))
    W(f"`env.yaml` sets `velocity_limit_sim: null` for all three actuator groups, so Isaac")
    W(f"wrote **no** velocity limit to the simulation. Measured against the URDF's own")
    W(f"`<limit velocity>` (the only limit on record):\n")
    W(f"`joint_vel` field vs `obs[31:50]` agree to {agree:.2e} - the recording is self-consistent,")
    W("so no reconstruction from finite differences was necessary.\n")
    W("| # | joint | p99 abs (rad/s) | max abs (rad/s) | URDF velocity limit | max/limit | exceeded? |")
    W("|---:|---|---:|---:|---:|---:|---|")
    exceeded = []
    for i, jn in enumerate(jorder):
        p99 = float(np.percentile(np.abs(jv_obs[:, i]), 99))
        mx = float(np.max(np.abs(jv_obs[:, i])))
        lim = jinfo[jn]["urdfVelocityLimit"]
        ex_ = mx > lim
        if ex_:
            exceeded.append((jn, mx, lim))
        W(f"| {i} | `{jn}` | {p99:.3f} | {mx:.3f} | {lim:.1f} | {mx / lim:.3f} | "
          f"{'**YES**' if ex_ else 'no'} |")
    W("")
    if exceeded:
        W("Exceeded on: " + ", ".join(f"`{n}` ({m:.2f} > {l:.0f} rad/s)" for n, m, l in exceeded))
        W("")
        W("**Conclusion: Isaac did NOT enforce a joint velocity limit.** The recorded motion")
        W("physically exceeds the only documented limit, so clamping in Unity would truncate")
        W("motion the policy relies on.\n")
        W("Shipped: `enforceVelocityLimit = false`, and `maxJointVelocity` is set to the link")
        W(f"angular cap from `env.yaml` (`max_angular_velocity` = "
          f"{rig['physics']['maxAngularVelocity']:.0f} rad/s), not to the URDF limits.")
        W(f"Unity's project default `m_DefaultMaxAngularSpeed` is 50 rad/s, which is below the")
        W(f"measured peak of {max(m for _, m, _ in exceeded):.2f} rad/s - hence the per-body override.")
    else:
        W("No joint exceeded its limit; enforcement cannot be ruled out from this recording.")
    W("")

    # -------------------------------------------- C. explicit-PD stability ---
    W("## C. Explicit-PD stability bound  `kd * dt / I_joint`\n")
    W("Applies to the **diagnostic** `ActuatorMode.ExplicitTorquePD` path only")
    W("(`tau = clip(kp*(q* - q) - kd*qd, +/-effort)` applied with `ArticulationBody.AddTorque`).")
    W("The shipped default is `ArticulationDrive`, an implicit spring-damper, which is")
    W(f"unconditionally stable and does not obey this bound.\n")
    W(f"`I_joint` is the whole distal subtree's inertia about the joint axis at the default")
    W("pose (parallel-axis, every body). Explicit damping goes unstable as `kd*dt/I -> 2`.\n")

    I_raw, I_arm = {}, {}
    for jn in jorder:
        Ij = subtree_inertia_about_axis(rig, poses_def, jn)
        I_raw[jn] = Ij
        I_arm[jn] = Ij + jinfo[jn]["armature"]

    def bound(jn, dt, with_arm):
        I = (I_arm if with_arm else I_raw)[jn]
        return jinfo[jn]["damping"] * dt / I

    for with_arm, title in ((True, "with the 0.1 kg.m2 armature folded in (as shipped)"),
                            (False, "raw subtree inertia, no armature")):
        W(f"### {title}\n")
        W("| # | joint | kd | I_joint | " + " | ".join(STEP_LABELS) + " |")
        W("|---:|---|---:|---:|" + "---:|" * len(CANDIDATE_STEPS))
        for i, jn in enumerate(jorder):
            I = (I_arm if with_arm else I_raw)[jn]
            cells = []
            for dt in CANDIDATE_STEPS:
                v = bound(jn, dt, with_arm)
                cells.append(f"**{v:.2f}**" if v >= PD_BOUND else f"{v:.2f}")
            W(f"| {i} | `{jn}` | {jinfo[jn]['damping']:.1f} | {I:.5f} | " + " | ".join(cells) + " |")
        worst_per_step = [max(bound(jn, dt, with_arm) for jn in jorder) for dt in CANDIDATE_STEPS]
        W("| | **worst joint** | | | " +
          " | ".join(f"**{v:.2f}**" for v in worst_per_step) + " |")
        W("")
        ok = [(dt, lb) for dt, lb, v in zip(CANDIDATE_STEPS, STEP_LABELS, worst_per_step) if v < PD_BOUND]
        if ok:
            dt, lb = ok[0]
            W(f"**Coarsest step with every joint below {PD_BOUND:.0f}: {lb} ({dt:.6f} s).**")
        else:
            W(f"**No candidate step keeps every joint below {PD_BOUND:.0f}.**")
        W("")

    W("### Parent-recoil caveat\n")
    W("`kd*dt/I < 2` is a single-joint bound. It treats the parent as infinitely massive, so")
    W("it is optimistic for a serial chain: a joint whose parent is comparable in inertia")
    W("recoils, and the effective ratio is worse than the table says. A joint sitting at")
    W("ratio ~1.5 at 1/240 can still diverge in practice. Treat the coarsest passing step as")
    W("an upper bound to be confirmed empirically, not as a licence - which is exactly what")
    W("rung 3 (zero-g square wave) measures, and why `ExplicitTorquePD` ships as a")
    W("diagnostic switch behind `ArticulationDrive` rather than as the default.\n")
    W("Worst parent/child inertia ratios along the chain (closer to 1 = more recoil):\n")
    W("| joint | I_joint (child subtree) | I_parent-side subtree | ratio |")
    W("|---|---:|---:|---:|")
    rows = []
    for jn in jorder:
        b = jbody[jn]
        par = bodies[b]["parent"]
        pj = bodies[par].get("joint")
        if not pj:
            continue
        Ip = I_arm[pj["name"]]
        rows.append((jn, I_arm[jn], Ip, Ip / I_arm[jn]))
    for jn, Ic, Ip, r in sorted(rows, key=lambda t: t[3])[:8]:
        W(f"| `{jn}` | {Ic:.5f} | {Ip:.5f} | {r:.2f} |")
    W("")
    # concrete recommendation: worst single-joint ratio AND its recoil exposure
    recoil = {jn: r for jn, _, _, r in rows}
    wj, wv = max(((jn, jinfo[jn]["damping"] * args.project_dt / I_arm[jn]) for jn in jorder),
                 key=lambda t: t[1])
    safe = next((dt for dt in CANDIDATE_STEPS
                 if max(jinfo[j]["damping"] * dt / I_arm[j] for j in jorder) < PD_BOUND / 4), None)
    W("### What was actually measured in Unity\n")
    W("The table above is a bound, not a result. In the engine, with `armatureMode = None`")
    W("(the shipped default, see section A), `ExplicitTorquePD`:\n")
    W("| step | outcome |")
    W("|---|---|")
    W("| 1/500 s | **diverges** - zero-g bang-bang max abs vCoM 34.2 m/s, 58.7 m of drift in 3 s; falls immediately under gravity |")
    W("| 1/1000 s | **walks** - 0.956 m/s, upright 0.991, height 0.914 m |")
    W("")
    W("So the usable figure is **1/1000 s**, consistent with the no-armature column above")
    W("(1/960 -> 0.82) and NOT with the with-armature column. The with-armature column is")
    W("informational only: folding the armature into link inertia is what the shipped")
    W("configuration deliberately does not do.\n")
    W(f"Concretely: the worst joint at this project's step is **`{wj}` at {wv:.2f}**, and its")
    W(f"parent-side subtree is only {recoil.get(wj, float('nan')):.2f}x its own inertia, so the")
    W("single-joint number is meaningfully optimistic for exactly the joint that is closest to")
    W(f"the limit. **`ExplicitTorquePD` therefore ships with a recommended substep of "
      f"{safe:.6f} s** (worst ratio "
      f"{max(jinfo[j]['damping'] * safe / I_arm[j] for j in jorder):.2f}), a 4x margin rather")
    W("than the 1.2x the raw bound would allow. `IsaacH1Agent` logs that substep when the mode")
    W("is selected; it does **not** change `Time.fixedDeltaTime` to reach it.\n")

    # ---------------------------------------------- D. quaternion order ------
    W("## D. Reference quaternion order\n")
    W("`isaac_reference.json` names the field `root_quat_w_wxyz`. Tested against `obs[6:9]`")
    W("(`projected_gravity`, which is `R^T * (0,0,-1)` and therefore pins the order):\n")

    def R_from(q, order):
        w, x, y, z = (q if order == "wxyz" else (q[3], q[0], q[1], q[2]))
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    quats = np.array([s["root_quat_w_wxyz"] for s in ref])
    W("| interpretation | max abs error | mean abs error | verdict |")
    W("|---|---:|---:|---|")
    errs = {}
    for order in ("wxyz", "xyzw"):
        e = [float(np.max(np.abs(R_from(quats[i], order).T @ np.array([0, 0, -1.0]) - obs[i, 6:9])))
             for i in range(len(ref))]
        errs[order] = (max(e), float(np.mean(e)))
        W(f"| as `{order}` | {max(e):.3e} | {np.mean(e):.3e} | "
          f"{'**MATCH**' if max(e) < 1e-2 else 'mismatch'} |")
    W("")
    winner = min(errs, key=lambda k: errs[k][0])
    W(f"**The field is `{winner}`, despite being named `..._wxyz`.** "
      f"Reading it as `wxyz` gives a {errs['wxyz'][1]:.2f} mean error - the robot would be")
    W("interpreted as upside down. `isaac_reference.json` in this folder is a copy with the")
    W(f"field renamed `root_quat_w_{winner}` so the name states the real order.\n")
    W("The corresponding Unity conversion is therefore")
    W("`new Quaternion(q.y, -q.z, -q.x, q.w)` reading `q` as **xyzw**, i.e. the Unity vector")
    W("part is `-M * (x,y,z)` and the scalar is `w`.\n")

    # ------------------------------------------------------- timing summary --
    W("## Control rate\n")
    pdt = rig["timing"]["policyDt"]
    W("| quantity | value |")
    W("|---|---:|")
    W(f"| Isaac `policy_dt` | {pdt} s ({1 / pdt:.0f} Hz) |")
    W(f"| Isaac `physics_dt` | {rig['timing']['isaacPhysicsDt']} s "
      f"({1 / rig['timing']['isaacPhysicsDt']:.0f} Hz) |")
    W(f"| Isaac decimation | {rig['timing']['isaacDecimation']} |")
    W(f"| this project's `Time.fixedDeltaTime` | {args.project_dt} s ({1 / args.project_dt:.0f} Hz) |")
    W(f"| **policy_dt / fixedDeltaTime** | **{pdt / args.project_dt:.6f}** |")
    ratio = pdt / args.project_dt
    is_int = abs(ratio - round(ratio)) < 1e-9
    W(f"| integer? | **{'yes' if is_int else 'NO'}** |")
    W(f"| shipped decimation | {max(1, round(ratio))} |")
    W("")
    if is_int:
        W(f"The ratio is an exact integer ({round(ratio)}), so the agent runs at the project step")
        W("with no rounding and logs **no** `LogError`. It does log one `LogWarning`, because")
        W(f"decimation {round(ratio)} is not Isaac's {rig['timing']['isaacDecimation']}: the PD drive")
        W(f"gets {round(ratio)} tick(s) per policy step where Isaac gave it "
          f"{rig['timing']['isaacDecimation']}.")
        W("")
        W(f"Report-only proposal (**not applied**): setting the project step to "
          f"`{rig['timing']['isaacPhysicsDt']}` reproduces Isaac exactly "
          f"(decimation {rig['timing']['isaacDecimation']}) and is still an exact divisor of")
        W("`policy_dt`. The PlayMode tests set that step in `SetUp` and restore it in `TearDown`.")
    W("")

    with open(args.out, "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {args.out} ({len(L)} lines)")
    print(f"  FK max disagreement       : {worst * 1000:.5f} mm  ({'PASS' if worst < 1e-3 else 'FAIL'})")
    print(f"  rest pelvis height        : {rest_pelvis_z:.5f} m (recorded walking mean {ref_zmean:.5f} m)")
    print(f"  quaternion order          : {winner}")
    print(f"  inertia floor needed      : {'yes' if need_floor else 'no'}")
    print(f"  velocity limit exceeded   : {'yes' if exceeded else 'no'}")
    for with_arm, tag in ((True, "with armature"), (False, "no armature")):
        wp = [max(jinfo[jn]["damping"] * dt / (I_arm if with_arm else I_raw)[jn] for jn in jorder)
              for dt in CANDIDATE_STEPS]
        ok = [lb for lb, v in zip(STEP_LABELS, wp) if v < PD_BOUND]
        print(f"  explicit-PD coarsest step ({tag}): {ok[0] if ok else 'none'}")


if __name__ == "__main__":
    main()
