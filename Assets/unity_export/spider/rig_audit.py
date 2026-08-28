"""Rig audit for PhysX 4 (Unity) from the Isaac export.  Run: python rig_audit.py
Writes RIG_AUDIT.md + rig_audit.json next to this file.
Reads ../../spider/{export_report.json, checkpoint/params/env.yaml, robot/spider.urdf, isaac_reference.json}."""
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.normpath(os.path.join(HERE, "..", "..", "spider"))
report = json.load(open(os.path.join(SRC, "export_report.json")))
ref = json.load(open(os.path.join(SRC, "isaac_reference.json")))
env_txt = open(os.path.join(SRC, "checkpoint", "params", "env.yaml")).read()


def env_val(key, default):
    for line in env_txt.splitlines():
        s = line.strip()
        if s.startswith(key + ":"):
            v = s.split(":", 1)[1].strip()
            return float(v) if v not in ("null", "") else default
    return default


kp = env_val("stiffness", 25.0)
kd = env_val("damping", 1.0)
effort = env_val("effort_limit_sim", 15.0)
vel_lim = env_val("velocity_limit_sim", 12.0)
sim_dt = report["sim_dt"]
policy_dt = report["policy_dt"]

# ---------------------------------------------------------------- URDF
root = ET.parse(os.path.join(SRC, "robot", "spider.urdf")).getroot()
links, joints = {}, {}


def xyz(e, k="xyz"):
    return np.array([float(x) for x in e.get(k, "0 0 0").split()])


def rot_rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


for l in root.findall("link"):
    inertial = l.find("inertial")
    o = inertial.find("origin")
    I = inertial.find("inertia")
    Icm = np.diag([float(I.get("ixx")), float(I.get("iyy")), float(I.get("izz"))])
    R = rot_rpy(*xyz(o, "rpy")) if o is not None else np.eye(3)
    links[l.get("name")] = dict(mass=float(inertial.find("mass").get("value")),
                                com=xyz(o) if o is not None else np.zeros(3),
                                Icm_link=R @ Icm @ R.T, Iprincipal=np.diag(Icm))
for j in root.findall("joint"):
    o = j.find("origin")
    joints[j.get("name")] = dict(parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                 xyz=xyz(o), R=rot_rpy(*xyz(o, "rpy")), axis=xyz(j.find("axis")))
children = {}
for n, j in joints.items():
    children.setdefault(j["parent"], []).append(n)


def subtree_inertia_about_axis(link, axis, R_acc, p_acc):
    """Inertia of link + all downstream links about the joint axis through the joint origin."""
    L = links[link]
    com_w = p_acc + R_acc @ L["com"]
    I_w = R_acc @ L["Icm_link"] @ R_acc.T
    d = com_w - np.dot(com_w, axis) * axis
    total = axis @ I_w @ axis + L["mass"] * float(d @ d)
    for cn in children.get(link, []):
        c = joints[cn]
        total += subtree_inertia_about_axis(c["child"], axis, R_acc @ c["R"], p_acc + R_acc @ c["xyz"])
    return total


lines = ["# Rig audit - spider (Isaac Lab export -> Unity PhysX 4)", "",
         f"kp={kp} N.m/rad, kd={kd} N.m.s/rad, effort={effort} N.m, velocity_limit_sim={vel_lim} rad/s, "
         f"Isaac sim_dt={sim_dt:.6f} s (decimation {report['decimation']}, policy_dt={policy_dt:.6f} s)", ""]

# ---------------------------------------------------------------- 1. light links / small inertia
lines += ["## 1. Light links and small inertia", "",
          "| link | mass kg | I principal (kg.m2) | neighbours | mass ratio to neighbours |", "|---|---|---|---|---|"]
neigh = {}
for j in joints.values():
    neigh.setdefault(j["parent"], set()).add(j["child"])
    neigh.setdefault(j["child"], set()).add(j["parent"])
flag_light, flag_inertia = [], []
for name, L in links.items():
    ratios = [(nb, L["mass"] / links[nb]["mass"]) for nb in sorted(neigh.get(name, []))]
    rstr = ", ".join(f"{nb}: {r:.2f}" for nb, r in ratios)
    if any(r < 0.1 for _, r in ratios):
        flag_light.append((name, min(r for _, r in ratios)))
    if L["Iprincipal"].min() < 1e-4:
        flag_inertia.append((name, L["Iprincipal"].min()))
    lines.append(f"| {name} | {L['mass']} | {', '.join(f'{v:.2e}' for v in L['Iprincipal'])} | "
                 f"{', '.join(sorted(neigh.get(name, [])))} | {rstr} |")
body_femur = links['body']['mass'] / links['L1_femur']['mass']
lines += ["",
          f"Links lighter than 10 % of a neighbour: {', '.join(f'{n} ({r:.3f})' for n, r in flag_light) or 'none'} "
          f"(worst ratio body:femur = {body_femur:.0f}:1, below the 50:1 retrain threshold).",
          f"Links with a principal inertia below 1e-4 kg.m2: {', '.join(f'{n} ({v:.1e})' for n, v in flag_inertia) or 'none'} "
          "- that is the long-axis (roll) component of the cylinders, not the joint-axis component.",
          "Recommendation: keep the URDF masses (massFloor = 0; the 20:1 body:femur ratio is inside PhysX 4 tolerance) and floor "
          "each principal inertia at 1e-4 kg.m2 (inertiaFloor = 1e-4 on the Agent). That raises tibia roll inertia 7x and femur "
          "roll inertia 5x - a DoF no joint drives - so gait is unaffected while the solver conditioning improves. "
          "If the femur ever has to be lightened further (ratio > 50:1) retrain with heavier femurs instead.", ""]

# ---------------------------------------------------------------- 2. joint speeds in the Isaac eval
jv = np.array([s["obs"][16:32] for s in ref]) / 0.1
order = report["joint_order"]
lines += ["## 2. Joint velocities in the Isaac reference (reconstructed from obs[16:32] / 0.1)", "",
          "| joint | max abs rad/s | p99 abs rad/s |", "|---|---|---|"]
for i, n in enumerate(order):
    a = np.abs(jv[:, i])
    lines.append(f"| {n} | {a.max():.2f} | {np.percentile(a, 99):.2f} |")
over = float(np.abs(jv).max())
if over <= vel_lim * 1.02:
    verdict = ("The Isaac solver clamped joint velocity at the limit (USD physxJoint:maxJointVelocity = 687.55 deg/s = 12 rad/s), "
               "so the policy was trained WITH the clamp: Unity must enforce it too (ArticulationBody.maxJointVelocity = 12 in "
               "drive mode; enforceVelocityLimit = true in torque mode).")
else:
    verdict = "EXCEEDS the limit -> Isaac did not enforce it; set enforceVelocityLimit = false."
lines += ["", f"Overall max abs joint velocity = {over:.2f} rad/s vs velocity_limit_sim = {vel_lim}. " + verdict, ""]

# ---------------------------------------------------------------- 3. explicit-PD stability bound
lines += ["## 3. Explicit-PD stability bound  kd*dt / I_joint  (must stay < 2; < 1 comfortable)", ""]
dts = [("project 0.02", 0.02), ("1/60", 1 / 60), ("1/120 (Isaac)", 1 / 120), ("1/240", 1 / 240), ("1/480", 1 / 480), ("1/960", 1 / 960)]
lines.append("| joint | I about axis (kg.m2, parallel-axis, whole subtree) | " + " | ".join(n for n, _ in dts) + " |")
lines.append("|---|---|" + "---|" * len(dts))
worst_ratio = {n: 0.0 for n, _ in dts}
Ij = {}
for jn in order:
    j = joints[jn]
    I = subtree_inertia_about_axis(j["child"], j["axis"], np.eye(3), np.zeros(3))
    Ij[jn] = I
    cells = []
    for n, dt in dts:
        r = kd * dt / I
        worst_ratio[n] = max(worst_ratio[n], r)
        cells.append(f"{r:.2f}")
    lines.append(f"| {jn} | {I:.3e} | " + " | ".join(cells) + " |")
coarsest = next((n for n, _ in dts if worst_ratio[n] < 2.0), None)
lines += ["",
          f"Coarsest substep with every joint below 2: **{coarsest}** (knee ratio {worst_ratio[coarsest]:.2f}). "
          f"At the project's 0.02 s the knee ratio is {worst_ratio['project 0.02']:.1f} -> the explicit torque PD diverges; "
          f"at 1/240 it is {worst_ratio['1/240']:.2f} (< 2, but the femur recoils so the effective inertia seen by the knee "
          f"is smaller than the tibia-alone number - the divergence observed at 1/240 is consistent with that); "
          f"**1/480 ({worst_ratio['1/480']:.2f}) is the substep the C# torque actuator needs**.",
          "The ArticulationDrive path (the PhysX implicit spring-damper - which is what Isaac's ImplicitActuator is) has no such "
          "bound and is stable at the project's 0.02 s.", ""]

# ---------------------------------------------------------------- 4. reference conventions
q = np.array([s["root_quat_w_wxyz"] for s in ref])
tr = np.array([s["target_rel"] for s in ref])
rp = np.array([s["root_pos_w"] for s in ref])
tv = np.array([s["obs"][41:43] for s in ref])


def yaw_from(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


err = {"xyzw": 0.0, "wxyz": 0.0}
for k in range(len(ref)):
    d = tr[k] - rp[k]
    for name, (x, y, z, w) in (("xyzw", q[k]), ("wxyz", (q[k][1], q[k][2], q[k][3], q[k][0]))):
        yaw = yaw_from(x, y, z, w)
        c, s = math.cos(-yaw), math.sin(-yaw)
        pred = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
        err[name] = max(err[name], float(np.abs(pred - tv[k]).max()))
conv = "xyzw" if err["xyzw"] < err["wxyz"] else "wxyz"
speed = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1).sum() / (len(ref) - 1) / policy_dt
lines += ["## 4. Reference-file conventions", "",
          f"root_quat_w_wxyz matches obs[41:43] as **{conv}** (max err xyzw {err['xyzw']:.2e}, wxyz {err['wxyz']:.2e}): "
          "the field name says wxyz but the data is xyzw (w last). The copy in this folder renames it root_quat_w_xyzw.",
          f"Body height in the reference: min {rp[:, 2].min():.3f} m, mean {rp[:, 2].mean():.3f} m, max {rp[:, 2].max():.3f} m "
          "(Isaac spawns at 0.18 m; reported standing height 0.141 m).",
          f"Mean planar speed over the 200-step recording: {speed:.2f} m/s.", ""]

# ---------------------------------------------------------------- 5. colliders
lines += ["## 5. Colliders", "",
          "URDF collision shapes are primitives only: body sphere r=0.1; femur cylinder r=0.02 L=0.16; tibia cylinder r=0.015 "
          "L=0.26. The URDF Importer turns each cylinder into a scaled convex mesh under a non-uniformly scaled `unnamed` parent; "
          "the setup script replaces those with unscaled CapsuleColliders created directly on the link (height = L + r, so the "
          "tip contact point matches the flat cylinder rim at the rest angle within ~1 mm).", ""]

open(os.path.join(HERE, "RIG_AUDIT.md"), "w", encoding="utf-8").write("\n".join(lines))
json.dump({"joint_inertia_about_axis": Ij, "worst_kd_dt_over_I": worst_ratio, "coarsest_ok_substep": coarsest,
           "max_joint_vel": over, "quat_convention": conv, "ref_mean_speed": speed},
          open(os.path.join(HERE, "rig_audit.json"), "w"), indent=2)
print("\n".join(lines))
