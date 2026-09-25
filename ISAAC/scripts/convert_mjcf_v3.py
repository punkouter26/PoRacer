#!/usr/bin/env python3
"""Kit-less MJCF -> USD for Isaac Lab 3 / Newton (MuJoCo-Warp solver), for ANY simple MJCF creature.

    ISAAC\\isaaclab3\\.venv\\Scripts\\python.exe ISAAC\\scripts\\convert_mjcf_v3.py --mjcf training\\quad\\quad.xml ^
        --out ISAAC\\quad_tasks_v3\\assets\\quad_v3.usda

The generalised form of convert_worm_v3.py (which stays as the worm's converter of record). Isaac Lab 3's
MjcfConverter needs Kit (Isaac Sim's importer extension); this install is kit-less, so the import is done
here from the COMPILED MuJoCo model (``mujoco.MjModel.from_xml_path``): masses, inertias, frames, axes,
ranges, solref/solimp, friction and contact exclusions are read exactly as MuJoCo simulates them.

Supported (anything else aborts with a clear message rather than being silently dropped):
  * one kinematic tree under the world, a free joint on the root body, hinge joints elsewhere
    (any axis; bodies may be rotated), at most one joint per body;
  * geoms: box, capsule, sphere, cylinder (the world's plane is skipped: the task supplies the ground);
  * <contact><exclude> pairs and MuJoCo's parent-child filter (filterparent, on by default);
  * position actuators (kp, kv, forcerange) and passive joint damping / armature / frictionloss.

What is written, and which Newton default each value pins (the worm port found that several Newton
defaults silently differ from MuJoCo's):
  * RigidBodyAPI + MassAPI per body: mass, centre of mass, principal inertia, principal axes (from MuJoCo).
  * colliders: UsdGeom Capsule/Cylinder (axis Z, the MuJoCo geom frame), Cube (size 2, scaled to the
    half-sizes), Sphere, posed by geom_pos/geom_quat inside the body. Per collider:
      newton:contact_ke/kd  = geom solref as Newton gains (SolverMuJoCo.convert_solref inverts them
                              exactly; Newton's own default is a different, softer contact)
      newton:contactMargin  = geom margin (0)
      mjc:condim            = geom condim
  * physics materials (one per distinct friction triple): static = dynamic = sliding friction,
    restitution 0, newton:torsionalFriction / newton:rollingFriction = MuJoCo's 2nd/3rd friction.
  * RevoluteJoint per hinge (limits in degrees) with:
      mjc:damping            = dof_damping (PASSIVE joint damping, not capped by the force limit)
      newton:armature        = dof_armature
      newton:friction        = dof_frictionloss
      newton:angular:limitStiffness / limitDamping = the joint's solreflimit as Newton limit gains,
                              per DEGREE (Newton divides by pi/180). SolverMuJoCo writes them back as the
                              direct-form solref (-ke, -kd), which gives MuJoCo the same K and B as the
                              time-constant form (see limit_gains()). Newton's default (1e4/10 per rad)
                              would be a different limit; 0 would fall back to MuJoCo's (0.02, 1), which is
                              only right when the MJCF itself uses the default.
      DriveAPI angular (force): stiffness kp, damping kv (per degree), maxForce = the actuator force
                              range. The task's ImplicitActuatorCfg sets these again at runtime.
  * FilteredPairsAPI on the collider prims for <exclude> pairs (Newton reads shape-level filters only).
    Parent-child pairs are filtered by Newton's joint builder and again by MuJoCo's filterparent.
  * articulation root on the asset root Xform, self-collision enabled.

USD prim names must be identifiers, so MJCF names are sanitised ('-' -> 'm', any other invalid character
-> '_'; "Upper_-1_1" -> "Upper_m1_1"). The mapping is written next to the USD as <out>.names.json and
the task code uses it, so nothing downstream re-derives names.

After writing, the USD is re-opened and checked against the compiled MJCF (bodies, masses, inertias,
colliders, joints, limits, damping, armature, limit gains, filters); exits non-zero on any mismatch.
The runtime (Newton-built mjModel) checks live in check_quad_physics_v3.py.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

DEG = math.pi / 180.0


def sanitize(name: str) -> str:
    s = name.replace("-", "m")
    s = re.sub(r"[^A-Za-z0-9_]", "_", s)
    return s if re.match(r"[A-Za-z_]", s) else "_" + s


def quat_gf(q) -> Gf.Quatf:
    """MuJoCo (w, x, y, z) -> Gf.Quatf."""
    return Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3])))


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return out


def rot_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, float))
    return m.reshape(3, 3)


def quat_x_to(axis) -> np.ndarray:
    """Unit quaternion (w, x, y, z) rotating +X onto ``axis``."""
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    x = np.array([1.0, 0.0, 0.0])
    c = float(np.dot(x, a))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])  # 180 deg about z
    v = np.cross(x, a)
    q = np.array([1.0 + c, *v])
    return q / np.linalg.norm(q)


AXIS_TOKENS = {(1, 0, 0): "X", (0, 1, 0): "Y", (0, 0, 1): "Z"}


def limit_gains(model: mujoco.MjModel, jid: int) -> tuple[float, float]:
    """(ke, kd) per RADIAN such that MuJoCo's direct-form solref (-ke, -kd) equals the joint's solreflimit.

    MuJoCo (engine_core_constraint.c): positive solref (timeconst, dampratio) ->
        K = 1 / (dmax^2 timeconst^2 dampratio^2),  B = 2 / (dmax timeconst),
    with timeconst raised to 2*timestep unless mjDSBL_REFSAFE; negative solref (-ke, -kd) ->
        K = ke / dmax^2,  B = kd / dmax.
    So ke = 1 / (timeconst^2 dampratio^2) and kd = 2 / timeconst give identical K and B for every dmax.
    """
    tc, dr = (float(v) for v in model.jnt_solref[jid])
    if tc <= 0.0:
        return -tc, -dr
    if not (model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_REFSAFE):
        tc = max(tc, 2.0 * float(model.opt.timestep))
    return 1.0 / (tc * tc * dr * dr), 2.0 / tc


def geom_gains(model: mujoco.MjModel, gid: int) -> tuple[float, float]:
    """Newton contact (ke, kd) that SolverMuJoCo.convert_solref(ke, kd, 1, 1) maps back to geom_solref."""
    tc, dr = (float(v) for v in model.geom_solref[gid])
    if tc <= 0.0:
        raise SystemExit(f"geom {model.geom(gid).name}: direct-form geom solref is not supported")
    return 1.0 / (tc * tc * dr * dr), 2.0 / tc


def excluded_body_pairs(m: mujoco.MjModel) -> list[tuple[int, int]]:
    return [(int(s >> 16), int(s & 0xFFFF)) for s in m.exclude_signature]


def collidable_body_pairs(m: mujoco.MjModel) -> set[tuple[str, str]]:
    """Body pairs (by name, sorted) MuJoCo's broadphase filter lets through (mj_collision's rules).

    Same body -> no; contype/conaffinity mismatch -> no; <exclude> -> no; parent-child of the weld tree
    with a non-world parent -> no (filterparent, unless disabled). World geoms are left out.
    """
    excl = {tuple(sorted(p)) for p in excluded_body_pairs(m)}
    filterparent = not (m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    out = set()
    for g1 in range(m.ngeom):
        for g2 in range(g1 + 1, m.ngeom):
            b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
            if b1 == 0 or b2 == 0 or b1 == b2:
                continue
            if not ((m.geom_contype[g1] & m.geom_conaffinity[g2]) or (m.geom_contype[g2] & m.geom_conaffinity[g1])):
                continue
            if tuple(sorted((b1, b2))) in excl:
                continue
            w1, w2 = int(m.body_weldid[b1]), int(m.body_weldid[b2])
            if filterparent and w1 != 0 and w2 != 0:
                p1, p2 = int(m.body_weldid[m.body_parentid[w1]]), int(m.body_weldid[m.body_parentid[w2]])
                if (p1 == w2 and p1 != 0) or (p2 == w1 and p2 != 0):
                    continue
            out.add(tuple(sorted((m.body(b1).name, m.body(b2).name))))
    return out


def position_actuators(m: mujoco.MjModel) -> dict[int, dict]:
    """joint id -> {kp, kv, force range} for the model's position actuators."""
    out = {}
    for a in range(m.nu):
        if m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            raise SystemExit(f"actuator {m.actuator(a).name}: only joint transmissions are supported")
        kp = float(m.actuator_gainprm[a][0])
        bias = m.actuator_biasprm[a]
        if not (abs(bias[0]) < 1e-12 and abs(bias[1] + kp) < 1e-9 and m.actuator_gear[a][0] == 1.0):
            raise SystemExit(f"actuator {m.actuator(a).name}: only unit-gear <position> servos are supported")
        jid = int(m.actuator_trnid[a][0])
        if jid in out:
            raise SystemExit(f"joint {m.joint(jid).name} has more than one actuator")
        if m.actuator_forcelimited[a]:
            lo, hi = (float(v) for v in m.actuator_forcerange[a])
        elif m.jnt_actfrclimited[jid]:
            lo, hi = (float(v) for v in m.jnt_actfrcrange[jid])
        else:
            lo, hi = -1e9, 1e9
        if abs(lo + hi) > 1e-9:
            raise SystemExit(f"actuator {m.actuator(a).name}: asymmetric force range {lo, hi}")
        out[jid] = {"actuator": m.actuator(a).name, "kp": kp, "kv": float(-bias[2]), "force": hi}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Kit-less MJCF -> USD for Isaac Lab 3 / Newton (MuJoCo-Warp).")
    ap.add_argument("--mjcf", required=True)
    ap.add_argument("--out", required=True, help=".usda path; <out>.names.json is written next to it")
    ap.add_argument("--root", default=None, help="USD root prim name (default: the MJCF model name)")
    ap.add_argument("--color", default="0.3 0.8 0.35", help="display colour of the colliders")
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(os.path.abspath(args.mjcf))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # world poses at qpos0

    # ---------------------------------------------------------------- supported-feature gate --
    for feat, n in (("equality constraints", m.neq), ("tendons", m.ntendon), ("explicit <pair> contacts", m.npair)):
        if n:
            raise SystemExit(f"{args.mjcf}: {feat} are not supported by this converter")
    roots = [b for b in range(1, m.nbody) if m.body_parentid[b] == 0]
    if len(roots) != 1:
        raise SystemExit(f"expected one root body under the world, found {[m.body(b).name for b in roots]}")
    root_body = roots[0]
    for b in range(1, m.nbody):
        nj = int(m.body_jntnum[b])
        if b == root_body:
            if nj != 1 or m.jnt_type[m.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_FREE:
                raise SystemExit(f"root body {m.body(b).name} must have exactly one free joint")
        elif nj != 1 or m.jnt_type[m.body_jntadr[b]] != mujoco.mjtJoint.mjJNT_HINGE:
            raise SystemExit(f"body {m.body(b).name}: exactly one hinge joint per non-root body is supported")
    for gid in range(m.ngeom):
        if m.geom_bodyid[gid] == 0:
            if m.geom_type[gid] != mujoco.mjtGeom.mjGEOM_PLANE:
                raise SystemExit(f"world geom {m.geom(gid).name}: only the ground plane may live in the world")
            continue
        if abs(m.geom_gap[gid]) > 0.0:
            raise SystemExit(f"geom {m.geom(gid).name}: gap is not supported")
        if not np.allclose(m.geom_solimp[gid], [0.9, 0.95, 0.001, 0.5, 2.0]):
            raise SystemExit(f"geom {m.geom(gid).name}: non-default solimp is not written by this converter")
    hinges = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    for j in hinges:
        if not np.allclose(m.jnt_solimp[j], [0.9, 0.95, 0.001, 0.5, 2.0]) or m.jnt_margin[j] != 0.0:
            raise SystemExit(f"joint {m.joint(j).name}: non-default solimplimit / margin is not supported")
        if not m.jnt_limited[j]:
            raise SystemExit(f"joint {m.joint(j).name}: unlimited hinges are not supported")
        if m.qpos0[m.jnt_qposadr[j]] != 0.0 or m.jnt_stiffness[j] != 0.0:
            raise SystemExit(f"joint {m.joint(j).name}: ref / springs are not supported")
    acts = position_actuators(m)

    names = {"bodies": {}, "joints": {}, "geoms": {}}
    for b in range(1, m.nbody):
        names["bodies"][m.body(b).name] = sanitize(m.body(b).name)
    for j in hinges:
        names["joints"][m.joint(j).name] = sanitize(m.joint(j).name)
    for kind, table in names.items():
        if len(set(table.values())) != len(table):
            raise SystemExit(f"sanitised {kind} names collide: {table}")

    # ------------------------------------------------------------------------------- stage --
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)  # always regenerated from the MJCF
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root_name = sanitize(args.root or m.names[: m.names.index(b"\0")].decode() or "robot")
    R = f"/{root_name}"
    root = UsdGeom.Xform.Define(stage, R)
    stage.SetDefaultPrim(root.GetPrim())
    rp = root.GetPrim()
    UsdPhysics.ArticulationRootAPI.Apply(rp)
    rp.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    rp.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)
    color = Gf.Vec3f(*(float(v) for v in args.color.split()))

    materials = {}

    def material_for(fr) -> UsdShade.Material:
        key = tuple(round(float(v), 9) for v in fr)
        if key not in materials:
            mat = UsdShade.Material.Define(stage, f"{R}/Looks/physicsMaterial_{len(materials)}")
            api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
            api.CreateStaticFrictionAttr(key[0])
            api.CreateDynamicFrictionAttr(key[0])
            api.CreateRestitutionAttr(0.0)
            mat.GetPrim().CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(key[1])
            mat.GetPrim().CreateAttribute("newton:rollingFriction", Sdf.ValueTypeNames.Float).Set(key[2])
            materials[key] = mat
        return materials[key]

    body_prims, geom_prims = {}, {}
    for b in range(1, m.nbody):
        bname = names["bodies"][m.body(b).name]
        xf = UsdGeom.Xform.Define(stage, f"{R}/{bname}")
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, d.xpos[b])))
        xf.AddOrientOp().Set(quat_gf(d.xquat[b]))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(m.body_mass[b]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, m.body_ipos[b])))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, m.body_inertia[b])))
        mass.CreatePrincipalAxesAttr(quat_gf(m.body_iquat[b]))
        body_prims[m.body(b).name] = prim

        for gid in (g for g in range(m.ngeom) if m.geom_bodyid[g] == b):
            gname = sanitize(m.geom(gid).name or f"geom{gid}")
            names["geoms"][m.geom(gid).name or f"geom{gid}"] = f"{bname}/{gname}"
            path = f"{R}/{bname}/{gname}"
            t, size = int(m.geom_type[gid]), m.geom_size[gid]
            if t == mujoco.mjtGeom.mjGEOM_CAPSULE:
                g = UsdGeom.Capsule.Define(stage, path)
                g.CreateAxisAttr(UsdGeom.Tokens.z)
                g.CreateRadiusAttr(float(size[0]))
                g.CreateHeightAttr(float(2.0 * size[1]))
            elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
                g = UsdGeom.Cylinder.Define(stage, path)
                g.CreateAxisAttr(UsdGeom.Tokens.z)
                g.CreateRadiusAttr(float(size[0]))
                g.CreateHeightAttr(float(2.0 * size[1]))
            elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
                g = UsdGeom.Sphere.Define(stage, path)
                g.CreateRadiusAttr(float(size[0]))
            elif t == mujoco.mjtGeom.mjGEOM_BOX:
                g = UsdGeom.Cube.Define(stage, path)
                g.CreateSizeAttr(2.0)  # extent -1..1, scaled to the half-sizes below
            else:
                raise SystemExit(f"geom {m.geom(gid).name}: type {t} is not supported")
            g.AddTranslateOp().Set(Gf.Vec3d(*map(float, m.geom_pos[gid])))
            g.AddOrientOp().Set(quat_gf(m.geom_quat[gid]))
            if t == mujoco.mjtGeom.mjGEOM_BOX:
                g.AddScaleOp().Set(Gf.Vec3f(*map(float, size)))
            gp = g.GetPrim()
            UsdPhysics.CollisionAPI.Apply(gp)
            ke, kd = geom_gains(m, gid)
            gp.CreateAttribute("newton:contact_ke", Sdf.ValueTypeNames.Float).Set(ke)
            gp.CreateAttribute("newton:contact_kd", Sdf.ValueTypeNames.Float).Set(kd)
            gp.CreateAttribute("newton:contactMargin", Sdf.ValueTypeNames.Float).Set(float(m.geom_margin[gid]))
            gp.CreateAttribute("mjc:condim", Sdf.ValueTypeNames.Int).Set(int(m.geom_condim[gid]))
            if not (m.geom_contype[gid] or m.geom_conaffinity[gid]):
                UsdPhysics.CollisionAPI(gp).CreateCollisionEnabledAttr(False)
            UsdShade.MaterialBindingAPI.Apply(gp).Bind(
                material_for(m.geom_friction[gid]), bindingStrength=UsdShade.Tokens.weakerThanDescendants,
                materialPurpose="physics")
            g.CreateDisplayColorAttr([color])
            geom_prims[gid] = gp

    # ------------------------------------------------------------------------------ joints --
    UsdGeom.Scope.Define(stage, f"{R}/joints")
    joint_info = []
    for j in hinges:
        child = int(m.jnt_bodyid[j])
        parent = int(m.body_parentid[child])
        jn = names["joints"][m.joint(j).name]
        anchor_c = np.array(m.jnt_pos[j], float)
        axis_c = np.array(m.jnt_axis[j], float)
        rot_pc = rot_mat(m.body_quat[child])  # child frame expressed in the parent frame (qpos0)
        anchor_p = np.array(m.body_pos[child], float) + rot_pc @ anchor_c
        # a positive coordinate axis keeps the plain token (identity joint frames, as the worm);
        # anything else gets joint frames whose +X is the axis
        tok = AXIS_TOKENS.get(tuple(int(v) for v in np.round(axis_c))) if np.allclose(axis_c, np.round(axis_c)) else None
        if tok is not None:
            q1 = np.array([1.0, 0.0, 0.0, 0.0])
        else:  # general axis: joint frame X along the axis on both sides
            tok, q1 = "X", quat_x_to(axis_c)
        q0 = quat_mul(m.body_quat[child], q1)
        rj = UsdPhysics.RevoluteJoint.Define(stage, f"{R}/joints/{jn}")
        rj.CreateBody0Rel().SetTargets([body_prims[m.body(parent).name].GetPath()])
        rj.CreateBody1Rel().SetTargets([body_prims[m.body(child).name].GetPath()])
        rj.CreateLocalPos0Attr(Gf.Vec3f(*map(float, anchor_p)))
        rj.CreateLocalRot0Attr(quat_gf(q0))
        rj.CreateLocalPos1Attr(Gf.Vec3f(*map(float, anchor_c)))
        rj.CreateLocalRot1Attr(quat_gf(q1))
        rj.CreateAxisAttr(tok)
        lo, hi = (math.degrees(float(v)) for v in m.jnt_range[j])
        rj.CreateLowerLimitAttr(lo)
        rj.CreateUpperLimitAttr(hi)
        jp = rj.GetPrim()
        dof = int(m.jnt_dofadr[j])
        jp.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Float).Set(float(m.dof_damping[dof]))
        jp.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(float(m.dof_armature[dof]))
        jp.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(float(m.dof_armature[dof]))
        jp.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(float(m.dof_frictionloss[dof]))
        lke, lkd = limit_gains(m, j)
        jp.CreateAttribute("newton:angular:limitStiffness", Sdf.ValueTypeNames.Float).Set(lke * DEG)
        jp.CreateAttribute("newton:angular:limitDamping", Sdf.ValueTypeNames.Float).Set(lkd * DEG)
        act = acts.get(j)
        if act is not None:
            drive = UsdPhysics.DriveAPI.Apply(jp, "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(act["kp"] * DEG)  # USD angular drive gains are per degree
            drive.CreateDampingAttr(act["kv"] * DEG)
            drive.CreateMaxForceAttr(act["force"])
            drive.CreateTargetPositionAttr(0.0)
        joint_info.append((j, jn, tok, q1, lke, lkd, act))

    # <exclude> -> FilteredPairsAPI between every collider of the two bodies (shape level: Newton's USD
    # importer ignores body-level filters).
    excl = excluded_body_pairs(m)
    for b1, b2 in excl:
        for g1 in (g for g in range(m.ngeom) if m.geom_bodyid[g] == b1):
            for g2 in (g for g in range(m.ngeom) if m.geom_bodyid[g] == b2):
                for s, t in ((g1, g2), (g2, g1)):
                    UsdPhysics.FilteredPairsAPI.Apply(geom_prims[s]).CreateFilteredPairsRel().AddTarget(
                        geom_prims[t].GetPath())
    stage.GetRootLayer().Save()

    sidecar = {
        "mjcf": os.path.relpath(os.path.abspath(args.mjcf), os.path.dirname(os.path.abspath(args.out))).replace("\\", "/"),
        "root": root_name,
        "bodies": names["bodies"],
        "joints": names["joints"],
        "geoms": names["geoms"],
        "actuatorOrder": [m.actuator(a).name for a in range(m.nu)],
        "actuatorJointsUsd": [names["joints"][m.joint(int(m.actuator_trnid[a][0])).name] for a in range(m.nu)],
    }
    with open(args.out + ".names.json", "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=1)
    print(f"[convert_mjcf_v3] wrote {args.out} (+ .names.json): {m.nbody - 1} bodies, {len(hinges)} hinges, "
          f"{sum(1 for g in range(m.ngeom) if m.geom_bodyid[g])} colliders, {len(excl)} <exclude> pairs, "
          f"{len(materials)} material(s)")
    return verify(args.out, m, names, joint_info, excl)


def verify(path, m, names, joint_info, excl) -> int:
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'OK ' if ok else 'FAIL'}] {label} {detail}")
        if not ok:
            fails.append(label)

    stage = Usd.Stage.Open(path)
    inv_b = {v: k for k, v in names["bodies"].items()}
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    check(f"{m.nbody - 1} rigid bodies", len(bodies) == m.nbody - 1, str([b.GetName() for b in bodies]))
    total = 0.0
    for b in bodies:
        mb = m.body(inv_b[b.GetName()])
        api = UsdPhysics.MassAPI(b)
        mm, ii = api.GetMassAttr().Get(), api.GetDiagonalInertiaAttr().Get()
        total += mm
        ok = abs(mm - mb.mass[0]) < 1e-5 and np.allclose(ii, mb.inertia, rtol=1e-6, atol=1e-9)
        ok &= np.allclose(api.GetCenterOfMassAttr().Get(), mb.ipos, atol=1e-7)
        check(f"{b.GetName():14s} mass/inertia/com == MuJoCo", ok, f"m={mm:.4f} I=({ii[0]:.5g}, {ii[1]:.5g}, {ii[2]:.5g})")
    check("total mass == MuJoCo", abs(total - float(m.body_mass[1:].sum())) < 1e-4, f"{total:.4f} kg")

    cols = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI)]
    n_body_geoms = sum(1 for g in range(m.ngeom) if m.geom_bodyid[g])
    check(f"{n_body_geoms} colliders", len(cols) == n_body_geoms)
    for gid in (g for g in range(m.ngeom) if m.geom_bodyid[g]):
        prim = stage.GetPrimAtPath(f"/{stage.GetDefaultPrim().GetName()}/{names['geoms'][m.geom(gid).name or f'geom{gid}']}")
        t, size = int(m.geom_type[gid]), m.geom_size[gid]
        if t == mujoco.mjtGeom.mjGEOM_BOX:
            s = prim.GetAttribute("xformOp:scale").Get()
            ok = prim.IsA(UsdGeom.Cube) and np.allclose(s, size, atol=1e-7) and UsdGeom.Cube(prim).GetSizeAttr().Get() == 2.0
            desc = f"box half {np.round(size, 4).tolist()}"
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            c = UsdGeom.Capsule(prim)
            ok = prim.IsA(UsdGeom.Capsule) and abs(c.GetRadiusAttr().Get() - size[0]) < 1e-7 and \
                abs(c.GetHeightAttr().Get() - 2 * size[1]) < 1e-7 and c.GetAxisAttr().Get() == "Z"
            desc = f"capsule r {size[0]:.4f} half {size[1]:.4f}"
        elif t == mujoco.mjtGeom.mjGEOM_SPHERE:
            ok = prim.IsA(UsdGeom.Sphere) and abs(UsdGeom.Sphere(prim).GetRadiusAttr().Get() - size[0]) < 1e-7
            desc = f"sphere r {size[0]:.4f}"
        else:
            c = UsdGeom.Cylinder(prim)
            ok = abs(c.GetRadiusAttr().Get() - size[0]) < 1e-7 and abs(c.GetHeightAttr().Get() - 2 * size[1]) < 1e-7
            desc = f"cylinder r {size[0]:.4f} half {size[1]:.4f}"
        ok &= np.allclose(prim.GetAttribute("xformOp:translate").Get(), m.geom_pos[gid], atol=1e-7)
        q = prim.GetAttribute("xformOp:orient").Get()
        qq = np.array([q.GetReal(), *q.GetImaginary()])
        ok &= np.allclose(qq, m.geom_quat[gid], atol=1e-6) or np.allclose(qq, -np.asarray(m.geom_quat[gid]), atol=1e-6)
        ke, kd = geom_gains(m, gid)
        ok &= abs(prim.GetAttribute("newton:contact_ke").Get() - ke) < 1e-3 * ke and abs(prim.GetAttribute("newton:contact_kd").Get() - kd) < 1e-6 * kd
        mat = UsdShade.MaterialBindingAPI(prim).GetDirectBinding(materialPurpose="physics").GetMaterial()
        mapi = UsdPhysics.MaterialAPI(mat.GetPrim())
        ok &= abs(mapi.GetStaticFrictionAttr().Get() - m.geom_friction[gid][0]) < 1e-6 and mapi.GetRestitutionAttr().Get() == 0.0
        check(f"{m.geom(gid).name:18s} {desc}, pose, solref->ke/kd, friction {m.geom_friction[gid][0]:.3f}", ok)

    for j, jn, tok, q1, lke, lkd, act in joint_info:
        prim = stage.GetPrimAtPath(f"/{stage.GetDefaultPrim().GetName()}/joints/{jn}")
        rj = UsdPhysics.RevoluteJoint(prim)
        dof = int(m.jnt_dofadr[j])
        lo, hi = (math.degrees(float(v)) for v in m.jnt_range[j])
        ok = abs(rj.GetLowerLimitAttr().Get() - lo) < 1e-4 and abs(rj.GetUpperLimitAttr().Get() - hi) < 1e-4
        ok &= abs(prim.GetAttribute("mjc:damping").Get() - m.dof_damping[dof]) < 1e-6
        ok &= abs(prim.GetAttribute("newton:armature").Get() - m.dof_armature[dof]) < 1e-9
        ok &= abs(prim.GetAttribute("newton:angular:limitStiffness").Get() / DEG - lke) < 1e-3 * lke
        ok &= abs(prim.GetAttribute("newton:angular:limitDamping").Get() / DEG - lkd) < 1e-3 * lkd
        # the joint axis in the child frame: localRot1 applied to the token axis
        tv = {"X": [1, 0, 0], "Y": [0, 1, 0], "Z": [0, 0, 1]}[rj.GetAxisAttr().Get()]
        r1 = rj.GetLocalRot1Attr().Get()
        axis = rot_mat([r1.GetReal(), *r1.GetImaginary()]) @ np.array(tv, float)
        ok &= np.allclose(axis, m.jnt_axis[j], atol=1e-6)
        if act is not None:
            drv = UsdPhysics.DriveAPI(prim, "angular")
            ok &= abs(drv.GetStiffnessAttr().Get() / DEG - act["kp"]) < 1e-3 and abs(drv.GetMaxForceAttr().Get() - act["force"]) < 1e-6
        check(f"{jn:14s} [{lo:+.2f}, {hi:+.2f}] deg, axis {np.round(axis, 3).tolist()}, damping {m.dof_damping[dof]:g}, "
              f"armature {m.dof_armature[dof]:g}, limit gains ({lke:.4g}, {lkd:.4g}) = solreflimit {m.jnt_solref[j].tolist()}"
              + (f", kp {act['kp']:g}, force {act['force']:g}" if act else ""), ok)

    filt = sorted(tuple(sorted((c.GetParent().GetName(), t.GetParentPath().name))) for c in cols
                  if c.HasAPI(UsdPhysics.FilteredPairsAPI) for t in UsdPhysics.FilteredPairsAPI(c).GetFilteredPairsRel().GetTargets())
    want = []
    for b1, b2 in excl:
        n1 = sum(1 for g in range(m.ngeom) if m.geom_bodyid[g] == b1)
        n2 = sum(1 for g in range(m.ngeom) if m.geom_bodyid[g] == b2)
        want += [tuple(sorted((names["bodies"][m.body(b1).name], names["bodies"][m.body(b2).name])))] * (2 * n1 * n2)
    check(f"filtered pairs == MJCF <exclude> ({len(excl)} pairs)", filt == sorted(want))
    print("ALL USD CHECKS PASSED" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
