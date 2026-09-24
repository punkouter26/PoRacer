#!/usr/bin/env python3
"""Kit-less MJCF -> USD for the Isaac Lab 3 / Newton worm (training/worm/worm.xml -> worm_v3.usda).

    ISAAC\\isaaclab3\\.venv\\Scripts\\python.exe ISAAC\\scripts\\convert_worm_v3.py

Why not Isaac Lab's MjcfConverter: in Isaac Lab 3 it wraps Isaac Sim 6's ``isaacsim.asset.importer.mjcf``
extension, i.e. it needs Kit. The Newton install here is kit-less (no Isaac Sim), so this script does
the import itself:

1. ``mujoco.MjModel.from_xml_path(worm.xml)`` compiles the SHARED MJCF. Masses, inertias, body
   frames, joint axes/ranges and the <contact><exclude> pairs are read from the compiled model, so the
   USD carries exactly what MuJoCo simulates (no re-derivation of the capsule inertia).
2. ``pxr`` (usd-core) writes a flat UsdPhysics articulation that Newton's ``ModelBuilder.add_usd``
   ingests: one rigid body per MJCF body, capsule colliders, revolute joints with limits,
   FilteredPairsAPI for the excluded adjacent segments, and the MuJoCo-only joint terms as the
   ``mjc:`` / ``newton:`` attributes Newton's MuJoCo solver reads:

   * ``mjc:damping``            -> MuJoCo ``dof_damping`` (PASSIVE joint damping, 2.0; not capped
                                   by the 6 N*m force limit - exactly worm.xml's <joint damping>)
   * ``newton:armature``        -> ``dof_armature`` 0.01
   * ``newton:angular:limit*``  = 0 -> Newton leaves MuJoCo's default limit solref (0.02, 1), like
                                   worm.xml (Newton's own default would switch to a soft direct-form
                                   1e4/10 limit)
   * ``newton:contact_ke/kd``   = 1e4 / 200 -> geom solref (0.01, 1), worm.xml's default solref
   * UsdPhysics material 0.9/0.9, restitution 0; torsional/rolling 0.005 / 0.0001 as in worm.xml

The drive (kp 30, damping 0, force limit 6) is authored too, but the task's ImplicitActuatorCfg is
what sets it at runtime. worm.xml is never modified. Writes ISAAC/worm_tasks_v3/assets/worm_v3.usda
and then re-opens it and checks it against the compiled MJCF; exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import mujoco
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

parser = argparse.ArgumentParser(description="Kit-less MJCF -> USD for the Newton worm task.")
parser.add_argument("--mjcf", default=os.path.join(REPO, "training", "worm", "worm.xml"))
parser.add_argument("--out", default=os.path.join(REPO, "ISAAC", "worm_tasks_v3", "assets", "worm_v3.usda"))
args = parser.parse_args()

with open(os.path.join(os.path.dirname(os.path.abspath(args.mjcf)), "worm_rig.json"), "r", encoding="utf-8") as f:
    RIG = json.load(f)

ACTION_ORDER = [f"j{i}_{ax}" for i in range(4) for ax in ("pitch", "yaw")]
# solref (timeconst, dampratio) -> Newton ke/kd as SolverMuJoCo.convert_solref inverts it
# (d_width = d_r = 1): kd = 2 / timeconst, ke = 1 / (timeconst^2 * dampratio^2)
GEOM_SOLREF = (0.01, 1.0)
CONTACT_KE = 1.0 / (GEOM_SOLREF[0] ** 2 * GEOM_SOLREF[1] ** 2)
CONTACT_KD = 2.0 / GEOM_SOLREF[0]


def quat_wxyz_to_gf(q) -> Gf.Quatf:
    return Gf.Quatf(float(q[0]), Gf.Vec3f(float(q[1]), float(q[2]), float(q[3])))


def main() -> int:
    m = mujoco.MjModel.from_xml_path(os.path.abspath(args.mjcf))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)  # world poses of every body at qpos0 (straight worm, head at z 0.05)

    body_names = [m.body(i).name for i in range(1, m.nbody)]  # skip world
    assert body_names == ["seg0", "link0", "seg1", "link1", "seg2", "link2", "seg3", "link3", "seg4"], body_names

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if os.path.exists(args.out):
        os.remove(args.out)  # always regenerate from worm.xml
    stage = Usd.Stage.CreateNew(args.out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdPhysics.SetStageKilogramsPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/worm")
    stage.SetDefaultPrim(root.GetPrim())
    root_prim = root.GetPrim()
    # articulation root on the asset root Xform (PhysX / Newton convention for floating bases)
    UsdPhysics.ArticulationRootAPI.Apply(root_prim)
    root_prim.CreateAttribute("newton:selfCollisionEnabled", Sdf.ValueTypeNames.Bool).Set(True)
    root_prim.CreateAttribute("physxArticulation:enabledSelfCollisions", Sdf.ValueTypeNames.Bool).Set(True)

    # physics material shared by every worm shape: 0.9 / 0.9 / bounce 0 (+ worm.xml torsion/rolling)
    friction = m.geom_friction[m.geom("seg0").id]
    mat = UsdShade.Material.Define(stage, "/worm/Looks/wormPhysicsMaterial")
    mapi = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    mapi.CreateStaticFrictionAttr(float(friction[0]))
    mapi.CreateDynamicFrictionAttr(float(friction[0]))
    mapi.CreateRestitutionAttr(0.0)
    mat.GetPrim().CreateAttribute("newton:torsionalFriction", Sdf.ValueTypeNames.Float).Set(float(friction[1]))
    mat.GetPrim().CreateAttribute("newton:rollingFriction", Sdf.ValueTypeNames.Float).Set(float(friction[2]))

    body_prims = {}
    for bid in range(1, m.nbody):
        name = m.body(bid).name
        xf = UsdGeom.Xform.Define(stage, f"/worm/{name}")
        p = d.xpos[bid]
        q = d.xquat[bid]
        xf.AddTranslateOp().Set(Gf.Vec3d(*map(float, p)))
        xf.AddOrientOp().Set(quat_wxyz_to_gf(q))
        prim = xf.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(float(m.body_mass[bid]))
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*map(float, m.body_ipos[bid])))
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*map(float, m.body_inertia[bid])))
        mass.CreatePrincipalAxesAttr(quat_wxyz_to_gf(m.body_iquat[bid]))
        body_prims[name] = prim

        # colliders: MJCF capsule fromto along x -> UsdGeom.Capsule axis X, height = cylinder length
        for gid in range(m.ngeom):
            if m.geom_bodyid[gid] != bid:
                continue
            assert m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE, m.geom(gid).name
            cap = UsdGeom.Capsule.Define(stage, f"/worm/{name}/collision")
            cap.CreateAxisAttr(UsdGeom.Tokens.x)
            cap.CreateRadiusAttr(float(m.geom_size[gid][0]))
            cap.CreateHeightAttr(float(2.0 * m.geom_size[gid][1]))
            gq = m.geom_quat[gid]
            # MuJoCo capsules are along the geom's local z; fromto along body x gives a geom quat
            # that maps z -> x. Rebuild the pose so the USD capsule (axis X) lies along body x.
            gz = np.zeros(9)
            mujoco.mju_quat2Mat(gz, gq)
            geom_axis = gz.reshape(3, 3)[:, 2]
            assert np.allclose(np.abs(geom_axis), [1.0, 0.0, 0.0], atol=1e-9), geom_axis
            cap.AddTranslateOp().Set(Gf.Vec3d(*map(float, m.geom_pos[gid])))
            cprim = cap.GetPrim()
            UsdPhysics.CollisionAPI.Apply(cprim)
            cprim.CreateAttribute("newton:contact_ke", Sdf.ValueTypeNames.Float).Set(CONTACT_KE)
            cprim.CreateAttribute("newton:contact_kd", Sdf.ValueTypeNames.Float).Set(CONTACT_KD)
            cprim.CreateAttribute("newton:contactMargin", Sdf.ValueTypeNames.Float).Set(float(m.geom_margin[gid]))
            UsdShade.MaterialBindingAPI.Apply(cprim).Bind(
                mat, bindingStrength=UsdShade.Tokens.weakerThanDescendants, materialPurpose="physics"
            )
            cap.CreateDisplayColorAttr([Gf.Vec3f(0.35, 0.55, 0.9)])

    # joints: MJCF hinge on child body c at child-frame anchor a, axis ax (child frame)
    UsdGeom.Scope.Define(stage, "/worm/joints")
    for jn in ACTION_ORDER:
        jid = m.joint(jn).id
        child = m.jnt_bodyid[jid]
        parent = m.body_parentid[child]
        cname, pname = m.body(child).name, m.body(parent).name
        anchor_c = np.array(m.jnt_pos[jid], dtype=float)
        axis_c = np.array(m.jnt_axis[jid], dtype=float)
        # the anchor in the parent frame at qpos0 (child pose relative to parent from body_pos/quat)
        rot = np.zeros(9)
        mujoco.mju_quat2Mat(rot, m.body_quat[child])
        rot = rot.reshape(3, 3)
        anchor_p = m.body_pos[child] + rot @ anchor_c
        axis_p = rot @ axis_c
        assert np.allclose(m.body_quat[child], [1, 0, 0, 0]), "worm bodies are all aligned at qpos0"
        tok = {(0, 1, 0): UsdPhysics.Tokens.y, (0, 0, 1): UsdPhysics.Tokens.z, (1, 0, 0): UsdPhysics.Tokens.x}[
            tuple(int(round(v)) for v in axis_c)
        ]
        assert np.allclose(axis_p, axis_c)
        j = UsdPhysics.RevoluteJoint.Define(stage, f"/worm/joints/{jn}")
        j.CreateBody0Rel().SetTargets([body_prims[pname].GetPath()])
        j.CreateBody1Rel().SetTargets([body_prims[cname].GetPath()])
        j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, anchor_p)))
        j.CreateLocalRot0Attr(Gf.Quatf(1.0))
        j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, anchor_c)))
        j.CreateLocalRot1Attr(Gf.Quatf(1.0))
        j.CreateAxisAttr(tok)
        lo, hi = (math.degrees(float(v)) for v in m.jnt_range[jid])
        j.CreateLowerLimitAttr(lo)
        j.CreateUpperLimitAttr(hi)
        jp = j.GetPrim()
        drive = UsdPhysics.DriveAPI.Apply(jp, "angular")
        drive.CreateTypeAttr("force")
        drive.CreateStiffnessAttr(float(RIG["kp"]) * math.pi / 180.0)  # USD angular drives are per degree
        drive.CreateDampingAttr(0.0)
        drive.CreateMaxForceAttr(float(RIG["forceLimit"]))
        drive.CreateTargetPositionAttr(0.0)
        dof = m.jnt_dofadr[jid]
        jp.CreateAttribute("mjc:damping", Sdf.ValueTypeNames.Float).Set(float(m.dof_damping[dof]))
        jp.CreateAttribute("newton:armature", Sdf.ValueTypeNames.Float).Set(float(m.dof_armature[dof]))
        jp.CreateAttribute("physxJoint:armature", Sdf.ValueTypeNames.Float).Set(float(m.dof_armature[dof]))
        jp.CreateAttribute("newton:friction", Sdf.ValueTypeNames.Float).Set(float(m.dof_frictionloss[dof]))
        jp.CreateAttribute("newton:angular:limitStiffness", Sdf.ValueTypeNames.Float).Set(0.0)
        jp.CreateAttribute("newton:angular:limitDamping", Sdf.ValueTypeNames.Float).Set(0.0)

    # <contact><exclude body1 body2/> -> FilteredPairsAPI. Authored on the COLLIDER prims (shape ->
    # shape): Newton's USD importer only reads physics:filteredPairs on shapes and maps the targets
    # through its shape table (a body-level filter is silently ignored and the pair keeps colliding).
    pairs = [(m.body(int(m.exclude_signature[i] >> 16)).name, m.body(int(m.exclude_signature[i] & 0xFFFF)).name)
             for i in range(m.nexclude)]
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            src = stage.GetPrimAtPath(f"/worm/{x}/collision")
            api = UsdPhysics.FilteredPairsAPI.Apply(src)
            rel = api.CreateFilteredPairsRel()
            rel.AddTarget(Sdf.Path(f"/worm/{y}/collision"))
    stage.GetRootLayer().Save()
    print(f"[convert_worm_v3] wrote {args.out}")
    print(f"[convert_worm_v3] excluded pairs from MJCF: {pairs}")
    return verify(m, pairs)


def verify(m, pairs) -> int:
    fails = []

    def check(label, ok, detail=""):
        print(f"  [{'OK ' if ok else 'FAIL'}] {label} {detail}")
        if not ok:
            fails.append(label)

    stage = Usd.Stage.Open(args.out)
    bodies = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    check("9 rigid bodies", len(bodies) == 9, str([b.GetName() for b in bodies]))
    total = sum(UsdPhysics.MassAPI(b).GetMassAttr().Get() for b in bodies)
    check("total mass 7.08 kg", abs(total - RIG["totalMass"]) < 1e-5, f"{total:.6f}")
    for b in bodies:
        mb = m.body(b.GetName())
        mm = UsdPhysics.MassAPI(b).GetMassAttr().Get()
        ii = UsdPhysics.MassAPI(b).GetDiagonalInertiaAttr().Get()
        check(f"{b.GetName()} mass/inertia == MuJoCo", abs(mm - mb.mass[0]) < 1e-6 and np.allclose(ii, mb.inertia, atol=1e-9),
              f"m={mm:.6f} I=({ii[0]:.6g}, {ii[1]:.6g}, {ii[2]:.6g})")
    caps = [p for p in stage.Traverse() if p.IsA(UsdGeom.Capsule) and p.HasAPI(UsdPhysics.CollisionAPI)]
    check("5 capsule colliders r 0.045 h 0.15", len(caps) == 5 and all(
        abs(UsdGeom.Capsule(c).GetRadiusAttr().Get() - RIG["radius"]) < 1e-7
        and abs(UsdGeom.Capsule(c).GetHeightAttr().Get() - 2 * RIG["capsuleHalfLength"]) < 1e-7 for c in caps))
    joints = [p for p in stage.Traverse() if p.IsA(UsdPhysics.RevoluteJoint)]
    check("8 revolute joints in action order", [j.GetName() for j in joints] == ACTION_ORDER, str([j.GetName() for j in joints]))
    for j in joints:
        rj = UsdPhysics.RevoluteJoint(j)
        ok = abs(rj.GetLowerLimitAttr().Get() + 45.0) < 1e-4 and abs(rj.GetUpperLimitAttr().Get() - 45.0) < 1e-4
        ok &= abs(j.GetAttribute("mjc:damping").Get() - RIG["jointDamping"]) < 1e-7
        ok &= abs(j.GetAttribute("newton:armature").Get() - RIG["armature"]) < 1e-7
        want_axis = "Y" if j.GetName().endswith("pitch") else "Z"
        ok &= rj.GetAxisAttr().Get() == want_axis
        check(f"{j.GetName()} +/-45 deg, axis {want_axis}, damping 2.0, armature 0.01", ok)
    filt = sorted(tuple(sorted((c.GetParent().GetName(), t.GetParentPath().name))) for c in caps
                  if c.HasAPI(UsdPhysics.FilteredPairsAPI)
                  for t in UsdPhysics.FilteredPairsAPI(c).GetFilteredPairsRel().GetTargets())
    want = sorted(tuple(sorted(p)) for p in pairs for _ in range(2))
    check("filtered pairs = MJCF excludes (adjacent segments)", filt == want and len(pairs) == 4, str(sorted(set(filt))))
    print("ALL USD CHECKS PASSED" if not fails else f"FAILED: {fails}")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
