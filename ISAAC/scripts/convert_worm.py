#!/usr/bin/env python3
"""Convert training/worm/worm.xml (the shared MJCF) to the USD the Isaac Lab worm task loads.

Run with the Isaac Lab venv's python (no isaaclab.bat needed)::

    set OMNI_KIT_ACCEPT_EULA=YES
    ISAAC\\isaaclab\\.venv\\Scripts\\python.exe ISAAC\\scripts\\convert_worm.py

Writes ``ISAAC/worm_tasks/assets/worm.usd`` and then VERIFIES it against WORM_SPEC.md by reading
the USD back with pxr: 5 capsule segments + 4 collider-free links, 8 revolute joints at +/-45 deg,
masses (7.08 kg), capsule sizes, and the adjacent-segment collision filters. The importer's own
floor plane (worldbody geom "floor") is stripped from the asset: the task supplies the ground.

worm.xml itself is never modified. If the checks fail the script exits non-zero.
"""

import argparse
import math
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

parser = argparse.ArgumentParser(description="MJCF -> USD for the worm, with verification.")
parser.add_argument("--mjcf", default=os.path.join(REPO, "training", "worm", "worm.xml"))
parser.add_argument("--out_dir", default=os.path.join(REPO, "ISAAC", "worm_tasks", "assets"))
parser.add_argument("--verify_only", action="store_true", help="Skip conversion, only check the existing USD.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # noqa: E402

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg  # noqa: E402

SEGMENTS = [f"seg{i}" for i in range(5)]
LINKS = [f"link{i}" for i in range(4)]
ACTION_ORDER = [f"j{i}_{ax}" for i in range(4) for ax in ("pitch", "yaw")]
import json  # noqa: E402

with open(os.path.join(os.path.dirname(os.path.abspath(args.mjcf)), "worm_rig.json"), "r", encoding="utf-8") as _f:
    RIG = json.load(_f)
SEG_MASS = float(RIG["segmentMass"])
LINK_MASS = float(RIG["linkMass"])
RANGE_DEG = math.degrees(float(RIG["jointRangeRad"]))
RADIUS = float(RIG["radius"])
HALF = float(RIG["capsuleHalfLength"])
TOTAL_MASS = float(RIG["totalMass"])


def convert() -> str:
    # the headless experience file does not load the importer; its commands are unregistered
    # (MJCFCreateImportConfig returns None) until the extension is enabled explicitly.
    import omni.kit.app

    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(
        "isaacsim.asset.importer.mjcf", True
    )
    cfg = MjcfConverterCfg(
        asset_path=os.path.abspath(args.mjcf),
        usd_dir=os.path.join(os.path.abspath(args.out_dir), "_import"),
        usd_file_name="worm.usd",
        fix_base=False,
        self_collision=True,  # everything collides except the <contact><exclude> pairs
        make_instanceable=False,  # primitives only; nothing to share
        import_sites=False,
        import_inertia_tensor=True,
        force_usd_conversion=True,
    )
    conv = MjcfConverter(cfg)
    print(f"[convert] wrote {conv.usd_path}")
    return conv.usd_path


def flatten_and_clean(imported: str, out_path: str) -> list[str]:
    """Flatten the importer's layered output (worm.usd + configuration/*.usd) into ONE
    self-contained ``worm.usda`` and remove what the task must not inherit:

    * ``/worm/worldBody`` - the importer turns the MJCF worldbody (the ``floor`` plane) into a
      second rigid body carrying its OWN ArticulationRootAPI. The task has its own ground, and a
      second articulation root in the asset breaks Isaac Lab's single-root articulation parse.
    """
    import shutil

    src = Usd.Stage.Open(imported)
    # de-instance so the flattened file holds real prims rather than prototypes
    for prim in src.Traverse(Usd.TraverseInstanceProxies()):
        if prim.IsInstance():
            prim.SetInstanceable(False)
    for prim in list(src.Traverse()):
        if prim.IsInstanceable():
            prim.SetInstanceable(False)
    layer = src.Flatten()
    layer.Export(out_path)
    del src

    stage = Usd.Stage.Open(out_path)
    removed = []
    for path in ("/worm/worldBody",):
        if stage.GetPrimAtPath(path):
            stage.RemovePrim(path)
            removed.append(path)
    # The importer ignores <geom mass=...> on the capsules: it writes physics:density = 0 and no
    # mass, leaving PhysX to fall back to its default density. That happens to give the same
    # 1.336 kg (build_worm.py derives the mass from density 1000), but only by accident, so the
    # mass is authored explicitly. Inertia stays unauthored: PhysX then computes it from the
    # capsule collider scaled to this mass, i.e. the solid-capsule inertia MuJoCo also uses.
    for i in range(5):
        prim = next((p for p in stage.Traverse() if p.GetName() == f"seg{i}" and p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
        if prim is None:
            raise RuntimeError(f"seg{i} rigid body not found in the imported USD")
        UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(SEG_MASS))
    root = stage.GetPrimAtPath("/worm")
    stage.SetDefaultPrim(root)
    stage.GetRootLayer().Save()
    shutil.rmtree(os.path.dirname(imported), ignore_errors=True)
    return removed


def verify(path: str) -> bool:
    stage = Usd.Stage.Open(path)
    ok = True
    rows = []

    def check(label, expected, got, good):
        nonlocal ok
        ok &= bool(good)
        rows.append((label, str(expected), str(got), "OK" if good else "FAIL"))

    bodies, joints, colliders = {}, {}, {}
    filtered = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            bodies[prim.GetName()] = prim
        if prim.IsA(UsdPhysics.RevoluteJoint):
            joints[prim.GetName()] = prim
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            colliders[str(prim.GetPath())] = prim
        if prim.HasAPI(UsdPhysics.FilteredPairsAPI):
            tg = UsdPhysics.FilteredPairsAPI(prim).GetFilteredPairsRel().GetTargets()
            filtered[prim.GetName()] = sorted(Sdf.Path(t).name for t in tg)
    print("[verify] prims:")
    for prim in stage.Traverse():
        print(f"    {prim.GetPath()}  <{prim.GetTypeName()}>  apis={[a for a in prim.GetAppliedSchemas()]}")

    check("segment bodies", SEGMENTS, sorted(b for b in bodies if b.startswith("seg")),
          sorted(b for b in bodies if b.startswith("seg")) == SEGMENTS)
    check("link bodies", LINKS, sorted(b for b in bodies if b.startswith("link")),
          sorted(b for b in bodies if b.startswith("link")) == LINKS)
    check("revolute joints", ACTION_ORDER, sorted(joints), sorted(joints) == sorted(ACTION_ORDER))

    total = 0.0
    for name, prim in sorted(bodies.items()):
        m = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if prim.HasAPI(UsdPhysics.MassAPI) else None
        total += m or 0.0
        want = SEG_MASS if name.startswith("seg") else LINK_MASS
        check(f"mass {name}", f"{want:.4f}", f"{m:.4f}" if m is not None else None,
              m is not None and abs(m - want) < 1e-3)
    check("total mass [kg]", f"{TOTAL_MASS:.3f}", f"{total:.3f}", abs(total - TOTAL_MASS) < 1e-3)

    for name in ACTION_ORDER:
        prim = joints.get(name)
        if prim is None:
            continue
        j = UsdPhysics.RevoluteJoint(prim)
        lo, hi = j.GetLowerLimitAttr().Get(), j.GetUpperLimitAttr().Get()
        axis = j.GetAxisAttr().Get()
        b0 = [str(t) for t in j.GetBody0Rel().GetTargets()]
        b1 = [str(t) for t in j.GetBody1Rel().GetTargets()]
        # the joint axis in the PARENT body frame (localRot0 rotates the joint frame)
        r0 = j.GetLocalRot0Attr().Get()
        ax_vec = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}[axis]
        world_ax = Gf.Rotation(Gf.Quatd(r0)).TransformDir(ax_vec) if r0 is not None else ax_vec
        want_ax = (0, 1, 0) if name.endswith("pitch") else (0, 0, 1)
        ax_good = all(abs(world_ax[k] - want_ax[k]) < 1e-4 for k in range(3))
        check(f"limits {name} [deg]", f"+/-{RANGE_DEG}", f"[{lo:.3f}, {hi:.3f}]",
              abs(lo + RANGE_DEG) < 1e-2 and abs(hi - RANGE_DEG) < 1e-2)
        check(f"axis {name} (parent frame)", want_ax, tuple(round(v, 4) for v in world_ax), ax_good)
        check(f"bodies {name}", "parent->child",
              f"{b0[0].split('/')[-1] if b0 else None}->{b1[0].split('/')[-1] if b1 else None}", bool(b0 and b1))
        lp0 = j.GetLocalPos0Attr().Get()
        rows.append((f"localPos0 {name}", "", str(tuple(round(v, 4) for v in lp0)), "info"))
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if drive:
            rows.append((f"USD drive {name}", "(overridden by task)",
                         f"kp={drive.GetStiffnessAttr().Get()} kd={drive.GetDampingAttr().Get()} "
                         f"max={drive.GetMaxForceAttr().Get()}", "info"))
        arm = prim.GetAttribute("physxJoint:armature")
        if arm and arm.Get() is not None:
            rows.append((f"USD armature {name}", "(overridden by task)", arm.Get(), "info"))

    caps = [p for p in colliders.values() if p.IsA(UsdGeom.Capsule)]
    check("capsule colliders", 5, len(caps), len(caps) == 5)
    for p in caps:
        c = UsdGeom.Capsule(p)
        r, h, ax = c.GetRadiusAttr().Get(), c.GetHeightAttr().Get(), c.GetAxisAttr().Get()
        check(f"capsule {p.GetPath()}", f"r={RADIUS} h={2*HALF} axis X",
              f"r={r:.4f} h={h:.4f} axis {ax}", abs(r - RADIUS) < 1e-4 and abs(h - 2 * HALF) < 1e-4)
    others = [str(p.GetPath()) for p in colliders.values() if not p.IsA(UsdGeom.Capsule)]
    check("non-capsule colliders", [], others, not others)

    rows.append(("FilteredPairsAPI", "seg_i <-> seg_i+1", str(filtered) or "none", "info"))
    pairs_ok = all(
        (f"seg{i+1}" in filtered.get(f"seg{i}", [])) or (f"seg{i}" in filtered.get(f"seg{i+1}", []))
        for i in range(4)
    )
    check("adjacent segments filtered", "4 pairs", pairs_ok, pairs_ok)

    art = [p for p in stage.Traverse() if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    check("articulation root", 1, [str(p.GetPath()) for p in art], len(art) == 1)
    for p in art:
        sc = p.GetAttribute("physxArticulation:enabledSelfCollisions")
        rows.append(("enabledSelfCollisions (USD)", True, sc.Get() if sc else None, "info"))

    w = max(len(r[0]) for r in rows)
    print("\n[verify] " + "-" * 90)
    for r in rows:
        print(f"  {r[0]:<{w}}  expected {r[1]:<28} got {r[2]:<40} {r[3]}")
    print("[verify] " + ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    return ok


def main():
    path = os.path.join(os.path.abspath(args.out_dir), "worm.usda")
    if not args.verify_only:
        imported = convert()
        removed = flatten_and_clean(imported, path)
        print(f"[convert] flattened -> {path}; removed {removed or 'nothing'}")
    return 0 if verify(path) else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        simulation_app.close()
        os._exit(code)
