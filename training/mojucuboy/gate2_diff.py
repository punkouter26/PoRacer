"""Gate 2: diff the authored MJCF against the one Unity exported back out.

A textual diff of the two files is close to useless -- org.mujoco renames every
element, writes every default explicitly, and reorders sections. What matters is
whether the two COMPILE to the same physics, so this compares the compiled
mjModel arrays element by element, matching by name.

The exporter appends "_<n>" to every name, so names are matched with that suffix
stripped.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
TOL = 1e-4


def strip(name: str) -> str:
    return re.sub(r"_\d+$", "", name or "")


def names(model, objtype, count):
    return [strip(mujoco.mj_id2name(model, objtype, i)) for i in range(count)]


def compare(label, a_vals, b_vals, a_names, b_names, findings, mode="abs"):
    """Match rows by name and report the worst difference.

    mode="quat" compares up to sign: MJCF writes the root bodies as
    quat="-1 0 0 0", and q and -q are the same rotation, so a raw absolute diff
    reports a spurious 2.0.

    mode="rel" compares relatively: org.mujoco writes floats to ~6 significant
    figures, so a derived damping term of -112.06149 comes back as -112.061. That
    is text precision, not a physics difference, and an absolute tolerance would
    flag it.
    """
    b_index = {n: i for i, n in enumerate(b_names)}
    missing = [n for n in a_names if n not in b_index]
    if missing:
        findings.append((label, "MISSING", f"not in export: {missing[:6]}", np.inf, ""))
        return
    worst, worst_at = 0.0, ""
    for i, n in enumerate(a_names):
        av = np.atleast_1d(np.asarray(a_vals[i], dtype=float)).ravel()
        bv = np.atleast_1d(np.asarray(b_vals[b_index[n]], dtype=float)).ravel()
        if av.shape != bv.shape:
            findings.append((label, "SHAPE", f"{n}: {av.shape} vs {bv.shape}", np.inf, n))
            continue
        if av.size == 0:
            continue
        if mode == "quat":
            d = float(min(np.max(np.abs(av - bv)), np.max(np.abs(av + bv))))
        elif mode == "rel":
            scale = np.maximum(np.abs(av), 1.0)
            d = float(np.max(np.abs(av - bv) / scale))
        else:
            d = float(np.max(np.abs(av - bv)))
        if d > worst:
            worst, worst_at = d, n
    unit = "max rel" if mode == "rel" else "max |d|"
    status = "OK" if worst < TOL else "DIFF"
    findings.append((label, status, f"{unit} = {worst:.3e}", worst, worst_at))


def main() -> int:
    src = mujoco.MjModel.from_xml_path(str(HERE / "mojucuboy.xml"))
    dst = mujoco.MjModel.from_xml_path(str(HERE / "mojucuboy_roundtrip.xml"))

    print("=== CENSUS ===")
    print(f"{'quantity':<12}{'authored':>10}{'roundtrip':>11}   match")
    census_ok = True
    for field in ("nbody", "njnt", "nu", "nq", "nv", "ngeom"):
        a, b = getattr(src, field), getattr(dst, field)
        census_ok &= a == b
        print(f"{field:<12}{a:>10}{b:>11}   {'OK' if a == b else 'MISMATCH'}")

    print("\n=== OPTIONS ===")
    for field in ("timestep", "gravity", "integrator", "cone", "jacobian", "iterations"):
        a = getattr(src.opt, field)
        b = getattr(dst.opt, field)
        same = np.allclose(np.atleast_1d(a), np.atleast_1d(b))
        print(f"  {field:<12} authored={a}  roundtrip={b}   {'OK' if same else 'DIFF'}")

    OBJ = mujoco.mjtObj
    bn_a, bn_b = names(src, OBJ.mjOBJ_BODY, src.nbody), names(dst, OBJ.mjOBJ_BODY, dst.nbody)
    jn_a, jn_b = names(src, OBJ.mjOBJ_JOINT, src.njnt), names(dst, OBJ.mjOBJ_JOINT, dst.njnt)
    gn_a, gn_b = names(src, OBJ.mjOBJ_GEOM, src.ngeom), names(dst, OBJ.mjOBJ_GEOM, dst.ngeom)
    an_a, an_b = names(src, OBJ.mjOBJ_ACTUATOR, src.nu), names(dst, OBJ.mjOBJ_ACTUATOR, dst.nu)

    findings: list[tuple] = []
    for label, mode in (
        ("body_pos", "abs"), ("body_quat", "quat"), ("body_mass", "abs"),
        ("body_inertia", "abs"), ("body_ipos", "abs"), ("body_iquat", "quat"),
    ):
        compare(label, getattr(src, label), getattr(dst, label), bn_a, bn_b, findings, mode)
    for label in ("jnt_type", "jnt_axis", "jnt_pos", "jnt_range", "jnt_limited",
                  "jnt_stiffness", "jnt_margin"):
        compare(label, getattr(src, label), getattr(dst, label), jn_a, jn_b, findings)
    compare("dof_armature", src.dof_armature, dst.dof_armature,
            [jn_a[src.dof_jntid[i]] for i in range(src.nv)],
            [jn_b[dst.dof_jntid[i]] for i in range(dst.nv)], findings)
    compare("dof_damping", src.dof_damping, dst.dof_damping,
            [jn_a[src.dof_jntid[i]] for i in range(src.nv)],
            [jn_b[dst.dof_jntid[i]] for i in range(dst.nv)], findings)
    for label, mode in (("geom_type", "abs"), ("geom_size", "abs"), ("geom_pos", "abs"),
                        ("geom_quat", "quat"), ("geom_friction", "abs"),
                        ("geom_condim", "abs"), ("geom_solref", "abs"), ("geom_solimp", "abs")):
        compare(label, getattr(src, label), getattr(dst, label), gn_a, gn_b, findings, mode)
    for label, mode in (("actuator_gainprm", "abs"), ("actuator_biasprm", "rel"),
                        ("actuator_ctrlrange", "abs"), ("actuator_forcerange", "abs"),
                        ("actuator_gear", "abs"), ("actuator_gaintype", "abs"),
                        ("actuator_biastype", "abs")):
        compare(label, getattr(src, label), getattr(dst, label), an_a, an_b, findings, mode)

    print("\n=== COMPILED MODEL ARRAYS (matched by name) ===")
    print(f"{'array':<20}{'status':<8}{'detail':<24}worst-at")
    failures = []
    for label, status, detail, worst, at in findings:
        print(f"{label:<20}{status:<8}{detail:<24}{at}")
        if status != "OK":
            failures.append((label, detail, at))

    # The joint-axis defect the brief warns about is worth naming explicitly, since
    # a collapsed axis silently turns a yaw joint into a duplicate pitch joint.
    print("\n=== JOINT AXES (authored vs round-trip) ===")
    b_index = {n: i for i, n in enumerate(jn_b)}
    axis_bad = []
    for i, n in enumerate(jn_a):
        if src.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        a_axis = src.jnt_axis[i]
        b_axis = dst.jnt_axis[b_index[n]]
        d = float(np.max(np.abs(a_axis - b_axis)))
        flag = "" if d < TOL else "   <-- COLLAPSED"
        if d >= TOL:
            axis_bad.append(n)
        print(f"  {n:<16} authored [{a_axis[0]:5.2f}{a_axis[1]:6.2f}{a_axis[2]:6.2f}]"
              f"   roundtrip [{b_axis[0]:5.2f}{b_axis[1]:6.2f}{b_axis[2]:6.2f}]{flag}")

    print("\n=== GATE 2 VERDICT ===")
    if not census_ok:
        print("  FAIL: census mismatch")
    if axis_bad:
        print(f"  FAIL: {len(axis_bad)} joint axes did not survive the round trip: {axis_bad}")
    for label, detail, at in failures:
        print(f"  FAIL: {label} {detail} (worst at {at})")
    if census_ok and not failures:
        print(f"  PASS: every compared array agrees within {TOL:g}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
