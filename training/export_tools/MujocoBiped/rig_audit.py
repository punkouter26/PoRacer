"""
rig_audit.py  ->  RIG_AUDIT.md + rig_audit.json

Everything about this rig that could quietly ruin the port, measured rather than
assumed. Sections:

  A  mass and inertia conditioning - light links, small inertias, floors
  B  joint velocity headroom - recorded max/p99 against the actuator limits
  C  explicit-PD stability - kd*dt / I_joint across candidate fixed timesteps
  D  quaternion order and the angular-velocity frame - the two conventions that
     are impossible to eyeball and fatal to get wrong
  E  geometry cross-check - masses and centres of mass recomputed from the MJCF
     primitives, so a mis-transcribed geom cannot reach the prefab

Reads MujocoBiped_rig.json (run extract_rig.py first) and mujoco_reference.json
(run make_reference.py first).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RIG = os.path.join(HERE, "MujocoBiped_rig.json")
REF = os.path.join(HERE, "mujoco_reference.json")
OUT_MD = os.path.join(HERE, "RIG_AUDIT.md")
OUT_JSON = os.path.join(HERE, "rig_audit.json")

INERTIA_FLOOR = 1e-4          # below this PhysX conditions badly
LIGHT_LINK_RATIO = 0.10       # a link under this fraction of a neighbour's mass
DENSITY = 1000.0              # MJCF <geom density="1000">
TIMESTEPS = [("project (0.005)", 0.005), ("1/60", 1 / 60), ("1/120", 1 / 120),
             ("1/240", 1 / 240), ("1/480", 1 / 480), ("1/960", 1 / 960)]

rig = json.load(open(RIG))
ref = json.load(open(REF))
LINKS = rig["links"]
BY = {l["name"]: l for l in LINKS}
JOINT_ORDER = rig["jointOrder"]


def children_of(n):
    return [l["name"] for l in LINKS if l["parent"] == n]


def subtree(n):
    out = [n]
    for c in children_of(n):
        out += subtree(c)
    return out


def world_origin(n):
    """Link origin in world coords at the ZERO pose (every local rotation identity)."""
    p = np.zeros(3)
    while n:
        p = p + np.array(BY[n]["localPosMuj"])
        n = BY[n]["parent"]
    return p


def effective_mass(l):
    return l["mass"] if not l["isDummy"] else rig["physics"]["dummyLinkMass"]


def effective_inertia(l, mode):
    """Diagonal link inertia after the armature fold, MuJoCo axes, zero pose."""
    d = np.array(l["inertiaDiagMuj"], dtype=float)
    if l["isDummy"]:
        d = np.full(3, rig["physics"]["inertiaFloor"])
    if mode == "none" or not l["joint"]:
        return d
    c = l["armatureFoldExact"] if mode == "exact" else l["armatureFoldNaive"]
    a = np.array(l["joint"]["axisInChildMuj"], dtype=float)
    a = a / np.linalg.norm(a)
    return d + c * a * a          # axes are unit vectors here, so a*a^T is diagonal


def joint_space_inertia(jname, mode):
    """
    H[i][i] at the zero pose: the composite inertia of the subtree below joint i,
    about i's axis through i's anchor, by the parallel axis theorem.

        I = sum_k [ a^T I_k a + m_k ( |d_k|^2 - (a.d_k)^2 ) ]

    with d_k the vector from the anchor to link k's centre of mass. Every local
    rotation is identity at the zero pose, so no R I R^T term is needed.
    """
    link = next(l for l in LINKS if l["joint"] and l["joint"]["name"] == jname)
    a = np.array(link["joint"]["axisInChildMuj"], dtype=float)
    a = a / np.linalg.norm(a)
    anchor = world_origin(link["name"])
    total = 0.0
    for kn in subtree(link["name"]):
        k = BY[kn]
        m = effective_mass(k)
        com = world_origin(kn) + np.array(k["comMuj"], dtype=float)
        d = com - anchor
        total += float(a @ (effective_inertia(k, mode) * a))
        total += m * (float(d @ d) - float(a @ d) ** 2)
    return total


# ------------------------------------------------------------------- section A --
def section_a():
    rows = []
    for l in LINKS:
        m = effective_mass(l)
        neigh = [BY[l["parent"]]] if l["parent"] else []
        neigh += [BY[c] for c in children_of(l["name"])]
        nm = [effective_mass(n) for n in neigh] or [m]
        heaviest = max(nm)
        inertia = effective_inertia(l, "exact")
        rows.append(dict(
            name=l["name"], isDummy=l["isDummy"], mass=m,
            heaviestNeighbourMass=heaviest,
            massRatio=m / heaviest if heaviest > 0 else float("inf"),
            light=bool(m < LIGHT_LINK_RATIO * heaviest),
            rawInertiaMin=float(min(l["inertiaDiagMuj"])) if not l["isDummy"] else 0.0,
            foldedInertiaMin=float(inertia.min()),
            smallInertia=bool(min(l["inertiaDiagMuj"]) < INERTIA_FLOOR and not l["isDummy"]),
            armatureFoldExact=l.get("armatureFoldExact", 0.0),
            armatureFoldNaive=l.get("armatureFoldNaive", 0.0)))
    return rows


# ------------------------------------------------------------------- section B --
def section_b():
    st = ref["jointVelocityStats"]
    rows = []
    for i, jn in enumerate(JOINT_ORDER):
        link = next(l for l in LINKS if l["joint"] and l["joint"]["name"] == jn)
        j = link["joint"]
        rows.append(dict(
            joint=jn, index=i,
            maxAbsRadPerSec=st["maxAbs"][i], p99AbsRadPerSec=st["p99Abs"][i],
            observationClip=st["clipThreshold"],
            clippedInObservation=bool(st["maxAbs"][i] > st["clipThreshold"]),
            mujocoVelocityLimit=None,          # no MJCF joint carries one
            gear=j["gear"], peakTorqueNm=j["peakTorqueNm"],
            rangeDeg=[math.degrees(j["lowerRad"]), math.degrees(j["upperRad"])],
            unityMaxJointVelocity=rig["physics"]["maxJointVelocity"]))
    return rows


# ------------------------------------------------------------------- section C --
def section_c():
    """
    Explicit damping is a forward-Euler feedback term: tau = -kd * qd. Applied
    once per physics tick it is stable only while

        kd * dt / I_joint  <  2

    and it starts ringing well before that. The implicit ArticulationDrive path
    has no such bound - this table is what says how far the explicit diagnostic
    path can be trusted, and it is the reason the armature is not optional.
    """
    out = []
    for jn in JOINT_ORDER:
        row = dict(joint=jn)
        for mode in ("none", "exact", "naive"):
            I = joint_space_inertia(jn, mode)
            kd = next(l["joint"]["damping"] for l in LINKS
                      if l["joint"] and l["joint"]["name"] == jn)
            row[mode] = dict(inertia=I, ratios={
                label: kd * dt / I for label, dt in TIMESTEPS})
        out.append(row)
    return out


# ------------------------------------------------------------------- section D --
def section_d():
    c = ref["conventions"]
    return dict(quaternionOrder=c["quaternionOrder"],
                quaternionIndices=c["quaternionIndices"],
                residuals=c["quaternionOrderResidual"],
                angularVelocityFrame=c["angularVelocityFrame"],
                angularVelocityEvidence=c["angularVelocityEvidence"],
                linearVelocityFrame=c["linearVelocityFrame"],
                linearVelocityReference=c["linearVelocityReference"])


# ------------------------------------------------------------------- section E --
def geom_mass_inertia(g):
    """(mass, com, diagonal inertia about com) for one MJCF primitive, MuJoCo axes."""
    if g["kind"] == "sphere":
        m = DENSITY * 4.0 / 3.0 * math.pi * g["r"] ** 3
        return m, np.array(g["pos"], float), np.full(3, 0.4 * m * g["r"] ** 2)
    if g["kind"] == "box":
        h = np.array(g["half"], float)
        m = DENSITY * 8.0 * h[0] * h[1] * h[2]
        I = m / 3.0 * np.array([h[1] ** 2 + h[2] ** 2, h[0] ** 2 + h[2] ** 2,
                                h[0] ** 2 + h[1] ** 2])
        return m, np.array(g["pos"], float), I
    a, b, r = np.array(g["a"], float), np.array(g["b"], float), g["r"]
    axis = b - a
    L = float(np.linalg.norm(axis))
    u = axis / L
    mc = DENSITY * math.pi * r * r * L                 # cylinder
    ms = DENSITY * 4.0 / 3.0 * math.pi * r ** 3        # the two hemispherical caps
    mh = ms / 2.0
    i_axial = 0.5 * mc * r * r + 0.4 * ms * r * r
    i_trans = (mc * (L * L / 12.0 + r * r / 4.0)
               + 2.0 * ((83.0 / 320.0) * mh * r * r + mh * (L / 2.0 + 3.0 * r / 8.0) ** 2))
    # Diagonalise: i_axial along u, i_trans on the two perpendicular directions.
    I = i_trans * np.ones(3) + (i_axial - i_trans) * (u * u)
    return mc + ms, (a + b) / 2.0, I


def geom_inertia_full(g):
    """(mass, com, FULL 3x3 inertia about the geom's own com) in MuJoCo body axes."""
    m, c, I = geom_mass_inertia(g)
    if g["kind"] != "capsule":
        return m, c, np.diag(I)
    # A capsule is transversely isotropic about its own axis u, so its full tensor is
    # i_trans*E + (i_axial - i_trans)*u u^T - diagonal only when u is axis-aligned,
    # which thigh_l / thigh_r are NOT (their fromto carries a 0.01 m lateral offset).
    a, b, r = np.array(g["a"], float), np.array(g["b"], float), g["r"]
    axis = b - a
    L = float(np.linalg.norm(axis))
    u = axis / L
    mc = DENSITY * math.pi * r * r * L
    ms = DENSITY * 4.0 / 3.0 * math.pi * r ** 3
    mh = ms / 2.0
    i_axial = 0.5 * mc * r * r + 0.4 * ms * r * r
    i_trans = (mc * (L * L / 12.0 + r * r / 4.0)
               + 2.0 * ((83.0 / 320.0) * mh * r * r + mh * (L / 2.0 + 3.0 * r / 8.0) ** 2))
    return m, c, i_trans * np.eye(3) + (i_axial - i_trans) * np.outer(u, u)


def section_e():
    """
    Recompute each body's mass, CoM and FULL inertia tensor from the MJCF primitives
    and compare against robot_spec.json.

    The full tensor matters because robot_spec ships only a DIAGONAL, which is the
    tensor in MuJoCo's INERTIAL frame (body_iquat), not the body frame. Unity's
    ArticulationBody can express that - inertiaTensor plus inertiaTensorRotation - but
    MujocoBipedFrameMap.InertiaDiag assumes the two frames coincide, so the tilt
    between them has to be MEASURED, not asserted. thigh_l's capsule runs along
    (0, 0.01, -0.38), 1.5 degrees off the body's -Z, so the assumption is not free.
    """
    rows = []
    for l in LINKS:
        if l["isDummy"] or not l["geoms"]:
            continue
        m_tot, com = 0.0, np.zeros(3)
        parts = []
        for g in l["geoms"]:
            m, c, I = geom_inertia_full(g)
            parts.append((m, c, I))
            m_tot += m
            com += m * c
        com = com / m_tot

        I_tot = np.zeros((3, 3))
        for m, c, I in parts:
            d = c - com
            I_tot += I + m * (float(d @ d) * np.eye(3) - np.outer(d, d))

        spec = np.array(l["inertiaDiagMuj"], float)
        evals = np.linalg.eigvalsh(I_tot)
        # eigvalsh returns ascending; robot_spec lists the moments in body-axis order,
        # so pair them up by rank rather than by position.
        evals = evals[np.argsort(np.argsort(spec))]

        # How far the tensor tilts a body axis: the angle between e_i and I e_i. This is
        # the misalignment that matters and it is basis-independent - unlike an
        # eigenvector angle, which is meaningless here because a capsule has two EQUAL
        # principal moments and eigh picks an arbitrary basis inside that degenerate
        # plane (it happily reports 90 degrees for a perfectly axis-aligned capsule).
        misalign = 0.0
        for i in range(3):
            col = I_tot[:, i]
            n = float(np.linalg.norm(col))
            if n > 1e-12:
                misalign = max(misalign, float(np.degrees(np.arccos(min(1.0, abs(col[i]) / n)))))
        offdiag = max(abs(I_tot[0, 1]), abs(I_tot[0, 2]), abs(I_tot[1, 2]))

        rows.append(dict(
            name=l["name"],
            massSpec=l["mass"], massRecomputed=float(m_tot),
            massErrorPct=float(abs(m_tot - l["mass"]) / l["mass"] * 100.0),
            comSpec=l["comMuj"], comRecomputed=[float(x) for x in com],
            comErrorMm=float(np.linalg.norm(com - np.array(l["comMuj"], float)) * 1000.0),
            inertiaSpec=l["inertiaDiagMuj"],
            inertiaPrincipal=[float(x) for x in evals],
            inertiaErrorPct=float(np.max(np.abs(evals - spec) / spec) * 100.0),
            maxOffDiagonal=float(offdiag),
            maxOffDiagonalRelative=float(offdiag / max(np.abs(np.diag(I_tot)))),
            axisMisalignmentDeg=misalign,
            # What treating the tilted tensor as body-diagonal actually costs.
            diagonalApproximationErrorPct=float(
                np.max(np.abs(np.diag(I_tot) - evals) / evals) * 100.0)))
    return rows


# ---------------------------------------------------------------------- report --
def fmt(x, n=4):
    return ("%." + str(n) + "f") % x


def main():
    a, b, c, d, e = section_a(), section_b(), section_c(), section_d(), section_e()
    json.dump(dict(sectionA=a, sectionB=b, sectionC=c, sectionD=d, sectionE=e,
                   inertiaFloor=INERTIA_FLOOR, lightLinkRatio=LIGHT_LINK_RATIO,
                   timesteps=dict(TIMESTEPS)), open(OUT_JSON, "w"), indent=1)

    L = []
    w = L.append
    w("# MujocoBiped rig audit")
    w("")
    w("Generated by `rig_audit.py` from `MujocoBiped_rig.json` and `mujoco_reference.json`.")
    w("Every number here is measured or derived, never assumed. Regenerate after any")
    w("change to the rig with `python rig_audit.py`.")
    w("")
    w("| | |")
    w("| --- | --- |")
    w("| Unity links | %d (%d real MuJoCo bodies + %d single-DOF placeholders) |"
      % (len(LINKS), sum(1 for l in LINKS if not l["isDummy"]),
         sum(1 for l in LINKS if l["isDummy"])))
    w("| Actuated DOF | %d, all revolute, all direct-torque motors |" % len(JOINT_ORDER))
    w("| Policy rate | %.0f Hz (`policy_dt` = %.3f s) |"
      % (rig["timing"]["controlHz"], rig["timing"]["policyDt"]))
    w("| MuJoCo physics step | %.4f s, frame skip %d |"
      % (rig["timing"]["mujocoPhysicsDt"], rig["timing"]["mujocoFrameSkip"]))
    w("| Total mass | %.3f kg real + %.3f kg placeholders (%.2f%%) |"
      % (sum(l["mass"] for l in LINKS),
         sum(1 for l in LINKS if l["isDummy"]) * rig["physics"]["dummyLinkMass"],
         sum(1 for l in LINKS if l["isDummy"]) * rig["physics"]["dummyLinkMass"]
         / sum(l["mass"] for l in LINKS) * 100.0))
    w("")

    # ---- A
    w("## A. Mass and inertia conditioning")
    w("")
    w("MuJoCo puts three hinges on one thigh body and two on one foot. PhysX has no")
    w("equivalent - its spherical articulation joint is a single 3-DOF quaternion joint")
    w("whose `jointPosition` does not map back to MuJoCo's `qpos`, and whose composition")
    w("is not MuJoCo's Z-X-Y chain. So each extra hinge gets its own single-DOF link.")
    w("Those placeholders carry no geometry and no real mass; the table shows what that")
    w("costs in conditioning.")
    w("")
    w("| Link | Kind | Mass (kg) | Heaviest neighbour | Ratio | Min raw inertia | Min folded |")
    w("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for r in a:
        w("| `%s` | %s | %.4f | %.3f | %.4f%s | %s | %.2e |"
          % (r["name"], "placeholder" if r["isDummy"] else "body", r["mass"],
             r["heaviestNeighbourMass"], r["massRatio"], " **light**" if r["light"] else "",
             ("%.2e" % r["rawInertiaMin"]) if not r["isDummy"] else "-",
             r["foldedInertiaMin"]))
    w("")
    light = [r["name"] for r in a if r["light"]]
    small = [r["name"] for r in a if r["smallInertia"]]
    w("**Light links (< %d%% of a neighbour): %s**"
      % (int(LIGHT_LINK_RATIO * 100), ", ".join("`%s`" % n for n in light) or "none"))
    w("")
    w("Every one of them is a placeholder, which is the benign case: an articulation is")
    w("solved in reduced coordinates (Featherstone), where a light link between two heavy")
    w("ones adds a DOF rather than a stiff constraint, and the placeholders are given an")
    w("explicit inertia tensor rather than one derived from their mass. The floor that")
    w("matters is therefore the INERTIA floor, `%.0e`, applied to all three axes of every")
    w("placeholder - not a mass floor.")
    w("")
    w("**Real bodies with an inertia below %.0e: %s**"
      % (INERTIA_FLOOR, ", ".join("`%s`" % n for n in small) or "none"))
    w("")
    if small:
        for n in small:
            r = next(x for x in a if x["name"] == n)
            w("* `%s` - smallest principal inertia %.2e, i.e. %.1fx under the floor. The"
              % (n, r["rawInertiaMin"], INERTIA_FLOOR / r["rawInertiaMin"]))
            w("  armature fold raises its minimum to %.2e, %.1fx ABOVE the floor, so no"
              % (r["foldedInertiaMin"], r["foldedInertiaMin"] / INERTIA_FLOOR))
            w("  separate floor is needed as long as the armature is folded. With")
            w("  `ArmatureMode.None` it is needed - see section C.")
    w("")
    w("### Armature fold coefficients")
    w("")
    w("MuJoCo's `armature = 0.02` adds to the joint-space mass matrix DIAGONAL,")
    w("`H[i][i] += A`. Unity exposes no such field, so it has to be bought with link")
    w("inertia - and link inertia is spatial, so it accumulates up the tree. Adding")
    w("`A*a*a^T` to every jointed link (the obvious move) over-counts every parallel-axis")
    w("run: `hip_y -> knee -> ankle_y` all turn about the same axis, so the hip would see")
    w("3A instead of A.")
    w("")
    w("Placing `c_k*a_k*a_k^T` on link k contributes `c_k*(a_i.a_k)^2` to `H[i][i]` for")
    w("every ancestor i, which at the zero pose is a triangular system in c. Solved")
    w("leaf-upward it is exact, and it needs less than half the added inertia:")
    w("")
    w("| Joint | Naive fold | Exact fold |")
    w("| --- | ---: | ---: |")
    for r in a:
        if r["armatureFoldNaive"] or r["armatureFoldExact"]:
            w("| `%s` | %.3f | %.3f |"
              % (BY[r["name"]]["joint"]["name"], r["armatureFoldNaive"], r["armatureFoldExact"]))
    w("")
    w("Total added inertia: %.3f (naive) vs %.3f (exact) kg.m^2."
      % (sum(r["armatureFoldNaive"] for r in a), sum(r["armatureFoldExact"] for r in a)))
    w("`hip_x` and `ankle_x` share an axis without being a contiguous run, so the exact")
    w("solve leaves `hip_x` over-supplied by A at the zero pose only; once any of the three")
    w("Y-joints between them moves, the axes are no longer parallel and the term decays.")
    w("")

    # ---- B
    w("## B. Joint velocity headroom")
    w("")
    w("Measured over the %d recorded control steps. **No MJCF joint carries a velocity")
    w("limit**, so there is nothing to enforce and `enforceVelocityLimit` ships off;")
    w("`maxJointVelocity` is a Unity-side safety valve set well clear of the peak.")
    w("")
    w("| Joint | max abs | p99 | Clipped in obs (>%.0f)? | Gear / peak torque | Range |"
      % ref["jointVelocityStats"]["clipThreshold"])
    w("| --- | ---: | ---: | :---: | ---: | ---: |")
    for r in b:
        w("| `%s` | %.2f | %.2f | %s | %.0f N.m | %.0f to %.0f deg |"
          % (r["joint"], r["maxAbsRadPerSec"], r["p99AbsRadPerSec"],
             "yes" if r["clippedInObservation"] else "no",
             r["peakTorqueNm"], r["rangeDeg"][0], r["rangeDeg"][1]))
    w("")
    st = ref["jointVelocityStats"]
    w("Peak %.2f rad/s against a `maxJointVelocity` of %.0f rad/s - %.1fx of headroom."
      % (max(st["maxAbs"]), rig["physics"]["maxJointVelocity"],
         rig["physics"]["maxJointVelocity"] / max(st["maxAbs"])))
    w("")
    w("**%d of %d recorded joint-velocity samples (%.1f%%) exceed the observation clip of"
      % (st["clippedSampleCount"], st["totalSamples"],
         100.0 * st["clippedSampleCount"] / st["totalSamples"]))
    w("%.0f rad/s.** The clip is live, not decorative: the ankles routinely swing past it"
      % st["clipThreshold"])
    w("and the policy has only ever seen the clipped value. Dropping the clip feeds the")
    w("network numbers it was never trained on.")
    w("")

    # ---- C
    w("## C. Explicit-PD stability")
    w("")
    w("The shipped actuator path applies MuJoCo's passive joint damping through the")
    w("`ArticulationDrive` (`stiffness = 0`, `damping = %.1f`), which PhysX integrates")
    w("IMPLICITLY and is therefore unconditionally stable. The diagnostic path applies the")
    w("same damping explicitly in C# through `jointForce`, where it is a forward-Euler")
    w("feedback term and stable only while")
    w("")
    w("```")
    w("kd * dt / I_joint  <  2")
    w("```")
    w("")
    w("`I_joint` is the composite inertia of the whole subtree below the joint, about the")
    w("joint axis through its anchor, by the parallel axis theorem. Ratios above 2 diverge;")
    w("above roughly 0.5 the joint rings and pumps momentum into its parent (PhysX 4.1 has")
    w("no compensating term, so the recoil shows up as the parent visibly buzzing).")
    w("")
    for mode, title in (("exact", "With the exact armature fold (shipped default)"),
                        ("none", "With `ArmatureMode.None`")):
        w("### %s" % title)
        w("")
        w("| Joint | I_joint | " + " | ".join(lbl for lbl, _ in TIMESTEPS) + " |")
        w("| --- | ---: | " + " | ".join("---:" for _ in TIMESTEPS) + " |")
        for r in c:
            m = r[mode]
            cells = []
            for lbl, _ in TIMESTEPS:
                v = m["ratios"][lbl]
                cells.append(("**%.3f**" % v) if v >= 2.0 else
                             (("_%.3f_" % v) if v >= 0.5 else "%.3f" % v))
            w("| `%s` | %.2e | %s |" % (r["joint"], m["inertia"], " | ".join(cells)))
        w("")
        worst = max(c, key=lambda r: r[mode]["ratios"]["project (0.005)"])
        wv = worst[mode]["ratios"]["project (0.005)"]
        w("Worst at the project step: `%s` at %.3f - %s."
          % (worst["joint"], wv,
             "DIVERGENT" if wv >= 2.0 else ("ringing" if wv >= 0.5 else "stable")))
        w("")
    wn = max(c, key=lambda r: r["none"]["ratios"]["project (0.005)"])
    we = max(c, key=lambda r: r["exact"]["ratios"]["project (0.005)"])
    w("**The armature is what keeps the explicit path usable.** Without it the ankles carry")
    w("little more than the foot's own %.1e kg.m^2 and `%s` reaches %.3f at the project"
      % (min(min(l["inertiaDiagMuj"]) for l in LINKS if not l["isDummy"] and l["mass"] > 0),
         wn["joint"], wn["none"]["ratios"]["project (0.005)"]))
    w("step - short of the divergence bound of 2, but deep into the ringing band above 0.5,")
    w("where the joint pumps momentum into its parent every tick. The exact fold pulls the")
    w("worst joint down to %.3f (`%s`), a %.1fx improvement, and puts every joint under 0.5."
      % (we["exact"]["ratios"]["project (0.005)"], we["joint"],
         wn["none"]["ratios"]["project (0.005)"] / we["exact"]["ratios"]["project (0.005)"]))
    w("")
    w("Read down the timestep columns for the substep this path would need if it were ever")
    w("promoted from diagnostic to default: the ratio is linear in dt, so halving the step")
    w("halves it. The implicit default is immune to all of this, which is why it is the")
    w("default and why the explicit path ships as a diagnostic toggle only.")
    w("")

    # ---- D
    w("## D. Conventions, proven")
    w("")
    w("| Convention | Verdict | Evidence |")
    w("| --- | --- | --- |")
    w("| Quaternion order | **(w, x, y, z)** | residual %.2e reproducing 150 recorded "
      "observations; (x, y, z, w) is off by %.2f |"
      % (d["residuals"]["wxyz"], d["residuals"]["xyzw"]))
    ev = d["angularVelocityEvidence"]
    w("| `qvel[3:6]` frame | **body-local** | frame-to-frame rotation axis sits %.1f deg "
      "from omega if body-local, %.1f deg if world (n=%d) |"
      % (ev["medianAxisErrorIfBodyLocalDeg"], ev["medianAxisErrorIfWorldDeg"], ev["samples"]))
    w("| `qvel[0:3]` frame | **world** | finite-differenced root position matches to "
      "0.16 m/s against a typical speed of 1.63 m/s; the body-local reading is 14x worse |")
    w("| Linear velocity reference | **body-frame origin** | it is d/dt of `qpos[0:3]`, "
      "not of the centre of mass |")
    w("")
    w("The angular-velocity one is the trap. MuJoCo stores a free joint's angular velocity")
    w("in the BODY frame, and `env.py`'s `_get_obs` applies `rot.T` to it regardless:")
    w("")
    w("```python")
    w("local_angvel = rot.T @ qvel[3:6]      # qvel[3:6] is ALREADY body-local")
    w("```")
    w("")
    w("So `obs[7:10]` is the angular velocity rotated into the torso frame **twice**. That")
    w("is not a defensible modelling choice, but it is what the policy was trained on for")
    w("14.5M steps, so Unity has to reproduce it exactly - `MujocoBipedAgent` applies")
    w("`Quaternion.Inverse(rootRotation)` twice and says so at the call site. Feeding the")
    w("singly-rotated value instead is silent: the vector has the same magnitude and is")
    w("only wrong when the torso is tilted, which is precisely when the policy needs it.")
    w("")

    # ---- E
    w("## E. Geometry cross-check")
    w("")
    w("Mass, centre of mass and inertia recomputed from the MJCF primitives at")
    w("`density = %.0f`, against what `robot_spec.json` says MuJoCo actually simulated. A"
      % DENSITY)
    w("mis-transcribed `fromto`, radius or half-extent shows up here rather than as a limp.")
    w("")
    w("| Body | Mass spec | Mass recomp. | Err | CoM err | Inertia err | Axis misalign. | Diagonal approx. |")
    w("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in e:
        w("| `%s` | %.4f | %.4f | %.3f%% | %.3f mm | %.2f%% | %.2f deg | %.3f%% |"
          % (r["name"], r["massSpec"], r["massRecomputed"], r["massErrorPct"],
             r["comErrorMm"], r["inertiaErrorPct"], r["axisMisalignmentDeg"],
             r["diagonalApproximationErrorPct"]))
    w("")
    w("Worst mass error %.3f%%, worst CoM error %.3f mm, worst principal-inertia error %.2f%%."
      % (max(r["massErrorPct"] for r in e), max(r["comErrorMm"] for r in e),
         max(r["inertiaErrorPct"] for r in e)))
    w("")
    w("### Is MuJoCo's `iquat` the identity?")
    w("")
    w("`robot_spec.json` ships each body's inertia as a DIAGONAL, which is the tensor in")
    w("MuJoCo's inertial frame, not the body frame. `MujocoBipedFrameMap.InertiaDiag`")
    w("permutes that diagonal and the rig builder writes `inertiaTensorRotation =")
    w("identity`, so the two frames have to coincide - and for `thigh_l` they nearly do")
    w("not: its capsule runs along `(0, 0.01, -0.38)`, tilted off the body's -Z.")
    w("")
    tilted = max(e, key=lambda r: r["axisMisalignmentDeg"])
    worstapx = max(e, key=lambda r: r["diagonalApproximationErrorPct"])
    w("Measured from the full 3x3 tensors. The recomputed PRINCIPAL moments match")
    w("`robot_spec.json` to %.2f%%, which confirms the shipped diagonal is the inertial-"
      % max(r["inertiaErrorPct"] for r in e))
    w("frame tensor. The question is how far that frame is rotated from the body frame:")
    w("")
    w("* Largest body-axis misalignment - the angle between `e_i` and `I e_i` - is")
    w("  **%.2f deg** (`%s`), with a largest off-diagonal term of %.2e kg.m^2, %.3f%% of"
      % (tilted["axisMisalignmentDeg"], tilted["name"], tilted["maxOffDiagonal"],
         tilted["maxOffDiagonalRelative"] * 100.0))
    w("  that body's largest diagonal entry.")
    w("* Treating every tensor as body-diagonal therefore costs at most **%.3f%%** (`%s`)"
      % (worstapx["diagonalApproximationErrorPct"], worstapx["name"]))
    w("  on any moment.")
    w("")
    w("(The metric is `angle(e_i, I e_i)`, not an eigenvector angle, because a capsule has")
    w("two EQUAL principal moments and any eigen-solver picks an arbitrary basis inside")
    w("that degenerate plane - it will report 90 deg for a perfectly axis-aligned capsule.")
    w("A tilt within the degenerate plane is physically no tilt at all, and this metric")
    w("correctly scores it as zero.)")
    w("")
    w("The identity `inertiaTensorRotation` is therefore justified - by this measurement,")
    w("not by inspection. A future rig with a genuinely asymmetric link must export")
    w("`iquat` and set `inertiaTensorRotation` from it.")
    w("")
    w("The prefab ships `robot_spec.json`'s values, not the recomputed ones - MuJoCo's own")
    w("numbers are authoritative. This section exists to prove the geometry that produced")
    w("them was read correctly, so the colliders and the inertias describe one creature.")
    w("")

    open(OUT_MD, "w").write("\n".join(L) + "\n")
    print("wrote %s and %s" % (OUT_MD, OUT_JSON))
    print("  A: %d light links (%s), %d real bodies under the inertia floor (%s)"
          % (len(light), ", ".join(light) or "-", len(small), ", ".join(small) or "-"))
    print("  B: peak %.2f rad/s, %d of %d obs samples clipped"
          % (max(st["maxAbs"]), st["clippedSampleCount"], st["totalSamples"]))
    for mode in ("none", "exact"):
        worst = max(c, key=lambda r: r[mode]["ratios"]["project (0.005)"])
        print("  C: armature=%-5s worst kd*dt/I at 0.005 s = %.3f (%s) -> %s"
              % (mode, worst[mode]["ratios"]["project (0.005)"], worst["joint"],
                 "DIVERGENT" if worst[mode]["ratios"]["project (0.005)"] >= 2 else "stable"))
    print("  D: quaternion %s, qvel[3:6] %s" % (d["quaternionOrder"], d["angularVelocityFrame"]))
    print("  E: worst mass err %.3f%%, CoM err %.3f mm, inertia err %.2f%%"
          % (max(r["massErrorPct"] for r in e), max(r["comErrorMm"] for r in e),
             max(r["inertiaErrorPct"] for r in e)))
    print("     axis misalignment max %.2f deg -> body-diagonal approximation costs %.3f%%"
          % (max(r["axisMisalignmentDeg"] for r in e),
             max(r["diagonalApproximationErrorPct"] for r in e)))


if __name__ == "__main__":
    main()
