"""Builds the one quadruped both trainers use: quad.xml (MJCF) and quad_rig.json.

Contract: training/quad/QUAD_SPEC.md. The body is training/bugs/Quad_v01.xml (a
straight conversion of Assets/Prefabs/Quad_v01.prefab) with exactly the spec's
changes and nothing else:

  * force limit 300 N*m on every actuator (the prefab has 900);
  * kp 1500 N*m/rad and joint damping 100 N*m*s/rad (the prefab has 4500 / 300), scaled
    down by 3 with the force limit so the servo is proportional, not an on/off switch;
  * soft ground (round 9): the FLOOR geom gets solref "0.03 1" (a paw-pad-like contact, 3x the
    rigid 0.01) and priority 1, so every floor contact uses the floor's solref/solimp/friction.
    All body geoms keep solref 0.01 / priority 0, so body-body contacts (leg on leg) stay rigid.
    (Soft FEET - the lower legs at priority 1 - were tried first and rejected: two soft feet
    are soft against each other too, and lower legs overlapped 40-65 mm under random actions.)
    Because the floor supplies the friction of every floor contact, the per-episode friction
    randomisation acts on the floor geom only (see QUAD_SPEC round 9).
  * friction 0.9 on every geom, floor included, no bounce (the bugs used Unity's 0.6);
  * solver iterations written out explicitly (Newton, 10 / 8 line-search), the same
    <option> as training/worm/worm.xml, so both trainers configure the MuJoCo-Warp
    solver from the file instead of from defaults;
  * keyframe "spawn" = the spec reset height (rest 0.90 m + 2 cm).

Armature (0), joint ranges (hip +-45 deg, knee
+-60 deg), masses (90 kg), shapes, timestep (0.005 s), integrator (implicitfast)
and cone (pyramidal) are copied unchanged from Quad_v01.xml. Visual-only extras
of the bugs file (texture, material, light, camera, site) are dropped so every
importer (MuJoCo, MJWarp, Newton) sees the same plain model.

The MuJoCo trainer loads quad.xml; the Isaac Lab 3 task (Newton + MuJoCo-Warp)
builds from quad.xml / quad_rig.json; Unity rebuilds the body from quad_rig.json.

  .venv-mjwarp\\Scripts\\python.exe training/quad/build_quad.py
"""
from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "training" / "bugs" / "Quad_v01.xml"
SOURCE_RIG = REPO / "training" / "bugs" / "Quad_v01_rig.json"
OUT_XML = HERE / "quad.xml"
OUT_RIG = HERE / "quad_rig.json"

FORCE_LIMIT = 300.0            # N*m, QUAD_SPEC (prefab: 900)
KP = 1500.0                    # N*m/rad, QUAD_SPEC (prefab: 4500); full force at 11 deg error
JOINT_DAMPING = 100.0          # N*m*s/rad, QUAD_SPEC (prefab: 300)
FLOOR_SOLREF = (0.03, 1.0)     # round 9: soft ground (time constant s, damping ratio)
FLOOR_PRIORITY = 1             # the floor's contact parameters win in every floor contact
FRICTION = (0.9, 0.005, 0.0001)  # sliding, torsional, rolling; AGENTS 2E; bounce 0
SOLVER = {"solver": "Newton", "iterations": "10", "ls_iterations": "8"}  # as worm.xml
REST_HEIGHT = 0.90             # root (torso centre) height with the feet on the floor
SPAWN_CLEARANCE = 0.02         # QUAD_SPEC reset: rest + 2 cm
ACTION_SCALE = 0.785398        # rad per unit action (45 deg), QUAD_SPEC
LEG_LENGTH = 0.90              # hip-free stance height used for the Froude speed
FROUDE = 0.25
GRAVITY = 9.81
# QUAD_SPEC's number, used verbatim by both trainers: sqrt(0.25 * 9.81 * 0.90) = 1.486, rounded
# to 1.49 as the spec prints it. Observed as TARGET_SPEED / 2 = 0.745.
TARGET_SPEED = round(math.sqrt(FROUDE * GRAVITY * LEG_LENGTH), 2)   # 1.49 m/s
DECIMATION = 10               # policy 20 Hz at dt 0.005 (QUAD_SPEC round 3; was 4 = 50 Hz)
FALL_UP_DOT = 0.5
FALL_HEIGHT = 0.45


def fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in values)


def build_xml() -> str:
    tree = ET.parse(SOURCE)
    root = tree.getroot()
    root.set("model", "quad")

    option = root.find("option")
    for key, value in SOLVER.items():
        option.set(key, value)

    # Visual-only elements: not part of the body, not understood by every importer.
    for tag in ("size", "visual", "asset"):
        for element in root.findall(tag):
            root.remove(element)

    friction = fmt(FRICTION)
    default = root.find("default")
    default.find("geom").set("friction", friction)
    default.find("geom").set("rgba", "0.6 0.6 0.6 1")    # neutral grey: rule D reserves red/green
    # Every geom, floor included (MuJoCo combines by max, so all must be 0.9).
    for geom in root.iter("geom"):
        if "friction" in geom.attrib:
            geom.set("friction", friction)
        geom.attrib.pop("material", None)

    worldbody = root.find("worldbody")
    for element in list(worldbody):
        if element.tag == "light":
            worldbody.remove(element)
    floor = worldbody.find("geom[@name='floor']")
    floor.set("rgba", "0.3 0.32 0.35 1")
    for body in worldbody.iter("body"):
        for element in list(body):
            if element.tag in ("site", "camera"):
                body.remove(element)

    for actuator in root.find("actuator"):
        actuator.set("forcerange", f"{-FORCE_LIMIT:g} {FORCE_LIMIT:g}")
        actuator.set("kp", f"{KP:g}")
    for joint in worldbody.iter("joint"):
        if "damping" in joint.attrib:
            joint.set("damping", f"{JOINT_DAMPING:g}")
    floor.set("solref", fmt(FLOOR_SOLREF))
    floor.set("priority", str(FLOOR_PRIORITY))

    keyframe = root.find("keyframe")
    for key in keyframe.findall("key"):
        qpos = key.get("qpos").split()
        qpos[2] = f"{REST_HEIGHT + SPAWN_CLEARANCE if key.get('name') == 'spawn' else REST_HEIGHT:g}"
        key.set("qpos", " ".join(qpos))

    ET.indent(tree, space="  ")
    header = (
        "<!-- quad: generated by training/quad/build_quad.py from training/bugs/Quad_v01.xml - do not hand-edit.\n"
        "     Contract: training/quad/QUAD_SPEC.md. Changes from Quad_v01.xml: forcerange +-300 N*m (was 900),\n"
        "     friction 0.9 on every geom incl. the floor (was 0.6), solver Newton 10/8 written out (as worm.xml),\n"
        "     kp 1500 (was 4500), joint damping 100 (was 300), soft ground (floor solref 0.03 1, priority 1),\n"
        "     spawn keyframe at 0.92 m, visual-only extras\n"
        "     dropped. Armature, ranges, masses, shapes, timestep unchanged.\n"
        "     Frames: MuJoCo x forward, y left, z up. Unity (x,y,z) = MuJoCo (-y, z, x); see quad_rig.json. -->\n"
    )
    return header + ET.tostring(root, encoding="unicode") + "\n"


# ---------------------------------------------------------------- MuJoCo -> Unity
def pos_to_unity(p):
    return [-float(p[1]), float(p[2]), float(p[0])]


def axis_to_unity(a):
    # inverse of Unity a -> MuJoCo (-a.z, a.x, -a.y)
    return [float(a[1]), -float(a[2]), -float(a[0])]


def quat_to_unity(q):
    # MuJoCo (w, x, y, z) -> Unity (x, y, z, w); inverse of Unity (x,y,z,w) -> MuJoCo (w,-z,x,-y)
    w, a, b, c = (float(v) for v in q)
    return [b, -c, -a, w]


def box_to_unity(h):
    return [float(h[1]), float(h[2]), float(h[0])]


def clean(values, digits=9):
    return [round(float(v), digits) + 0.0 for v in values]


def build_rig(xml_text: str, mjm: mujoco.MjModel) -> dict:
    src_rig = json.loads(SOURCE_RIG.read_text())
    action_order = list(src_rig["actuator_order"])
    root = ET.fromstring(xml_text)

    # fromto of capsules as authored (the compiled model keeps pos/quat/size only).
    fromto = {g.get("name"): [float(v) for v in g.get("fromto").split()]
              for g in root.iter("geom") if g.get("fromto")}

    def name(obj, i):
        return mujoco.mj_id2name(mjm, obj, i)

    bodies = []
    for b in range(1, mjm.nbody):
        parent = int(mjm.body_parentid[b])
        bodies.append({
            "name": name(mujoco.mjtObj.mjOBJ_BODY, b),
            "parent": None if parent == 0 else name(mujoco.mjtObj.mjOBJ_BODY, parent),
            "pos": clean(mjm.body_pos[b]),
            "quat": clean(mjm.body_quat[b]),
            "mass": float(mjm.body_mass[b]),
            "inertiaDiag": clean(mjm.body_inertia[b]),
            "ipos": clean(mjm.body_ipos[b]),
            "iquat": clean(mjm.body_iquat[b]),
            "freejoint": bool(mjm.body_jntnum[b] and mjm.jnt_type[mjm.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_FREE),
            "unity": {"pos": clean(pos_to_unity(mjm.body_pos[b])),
                      "rot": clean(quat_to_unity(mjm.body_quat[b])),
                      "centerOfMass": clean(pos_to_unity(mjm.body_ipos[b]))},
        })

    geom_type = {int(mujoco.mjtGeom.mjGEOM_PLANE): "plane", int(mujoco.mjtGeom.mjGEOM_BOX): "box",
                 int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
                 int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere"}
    geoms = []
    for g in range(mjm.ngeom):
        gname = name(mujoco.mjtObj.mjOBJ_GEOM, g)
        gtype = geom_type[int(mjm.geom_type[g])]
        entry = {
            "name": gname,
            "body": name(mujoco.mjtObj.mjOBJ_BODY, int(mjm.geom_bodyid[g])) or "world",
            "type": gtype,
            "size": clean(mjm.geom_size[g]),
            "pos": clean(mjm.geom_pos[g]),
            "quat": clean(mjm.geom_quat[g]),
            "friction": clean(mjm.geom_friction[g]),
            "condim": int(mjm.geom_condim[g]),
            "contype": int(mjm.geom_contype[g]),
            "conaffinity": int(mjm.geom_conaffinity[g]),
            "solref": clean(mjm.geom_solref[g]),
            "solimp": clean(mjm.geom_solimp[g]),
            "solmix": float(mjm.geom_solmix[g]),
            "priority": int(mjm.geom_priority[g]),
            "margin": float(mjm.geom_margin[g]),
        }
        unity = {"center": clean(pos_to_unity(mjm.geom_pos[g]))}
        if gtype == "capsule":
            entry["fromto"] = clean(fromto[gname])
            entry["radius"] = float(mjm.geom_size[g][0])
            entry["halfLength"] = float(mjm.geom_size[g][1])
            a, b = np.array(fromto[gname][:3]), np.array(fromto[gname][3:])
            axis = (b - a) / np.linalg.norm(b - a)
            unity.update({"radius": entry["radius"],
                          "height": 2 * (entry["halfLength"] + entry["radius"]),
                          "direction": int(np.argmax(np.abs(pos_to_unity(axis)))),
                          "directionNote": "CapsuleCollider.direction: 0 = X, 1 = Y, 2 = Z"})
        elif gtype == "box":
            unity["halfExtents"] = clean(box_to_unity(mjm.geom_size[g]))
        entry["unity"] = unity
        geoms.append(entry)

    joints = []
    for j in range(mjm.njnt):
        if mjm.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        dof = int(mjm.jnt_dofadr[j])
        joints.append({
            "name": name(mujoco.mjtObj.mjOBJ_JOINT, j),
            "body": name(mujoco.mjtObj.mjOBJ_BODY, int(mjm.jnt_bodyid[j])),
            "type": "hinge",
            "axis": clean(mjm.jnt_axis[j]),
            "pos": clean(mjm.jnt_pos[j]),
            "range": clean(mjm.jnt_range[j]),
            "rangeDeg": clean(np.degrees(mjm.jnt_range[j]), 4),
            "damping": float(mjm.dof_damping[dof]),
            "armature": float(mjm.dof_armature[dof]),
            "frictionloss": float(mjm.dof_frictionloss[dof]),
            "solreflimit": clean(mjm.jnt_solref[j]),
            "unity": {"axis": clean(axis_to_unity(mjm.jnt_axis[j])),
                      "anchor": clean(pos_to_unity(mjm.jnt_pos[j])),
                      "angleSign": "Unity jointPosition (rad) == MuJoCo qpos, same sign"},
        })

    actuators = []
    for i, act_name in enumerate(action_order):
        a = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        assert a == i, f"actuator {act_name} is #{a}, expected action index {i}"
        jid = int(mjm.actuator_trnid[a, 0])
        actuators.append({
            "index": i,
            "name": act_name,
            "joint": name(mujoco.mjtObj.mjOBJ_JOINT, jid),
            "type": "position",
            "kp": float(mjm.actuator_gainprm[a, 0]),
            "kv": 0.0,
            "ctrlrange": clean(mjm.actuator_ctrlrange[a]),
            "forcerange": clean(mjm.actuator_forcerange[a]),
            "law": "force = clip(kp * (ctrl - qpos), forcerange); joint damping acts separately "
                   "(passive, not capped)",
        })

    lower_legs = [b["name"] for b in bodies if b["name"].startswith("Lower_")]
    feet = [g["name"] for g in geoms if g["body"] in lower_legs]
    # Foot point = centre of the lower capsule's bottom end-cap, in the lower-leg frame.
    foot_points = {}
    for g in geoms:
        if g["body"] in lower_legs:
            ends = [g["fromto"][:3], g["fromto"][3:]]
            foot_points[g["body"]] = clean(min(ends, key=lambda e: e[2]))
    torso = bodies[0]["name"]

    return {
        "model": "quad",
        "source": "training/bugs/Quad_v01.xml (from Assets/Prefabs/Quad_v01.prefab)",
        "generator": "training/quad/build_quad.py",
        "mjcf": "training/quad/quad.xml",
        "spec": "training/quad/QUAD_SPEC.md",
        "changesFromSource": [
            f"actuator forcerange +-{FORCE_LIMIT:g} N*m (source 900)",
            f"actuator kp {KP:g} N*m/rad (source 4500) and joint damping {JOINT_DAMPING:g} N*m*s/rad "
            "(source 300): scaled by 1/3 with the force limit (QUAD_SPEC)",
            f"soft ground (round 9): floor geom solref {FLOOR_SOLREF[0]:g} {FLOOR_SOLREF[1]:g} (source 0.01 1) "
            f"and priority {FLOOR_PRIORITY}, so the floor's solref/solimp/friction are used in every floor "
            "contact; body geoms unchanged (0.01 1, priority 0)",
            f"friction {FRICTION[0]} on every geom incl. floor (source 0.6)",
            "solver Newton, iterations 10, ls_iterations 8 written explicitly (as worm.xml)",
            f"keyframe 'spawn' at {REST_HEIGHT + SPAWN_CLEARANCE:.2f} m (rest + 2 cm)",
            "visual-only texture/material/light/camera/site removed; body colour neutral grey (rule D reserves red/green)",
        ],
        "frames": {
            "mujoco": "right-handed, Z-up, +X forward, +Y left",
            "unity": "left-handed, Y-up, +X right, +Z forward",
            "position": "Unity (x, y, z) = MuJoCo (-y, z, x)   [MuJoCo = (z_u, -x_u, y_u)]",
            "axis": "Unity axis a -> MuJoCo (-a.z, a.x, -a.y); MuJoCo m -> Unity (m.y, -m.z, -m.x). "
                    "MuJoCo hinge axis +Y (pitch) is Unity +X",
            "quat": "Unity (x, y, z, w) -> MuJoCo (w, -z, x, -y); MuJoCo (w, a, b, c) -> Unity (b, -c, -a, w)",
            "boxHalfSizes": "Unity (sx, sy, sz) -> MuJoCo (sz, sx, sy)",
            "jointSign": "a positive Unity jointPosition / xDrive.target equals a positive MuJoCo qpos / ctrl "
                         "for the same physical motion (the axis map carries the handedness flip). Same map as "
                         "training/bugs/README.md and WORM_SPEC.md. Hip +20 deg swings the foot backward "
                         "(MuJoCo -x, Unity -z).",
            "unityBlocks": "each body/geom/joint carries a 'unity' block already converted with these maps",
        },
        "units": "SI: m, kg, s, rad, N*m. kp in N*m/rad, damping in N*m*s/rad",
        "physics": {
            "timestep": float(mjm.opt.timestep),
            "decimation": DECIMATION,
            "policyDt": float(mjm.opt.timestep) * DECIMATION,
            "gravity": clean(mjm.opt.gravity),
            "integrator": "implicitfast",
            "cone": "pyramidal",
            "solver": "Newton",
            "iterations": int(mjm.opt.iterations),
            "lsIterations": int(mjm.opt.ls_iterations),
            "tolerance": float(mjm.opt.tolerance),
            "frictionCombine": "max (MuJoCo); every geom and the floor are 0.9, so the combine rule does not matter",
            "restitution": 0.0,
            "contactParameterRule": "MuJoCo: if two geoms' priorities differ, the higher-priority geom's "
                                    "solref/solimp/friction are used; if equal, solref/solimp are averaged "
                                    "(solmix weights) and friction is the max. The floor has priority 1 and "
                                    "solref 0.03 1; every body geom priority 0 and solref 0.01 1. So every "
                                    "floor contact uses the floor's 0.03 1 / solimp / friction, and every "
                                    "body-body contact uses 0.01 1.",
            "selfCollision": "all body pairs collide except parent-child (MuJoCo filterparent default); "
                             "no <exclude> pairs",
        },
        "totalMass": float(sum(b["mass"] for b in bodies)),
        "torso": torso,
        "restRootHeight": REST_HEIGHT,
        "spawnRootHeight": REST_HEIGHT + SPAWN_CLEARANCE,
        "actionScaleRad": ACTION_SCALE,
        "actionToCtrl": "ctrl[i] = rest[i] + clip(action[i], -1, 1) * actionScaleRad; rest = 0 for every joint",
        "restPose": {j["name"]: 0.0 for j in joints},
        "actionOrder": action_order,
        "forceLimit": FORCE_LIMIT,
        "lowerLegs": lower_legs,
        "footGeoms": feet,
        "footPointsLocal": foot_points,
        "footSlip": ("sum over lower legs in ground contact of the horizontal (world x, y) speed of the "
                     "foot point (footPointsLocal, the bottom end-cap centre of the lower capsule): "
                     "v_foot = v_body + w x (R p). 'In ground contact' = MuJoCo lists a floor contact "
                     "for that lower-leg geom with dist < 0 (a force-carrying contact); an Isaac/PhysX "
                     "contact sensor equivalent is ground normal force on the lower leg > 1 N"),
        "legNames": {
            "Upper_-1_-1": "rear-left hip", "Lower_-1_-1": "rear-left knee",
            "Upper_-1_1": "front-left hip", "Lower_-1_1": "front-left knee",
            "Upper_1_-1": "rear-right hip", "Lower_1_-1": "rear-right knee",
            "Upper_1_1": "front-right hip", "Lower_1_1": "front-right knee",
        },
        "task": {
            "goal": "world +x (MuJoCo) = Unity +z",
            "targetSpeed": TARGET_SPEED,
            "targetSpeedFormula": f"sqrt({FROUDE} * {GRAVITY} * {LEG_LENGTH}) = 1.486, rounded to 1.49 as QUAD_SPEC prints it (Froude 0.25)",
            "targetSpeedObs": "target speed / 2 (obs index 35)",
            "speedKernel": "exp(-((v_x - targetSpeed) / 0.5)^2), two-sided",
            "episodeSeconds": 20.0,
            "fall": f"torso up . world up < {FALL_UP_DOT} or torso z < {FALL_HEIGHT} m",
            "obsSize": 36,
            "actionSize": 8,
        },
        "floor": {
            "geom": "floor",
            "solref": list(FLOOR_SOLREF),
            "solimp": clean(mjm.geom_solimp[0]),
            "priority": FLOOR_PRIORITY,
            "friction": list(FRICTION),
            "restitution": 0.0,
            "note": "Unity builds its own floor: give it these contact parameters. Training randomises the "
                    "floor's sliding friction per episode (0.9 x U(0.85, 1.15)); body geoms stay at 0.9.",
        },
        "contactPairs": [{"geom1": mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(mjm.pair_geom1[i])),
                          "geom2": mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_GEOM, int(mjm.pair_geom2[i])),
                          "condim": int(mjm.pair_dim[i]), "friction": clean(mjm.pair_friction[i]),
                          "solref": clean(mjm.pair_solref[i]), "solimp": clean(mjm.pair_solimp[i]),
                          "margin": float(mjm.pair_margin[i]), "gap": float(mjm.pair_gap[i])}
                         for i in range(mjm.npair)],
        "contactExcludes": [{"body1": mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(e >> 16)) or "world",
                             "body2": mujoco.mj_id2name(mjm, mujoco.mjtObj.mjOBJ_BODY, int(e & 0xFFFF)) or "world"}
                            for e in mjm.exclude_signature],
        "bodies": bodies,
        "geoms": geoms,
        "joints": joints,
        "actuators": actuators,
    }


def main() -> None:
    xml_text = build_xml()
    OUT_XML.write_text(xml_text, encoding="utf-8")
    mjm = mujoco.MjModel.from_xml_path(str(OUT_XML))
    rig = build_rig(xml_text, mjm)
    OUT_RIG.write_text(json.dumps(rig, indent=1), encoding="utf-8")
    print(f"{OUT_XML.name} + {OUT_RIG.name}: {mjm.nbody - 1} bodies, {len(rig['joints'])} hinges, "
          f"{rig['totalMass']:.1f} kg, force limit {FORCE_LIMIT:g} N*m, friction {FRICTION[0]}, "
          f"target speed {TARGET_SPEED:.3f} m/s")


if __name__ == "__main__":
    main()
