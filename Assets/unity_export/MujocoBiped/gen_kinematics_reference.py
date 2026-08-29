"""
Independent forward kinematics for model/biped.xml  ->  kinematics_reference.json

Deliberately INDEPENDENT of extract_rig.py and of MujocoBiped_rig.json: it parses
the MJCF with ElementTree and walks the body tree itself. If the Unity rig has a
transposed quaternion, a mirrored axis, a dropped body offset or a wrong joint
composition order, the PlayMode kinematics test fails against these numbers -
which it could not do if both sides came from the same intermediate file.

MuJoCo composition, from mj_kinematics: a body starts at the parent frame plus
its own pos/quat, then each joint is applied IN THE ORDER LISTED by
POST-multiplying a rotation about that joint's axis, so the axis of joint k is
expressed in the frame left behind by joints 1..k-1. A three-hinge hip is
therefore an intrinsic Z-X-Y Euler chain, not a PhysX spherical joint.

Output is in MuJoCo coordinates (right-handed, Z-up, X-forward, metres, radians).
"""
import json
import math
import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
MJCF = os.path.abspath(os.path.join(HERE, "..", "..", "biped_sentis", "model", "biped.xml"))
OUT = os.path.join(HERE, "kinematics_reference.json")

JOINT_ORDER = ["hip_z_l", "hip_x_l", "hip_y_l", "knee_l", "ankle_y_l", "ankle_x_l",
               "hip_z_r", "hip_x_r", "hip_y_r", "knee_r", "ankle_y_r", "ankle_x_r"]


def mat_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]


def mat_vec(m, v):
    return [sum(m[i][k] * v[k] for k in range(3)) for i in range(3)]


def axis_angle(axis, ang):
    """Rodrigues. Right-handed, which is MuJoCo's convention."""
    n = math.sqrt(sum(c * c for c in axis))
    x, y, z = (c / n for c in axis)
    c, s, t = math.cos(ang), math.sin(ang), 1.0 - math.cos(ang)
    return [[t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c]]


def mat_to_quat_wxyz(m):
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2][1] - m[1][2]) / s
        y = (m[0][2] - m[2][0]) / s
        z = (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w = (m[2][1] - m[1][2]) / s
        x = 0.25 * s
        y = (m[0][1] + m[1][0]) / s
        z = (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w = (m[0][2] - m[2][0]) / s
        x = (m[0][1] + m[1][0]) / s
        y = 0.25 * s
        z = (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w = (m[1][0] - m[0][1]) / s
        x = (m[0][2] + m[2][0]) / s
        y = (m[1][2] + m[2][1]) / s
        z = 0.25 * s
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    return [w, x, y, z]


def floats(text, n=None):
    v = [float(t) for t in text.split()]
    if n is not None and len(v) != n:
        raise ValueError("expected %d floats, got %r" % (n, text))
    return v


def parse():
    """(bodies, is_degrees). Each body: name, parent, pos, joints[(name, axis)]."""
    root = ET.parse(MJCF).getroot()
    compiler = root.find("compiler")
    is_deg = compiler is not None and compiler.get("angle", "degree") == "degree"
    world = root.find("worldbody")

    bodies = []

    def walk(elem, parent):
        for b in elem.findall("body"):
            if b.get("mocap") == "true":         # the goal marker is not part of the rig
                continue
            name = b.get("name")
            pos = floats(b.get("pos", "0 0 0"), 3)
            joints = [(j.get("name"), floats(j.get("axis", "0 0 1"), 3))
                      for j in b.findall("joint")]
            bodies.append(dict(name=name, parent=parent, pos=pos, joints=joints))
            walk(b, name)

    walk(world, None)
    return bodies, is_deg


def fk(bodies, q, root_pos, root_quat_wxyz):
    """q: joint name -> radians. Returns name -> {pos, quat_wxyz} in the world."""
    w, x, y, z = root_quat_wxyz
    r0 = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
          [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
          [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]

    out = {}
    for b in bodies:
        if b["parent"] is None:
            # The torso's own <body pos> is folded into the free joint, which is
            # what qpos[0:3] measures, so the caller supplies the root pose.
            pos, rot = list(root_pos), r0
        else:
            p = out[b["parent"]]
            pos = [p["pos"][i] + mat_vec(p["rot"], b["pos"])[i] for i in range(3)]
            rot = p["rot"]
        for jname, axis in b["joints"]:
            rot = mat_mul(rot, axis_angle(axis, q.get(jname, 0.0)))
        out[b["name"]] = dict(pos=pos, rot=rot)
    return out


def main():
    bodies, is_deg = parse()
    names = [b["name"] for b in bodies]
    if names[0] != "torso":
        raise SystemExit("expected 'torso' to be the articulation root, got %r" % names[0])

    # Three poses: the zero pose, a per-joint staircase that exercises every DOF
    # in isolation, and a squat-like pose where several parallel-axis joints move
    # together (the case a wrong composition order gets away with at zero).
    d = math.radians
    poses = [
        dict(label="zero", q={}),
        dict(label="staircase", q={
            "hip_z_l": d(20), "hip_x_l": d(-15), "hip_y_l": d(-40), "knee_l": d(-60),
            "ankle_y_l": d(15), "ankle_x_l": d(-10),
            "hip_z_r": d(-25), "hip_x_r": d(12), "hip_y_r": d(30), "knee_r": d(-100),
            "ankle_y_r": d(-30), "ankle_x_r": d(20)}),
        dict(label="parallel_axis_run", q={
            "hip_y_l": d(-70), "knee_l": d(-120), "ankle_y_l": d(30),
            "hip_y_r": d(-70), "knee_r": d(-120), "ankle_y_r": d(30),
            "hip_x_l": d(-8), "hip_x_r": d(8)}),
    ]

    # A non-identity root orientation on the last pose, so the test also catches a
    # root quaternion that is right up to a transpose.
    roots = [([0.0, 0.0, 0.88], [1.0, 0.0, 0.0, 0.0]),
             ([0.0, 0.0, 0.88], [1.0, 0.0, 0.0, 0.0]),
             ([1.5, -0.75, 0.83], mat_to_quat_wxyz(axis_angle([0.0, 0.0, 1.0], d(37))))]

    out = []
    for pose, (rp, rq) in zip(poses, roots):
        res = fk(bodies, pose["q"], rp, rq)
        out.append(dict(
            label=pose["label"],
            rootPosMuj=rp, rootQuatMujWxyz=rq,
            jointsRad=[pose["q"].get(n, 0.0) for n in JOINT_ORDER],
            bodies=[dict(name=n, posMuj=res[n]["pos"],
                         quatMujWxyz=mat_to_quat_wxyz(res[n]["rot"])) for n in names]))

    json.dump(dict(
        description="Independent forward kinematics of model/biped.xml. MuJoCo frame "
                    "(right-handed, Z-up, X-forward), metres, radians, quaternions (w,x,y,z).",
        source=os.path.basename(MJCF),
        mjcfAngleUnits="degree" if is_deg else "radian",
        jointOrder=JOINT_ORDER,
        bodyOrder=names,
        toleranceMetres=1e-3,
        poses=out), open(OUT, "w"), indent=1)

    print("wrote " + OUT)
    print("  %d bodies, %d poses, joint order %s" % (len(names), len(out), JOINT_ORDER[0] + "..."))
    for p in out:
        far = max(p["bodies"], key=lambda b: sum(c * c for c in b["posMuj"]))
        print("  %-10s furthest body %-8s at (%.4f, %.4f, %.4f)"
              % (p["label"], far["name"], far["posMuj"][0], far["posMuj"][1], far["posMuj"][2]))


if __name__ == "__main__":
    main()
