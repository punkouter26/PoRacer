"""Generate biped.urdf: a torso carried by two 5-DOF legs.

Per leg: hip_yaw (Z) -> hip_roll (X) -> hip_pitch (Y) -> knee (Y) -> ankle (Y),
i.e. 10 actuated joints in total. Link frames follow the usual convention: a link's
origin sits at its parent joint and the geometry hangs from there along -Z.

Run with plain python:  python biped/make_urdf.py
"""

import math
import os

# --- torso (base link): box whose bottom face sits on the hip line (the link origin) ---
TORSO_X, TORSO_Y, TORSO_Z, TORSO_M = 0.16, 0.24, 0.32, 6.0
HIP_Y = 0.09  # lateral hip offset from the mid-line [m]
HUB_R, HUB_M = 0.04, 0.15  # tiny yaw/roll hub links between the torso and the thigh

THIGH_L, THIGH_R, THIGH_M = 0.32, 0.045, 2.0
SHANK_L, SHANK_R, SHANK_M = 0.32, 0.035, 1.5
FOOT_X, FOOT_Y, FOOT_Z, FOOT_M = 0.20, 0.09, 0.04, 0.6
FOOT_AHEAD = 0.04  # foot box centre ahead of the ankle [m]: heel behind, toe in front
# collision cylinders are shorter than the segment so the capsule end-caps (the URDF
# converter turns cylinders into capsules) do not poke through the foot or the hip
SEG_SHRINK = 0.06

# nominal standing crouch [rad]; the policy acts around this pose.
# hip + knee + ankle == 0 keeps the foot sole flat on the ground.
Q_HIP_PITCH, Q_KNEE, Q_ANKLE = -0.25, 0.50, -0.25

# joint -> (lower, upper, effort [N*m], velocity [rad/s])
LIMITS = {
    "hip_yaw": (-0.6, 0.6, 50.0, 12.0),
    "hip_roll": (-0.5, 0.5, 100.0, 12.0),
    "hip_pitch": (-1.5, 1.0, 150.0, 15.0),
    "knee": (0.0, 2.2, 150.0, 15.0),  # knee bends one way only, like a human knee
    "ankle": (-0.8, 0.8, 80.0, 15.0),
}


def box_inertia(m, x, y, z):
    return m * (y * y + z * z) / 12, m * (x * x + z * z) / 12, m * (x * x + y * y) / 12


def cyl_inertia(m, r, h):
    ixx = m * (3 * r * r + h * h) / 12
    return ixx, ixx, m * r * r / 2


def inertial(m, i, origin):
    ixx, iyy, izz = i
    return (
        f'<inertial>{origin}<mass value="{m}"/>'
        f'<inertia ixx="{ixx:.5e}" iyy="{iyy:.5e}" izz="{izz:.5e}" ixy="0" ixz="0" iyz="0"/></inertial>'
    )


def box_link(name, sx, sy, sz, mass, cx, cz, color):
    """Box link with the box centred at (cx, 0, cz) in the link frame."""
    o = f'<origin xyz="{cx:.4f} 0 {cz:.4f}"/>'
    g = f'<geometry><box size="{sx} {sy} {sz}"/></geometry>'
    return f"""  <link name="{name}">
    <visual>{o}{g}<material name="{color}"/></visual>
    <collision>{o}{g}</collision>
    {inertial(mass, box_inertia(mass, sx, sy, sz), o)}
  </link>
"""


def segment_link(name, length, radius, mass, color):
    """Limb segment hanging straight down (-Z) from the link origin."""
    clen = length - SEG_SHRINK
    o = f'<origin xyz="0 0 {-length / 2:.4f}"/>'
    g = f'<geometry><cylinder radius="{radius}" length="{clen:.4f}"/></geometry>'
    return f"""  <link name="{name}">
    <visual>{o}{g}<material name="{color}"/></visual>
    <collision>{o}{g}</collision>
    {inertial(mass, cyl_inertia(mass, radius, length), o)}
  </link>
"""


def hub_link(name):
    """Massive-enough stub link carrying a hip DOF; visual only, no collider."""
    i = 0.4 * HUB_M * HUB_R**2
    return f"""  <link name="{name}">
    <visual><geometry><sphere radius="{HUB_R}"/></geometry><material name="grey"/></visual>
    {inertial(HUB_M, (i, i, i), "<origin xyz='0 0 0'/>")}
  </link>
"""


def joint(name, parent, child, xyz, axis):
    lo, hi, eff, vel = LIMITS[name.split("_", 1)[1]]
    return f"""  <joint name="{name}" type="revolute">
    <parent link="{parent}"/><child link="{child}"/>
    <origin xyz="{xyz}"/><axis xyz="{axis}"/>
    <limit lower="{lo}" upper="{hi}" effort="{eff}" velocity="{vel}"/>
  </joint>
"""


def build():
    out = [
        '<?xml version="1.0"?>\n<robot name="biped">\n',
        '  <material name="navy"><color rgba="0.16 0.22 0.42 1"/></material>\n',
        '  <material name="orange"><color rgba="0.85 0.42 0.12 1"/></material>\n',
        '  <material name="grey"><color rgba="0.35 0.35 0.38 1"/></material>\n',
        '  <material name="dark"><color rgba="0.12 0.12 0.14 1"/></material>\n',
        box_link("torso", TORSO_X, TORSO_Y, TORSO_Z, TORSO_M, 0.0, TORSO_Z / 2, "navy"),
    ]
    for side, sy in (("L", 1.0), ("R", -1.0)):
        out += [
            hub_link(f"{side}_hip_yaw_link"),
            hub_link(f"{side}_hip_roll_link"),
            segment_link(f"{side}_thigh", THIGH_L, THIGH_R, THIGH_M, "orange"),
            segment_link(f"{side}_shank", SHANK_L, SHANK_R, SHANK_M, "grey"),
            box_link(f"{side}_foot", FOOT_X, FOOT_Y, FOOT_Z, FOOT_M, FOOT_AHEAD, -FOOT_Z / 2, "dark"),
            joint(f"{side}_hip_yaw", "torso", f"{side}_hip_yaw_link", f"0 {HIP_Y * sy:.4f} 0", "0 0 1"),
            joint(f"{side}_hip_roll", f"{side}_hip_yaw_link", f"{side}_hip_roll_link", "0 0 0", "1 0 0"),
            joint(f"{side}_hip_pitch", f"{side}_hip_roll_link", f"{side}_thigh", "0 0 0", "0 1 0"),
            joint(f"{side}_knee", f"{side}_thigh", f"{side}_shank", f"0 0 {-THIGH_L:.4f}", "0 1 0"),
            joint(f"{side}_ankle", f"{side}_shank", f"{side}_foot", f"0 0 {-SHANK_L:.4f}", "0 1 0"),
        ]
    out.append("</robot>\n")
    return "".join(out)


def standing_hip_height():
    """Hip height when the legs hold the nominal crouch and the soles are flat."""
    thigh_w = Q_HIP_PITCH  # world pitch of each segment = sum of the joints above it
    shank_w = Q_HIP_PITCH + Q_KNEE
    drop = THIGH_L * math.cos(thigh_w) + SHANK_L * math.cos(shank_w)
    return drop + FOOT_Z


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "biped.urdf")
    with open(path, "w") as f:
        f.write(build())
    mass = TORSO_M + 2 * (2 * HUB_M + THIGH_M + SHANK_M + FOOT_M)
    h = standing_hip_height()
    print(f"[make_urdf] wrote {path}")
    print(f"[make_urdf] total mass {mass:.2f} kg; standing hip height {h:.3f} m -> spawn torso at {h + 0.02:.2f} m")
