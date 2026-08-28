"""Generate spider.urdf: a spherical body with 8 legs, each leg = hip swing joint (about Z) + knee joint (pitch).

Run with plain python:  python spider/make_urdf.py
"""
import math
import os

BODY_R, BODY_M = 0.10, 2.0
FEMUR_L, FEMUR_R, FEMUR_M, FEMUR_ELEV = 0.16, 0.02, 0.10, math.radians(35)  # femur points outward & up
TIBIA_L, TIBIA_R, TIBIA_M, TIBIA_ELEV = 0.26, 0.015, 0.12, math.radians(-60)  # tibia points outward & down
# leg mount angles around the body (deg, CCW from +X = forward). L = left (+Y), R = right (-Y)
LEG_ANGLES = {"L1": 30, "L2": 70, "L3": 110, "L4": 150, "R1": -30, "R2": -70, "R3": -110, "R4": -150}


def cyl_inertia(m, r, h):
    ixx = m * (3 * r * r + h * h) / 12
    return ixx, ixx, m * r * r / 2


def cyl_link(name, length, radius, mass, elev, color):
    """Cylinder link starting at the link origin and pointing along (cos(elev), 0, sin(elev)) in its own frame."""
    cx, cz = 0.5 * length * math.cos(elev), 0.5 * length * math.sin(elev)
    pitch = math.pi / 2 - elev  # cylinder axis is Z; rotate about Y so it points along the leg direction
    ixx, iyy, izz = cyl_inertia(mass, radius, length)
    origin = f'<origin xyz="{cx:.4f} 0 {cz:.4f}" rpy="0 {pitch:.5f} 0"/>'
    return f"""  <link name="{name}">
    <visual>{origin}<geometry><cylinder radius="{radius}" length="{length}"/></geometry><material name="{color}"/></visual>
    <collision>{origin}<geometry><cylinder radius="{radius}" length="{length}"/></geometry></collision>
    <inertial>{origin}<mass value="{mass}"/><inertia ixx="{ixx:.3e}" iyy="{iyy:.3e}" izz="{izz:.3e}" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
"""


def build():
    bi = 0.4 * BODY_M * BODY_R**2
    out = [
        '<?xml version="1.0"?>\n<robot name="spider">\n',
        '  <material name="black"><color rgba="0.15 0.1 0.1 1"/></material>\n',
        '  <material name="brown"><color rgba="0.45 0.25 0.1 1"/></material>\n',
        f"""  <link name="body">
    <visual><geometry><sphere radius="{BODY_R}"/></geometry><material name="black"/></visual>
    <collision><geometry><sphere radius="{BODY_R}"/></geometry></collision>
    <inertial><mass value="{BODY_M}"/><inertia ixx="{bi:.4f}" iyy="{bi:.4f}" izz="{bi:.4f}" ixy="0" ixz="0" iyz="0"/></inertial>
  </link>
""",
    ]
    for leg, deg in LEG_ANGLES.items():
        yaw = math.radians(deg)
        hx, hy = BODY_R * 0.9 * math.cos(yaw), BODY_R * 0.9 * math.sin(yaw)
        kx, kz = FEMUR_L * math.cos(FEMUR_ELEV), FEMUR_L * math.sin(FEMUR_ELEV)
        out.append(cyl_link(f"{leg}_femur", FEMUR_L, FEMUR_R, FEMUR_M, FEMUR_ELEV, "brown"))
        out.append(cyl_link(f"{leg}_tibia", TIBIA_L, TIBIA_R, TIBIA_M, TIBIA_ELEV, "black"))
        out.append(f"""  <joint name="{leg}_hip" type="revolute">
    <parent link="body"/><child link="{leg}_femur"/>
    <origin xyz="{hx:.4f} {hy:.4f} 0" rpy="0 0 {yaw:.5f}"/><axis xyz="0 0 1"/>
    <limit lower="-0.8" upper="0.8" effort="15" velocity="12"/>
  </joint>
  <joint name="{leg}_knee" type="revolute">
    <parent link="{leg}_femur"/><child link="{leg}_tibia"/>
    <origin xyz="{kx:.4f} 0 {kz:.4f}"/><axis xyz="0 1 0"/>
    <limit lower="-1.0" upper="1.0" effort="15" velocity="12"/>
  </joint>
""")
    out.append("</robot>\n")
    return "".join(out)


if __name__ == "__main__":
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "spider.urdf")
    with open(path, "w") as f:
        f.write(build())
    tip_z = FEMUR_L * math.sin(FEMUR_ELEV) + TIBIA_L * math.sin(TIBIA_ELEV)
    print(f"[make_urdf] wrote {path}; foot tip at z={tip_z:+.3f} m below body centre -> spawn body at ~{-tip_z + 0.02:.2f} m")
