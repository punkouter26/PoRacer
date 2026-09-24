"""Builds the one worm both trainers use: worm.xml (MJCF) and worm_rig.json.

The MuJoCo trainer loads worm.xml directly; the Isaac Lab task converts the same
worm.xml to USD; Unity rebuilds the body from worm_rig.json. One source, so the two
tools train the same animal and the race compares training, not bodies.

Body: 5 capsule segments in a line along +x (segment 0 = head, at the front), joined
by 4 two-axis joints. Each joint is a pitch hinge (up-down, axis y) on a small
intermediate link, then a yaw hinge (side-to-side, axis z) into the next segment.
Both bend +/-45 degrees. The intermediate link is explicit (rather than two hinges
in one body) so that MuJoCo, Isaac's MJCF importer and Unity's ArticulationBody all
build the same kinematic tree.

Units SI, gravity -9.81, MuJoCo frame: x forward, z up.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent

SEGMENTS = 5
SEGMENT_SPACING = 0.20       # joint to joint, metres
CAPSULE_HALF = 0.075         # half-length of the capsule's cylinder part
RADIUS = 0.045
DENSITY = 1000.0             # a water-filled body, like real soft-bodied animals
LINK_MASS = 0.10             # intermediate pitch link
LINK_INERTIA = 2e-4
JOINT_RANGE_DEG = 45.0
KP = 30.0                    # N*m/rad
FORCE_LIMIT = 12.0           # N*m; lifts ~2 segments at arm's length, no more
JOINT_DAMPING = 1.0          # N*m*s/rad, on the joint (stable at 5 ms, see training/bugs)
ARMATURE = 0.01
FRICTION = 0.9               # AGENTS 2E: static 0.8-1.0; no bounce
SPAWN_Z = RADIUS + 0.005     # lying on the floor, 5 mm clear

JOINT_RANGE = math.radians(JOINT_RANGE_DEG)


def capsule_mass() -> float:
    cylinder = math.pi * RADIUS ** 2 * (2 * CAPSULE_HALF)
    caps = 4.0 / 3.0 * math.pi * RADIUS ** 3
    return DENSITY * (cylinder + caps)


def build() -> tuple[str, dict]:
    segment_mass = capsule_mass()
    lines = [
        '<?xml version="1.0" ?>',
        '<mujoco model="worm5">',
        '  <compiler angle="radian" autolimits="true"/>',
        '  <option timestep="0.005" gravity="0 0 -9.81" integrator="implicitfast" '
        'solver="Newton" iterations="10" ls_iterations="8" cone="pyramidal"/>',
        '  <default>',
        f'    <geom condim="3" friction="{FRICTION} 0.005 0.0001" solref="0.01 1" '
        'solimp="0.9 0.95 0.001"/>',
        f'    <joint type="hinge" limited="true" range="{-JOINT_RANGE:.6f} {JOINT_RANGE:.6f}" '
        f'damping="{JOINT_DAMPING}" armature="{ARMATURE}"/>',
        f'    <position kp="{KP}" ctrlrange="{-JOINT_RANGE:.6f} {JOINT_RANGE:.6f}" '
        f'forcerange="{-FORCE_LIMIT} {FORCE_LIMIT}"/>',
        '  </default>',
        '  <worldbody>',
        f'    <geom name="floor" type="plane" size="0 0 1" friction="{FRICTION} 0.005 0.0001" '
        'rgba="0.3 0.32 0.35 1"/>',
    ]
    indent = "    "
    rig_bodies = []
    joints = []

    def segment_geom(index: int) -> str:
        return (f'<geom name="seg{index}" type="capsule" fromto="{-CAPSULE_HALF} 0 0 {CAPSULE_HALF} 0 0" '
                f'size="{RADIUS}" mass="{segment_mass:.6f}" rgba="0.35 0.55 0.9 1"/>')

    lines.append(f'{indent}<body name="seg0" pos="0 0 {SPAWN_Z:.4f}">')
    lines.append(f'{indent}  <freejoint name="root"/>')
    lines.append(f'{indent}  {segment_geom(0)}')
    rig_bodies.append({"name": "seg0", "parent": None, "pos": [0, 0, SPAWN_Z], "mass": segment_mass,
                       "capsule": {"halfLength": CAPSULE_HALF, "radius": RADIUS, "axis": "x"}})
    depth = 1
    for joint_index in range(SEGMENTS - 1):
        pad = indent + "  " * depth
        link = f"link{joint_index}"
        child = f"seg{joint_index + 1}"
        # The pitch link sits at the joint, half a spacing behind the parent's centre.
        lines.append(f'{pad}<body name="{link}" pos="{-SEGMENT_SPACING / 2} 0 0">')
        lines.append(f'{pad}  <inertial pos="0 0 0" mass="{LINK_MASS}" '
                     f'diaginertia="{LINK_INERTIA} {LINK_INERTIA} {LINK_INERTIA}"/>')
        lines.append(f'{pad}  <joint name="j{joint_index}_pitch" axis="0 1 0"/>')
        lines.append(f'{pad}  <body name="{child}" pos="{-SEGMENT_SPACING / 2} 0 0">')
        lines.append(f'{pad}    <joint name="j{joint_index}_yaw" axis="0 0 1" pos="{SEGMENT_SPACING / 2} 0 0"/>')
        lines.append(f'{pad}    {segment_geom(joint_index + 1)}')
        rig_bodies.append({"name": link, "parent": f"seg{joint_index}", "pos": [-SEGMENT_SPACING / 2, 0, 0],
                           "mass": LINK_MASS, "inertia": LINK_INERTIA})
        rig_bodies.append({"name": child, "parent": link, "pos": [-SEGMENT_SPACING / 2, 0, 0],
                           "mass": segment_mass,
                           "capsule": {"halfLength": CAPSULE_HALF, "radius": RADIUS, "axis": "x"}})
        joints.append({"name": f"j{joint_index}_pitch", "body": link, "axis": [0, 1, 0], "anchor": [0, 0, 0]})
        joints.append({"name": f"j{joint_index}_yaw", "body": child, "axis": [0, 0, 1],
                       "anchor": [SEGMENT_SPACING / 2, 0, 0]})
        depth += 2
    for close_depth in range(depth - 1, -1, -1):
        lines.append(f'{indent}{"  " * close_depth}</body>')
    lines.append('  </worldbody>')

    # Adjacent segments overlap at their joint by design; they are not parent and
    # child (the pitch link sits between them), so MuJoCo would collide them.
    lines.append('  <contact>')
    for joint_index in range(SEGMENTS - 1):
        lines.append(f'    <exclude body1="seg{joint_index}" body2="seg{joint_index + 1}"/>')
    lines.append('  </contact>')

    lines.append('  <actuator>')
    for joint in joints:
        lines.append(f'    <position name="{joint["name"]}" joint="{joint["name"]}"/>')
    lines.append('  </actuator>')
    lines.append('</mujoco>')

    rig = {
        "model": "worm5",
        "frame": "MuJoCo: x forward (head at +x), y left, z up. Unity (x, y, z) = MuJoCo (-y, z, x); see WORM_SPEC.md",
        "segments": SEGMENTS,
        "segmentSpacing": SEGMENT_SPACING,
        "capsuleHalfLength": CAPSULE_HALF,
        "radius": RADIUS,
        "segmentMass": segment_mass,
        "linkMass": LINK_MASS,
        "linkInertia": LINK_INERTIA,
        "totalMass": segment_mass * SEGMENTS + LINK_MASS * (SEGMENTS - 1),
        "jointRangeRad": JOINT_RANGE,
        "kp": KP,
        "forceLimit": FORCE_LIMIT,
        "jointDamping": JOINT_DAMPING,
        "armature": ARMATURE,
        "friction": FRICTION,
        "spawnHeight": SPAWN_Z,
        "physicsDt": 0.005,
        "decimation": 4,
        "bodies": rig_bodies,
        "joints": joints,
        "actionOrder": [joint["name"] for joint in joints],
    }
    return "\n".join(lines) + "\n", rig


def main() -> None:
    xml, rig = build()
    (HERE / "worm.xml").write_text(xml, encoding="utf-8")
    (HERE / "worm_rig.json").write_text(json.dumps(rig, indent=1), encoding="utf-8")
    print(f"worm.xml + worm_rig.json: {rig['segments']} segments, {len(rig['joints'])} joints, "
          f"{rig['totalMass']:.2f} kg, {SEGMENT_SPACING * SEGMENTS:.2f} m long")


if __name__ == "__main__":
    main()
