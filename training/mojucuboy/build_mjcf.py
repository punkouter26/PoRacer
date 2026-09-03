"""Generate mojucuboy.xml -- the MJCF for the Boy_Character_mujoco rig.

Re-runnable: everything is derived from the GLB skeleton and the shoe mesh, so a
change to the source art reproduces a matching model. Nothing here is measured by
hand.

Body plan (settled 2026-09-03): 13 bodies, 21 actuated DOF, 45 kg, facing
MuJoCo +Y == Unity +Z. The rig's shoulder and hand bones carry no joint -- the
clavicles are welded into the torso and the hands into the forearms -- which is
what brings 19 bones down to the 21 DOF that matches IsaacBox.

The body tree is authored in the GLB BIND pose, so every body frame is
axis-aligned with the world at rest and each MuJoCo body frame coincides with its
bone's bind frame. That makes the Unity bone binding a translation offset instead
of a rotation, and it is why the standing stance is emitted as a qpos vector in
mojucuboy_rig.json rather than as an MJCF <keyframe>: org.mujoco's importer has no
keyframe case (MjcfImporter.ParseRoot) and would silently drop it.
"""

from __future__ import annotations

import json
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import rig_parse

GLB = "Assets/Boy_Character_mujoco.glb"
OUT_DIR = Path(__file__).resolve().parent

TOTAL_MASS = 45.0
DEG = np.pi / 180.0

# Torso-local axis roles at rest. Forward is +Y, so pitch is about X, roll about Y
# and yaw about Z. Everything downstream -- reward, observation, the Unity
# controller -- reads forward from this one constant.
FORWARD_AXIS = 1

# dt_physics 0.005 s matches the project's Time.fixedDeltaTime exactly, so
# org.mujoco overwriting <option timestep> with Time.fixedDeltaTime is a no-op
# here rather than a silent change of dynamics.
TIMESTEP = 0.005
DECIMATION = 4

# Position-servo stiffness. Calibrated, not guessed: the first cut ran 4x softer
# and could not hold the standing stance at all -- every joint sagged under
# gravity, the sag accumulated into a forward pitch, and the model toppled in
# 1.58 s. Sweeping the multiplier, 2.5x is the softest that holds an 8 s static
# hold; 4x is taken for headroom and settles to 0.018 m of drift. The static hold
# then peaks at 23.9 N.m, under 10% of the hip force limit, so the forceranges
# below are sized for dynamic locomotion rather than for standing up.
#
# (joint suffix, axis, lo_deg, hi_deg, kp, |forcerange|)
ARM_JOINTS = [
    ("shoulder_y", (0, 1, 0), -100.0, 100.0, 480.0, 80.0),
    ("shoulder_z", (0, 0, 1), -85.0, 85.0, 480.0, 80.0),
]
LEG_JOINTS = [
    ("hip_z", (0, 0, 1), -45.0, 45.0, 1400.0, 250.0),
    ("hip_x", (1, 0, 0), -35.0, 110.0, 1400.0, 250.0),
    ("hip_y", (0, 1, 0), -40.0, 40.0, 1400.0, 250.0),
]

# Standing stance for the legs, in degrees. Chosen by sweeping (hip_x, knee) with
# the ankle set to -(hip_x + knee) so the sole stays flat, and keeping the pose
# whose centre of mass sits furthest from the edge of the support polygon:
# 0.109 m of margin, against 0.046 m for the first cut. The first cut raked the
# legs backward (hip_x negative), which put the feet behind the torso and left
# the CoM almost on the toe edge -- the model pitched forward within a second.
STANCE_HIP_X = 7.5
STANCE_KNEE = -20.0
STANCE_ANKLE = -(STANCE_HIP_X + STANCE_KNEE)


def _fmt(values) -> str:
    return " ".join(f"{float(v):.6g}" for v in np.asarray(values, dtype=float).ravel())


class Builder:
    def __init__(self) -> None:
        self.rig = rig_parse.load(GLB)
        self.shoes = rig_parse.mesh_points(GLB, "Shoes")
        # Lift the whole model so the sole of the shoe rests exactly on z = 0.
        self.lift = -float(self.shoes[:, 2].min())
        self.joints: list[dict] = []

    # ---- geometry helpers -------------------------------------------------
    def world(self, bone: str) -> np.ndarray:
        return self.rig.pos(bone) + np.array([0.0, 0.0, self.lift])

    def side_sign(self, bone: str) -> float:
        """+1 if the bone sits on +x, -1 on -x. Read from the measured rest pose,
        never from the bone's L/R name: the 180 deg facing yaw put the
        character's left side on -x."""
        return 1.0 if self.rig.pos(bone)[0] >= 0 else -1.0

    def foot_box(self, bone: str) -> tuple[np.ndarray, np.ndarray]:
        """Half-extents and body-local centre of a foot box, off the shoe mesh."""
        sign = self.side_sign(bone)
        sel = self.shoes[self.shoes[:, 0] * sign > 0]
        lo, hi = sel.min(0), sel.max(0)
        # Cap the top just under the ankle: the shoe mesh climbs the shin, and an
        # uncapped box would be tall enough for the feet to trip on each other.
        top = min(float(hi[2]), float(self.rig.pos(bone)[2]) - 0.015)
        centre = np.array([(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0, (float(lo[2]) + top) / 2.0])
        half = np.array([(hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0, (top - float(lo[2])) / 2.0])
        return half, centre + np.array([0.0, 0.0, self.lift]) - self.world(bone)

    # ---- MJCF emission ----------------------------------------------------
    def add_joint(self, parent, name, axis, lo, hi, kp, frc, stance) -> None:
        ET.SubElement(parent, "joint", {
            "name": name,
            "type": "hinge",
            "axis": _fmt(axis),
            "range": f"{lo * DEG:.6g} {hi * DEG:.6g}",
        })
        self.joints.append({
            "name": name,
            "axis": list(axis),
            "range_deg": [lo, hi],
            "range_rad": [lo * DEG, hi * DEG],
            "kp": kp,
            "forcerange": frc,
            "stance_deg": stance,
            "stance_rad": stance * DEG,
        })

    def body(self, parent, name, bone, parent_bone):
        origin = self.world(parent_bone) if parent_bone else np.zeros(3)
        return ET.SubElement(parent, "body", {
            "name": name, "pos": _fmt(self.world(bone) - origin),
        })

    def capsule(self, body, name, origin_bone, a_bone, b_bone, radius, mass) -> None:
        base = self.world(origin_bone)
        fromto = np.concatenate([self.world(a_bone) - base, self.world(b_bone) - base])
        ET.SubElement(body, "geom", {
            "name": name, "type": "capsule", "fromto": _fmt(fromto),
            "size": f"{radius:.6g}", "mass": f"{mass:.6g}",
        })

    def build(self) -> ET.Element:
        rig = self.rig
        root = ET.Element("mujoco", {"model": "boy"})
        ET.SubElement(root, "compiler", {
            "angle": "radian", "coordinate": "local",
            "inertiafromgeom": "true", "autolimits": "true",
        })
        ET.SubElement(root, "option", {
            "timestep": str(TIMESTEP), "gravity": "0 0 -9.81",
            "integrator": "implicitfast", "solver": "Newton",
            "iterations": "10", "ls_iterations": "8",
            "cone": "pyramidal", "jacobian": "dense",
        })
        default = ET.SubElement(root, "default")
        ET.SubElement(default, "geom", {
            "condim": "3", "friction": "1.0 0.005 0.0001",
            "solref": "0.01 1", "solimp": "0.9 0.95 0.001",
            "rgba": "0.7 0.7 0.75 1",
        })
        # armature keeps the light distal links from ringing at 200 Hz; damping is
        # the passive term the position actuators' dampratio adds on top of.
        ET.SubElement(default, "joint", {"armature": "0.02", "damping": "1", "limited": "true"})

        world = ET.SubElement(root, "worldbody")
        ET.SubElement(world, "geom", {
            "name": "floor", "type": "plane", "size": "0 0 1", "pos": "0 0 0",
            "condim": "3", "friction": "1.0 0.005 0.0001", "rgba": "0.3 0.32 0.35 1",
        })

        hips_pos = self.world("hips")
        hips = ET.SubElement(world, "body", {"name": "hips", "pos": _fmt(hips_pos)})
        ET.SubElement(hips, "freejoint", {"name": "root"})
        pelvis_h = float(self.world("spine")[2] - hips_pos[2])
        ET.SubElement(hips, "geom", {
            "name": "pelvis", "type": "box",
            "pos": _fmt([0.0, 0.0, pelvis_h / 2.0]),
            "size": _fmt([abs(rig.pos("thigh.L")[0]) + 0.030, 0.070, pelvis_h / 2.0 + 0.020]),
            "mass": "8.0",
        })

        # ---- torso: spine + chest + both clavicles, 3 DOF ------------------
        torso = self.body(hips, "torso", "spine", "hips")
        for jname, axis, lo, hi in (
            ("abdomen_z", (0, 0, 1), -45.0, 45.0),
            ("abdomen_y", (0, 1, 0), -35.0, 35.0),
            ("abdomen_x", (1, 0, 0), -45.0, 25.0),
        ):
            self.add_joint(torso, jname, axis, lo, hi, 1200.0, 200.0, 0.0)
        self.capsule(torso, "spine", "spine", "spine", "chest", 0.080, 5.0)
        self.capsule(torso, "chest", "spine", "chest", "neck", 0.095, 6.6)
        for side in ("L", "R"):
            self.capsule(torso, f"clavicle_{side}", "spine",
                         f"shoulder.{side}", f"upper_arm.{side}", 0.035, 0.7)

        # ---- head: neck + head fused, 2 DOF --------------------------------
        head = self.body(torso, "head", "neck", "spine")
        self.add_joint(head, "neck_x", (1, 0, 0), -35.0, 35.0, 240.0, 40.0, 0.0)
        self.add_joint(head, "neck_z", (0, 0, 1), -50.0, 50.0, 240.0, 40.0, 0.0)
        ET.SubElement(head, "geom", {
            "name": "head", "type": "sphere", "pos": "0 0 0.09",
            "size": "0.1", "mass": "4.0",
        })

        # ---- arms: 2 DOF shoulder + 1 DOF elbow, hand fused into forearm ----
        for side in ("L", "R"):
            sign = self.side_sign(f"upper_arm.{side}")
            upper = self.body(torso, f"upper_arm_{side}", f"upper_arm.{side}", "spine")
            for jname, axis, lo, hi, kp, frc in ARM_JOINTS:
                stance = 75.0 * sign if jname == "shoulder_y" else 0.0
                self.add_joint(upper, f"{jname}_{side}", axis, lo, hi, kp, frc, stance)
            self.capsule(upper, f"upper_arm_{side}", f"upper_arm.{side}",
                         f"upper_arm.{side}", f"forearm.{side}", 0.048, 1.8)

            fore = self.body(upper, f"forearm_{side}", f"forearm.{side}", f"upper_arm.{side}")
            # Elbow flexion has to swing the hand toward +y (forward) on both
            # sides, and the two arms point opposite ways along x, so the allowed
            # range is signed rather than symmetric.
            lo, hi = (-140.0, 0.0) if sign < 0 else (0.0, 140.0)
            self.add_joint(fore, f"elbow_{side}", (0, 0, 1), lo, hi, 320.0, 60.0, 20.0 * sign)
            self.capsule(fore, f"forearm_{side}", f"forearm.{side}",
                         f"forearm.{side}", f"hand.{side}", 0.040, 1.15)
            ET.SubElement(fore, "geom", {
                "name": f"hand_{side}", "type": "sphere",
                "pos": _fmt(self.world(f"hand.{side}") - self.world(f"forearm.{side}")),
                "size": "0.045", "mass": "0.25",
            })

        # ---- legs: 3 DOF hip + 1 DOF knee + 1 DOF ankle --------------------
        for side in ("L", "R"):
            thigh = self.body(hips, f"thigh_{side}", f"thigh.{side}", "hips")
            for jname, axis, lo, hi, kp, frc in LEG_JOINTS:
                self.add_joint(thigh, f"{jname}_{side}", axis, lo, hi, kp, frc,
                               STANCE_HIP_X if jname == "hip_x" else 0.0)
            self.capsule(thigh, f"thigh_{side}", f"thigh.{side}",
                         f"thigh.{side}", f"shin.{side}", 0.060, 4.5)

            shin = self.body(thigh, f"shin_{side}", f"shin.{side}", f"thigh.{side}")
            self.add_joint(shin, f"knee_{side}", (1, 0, 0), -160.0, 0.0, 1200.0, 250.0, STANCE_KNEE)
            self.capsule(shin, f"shin_{side}", f"shin.{side}",
                         f"shin.{side}", f"foot.{side}", 0.048, 1.8)

            foot = self.body(shin, f"foot_{side}", f"foot.{side}", f"shin.{side}")
            self.add_joint(foot, f"ankle_{side}", (1, 0, 0), -45.0, 45.0, 480.0, 100.0, STANCE_ANKLE)
            half, centre = self.foot_box(f"foot.{side}")
            ET.SubElement(foot, "geom", {
                "name": f"foot_{side}", "type": "box",
                "pos": _fmt(centre), "size": _fmt(half), "mass": "0.5",
                "friction": "1.2 0.02 0.001",
            })

        # ---- actuators: position, dampratio 1, in joint declaration order ---
        actuators = ET.SubElement(root, "actuator")
        for spec in self.joints:
            ET.SubElement(actuators, "position", {
                "name": f"act_{spec['name']}", "joint": spec["name"],
                "kp": f"{spec['kp']:.6g}", "dampratio": "1",
                "ctrlrange": f"{spec['range_rad'][0]:.6g} {spec['range_rad'][1]:.6g}",
                "forcerange": f"{-spec['forcerange']:.6g} {spec['forcerange']:.6g}",
            })
        return root


def main() -> None:
    builder = Builder()
    root = builder.build()
    xml_text = minidom.parseString(ET.tostring(root, "unicode")).toprettyxml(indent="  ")
    xml_text = "\n".join(line for line in xml_text.splitlines() if line.strip())
    xml_path = OUT_DIR / "mojucuboy.xml"
    xml_path.write_text(xml_text + "\n", encoding="utf-8")

    meta = {
        "source_glb": GLB,
        "fk_vs_ibm_residual_m": builder.rig.residual,
        "ground_lift_m": builder.lift,
        "total_mass_kg": TOTAL_MASS,
        "timestep": TIMESTEP,
        "decimation": DECIMATION,
        "policy_dt": TIMESTEP * DECIMATION,
        "forward_axis": FORWARD_AXIS,
        "actuator_order": [spec["name"] for spec in builder.joints],
        "stance_deg": [spec["stance_deg"] for spec in builder.joints],
        "joints": builder.joints,
    }
    (OUT_DIR / "mojucuboy_rig.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {xml_path} -- {len(builder.joints)} actuated joints")
    print(f"ground lift {builder.lift:.4f} m; fk-vs-ibm residual {builder.rig.residual:.3e} m")


if __name__ == "__main__":
    main()
