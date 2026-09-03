"""Produce the MJCF that org.mujoco can import without corrupting it.

Works around a genuine org.mujoco importer defect, characterised on 2026-09-03:

  MuJoCo's own XML writer (mj_saveLastXML) omits a hinge's `axis` attribute when
  it equals MuJoCo's default of (0, 0, 1). org.mujoco's MjHingeJoint.OnParseMjcf
  then reads it as GetVector3Attribute("axis", defaultValue: Vector3.right) --
  Unity's +X, which is MuJoCo's +X, not MuJoCo's +Z. So every Z-axis hinge is
  silently imported as an X-axis hinge, with no error anywhere.

  On this rig that hit 8 of 21 hinges: both hip yaws, both shoulder fore/aft
  swings, both elbows, the abdomen yaw and the neck yaw. The abdomen yaw became a
  duplicate of the abdomen pitch.

Authoring `axis` explicitly does not help on its own, because MjImporterWithAssets
.ImportFile compiles the file and re-saves it through MuJoCo before importing --
which drops the attribute again. The fix is therefore twofold:

  1. here: take MuJoCo's own flattened output (so <default> classes and derived
     actuator gains are resolved exactly as MuJoCo resolves them) and write an
     explicit `axis` onto every hinge that lacks one;
  2. in Unity: import with ImportString, NOT ImportFile, so this file reaches the
     importer verbatim instead of being round-tripped through the writer again.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "mojucuboy.xml"
FLATTENED = HERE / "_savedlast.xml"
UNITY_COPY = HERE.parents[1] / "Assets" / "Agents" / "MojucuBoy_v01" / "mojucuboy_unity.xml"

# MuJoCo's documented default hinge axis, and the one its writer omits.
MJC_DEFAULT_AXIS = "0 0 1"


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SOURCE))
    mujoco.mj_saveLastXML(str(FLATTENED), model)

    tree = ET.parse(FLATTENED)
    root = tree.getroot()

    patched = []
    for joint in root.iter("joint"):
        # Skip the <default> section's template joint: it has no name and setting
        # an axis there would apply to every joint that omits one.
        if joint.get("name") is None:
            continue
        if joint.get("type") == "free":
            continue
        if joint.get("axis") is None:
            joint.set("axis", MJC_DEFAULT_AXIS)
            patched.append(joint.get("name"))

    UNITY_COPY.parent.mkdir(parents=True, exist_ok=True)
    tree.write(UNITY_COPY, encoding="utf-8", xml_declaration=True)

    # Prove the patch changed nothing about the physics: the patched file must
    # compile to the same joint axes as the source.
    check = mujoco.MjModel.from_xml_path(str(UNITY_COPY))
    import numpy as np

    delta = float(np.max(np.abs(model.jnt_axis - check.jnt_axis)))
    print(f"wrote {UNITY_COPY}")
    print(f"made axis explicit on {len(patched)} hinge(s): {patched}")
    print(f"jnt_axis delta source vs patched = {delta:.3e}  (must be 0)")
    print(f"njnt {model.njnt} -> {check.njnt}   nu {model.nu} -> {check.nu}")


if __name__ == "__main__":
    main()
