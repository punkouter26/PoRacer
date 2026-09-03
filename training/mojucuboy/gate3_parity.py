"""Gate 3: numerical parity between the Python and Unity runtimes.

  1. `python gate3_parity.py emit`     -> writes _parity_state.json + _parity_python.json
  2. `unity cmd eval --code "CreatureEditor.MojucuBoyParityHarness.Run();"`
                                       -> writes _parity_unity.json
  3. `python gate3_parity.py compare`  -> element-wise diff, Gate 3 verdict

Both sides load mojucuboy_roundtrip.xml -- the round-trip export, which Gate 2 made the
ground-truth training model -- and are driven to the same raw qpos/qvel. The test
state is deliberately non-trivial: the root is tilted about all three axes and
translated off the origin, every joint is displaced away from its stance, and both
the root and every joint carry velocity. A state that is upright, centred or at
rest would let a transposed rotation or a dropped velocity term pass unnoticed.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

import mojucuboy_obs

HERE = Path(__file__).resolve().parent
MODEL = HERE / "mojucuboy_roundtrip.xml"
STATE = HERE / "_parity_state.json"
PY_OUT = HERE / "_parity_python.json"
UNITY_OUT = HERE / "_parity_unity.json"

TOLERANCE = 1e-4

COMMAND_HEADING = 0.6435011
COMMAND_SPEED = 1.5


def test_state(model) -> tuple[np.ndarray, np.ndarray]:
    """A deliberately awkward state: tilted, translated, and moving everywhere."""
    meta = json.loads((HERE / "mojucuboy_rig.json").read_text())
    qpos = np.zeros(model.nq)
    qvel = np.zeros(model.nv)

    qpos[0:3] = [0.12, -0.34, 0.83]
    # Tilted about all three axes -- 12 deg roll, -25 deg pitch, 8 deg yaw -- so a
    # transposed or mis-ordered rotation matrix cannot survive the comparison.
    euler = np.radians([12.0, -25.0, 8.0])
    quat = np.zeros(4)
    mat = np.zeros(9)
    mujoco.mju_euler2Quat(quat, euler, "xyz")
    mujoco.mju_quat2Mat(mat, quat)
    qpos[3:7] = quat

    stance = np.array([spec["stance_rad"] for spec in meta["joints"]])
    lo = np.array([spec["range_rad"][0] for spec in meta["joints"]])
    hi = np.array([spec["range_rad"][1] for spec in meta["joints"]])
    offsets = 0.23 * np.sin(np.arange(len(stance)) * 1.7 + 0.4)
    joints = np.clip(stance + offsets, lo + 1e-3, hi - 1e-3)

    qpos_addr, dof_addr = mojucuboy_obs.addresses(model, meta["actuator_order"])
    qpos[qpos_addr] = joints

    qvel[0:3] = [0.70, 1.30, -0.25]
    qvel[3:6] = [0.40, -0.60, 0.90]
    qvel[dof_addr] = 0.55 * np.cos(np.arange(len(stance)) * 2.1 + 0.9)
    return qpos, qvel


def emit() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    meta = json.loads((HERE / "mojucuboy_rig.json").read_text())
    order = meta["actuator_order"]

    qpos, qvel = test_state(model)
    last_action = (0.31 * np.sin(np.arange(mojucuboy_obs.ACTION_SIZE) * 0.8 + 0.2)).astype(np.float64)

    STATE.write_text(json.dumps({
        "root_body": mojucuboy_obs.ROOT_BODY,
        "actuator_order": order,
        "command_heading": COMMAND_HEADING,
        "command_speed": COMMAND_SPEED,
        "qpos": [float(v) for v in qpos],
        "qvel": [float(v) for v in qvel],
        "last_action": [float(v) for v in last_action],
    }, indent=2))

    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, mojucuboy_obs.ROOT_BODY)
    qpos_addr, dof_addr = mojucuboy_obs.addresses(model, order)
    obs = mojucuboy_obs.build(data, root_id, qpos_addr, dof_addr,
                        COMMAND_HEADING, COMMAND_SPEED, last_action)

    PY_OUT.write_text(json.dumps({
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "root_body_id": int(root_id),
        "timestep": float(model.opt.timestep),
        "obs": [float(v) for v in obs],
    }, indent=2))

    print(f"wrote {STATE.name} and {PY_OUT.name}")
    print(f"  nq={model.nq} nv={model.nv} nu={model.nu} root_body_id={root_id}")
    print(f"  obs size {len(obs)}  |obs|inf = {np.max(np.abs(obs)):.6f}")
    return 0


LABELS = (
    ("gravity_local", 0, 3), ("linvel_local", 3, 6), ("angvel_local", 6, 9),
    ("command", 9, 12), ("joint_pos", 12, 33), ("joint_vel", 33, 54),
    ("last_action", 54, 75),
)


def compare() -> int:
    if not UNITY_OUT.exists():
        print(f"ERROR: {UNITY_OUT.name} missing -- run the Unity harness first")
        return 2
    py = json.loads(PY_OUT.read_text())
    un = json.loads(UNITY_OUT.read_text())

    print("=== MODEL AGREEMENT ===")
    ok = True
    for key in ("nq", "nv", "nu", "root_body_id"):
        same = py[key] == un[key]
        ok &= same
        print(f"  {key:<14} python={py[key]:<6} unity={un[key]:<6} {'OK' if same else 'MISMATCH'}")
    dt_delta = abs(py["timestep"] - un["timestep"])
    print(f"  {'timestep':<14} python={py['timestep']:<12.9g} unity={un['timestep']:<12.9g}"
          f" delta={dt_delta:.3e}")

    a = np.asarray(py["obs"], dtype=np.float64)
    b = np.asarray(un["obs"], dtype=np.float64)
    if a.shape != b.shape:
        print(f"  FAIL: obs length {a.shape} vs {b.shape}")
        return 1
    diff = np.abs(a - b)

    print("\n=== OBSERVATION, PER BLOCK ===")
    print(f"{'block':<16}{'range':<12}{'max |d|':>12}{'at idx':>8}")
    for name, start, stop in LABELS:
        block = diff[start:stop]
        at = int(np.argmax(block)) + start
        print(f"{name:<16}{f'[{start}:{stop})':<12}{block.max():>12.3e}{at:>8}")

    worst = int(np.argmax(diff))
    print("\n=== GATE 3 VERDICT ===")
    print(f"  max |delta|      = {diff.max():.6e}")
    print(f"  at index         = {worst}"
          f"  ({next(n for n, s, e in LABELS if s <= worst < e)})")
    print(f"  python[{worst}]     = {a[worst]:+.9f}")
    print(f"  unity [{worst}]     = {b[worst]:+.9f}")
    print(f"  tolerance        = {TOLERANCE:g}")
    if ok and diff.max() < TOLERANCE:
        print("  PASS")
        return 0
    print("  FAIL")
    return 1


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    sys.exit(emit() if mode == "emit" else compare())
