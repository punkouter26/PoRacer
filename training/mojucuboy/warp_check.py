"""Contact-integrity check for MuJoCo Warp, before any training is attempted.

This repository carries a recorded, measured failure of a Warp path
(training/fido/train.py): MJX driven through Brax/Playground exposes only a
batch-wide contact budget and never a per-world nconmax, so the broadphase
overflowed every step, contacts were silently dropped, and the creature sank to a
torso height of -6.9 m -- while appearing FASTER because it skipped collision work.

That was MJX-with-a-Warp-backend under jax.vmap. This is native mujoco_warp, whose
put_data takes explicit per-batch nconmax/njmax. Structurally it should not have
the same defect, but "should" is not evidence, so this compares a batched GPU
rollout against single-world CPU MuJoCo on the identical initial state.

Abort criteria -- any one of these means do not train:
  * the GPU torso height diverges from CPU by more than DIVERGENCE_TOL
  * the torso ever goes below the floor
  * the contact buffer reports an overflow
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "mojucuboy_roundtrip.xml"

NWORLD = 512
STEPS = 400          # 2 s at 0.005 s
DIVERGENCE_TOL = 1e-3
NCONMAX = 256
NJMAX = 512


def cpu_rollout(model, qpos, ctrl):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    mujoco.mj_forward(model, data)
    heights, contacts = [], []
    for _ in range(STEPS):
        mujoco.mj_step(model, data)
        heights.append(float(data.qpos[2]))
        contacts.append(int(data.ncon))
    return np.array(heights), np.array(contacts)


def warp_rollout(model, qpos, ctrl):
    import mujoco_warp as mjw
    import warp as wp

    seed = mujoco.MjData(model)
    seed.qpos[:] = qpos
    seed.ctrl[:] = ctrl
    mujoco.mj_forward(model, seed)

    wm = mjw.put_model(model)
    wd = mjw.put_data(model, seed, nworld=NWORLD, nconmax=NCONMAX, njmax=NJMAX)

    heights = np.zeros((STEPS, NWORLD), dtype=np.float64)
    max_ncon = 0
    wp.synchronize()
    started = time.perf_counter()
    for step in range(STEPS):
        mjw.step(wm, wd)
        heights[step] = wd.qpos.numpy()[:, 2]
        max_ncon = max(max_ncon, int(np.max(wd.nacon.numpy())))
    wp.synchronize()
    elapsed = time.perf_counter() - started
    return heights, max_ncon, elapsed, int(wd.naconmax)


def main() -> int:
    meta = json.loads((HERE / "mojucuboy_rig.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    qpos = np.array(meta["stance_qpos"])
    ctrl = qpos[7:].copy()

    print(f"model: {MODEL.name}  nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"batch: nworld={NWORLD} steps={STEPS} nconmax={NCONMAX} njmax={NJMAX}\n")

    cpu_h, cpu_ncon = cpu_rollout(model, qpos, ctrl)
    print("=== CPU MuJoCo (reference) ===")
    print(f"  torso height  start {cpu_h[0]:.4f}  min {cpu_h.min():.4f}  end {cpu_h[-1]:.4f} m")
    print(f"  contacts      min {cpu_ncon.min()}  max {cpu_ncon.max()}")

    warp_h, max_ncon, elapsed, naconmax = warp_rollout(model, qpos, ctrl)
    w0 = warp_h[:, 0]
    print("\n=== MuJoCo Warp (GPU, batched) ===")
    print(f"  torso height  start {w0[0]:.4f}  min {warp_h.min():.4f}  end {w0[-1]:.4f} m")
    print(f"  peak contacts {max_ncon} of naconmax {naconmax}"
          f"   {'OVERFLOW' if max_ncon >= naconmax else 'headroom OK'}")
    spread = float(np.max(np.abs(warp_h - warp_h[:, :1])))
    print(f"  world spread  {spread:.3e} m  (identical inputs, so this should be ~0)")
    print(f"  throughput    {NWORLD * STEPS / elapsed / 1e6:.2f}M steps/s"
          f"   ({elapsed:.2f} s wall)")

    divergence = float(np.max(np.abs(w0 - cpu_h)))
    print("\n=== GPU vs CPU ===")
    print(f"  max |delta| torso height = {divergence:.3e} m  (tolerance {DIVERGENCE_TOL:g})")

    problems = []
    if warp_h.min() < -0.01:
        problems.append(f"torso went below the floor: {warp_h.min():.3f} m "
                        f"-- this is the recorded sink-through failure")
    if max_ncon >= naconmax:
        problems.append(f"contact buffer overflow: {max_ncon} >= naconmax {naconmax}")
    if divergence > DIVERGENCE_TOL:
        problems.append(f"GPU diverged from CPU by {divergence:.3e} m")

    print("\n=== VERDICT ===")
    if problems:
        for p in problems:
            print(f"  ABORT: {p}")
        print("  Native MuJoCo Warp is NOT usable for this model. Do not train.")
        return 1
    print("  PASS: native MuJoCo Warp keeps its contacts and tracks CPU MuJoCo.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
