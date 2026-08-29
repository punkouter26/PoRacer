r"""Interactive MuJoCo viewer for the creature. Runs on WINDOWS (uses the GPU
directly for rendering); no WSL needed.

    ..\.venv\Scripts\python.exe view_creature.py
    ..\.venv\Scripts\python.exe view_creature.py --policy ..\Assets\Creature\policy.json

With --policy it runs the exported JSON through a pure-numpy copy of the
inference math. This is the last checkpoint before Unity: if the creature walks
here, the JSON is good and any remaining problem is on the Unity side.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

HERE = Path(__file__).resolve().parent
XML = HERE.parent / "Assets" / "Creature" / "creature.xml"


class ExportedPolicy:
  """Pure-numpy twin of Assets/Creature/Scripts/CreaturePolicy.cs."""

  def __init__(self, path: Path):
    d = json.loads(Path(path).read_text())
    if d.get("format") != "mjx-unity-policy/1":
      raise SystemExit(f"unexpected policy format: {d.get('format')}")
    self.obs_size = d["obsSize"]
    self.action_size = d["actionSize"]
    self.n_substeps = d.get("nSubsteps", 5)
    self.mean = np.asarray(d["mean"], np.float32)
    self.std = np.asarray(d["std"], np.float32)
    self.layers = [
        (np.asarray(l["kernel"], np.float32).reshape(l["inSize"], l["outSize"]),
         np.asarray(l["bias"], np.float32))
        for l in d["layers"]
    ]

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    x = (obs - self.mean) / self.std
    for w, b in self.layers[:-1]:
      x = x @ w + b
      x = x / (1.0 + np.exp(-x))  # silu
    w, b = self.layers[-1]
    return np.tanh((x @ w + b)[: self.action_size])


def observe(model, data, torso_id, last_action) -> np.ndarray:
  """Mirror of OBS_LAYOUT in creature_env.py / CreatureAgent.cs."""
  R = data.xmat[torso_id].reshape(3, 3)
  return np.concatenate([
      R.T @ np.array([0.0, 0.0, -1.0]),
      R.T @ data.qvel[0:3],
      data.qvel[3:6],
      data.qpos[7:],
      data.qvel[6:],
      last_action,
  ]).astype(np.float32)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument("--policy", default=None, help="path to policy.json")
  ap.add_argument("--xml", default=str(XML))
  args = ap.parse_args()

  model = mujoco.MjModel.from_xml_path(args.xml)
  data = mujoco.MjData(model)
  mujoco.mj_resetDataKeyframe(model, data, 0)
  mujoco.mj_forward(model, data)

  torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
  policy = ExportedPolicy(Path(args.policy)) if args.policy else None

  if policy:
    if policy.obs_size != 9 + 2 * model.nu + model.nu:
      raise SystemExit(
          f"policy obs {policy.obs_size} does not match this model "
          f"({9 + 3 * model.nu} expected for nu={model.nu})")
    print(f"policy loaded: obs={policy.obs_size} act={policy.action_size} "
          f"decimation={policy.n_substeps}")
  else:
    print("no --policy given: showing the passive model (zero controls)")

  last_action = np.zeros(model.nu, np.float32)
  step = 0

  with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
      tick = time.time()

      if policy is not None and step % policy.n_substeps == 0:
        last_action = policy(observe(model, data, torso_id, last_action))
      data.ctrl[:] = last_action if policy is not None else 0.0

      mujoco.mj_step(model, data)
      step += 1
      viewer.sync()

      # keep roughly real time
      lag = model.opt.timestep - (time.time() - tick)
      if lag > 0:
        time.sleep(lag)


if __name__ == "__main__":
  main()
