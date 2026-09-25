"""Evaluate ANY quad ONNX (obs [1,36] -> actions [1,8], QUAD_SPEC.md) in MuJoCo Warp.

  # the MuJoCo-trained quad (defaults): 100 x 20 s -> training/quad/export/quad_mujoco_report.json
  .venv-mjwarp\\Scripts\\python.exe training/quad/mujoco/eval_quad_mujoco.py

  # cross-check the Isaac Lab 3 brain in plain MuJoCo
  .venv-mjwarp\\Scripts\\python.exe training/quad/mujoco/eval_quad_mujoco.py ^
      --onnx training/quad/export/quad_isaaclab3.onnx --tag isaaclab3_in_mujoco --tool isaaclab3

Deterministic (actor mean clipped to [-1, 1]), seed 12345, spec reset, no randomisation,
no pushes (--pushes adds them). A fall ends that racer's run: its distance stops there
and it still counts in meanSpeed (distance / 20 s); fallRate is reported beside it.
The report carries a plausibility block (rule I). See training/creature/evaluate.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from quad_env import QuadEnv, QuadPlausibility  # noqa: E402
from creature.evaluate import main  # noqa: E402

EXPORT_DIR = HERE.parent / "export"


def make_env(n: int, seed: int, pushes: bool = False) -> QuadEnv:
    return QuadEnv(n, seed=seed, randomize=False, random_initial_episode=False, pushes=pushes)


if __name__ == "__main__":
    sys.exit(main("quad", make_env, QuadPlausibility, EXPORT_DIR))
