"""Export a quad checkpoint to ONNX: obs [1,36] -> actions [1,8], normaliser baked in.

  # final policy of the newest run -> training/quad/export/quad_mujoco.onnx
  .venv-mjwarp\Scripts\python.exe training/quad/mujoco/export_quad_mujoco.py

  # the best checkpoint before a collapse, to its own file
  .venv-mjwarp\Scripts\python.exe training/quad/mujoco/export_quad_mujoco.py --pick best ^
      --onnx-out training/quad/export/quad_mujoco_best.onnx

The trainer already exports its final policy; this re-exports any checkpoint.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from quad_env import OBS_DESCRIPTION  # noqa: E402  (also puts training/ on sys.path)
from creature.export import export_main  # noqa: E402

if __name__ == "__main__":
    sys.exit(export_main("quad", HERE / "runs", HERE.parent / "export" / "quad_mujoco.onnx",
                         OBS_DESCRIPTION, "training/quad/mujoco/train_quad_mujoco.py"))
