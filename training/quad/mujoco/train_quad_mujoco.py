"""PPO for the Quad pilot on MuJoCo Warp (training/quad/QUAD_SPEC.md), on the creature template.

  .venv-mjwarp\\Scripts\\python.exe training/quad/mujoco/train_quad_mujoco.py --minutes 30 --num-envs 4096 --tensorboard-port 6006

TensorBoard starts first (rule C) on --tensorboard-port, logdir training/quad/mujoco/runs; a
busy port is refused, nothing is killed. Checkpoints every 50 iterations (model_<it>.pt,
with checkpoints.jsonl), model_final.pt at the end, and the final policy exported to
training/quad/export/quad_mujoco.onnx (obs [1,36] -> actions [1,8]). A 30-minute run takes
over the machine: close the Unity editor first (rule L).
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from quad_env import OBS_DESCRIPTION, QuadEnv  # noqa: E402  (also puts training/ on sys.path)
from creature.ppo import TrainSetup, main  # noqa: E402

EXPORT_DIR = HERE.parent / "export"

SETUP = TrainSetup(
    name="quad",
    make_env=lambda n, seed: QuadEnv(n, seed=seed, randomize=True, random_initial_episode=True),
    runs_dir=HERE / "runs",
    onnx_out=EXPORT_DIR / "quad_mujoco.onnx",
    trainer_script="training/quad/mujoco/train_quad_mujoco.py",
    obs_description=OBS_DESCRIPTION,
    eval_command=".venv-mjwarp\\Scripts\\python.exe training/quad/mujoco/eval_quad_mujoco.py",
    tool="mujoco",
    default_port=6006,
    lr_schedule="adaptive",       # QUAD_SPEC round 4: adaptive KL, target 0.01, lr <= 3e-4
    desired_kl=0.01,
)

if __name__ == "__main__":
    sys.exit(main(SETUP))
