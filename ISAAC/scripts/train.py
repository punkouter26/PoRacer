#!/usr/bin/env python3
"""Train the Boy target-chasing policy with RSL-RL PPO in Isaac Lab, headless.

Run from the Isaac Lab checkout's Python (install.ps1 sets it up)::

    ISAAC\\isaaclab\\isaaclab.bat -p ISAAC\\scripts\\train.py --num_envs 4096 --max_iterations 3000

TensorBoard is started FIRST, before the simulator, on --tensorboard_port (project rule:
no training run without it). Teardown order is trainer -> env -> simulator -> TensorBoard.

Logs land in ``logs/rsl_rl/boy_chase_flat/<timestamp>/`` under the current directory, with the
resolved ``params/env.yaml`` and ``params/agent.yaml`` beside the checkpoints, which is what
``export_bundle.py`` reads back.
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

# Unbuffer stdout/stderr BEFORE anything prints. Isaac Sim ends the process inside
# simulation_app.close() with os._exit(), which skips Python's atexit flush - so when stdout
# is a file or a pipe (any redirected/detached run) every print() still sitting in the 8 KB
# block buffer is silently discarded. Kit's own C++ log lines bypass that buffer and survive,
# which makes a perfectly good run look like it died right after startup: no training output,
# no traceback, exit code 0. Only the checkpoints on disk give it away.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train the Boy chase policy (RSL-RL PPO).")
parser.add_argument("--task", type=str, default="Isaac-Chase-Flat-Boy-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=None, help="Override the runner cfg.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="", help="Suffix for the log folder.")
parser.add_argument("--resume", action="store_true", help="Resume from --load_run / --checkpoint.")
parser.add_argument("--load_run", type=str, default=".*", help="Run folder regex to resume from.")
parser.add_argument("--checkpoint", type=str, default="model_.*.pt", help="Checkpoint regex to resume from.")
parser.add_argument("--log_root", type=str, default=os.path.join("logs", "rsl_rl"))
parser.add_argument("--tensorboard_port", type=int, default=6006)
parser.add_argument("--no_tensorboard", action="store_true", help="NOT recommended; see CLAUDE.md.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if not getattr(args_cli, "headless", False):
    # training is headless by definition; --headless stays accepted for symmetry
    args_cli.headless = True

# ------------------------------------------------------------------ TensorBoard first --
_tb = None


def start_tensorboard(logdir, port):
    os.makedirs(logdir, exist_ok=True)
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", logdir, "--port", str(port), "--bind_all"]
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:  # noqa: BLE001
        print(f"[train] could not start TensorBoard ({exc}); install it with `pip install tensorboard`.")
        return None
    time.sleep(2.0)
    if proc.poll() is not None:
        print("[train] TensorBoard exited immediately (port busy?). Continuing without it is against "
              "project policy; pass --tensorboard_port <free port> and rerun.")
        return None
    print(f"[train] TensorBoard up: http://localhost:{port}  (logdir {logdir})")
    return proc


# TensorBoard points at the log ROOT, not the experiment directory. It scans recursively, so
# it still picks up <log_root>/<experiment>/<run> the moment the trainer creates it - and this
# way nothing from isaaclab_tasks has to be imported yet. That matters: reading the experiment
# name from the registry pulls isaaclab.envs -> isaaclab.utils.mesh -> `from pxr import Usd`,
# and pxr does not exist until the SimulationApp has initialised kit. Importing it here failed
# with "No module named 'pxr'" (or, with a PyPI usd-core installed to paper over that, the far
# more confusing "DLL load failed while importing _tf"). Simulator first, registry after.
if not args_cli.no_tensorboard:
    _tb = start_tensorboard(os.path.abspath(args_cli.log_root), args_cli.tensorboard_port)

# ------------------------------------------------------------------------ simulator --
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- everything below needs kit to be live ------------------------------------------
import importlib  # noqa: E402

import gymnasium as gym  # noqa: E402

import boy_tasks  # noqa: E402,F401  (registers the tasks)
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
try:
    import importlib.metadata as metadata
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
except Exception:  # noqa: BLE001  (older Isaac Lab: no shim needed)
    pass

experiment_dir = os.path.abspath(os.path.join(args_cli.log_root, agent_cfg.experiment_name))

import torch  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

try:
    from isaaclab_tasks.utils import get_checkpoint_path
except ImportError:  # pragma: no cover
    from isaaclab_rl.utils import get_checkpoint_path  # type: ignore

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def main():
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    env_cfg.sim.device = args_cli.device if hasattr(args_cli, "device") and args_cli.device else env_cfg.sim.device
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = env_cfg.sim.device
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = stamp + (f"_{args_cli.run_name}" if args_cli.run_name else "")
    log_dir = os.path.join(experiment_dir, run_dir)
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train] log dir: {log_dir}")

    resume_path = None
    if args_cli.resume:
        resume_path = get_checkpoint_path(experiment_dir, args_cli.load_run, args_cli.checkpoint)
        print(f"[train] resuming from {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    runner.add_git_repo_to_log(__file__)
    if resume_path:
        runner.load(resume_path)

    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # a fixed physics step is the contract with Unity: refuse to train anything else
    dt = env.unwrapped.physics_dt if hasattr(env.unwrapped, "physics_dt") else env_cfg.sim.dt
    assert abs(dt - 0.005) < 1e-9, f"physics dt {dt} != 0.005 s; the Unity port assumes 200 Hz / decimation 4"
    assert env_cfg.decimation == 4, env_cfg.decimation

    try:
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:  # noqa: BLE001
        # PRINT BEFORE THE finally. simulation_app.close() ends the process with os._exit()
        # inside Isaac Sim, which discards the in-flight exception and hands back exit code 0
        # - so an unguarded failure here looks like a clean run that simply never trained.
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()
        if _tb is not None:
            _tb.terminate()
            print("[train] TensorBoard stopped.")
