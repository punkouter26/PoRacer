#!/usr/bin/env python3
"""Train Worm5 with RSL-RL PPO in Isaac Lab 3 on Newton + MuJoCo-Warp (kit-less), headless.

The third trainer of training/worm/WORM_SPEC.md. Run with the Isaac Lab 3 venv's python::

    ISAAC\\isaaclab3\\.venv\\Scripts\\python.exe ISAAC\\scripts\\train_worm_v3.py --minutes 30

No Isaac Sim / Kit is involved: Newton is the physics, MuJoCo-Warp the solver.

Order of events (project rule C: TensorBoard first):

1. Check that --tensorboard_port is free. If anything already listens there, print an error and
   exit - no process is ever killed.
2. Start TensorBoard on ISAAC/logs/rsl_rl_v3/worm5_newton and wait until it answers.
3. Build Isaac-Worm5-Flat-Newton-v0, train until the wall-clock budget (--minutes, measured from the
   first rollout) would be exceeded by one more iteration, save.
4. Tear down: env -> TensorBoard (port verified released).

Besides RSL-RL's own scalars, every iteration logs ``Worm/mean_forward_speed_seg2_x``,
``Worm/mean_step_reward`` and ``health/diverged_worlds`` (WORM_SPEC.md item 12), exactly like the
2.3 trainer. The run folder gets ``train_summary.json`` for export_worm_v3.py.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from datetime import datetime

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
EXPERIMENT = "worm5_newton"

parser = argparse.ArgumentParser(description="Train Worm5 (Isaac Lab 3, Newton/MuJoCo-Warp, RSL-RL PPO) under a wall-clock budget.")
parser.add_argument("--task", type=str, default="Isaac-Worm5-Flat-Newton-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--minutes", type=float, default=30.0, help="Wall-clock training budget (from the first rollout).")
parser.add_argument("--max_iterations", type=int, default=None, help="Optional hard cap on PPO iterations.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--log_root", type=str, default=os.path.join(REPO, "ISAAC", "logs", "rsl_rl_v3"))
parser.add_argument("--tensorboard_port", type=int, default=6006)
parser.add_argument("--no_graph_decimation", action="store_true",
                    help="Use Isaac Lab's per-substep Python loop instead of Newton's CUDA-graphed decimation "
                         "(sim.use_newton_actuators=True, identical physics; see worm_env_cfg.py).")

from isaaclab_tasks.utils.sim_launcher import add_launcher_args, launch_simulation  # noqa: E402

add_launcher_args(parser)  # --device etc. (no --viz: headless)
args_cli = parser.parse_args()


# ------------------------------------------------------------------ TensorBoard first --
def port_in_use(port: int) -> bool:
    for host in ("127.0.0.1", "::1"):
        fam = socket.AF_INET6 if ":" in host else socket.AF_INET
        try:
            with socket.socket(fam, socket.SOCK_STREAM) as s:
                s.settimeout(0.5)
                if s.connect_ex((host, port)) == 0:
                    return True
        except OSError:
            pass
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("0.0.0.0", port))
    except OSError:
        return True
    return False


def start_tensorboard(logdir: str, port: int):
    os.makedirs(logdir, exist_ok=True)
    # --load_fast=false: no Rust data-server child process, so terminating this one process really
    # stops TensorBoard and frees the port.
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", logdir, "--port", str(port),
           "--bind_all", "--load_fast=false"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 60.0:
        if proc.poll() is not None:
            raise RuntimeError(f"TensorBoard exited immediately (code {proc.returncode}) - is port {port} free?")
        if port_in_use(port):
            print(f"[train_worm_v3] TensorBoard up: http://localhost:{port}  (logdir {logdir}, pid {proc.pid})")
            return proc
        time.sleep(0.5)
    proc.terminate()
    raise RuntimeError(f"TensorBoard did not start listening on port {port} within 60 s")


def stop_tensorboard(proc, port: int):
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=15)
    for _ in range(40):
        if not port_in_use(port):
            print(f"[train_worm_v3] TensorBoard stopped; port {port} released.")
            return
        time.sleep(0.25)
    print(f"[train_worm_v3] WARNING: TensorBoard (pid {proc.pid}) stopped but port {port} still answers.")


if port_in_use(args_cli.tensorboard_port):
    print(f"[train_worm_v3] ERROR: port {args_cli.tensorboard_port} is already in use by another process.\n"
          f"                Pick a free one with --tensorboard_port <port>. Nothing was started and no "
          f"process was touched.", file=sys.stderr)
    sys.exit(2)

TB_LOGDIR = os.path.abspath(os.path.join(args_cli.log_root, EXPERIMENT))
_tb = start_tensorboard(TB_LOGDIR, args_cli.tensorboard_port)

import importlib.metadata as metadata  # noqa: E402
import statistics  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import worm_tasks_v3  # noqa: E402,F401  (registers the tasks)
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl.utils import check_nan  # noqa: E402
from worm_tasks_v3 import spec  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def versions() -> dict:
    out = {}
    for p in ("isaaclab", "isaaclab_newton", "isaaclab_rl", "newton", "mujoco", "mujoco-warp", "warp-lang", "torch",
              "rsl-rl-lib"):
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


class WormRunner(OnPolicyRunner):
    """OnPolicyRunner.learn() with a wall-clock deadline and the worm scalars (same as the 2.3 trainer).

    The loop body is rsl-rl 5.0.1's own (rollout -> compute_returns -> update -> log -> save); only the
    stop condition and the extra logging are added.
    """

    def learn_budget(self, budget_s: float, max_iterations: int) -> dict:
        self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        robot = self.env.unwrapped.scene["robot"]
        seg2 = robot.body_names.index(spec.REF_BODY)


        t_start = time.time()
        last_iter_s = 0.0
        stop_reason = "max_iterations"
        it = self.current_learning_iteration
        total_it = it + max_iterations
        speed, step_rew = 0.0, 0.0
        tm = self.env.unwrapped.termination_manager
        diverged_total = 0
        while it < total_it:
            elapsed = time.time() - t_start
            if elapsed + last_iter_s > budget_s:
                stop_reason = "wall_clock_budget"
                break
            start = time.time()
            speed_sum = torch.zeros((), device=self.device)
            rew_sum = torch.zeros((), device=self.device)
            div_sum = torch.zeros((), device=self.device, dtype=torch.long)
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
                    speed_sum += robot.data.body_link_lin_vel_w.torch[:, seg2, 0].mean().to(self.device)
                    rew_sum += rewards.mean()
                    div_sum += tm.get_term("diverged").sum().to(self.device)
                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            last_iter_s = collect_time + learn_time
            self.current_learning_iteration = it

            n = self.cfg["num_steps_per_env"]
            speed, step_rew = float(speed_sum) / n, float(rew_sum) / n
            diverged = int(div_sum)
            diverged_total += diverged
            if self.logger.writer is not None:
                w = self.logger.writer
                w.add_scalar("Worm/mean_forward_speed_seg2_x", speed, it)
                w.add_scalar("Worm/mean_step_reward", step_rew, it)
                w.add_scalar("health/diverged_worlds", diverged, it)
                w.add_scalar("health/diverged_worlds_total", diverged_total, it)
                w.add_scalar("Worm/wall_minutes", (time.time() - t_start) / 60.0, it)
            ep_rew = statistics.mean(self.logger.rewbuffer) if len(self.logger.rewbuffer) else float("nan")
            self.logger.log(
                it=it, start_it=0, total_it=total_it, collect_time=collect_time, learn_time=learn_time,
                loss_dict=loss_dict, learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std, rnd_weight=None, print_minimal=True,
            )
            sps = n * self.env.num_envs / max(collect_time + learn_time, 1e-9)
            print(f"[train_worm_v3] it {it:5d}  steps {self.logger.tot_timesteps:>11d}  "
                  f"{(time.time() - t_start) / 60:6.2f} min  {sps:8.0f} steps/s (collect {collect_time:.2f}s learn {learn_time:.2f}s)  "
                  f"mean ep reward {ep_rew:8.3f}  step reward {step_rew:+.5f}  seg2 speed {speed:+.4f} m/s  diverged {diverged}")
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))
            it += 1

        wall_s = time.time() - t_start
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.writer.flush()
            self.logger.stop_logging_writer()
        return {
            "iterations": self.current_learning_iteration + 1,
            "stepsTrained": int(self.logger.tot_timesteps),
            "wallMinutes": wall_s / 60.0,
            "stopReason": stop_reason,
            "finalMeanEpisodeReward": statistics.mean(self.logger.rewbuffer) if len(self.logger.rewbuffer) else None,
            "finalMeanStepReward": step_rew,
            "finalMeanForwardSpeedSeg2": speed,
            "divergedWorldsTotal": diverged_total,
        }


def main():
    t_process = time.time()
    env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
    )
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.no_graph_decimation:
        env_cfg.sim.use_newton_actuators = False
    if getattr(args_cli, "device", None):
        env_cfg.sim.device = args_cli.device
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = env_cfg.sim.device
    max_it = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(TB_LOGDIR, stamp + (f"_{args_cli.run_name}" if args_cli.run_name else ""))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train_worm_v3] log dir: {log_dir}")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        try:
            u = env.unwrapped
            assert abs(u.physics_dt - spec.PHYSICS_DT) < 1e-9 and u.cfg.decimation == spec.DECIMATION, "timing != spec"
            assert abs(u.step_dt - spec.STEP_DT) < 1e-9 and u.max_episode_length == 1000, (u.step_dt, u.max_episode_length)
            wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = WormRunner(wenv, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
            runner.add_git_repo_to_log(__file__)
            dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

            print(f"[train_worm_v3] {args_cli.num_envs} envs, budget {args_cli.minutes:.1f} min, "
                  f"backend Newton + MuJoCo-Warp (kit-less)")
            summary = runner.learn_budget(args_cli.minutes * 60.0, max_it)
            summary.update({
                "tool": "isaaclab3-newton-mjwarp-rsl_rl",
                "task": args_cli.task,
                "numEnvs": args_cli.num_envs,
                "budgetMinutes": args_cli.minutes,
                "processMinutes": (time.time() - t_process) / 60.0,
                "stepsPerSecond": summary["stepsTrained"] / max(summary["wallMinutes"] * 60.0, 1e-9),
                "physics": "Newton SolverMuJoCo (mujoco_warp): Newton solver, 10 iterations, 8 ls iterations, "
                           "implicitfast, pyramidal cone, tolerance 1e-8, dt 0.005 s",
                "versions": versions(),
                "graphDecimation": bool(getattr(env_cfg.sim, "use_newton_actuators", False)),
                "seed": args_cli.seed,
            })
            with open(os.path.join(log_dir, "train_summary.json"), "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[train_worm_v3] done: {json.dumps(summary)}")
        finally:
            env.close()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        stop_tensorboard(_tb, args_cli.tensorboard_port)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
