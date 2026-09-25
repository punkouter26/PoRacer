#!/usr/bin/env python3
"""Train the quad with RSL-RL PPO in Isaac Lab 3 on Newton + MuJoCo-Warp (kit-less), headless.

The Isaac Lab 3 trainer of training/quad/QUAD_SPEC.md (the other is training/quad/mujoco). Run with the
Isaac Lab 3 venv's python::

    ISAAC\\isaaclab3\\.venv\\Scripts\\python.exe ISAAC\\scripts\\train_quad_v3.py --minutes 30

Order of events (project rule C: TensorBoard first):

1. Check that --tensorboard_port is free. If anything already listens there, print an error and exit -
   no process is ever killed.
2. Start TensorBoard on ISAAC/logs/rsl_rl_v3/quad_newton and wait until it answers.
3. Build Isaac-Quad-Flat-Newton-v0, train until the wall-clock budget (--minutes, measured from the first
   rollout) would be exceeded by one more iteration, checkpoint every 50 iterations, save the last one.
4. Tear down: env -> TensorBoard (port verified released).

Besides RSL-RL's scalars, every iteration logs Quad/* (torso forward speed, step reward, falls, feet in
contact), health/diverged_worlds (WORM_SPEC item 12) and Policy/kl_update: the mean KL between the
rollout policy and the updated policy on a 16k-sample slice of the batch, plus Policy/lr_used.

Learning rate (QUAD_SPEC round 4): RSL-RL's adaptive-KL rule, desired KL 0.01, applied per mini-batch,
floor 1e-5, CEILING 3e-4 (quad_tasks_v3/agents/capped_ppo.py). Why: the Isaac3Worm run collapsed at 18 min
with a constant 3e-4; by then its action std was 0.02-0.1 and a replay of its model_1350 checkpoint showed
per-update KL of median 0.030 with spikes to 0.15 and a positive surrogate loss. Under the adaptive rule
the same replay held KL at ~0.008 with no loss of speed.

train_summary.json (read by export_quad_v3.py) records the best checkpoint by the mean speed-tracking
reward of the 10 iterations before each save, so a collapsed run can still export its best brain
(QUAD_SPEC "PPO and budget").
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
EXPERIMENT = "quad_newton"
SPEC_LR = 3.0e-4

parser = argparse.ArgumentParser(description="Train the quad (Isaac Lab 3, Newton/MuJoCo-Warp, RSL-RL PPO) under a wall-clock budget.")
parser.add_argument("--task", type=str, default="Isaac-Quad-Flat-Newton-v0")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--minutes", type=float, default=30.0, help="Wall-clock training budget (from the first rollout).")
parser.add_argument("--max_iterations", type=int, default=None, help="Optional hard cap on PPO iterations.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", type=str, default="")
parser.add_argument("--log_root", type=str, default=os.path.join(REPO, "ISAAC", "logs", "rsl_rl_v3"))
parser.add_argument("--tensorboard_port", type=int, default=6006)
parser.add_argument("--no_graph_decimation", action="store_true",
                    help="Use Isaac Lab's per-substep Python loop instead of Newton's CUDA-graphed decimation.")

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
    # --load_fast=false: no Rust data-server child, so terminating this one process frees the port
    cmd = [sys.executable, "-m", "tensorboard.main", "--logdir", logdir, "--port", str(port),
           "--bind_all", "--load_fast=false"]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.time()
    while time.time() - t0 < 60.0:
        if proc.poll() is not None:
            raise RuntimeError(f"TensorBoard exited immediately (code {proc.returncode}) - is port {port} free?")
        if port_in_use(port):
            print(f"[train_quad_v3] TensorBoard up: http://localhost:{port}  (logdir {logdir}, pid {proc.pid})")
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
            print(f"[train_quad_v3] TensorBoard stopped; port {port} released.")
            return
        time.sleep(0.25)
    print(f"[train_quad_v3] WARNING: TensorBoard (pid {proc.pid}) stopped but port {port} still answers.")


if port_in_use(args_cli.tensorboard_port):
    print(f"[train_quad_v3] ERROR: port {args_cli.tensorboard_port} is already in use by another process.\n"
          f"               Pick a free one with --tensorboard_port <port>. Nothing was started and no "
          f"process was touched.", file=sys.stderr)
    sys.exit(2)

TB_LOGDIR = os.path.abspath(os.path.join(args_cli.log_root, EXPERIMENT))
_tb = start_tensorboard(TB_LOGDIR, args_cli.tensorboard_port)

import importlib.metadata as metadata  # noqa: E402
import statistics  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import quad_tasks_v3  # noqa: E402,F401  (registers the tasks)
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from quad_tasks_v3 import spec  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from rsl_rl.utils import check_nan  # noqa: E402

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


class QuadRunner(OnPolicyRunner):
    """OnPolicyRunner.learn() with a wall-clock deadline, the quad scalars, a measured per-update KL, the
    optional KL guard and best-checkpoint tracking. The loop body is rsl-rl 5.0.1's own
    (rollout -> compute_returns -> update -> log -> save)."""

    def update_kl(self, n_sample: int = 16384) -> float:
        """Mean KL(rollout policy || updated policy) on a random slice of the batch just trained on.
        ``storage.clear()`` only rewinds the write index, so the batch is still there after update()."""
        st = self.alg.storage
        t, n = st.actions.shape[:2]
        idx = torch.randint(0, t * n, (min(n_sample, t * n),), device=self.device)
        obs = st.observations.flatten(0, 1)[idx]
        old = tuple(p.flatten(0, 1)[idx] for p in st.distribution_params)
        with torch.inference_mode():
            self.alg.actor(obs, stochastic_output=True)
            new = self.alg.actor.output_distribution_params
            return float(self.alg.actor.get_kl_divergence(old, new).mean())

    def learn_budget(self, budget_s: float, max_iterations: int) -> dict:
        self.env.episode_length_buf = torch.randint_like(self.env.episode_length_buf, high=int(self.env.max_episode_length))
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        self.logger.init_logging_writer()
        u = self.env.unwrapped
        robot = u.scene["robot"]
        torso = robot.body_names.index(spec.TORSO)
        tm = u.termination_manager
        rm = u.reward_manager
        foot = rm.get_term_cfg("foot_slip").func
        speed_col = rm.active_terms.index("speed")
        air_col = rm.active_terms.index("feet_air_time")
        flight_col = rm.active_terms.index("flight")

        t_start = time.time()
        last_iter_s = 0.0
        stop_reason = "max_iterations"
        it = self.current_learning_iteration
        total_it = it + max_iterations
        speed = step_rew = 0.0
        diverged_total = falls_total = 0
        recent_track = []
        best = {"checkpoint": None, "iteration": None, "meanSpeedTrackReward": -1.0, "meanForwardSpeed": None}
        saves = []
        while it < total_it:
            elapsed = time.time() - t_start
            if elapsed + last_iter_s > budget_s:
                stop_reason = "wall_clock_budget"
                break
            start = time.time()
            n = self.cfg["num_steps_per_env"]
            speed_sum = torch.zeros((), device=self.device)
            rew_sum = torch.zeros((), device=self.device)
            track_sum = torch.zeros((), device=self.device)
            feet_sum = torch.zeros((), device=self.device)
            air_sum = torch.zeros((), device=self.device)
            flight_sum = torch.zeros((), device=self.device)
            div_sum = torch.zeros((), device=self.device, dtype=torch.long)
            fall_sum = torch.zeros((), device=self.device, dtype=torch.long)
            with torch.inference_mode():
                for _ in range(n):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
                    speed_sum += robot.data.body_link_lin_vel_w.torch[:, torso, 0].mean().to(self.device)
                    rew_sum += rewards.mean()
                    # RewardManager._step_reward holds weight * raw term (per second); raw speed kernel in [0, 1]
                    track_sum += rm._step_reward[:, speed_col].mean().to(self.device) / spec.W_SPEED
                    feet_sum += foot.contact.float().sum(dim=1).mean().to(self.device)
                    air_sum += rm._step_reward[:, air_col].mean().to(self.device) * u.step_dt  # touchdown terms per step
                    flight_sum += rm._step_reward[:, flight_col].mean().to(self.device) / spec.W_FLIGHT  # flight fraction
                    div_sum += tm.get_term("diverged").sum().to(self.device)
                    fall_sum += tm.get_term("fall").sum().to(self.device)
                collect_time = time.time() - start
                start = time.time()
                self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            kl = self.update_kl()
            lr_used = self.alg.learning_rate  # after this update's per-mini-batch adaptive steps (<= 3e-4)
            learn_time = time.time() - start
            last_iter_s = collect_time + learn_time
            self.current_learning_iteration = it

            speed, step_rew = float(speed_sum) / n, float(rew_sum) / n
            track, feet = float(track_sum) / n, float(feet_sum) / n
            diverged, falls = int(div_sum), int(fall_sum)
            diverged_total += diverged
            falls_total += falls
            recent_track = (recent_track + [track])[-10:]
            if self.logger.writer is not None:
                w = self.logger.writer
                w.add_scalar("Quad/mean_forward_speed_torso_x", speed, it)
                w.add_scalar("Quad/mean_step_reward", step_rew, it)
                w.add_scalar("Quad/mean_speed_track_reward", track, it)
                w.add_scalar("Quad/mean_feet_in_contact", feet, it)
                w.add_scalar("Quad/air_time_reward_per_s", float(air_sum) / (n * u.step_dt), it)
                w.add_scalar("Quad/flight_fraction_debounced", float(flight_sum) / n, it)
                w.add_scalar("Quad/falls_per_1000_env_steps", 1000.0 * falls / (n * self.env.num_envs), it)
                w.add_scalar("Quad/wall_minutes", (time.time() - t_start) / 60.0, it)
                w.add_scalar("health/diverged_worlds", diverged, it)
                w.add_scalar("health/diverged_worlds_total", diverged_total, it)
                w.add_scalar("Policy/kl_update", kl, it)
                w.add_scalar("Policy/lr_used", lr_used, it)
            ep_rew = statistics.mean(self.logger.rewbuffer) if len(self.logger.rewbuffer) else float("nan")
            ep_len = statistics.mean(self.logger.lenbuffer) if len(self.logger.lenbuffer) else float("nan")
            self.logger.log(
                it=it, start_it=0, total_it=total_it, collect_time=collect_time, learn_time=learn_time,
                loss_dict=loss_dict, learning_rate=lr_used,
                action_std=self.alg.get_policy().output_std, rnd_weight=None, print_minimal=True,
            )
            sps = n * self.env.num_envs / max(collect_time + learn_time, 1e-9)
            print(f"[train_quad_v3] it {it:5d}  steps {self.logger.tot_timesteps:>11d}  {(time.time() - t_start) / 60:6.2f} min  "
                  f"{sps:8.0f} steps/s  ep reward {ep_rew:8.3f} len {ep_len:6.1f}  speed {speed:+.3f} m/s  "
                  f"track {track:.3f}  feet {feet:.2f}  falls {falls:5d}  kl {kl:.4f} lr {lr_used:.2e}  diverged {diverged}")
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                path = os.path.join(self.logger.log_dir, f"model_{it}.pt")
                self.save(path)
                score = sum(recent_track) / len(recent_track)
                saves.append({"iteration": it, "meanSpeedTrackReward": score, "meanForwardSpeed": speed})
                if score > best["meanSpeedTrackReward"]:
                    best = {"checkpoint": os.path.relpath(path, REPO), "iteration": it,
                            "meanSpeedTrackReward": score, "meanForwardSpeed": speed}
            it += 1

        wall_s = time.time() - t_start
        final = None
        if self.logger.writer is not None:
            final = os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt")
            self.save(final)
            self.logger.writer.flush()
            self.logger.stop_logging_writer()
        final_score = sum(recent_track) / max(len(recent_track), 1)
        return {
            "iterations": self.current_learning_iteration + 1,
            "stepsTrained": int(self.logger.tot_timesteps),
            "wallMinutes": wall_s / 60.0,
            "stopReason": stop_reason,
            "finalCheckpoint": os.path.relpath(final, REPO) if final else None,
            "finalMeanEpisodeReward": statistics.mean(self.logger.rewbuffer) if len(self.logger.rewbuffer) else None,
            "finalMeanStepReward": step_rew,
            "finalMeanForwardSpeedTorso": speed,
            "finalMeanSpeedTrackReward": final_score,
            "bestCheckpoint": best,
            "collapseSuspected": bool(best["meanSpeedTrackReward"] > 0.2 and final_score < 0.5 * best["meanSpeedTrackReward"]),
            "savedCheckpoints": saves,
            "divergedWorldsTotal": diverged_total,
            "fallsTotal": falls_total,
            "lrSchedule": "adaptive KL 0.01, per mini-batch, [1e-5, 3e-4]",
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
    a = agent_cfg.algorithm
    assert (a.learning_rate == SPEC_LR and a.schedule == "adaptive" and a.desired_kl == 0.01
            and a.class_name.endswith(":CappedPPO")), "PPO cfg != QUAD_SPEC (round 4+: adaptive KL 0.01, lr <= 3e-4)"
    max_it = args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = os.path.join(TB_LOGDIR, stamp + (f"_{args_cli.run_name}" if args_cli.run_name else ""))
    os.makedirs(log_dir, exist_ok=True)
    print(f"[train_quad_v3] log dir: {log_dir}")

    with launch_simulation(env_cfg, args_cli):
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
        try:
            u = env.unwrapped
            assert abs(u.physics_dt - spec.PHYSICS_DT) < 1e-9 and u.cfg.decimation == spec.DECIMATION, "timing != spec"
            assert abs(u.step_dt - spec.STEP_DT) < 1e-9 and u.max_episode_length == spec.EPISODE_STEPS, (u.step_dt, u.max_episode_length)
            wenv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
            runner = QuadRunner(wenv, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
            runner.add_git_repo_to_log(__file__)
            dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
            dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

            print(f"[train_quad_v3] {args_cli.num_envs} envs, budget {args_cli.minutes:.1f} min, backend Newton + MuJoCo-Warp "
                  f"(kit-less), lr adaptive KL 0.01 in [1e-5, 3e-4]")
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
            print(f"[train_quad_v3] done: {json.dumps({k: v for k, v in summary.items() if k != 'savedCheckpoints'})}")
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
