"""PPO trainer for any CreatureEnv (lifted from training/worm/mujoco/train_worm_mujoco.py).

TensorBoard starts FIRST (AGENTS.md rule C) on --tensorboard-port with logdir <runs_dir>.
If the port is taken the script says so and exits: it never kills a process it did not
start, and it stops the TensorBoard it started when training ends.

PPO is the worm spec's, following RSL-RL where the spec is silent (item 11):
  * Gaussian policy, one state-independent log-std per action (init 0.5), actions not
    squashed (the env clips). 3x128 ELU actor and critic, default PyTorch init.
  * Observation normaliser: running mean/var over every returned observation,
    (x - mean)/(std + 0.01), updated throughout training, baked into the ONNX.
  * Timeout bootstrap r += gamma * V(s_t) on timeout steps (RSL-RL); terminations
    (falls, health guard) get no bootstrap.
  * Advantages normalised over the whole rollout; clipped value loss (0.2);
    loss = surrogate + 1.0 * value - 0.005 * entropy; Adam; grad-norm clip 1.0;
    5 epochs x 4 shuffled minibatches; gamma 0.99, lambda 0.95.
  * Learning rate: TrainSetup.lr_schedule = "fixed" (3e-4, the worm) or "adaptive"
    (RSL-RL's adaptive-KL schedule, capped): before each minibatch's step, the analytic KL
    between the rollout's Gaussian (stored means, std at rollout time) and the current one,
    summed over actions and averaged over the minibatch; KL > 2 x desired -> lr / 1.5
    (floor lr_min 1e-5); KL < desired / 2 -> lr x 1.5 (ceiling lr_max = 3e-4). The lr
    starts at lr_max and carries over between iterations.
Wall-clock budget counts from the first rollout (item 9); the loop stops before an
iteration it could not finish inside the budget.

Outputs in <runs_dir>/<run_id>/: TensorBoard events, config.json, metrics.jsonl (one line
per iteration), model_<it>.pt every --save-every iterations plus model_final.pt,
checkpoints.jsonl (the metrics at each checkpoint, for picking the best one before a
collapse), train_info.json; and the ONNX of the final policy.
"""

from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from .export import HIDDEN, INIT_STD, ActorCritic, export_onnx, save_checkpoint

REPO = Path(__file__).resolve().parents[2]
TENSORBOARD_EXE = REPO / ".venv-mjwarp" / "Scripts" / "tensorboard.exe"

# Spec PPO constants. Deliberately not flags: both trainers of a creature must use these.
GAMMA, LAMBDA, CLIP = 0.99, 0.95, 0.2
EPOCHS, MINIBATCHES, LR = 5, 4, 3e-4
ENTROPY_COEF, VALUE_COEF, MAX_GRAD_NORM = 0.005, 1.0, 1.0


@dataclass
class TrainSetup:
    name: str                                   # creature name, e.g. "quad"
    make_env: Callable[[int, int], object]      # (num_envs, seed) -> CreatureEnv (training mode)
    runs_dir: Path
    onnx_out: Path
    trainer_script: str
    obs_description: str
    eval_command: str = ""
    tool: str = "mujoco"
    default_port: int = 6006
    default_minutes: float = 30.0
    lr_schedule: str = "fixed"                  # "fixed" | "adaptive" (KL-controlled, capped)
    desired_kl: float = 0.01
    lr_min: float = 1e-5
    lr_max: float = LR


# ---------------------------------------------------------------- tensorboard
def port_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def start_tensorboard(runs_dir: Path, port: int) -> subprocess.Popen:
    if not TENSORBOARD_EXE.exists():
        print(f"ERROR: TensorBoard not found at {TENSORBOARD_EXE}. Refusing to train blind.")
        sys.exit(2)
    if not port_free(port):
        print(f"ERROR: port {port} is already in use by another process.")
        print("       This script will not stop a process it did not start.")
        print("       Free the port yourself, or pass --tensorboard-port <free port>.")
        sys.exit(2)
    runs_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([str(TENSORBOARD_EXE), "--logdir", str(runs_dir), "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):                         # up to ~30 s for it to bind
        if proc.poll() is not None:
            print(f"ERROR: TensorBoard exited immediately (code {proc.returncode}).")
            sys.exit(2)
        if not port_free(port):
            break
        time.sleep(0.5)
    print(f"TensorBoard: http://localhost:{port}/  (logdir {runs_dir})", flush=True)
    return proc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def prune_runs(runs_dir: Path, keep: int, current: str) -> None:
    """Rule C: clear dead runs so the live curve is readable. Keeps the newest `keep`,
    never touches a run written to in the last 30 minutes."""
    if keep < 0 or not runs_dir.exists():
        return
    now = time.time()

    def newest(run: Path) -> float:
        try:
            return max((p.stat().st_mtime for p in run.rglob("*")), default=run.stat().st_mtime)
        except OSError:
            return now

    runs = [p for p in runs_dir.iterdir() if p.is_dir() and p.name != current
            and now - newest(p) > 1800.0]
    runs.sort(key=newest)
    for old in runs[:max(0, len(runs) - keep)]:
        try:
            shutil.rmtree(old)
            print(f"pruned stale run {old.name}")
        except OSError as exc:
            print(f"!! could not prune {old.name}: {exc}")


# ---------------------------------------------------------------- entry point
def main(setup: TrainSetup, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"PPO for {setup.name} on MuJoCo Warp")
    parser.add_argument("--minutes", type=float, default=setup.default_minutes,
                        help="wall-clock training budget, from the first rollout")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--tensorboard-port", type=int, default=setup.default_port)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout", type=int, default=24)
    parser.add_argument("--save-every", type=int, default=50, help="checkpoint period, iterations")
    parser.add_argument("--keep-runs", type=int, default=3,
                        help="older idle runs kept when pruning (-1 = never prune)")
    parser.add_argument("--onnx-out", type=Path, default=setup.onnx_out)
    args = parser.parse_args(argv)

    run_id = args.run_id or time.strftime(f"{setup.name}_{setup.tool}_%Y%m%d_%H%M")
    logdir = setup.runs_dir / run_id

    tb = start_tensorboard(setup.runs_dir, args.tensorboard_port)   # rule C: first
    try:
        prune_runs(setup.runs_dir, args.keep_runs, run_id)
        logdir.mkdir(parents=True, exist_ok=True)
        return train(setup, args, logdir, run_id)
    finally:
        stop_process(tb)                                           # stop what we started
        print("TensorBoard stopped.")


def train(setup: TrainSetup, args, logdir: Path, run_id: str) -> int:
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    n, horizon = args.num_envs, args.rollout

    setup_started = time.perf_counter()
    env = setup.make_env(n, args.seed)
    obs_size, act_size = env.obs_size, env.action_size
    term_names = tuple(env.term_names)
    policy = ActorCritic(obs_size, act_size).to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=LR)
    writer = SummaryWriter(str(logdir))
    (logdir / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "creature": setup.name, "obs": obs_size, "act": act_size, "hidden": list(HIDDEN),
        "activation": "elu", "init_std": INIT_STD, "gamma": GAMMA, "lambda": LAMBDA,
        "clip": CLIP, "epochs": EPOCHS, "minibatches": MINIBATCHES, "lr": LR,
        "entropy_coef": ENTROPY_COEF, "value_coef": VALUE_COEF, "max_grad_norm": MAX_GRAD_NORM,
        "policy_dt": env.dt, "episode_steps": env.episode_steps,
        "cuda_graph": env._graph is not None, "pushes": env.pushes,
        "lr_schedule": setup.lr_schedule, "desired_kl": setup.desired_kl,
        "lr_min": setup.lr_min, "lr_max": setup.lr_max,
        "mjcf": str(env.cfg.mjcf),
    }, indent=2))
    print(f"run {run_id}: {n} envs x {horizon} steps = {n * horizon} samples/iter  "
          f"(setup {time.perf_counter() - setup_started:.1f} s, "
          f"cuda graph {'on' if env._graph is not None else 'off'})", flush=True)

    obs_buf = torch.zeros(horizon, n, obs_size, device=device)
    act_buf = torch.zeros(horizon, n, act_size, device=device)
    mu_buf = torch.zeros(horizon, n, act_size, device=device)
    adaptive = setup.lr_schedule == "adaptive"
    if setup.lr_schedule not in ("fixed", "adaptive"):
        raise ValueError(f"unknown lr_schedule {setup.lr_schedule!r}")
    lr = setup.lr_max if adaptive else LR
    for group in optimiser.param_groups:
        group["lr"] = lr
    logp_buf = torch.zeros(horizon, n, device=device)
    val_buf = torch.zeros(horizon, n, device=device)
    rew_buf = torch.zeros(horizon, n, device=device)
    done_buf = torch.zeros(horizon, n, device=device)
    nefc = getattr(env.wd, "nefc", None)
    nefc_view = None
    if nefc is not None:
        import warp as wp
        nefc_view = wp.to_torch(nefc)

    obs = env.observation()
    budget = args.minutes * 60.0
    started = time.perf_counter()
    total_steps, iteration, last_iter_time = 0, 0, 0.0
    diverged_total, falls_total, episodes_total = 0, 0, 0
    metrics_file = open(logdir / "metrics.jsonl", "w", encoding="utf-8")
    recent: list[dict] = []

    while True:
        elapsed = time.perf_counter() - started
        if iteration > 0 and elapsed + last_iter_time > budget:
            break
        iteration += 1
        iter_started = time.perf_counter()
        sums = {k: torch.zeros((), device=device) for k in ("reward",) + term_names}
        ep_sum = {k: torch.zeros((), device=device) for k in
                  ("return", "speed", "distance", "length", "count", "falls")}
        diverged = torch.zeros((), device=device)
        pushes = torch.zeros((), device=device)
        illegal = torch.zeros((), device=device)

        # ---- rollout
        with torch.no_grad():
            for t in range(horizon):
                dist = policy.distribution(obs)
                action = dist.sample()
                obs_buf[t] = obs
                act_buf[t] = action
                mu_buf[t] = dist.mean
                logp_buf[t] = dist.log_prob(action).sum(-1)
                val_buf[t] = policy.value(obs)

                obs, reward, done, timeouts, info = env.step(action)
                policy.norm.update(obs)
                reward = torch.nan_to_num(reward)
                sums["reward"] += reward.mean()
                for key in term_names:
                    sums[key] += info[key].mean()
                rew_buf[t] = reward + GAMMA * val_buf[t] * timeouts.float()
                done_buf[t] = done.float()
                # Completed episodes only: a health-guard reset is not a finished run.
                d = (done & ~info["diverged"]).float()
                ep_sum["return"] += (info["ep_return"] * d).sum()
                ep_sum["speed"] += (info["ep_speed"] * d).sum()
                ep_sum["distance"] += (info["ep_distance"] * d).sum()
                ep_sum["length"] += (info["ep_length"] * d).sum()
                ep_sum["count"] += d.sum()
                ep_sum["falls"] += info["terminated"].float().sum()
                diverged += info["diverged"].float().sum()
                pushes += info["pushed"].float().sum()
                illegal += info["illegal_contact"].float().sum()
            last_value = policy.value(obs)

        peak = env.contact_peak()
        if peak >= env.naconmax:
            print(f"ABORT iteration {iteration}: contact overflow {peak} >= {env.naconmax}. "
                  f"Raise nconmax in the creature config.")
            return 1
        if nefc_view is not None:
            efc_peak = int(nefc_view.max().item())
            if efc_peak >= env.cfg.njmax:
                print(f"ABORT iteration {iteration}: constraint rows {efc_peak} >= njmax "
                      f"{env.cfg.njmax}. Raise njmax in the creature config.")
                return 1
        else:
            efc_peak = -1

        # ---- GAE
        adv_buf = torch.zeros_like(rew_buf)
        gae = torch.zeros(n, device=device)
        for t in reversed(range(horizon)):
            next_value = last_value if t == horizon - 1 else val_buf[t + 1]
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + not_done * GAMMA * next_value - val_buf[t]
            gae = delta + not_done * GAMMA * LAMBDA * gae
            adv_buf[t] = gae
        ret_buf = adv_buf + val_buf

        b_obs = obs_buf.reshape(-1, obs_size)
        b_act = act_buf.reshape(-1, act_size)
        b_mu = mu_buf.reshape(-1, act_size)
        old_std = policy.log_std.exp().detach().clone()
        b_logp = logp_buf.reshape(-1)
        b_val = val_buf.reshape(-1)
        b_ret = ret_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ---- PPO update
        total = b_obs.shape[0]
        mb = total // MINIBATCHES
        stats = []
        for _ in range(EPOCHS):
            perm = torch.randperm(total, device=device)
            for k in range(MINIBATCHES):
                sel = perm[k * mb:(k + 1) * mb]
                dist = policy.distribution(b_obs[sel])
                with torch.no_grad():
                    new_std = dist.stddev
                    kl = (torch.log(new_std / old_std)
                          + (old_std.pow(2) + (b_mu[sel] - dist.mean).pow(2)) / (2.0 * new_std.pow(2))
                          - 0.5).sum(-1).mean()
                if adaptive:
                    kl_value = kl.item()
                    if kl_value > 2.0 * setup.desired_kl:
                        lr = max(setup.lr_min, lr / 1.5)
                    elif 0.0 < kl_value < 0.5 * setup.desired_kl:
                        lr = min(setup.lr_max, lr * 1.5)
                    for group in optimiser.param_groups:
                        group["lr"] = lr
                logp = dist.log_prob(b_act[sel]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = torch.exp(logp - b_logp[sel])
                adv = b_adv[sel]
                surrogate = torch.max(-adv * ratio,
                                      -adv * torch.clamp(ratio, 1.0 - CLIP, 1.0 + CLIP)).mean()
                value = policy.value(b_obs[sel])
                v_clipped = b_val[sel] + (value - b_val[sel]).clamp(-CLIP, CLIP)
                value_loss = torch.max((value - b_ret[sel]).pow(2),
                                       (v_clipped - b_ret[sel]).pow(2)).mean()
                loss = surrogate + VALUE_COEF * value_loss - ENTROPY_COEF * entropy
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), MAX_GRAD_NORM)
                optimiser.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (logp - b_logp[sel])).mean()
                stats.append(torch.stack((surrogate.detach(), value_loss.detach(),
                                          entropy.detach(), approx_kl, kl)))

        total_steps += n * horizon
        last_iter_time = time.perf_counter() - iter_started
        elapsed = time.perf_counter() - started

        # ---- logging (one host sync per iteration)
        stacked = torch.stack(stats)
        surr, vloss, ent, kl, kl_mean = stacked.mean(0).tolist()
        kl_max = stacked[:, 4].max().item()
        means = {k: (v / horizon).item() for k, v in sums.items()}
        episodes = int(ep_sum["count"].item())
        falls = int(ep_sum["falls"].item())
        diverged_n = int(diverged.item())
        diverged_total += diverged_n
        falls_total += falls
        episodes_total += episodes
        sps = n * horizon / last_iter_time
        step = total_steps
        std = policy.log_std.exp().mean().item()
        writer.add_scalar("train/mean_reward_per_step", means["reward"], step)
        if env.cfg.speed_term in means:
            writer.add_scalar("train/mean_forward_speed", means[env.cfg.speed_term], step)
        writer.add_scalar("perf/steps_per_second", sps, step)
        writer.add_scalar("perf/total_steps", total_steps, step)
        writer.add_scalar("perf/wall_minutes", elapsed / 60.0, step)
        for key in term_names:
            writer.add_scalar(f"terms/{key}", means[key], step)
        writer.add_scalar("loss/surrogate", surr, step)
        writer.add_scalar("loss/value", vloss, step)
        writer.add_scalar("loss/entropy", ent, step)
        writer.add_scalar("loss/approx_kl", kl, step)
        writer.add_scalar("loss/kl_mean", kl_mean, step)
        writer.add_scalar("loss/kl_max", kl_max, step)
        writer.add_scalar("loss/learning_rate", lr, step)
        writer.add_scalar("policy/mean_std", std, step)
        writer.add_scalar("health/diverged_worlds", diverged_n, step)
        writer.add_scalar("health/contact_peak", peak, step)
        if efc_peak >= 0:
            writer.add_scalar("health/constraint_rows_peak", efc_peak, step)
        writer.add_scalar("episode/falls", falls, step)
        writer.add_scalar("episode/pushes", pushes.item(), step)
        writer.add_scalar("episode/falls_by_contact", illegal.item(), step)
        record = {"it": iteration, "steps": total_steps, "minutes": elapsed / 60.0,
                  "reward": means["reward"], "sps": sps, "std": std, "falls": falls,
                  "episodes": episodes, "diverged": diverged_n, "lr": lr, "kl_mean": kl_mean,
                  "falls_by_contact": int(illegal.item()),
                  "kl_max": kl_max,
                  **{f"term_{k}": means[k] for k in term_names}}
        ep_text = ""
        if episodes > 0:
            ep_ret = (ep_sum["return"] / episodes).item()
            ep_spd = (ep_sum["speed"] / episodes).item()
            ep_dist = (ep_sum["distance"] / episodes).item()
            ep_len = (ep_sum["length"] / episodes).item() * env.dt
            fall_rate = falls / episodes
            writer.add_scalar("episode/mean_return", ep_ret, step)
            writer.add_scalar("episode/mean_forward_speed", ep_spd, step)
            writer.add_scalar("episode/mean_distance", ep_dist, step)
            writer.add_scalar("episode/mean_length_s", ep_len, step)
            writer.add_scalar("episode/fall_rate", fall_rate, step)
            record.update(ep_return=ep_ret, ep_speed=ep_spd, ep_distance=ep_dist,
                          ep_length_s=ep_len, fall_rate=fall_rate)
            ep_text = (f"  ep {episodes:5d} ret {ep_ret:6.2f} len {ep_len:4.1f}s "
                       f"dist {ep_dist:5.2f} m fall {fall_rate:4.0%}")
        metrics_file.write(json.dumps(record) + "\n")
        metrics_file.flush()
        recent = (recent + [record])[-10:]
        speed_now = means.get(env.cfg.speed_term, float("nan"))
        print(f"it {iteration:4d}  steps {total_steps / 1e6:7.2f}M  "
              f"rew/step {means['reward']:+.5f}  speed {speed_now:+.3f} m/s  "
              f"std {std:.3f}  lr {lr:.1e} kl {kl_mean:.4f}  {sps:8,.0f} steps/s  "
              f"{elapsed / 60:5.1f} min{ep_text}"
              + (f"  DIVERGED {diverged_n}" if diverged_n else ""), flush=True)

        if not all(np.isfinite([means["reward"], surr, vloss])):
            print("ABORT: non-finite reward or loss")
            return 1
        if iteration % args.save_every == 0:
            path = save_checkpoint(logdir / f"model_{iteration:05d}.pt", policy, optimiser,
                                   iteration, total_steps, elapsed)
            log_checkpoint(logdir, path, recent)

    wall_minutes = (time.perf_counter() - started) / 60.0
    checkpoint = save_checkpoint(logdir / "model_final.pt", policy, optimiser, iteration,
                                 total_steps, wall_minutes * 60.0)
    log_checkpoint(logdir, checkpoint, recent)
    metrics_file.close()
    writer.close()
    print(f"trained {iteration} iterations, {total_steps:,} steps in {wall_minutes:.2f} min "
          f"({total_steps / (wall_minutes * 60):,.0f} steps/s overall); "
          f"{episodes_total} episodes, {falls_total} falls, {diverged_total} diverged")

    # ---- export: the deterministic mean, normaliser baked in
    parity_obs = obs_buf.reshape(-1, obs_size)[torch.randperm(n * horizon, device=device)[:512]]
    try:
        checkpoint_rel = checkpoint.relative_to(REPO).as_posix()
    except ValueError:
        checkpoint_rel = str(checkpoint)
    metadata = {"tool": setup.tool, "creature": setup.name, "trainer": setup.trainer_script,
                "run_id": run_id, "stepsTrained": total_steps, "iterations": iteration,
                "wallMinutes": round(wall_minutes, 3), "checkpoint": checkpoint_rel,
                "obs": setup.obs_description, "actions": f"{act_size}, unclipped mean"}
    err = export_onnx(policy.cpu(), args.onnx_out, metadata, parity_obs)
    info = {**metadata, "onnx": str(args.onnx_out), "onnxMaxAbsErr": err,
            "divergedWorlds": diverged_total, "episodes": episodes_total, "falls": falls_total}
    (logdir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"ONNX: {args.onnx_out}  (onnx vs torch max |err| {err:.2e} over 512 rollout obs)")
    if setup.eval_command:
        print(f"next: {setup.eval_command}")
    return 0


def log_checkpoint(logdir: Path, path: Path, recent: list[dict]) -> None:
    """Append the checkpoint with the mean of the last iterations' episode metrics."""
    keys = ("ep_return", "ep_speed", "ep_distance", "fall_rate", "reward")
    summary = {"checkpoint": path.name, "it": recent[-1]["it"] if recent else 0,
               "steps": recent[-1]["steps"] if recent else 0,
               "minutes": recent[-1]["minutes"] if recent else 0.0}
    for key in keys:
        values = [r[key] for r in recent if key in r]
        summary[key] = float(np.mean(values)) if values else None
    with open(logdir / "checkpoints.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary) + "\n")
