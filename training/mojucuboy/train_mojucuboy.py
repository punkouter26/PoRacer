"""PPO for MojucuBoy on MuJoCo Warp. All GPU, no host sync inside the rollout.

  .venv-mjwarp\\Scripts\\python.exe training/mojucuboy/train_mojucuboy.py --iterations 900

TensorBoard is started by this script BEFORE the trainer, per CLAUDE.md: a run you
cannot watch is a run you cannot judge. Port 6006 is checked first -- a crashed run
leaks its TensorBoard and the next launch would otherwise train blind.

Network shape is 75 -> 128 -> 128 -> 128 -> 21, matching IsaacBox so the two
humanoids are comparable and the exported ONNX has the same footprint.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mojucuboy_env  # noqa: E402
from mojucuboy_env import ACTION_SIZE, OBS_SIZE, MojucuBoyEnv  # noqa: E402

RESULTS = HERE / "runs"
HIDDEN = (128, 128, 128)


class RunningNorm(nn.Module):
    """Observation normaliser with the statistics kept as buffers, so they are
    saved with the checkpoint and exported into the ONNX graph rather than being
    re-derived (or forgotten) on the Unity side."""

    def __init__(self, size: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("count", torch.tensor(1e-4))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = torch.tensor(float(x.shape[0]), device=x.device)
        delta = batch_mean - self.mean
        total = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / total
        self.mean.copy_(new_mean)
        self.var.copy_(m2 / total)
        self.count.copy_(total)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -10.0, 10.0)


def mlp(sizes, out_size, gain_last):
    layers, last = [], sizes[0]
    for size in sizes[1:]:
        layers += [nn.Linear(last, size), nn.ELU()]
        last = size
    head = nn.Linear(last, out_size)
    nn.init.orthogonal_(head.weight, gain_last)
    nn.init.zeros_(head.bias)
    return nn.Sequential(*layers, head)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = RunningNorm(OBS_SIZE)
        self.actor = mlp((OBS_SIZE,) + HIDDEN, ACTION_SIZE, 0.01)
        self.critic = mlp((OBS_SIZE,) + HIDDEN, 1, 1.0)
        # Start quiet. At log_std = -0.5 (sigma 0.61) the initial policy knocked the
        # racer over in 0.7 s on every world, when a zero action holds the stance for
        # 3 s -- so the first rollouts carried almost no signal about walking, only
        # about falling. sigma 0.22 explores around the calibrated stance instead.
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), -1.5))

    def forward(self, obs):
        """Deterministic action. This is the ONNX entry point: normalisation is
        inside the graph, so Unity feeds raw observations."""
        return torch.tanh(self.actor(self.norm(obs)))

    def distribution(self, obs):
        normed = self.norm(obs)
        mean = self.actor(normed)
        return torch.distributions.Normal(mean, self.log_std.exp()), normed

    def value(self, obs):
        return self.critic(self.norm(obs)).squeeze(-1)


def port_free(port: int) -> bool:
    with socket.socket() as probe:
        return probe.connect_ex(("127.0.0.1", port)) != 0


def start_tensorboard(logdir: Path, port: int):
    """CLAUDE.md makes this non-optional, and makes it the launcher's job."""
    exe = HERE.parents[1] / ".venv-mjwarp" / "Scripts" / "tensorboard.exe"
    if not exe.exists():
        print(f"!! tensorboard not found at {exe} -- install it; refusing to train blind")
        return None
    if not port_free(port):
        print(f"!! port {port} is busy -- kill the leaked TensorBoard before launching.")
        print("   Refusing to start a run that cannot be watched.")
        sys.exit(2)
    proc = subprocess.Popen(
        [str(exe), "--logdir", str(logdir), "--port", str(port)],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(3.0)
    if proc.poll() is not None:
        print(f"!! TensorBoard exited immediately (port {port} busy?)")
        sys.exit(2)
    print(f"TensorBoard: http://localhost:{port}/  (logdir {logdir})")
    return proc


def prune_runs(keep: int = 3) -> None:
    """CLAUDE.md: prune obsolete runs before starting a new one, so the curve being
    watched is not buried under dead experiments."""
    if not RESULTS.exists():
        return
    runs = sorted((p for p in RESULTS.iterdir() if p.is_dir()),
                  key=lambda p: p.stat().st_mtime)
    for old in runs[:max(0, len(runs) - keep)]:
        shutil.rmtree(old, ignore_errors=True)
        print(f"pruned stale run {old.name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=900)
    parser.add_argument("--worlds", type=int, default=8192)
    parser.add_argument("--rollout", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=2e-3)
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    run_id = args.run_id or f"boy_{time.strftime('%Y%m%d_%H%M')}"
    logdir = RESULTS / run_id
    logdir.mkdir(parents=True, exist_ok=True)

    prune_runs()
    tb = start_tensorboard(RESULTS, args.port)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(str(logdir))

    env = MojucuBoyEnv(args.worlds, seed=args.seed)
    policy = ActorCritic().to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=args.lr)

    (logdir / "config.json").write_text(json.dumps({
        **vars(args), "obs": OBS_SIZE, "act": ACTION_SIZE, "hidden": list(HIDDEN),
        "policy_dt": env.dt, "decimation": mojucuboy_env.DECIMATION,
        "episode_steps": mojucuboy_env.EPISODE_STEPS, "target_speed": mojucuboy_env.TARGET_SPEED,
    }, indent=2))

    obs = env.observation()
    ep_return = torch.zeros(args.worlds, device=device)
    ep_length = torch.zeros(args.worlds, device=device)
    ep_speed = torch.zeros(args.worlds, device=device)
    done_returns, done_lengths, done_speeds, done_falls = [], [], [], []

    total_steps = 0
    started = time.perf_counter()
    print(f"run {run_id}: {args.worlds} worlds x {args.rollout} steps "
          f"= {args.worlds * args.rollout} samples/iter")

    for iteration in range(1, args.iterations + 1):
        buf_obs = torch.zeros(args.rollout, args.worlds, OBS_SIZE, device=device)
        buf_act = torch.zeros(args.rollout, args.worlds, ACTION_SIZE, device=device)
        buf_logp = torch.zeros(args.rollout, args.worlds, device=device)
        buf_rew = torch.zeros(args.rollout, args.worlds, device=device)
        buf_val = torch.zeros(args.rollout, args.worlds, device=device)
        buf_done = torch.zeros(args.rollout, args.worlds, device=device)

        with torch.no_grad():
            for t in range(args.rollout):
                dist, _ = policy.distribution(obs)
                raw = torch.nan_to_num(dist.sample(), nan=0.0)
                buf_obs[t] = obs
                buf_act[t] = raw
                buf_logp[t] = dist.log_prob(raw).sum(-1)
                buf_val[t] = policy.value(obs)

                obs, reward, done, terms = env.step(torch.tanh(raw))
                buf_rew[t] = torch.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)
                buf_done[t] = done.float()

                ep_return += reward
                ep_length += 1
                ep_speed += terms["speed_along"]
                if done.any():
                    idx = done.nonzero(as_tuple=True)[0]
                    done_returns.append(ep_return[idx].clone())
                    done_lengths.append(ep_length[idx].clone())
                    done_speeds.append((ep_speed[idx] / ep_length[idx]).clone())
                    done_falls.append(terms["fallen"][idx].clone())
                    ep_return[idx] = 0
                    ep_length[idx] = 0
                    ep_speed[idx] = 0
                    env.reset(idx)
                    obs = env.observation()

            last_value = policy.value(obs)

        # An overflow silently drops contacts, which is how a Warp run in this repo
        # previously trained a creature that sank through the floor. Fail loudly.
        peak = env.contact_overflow()
        if peak >= env.naconmax:
            print(f"ABORT iter {iteration}: contact overflow {peak} >= {env.naconmax}. "
                  f"Raise NCONMAX in mojucuboy_env.py and restart -- results so far are suspect.")
            return 1

        advantages = torch.zeros_like(buf_rew)
        gae = torch.zeros(args.worlds, device=device)
        for t in reversed(range(args.rollout)):
            next_value = last_value if t == args.rollout - 1 else buf_val[t + 1]
            not_done = 1.0 - buf_done[t]
            delta = buf_rew[t] + args.gamma * next_value * not_done - buf_val[t]
            gae = delta + args.gamma * args.lam * not_done * gae
            advantages[t] = gae
        returns = advantages + buf_val

        flat_obs = buf_obs.reshape(-1, OBS_SIZE)
        flat_act = buf_act.reshape(-1, ACTION_SIZE)
        flat_logp = buf_logp.reshape(-1)
        flat_adv = advantages.reshape(-1)
        flat_ret = returns.reshape(-1)
        flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)
        policy.norm.update(flat_obs)

        total = flat_obs.shape[0]
        batch = total // args.minibatches
        losses = []
        for _ in range(args.epochs):
            order = torch.randperm(total, device=device)
            for start in range(0, total, batch):
                sel = order[start:start + batch]
                dist, _ = policy.distribution(flat_obs[sel])
                logp = dist.log_prob(flat_act[sel]).sum(-1)
                ratio = (logp - flat_logp[sel]).exp()
                a = flat_adv[sel]
                pg = -torch.min(ratio * a,
                                torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * a).mean()
                value_loss = (policy.value(flat_obs[sel]) - flat_ret[sel]).pow(2).mean()
                entropy = dist.entropy().sum(-1).mean()
                loss = pg + 0.5 * value_loss - args.entropy * entropy
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimiser.step()
                losses.append((pg.item(), value_loss.item(), entropy.item()))

        total_steps += total
        if done_returns and iteration % 5 == 0:
            rets = torch.cat(done_returns)
            lens = torch.cat(done_lengths)
            spds = torch.cat(done_speeds)
            falls = torch.cat(done_falls)
            pg, vl, ent = np.mean(losses, axis=0)
            elapsed = time.perf_counter() - started
            writer.add_scalar("rollout/mean_return", rets.mean().item(), total_steps)
            writer.add_scalar("rollout/mean_episode_length", lens.mean().item(), total_steps)
            writer.add_scalar("rollout/mean_forward_speed", spds.mean().item(), total_steps)
            writer.add_scalar("rollout/fall_rate", falls.mean().item(), total_steps)
            writer.add_scalar("loss/policy", pg, total_steps)
            writer.add_scalar("loss/value", vl, total_steps)
            writer.add_scalar("loss/entropy", ent, total_steps)
            writer.add_scalar("perf/steps_per_second", total_steps / elapsed, total_steps)
            print(f"iter {iteration:4d}  steps {total_steps/1e6:7.2f}M  "
                  f"ret {rets.mean():7.2f}  len {lens.mean():6.1f}  "
                  f"spd {spds.mean():5.2f} m/s  fall {falls.mean():4.2f}  "
                  f"{total_steps/elapsed/1e3:5.0f}k steps/s", flush=True)
            done_returns, done_lengths, done_speeds, done_falls = [], [], [], []

        if iteration % 50 == 0 or iteration == args.iterations:
            torch.save({"model": policy.state_dict(), "iteration": iteration},
                       logdir / "policy.pt")

    torch.save({"model": policy.state_dict(), "iteration": args.iterations},
               logdir / "policy.pt")
    writer.close()
    print(f"done: {logdir/'policy.pt'}")
    if tb is not None:
        tb.terminate()   # teardown order: trainer -> envs -> TensorBoard
    return 0


if __name__ == "__main__":
    sys.exit(main())
