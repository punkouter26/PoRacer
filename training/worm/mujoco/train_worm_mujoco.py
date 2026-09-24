"""PPO for Worm5 on MuJoCo Warp -- method C of the worm shoot-out (WORM_SPEC.md).

  .venv-mjwarp\\Scripts\\python.exe training/worm/mujoco/train_worm_mujoco.py --minutes 30

TensorBoard is started FIRST (AGENTS.md rule C) on --tensorboard-port with logdir
training/worm/mujoco/runs. If that port is already taken the script says so and
exits: it never kills a process it did not start.

The PPO is the spec's, and where the spec is silent it follows RSL-RL 5.0.1 (the
library the Isaac Lab side uses) so that the comparison is of simulators and
training tools, not of two different PPO variants:
  * Gaussian policy, one state-independent LOG-std parameter per action (init
    log 0.5), actions NOT squashed -- the env clips to [-1, 1]. Default PyTorch layer init.
  * Observation normaliser = RSL-RL EmpiricalNormalization: running mean/var over
    every env step's returned observation, (x - mean) / (std + 0.01), no clip.
    Updated after each env step; raw observations are stored in the rollout.
  * Timeout bootstrap = RSL-RL's: reward += gamma * V(s_t) on a timeout step,
    and the step counts as done for GAE.
  * Advantages normalised once over the whole rollout; clipped value loss
    (clip 0.2); loss = surrogate + 1.0 * value - 0.005 * entropy; Adam, constant
    lr 3e-4, global grad-norm clip 1.0; 5 epochs x 4 shuffled minibatches.

Wall-clock budget: --minutes counts from the first training iteration, i.e. it
excludes model loading and kernel compilation. The loop stops before starting an
iteration it could not finish inside the budget.
"""

from __future__ import annotations

import argparse
import copy
import json
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
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from worm_env import ACTION_SIZE, EPISODE_STEPS, OBS_SIZE, WormEnv  # noqa: E402

RUNS = HERE / "runs"
EXPORT_DIR = HERE.parent / "export"
DEFAULT_ONNX = EXPORT_DIR / "worm_mujoco.onnx"
TENSORBOARD_EXE = REPO / ".venv-mjwarp" / "Scripts" / "tensorboard.exe"
HIDDEN = (128, 128, 128)
INIT_STD = 0.5
NORM_EPS = 1e-2


# ---------------------------------------------------------------- networks
class EmpiricalNormalization(nn.Module):
    """RSL-RL's observation normaliser, stats kept as buffers so they travel with
    the checkpoint and are baked into the ONNX graph."""

    def __init__(self, size: int, eps: float = NORM_EPS):
        super().__init__()
        self.eps = eps
        self.register_buffer("mean", torch.zeros(size))
        self.register_buffer("var", torch.ones(size))
        self.register_buffer("std", torch.ones(size))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        n = x.shape[0]
        self.count += n
        rate = n / self.count
        var_x = x.var(dim=0, unbiased=False)
        mean_x = x.mean(dim=0)
        delta = mean_x - self.mean
        self.mean += rate * delta
        self.var += rate * (var_x - self.var + delta * (mean_x - self.mean))
        self.std.copy_(torch.sqrt(self.var))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)


def mlp(inp: int, out: int) -> nn.Sequential:
    layers, last = [], inp
    for size in HIDDEN:
        layers += [nn.Linear(last, size), nn.ELU()]
        last = size
    layers.append(nn.Linear(last, out))
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = EmpiricalNormalization(OBS_SIZE)
        self.actor = mlp(OBS_SIZE, ACTION_SIZE)
        self.critic = mlp(OBS_SIZE, 1)
        self.log_std = nn.Parameter(torch.full((ACTION_SIZE,), float(np.log(INIT_STD))))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action (the mean), unclipped. Same as the ONNX graph."""
        return self.actor(self.norm(obs))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.actor(self.norm(obs))
        return torch.distributions.Normal(mean, self.log_std.exp().expand_as(mean),
                                          validate_args=False)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self.norm(obs)).squeeze(-1)


class OnnxPolicy(nn.Module):
    """obs [1, 35] -> actions [1, 8]: normaliser + actor mean, nothing else."""

    def __init__(self, policy: ActorCritic):
        super().__init__()
        self.norm = copy.deepcopy(policy.norm).cpu()
        self.actor = copy.deepcopy(policy.actor).cpu()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.norm(obs))


def load_policy(checkpoint: Path, device="cpu") -> ActorCritic:
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    policy = ActorCritic().to(device)
    policy.load_state_dict(ck["model"])
    policy.eval()
    return policy


def export_onnx(policy: ActorCritic, path: Path, metadata: dict,
                parity_obs: torch.Tensor) -> float:
    """Write the ONNX and return max |onnx - torch| over `parity_obs` (N, 35)."""
    import onnx
    import onnxruntime as ort

    path.parent.mkdir(parents=True, exist_ok=True)
    module = OnnxPolicy(policy).eval()
    torch.onnx.export(module, (torch.zeros(1, OBS_SIZE),), str(path),
                      input_names=["obs"], output_names=["actions"],
                      opset_version=15, dynamo=False, dynamic_axes=None)
    model = onnx.load(str(path))
    for key, value in metadata.items():
        entry = model.metadata_props.add()
        entry.key, entry.value = key, str(value)
    onnx.checker.check_model(model)
    onnx.save(model, str(path))

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    obs_np = parity_obs.detach().cpu().float().numpy()
    with torch.no_grad():
        ref = module(torch.from_numpy(obs_np)).numpy()
    err = 0.0
    for row in range(obs_np.shape[0]):
        out = session.run(["actions"], {"obs": obs_np[row:row + 1]})[0]
        err = max(err, float(np.abs(out - ref[row:row + 1]).max()))
    return err


# ---------------------------------------------------------------- tensorboard
def port_free(port: int) -> bool:
    with socket.socket() as probe:
        probe.settimeout(0.5)
        return probe.connect_ex(("127.0.0.1", port)) != 0


def start_tensorboard(port: int) -> subprocess.Popen:
    if not TENSORBOARD_EXE.exists():
        print(f"ERROR: TensorBoard not found at {TENSORBOARD_EXE}. Refusing to train blind.")
        sys.exit(2)
    if not port_free(port):
        print(f"ERROR: port {port} is already in use by another process.")
        print(f"       This script will not stop a process it did not start.")
        print(f"       Free the port yourself, or pass --tensorboard-port <free port>.")
        sys.exit(2)
    RUNS.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([str(TENSORBOARD_EXE), "--logdir", str(RUNS), "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):                         # up to ~30 s for it to bind
        if proc.poll() is not None:
            print(f"ERROR: TensorBoard exited immediately (code {proc.returncode}).")
            sys.exit(2)
        if not port_free(port):
            break
        time.sleep(0.5)
    print(f"TensorBoard: http://localhost:{port}/  (logdir {RUNS})", flush=True)
    return proc


def stop_process(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def prune_runs(keep: int, current: str) -> None:
    """AGENTS.md rule C: clear dead runs so the live curve is readable. Keeps the
    newest `keep`, never touches a run written to in the last 30 minutes."""
    if keep < 0 or not RUNS.exists():
        return
    now = time.time()

    def newest(run: Path) -> float:
        try:
            return max((p.stat().st_mtime for p in run.rglob("*")), default=run.stat().st_mtime)
        except OSError:
            return now

    runs = [p for p in RUNS.iterdir() if p.is_dir() and p.name != current
            and now - newest(p) > 1800.0]
    runs.sort(key=newest)
    for old in runs[:max(0, len(runs) - keep)]:
        try:
            shutil.rmtree(old)
            print(f"pruned stale run {old.name}")
        except OSError as exc:
            print(f"!! could not prune {old.name}: {exc}")


# ---------------------------------------------------------------- training
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--minutes", type=float, default=30.0, help="wall-clock training budget")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--tensorboard-port", type=int, default=6006)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rollout", type=int, default=24)
    parser.add_argument("--save-every", type=int, default=50, help="checkpoint period, iterations")
    parser.add_argument("--keep-runs", type=int, default=3,
                        help="older idle runs kept when pruning (-1 = never prune)")
    parser.add_argument("--onnx-out", type=Path, default=DEFAULT_ONNX)
    args = parser.parse_args()

    # Spec PPO constants. Deliberately not flags: both trainers must use these.
    gamma, lam, clip = 0.99, 0.95, 0.2
    epochs, minibatches, lr = 5, 4, 3e-4
    entropy_coef, value_coef, max_grad_norm = 0.005, 1.0, 1.0

    run_id = args.run_id or time.strftime("worm_mujoco_%Y%m%d_%H%M")
    logdir = RUNS / run_id

    tb = start_tensorboard(args.tensorboard_port)          # rule C: before anything else
    try:
        prune_runs(args.keep_runs, run_id)
        logdir.mkdir(parents=True, exist_ok=True)
        return train(args, logdir, run_id, gamma, lam, clip, epochs, minibatches, lr,
                     entropy_coef, value_coef, max_grad_norm)
    finally:
        stop_process(tb)                                   # stop what we started
        print("TensorBoard stopped.")


def train(args, logdir, run_id, gamma, lam, clip, epochs, minibatches, lr,
          entropy_coef, value_coef, max_grad_norm) -> int:
    from torch.utils.tensorboard import SummaryWriter

    torch.manual_seed(args.seed)
    device = torch.device("cuda:0")
    n, horizon = args.num_envs, args.rollout

    setup_started = time.perf_counter()
    env = WormEnv(n, seed=args.seed, randomize=True, random_initial_episode=True)
    policy = ActorCritic().to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    writer = SummaryWriter(str(logdir))
    (logdir / "config.json").write_text(json.dumps({
        **{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "obs": OBS_SIZE, "act": ACTION_SIZE, "hidden": list(HIDDEN), "activation": "elu",
        "init_std": INIT_STD, "gamma": gamma, "lambda": lam, "clip": clip, "epochs": epochs,
        "minibatches": minibatches, "lr": lr, "entropy_coef": entropy_coef,
        "value_coef": value_coef, "max_grad_norm": max_grad_norm, "policy_dt": env.dt,
        "episode_steps": EPISODE_STEPS, "cuda_graph": env._graph is not None,
    }, indent=2))
    print(f"run {run_id}: {n} envs x {horizon} steps = {n * horizon} samples/iter  "
          f"(setup {time.perf_counter() - setup_started:.1f} s, "
          f"cuda graph {'on' if env._graph is not None else 'off'})", flush=True)

    obs_buf = torch.zeros(horizon, n, OBS_SIZE, device=device)
    act_buf = torch.zeros(horizon, n, ACTION_SIZE, device=device)
    logp_buf = torch.zeros(horizon, n, device=device)
    val_buf = torch.zeros(horizon, n, device=device)
    rew_buf = torch.zeros(horizon, n, device=device)
    done_buf = torch.zeros(horizon, n, device=device)

    obs = env.observation()
    budget = args.minutes * 60.0
    started = time.perf_counter()
    total_steps, iteration, last_iter_time = 0, 0, 0.0
    diverged_total = 0

    while True:
        elapsed = time.perf_counter() - started
        if iteration > 0 and elapsed + last_iter_time > budget:
            break
        iteration += 1
        iter_started = time.perf_counter()
        sums = {k: torch.zeros((), device=device) for k in
                ("reward", "speed_x", "progress", "heading", "action_rate", "effort",
                 "lateral", "roll_rate", "belly_down", "torque_abs")}
        ep_sum = {k: torch.zeros((), device=device) for k in
                  ("return", "speed", "distance", "count")}
        diverged = torch.zeros((), device=device)

        # ---- rollout
        with torch.no_grad():
            for t in range(horizon):
                dist = policy.distribution(obs)
                action = dist.sample()
                obs_buf[t] = obs
                act_buf[t] = action
                logp_buf[t] = dist.log_prob(action).sum(-1)
                val_buf[t] = policy.value(obs)

                obs, reward, done, timeouts, info = env.step(action)
                policy.norm.update(obs)
                reward = torch.nan_to_num(reward)
                sums["reward"] += reward.mean()
                for key in ("speed_x", "progress", "heading", "action_rate", "effort",
                            "lateral", "roll_rate", "belly_down", "torque_abs"):
                    sums[key] += info[key].mean()
                # RSL-RL timeout bootstrap: r += gamma * V(s_t) where the episode was cut.
                rew_buf[t] = reward + gamma * val_buf[t] * timeouts.float()
                done_buf[t] = done.float()
                # Episode statistics from COMPLETED episodes only (a health-guard
                # reset is not a finished 20 s run).
                d = (done & ~info["diverged"]).float()
                ep_sum["return"] += (info["ep_return"] * d).sum()
                ep_sum["speed"] += (info["ep_speed"] * d).sum()
                ep_sum["distance"] += (info["ep_distance"] * d).sum()
                ep_sum["count"] += d.sum()
                diverged += info["diverged"].float().sum()
            last_value = policy.value(obs)

        peak = env.contact_peak()
        if peak >= env.naconmax:
            print(f"ABORT iteration {iteration}: contact overflow {peak} >= {env.naconmax}. "
                  f"Raise NCONMAX in worm_env.py.")
            return 1

        # ---- GAE
        adv_buf = torch.zeros_like(rew_buf)
        gae = torch.zeros(n, device=device)
        for t in reversed(range(horizon)):
            next_value = last_value if t == horizon - 1 else val_buf[t + 1]
            not_done = 1.0 - done_buf[t]
            delta = rew_buf[t] + not_done * gamma * next_value - val_buf[t]
            gae = delta + not_done * gamma * lam * gae
            adv_buf[t] = gae
        ret_buf = adv_buf + val_buf

        b_obs = obs_buf.reshape(-1, OBS_SIZE)
        b_act = act_buf.reshape(-1, ACTION_SIZE)
        b_logp = logp_buf.reshape(-1)
        b_val = val_buf.reshape(-1)
        b_ret = ret_buf.reshape(-1)
        b_adv = adv_buf.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        # ---- PPO update
        total = b_obs.shape[0]
        mb = total // minibatches
        stats = []
        for _ in range(epochs):
            perm = torch.randperm(total, device=device)
            for k in range(minibatches):
                sel = perm[k * mb:(k + 1) * mb]
                dist = policy.distribution(b_obs[sel])
                logp = dist.log_prob(b_act[sel]).sum(-1)
                entropy = dist.entropy().sum(-1).mean()
                ratio = torch.exp(logp - b_logp[sel])
                adv = b_adv[sel]
                surrogate = torch.max(-adv * ratio,
                                      -adv * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)).mean()
                value = policy.value(b_obs[sel])
                v_clipped = b_val[sel] + (value - b_val[sel]).clamp(-clip, clip)
                value_loss = torch.max((value - b_ret[sel]).pow(2),
                                       (v_clipped - b_ret[sel]).pow(2)).mean()
                loss = surrogate + value_coef * value_loss - entropy_coef * entropy
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
                optimiser.step()
                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - (logp - b_logp[sel])).mean()
                stats.append(torch.stack((surrogate.detach(), value_loss.detach(),
                                          entropy.detach(), approx_kl)))

        total_steps += n * horizon
        last_iter_time = time.perf_counter() - iter_started
        elapsed = time.perf_counter() - started

        # ---- logging (one host sync per iteration)
        surr, vloss, ent, kl = torch.stack(stats).mean(0).tolist()
        means = {k: (v / horizon).item() for k, v in sums.items()}
        episodes = int(ep_sum["count"].item())
        diverged_n = int(diverged.item())
        diverged_total += diverged_n
        sps = n * horizon / last_iter_time
        step = total_steps
        writer.add_scalar("train/mean_reward_per_step", means["reward"], step)
        writer.add_scalar("train/mean_forward_speed", means["speed_x"], step)
        writer.add_scalar("perf/steps_per_second", sps, step)
        writer.add_scalar("perf/total_steps", total_steps, step)
        writer.add_scalar("perf/wall_minutes", elapsed / 60.0, step)
        for key in ("progress", "heading", "action_rate", "effort", "lateral", "roll_rate",
                    "belly_down", "torque_abs"):
            writer.add_scalar(f"terms/{key}", means[key], step)
        writer.add_scalar("loss/surrogate", surr, step)
        writer.add_scalar("loss/value", vloss, step)
        writer.add_scalar("loss/entropy", ent, step)
        writer.add_scalar("loss/approx_kl", kl, step)
        writer.add_scalar("policy/mean_std", policy.log_std.exp().mean().item(), step)
        writer.add_scalar("health/diverged_worlds", diverged_n, step)
        writer.add_scalar("health/contact_peak", peak, step)
        ep_text = ""
        if episodes > 0:
            ep_ret = (ep_sum["return"] / episodes).item()
            ep_spd = (ep_sum["speed"] / episodes).item()
            ep_dist = (ep_sum["distance"] / episodes).item()
            writer.add_scalar("episode/mean_return", ep_ret, step)
            writer.add_scalar("episode/mean_forward_speed", ep_spd, step)
            writer.add_scalar("episode/mean_distance", ep_dist, step)
            ep_text = f"  ep {episodes:5d} ret {ep_ret:6.2f} dist {ep_dist:5.2f} m"
        print(f"it {iteration:4d}  steps {total_steps / 1e6:7.2f}M  "
              f"rew/step {means['reward']:+.5f}  speed {means['speed_x']:+.3f} m/s  "
              f"std {policy.log_std.exp().mean().item():.3f}  {sps:8,.0f} steps/s  "
              f"{elapsed / 60:5.1f} min{ep_text}"
              + (f"  DIVERGED {diverged_n}" if diverged_n else ""), flush=True)

        if not all(np.isfinite([means["reward"], means["speed_x"], surr, vloss])):
            print("ABORT: non-finite reward, speed or loss")
            return 1
        if iteration % args.save_every == 0:
            save_checkpoint(logdir, policy, optimiser, iteration, total_steps, elapsed)

    wall_minutes = (time.perf_counter() - started) / 60.0
    checkpoint = save_checkpoint(logdir, policy, optimiser, iteration, total_steps,
                                 wall_minutes * 60.0)
    writer.close()
    print(f"trained {iteration} iterations, {total_steps:,} steps in {wall_minutes:.2f} min "
          f"({total_steps / (wall_minutes * 60):,.0f} steps/s overall)")

    # ---- export: the deterministic mean, normaliser baked in
    parity_obs = obs_buf.reshape(-1, OBS_SIZE)[
        torch.randperm(n * horizon, device=device)[:512]]
    try:
        checkpoint_rel = checkpoint.relative_to(REPO).as_posix()
    except ValueError:
        checkpoint_rel = str(checkpoint)
    metadata = {"tool": "mujoco", "trainer": "train_worm_mujoco.py", "run_id": run_id,
                "stepsTrained": total_steps, "iterations": iteration,
                "wallMinutes": round(wall_minutes, 3), "checkpoint": checkpoint_rel,
                "obs": "WORM_SPEC.md 35-float observation", "actions": "8, unclipped mean"}
    err = export_onnx(policy.cpu(), args.onnx_out, metadata, parity_obs)
    info = {**metadata, "onnx": str(args.onnx_out), "onnxMaxAbsErr": err,
            "divergedWorlds": diverged_total}
    (logdir / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"ONNX: {args.onnx_out}  (onnx vs torch max |err| {err:.2e} over 512 rollout obs)")
    print(f"next: .venv-mjwarp\\Scripts\\python.exe training/worm/mujoco/eval_worm_mujoco.py "
          f"--onnx {args.onnx_out}")
    return 0


def save_checkpoint(logdir: Path, policy, optimiser, iteration, total_steps, seconds) -> Path:
    path = logdir / "policy.pt"
    torch.save({"model": policy.state_dict(), "optimiser": optimiser.state_dict(),
                "iteration": iteration, "total_steps": total_steps,
                "wall_seconds": seconds}, path)
    return path


if __name__ == "__main__":
    sys.exit(main())
