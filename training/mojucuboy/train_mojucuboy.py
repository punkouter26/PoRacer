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
import math
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

# Reward terms and stability measures logged per iteration. Every quantity the
# reward charges for appears here, plus the derived heading angle, so the ten
# weights in mojucuboy_env.py can be tuned against evidence rather than against
# the single scalar return. "heading_err_deg" is derived in the rollout loop.
KPI_TERMS = (
    "track", "facing", "speed_along", "drift", "lateral",
    "accel", "impact", "ctrl", "action_rate", "upright", "standing", "height",
    # `torque`/`torque_abs` are the APPLIED-TORQUE measures. `ctrl` is the commanded
    # joint angle and is not the same quantity -- see mojucuboy_env.py.
    "torque", "torque_abs",
)
KPI_LOGGED = KPI_TERMS + ("heading_err_deg",)


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


# A run holding any of these has graduated: it produced a shipped brain or a
# recorded grade, and deleting it destroys something no rerun can reproduce.
# boy_chase01 has no policy.pt at all -- its .onnx IS the artifact.
# `speed_sweep*.json` is here for the same reason as `gate*.json`: it is a recorded
# grade. It is also the only artifact that distinguishes a policy which TRACKS a
# commanded speed from one that merely runs at 1.5 m/s (M11), so losing it loses the
# measurement that the single-point KPI cannot reproduce.
PROTECTED_GLOBS = ("*.onnx", "gate*.json", "speed_sweep*.json", "mujoco_reference.json")


def is_protected(run: Path) -> bool:
    return any(next(run.glob(pattern), None) is not None for pattern in PROTECTED_GLOBS)


ACTIVE_WINDOW_S = 1800.0


def is_active(run: Path) -> bool:
    """True if anything inside this run was written recently — i.e. it is very
    probably still training.

    This exists because `prune_runs` could delete a RUNNING experiment, and with
    concurrent launches it reliably would. Worked example, launching four cells
    30 s apart with five disposable runs already on disk: cell A starts (6
    candidates, 3 deleted), B starts (4 candidates, 1 deleted), C starts (4, 1),
    then D starts and the three oldest candidates are A, B, C — so D's own prune
    deletes cell A out from under a live process. The tfevents file vanishes and
    the run keeps writing into a deleted directory.

    Directory mtime is not enough: on Windows it does not change when a file
    inside it is appended to, which is exactly what a training run does to its
    tfevents. So the newest mtime among the CONTENTS is what gets checked.
    """
    try:
        newest = max((child.stat().st_mtime for child in run.iterdir()),
                     default=run.stat().st_mtime)
    except OSError:
        return True          # Unreadable: refuse to delete it.
    return (time.time() - newest) < ACTIVE_WINDOW_S


def prune_runs(keep: int = 3) -> None:
    """CLAUDE.md: prune obsolete runs before starting a new one, so the curve being
    watched is not buried under dead experiments.

    Prunes only disposable runs. This used to delete the oldest directories by
    mtime with no exemptions, and it silently destroyed runs/boy_chase01 -- the
    SHIPPED brain's mojucuboy_policy.onnx plus its Gate 4 and Gate 5 records --
    on the fourth training run of a session. Those files are tracked in git,
    which is the only reason they came back. A training script must not be able
    to delete a shipped artifact as a side effect of starting.
    """
    if not RESULTS.exists():
        return
    candidates = [p for p in RESULTS.iterdir()
                  if p.is_dir() and not is_protected(p) and not is_active(p)]
    candidates.sort(key=lambda p: p.stat().st_mtime)
    for old in candidates[:max(0, len(candidates) - keep)]:
        try:
            shutil.rmtree(old)
            print(f"pruned stale run {old.name}")
        except OSError as exc:
            # Was ignore_errors=True, which hid a failed wipe -- exactly the
            # Windows file-handle case CLAUDE.md warns about with TensorBoard.
            print(f"!! could not prune {old.name}: {exc}")


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
    parser.add_argument("--init-from", type=str, default=None,
                        help="run-id to load policy.pt from, for curriculum stage two: "
                             "learn balance with --terminate-on-fall, then relearn get-up "
                             "with it off, starting from those weights.")
    parser.add_argument("--terminate-on-fall", action="store_true",
                        help="end the episode on a fall. Stage one of a curriculum: "
                             "learn balance with it on, then relearn get-up with it off. "
                             "Without it, and with the M9 inversion exploit closed, two "
                             "full runs settled on lying still for 1000 steps.")
    parser.add_argument("--upright-weight", type=float, default=mojucuboy_env.W_UPRIGHT,
                        help="weight on the ungated uprightness term. This is the only "
                             "positive reward a fallen racer can earn, so it is the whole "
                             "gradient back to its feet; the shipped 0.05 is half of the "
                             "unconditional W_ALIVE and measurably too weak once the M9 "
                             "inversion exploit is closed.")
    parser.add_argument("--reset-fallen", type=float,
                        default=mojucuboy_env.RESET_FALLEN_FRACTION,
                        help="fraction of resets that start the racer sprawled. Lower it "
                             "to let the policy learn balance before recovery.")
    parser.add_argument("--sprawl-tilt-start", type=float, default=180.0,
                        help="Sprawl difficulty at iteration 1, DEGREES off vertical. "
                             "180 = the shipped fully-random orientation. Lower values "
                             "start the get-up curriculum from a recoverable tip.")
    parser.add_argument("--sprawl-tilt-end", type=float, default=180.0,
                        help="Sprawl difficulty the ramp finishes at, degrees.")
    parser.add_argument("--sprawl-tilt-full", type=float, default=0.6,
                        help="Fraction of the run over which the tilt ramps from start "
                             "to end; it holds at --sprawl-tilt-end afterwards.")
    parser.add_argument("--scale-drift", type=float, default=mojucuboy_env.SCALE_DRIFT,
                        help="Normalising scale for the drift penalty tanh. The shipped 8.0 "
                             "was calibrated on random-action rollouts and leaves a trained "
                             "policy at 2-5 %% of the tanh range, i.e. no penalty at all (M14).")
    parser.add_argument("--ctrl-weight", type=float, default=mojucuboy_env.W_CTRL,
                        help="Weight on mean(action^2). The shipped 0.005 prices E12's 8.6x "
                             "actuator effort at 0.004 per step against W_TRACK 1.8 (M14).")
    parser.add_argument("--command-speed-range", type=float, nargs=2, default=None,
                        metavar=("LOW", "HIGH"),
                        help="Sample the commanded speed per episode from [LOW, HIGH] m/s. "
                             "Without it the command is a constant and observation 11 has "
                             "zero variance, which is why E12 collapses when commanded "
                             "anything other than 1.5 (M11).")
    parser.add_argument("--drift-yaw-weight", type=float, default=1.0,
                        help="Share of the drift penalty that is YAW RATE (1.0 = "
                             "shipped). Yaw rate is how heading gets corrected, so "
                             "taxing it trades K4 away for K5 -- measured at "
                             "scale_drift 0.5: K5 0.365->0.225 (passes) while K4 "
                             "13.5->16.7 deg (fails).")
    parser.add_argument("--heading-weight", type=float,
                        default=mojucuboy_env.W_HEADING,
                        help="W_HEADING (default 0.4). The only term paying for heading "
                             "accuracy directly; K4/K5 trade along a scale_drift frontier "
                             "that misses the gate corner, so this is the independent "
                             "lever for K4.")
    parser.add_argument("--drift-weight", type=float, default=mojucuboy_env.W_DRIFT,
                        help="W_DRIFT (default 0.15). The WEIGHT on the drift penalty, as "
                             "opposed to --scale-drift which is the normaliser inside its "
                             "tanh. At 0.15 the term is under 7%% of the reward even "
                             "saturated; this is the direct lever for K5.")
    parser.add_argument("--two-sided-speed", action="store_true",
                        help="penalise overshooting the commanded speed as well as "
                             "undershooting it. The shipped one-sided kernel clamps "
                             "positive error away, which is why the shipped brain "
                             "settles at 2.04 m/s against a 1.5 m/s command.")
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

    env = MojucuBoyEnv(args.worlds, seed=args.seed,
                       two_sided_speed=args.two_sided_speed,
                       upright_weight=args.upright_weight,
                       reset_fallen_fraction=args.reset_fallen,
                       terminate_on_fall=args.terminate_on_fall,
                       sprawl_max_tilt=math.radians(args.sprawl_tilt_start),
                       command_speed_range=(tuple(args.command_speed_range)
                                            if args.command_speed_range else None),
                       scale_drift=args.scale_drift,
                       ctrl_weight=args.ctrl_weight,
                       drift_yaw_weight=args.drift_yaw_weight,
                       heading_weight=args.heading_weight,
                       drift_weight=args.drift_weight)
    policy = ActorCritic().to(device)
    if args.init_from:
        # Curriculum stage two: carry stage one's weights over rather than
        # restarting. The observation normaliser statistics travel with the
        # state_dict (RunningNorm keeps them as buffers, deliberately), so the
        # resumed policy sees inputs scaled the way it was trained on.
        src = RESULTS / args.init_from / "policy.pt"
        if not src.exists():
            print(f"!! --init-from {args.init_from}: no policy.pt at {src}")
            return 1
        ck = torch.load(src, map_location=device, weights_only=False)
        policy.load_state_dict(ck["model"])
        print(f"resumed from {args.init_from} at iteration {ck.get('iteration', '?')}")
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

    best_score = float("-inf")
    for iteration in range(1, args.iterations + 1):
        # Sprawl DIFFICULTY ramp. E11/E12/E13 escalated how OFTEN the racer starts
        # fallen (0.0 -> 0.3 -> 0.6) and never how HARD the fall was: all three used
        # a fully random orientation, and K10 measured 0 % recovery from all three,
        # with `ever_stood_again` also 0 % -- the policy never once reached standing
        # from the floor. That is exploration, not reward: the payoff for standing is
        # large and dense, but the motor sequence from supine is too long to stumble
        # onto. Ramping the tilt makes the first rung reachable and then raises it.
        if args.sprawl_tilt_end > args.sprawl_tilt_start and args.iterations > 1:
            ramp_frac = min(1.0, (iteration - 1) / max(1, args.sprawl_tilt_full * args.iterations))
            env.sprawl_max_tilt = math.radians(
                args.sprawl_tilt_start
                + (args.sprawl_tilt_end - args.sprawl_tilt_start) * ramp_frac)
        buf_obs = torch.zeros(args.rollout, args.worlds, OBS_SIZE, device=device)
        buf_act = torch.zeros(args.rollout, args.worlds, ACTION_SIZE, device=device)
        buf_logp = torch.zeros(args.rollout, args.worlds, device=device)
        buf_rew = torch.zeros(args.rollout, args.worlds, device=device)
        buf_val = torch.zeros(args.rollout, args.worlds, device=device)
        buf_done = torch.zeros(args.rollout, args.worlds, device=device)

        # Per-term KPI accumulators. Ten reward weights were being tuned against
        # one scalar return; these make each charge visible. Sums stay ON THE GPU
        # and are read once at logging time -- a .item() per step would cost more
        # than the physics, which is the same reason the env keeps its rollout
        # host-free.
        kpi_sum = {k: torch.zeros((), device=device) for k in KPI_LOGGED}
        kpi_steps = 0

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

                for key in KPI_TERMS:
                    kpi_sum[key] += terms[key].float().mean()
                # Accumulated as an angle per world, not as arccos of the mean
                # cosine -- those differ, and the angle is the one with units.
                kpi_sum["heading_err_deg"] += torch.rad2deg(
                    terms["facing"].float().clamp(-1.0, 1.0).arccos()).mean()
                kpi_steps += 1

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
                # Kept ON THE GPU and reduced once per iteration, for exactly the
                # reason the KPI accumulators above give: a `.item()` here is a
                # blocking host sync, and there are epochs x minibatches of them
                # (96 at the shipped 4 x 8). Each one drains the queue, so the
                # update phase ran serialised against the host rather than
                # pipelined. `torch.stack(...).mean(0)` at logging time is the
                # same arithmetic as the `np.mean(losses, axis=0)` it replaces.
                losses.append(torch.stack(
                    (pg.detach(), value_loss.detach(), entropy.detach())))

        total_steps += total
        # Per-step KPIs and losses are logged UNCONDITIONALLY. Episode statistics
        # cannot be: `done` is timeout-only (a fall deliberately does not end the
        # episode), so with EPISODE_STEPS=1000 and rollout=24 the first `done`
        # lands at iteration 42 and the first episode log at iteration 45. Gating
        # the whole logging block on `done_returns`, as this did, left every run
        # shorter than ~45 iterations with no telemetry whatsoever -- which is
        # every validation run worth doing.
        if iteration % 5 == 0:
            pg, vl, ent = torch.stack(losses).mean(0).tolist()
            elapsed = time.perf_counter() - started
            writer.add_scalar("loss/policy", pg, total_steps)
            writer.add_scalar("loss/value", vl, total_steps)
            writer.add_scalar("loss/entropy", ent, total_steps)
            writer.add_scalar("perf/steps_per_second", total_steps / elapsed, total_steps)
            for key in KPI_LOGGED:
                writer.add_scalar(f"kpi/{key}", (kpi_sum[key] / kpi_steps).item(), total_steps)

            ep = ""
            if done_returns:
                rets = torch.cat(done_returns)
                lens = torch.cat(done_lengths)
                spds = torch.cat(done_speeds)
                falls = torch.cat(done_falls)
                writer.add_scalar("rollout/mean_return", rets.mean().item(), total_steps)
                writer.add_scalar("rollout/mean_episode_length", lens.mean().item(), total_steps)
                writer.add_scalar("rollout/mean_forward_speed", spds.mean().item(), total_steps)
                writer.add_scalar("rollout/fall_rate", falls.mean().item(), total_steps)
                # K1: survival as a ratio of the 1000-step episode, so the
                # threshold reads directly rather than needing division by eye.
                writer.add_scalar("kpi/survival_ratio",
                                  lens.mean().item() / mojucuboy_env.EPISODE_STEPS, total_steps)
                # K2: signed tracking error against the commanded speed.
                writer.add_scalar("kpi/speed_error_frac",
                                  (spds.mean().item() - mojucuboy_env.TARGET_SPEED)
                                  / mojucuboy_env.TARGET_SPEED, total_steps)
                ep = (f"ret {rets.mean():7.2f}  len {lens.mean():6.1f}  "
                      f"spd {spds.mean():5.2f} m/s  fall {falls.mean():4.2f}  ")
                done_returns, done_lengths, done_speeds, done_falls = [], [], [], []
            print(f"iter {iteration:4d}  steps {total_steps/1e6:7.2f}M  " + ep +
                  f"trk {(kpi_sum['track']/kpi_steps).item():4.2f}  "
                  f"vel {(kpi_sum['speed_along']/kpi_steps).item():5.2f} m/s  "
                  f"hdg {(kpi_sum['heading_err_deg']/kpi_steps).item():5.1f}deg  "
                  f"lat {(kpi_sum['lateral']/kpi_steps).item():4.2f}  "
                  f"upr {(kpi_sum['upright']/kpi_steps).item():4.2f}  "
                  f"std {(kpi_sum['standing']/kpi_steps).item():4.2f}  "
                  f"{total_steps/elapsed/1e3:5.0f}k steps/s", flush=True)

        if iteration % 50 == 0 or iteration == args.iterations:
            torch.save({"model": policy.state_dict(), "iteration": iteration},
                       logdir / "policy.pt")
            # BEST checkpoint, alongside the last one.
            #
            # This harness saved last-only for its whole history, which silently
            # assumes training improves monotonically. It does not: E15 peaked at
            # iteration ~500 (standing 0.80, heading 23.6 deg) and had decayed by
            # 700 (standing 0.73, heading 30.3 deg) -- so evaluating its final
            # checkpoint would have scored a policy materially worse than the one
            # the run actually produced, and the peak would be unrecoverable.
            # Every result in this log before tonight is a LAST-iteration number
            # for the same reason.
            #
            # Scored on mean standing, which is K1 and the stability gate the
            # convergence criteria are built on. Deliberately not on return:
            # return mixes ten weighted terms and a reward change makes runs
            # incomparable, whereas `standing` means the same thing across every
            # experiment in this project.
            score = (kpi_sum["standing"] / kpi_steps).item()
            if score > best_score:
                best_score = score
                torch.save({"model": policy.state_dict(), "iteration": iteration,
                            "standing": score}, logdir / "policy_best.pt")

    torch.save({"model": policy.state_dict(), "iteration": args.iterations},
               logdir / "policy.pt")
    print(f"best checkpoint: standing {best_score:.4f} -> {logdir/'policy_best.pt'}")
    writer.close()
    print(f"done: {logdir/'policy.pt'}")
    if tb is not None:
        tb.terminate()   # teardown order: trainer -> envs -> TensorBoard
    return 0


if __name__ == "__main__":
    sys.exit(main())
