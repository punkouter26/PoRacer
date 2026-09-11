"""Where does the wall clock actually go, and how many worlds should we train at?

  .venv-mjwarp\\Scripts\\python.exe training/mojucuboy/throughput_sweep.py

WHY THIS EXISTS. The worlds count was set to 8192 by habit and then lowered to 6144
on a THERMAL argument that measurement did not support:

  E15 @ 8192 : 47.1k steps/s, throttle flags 0x20 (SwThermalSlowdown), 87 C
  E16 @ 6144 : 44.0k steps/s, throttle flags 0x01 (GpuIdle only),     83 C

Removing the thermal throttle made the run 7 % SLOWER, so heat was never the binding
constraint. Two further readings say what is:

  * GPU utilisation sits at 54.9 % with a very tight spread (p5 52 %, p95 59 %,
    zero samples below 40 % or above 90 % across 340 samples / ~18 iterations).
    A flat 55 % is not "saturated in rollout, idle in update" -- that would be
    bimodal. It is idle finely interleaved THROUGHOUT, which is the signature of
    launch-latency-bound execution: many small kernels with host gaps between them.
  * 721 MiB of 6144 MiB VRAM in use. There is ~5.4 GB of headroom.

If the run is launch-bound rather than compute-bound, then the kernel COUNT is
roughly fixed per iteration and only the work per kernel grows with worlds -- so
steps/s should keep climbing with worlds until the GPU genuinely saturates. That is
a prediction, and this script tests it instead of assuming it.

WHAT IT DOES NOT CLAIM. steps/s is throughput, not learning. A larger world count
means a larger batch and therefore FEWER gradient updates per sample, which can make
a faster run learn more slowly per sample. So this reports `updates_per_1k_samples`
alongside, and `--hold-updates` scales `minibatches` with worlds to hold that figure
constant -- which is the setting a throughput win should actually be taken at.

Timing is phase-split with torch.cuda.synchronize() around each phase, because
without it the asynchronous queue assigns the cost of the rollout to whichever line
next touches the host.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import mojucuboy_env  # noqa: E402
from mojucuboy_env import MojucuBoyEnv  # noqa: E402
from train_mojucuboy import ACTION_SIZE, OBS_SIZE, ActorCritic  # noqa: E402

WARMUP_ITERS = 6      # Warp kernel load, autotune, and clock ramp all land here.
MEASURE_ITERS = 14


def gpu_state() -> dict:
    """Temperature / power / throttle at the moment of sampling.

    Read at the END of the measured window, so it reflects the thermally settled
    state rather than the cold one -- the 6144-vs-8192 mistake came from comparing
    a settled run against an unsettled assumption.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,"
             "clocks_throttle_reasons.active,clocks.sm,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20).stdout.strip().split(", ")
        return {"temp_c": out[0], "power_w": out[1], "throttle": out[2],
                "sm_mhz": out[3], "util_pct": out[4], "vram_mib": out[5]}
    except (OSError, subprocess.SubprocessError, IndexError) as exc:
        return {"error": str(exc)}


def bench(worlds: int, rollout: int, epochs: int, minibatches: int,
          lr: float) -> dict:
    device = torch.device("cuda:0")
    env = MojucuBoyEnv(worlds, seed=0, two_sided_speed=True)
    policy = ActorCritic().to(device)
    optimiser = torch.optim.Adam(policy.parameters(), lr=lr)
    obs = env.observation()

    roll_s = 0.0
    upd_s = 0.0
    for iteration in range(WARMUP_ITERS + MEASURE_ITERS):
        measuring = iteration >= WARMUP_ITERS
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        buf_obs = torch.zeros(rollout, worlds, OBS_SIZE, device=device)
        buf_act = torch.zeros(rollout, worlds, ACTION_SIZE, device=device)
        buf_logp = torch.zeros(rollout, worlds, device=device)
        with torch.no_grad():
            for t in range(rollout):
                dist, _ = policy.distribution(obs)
                raw = torch.nan_to_num(dist.sample(), nan=0.0)
                buf_obs[t] = obs
                buf_act[t] = raw
                buf_logp[t] = dist.log_prob(raw).sum(-1)
                obs, _reward, _done, _terms = env.step(torch.tanh(raw))

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        # Same shape of work as the real update: 4 epochs x N minibatches of
        # forward + backward + clip + step. The loss is not the real PPO loss --
        # this measures the COST of the update phase, not its correctness.
        flat_obs = buf_obs.reshape(-1, OBS_SIZE)
        flat_act = buf_act.reshape(-1, ACTION_SIZE)
        flat_logp = buf_logp.reshape(-1)
        total = flat_obs.shape[0]
        batch = total // minibatches
        for _ in range(epochs):
            order = torch.randperm(total, device=device)
            for start in range(0, total, batch):
                sel = order[start:start + batch]
                dist, _ = policy.distribution(flat_obs[sel])
                logp = dist.log_prob(flat_act[sel]).sum(-1)
                loss = ((logp - flat_logp[sel]).exp().mean()
                        + policy.value(flat_obs[sel]).pow(2).mean()
                        - dist.entropy().sum(-1).mean())
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimiser.step()

        torch.cuda.synchronize()
        t2 = time.perf_counter()
        if measuring:
            roll_s += t1 - t0
            upd_s += t2 - t1

    samples = worlds * rollout * MEASURE_ITERS
    iter_s = (roll_s + upd_s) / MEASURE_ITERS
    state = gpu_state()
    del env, policy, optimiser
    torch.cuda.empty_cache()
    return {
        "worlds": worlds, "rollout": rollout, "minibatches": minibatches,
        "samples_per_iter": worlds * rollout,
        "iter_s": iter_s,
        "rollout_s": roll_s / MEASURE_ITERS,
        "update_s": upd_s / MEASURE_ITERS,
        "update_frac": upd_s / (roll_s + upd_s),
        "steps_per_s": samples / (roll_s + upd_s),
        # The learning-side cost of a bigger batch, so a throughput win cannot be
        # reported without the thing it trades against being visible beside it.
        "updates_per_1k_samples": 1000.0 * epochs * minibatches / (worlds * rollout),
        **state,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worlds", type=int, nargs="+",
                        default=[4096, 6144, 8192, 12288, 16384])
    parser.add_argument("--rollout", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hold-updates", action="store_true",
                        help="Scale minibatches with worlds so updates per sample "
                             "stay constant -- the honest setting for a throughput "
                             "comparison that is meant to inform a real run.")
    parser.add_argument("--base-worlds", type=int, default=6144,
                        help="The worlds count --hold-updates holds the ratio of.")
    args = parser.parse_args()

    print(f"rollout {args.rollout}, epochs {args.epochs}, "
          f"minibatches {args.minibatches}"
          f"{' (scaled to hold updates/sample)' if args.hold_updates else ''}\n")
    header = (f"{'worlds':>7} {'mb':>4} {'iter s':>7} {'roll s':>7} {'upd s':>7} "
              f"{'upd%':>5} {'ksteps/s':>9} {'upd/1k':>7} {'C':>4} {'W':>6} "
              f"{'util':>5} {'VRAM':>6} throttle")
    print(header)
    rows = []
    for worlds in args.worlds:
        mb = args.minibatches
        if args.hold_updates:
            mb = max(1, round(args.minibatches * worlds / args.base_worlds))
        try:
            row = bench(worlds, args.rollout, args.epochs, mb, args.lr)
        except torch.cuda.OutOfMemoryError:
            print(f"{worlds:7d}  OUT OF MEMORY -- this is the ceiling")
            torch.cuda.empty_cache()
            break
        except RuntimeError as exc:
            # Warp raises plain RuntimeErrors for its own allocation failures, and
            # swallowing the text here would hide the reason the ceiling was hit.
            print(f"{worlds:7d}  FAILED: {exc}")
            torch.cuda.empty_cache()
            break
        rows.append(row)
        print(f"{row['worlds']:7d} {row['minibatches']:4d} {row['iter_s']:7.3f} "
              f"{row['rollout_s']:7.3f} {row['update_s']:7.3f} "
              f"{row['update_frac']:5.0%} {row['steps_per_s']/1000:9.1f} "
              f"{row['updates_per_1k_samples']:7.3f} "
              f"{row.get('temp_c','?'):>4} {row.get('power_w','?'):>6} "
              f"{row.get('util_pct','?'):>4}% {row.get('vram_mib','?'):>6} "
              f"{row.get('throttle','?')}")

    if rows:
        best = max(rows, key=lambda r: r["steps_per_s"])
        print(f"\n  fastest: {best['worlds']} worlds at "
              f"{best['steps_per_s']/1000:.1f}k steps/s "
              f"({best['iter_s']:.3f} s/iter, update is {best['update_frac']:.0%})")
        print("  NOTE: fastest in steps/s is not automatically fastest in LEARNING "
              "-- compare updates/1k samples before adopting it.")
        out = HERE / "runs" / "throughput_sweep.json"
        out.write_text(json.dumps(rows, indent=2))
        print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
