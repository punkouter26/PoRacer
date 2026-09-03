"""Measure MojucuBoy's actual steering envelope.

The policy is heading-tracked: its observation carries cos/sin of the heading
error, not a goal position. During training the commanded heading is sampled
within +/-0.6 rad of the spawn yaw, so the interesting question is what happens
OUTSIDE that band -- can it turn 90 degrees? 180?

Each trial spawns the racer facing +y with a fixed commanded heading offset, runs
a deterministic rollout, and reports where it actually went. Reported per offset:

  final heading error  how far it still is from the commanded direction
  speed along command  progress in the direction it was told to go
  cross-track drift    how far it slid sideways
  survived             whether it stayed upright for the whole rollout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mojucuboy_env  # noqa: E402
from mojucuboy_env import MojucuBoyEnv  # noqa: E402
from train_mojucuboy import RESULTS, ActorCritic  # noqa: E402

OFFSETS_DEG = [0, 15, 30, 45, 60, 90, 135, 180, -30, -60, -90, -180]
STEPS = 400   # 8 s


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="boy_chase01")
    parser.add_argument("--repeats", type=int, default=8)
    args = parser.parse_args()

    checkpoint = torch.load(RESULTS / args.run / "policy.pt",
                            map_location="cuda:0", weights_only=False)
    policy = ActorCritic().to("cuda:0")
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    n = len(OFFSETS_DEG) * args.repeats
    env = MojucuBoyEnv(n, seed=99)
    device = env.device

    offsets = torch.tensor(
        np.repeat(np.radians(OFFSETS_DEG), args.repeats), device=device, dtype=torch.float32)

    # Spawn every world facing +y (yaw = pi/2 about z) so the commanded offset is
    # the only thing that differs between trials.
    env.reset(torch.arange(n, device=device))
    qpos = env.qpos
    spawn_yaw = torch.full((n,), float(np.pi / 2), device=device)
    qpos[:, 3] = torch.cos(spawn_yaw / 2).to(qpos.dtype)
    qpos[:, 4] = 0
    qpos[:, 5] = 0
    qpos[:, 6] = torch.sin(spawn_yaw / 2).to(qpos.dtype)
    env.qvel[:] = 0
    env.command_heading[:] = spawn_yaw + offsets
    env.last_action[:] = 0
    start = qpos[:, 0:2].clone().float()

    alive = torch.ones(n, dtype=torch.bool, device=device)
    speed_sum = torch.zeros(n, device=device)
    steps = torch.zeros(n, device=device)
    obs = env.observation()
    with torch.no_grad():
        for _ in range(STEPS):
            obs, _, done, terms = env.step(policy(obs))
            speed_sum += terms["speed_along"] * alive.float()
            steps += alive.float()
            alive &= ~done

    rot = env.root_rotation()
    forward = rot[:, :, mojucuboy_env.FORWARD_AXIS]
    heading = torch.atan2(forward[:, 1], forward[:, 0])
    error = (env.command_heading - heading + np.pi) % (2 * np.pi) - np.pi

    travel = env.qpos[:, 0:2].float() - start
    cmd = torch.stack([torch.cos(env.command_heading), torch.sin(env.command_heading)], 1)
    along = (travel * cmd).sum(1)
    cross = (travel * torch.stack([-cmd[:, 1], cmd[:, 0]], 1)).sum(1)
    mean_speed = speed_sum / steps.clamp(min=1)

    print(f"run {args.run}, iteration {checkpoint.get('iteration', '?')}, "
          f"{args.repeats} repeats x {STEPS} steps ({STEPS * 0.02:.0f} s)\n")
    print("Trained heading band is +/-34 deg; everything outside it is extrapolation.\n")
    print(f"{'cmd offset':>11}{'final err':>11}{'speed along':>13}"
          f"{'dist along':>12}{'cross drift':>13}{'survived':>10}")
    for i, deg in enumerate(OFFSETS_DEG):
        sel = slice(i * args.repeats, (i + 1) * args.repeats)
        band = "" if abs(deg) <= 34 else "   (extrapolated)"
        print(f"{deg:>8} deg"
              f"{np.degrees(error[sel].abs().mean().item()):>8.1f} deg"
              f"{mean_speed[sel].mean().item():>10.2f} m/s"
              f"{along[sel].mean().item():>10.2f} m"
              f"{cross[sel].abs().mean().item():>11.2f} m"
              f"{alive[sel].float().mean().item():>9.0%}{band}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
