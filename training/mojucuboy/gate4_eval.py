"""Gate 4: deterministic evaluation of a trained policy.

  .venv-mjwarp\\Scripts\\python.exe training/mojucuboy/gate4_eval.py --run boy_chase01

Runs the DETERMINISTIC policy (no sampling) over a fixed set of episodes from a
fixed seed, and reports the agreed convergence metrics:

  mean episode length  >= 900 / 1000 steps   (>= 18 s of a 20 s episode)
  mean forward speed   >= 1.2 m/s            along the COMMANDED heading
  survival             >= 90 / 100 episodes

Domain randomisation stays ON. A policy that only survives the nominal model is
not the thing being shipped -- Unity's model is nominal but its solver, contacts
and timing are not bit-identical, so the margin randomisation buys is the margin
that carries the policy across.

The same routine is what Phase 5 compares against in Unity, so keep the metric
definitions here and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mojucuboy_env  # noqa: E402
from mojucuboy_env import ACTION_SIZE, MojucuBoyEnv  # noqa: E402
from train_mojucuboy import RESULTS, ActorCritic  # noqa: E402

TARGET_EP_LEN = 900
TARGET_SPEED = 1.2
TARGET_SURVIVAL = 0.90
# Uptime replaces episode length and survival as the stability gate. Both of
# those are degenerate while `done` is timeout-only: no world ever terminates
# early, so length is always EPISODE_STEPS and survival always 1.0, for any
# policy including an untrained one.
TARGET_UPTIME = 0.90
TARGET_FALLEN = 0.10


def evaluate(policy, episodes: int, seed: int, randomise: bool = True):
    """One deterministic pass. Each world runs exactly one episode; a world that
    terminates early is frozen rather than reset, so every episode is independent
    and the mean is not biased toward short ones."""
    env = MojucuBoyEnv(episodes, seed=seed)
    if not randomise:
        import warp as wp
        wp.to_torch(env.wm.actuator_gainprm).copy_(env.nominal_gain)
        wp.to_torch(env.wm.actuator_biasprm).copy_(env.nominal_bias)
        wp.to_torch(env.wm.body_mass).copy_(env.nominal_mass)
        wp.to_torch(env.wm.geom_friction).copy_(env.nominal_friction)

    device = torch.device("cuda:0")
    alive = torch.ones(episodes, dtype=torch.bool, device=device)
    length = torch.zeros(episodes, device=device)
    speed_sum = torch.zeros(episodes, device=device)
    ret = torch.zeros(episodes, device=device)
    # Uptime, not episode length, is what survival means in this env. `done` is
    # timeout-only -- a fall no longer ends the episode -- so `length` is always
    # EPISODE_STEPS and `survival_rate` below is always exactly 1.0, for any
    # policy including an untrained one. These two accumulate the per-step
    # standing/fallen fractions, which a policy lying on the floor genuinely fails.
    standing_sum = torch.zeros(episodes, device=device)
    fallen_sum = torch.zeros(episodes, device=device)
    obs = env.observation()

    with torch.no_grad():
        for _ in range(mojucuboy_env.EPISODE_STEPS):
            action = policy(obs)
            obs, reward, done, terms = env.step(action)
            length += alive.float()
            speed_sum += terms["speed_along"] * alive.float()
            standing_sum += terms["standing"] * alive.float()
            fallen_sum += terms["fallen"] * alive.float()
            ret += reward * alive.float()
            # Freeze on first termination; env.step keeps integrating those worlds
            # but their statistics stop accumulating.
            alive &= ~done
            if not alive.any():
                break

    survived = length >= mojucuboy_env.EPISODE_STEPS
    mean_speed = speed_sum / length.clamp(min=1)
    uptime = standing_sum / length.clamp(min=1)
    fallen = fallen_sum / length.clamp(min=1)
    return {
        "episodes": episodes,
        # Kept for continuity, but see the note above: both are degenerate while
        # `done` is timeout-only. Read mean_uptime / mean_fallen instead.
        "mean_episode_length": float(length.mean()),
        "median_episode_length": float(length.median()),
        "survival_rate": float(survived.float().mean()),
        "survivors": int(survived.sum()),
        "mean_forward_speed": float(mean_speed.mean()),
        "mean_return": float(ret.mean()),
        "mean_uptime": float(uptime.mean()),
        "mean_fallen": float(fallen.mean()),
        "uptime_over_90pct": float((uptime > 0.90).float().mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    run_dir = RESULTS / args.run
    checkpoint = torch.load(run_dir / "policy.pt", map_location="cuda:0", weights_only=False)
    policy = ActorCritic().to("cuda:0")
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    print(f"run {args.run}, iteration {checkpoint.get('iteration', '?')}, "
          f"{args.episodes} deterministic episodes of {mojucuboy_env.EPISODE_STEPS} steps "
          f"({mojucuboy_env.EPISODE_STEPS * 0.02:.0f} s)\n")

    results = {}
    for label, randomise in (("randomised", True), ("nominal", False)):
        stats = evaluate(policy, args.episodes, args.seed, randomise)
        results[label] = stats
        print(f"=== {label.upper()} MODEL ===")
        print(f"  mean episode length : {stats['mean_episode_length']:7.1f} / "
              f"{mojucuboy_env.EPISODE_STEPS}   (target >= {TARGET_EP_LEN})")
        print(f"  median ep length    : {stats['median_episode_length']:7.1f}")
        print(f"  mean forward speed  : {stats['mean_forward_speed']:7.3f} m/s"
              f"        (target >= {TARGET_SPEED})")
        print(f"  survival            : {stats['survivors']:4d} / {stats['episodes']}"
              f"          (degenerate: always 100%, see mean uptime)")
        print(f"  mean uptime         : {stats['mean_uptime']:7.3f}"
              f"            (target >= {TARGET_UPTIME})")
        print(f"  mean fallen         : {stats['mean_fallen']:7.3f}"
              f"            (target <= {TARGET_FALLEN})")
        print(f"  episodes >90% up    : {stats['uptime_over_90pct']:7.1%}")
        print(f"  mean return         : {stats['mean_return']:7.2f}\n")

    (run_dir / "gate4_eval.json").write_text(json.dumps(results, indent=2))

    primary = results["randomised"]
    # "mean episode length" and "survival rate" are deliberately NOT gated on:
    # both are constant by construction while `done` is timeout-only, so a gate
    # on them passes an untrained policy. Uptime and fallen replace them.
    checks = [
        ("mean forward speed", primary["mean_forward_speed"], TARGET_SPEED, "ge"),
        ("mean uptime", primary["mean_uptime"], TARGET_UPTIME, "ge"),
        ("mean fallen", primary["mean_fallen"], TARGET_FALLEN, "le"),
    ]
    print("=== GATE 4 VERDICT (randomised model) ===")
    passed = True
    for name, value, target, direction in checks:
        ok = value >= target if direction == "ge" else value <= target
        passed &= ok
        arrow = ">=" if direction == "ge" else "<="
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} {value:8.3f}  vs target {arrow} {target}")
    print(f"  {'PASS' if passed else 'FAIL'}: overall")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
