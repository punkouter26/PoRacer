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
    obs = env.observation()

    with torch.no_grad():
        for _ in range(mojucuboy_env.EPISODE_STEPS):
            action = policy(obs)
            obs, reward, done, terms = env.step(action)
            length += alive.float()
            speed_sum += terms["speed_along"] * alive.float()
            ret += reward * alive.float()
            # Freeze on first termination; env.step keeps integrating those worlds
            # but their statistics stop accumulating.
            alive &= ~done
            if not alive.any():
                break

    survived = length >= mojucuboy_env.EPISODE_STEPS
    mean_speed = speed_sum / length.clamp(min=1)
    return {
        "episodes": episodes,
        "mean_episode_length": float(length.mean()),
        "median_episode_length": float(length.median()),
        "mean_forward_speed": float(mean_speed.mean()),
        "mean_return": float(ret.mean()),
        "survival_rate": float(survived.float().mean()),
        "survivors": int(survived.sum()),
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
              f"          (target >= {TARGET_SURVIVAL:.0%})")
        print(f"  mean return         : {stats['mean_return']:7.2f}\n")

    (run_dir / "gate4_eval.json").write_text(json.dumps(results, indent=2))

    primary = results["randomised"]
    checks = [
        ("mean episode length", primary["mean_episode_length"], TARGET_EP_LEN),
        ("mean forward speed", primary["mean_forward_speed"], TARGET_SPEED),
        ("survival rate", primary["survival_rate"], TARGET_SURVIVAL),
    ]
    print("=== GATE 4 VERDICT (randomised model) ===")
    passed = True
    for name, value, target in checks:
        ok = value >= target
        passed &= ok
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} {value:8.3f}  vs target {target}")
    print(f"  {'PASS' if passed else 'FAIL'}: overall")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
