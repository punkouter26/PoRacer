"""Train the creature with Brax PPO on MJX.

  python train.py --num_timesteps 60_000_000 --num_envs 4096

Checkpoints land in mjx_training/runs/<name>/. Export to Unity afterwards with:
  python export_policy.py --run <name>
"""
from __future__ import annotations

import argparse, functools, json, os, time
from datetime import datetime

os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".70")

from pathlib import Path

import jax
from brax.io import model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from mujoco_playground import wrapper

from creature_env import Creature, ACTION_SIZE, OBS_LAYOUT, OBS_SIZE

RUNS = Path(__file__).resolve().parent / "runs"


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--name", default=None, help="run name (default: timestamp)")
  p.add_argument("--num_timesteps", type=int, default=60_000_000)
  p.add_argument("--num_envs", type=int, default=4096)
  p.add_argument("--episode_length", type=int, default=1000)
  # Keep batch_size * num_minibatches == num_envs, the ratio MuJoCo Playground
  # uses for Go1/Barkour (256 * 32 == 8192 envs). Going larger collects several
  # unrolls per training step and cuts the number of gradient updates
  # proportionally -- 1024 here gave 8x fewer updates and visibly slower learning.
  p.add_argument("--batch_size", type=int, default=128)
  p.add_argument("--num_minibatches", type=int, default=32)
  p.add_argument("--unroll_length", type=int, default=20)
  p.add_argument("--num_updates_per_batch", type=int, default=4)
  p.add_argument("--learning_rate", type=float, default=3e-4)
  p.add_argument("--entropy_cost", type=float, default=1e-2)
  p.add_argument("--discounting", type=float, default=0.97)
  p.add_argument("--num_evals", type=int, default=10)
  p.add_argument("--seed", type=int, default=0)
  p.add_argument("--impl", choices=["jax", "warp"], default="jax",
                 help="MJX physics backend")
  p.add_argument("--policy_hidden", type=int, nargs="+", default=[128, 128, 128])
  p.add_argument("--value_hidden", type=int, nargs="+", default=[256, 256, 256])
  args = p.parse_args()

  run_name = args.name or datetime.now().strftime("%Y%m%d-%H%M%S")
  out = RUNS / run_name
  out.mkdir(parents=True, exist_ok=True)

  print(f"jax {jax.__version__} | backend {jax.default_backend()} | {jax.devices()}")
  print(f"run: {out}")

  if args.impl == "warp":
    print(
        "\n*** WARNING: --impl warp produces WRONG PHYSICS in this training stack.\n"
        "    Brax/Playground drive envs through jax.vmap, and mjx.make_data exposes\n"
        "    only the batch-wide naconmax, never the per-world nconmax. The broadphase\n"
        "    overflows on every step, contacts are dropped, and the creature sinks\n"
        "    through the floor -- while appearing FASTER because it skips collision\n"
        "    work. Measured torso height: -6.9 m.\n"
        "    MJX-JAX is also 1.85x faster on this creature (3.11M vs 1.68M steps/s).\n"
        "    Use --impl jax. See the backend section of README.md.\n"
        "    Continuing anyway; grep the log for 'broadphase overflow'.\n",
        flush=True)

  env = Creature(config_overrides={"impl": args.impl})
  eval_env = Creature(config_overrides={"impl": args.impl})
  print(f"physics backend: {env.mjx_model.impl}")
  print(f"obs={OBS_SIZE} act={ACTION_SIZE} ctrl_dt={env.dt} n_substeps={env.n_substeps}")

  # Written BEFORE training: export_policy.py needs it, and a run that is
  # interrupted must still leave an exportable checkpoint behind.
  (out / "config.json").write_text(json.dumps({
      "run": run_name,
      "obs_size": OBS_SIZE,
      "action_size": ACTION_SIZE,
      "obs_layout": [list(x) for x in OBS_LAYOUT],
      "policy_hidden": args.policy_hidden,
      "value_hidden": args.value_hidden,
      "ctrl_dt": env.dt,
      "sim_dt": env.sim_dt,
      "n_substeps": env.n_substeps,
      "num_timesteps": args.num_timesteps,
      "num_envs": args.num_envs,
      "seed": args.seed,
      "impl": args.impl,
  }, indent=2))

  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      policy_hidden_layer_sizes=tuple(args.policy_hidden),
      value_hidden_layer_sizes=tuple(args.value_hidden),
  )

  history = []
  t_start = time.time()

  def checkpoint(current_step: int, make_policy, params):
    """Save at every eval, so killing a run never throws the policy away.

    Brax calls this after each evaluation. Without it, params only hit disk when
    train() returns -- an interrupted run loses everything it learned.
    """
    del make_policy
    model.save_params(str(out / "params"), params)
    model.save_params(str(out / f"params_{current_step}"), params)
    print(f"           checkpoint saved at {current_step:,} steps", flush=True)

  def progress(num_steps: int, metrics: dict):
    reward = float(metrics.get("eval/episode_reward", float("nan")))
    length = float(metrics.get("eval/avg_episode_length", float("nan")))
    elapsed = time.time() - t_start
    sps = num_steps / max(elapsed, 1e-9)
    print(f"[{elapsed:7.1f}s] steps={num_steps:>12,}  reward={reward:8.2f}  "
          f"ep_len={length:7.1f}  {sps:,.0f} steps/s", flush=True)
    history.append({"steps": int(num_steps), "reward": reward,
                    "episode_length": length, "elapsed_s": elapsed,
                    **{k: float(v) for k, v in metrics.items()
                       if k.startswith("eval/episode_")}})
    (out / "progress.json").write_text(json.dumps(history, indent=2))

  train_fn = functools.partial(
      ppo.train,
      num_timesteps=args.num_timesteps,
      num_envs=args.num_envs,
      episode_length=args.episode_length,
      batch_size=args.batch_size,
      num_minibatches=args.num_minibatches,
      unroll_length=args.unroll_length,
      num_updates_per_batch=args.num_updates_per_batch,
      learning_rate=args.learning_rate,
      entropy_cost=args.entropy_cost,
      discounting=args.discounting,
      num_evals=args.num_evals,
      normalize_observations=True,
      reward_scaling=1.0,
      action_repeat=1,
      seed=args.seed,
      network_factory=network_factory,
      wrap_env_fn=wrapper.wrap_for_brax_training,
  )

  make_inference_fn, params, _ = train_fn(
      environment=env, eval_env=eval_env, progress_fn=progress,
      policy_params_fn=checkpoint)

  model.save_params(str(out / "params"), params)

  print(f"\nsaved params -> {out/'params'}")
  print(f"next: python export_policy.py --run {run_name}")


if __name__ == "__main__":
  main()
