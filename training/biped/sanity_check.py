"""Rig sanity check: drop the biped, hold the nominal pose, see whether it stands.

Run: python biped/sanity_check.py --headless [--num_envs 64] [--steps 250]
"""
import argparse

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--steps", type=int, default=250)
p.add_argument("--random", action="store_true", help="drive random actions instead of holding the pose")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import biped  # noqa: F401,E402  registers the task

cfg = load_cfg_from_registry("Isaac-Biped-Direct-v0", "env_cfg_entry_point")
cfg.scene.num_envs = a.num_envs
env = gym.make("Isaac-Biped-Direct-v0", cfg=cfg).unwrapped
env.reset()

print(f"[sanity] bodies: {env.robot.data.body_names}")
print(f"[sanity] total mass: {env.robot.data.default_mass.torch[0].sum().item():.2f} kg")

heights, falls = [], 0
with torch.inference_mode():
    for i in range(a.steps):
        act = torch.zeros(a.num_envs, env.cfg.action_space, device=env.device)
        if a.random:
            act.uniform_(-1.0, 1.0)
        env.step(act)
        h = env.robot.data.root_pos_w.torch[:, 2]
        heights.append(h.mean().item())
        falls += int(env._fallen.sum().item())
        if i % 50 == 0:
            g = env.robot.data.projected_gravity_b.torch[:, 2]
            print(f"[sanity] step {i:4d}  torso z: mean {h.mean():.3f} min {h.min():.3f}  upright {-g.mean():.3f}")

settled = sum(heights[-50:]) / 50
print(f"\n[sanity] settled torso height over the last 50 steps: {settled:.3f} m (nominal {cfg.nominal_height:.3f})")
print(f"[sanity] fall events across {a.num_envs} robots x {a.steps} steps: {falls}")
print("[sanity] VERDICT:", "rig stands" if settled > cfg.min_height and falls == 0 else "rig is falling over")
env.close()
app.close()
