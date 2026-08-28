"""Headless evaluation: how many targets does the trained spider reach?
Run: python spider/eval_policy.py --headless --num_envs 256 --steps 900
"""
import argparse, glob, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256); p.add_argument("--steps", type=int, default=900)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app

import gymnasium as gym, torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
import importlib.metadata as metadata
from isaaclab_tasks.utils import load_cfg_from_registry
import spider  # noqa: registers task

env_cfg = load_cfg_from_registry("Isaac-Spider-Direct-v0", "env_cfg_entry_point"); env_cfg.scene.num_envs = a.num_envs
agent_cfg = load_cfg_from_registry("Isaac-Spider-Direct-v0", "rsl_rl_cfg_entry_point")
agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(gym.make("Isaac-Spider-Direct-v0", cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
ckpt = sorted(glob.glob("logs/rsl_rl/spider_direct/*/model_*.pt"), key=os.path.getmtime)[-1]
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.unwrapped.device); runner.load(ckpt)
policy = runner.get_inference_policy(device=env.unwrapped.device)
print("[eval] checkpoint:", ckpt)

raw = env.unwrapped; obs = env.get_observations(); reaches = 0; quick = 0; path = 0.0
age = torch.zeros(a.num_envs, device=raw.device); prev = raw.spider.data.root_pos_w.torch[:, :2].clone(); speed = 0.0
with torch.inference_mode():
    for i in range(a.steps):
        obs, _, _, _ = env.step(policy(obs))
        r = raw._reached
        reaches += int(r.sum().item()); quick += int((r & (age < 15)).sum().item())
        age += 1; age[r] = 0
        cur = raw.spider.data.root_pos_w.torch[:, :2]; path += (cur - prev).norm(dim=-1).mean().item(); prev = cur.clone()
        speed += raw.spider.data.root_lin_vel_b.torch[:, :2].norm(dim=-1).mean().item()
sim_s = a.steps * raw.step_dt
print(f"[eval] {a.num_envs} spiders x {sim_s:.0f}s: {reaches} targets reached -> {reaches / a.num_envs / sim_s * 60:.2f} targets/spider/minute")
print(f"[eval] reached within 0.5s of assignment (lucky spawns): {quick} ({100*quick/max(reaches,1):.0f}%)")
print(f"[eval] mean path length per spider: {path:.2f} m  -> {path/sim_s:.2f} m/s ; mean |v_xy| {speed/a.steps:.2f} m/s")
app.close()
