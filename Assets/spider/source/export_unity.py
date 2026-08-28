"""Export the trained spider policy for Unity + record reference data + evaluate.
Run: python spider/export_unity.py --headless --num_envs 256
Writes to unity_export/spider/: spider.onnx (single file), isaac_reference.json, export_report.json
"""
import argparse, glob, json, os
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256); p.add_argument("--eval_steps", type=int, default=900)
p.add_argument("--ref_steps", type=int, default=200); p.add_argument("--run", type=str, default=None)
AppLauncher.add_app_launcher_args(p); a = p.parse_args()
app = AppLauncher(a).app

import importlib.metadata as metadata
import gymnasium as gym, numpy as np, onnx, onnxruntime as ort, torch
from onnx.external_data_helper import convert_model_from_external_data
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
import isaaclab.sim as sim_utils
import spider  # noqa: registers task
from spider.spider_env import JOINT_NAMES

OUT = os.path.abspath("unity_export/spider"); os.makedirs(OUT, exist_ok=True)
TASK = "Isaac-Spider-Direct-v0"
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point"); env_cfg.scene.num_envs = a.num_envs
agent_cfg = handle_deprecated_rsl_rl_cfg(load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib"))
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
raw = env.unwrapped
run_dir = a.run or sorted(glob.glob("logs/rsl_rl/spider_direct/*"), key=os.path.getmtime)[-1]
ckpt = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=lambda f: int(os.path.basename(f)[6:-3]))[-1]
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=raw.device); runner.load(ckpt)
policy = runner.get_inference_policy(device=raw.device)
print("[export] checkpoint:", ckpt)

# ---- 1. ONNX: export (rsl_rl writes external data), then fold into a single file
tmp = os.path.join(OUT, "_tmp_onnx"); runner.export_policy_to_onnx(path=tmp, filename="policy.onnx")
m = onnx.load(os.path.join(tmp, "policy.onnx"), load_external_data=True); convert_model_from_external_data(m)
from onnx import version_converter
m = version_converter.convert_version(m, 15); m.ir_version = 8  # Unity Sentis supports opset 7-15
onnx_path = os.path.join(OUT, "spider.onnx"); onnx.save_model(m, onnx_path, save_as_external_data=False); onnx.checker.check_model(onnx_path)
for f in glob.glob(os.path.join(tmp, "*")): os.remove(f)
os.rmdir(tmp)
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
io = {"inputs": [(i.name, i.shape) for i in sess.get_inputs()], "outputs": [(o.name, o.shape) for o in sess.get_outputs()]}
print("[export] onnx io:", io, "size:", os.path.getsize(onnx_path), "bytes")

# ---- 2. Reference recording (env 0) + ONNX-vs-torch check on the same observations
obs = env.get_observations(); ref = []; max_diff = 0.0
with torch.inference_mode():
    for i in range(a.ref_steps):
        act = policy(obs)
        o0 = obs["policy"][0].detach().cpu().numpy().astype(np.float32)
        onnx_act = sess.run(None, {sess.get_inputs()[0].name: o0[None]})[0][0]
        max_diff = max(max_diff, float(np.abs(onnx_act - act[0].cpu().numpy()).max()))
        d = raw.spider.data
        ref.append({"step": i, "t": i * raw.step_dt, "obs": o0.tolist(), "action": act[0].cpu().numpy().tolist(),
                    "root_pos_w": (d.root_pos_w.torch[0] - raw.scene.env_origins[0]).cpu().tolist(),
                    "root_quat_w_wxyz": d.root_quat_w.torch[0].cpu().tolist(),
                    "joint_pos": d.joint_pos.torch[0, raw._joint_ids].cpu().tolist(),
                    "target_rel": (raw.targets_w[0] - raw.scene.env_origins[0]).cpu().tolist()})
        obs, _, _, _ = env.step(act)
print(f"[export] onnx vs torch max|diff| over {a.ref_steps} obs: {max_diff:.2e}")

# ---- 3. Ground-truth numbers from the live sim
gp = sim_utils.GroundPlaneCfg().physics_material
d = raw.spider.data
_, jn = raw.spider.find_joints(JOINT_NAMES, preserve_order=True)
lim = d.joint_pos_limits.torch[0, raw._joint_ids].cpu().tolist()
masses = raw.spider.data.default_mass.torch[0].cpu().tolist() if hasattr(raw.spider.data, "default_mass") else None
body_names = raw.spider.body_names
report = {
    "checkpoint": ckpt, "onnx": onnx_path, "onnx_io": io, "onnx_vs_torch_max_abs_diff": max_diff,
    "obs_dim": int(obs["policy"].shape[1]), "act_dim": 16, "joint_order": jn, "joint_pos_limits_rad": lim,
    "body_names": body_names, "body_masses_kg": masses,
    "sim_dt": raw.sim.get_physics_dt(), "decimation": raw.cfg.decimation, "policy_dt": raw.step_dt,
    "gravity": list(raw.cfg.sim.gravity), "action_scale": raw.cfg.action_scale,
    "actuator": {"stiffness": 25.0, "damping": 1.0, "effort_limit_sim": 15.0, "velocity_limit_sim": 12.0},
    "ground_material": {"static_friction": gp.static_friction, "dynamic_friction": gp.dynamic_friction, "restitution": gp.restitution,
                        "friction_combine_mode": gp.friction_combine_mode, "restitution_combine_mode": gp.restitution_combine_mode},
    "init_pos_isaac": [0.0, 0.0, 0.18], "standing_body_height_m": 0.141, "target_radius_range": list(raw.cfg.target_radius_range), "reach_threshold": raw.cfg.reach_threshold,
    "episode_length_s": raw.cfg.episode_length_s,
}

# ---- 4. Evaluation (all envs)
reaches = 0; quick = 0; path = 0.0; age = torch.zeros(a.num_envs, device=raw.device); prev = d.root_pos_w.torch[:, :2].clone()
with torch.inference_mode():
    for i in range(a.eval_steps):
        obs, _, _, _ = env.step(policy(obs))
        r = raw._reached; reaches += int(r.sum()); quick += int((r & (age < 15)).sum()); age += 1; age[r] = 0
        cur = d.root_pos_w.torch[:, :2]; path += (cur - prev).norm(dim=-1).mean().item(); prev = cur.clone()
sim_s = a.eval_steps * raw.step_dt
report["evaluation"] = {"num_envs": a.num_envs, "seconds": sim_s, "targets_reached": reaches,
                        "targets_per_spider_per_minute": reaches / a.num_envs / sim_s * 60, "lucky_spawn_fraction": quick / max(reaches, 1),
                        "mean_speed_m_s": path / sim_s}
print("[export] eval:", json.dumps(report["evaluation"]))
json.dump(ref, open(os.path.join(OUT, "isaac_reference.json"), "w"))
json.dump(report, open(os.path.join(OUT, "export_report.json"), "w"), indent=2)
print("[export] report:", json.dumps({k: v for k, v in report.items() if k not in ("evaluation",)}, default=str))
app.close()
