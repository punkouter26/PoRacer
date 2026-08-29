"""Export the trained biped policy as a self-contained Unity Sentis bundle.

Writes everything a Unity port needs into ``unity_export/biped/``: the ONNX policy,
a reference trajectory to validate against, a machine-readable report of every physical
constant, the robot description, the raw checkpoint, the task source, and a README.

Run: python biped/export_unity.py --headless --num_envs 256
"""

import argparse
import glob
import json
import os
import shutil

from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--num_envs", type=int, default=256)
p.add_argument("--eval_steps", type=int, default=1500, help="policy steps for the evaluation sweep")
p.add_argument("--ref_steps", type=int, default=250, help="policy steps recorded into the reference trajectory")
p.add_argument("--run", type=str, default=None, help="training run dir; defaults to the newest")
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
app = AppLauncher(a).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from onnx import version_converter  # noqa: E402
from onnx.external_data_helper import convert_model_from_external_data  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402

import biped  # noqa: F401,E402  registers the task
from biped.biped_env import JOINT_NAMES  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath("unity_export/biped")
TASK = "Isaac-Biped-Direct-v0"
OPSET, IR_VERSION = 15, 8  # Unity Sentis loads opset 7-15, IR <= 8

# Observation layout -- keep in sync with BipedEnv._get_observations.
OBS_SEGMENTS = [
    ("joint_pos", 10, "joint positions minus the standing pose [rad], joint_order below"),
    ("joint_vel", 10, "joint velocities [rad/s] x 0.1"),
    ("root_lin_vel_b", 3, "torso linear velocity in the torso frame [m/s]"),
    ("root_ang_vel_b", 3, "torso angular velocity in the torso frame [rad/s] x 0.25"),
    ("projected_gravity_b", 3, "gravity unit vector in the torso frame; (0,0,-1) upright"),
    ("target_dir_b", 2, "unit vector to the target in the torso yaw-only frame; x fwd, y left"),
    ("target_dist", 1, "horizontal distance to the target [m] / 5, clamped to 1"),
    ("prev_actions", 10, "previous action, clamped to [-1,1]"),
]

os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------- load
env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
env_cfg.scene.num_envs = a.num_envs
agent_cfg = handle_deprecated_rsl_rl_cfg(
    load_cfg_from_registry(TASK, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
)
env = RslRlVecEnvWrapper(gym.make(TASK, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
raw = env.unwrapped

run_dir = a.run or sorted(glob.glob("logs/rsl_rl/biped_direct/*"), key=os.path.getmtime)[-1]
ckpt = sorted(glob.glob(os.path.join(run_dir, "model_*.pt")), key=lambda f: int(os.path.basename(f)[6:-3]))[-1]
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=raw.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=raw.device)
print(f"[export] run dir:    {run_dir}")
print(f"[export] checkpoint: {ckpt}")

# ------------------------------------------------- 1. ONNX, single file, opset 15
tmp = os.path.join(OUT, "_tmp_onnx")
runner.export_policy_to_onnx(path=tmp, filename="policy.onnx")
model = onnx.load(os.path.join(tmp, "policy.onnx"), load_external_data=True)
convert_model_from_external_data(model)  # fold weights into the graph itself
model = version_converter.convert_version(model, OPSET)
model.ir_version = IR_VERSION
onnx_path = os.path.join(OUT, "biped.onnx")
onnx.save_model(model, onnx_path, save_as_external_data=False)
onnx.checker.check_model(onnx_path)
shutil.rmtree(tmp, ignore_errors=True)

model = onnx.load(onnx_path)
ops = sorted({n.op_type for n in model.graph.node})
opsets = {imp.domain or "ai.onnx": imp.version for imp in model.opset_import}
sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
onnx_io = {
    "inputs": [(i.name, i.shape, i.type) for i in sess.get_inputs()],
    "outputs": [(o.name, o.shape, o.type) for o in sess.get_outputs()],
}
in_shape, out_shape = sess.get_inputs()[0].shape, sess.get_outputs()[0].shape
fixed_batch = all(isinstance(d, int) for d in list(in_shape) + list(out_shape)) and in_shape[0] == 1
print(f"[export] onnx opset {opsets} ir {model.ir_version} ops {ops}")
print(f"[export] onnx io {onnx_io} fixed_batch_1={fixed_batch} size={os.path.getsize(onnx_path)} B")

# ----------------------- 2. reference trajectory + ONNX-vs-PyTorch on real observations
d = raw.robot.data
obs = env.get_observations()
ref, max_diff = [], 0.0
in_name = sess.get_inputs()[0].name
with torch.inference_mode():
    for i in range(a.ref_steps):
        act = policy(obs)
        o0 = obs["policy"][0].detach().cpu().numpy().astype(np.float32)
        onnx_act = sess.run(None, {in_name: o0[None]})[0][0]
        max_diff = max(max_diff, float(np.abs(onnx_act - act[0].cpu().numpy()).max()))
        ref.append({
            "step": i,
            "t": i * raw.step_dt,
            "obs": o0.tolist(),
            "action": act[0].cpu().numpy().tolist(),
            "root_pos_w": (d.root_pos_w.torch[0] - raw.scene.env_origins[0]).cpu().tolist(),
            "root_quat_w_wxyz": d.root_quat_w.torch[0].cpu().tolist(),
            "joint_pos": d.joint_pos.torch[0, raw._joint_ids].cpu().tolist(),
            "target_rel": (raw.targets_w[0] - raw.scene.env_origins[0]).cpu().tolist(),
        })
        obs, _, _, _ = env.step(act)
print(f"[export] onnx vs torch, max |diff| over {a.ref_steps} real observations: {max_diff:.3e}")

# ---------------------------------------------- 3. ground truth from the live sim
ids = raw._joint_ids
_, joint_order = raw.robot.find_joints(JOINT_NAMES, preserve_order=True)
gp = sim_utils.GroundPlaneCfg().physics_material
init = env_cfg.robot_cfg.init_state

joints = []
for k, name in enumerate(joint_order):
    j = ids[k]
    joints.append({
        "index": k,
        "name": name,
        "default_rad": float(d.default_joint_pos.torch[0, j]),
        "lower_rad": float(d.joint_pos_limits.torch[0, j, 0]),
        "upper_rad": float(d.joint_pos_limits.torch[0, j, 1]),
        "stiffness_nm_per_rad": float(d.joint_stiffness.torch[0, j]),
        "damping_nms_per_rad": float(d.joint_damping.torch[0, j]),
        "effort_limit_nm": float(d.joint_effort_limits.torch[0, j]),
        "velocity_limit_rad_per_s": float(d.joint_velocity_limits.torch[0, j]),
    })

masses = d.body_mass.torch[0].cpu().tolist()
bodies = [{"name": n, "mass_kg": float(m)} for n, m in zip(raw.robot.body_names, masses)]

offset, obs_layout = 0, []
for name, n, desc in OBS_SEGMENTS:
    obs_layout.append({"name": name, "start": offset, "end": offset + n - 1, "size": n, "description": desc})
    offset += n
assert offset == raw.cfg.observation_space, f"obs layout {offset} != {raw.cfg.observation_space}"

report = {
    "task": TASK,
    "checkpoint": os.path.relpath(ckpt),
    "run_dir": os.path.relpath(run_dir),
    "onnx": {
        "file": "biped.onnx",
        "opset": opsets,
        "ir_version": model.ir_version,
        "operators": ops,
        "io": onnx_io,
        "fixed_batch_size_1": fixed_batch,
        "bytes": os.path.getsize(onnx_path),
        "external_data": False,
        "checker_passed": True,
        "vs_pytorch_max_abs_diff": max_diff,
        "obs_normalization": "baked into the graph",
        "note": "output is the Gaussian mean action; clamp to [-1,1] before use",
    },
    "spaces": {"obs_dim": raw.cfg.observation_space, "act_dim": raw.cfg.action_space, "obs_layout": obs_layout},
    "joints": joints,
    "joint_order": joint_order,
    "bodies": bodies,
    "total_mass_kg": float(sum(masses)),
    "timing": {
        "physics_dt_s": raw.sim.get_physics_dt(),
        "physics_hz": 1.0 / raw.sim.get_physics_dt(),
        "decimation": raw.cfg.decimation,
        "policy_dt_s": raw.step_dt,
        "policy_hz": 1.0 / raw.step_dt,
        "episode_length_s": raw.cfg.episode_length_s,
    },
    "gravity_m_s2": list(raw.cfg.sim.gravity),
    "ground_material": {
        "static_friction": gp.static_friction,
        "dynamic_friction": gp.dynamic_friction,
        "restitution": gp.restitution,
        "friction_combine_mode": gp.friction_combine_mode,
        "restitution_combine_mode": gp.restitution_combine_mode,
    },
    "spawn": {
        "root_pos_isaac_m": list(init.pos),
        "root_quat_wxyz": list(init.rot),
        "joint_pos_rad": {k: v for k, v in init.joint_pos.items()},
        "standing_hip_height_m": raw.cfg.nominal_height,
        "note": "the torso link origin sits on the hip line, so root z is the hip height",
    },
    "task_params": {
        "action_scale_rad": raw.cfg.action_scale,
        "action_to_target": "joint_target[i] = default_joint_pos[i] + action_scale * clamp(action[i], -1, 1)",
        "target_radius_range_m": list(raw.cfg.target_radius_range),
        "reach_threshold_m": raw.cfg.reach_threshold,
        "min_height_m": raw.cfg.min_height,
        "max_tilt_projected_gravity_z": raw.cfg.max_tilt,
        "air_time_target_s": raw.cfg.air_time_target,
        "foot_contact_height_m": raw.cfg.foot_contact_height,
    },
}

# ------------------------------------------------------------- 4. evaluation
reaches = quick = falls = 0
path = 0.0
age = torch.zeros(a.num_envs, device=raw.device)
prev = d.root_pos_w.torch[:, :2].clone()
with torch.inference_mode():
    for i in range(a.eval_steps):
        obs, _, _, _ = env.step(policy(obs))
        r = raw._reached
        reaches += int(r.sum())
        quick += int((r & (age < 15)).sum())
        falls += int(raw._fallen.sum())
        age += 1
        age[r] = 0
        cur = d.root_pos_w.torch[:, :2]
        path += (cur - prev).norm(dim=-1).mean().item()
        prev = cur.clone()
sim_s = a.eval_steps * raw.step_dt
report["evaluation"] = {
    "num_envs": a.num_envs,
    "seconds": sim_s,
    "targets_reached": reaches,
    "targets_per_biped_per_minute": reaches / a.num_envs / sim_s * 60,
    "lucky_spawn_fraction": quick / max(reaches, 1),
    "falls": falls,
    "falls_per_biped_per_minute": falls / a.num_envs / sim_s * 60,
    "mean_speed_m_s": path / sim_s,
}
print("[export] evaluation:", json.dumps(report["evaluation"], indent=2))

# --------------------------------------------------------- 5. bundle the files
robot_dir = os.path.join(OUT, "robot")
os.makedirs(robot_dir, exist_ok=True)
shutil.copy2(os.path.join(HERE, "assets", "biped.urdf"), os.path.join(robot_dir, "biped.urdf"))
usd_src = os.path.join(HERE, "assets", "biped_usd")
usd_dst = os.path.join(robot_dir, "biped_usd")
shutil.rmtree(usd_dst, ignore_errors=True)
shutil.copytree(usd_src, usd_dst, ignore=shutil.ignore_patterns(".asset_hash"))

ck_dir = os.path.join(OUT, "checkpoint")
os.makedirs(ck_dir, exist_ok=True)
shutil.copy2(ckpt, os.path.join(ck_dir, os.path.basename(ckpt)))
params_src = os.path.join(run_dir, "params")
if os.path.isdir(params_src):
    shutil.rmtree(os.path.join(ck_dir, "params"), ignore_errors=True)
    shutil.copytree(params_src, os.path.join(ck_dir, "params"))

src_dir = os.path.join(OUT, "source")
shutil.rmtree(src_dir, ignore_errors=True)
shutil.copytree(HERE, src_dir, ignore=shutil.ignore_patterns("__pycache__", "assets", "*.pyc"))

json.dump(ref, open(os.path.join(OUT, "isaac_reference.json"), "w"))
json.dump(report, open(os.path.join(OUT, "export_report.json"), "w"), indent=2)

# ------------------------------------------------------------- 6. the README
ev = report["evaluation"]
obs_rows = "\n".join(
    f"| {s['start']}-{s['end']} | {s['size']} | `{s['name']}` | {s['description']} |"
    if s["size"] > 1 else f"| {s['start']} | 1 | `{s['name']}` | {s['description']} |"
    for s in obs_layout
)
joint_rows = "\n".join(
    "| {index} | `{name}` | {default_rad:+.4f} | {lower_rad:+.4f} | {upper_rad:+.4f} |"
    " {stiffness_nm_per_rad:.0f} | {damping_nms_per_rad:.0f} | {effort_limit_nm:.0f} |"
    " {velocity_limit_rad_per_s:.0f} |".format(**j)
    for j in joints
)
body_rows = "\n".join(f"| `{b['name']}` | {b['mass_kg']:.3f} |" for b in bodies)
t = report["timing"]

readme = f"""# Biped walk-to-target — Unity Sentis export bundle

Trained in Isaac Lab with RSL-RL PPO ({os.path.basename(run_dir)}, checkpoint
`{os.path.basename(ckpt)}`). Measured over {ev['num_envs']} robots x {ev['seconds']:.0f} s:
**{ev['targets_per_biped_per_minute']:.1f} targets/robot/minute at {ev['mean_speed_m_s']:.2f} m/s
with {ev['falls']} falls**.

The rig is primitives only — it is meant to have a skinned mesh bound to it.

## Contents

| path | what |
|---|---|
| `biped.onnx` | the policy: single self-contained file, opset {opsets.get('ai.onnx')}, IR {model.ir_version}, ops {{{', '.join(ops)}}} |
| `export_report.json` | every number below, machine readable |
| `isaac_reference.json` | {a.ref_steps}-step recording of one robot, to validate your port |
| `robot/biped.urdf` | robot description (metres, kg, radians) |
| `robot/biped_usd/` | the same robot in Isaac Sim USD form |
| `checkpoint/` | raw RSL-RL checkpoint plus the exact env/agent configs used |
| `source/` | the Isaac Lab task code, including the URDF generator and this exporter |

## The ONNX policy

Input `{onnx_io['inputs'][0][0]}` float32 `{list(in_shape)}` -> output `{onnx_io['outputs'][0][0]}` float32 `{list(out_shape)}`.
Batch size is fixed at 1. Observation normalisation is **baked into the graph**, so feed raw
observations. The output is the Gaussian **mean** action — clamp it to `[-1, 1]` before use.
Verified against PyTorch on {a.ref_steps} real observations: max abs diff **{max_diff:.2e}**.

## Observation layout ({raw.cfg.observation_space} floats)

Everything is in **Isaac's Z-up right-handed** frame; convert Unity state into it before filling
the vector (see the conversion section).

| idx | n | name | content |
|---|---|---|---|
{obs_rows}

## Actions to joint targets

The {raw.cfg.action_space} outputs are **not** torques and **not** absolute angles. They are offsets
around the standing pose:

```
joint_target[i] = default_joint_pos[i] + {raw.cfg.action_scale} * clamp(action[i], -1, 1)   // [rad]
```

An implicit PD drive tracks that target every physics tick:

```
tau[i] = kp[i] * (joint_target[i] - q[i]) - kd[i] * qd[i]      // clamped to effort_limit[i]
```

| # | joint | default [rad] | lower | upper | kp [N·m/rad] | kd [N·m·s/rad] | effort [N·m] | vel [rad/s] |
|---|---|---|---|---|---|---|---|---|
{joint_rows}

Joint order in the observation and action vectors is exactly the table above (left leg first,
then right, proximal to distal).

## Bodies

| link | mass [kg] |
|---|---|
{body_rows}

Total {report['total_mass_kg']:.2f} kg. Standing hip height {raw.cfg.nominal_height:.3f} m — note the
torso link origin sits *on the hip line*, so the root z coordinate is the hip height.

## Physics settings to match

| setting | value |
|---|---|
| physics timestep | {t['physics_dt_s']:.6f} s ({t['physics_hz']:.0f} Hz) |
| policy timestep | {t['policy_dt_s']:.6f} s ({t['policy_hz']:.0f} Hz) |
| decimation | {t['decimation']} physics ticks per action |
| gravity | {report['gravity_m_s2']} m/s² (Isaac Z-up) |
| ground static friction | {gp.static_friction} |
| ground dynamic friction | {gp.dynamic_friction} |
| ground restitution | {gp.restitution} |
| spawn root position | {list(init.pos)} (Isaac) |
| episode length | {t['episode_length_s']} s |

## Isaac (Z-up) to Unity (Y-up)

Isaac Lab is **right-handed, Z-up, X-forward**. Unity is **left-handed, Y-up, Z-forward**.

```csharp
// positions and linear velocities
Vector3 ToUnity(Vector3 isaac) => new Vector3(-isaac.y, isaac.z, isaac.x);
Vector3 ToIsaac(Vector3 unity) => new Vector3( unity.z, -unity.x, unity.y);

// quaternions are stored WXYZ in isaac_reference.json; Unity uses XYZW
Quaternion QuatToUnity(float w, float x, float y, float z) => new Quaternion(y, -z, -x, w);

// angular velocity flips sign with handedness
Vector3 AngVelToUnity(Vector3 isaac) => new Vector3(isaac.y, -isaac.z, -isaac.x);
```

`projected_gravity_b` is the world gravity direction expressed in the torso frame, normalised —
`(0, 0, -1)` while standing upright. In Unity compute it as
`ToIsaac(Quaternion.Inverse(torsoRotation) * Vector3.down)`.

`target_dir_b` is the unit vector from the robot to the target in the torso's **yaw-only** frame
(roll and pitch removed): x forward, y to the robot's left. `target_dist` is the horizontal
distance in metres divided by 5 and clamped to 1.

Spawn pose: Isaac `{list(init.pos)}` = Unity `({-init.pos[1] + 0.0:.3f}, {init.pos[2]:.3f}, {init.pos[0] + 0.0:.3f})`.

Joint axes follow the URDF: hip_yaw about Isaac Z (Unity Y), hip_roll about Isaac X (Unity Z),
and hip_pitch / knee / ankle about Isaac Y (Unity -X). Positive pitch swings the segment
**backward**; the knee bends one way only, `[0, {joints[3]['upper_rad']:.2f}]` rad.

## Verifying the port

`isaac_reference.json` is the contract, and separates two failure modes:

1. **Inference path** — feed each recorded `obs` through the ONNX model in Unity and compare
   against the recorded `action`. Should match to ~1e-6. A mismatch is a Sentis or tensor-layout
   problem, not a physics problem.
2. **Physics path** — replay the recorded actions from the recorded spawn pose and watch
   `root_pos_w` / `joint_pos` drift over the {a.ref_steps} steps. Divergence here means gains,
   timestep, masses or friction differ.

## The task

A target is placed {raw.cfg.target_radius_range[0]}–{raw.cfg.target_radius_range[1]} m away at a random
bearing; reaching within {raw.cfg.reach_threshold} m scores it and immediately spawns a new one, so a
single episode chains many targets. The robot starts each episode facing a random direction, so the
policy has to turn as well as walk. An episode ends if the torso drops below
{raw.cfg.min_height} m or leans past a projected-gravity z of {raw.cfg.max_tilt}.

| metric | value |
|---|---|
| targets / robot / minute | {ev['targets_per_biped_per_minute']:.2f} |
| reached <0.5 s after assignment (lucky spawns) | {ev['lucky_spawn_fraction'] * 100:.1f}% |
| falls in {ev['seconds']:.0f} s x {ev['num_envs']} robots | {ev['falls']} |
| mean speed | {ev['mean_speed_m_s']:.2f} m/s |
"""
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)

print(f"[export] bundle written to {OUT}")
for root, _, files in os.walk(OUT):
    for f in sorted(files):
        fp = os.path.join(root, f)
        print(f"[export]   {os.path.relpath(fp, OUT):<52} {os.path.getsize(fp):>10,} B")
app.close()
