# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export the trained Unitree H1 locomotion policy as a self-contained Unity Sentis bundle.

Run (headless)::

    run_example.bat h1_export\\export_h1_unity.py --num_envs 64

Writes to ``unity_export/h1/``: ``h1_policy.onnx`` (opset 15 / IR 8, batch 1), ``isaac_reference.json``,
``export_report.json``, ``README.md``, plus ``checkpoint/``, ``robot/`` and ``source/`` subfolders.
"""

import argparse
import glob
import json
import os
import shutil
import subprocess

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Export the H1 locomotion policy for Unity Sentis.")
parser.add_argument("--task", type=str, default="Isaac-Velocity-Flat-H1-Play-v0")
parser.add_argument("--train_task", type=str, default="Isaac-Velocity-Flat-H1-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--ref_steps", type=int, default=250, help="Reference trajectory length [policy steps].")
parser.add_argument("--eval_steps", type=int, default=1000, help="Evaluation rollout length [policy steps].")
parser.add_argument("--ref_command", type=float, nargs=3, default=[1.0, 0.0, 0.0], help="Fixed (vx, vy, wz) command.")
parser.add_argument("--out", type=str, default="unity_export/h1")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata

import gymnasium as gym
import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import version_converter
from onnx.external_data_helper import convert_model_from_external_data
from rsl_rl.runners import OnPolicyRunner

import isaaclab_tasks  # noqa: F401  (registers the tasks)
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry

OUT = os.path.abspath(args_cli.out)
os.makedirs(OUT, exist_ok=True)
for sub in ("checkpoint/params", "robot", "source"):
    os.makedirs(os.path.join(OUT, sub), exist_ok=True)


def T(x):
    """Return the torch view of an Isaac Lab data buffer (Newton buffers expose ``.torch``)."""
    return x.torch if hasattr(x, "torch") else x


def jsonable(x):
    """Best-effort conversion of cfg values to JSON-serializable primitives."""
    if isinstance(x, (int, float, str, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "to_dict"):
        return jsonable(x.to_dict())
    return str(x)


def curl(url, dest):
    """Download ``url`` to ``dest`` over IPv4 (see README_SETUP.md for the IPv6/CloudFront issue)."""
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    try:
        r = subprocess.run(["curl", "-4", "-sfL", url, "-o", dest], capture_output=True, timeout=300)
        return r.returncode == 0 and os.path.exists(dest) and os.path.getsize(dest) > 0
    except Exception:
        return False


def unwrap_obs(o):
    """Return the raw policy observation tensor from whatever the wrapper handed back.

    ``RslRlVecEnvWrapper`` returns a ``TensorDict`` (not a ``dict`` subclass), so probe for keys.
    """
    if isinstance(o, tuple):
        o = o[0]
    if hasattr(o, "keys") and "policy" in list(o.keys()):
        o = o["policy"]
    return T(o)


# --------------------------------------------------------------------------------------------------
# 0. environment + policy
# --------------------------------------------------------------------------------------------------
env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
agent_cfg = handle_deprecated_rsl_rl_cfg(
    load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
)

env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
raw = env.unwrapped
robot = raw.scene["robot"]

if args_cli.checkpoint:
    ckpt = os.path.abspath(args_cli.checkpoint)
else:
    runs = sorted(glob.glob(f"logs/rsl_rl/{agent_cfg.experiment_name}/*"), key=os.path.getmtime)
    models = sorted(
        glob.glob(os.path.join(runs[-1], "model_*.pt")) if runs else [],
        key=lambda f: int(os.path.basename(f)[6:-3]),
    )
    ckpt = (
        models[-1]
        if models
        else os.path.abspath(os.path.join(".pretrained_checkpoints", "rsl_rl", args_cli.train_task, "checkpoint.pt"))
    )
if not os.path.exists(ckpt):
    raise FileNotFoundError(f"No checkpoint found at {ckpt}")
print(f"[export] checkpoint: {ckpt}")

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=raw.device)
runner.load(ckpt)
policy = runner.get_inference_policy(device=raw.device)

# --------------------------------------------------------------------------------------------------
# 1. ONNX export: single file, opset 15, IR 8, fixed batch 1
# --------------------------------------------------------------------------------------------------
tmp = os.path.join(OUT, "_tmp_onnx")
runner.export_policy_to_onnx(path=tmp, filename="policy.onnx")
model = onnx.load(os.path.join(tmp, "policy.onnx"), load_external_data=True)
convert_model_from_external_data(model)
model = version_converter.convert_version(model, 15)
model.ir_version = 8  # Unity Sentis reads IR <= 8
onnx_path = os.path.join(OUT, "h1_policy.onnx")
onnx.save_model(model, onnx_path, save_as_external_data=False)
onnx.checker.check_model(onnx_path)
shutil.rmtree(tmp, ignore_errors=True)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
onnx_io = {
    "inputs": [[i.name, i.shape, i.type] for i in sess.get_inputs()],
    "outputs": [[o.name, o.shape, o.type] for o in sess.get_outputs()],
}
op_types = sorted({n.op_type for n in model.graph.node})
onnx_meta = {
    "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in model.opset_import],
    "ir_version": model.ir_version,
    "operators": op_types,
    "initializers": len(model.graph.initializer),
    "size_bytes": os.path.getsize(onnx_path),
}
print(f"[export] onnx: {json.dumps(onnx_meta)} io={json.dumps(onnx_io)}")

# --------------------------------------------------------------------------------------------------
# 2. observation layout (index table) straight from the observation manager
# --------------------------------------------------------------------------------------------------
om = raw.observation_manager
terms = list(om.active_terms["policy"])
dims = [int(np.prod(d)) for d in om.group_obs_term_dim["policy"]]
obs_layout, start = [], 0
for name, dim in zip(terms, dims):
    obs_layout.append({"term": name, "start": start, "end": start + dim, "dim": dim})
    start += dim
obs_dim = start
joint_names = list(robot.joint_names)
act_dim = raw.action_manager.total_action_dim
print(f"[export] obs_dim={obs_dim} act_dim={act_dim} terms={obs_layout}")

# --------------------------------------------------------------------------------------------------
# 3. reference trajectory (env 0) under a FIXED command + ONNX-vs-PyTorch check on real observations
# --------------------------------------------------------------------------------------------------
cmd_term = raw.command_manager.get_term("base_velocity")
fixed_cmd = torch.tensor(args_cli.ref_command, device=raw.device, dtype=torch.float32)

obs = env.get_observations()
ref, max_diff = [], 0.0
in_name = sess.get_inputs()[0].name
with torch.inference_mode():
    for i in range(args_cli.ref_steps):
        cmd_term.vel_command_b[:] = fixed_cmd  # hold the command constant for a reproducible reference
        obs_t = unwrap_obs(obs)
        act = policy(obs[0] if isinstance(obs, tuple) else obs)
        o0 = obs_t[0].detach().cpu().numpy().astype(np.float32)
        onnx_act = sess.run(None, {in_name: o0[None]})[0][0]
        act_np = T(act)[0].detach().cpu().numpy()
        max_diff = max(max_diff, float(np.abs(onnx_act - act_np).max()))
        d = robot.data
        ref.append({
            "step": i,
            "t": i * raw.step_dt,
            "obs": o0.tolist(),
            "action": act_np.tolist(),
            "root_pos_w": (T(d.root_pos_w)[0] - raw.scene.env_origins[0]).cpu().tolist(),
            "root_quat_w_wxyz": T(d.root_quat_w)[0].cpu().tolist(),
            "root_lin_vel_b": T(d.root_lin_vel_b)[0].cpu().tolist(),
            "root_ang_vel_b": T(d.root_ang_vel_b)[0].cpu().tolist(),
            "joint_pos": T(d.joint_pos)[0].cpu().tolist(),
            "joint_vel": T(d.joint_vel)[0].cpu().tolist(),
            "target_velocity_command": fixed_cmd.cpu().tolist(),
        })
        obs = env.step(act)[0]
print(f"[export] onnx-vs-torch max|diff| over {args_cli.ref_steps} real observations: {max_diff:.3e}")

# --------------------------------------------------------------------------------------------------
# 4. evaluation under the task's own random commands
# --------------------------------------------------------------------------------------------------
env.reset()
obs = env.get_observations()
lin_err, ang_err, speed, n, falls = 0.0, 0.0, 0.0, 0, 0
with torch.inference_mode():
    for i in range(args_cli.eval_steps):
        cmd = T(cmd_term.command)
        d = robot.data
        lin_err += float((T(d.root_lin_vel_b)[:, :2] - cmd[:, :2]).norm(dim=-1).mean())
        ang_err += float((T(d.root_ang_vel_b)[:, 2] - cmd[:, 2]).abs().mean())
        speed += float(T(d.root_lin_vel_b)[:, :2].norm(dim=-1).mean())
        n += 1
        obs = env.step(policy(obs[0] if isinstance(obs, tuple) else obs))[0]
        falls += int(T(raw.termination_manager.terminated).sum())
sim_s = args_cli.eval_steps * raw.step_dt
evaluation = {
    "num_envs": args_cli.num_envs,
    "policy_steps": args_cli.eval_steps,
    "sim_seconds": sim_s,
    "mean_lin_vel_tracking_error_m_s": lin_err / n,
    "mean_ang_vel_tracking_error_rad_s": ang_err / n,
    "mean_speed_m_s": speed / n,
    "falls": falls,
    "falls_per_robot_per_minute": falls / args_cli.num_envs / sim_s * 60,
    "mean_seconds_between_falls_per_robot": (sim_s * args_cli.num_envs / falls) if falls else None,
}
print("[export] evaluation:", json.dumps(evaluation, indent=2))

# --------------------------------------------------------------------------------------------------
# 5. report: joints, bodies, actuators, timing, physics, spawn, task parameters
# --------------------------------------------------------------------------------------------------
d = robot.data
lim = T(d.joint_pos_limits)[0].cpu().numpy()
try:
    masses = T(d.default_mass)[0].cpu().numpy().tolist()
except Exception:
    masses = None
try:
    stiffness = T(d.joint_stiffness)[0].cpu().numpy().tolist()
    damping = T(d.joint_damping)[0].cpu().numpy().tolist()
except Exception:
    stiffness = damping = None
try:
    effort = T(d.joint_effort_limits)[0].cpu().numpy().tolist()
except Exception:
    effort = None
default_joint_pos = T(d.default_joint_pos)[0].cpu().numpy().tolist()
physics_dt = raw.physics_dt if hasattr(raw, "physics_dt") else raw.sim.get_physics_dt()
terrain_mat = env_cfg.scene.terrain.physics_material
action_scale = jsonable(getattr(env_cfg.actions.joint_pos, "scale", None))

report = {
    "task": args_cli.task,
    "train_task": args_cli.train_task,
    "checkpoint": ckpt,
    "onnx": dict({"file": os.path.basename(onnx_path), "io": onnx_io, "onnx_vs_torch_max_abs_diff": max_diff}, **onnx_meta),
    "observations": {"dim": obs_dim, "layout": obs_layout, "corruption_enabled": bool(om.cfg.policy.enable_corruption)},
    "actions": {
        "dim": act_dim,
        "type": type(raw.action_manager.get_term("joint_pos")).__name__,
        "scale": action_scale,
        "use_default_offset": jsonable(getattr(env_cfg.actions.joint_pos, "use_default_offset", None)),
        "offset_default_joint_pos_rad": default_joint_pos,
        "formula": "joint_position_target[i] = offset[i] + scale * action[i]",
        "clip": jsonable(agent_cfg.clip_actions),
    },
    "joints": {
        "order": joint_names,
        "count": len(joint_names),
        "pos_limits_rad": lim.tolist(),
        "default_pos_rad": default_joint_pos,
        "soft_limit_factor": jsonable(getattr(env_cfg.scene.robot, "soft_joint_pos_limit_factor", None)),
    },
    "actuators": {
        "resolved_per_joint": {"stiffness": stiffness, "damping": damping, "effort_limit": effort},
        "cfg": {k: jsonable(v.to_dict()) for k, v in env_cfg.scene.robot.actuators.items()},
        "drive_model": "implicit PD: tau = kp*(q_target - q) - kd*qd, clamped to effort_limit",
    },
    "bodies": {
        "names": list(robot.body_names),
        "masses_kg": masses,
        "total_mass_kg": float(sum(masses)) if masses else None,
    },
    "timing": {
        "physics_dt_s": physics_dt,
        "physics_hz": 1.0 / physics_dt,
        "decimation": env_cfg.decimation,
        "policy_dt_s": raw.step_dt,
        "policy_hz": 1.0 / raw.step_dt,
        "episode_length_s": env_cfg.episode_length_s,
    },
    "physics": {
        "gravity_m_s2": list(env_cfg.sim.gravity),
        "up_axis": "Z (Isaac)",
        "ground_material": {
            "static_friction": terrain_mat.static_friction,
            "dynamic_friction": terrain_mat.dynamic_friction,
            "restitution": terrain_mat.restitution,
            "friction_combine_mode": jsonable(getattr(terrain_mat, "friction_combine_mode", None)),
            "restitution_combine_mode": jsonable(getattr(terrain_mat, "restitution_combine_mode", None)),
        },
        "terrain_type": jsonable(env_cfg.scene.terrain.terrain_type),
    },
    "spawn": {
        "pos_isaac_xyz_m": list(env_cfg.scene.robot.init_state.pos),
        "rot_wxyz": list(env_cfg.scene.robot.init_state.rot),
        "joint_pos_rad": jsonable(env_cfg.scene.robot.init_state.joint_pos),
        "usd_path": jsonable(env_cfg.scene.robot.spawn.usd_path),
    },
    "task_parameters": {
        "command": jsonable(env_cfg.commands.base_velocity.to_dict()),
        "reference_command_used": args_cli.ref_command,
        "terminations": list(raw.termination_manager.active_terms),
        "rewards": list(raw.reward_manager.active_terms),
    },
    "network": {
        "actor_hidden_dims": jsonable(agent_cfg.actor.hidden_dims),
        "activation": jsonable(agent_cfg.actor.activation),
        "obs_normalization": jsonable(agent_cfg.actor.obs_normalization),
    },
    "evaluation": evaluation,
}

json.dump(ref, open(os.path.join(OUT, "isaac_reference.json"), "w"))
json.dump(report, open(os.path.join(OUT, "export_report.json"), "w"), indent=2)

# --------------------------------------------------------------------------------------------------
# 6. bundle: checkpoint + configs, robot description, task source
# --------------------------------------------------------------------------------------------------
shutil.copy2(ckpt, os.path.join(OUT, "checkpoint", "checkpoint.pt"))
dump_yaml(os.path.join(OUT, "checkpoint", "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(OUT, "checkpoint", "params", "agent.yaml"), agent_cfg)

usd_path = env_cfg.scene.robot.spawn.usd_path
robot_dir = os.path.join(OUT, "robot")
assets = {"usd_source": usd_path, "files": []}
try:
    from pxr import Usd, UsdUtils

    Usd.Stage.Open(usd_path)
    layers, assets_used, _ = UsdUtils.ComputeAllDependencies(usd_path)
    deps = [lyr.identifier for lyr in layers] + list(assets_used)
    base = usd_path.rsplit("/", 1)[0]
    for dep in deps:
        rel = dep[len(base) + 1:] if dep.startswith(base) else os.path.basename(dep)
        dest = os.path.join(robot_dir, "usd", rel.replace("/", os.sep))
        if dep.startswith("http"):
            ok = curl(dep, dest)
        elif os.path.exists(dep):
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            shutil.copy2(dep, dest)
            ok = True
        else:
            ok = False
        assets["files"].append({"src": dep, "rel": rel, "ok": bool(ok)})
except Exception as exc:  # noqa: BLE001
    assets["usd_error"] = str(exc)
    if curl(usd_path, os.path.join(robot_dir, "usd", os.path.basename(usd_path))):
        assets["files"].append({"src": usd_path, "rel": os.path.basename(usd_path), "ok": True})

URDF_URL = "https://raw.githubusercontent.com/unitreerobotics/unitree_ros/master/robots/h1_description/urdf/h1.urdf"
assets["urdf"] = {"src": URDF_URL, "ok": curl(URDF_URL, os.path.join(robot_dir, "h1.urdf"))}

src_root = "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity"
copies = {
    f"{src_root}/velocity_env_cfg.py": "velocity_env_cfg.py",
    f"{src_root}/config/h1/rough_env_cfg.py": "config_h1/rough_env_cfg.py",
    f"{src_root}/config/h1/flat_env_cfg.py": "config_h1/flat_env_cfg.py",
    f"{src_root}/config/h1/__init__.py": "config_h1/__init__.py",
    f"{src_root}/config/h1/agents/rsl_rl_ppo_cfg.py": "config_h1/rsl_rl_ppo_cfg.py",
    "source/isaaclab_assets/isaaclab_assets/robots/unitree.py": "assets/unitree.py",
}
for rel, dest_rel in copies.items():
    if os.path.exists(rel):
        dest = os.path.join(OUT, "source", dest_rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        shutil.copy2(rel, dest)
if os.path.exists(f"{src_root}/mdp"):
    shutil.copytree(f"{src_root}/mdp", os.path.join(OUT, "source", "mdp"), dirs_exist_ok=True)
shutil.copy2(os.path.abspath(__file__), os.path.join(OUT, "source", os.path.basename(__file__)))

report["bundled_assets"] = assets
json.dump(report, open(os.path.join(OUT, "export_report.json"), "w"), indent=2)
print("[export] bundled assets:", json.dumps(assets)[:1500])

# --------------------------------------------------------------------------------------------------
# 7. README
# --------------------------------------------------------------------------------------------------
DESCRIPTIONS = {
    "base_lin_vel": "base linear velocity in the base frame [m/s]",
    "base_ang_vel": "base angular velocity in the base frame [rad/s]",
    "projected_gravity": "gravity unit vector rotated into the base frame (Isaac Z-up)",
    "velocity_commands": "user command: vx [m/s], vy [m/s], wz [rad/s]",
    "joint_pos": "joint positions **relative to the default pose** [rad]",
    "joint_vel": "joint velocities relative to default [rad/s]",
    "actions": "previous action vector (raw policy output)",
    "height_scan": "terrain height samples under the robot [m]",
}
obs_rows = "\n".join(
    "| {} | {}-{} | {} | {} |".format(
        "`" + t["term"] + "`", t["start"], t["end"] - 1, t["dim"], DESCRIPTIONS.get(t["term"], "")
    )
    for t in obs_layout
)
joint_rows = "\n".join(
    "| {} | `{}` | {:+.4f} | {:+.4f} | {:+.4f} |".format(i, nm, default_joint_pos[i], lim[i][0], lim[i][1])
    for i, nm in enumerate(joint_names)
)
pd_rows = "\n".join(
    "| `{}` | `{}` | {} | {} | {} |".format(
        k,
        jsonable(v.to_dict().get("joint_names_expr")),
        jsonable(v.to_dict().get("stiffness")),
        jsonable(v.to_dict().get("damping")),
        jsonable(v.to_dict().get("effort_limit_sim") or v.to_dict().get("effort_limit")),
    )
    for k, v in env_cfg.scene.robot.actuators.items()
)
spawn = list(env_cfg.scene.robot.init_state.pos)
unity_spawn = tuple(v + 0.0 if v != 0 else 0.0 for v in (-spawn[1], spawn[2], spawn[0]))

readme = f"""# Unitree H1 locomotion policy - Unity Sentis bundle

Exported from Isaac Lab task `{args_cli.task}` (checkpoint trained on `{args_cli.train_task}`).
Everything here was generated by `source/{os.path.basename(__file__)}`; every number comes from the
live simulation, not from documentation.

## Contents

| Path | What it is |
|---|---|
| `h1_policy.onnx` | The actor network. opset {onnx_meta['opset'][0]['version']}, IR {onnx_meta['ir_version']}, single file, fixed batch 1. |
| `export_report.json` | Everything below, machine-readable. |
| `isaac_reference.json` | {args_cli.ref_steps}-step ground-truth trajectory of one robot (obs, action, root pose, joint state). |
| `checkpoint/` | Raw `checkpoint.pt` plus the resolved `env.yaml` / `agent.yaml`. |
| `robot/` | `h1.urdf` (vendor description) and `usd/` (the exact USD Isaac Lab simulated). |
| `source/` | Task source: env cfgs, MDP terms, robot asset cfg, and this exporter. |

## Policy interface

* **Input** `obs` - `float32[1, {obs_dim}]`
* **Output** `actions` - `float32[1, {act_dim}]`
* Deterministic (the Gaussian policy's mean). No observation normalization
  (`obs_normalization={jsonable(agent_cfg.actor.obs_normalization)}`), so feed raw observations.
* Network: MLP {jsonable(agent_cfg.actor.hidden_dims)}, `{jsonable(agent_cfg.actor.activation)}` activations.
* Operators used: {", ".join(op_types)} - all within Sentis' supported operator set.
* ONNX vs PyTorch over {args_cli.ref_steps} real observations: **max abs diff {max_diff:.2e}**.
* Run inference at **{1.0 / raw.step_dt:.1f} Hz** ({raw.step_dt * 1000:.1f} ms per step).

## Observation layout ({obs_dim} floats, concatenated in this order)

| Term | Index range | Size | Meaning |
|---|---|---|---|
{obs_rows}

All observations are in **Isaac's Z-up right-handed** frame. Convert Unity state into that frame
*before* filling the vector (see the conversion section below).

## Actions to joint targets

The policy emits {act_dim} numbers. They are **not** torques and **not** absolute angles:

```
joint_position_target[i] = default_joint_pos[i] + {action_scale} * action[i]      # [rad]
```

An implicit PD drive then tracks that target every physics tick:

```
tau[i] = kp[i] * (joint_position_target[i] - q[i]) - kd[i] * qd[i]     # clamped to effort_limit[i]
```

### Joint order, defaults and limits

| # | Joint | default [rad] | lower [rad] | upper [rad] |
|---|---|---|---|---|
{joint_rows}

### PD drive settings

| Actuator group | Joints | Stiffness kp | Damping kd | Effort limit [N.m] |
|---|---|---|---|---|
{pd_rows}

Physics runs at **{1.0 / physics_dt:.0f} Hz** with decimation **{env_cfg.decimation}**, i.e. the PD drive
updates {env_cfg.decimation}x per policy step. In Unity set `Time.fixedDeltaTime = {physics_dt:.6f}` and run
inference every {env_cfg.decimation} fixed steps. Unity's `ArticulationDrive` works in **degrees**: multiply
target radians by `Mathf.Rad2Deg`, and divide the kp/kd above by `Mathf.Rad2Deg` when porting the gains.

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

The `velocity_commands` you feed the policy must be in the Isaac convention: `vx` forward,
`vy` to the robot's left, `wz` counter-clockwise seen from above.

`projected_gravity` is the world gravity direction expressed in the base frame, normalized - while
standing upright it is `(0, 0, -1)`. In Unity compute it as
`ToIsaac(Quaternion.Inverse(baseRotation) * Vector3.down)`.

Spawn pose: Isaac `{spawn}` = Unity `({unity_spawn[0]:.3f}, {unity_spawn[1]:.3f}, {unity_spawn[2]:.3f})`.

## Verifying the Unity port

`isaac_reference.json` is the contract, and it isolates two failure modes separately:

1. **Inference path** - feed each recorded `obs` through the ONNX model in Unity and compare with the
   recorded `action`. Should match to ~1e-6. Any mismatch is a Sentis/tensor-layout problem.
2. **Physics path** - replay the recorded actions in your Unity scene from the recorded spawn pose and
   compare `root_pos_w` / `joint_pos` drift over the {args_cli.ref_steps} steps. Divergence here means
   gains, timestep, mass or friction differ.

The reference was recorded under a constant command `{args_cli.ref_command}` (vx, vy, wz).

## Measured policy performance

Isaac Lab, {args_cli.num_envs} robots, {sim_s:.0f} s of simulated time, task's own random commands
(vx {jsonable(env_cfg.commands.base_velocity.ranges.lin_vel_x)}, vy {jsonable(env_cfg.commands.base_velocity.ranges.lin_vel_y)},
wz {jsonable(env_cfg.commands.base_velocity.ranges.ang_vel_z)}, resampled every
{jsonable(env_cfg.commands.base_velocity.resampling_time_range)} s):

| Metric | Value |
|---|---|
| Mean linear-velocity tracking error | {evaluation['mean_lin_vel_tracking_error_m_s']:.3f} m/s |
| Mean angular-velocity tracking error | {evaluation['mean_ang_vel_tracking_error_rad_s']:.3f} rad/s |
| Mean speed | {evaluation['mean_speed_m_s']:.3f} m/s |
| Falls | {evaluation['falls']} ({evaluation['falls_per_robot_per_minute']:.3f} per robot per minute) |
"""
open(os.path.join(OUT, "README.md"), "w", encoding="utf-8").write(readme)
print(f"[export] wrote {OUT}")

env.close()
simulation_app.close()
