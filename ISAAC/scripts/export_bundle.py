#!/usr/bin/env python3
"""Export the trained Boy policy as a validated Unity bundle.

Run from the Isaac Lab checkout's Python::

    ISAAC\\isaaclab\\isaaclab.bat -p ISAAC\\scripts\\export_bundle.py [--checkpoint <model_NNNN.pt>] --num_envs 64

What it does, in order:

  1. Builds the PLAY task (no noise, no pushes) and loads the newest checkpoint.
  2. Exports the inference policy to ONNX: **opset 15, IR 8, fixed batch 1**, one file,
     with the empirical observation normaliser baked into the graph (the traced module is
     the runner's own inference policy, so whatever it normalises, the graph normalises).
  3. Verifies onnxruntime against PyTorch on real observations (gate 1e-4).
  4. Records a 250-step single-robot reference trajectory under a FIXED straight-ahead
     target -> ``isaac_reference.json`` (obs, actions, root state, joint state, target).
     The root quaternion order is verified against projected_gravity and stored as XYZW.
  5. Evaluates the policy under the task's own random targets -> speed, tracking error,
     falls, targets reached per minute.
  6. Rewrites ``isaacbox_rig.json``: joint order/index straight from the simulator, default
     pose, limits, gains and effort limits cross-checked, ``eval`` block filled in.
  7. Writes ``export_report.json`` and ``CONTRACT.md``.
  8. Copies the bundle into ``Assets/unity_export/IsaacBox/`` (ONNX overwritten IN PLACE so its
     .meta GUID survives) and the checkpoint + resolved yaml into ``ISAAC/boy_rig/out/checkpoint/``.
"""

import argparse
import glob
import json
import math
import os
import shutil
import sys

# Line-buffer before anything prints: this script ends at simulation_app.close(), which exits
# via os._exit() inside Isaac Sim and therefore skips Python's atexit flush. Redirected to a
# file or pipe, stdout is block-buffered and every export message - including the ONNX parity
# numbers and the evaluation results - is discarded, leaving a silent exit code 0.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# onnx / onnxruntime MUST be imported BEFORE the SimulationApp starts. Kit loads its own
# ONNX Runtime DLLs, and a pip onnxruntime imported afterwards cannot initialise on top of
# them - it fails with "DLL load failed while importing onnxruntime_pybind11_state" even
# though the identical import succeeds in the same interpreter standalone. Loading ours
# first makes it the resident copy. Same failure family as usd-core shadowing kit's pxr.
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

from isaaclab.app import AppLauncher

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RIG_OUT = os.path.join(REPO, "ISAAC", "boy_rig", "out")

parser = argparse.ArgumentParser(description="Export the Boy chase policy for Unity Inference Engine.")
parser.add_argument("--task", type=str, default="Isaac-Chase-Flat-Boy-Play-v0")
parser.add_argument("--train_task", type=str, default="Isaac-Chase-Flat-Boy-v0")
parser.add_argument("--checkpoint", type=str, default=None, help="Explicit checkpoint path (default: newest run).")
parser.add_argument("--log_root", type=str, default=os.path.join("logs", "rsl_rl"))
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--ref_steps", type=int, default=250)
parser.add_argument("--eval_steps", type=int, default=1500)
parser.add_argument("--ref_target_distance", type=float, default=8.0, help="Fixed target distance for the recording [m].")
parser.add_argument("--out", type=str, default=os.path.join(REPO, "Assets", "unity_export", "IsaacBox"))
parser.add_argument("--rig_json", type=str, default=os.path.join(RIG_OUT, "isaacbox_rig.json"))
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import boy_tasks  # noqa: E402,F401
from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

try:
    from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
except ImportError:  # pragma: no cover
    def handle_deprecated_rsl_rl_cfg(cfg, _version):
        return cfg

OUT = os.path.abspath(args_cli.out)
os.makedirs(OUT, exist_ok=True)
os.makedirs(os.path.join(RIG_OUT, "checkpoint", "params"), exist_ok=True)


def T(x):
    return x.torch if hasattr(x, "torch") else x


def jsonable(x):
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


def unwrap_obs(o):
    if isinstance(o, tuple):
        o = o[0]
    if hasattr(o, "keys") and "policy" in list(o.keys()):
        o = o["policy"]
    return T(o)


def quat_rotate_inverse_np(q, v, order):
    """Rotate v by the inverse of q (numpy). order: 'wxyz' or 'xyzw'."""
    if order == "wxyz":
        w, x, y, z = q
    else:
        x, y, z, w = q
    # q^-1 * v * q for a unit quaternion
    qv = np.array([x, y, z])
    t = 2.0 * np.cross(qv, v)
    return v - w * t + np.cross(qv, t)


# --------------------------------------------------------------------------------------------------
# 0. environment + policy
# --------------------------------------------------------------------------------------------------
with open(args_cli.rig_json, "r", encoding="utf-8") as f:
    rig = json.load(f)

env_cfg = load_cfg_from_registry(args_cli.task, "env_cfg_entry_point")
env_cfg.scene.num_envs = args_cli.num_envs
agent_cfg = handle_deprecated_rsl_rl_cfg(
    load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
)

env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=getattr(agent_cfg, "clip_actions", None))
raw = env.unwrapped
robot = raw.scene["robot"]

if args_cli.checkpoint:
    ckpt = os.path.abspath(args_cli.checkpoint)
else:
    runs = sorted(glob.glob(os.path.join(args_cli.log_root, agent_cfg.experiment_name, "*")), key=os.path.getmtime)
    models = sorted(
        glob.glob(os.path.join(runs[-1], "model_*.pt")) if runs else [],
        key=lambda p: int(os.path.basename(p)[6:-3]),
    )
    if not models:
        raise FileNotFoundError(f"no model_*.pt under {os.path.join(args_cli.log_root, agent_cfg.experiment_name)}")
    ckpt = models[-1]
print(f"[export] checkpoint: {ckpt}")

runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=raw.device)
runner.load(ckpt)
_raw_policy = runner.get_inference_policy(device=raw.device)


def policy(obs):
    """Flat-tensor facade over the rsl-rl inference policy.

    rsl-rl-lib 5.x feeds its models a DICT keyed by observation group - MLPModel.get_latent
    does ``obs[group] for group in self.obs_groups`` - while every call site in this script
    (the ONNX trace, the reference recording, the evaluation rollout) holds a flat
    (N, obsDim) tensor, which is also what Unity's BoyAgent will hand the ONNX. Older rsl-rl
    took the flat tensor directly; passing one to 5.x dies with "IndexError: too many indices
    for tensor of dimension 2". Adapt once here so the graph and every rollout stay flat.
    """
    return _raw_policy({"policy": obs})

# --------------------------------------------------------------------------------------------------
# 1. joint order from the SIMULATOR, cross-checked against the rig JSON
# --------------------------------------------------------------------------------------------------
joint_names = list(robot.joint_names)
body_names = list(robot.body_names)
rig_joints = {b["joint"]["name"]: b for b in rig["bodies"] if b.get("joint")}
missing = sorted(set(rig_joints) - set(joint_names))
extra = sorted(set(joint_names) - set(rig_joints))
if missing or extra:
    raise RuntimeError(f"joint set mismatch between isaacbox_rig.json and the sim: missing {missing}, extra {extra}")
if joint_names != rig["jointOrder"]:
    print(f"[export] NOTE: simulator joint order differs from the provisional rig order; rewriting isaacbox_rig.json.\n"
          f"         sim: {joint_names}")
for i, n in enumerate(joint_names):
    rig_joints[n]["joint"]["index"] = i
rig["jointOrder"] = joint_names
rig["bodyOrder"] = body_names if set(body_names) == {b["name"] for b in rig["bodies"]} else rig["bodyOrder"]

d = robot.data
default_joint_pos = T(d.default_joint_pos)[0].cpu().numpy().tolist()
lim = T(d.joint_pos_limits)[0].cpu().numpy()
stiffness = T(d.joint_stiffness)[0].cpu().numpy().tolist()
damping = T(d.joint_damping)[0].cpu().numpy().tolist()
try:
    effort = T(d.joint_effort_limits)[0].cpu().numpy().tolist()
except AttributeError:
    effort = T(d.joint_effort_limits_sim)[0].cpu().numpy().tolist() if hasattr(d, "joint_effort_limits_sim") else None
try:
    armature = T(d.joint_armature)[0].cpu().numpy().tolist()
except AttributeError:
    armature = None
try:
    masses = T(d.default_mass)[0].cpu().numpy().tolist()
except Exception:  # noqa: BLE001
    masses = None

deviations = []
for i, n in enumerate(joint_names):
    j = rig_joints[n]["joint"]
    checks = [("defaultPosRad", j["defaultPosRad"], default_joint_pos[i], 1e-4),
              ("lowerRad", j["lowerRad"], float(lim[i][0]), 1e-3),
              ("upperRad", j["upperRad"], float(lim[i][1]), 1e-3),
              ("stiffness", j["stiffness"], stiffness[i], 1e-3),
              ("damping", j["damping"], damping[i], 1e-3)]
    if effort is not None:
        checks.append(("effortLimit", j["effortLimit"], effort[i], 1e-3))
    for key, rig_v, sim_v, tol in checks:
        if abs(float(rig_v) - float(sim_v)) > tol:
            deviations.append({"joint": n, "field": key, "rig": rig_v, "sim": sim_v})
            j[key] = float(sim_v)  # the simulator wins; Unity must match what was trained
if deviations:
    print(f"[export] {len(deviations)} rig/sim deviations resolved in favour of the sim:")
    for dev in deviations:
        print(f"         {dev}")
if masses is not None:
    total = float(sum(masses))
    print(f"[export] simulated total mass {total:.3f} kg (rig {rig['totalMass']} kg)")

# --------------------------------------------------------------------------------------------------
# 2. ONNX: trace the runner's own inference policy (normaliser included), opset 15, IR 8, batch 1
# --------------------------------------------------------------------------------------------------
om = raw.observation_manager
terms = list(om.active_terms["policy"])
dims = [int(np.prod(dd)) for dd in om.group_obs_term_dim["policy"]]
obs_layout, start = [], 0
for name, dim in zip(terms, dims):
    obs_layout.append({"term": name, "start": start, "end": start + dim, "dim": dim})
    start += dim
obs_dim = start
act_dim = raw.action_manager.total_action_dim
assert obs_dim == rig["obsDim"] and act_dim == rig["actDim"], (obs_dim, act_dim, rig["obsDim"], rig["actDim"])


class _Exported(torch.nn.Module):
    """Traceable shell that keeps the ONNX input flat.

    It wraps the RAW rsl-rl policy, not the `policy` shim above: a plain Python closure hides
    the module from the tracer, which then treats its weights as grad-requiring constants and
    fails with "Cannot insert a Tensor that requires grad as a constant". Holding the module
    registers the parameters properly, so the dict adaptation is repeated here instead.
    """

    def __init__(self, fn, obs_group="policy"):
        super().__init__()
        self.fn = fn
        self._obs_group = obs_group

    def forward(self, obs):
        return self.fn({self._obs_group: obs})


onnx_path = os.path.join(OUT, "IsaacBox.onnx")
dummy = torch.zeros(1, obs_dim, device=raw.device)
for _p in getattr(_raw_policy, "parameters", list)():
    _p.requires_grad_(False)
with torch.inference_mode():
    torch.onnx.export(
        _Exported(_raw_policy).eval(), dummy, onnx_path,
        export_params=True, opset_version=15, do_constant_folding=True,
        input_names=["obs"], output_names=["actions"], dynamic_axes=None, dynamo=False,
    )
model = onnx.load(onnx_path)
model.ir_version = 8
onnx.save_model(model, onnx_path, save_as_external_data=False)
onnx.checker.check_model(onnx_path)

sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
in_name = sess.get_inputs()[0].name
onnx_io = {"inputs": [[i.name, i.shape, i.type] for i in sess.get_inputs()],
           "outputs": [[o.name, o.shape, o.type] for o in sess.get_outputs()]}
op_types = sorted({n.op_type for n in model.graph.node})
onnx_meta = {
    "opset": [{"domain": o.domain or "ai.onnx", "version": o.version} for o in model.opset_import],
    "ir_version": model.ir_version,
    "operators": op_types,
    "initializers": len(model.graph.initializer),
    "size_bytes": os.path.getsize(onnx_path),
    "normalizer_baked": any(op in op_types for op in ("Sub", "Div")) or len(model.graph.initializer) > 8,
}
print(f"[export] onnx: {json.dumps(onnx_meta)} io={json.dumps(onnx_io)}")

# --------------------------------------------------------------------------------------------------
# 3. reference trajectory: env 0 chasing a FIXED target straight ahead, ONNX vs torch on real obs
# --------------------------------------------------------------------------------------------------
cmd_term = raw.command_manager.get_term("target")
env.reset()
obs = env.get_observations()
origin0 = raw.scene.env_origins[0]
fixed_target = T(robot.data.root_pos_w)[0].clone()
fixed_target[0] += args_cli.ref_target_distance

ref, max_diff = [], 0.0
quat_votes = {"wxyz": 0.0, "xyzw": 0.0}
with torch.inference_mode():
    for i in range(args_cli.ref_steps):
        # pin env 0's target; keep every env's timer from expiring so nothing resamples
        cmd_term.target_pos_w[0] = fixed_target
        cmd_term.time_left[:] = 1e6
        cmd_term._update_command()
        obs = raw.observation_manager.compute()
        obs_t = unwrap_obs(obs)
        act = policy(obs_t)
        o0 = obs_t[0].detach().cpu().numpy().astype(np.float32)
        onnx_act = sess.run(None, {in_name: o0[None]})[0][0]
        act_np = T(act)[0].detach().cpu().numpy()
        max_diff = max(max_diff, float(np.abs(onnx_act - act_np).max()))

        dd = robot.data
        q = T(dd.root_quat_w)[0].cpu().numpy()
        g = np.array([0.0, 0.0, -1.0])
        for order in quat_votes:
            pg = quat_rotate_inverse_np(q, g, order)
            quat_votes[order] += float(np.linalg.norm(pg - o0[6:9]))
        ref.append({
            "step": i,
            "t": i * raw.step_dt,
            "obs": o0.tolist(),
            "action": act_np.tolist(),
            "root_pos_w": (T(dd.root_pos_w)[0] - origin0).cpu().tolist(),
            "root_quat_w_raw": q.tolist(),
            "root_lin_vel_b": T(dd.root_lin_vel_b)[0].cpu().tolist(),
            "root_ang_vel_b": T(dd.root_ang_vel_b)[0].cpu().tolist(),
            "joint_pos": T(dd.joint_pos)[0].cpu().tolist(),
            "joint_vel": T(dd.joint_vel)[0].cpu().tolist(),
            "target_pos_w": (fixed_target - origin0).cpu().tolist(),
            "target_pos_b": o0[9:12].tolist(),
        })
        obs = env.step(act)[0]

quat_order = min(quat_votes, key=quat_votes.get)
for s in ref:
    q = s.pop("root_quat_w_raw")
    s["root_quat_w_xyzw"] = q if quat_order == "xyzw" else [q[1], q[2], q[3], q[0]]
print(f"[export] onnx-vs-torch max|diff| over {args_cli.ref_steps} real observations: {max_diff:.3e}  "
      f"(gate 1e-4)  quaternion storage order detected: {quat_order} (residuals {quat_votes})")
if max_diff >= 1e-4:
    raise RuntimeError(f"ONNX disagrees with PyTorch: {max_diff:.3e} >= 1e-4")

ref_speed = float(np.mean([s["root_lin_vel_b"][0] for s in ref[50:]])) if len(ref) > 50 else 0.0
ref_travel = ref[-1]["root_pos_w"][0] - ref[0]["root_pos_w"][0]

# --------------------------------------------------------------------------------------------------
# 4. evaluation under the task's own random targets
# --------------------------------------------------------------------------------------------------
speed_sum, along_sum, along_err, n, falls, reached = 0.0, 0.0, 0.0, 0, 0, 0
target_speed = float(rig["chase"]["targetSpeed"])
# reset INSIDE inference_mode: the reference recording above already ran under it, so the
# env's sim buffers are inference tensors. Resetting outside writes to them in place and
# PhysX raises "Inplace update to inference tensor outside InferenceMode is not allowed".
with torch.inference_mode():
    env.reset()
    obs = env.get_observations()
    for i in range(args_cli.eval_steps):
        dd = robot.data
        delta = (cmd_term.target_pos_w - T(dd.root_pos_w))[:, :2]
        dist = torch.norm(delta, dim=-1).clamp_min(1e-6)
        direction = delta / dist.unsqueeze(-1)
        v = T(dd.root_lin_vel_w)[:, :2]
        along = torch.sum(v * direction, dim=-1)
        speed_sum += float(torch.norm(v, dim=-1).mean())
        along_sum += float(along.mean())
        along_err += float((along - target_speed).abs().mean())
        n += 1
        obs = env.step(policy(unwrap_obs(obs)))[0]
        falls += int(T(raw.termination_manager.terminated).sum())
        reached += int((cmd_term.reached & (cmd_term.time_left <= 0.0)).sum())
sim_s = args_cli.eval_steps * raw.step_dt
evaluation = {
    "num_envs": args_cli.num_envs,
    "policy_steps": args_cli.eval_steps,
    "sim_seconds": sim_s,
    "mean_speed_m_s": speed_sum / n,
    "mean_speed_toward_target_m_s": along_sum / n,
    "mean_target_speed_error_m_s": along_err / n,
    "falls": falls,
    "falls_per_robot_per_minute": falls / args_cli.num_envs / sim_s * 60,
    "targets_reached": reached,
    "targets_reached_per_robot_per_minute": reached / args_cli.num_envs / sim_s * 60,
    "reference_run": {"target_distance_m": args_cli.ref_target_distance, "steps": args_cli.ref_steps,
                      "mean_forward_speed_after_1s_m_s": ref_speed, "x_travelled_m": ref_travel},
}
print("[export] evaluation:", json.dumps(evaluation, indent=2))

# --------------------------------------------------------------------------------------------------
# 5. rig json + report + contract
# --------------------------------------------------------------------------------------------------
physics_dt = raw.physics_dt if hasattr(raw, "physics_dt") else raw.sim.get_physics_dt()
rig["eval"] = {
    "meanSpeed": evaluation["mean_speed_m_s"],
    "meanSpeedTowardTarget": evaluation["mean_speed_toward_target_m_s"],
    "meanTargetSpeedError": evaluation["mean_target_speed_error_m_s"],
    "fallsPerRobotPerMinute": evaluation["falls_per_robot_per_minute"],
    "targetsReachedPerMinute": evaluation["targets_reached_per_robot_per_minute"],
    "referenceForwardSpeed": ref_speed,
    "referenceTargetDistance": args_cli.ref_target_distance,
}
rig["sourceTask"] = args_cli.task
rig["trainTask"] = args_cli.train_task
rig["checkpoint"] = os.path.basename(ckpt)
rig["obsLayout"] = obs_layout
rig["network"] = {
    "actor_hidden_dims": jsonable(getattr(getattr(agent_cfg, "actor", None), "hidden_dims", None)
                                  or getattr(getattr(agent_cfg, "policy", None), "actor_hidden_dims", None)),
    "activation": jsonable(getattr(getattr(agent_cfg, "actor", None), "activation", None)
                           or getattr(getattr(agent_cfg, "policy", None), "activation", None)),
    "obs_normalization_baked_into_onnx": onnx_meta["normalizer_baked"],
}

report = {
    "task": args_cli.task,
    "train_task": args_cli.train_task,
    "checkpoint": ckpt,
    "onnx": dict({"file": "IsaacBox.onnx", "io": onnx_io, "onnx_vs_torch_max_abs_diff": max_diff}, **onnx_meta),
    "observations": {"dim": obs_dim, "layout": obs_layout,
                     "corruption_enabled": bool(om.cfg.policy.enable_corruption)},
    "actions": {
        "dim": act_dim,
        "type": type(raw.action_manager.get_term("joint_pos")).__name__,
        "scale": jsonable(env_cfg.actions.joint_pos.scale),
        "use_default_offset": jsonable(env_cfg.actions.joint_pos.use_default_offset),
        "offset_default_joint_pos_rad": default_joint_pos,
        "formula": "joint_position_target[i] = default[i] + scale * action[i]",
        "clip": jsonable(getattr(agent_cfg, "clip_actions", None)),
    },
    "joints": {"order": joint_names, "count": len(joint_names), "pos_limits_rad": lim.tolist(),
               "default_pos_rad": default_joint_pos,
               "soft_limit_factor": jsonable(env_cfg.scene.robot.soft_joint_pos_limit_factor)},
    "actuators": {
        "resolved_per_joint": {"stiffness": stiffness, "damping": damping, "effort_limit": effort, "armature": armature},
        "cfg": {k: jsonable(v.to_dict()) for k, v in env_cfg.scene.robot.actuators.items()},
        "drive_model": "implicit PD: tau = kp*(q_target - q) - kd*qd, clamped to effort_limit",
    },
    "bodies": {"names": body_names, "masses_kg": masses,
               "total_mass_kg": float(sum(masses)) if masses else None,
               "rig_json_masses_kg": {b["name"]: b["mass"] for b in rig["bodies"]},
               "rig_json_inertias": {b["name"]: b["inertiaDiag"] for b in rig["bodies"]}},
    "timing": {"physics_dt_s": physics_dt, "physics_hz": 1.0 / physics_dt, "decimation": env_cfg.decimation,
               "policy_dt_s": raw.step_dt, "policy_hz": 1.0 / raw.step_dt,
               "episode_length_s": env_cfg.episode_length_s},
    "physics": {
        "gravity_m_s2": list(env_cfg.sim.gravity), "up_axis": "Z (Isaac)",
        "ground_material": jsonable(env_cfg.scene.terrain.physics_material.to_dict()),
        "robot_material_play": jsonable(env_cfg.events.physics_material.params),
        "terrain_type": jsonable(env_cfg.scene.terrain.terrain_type),
        "solver": jsonable(env_cfg.scene.robot.spawn.articulation_props.to_dict()),
        "rigid_props": jsonable(env_cfg.scene.robot.spawn.rigid_props.to_dict()),
    },
    "spawn": {"pos_isaac_xyz_m": list(env_cfg.scene.robot.init_state.pos),
              "rot_wxyz": list(env_cfg.scene.robot.init_state.rot),
              "joint_pos_rad": jsonable(env_cfg.scene.robot.init_state.joint_pos),
              "usd_path": jsonable(env_cfg.scene.robot.spawn.usd_path)},
    "task_parameters": {
        "command": jsonable(env_cfg.commands.target.to_dict()),
        "reference_target_distance_m": args_cli.ref_target_distance,
        "quaternion_order_in_reference": "xyzw",
        "terminations": list(raw.termination_manager.active_terms),
        "rewards": list(raw.reward_manager.active_terms),
    },
    "network": rig["network"],
    "evaluation": evaluation,
    "rig_sim_deviations_resolved": deviations,
}

with open(os.path.join(OUT, "isaac_reference.json"), "w", encoding="utf-8") as f:
    json.dump({
        "_comment": "Env 0 of the Play task chasing a FIXED target straight ahead. Isaac frame (Z-up). "
                    "Written by export_bundle.py; obs/action pairs are the inference contract.",
        "_quaternionOrder": "xyzw",
        "_target": {"distance_m": args_cli.ref_target_distance, "fixed": True},
        "steps": ref,
    }, f)
with open(os.path.join(OUT, "export_report.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
for path in (os.path.join(OUT, "isaacbox_rig.json"), args_cli.rig_json):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rig, f, indent=1)

# --------------------------------------------------------------------------------------------------
# 6. CONTRACT.md
# --------------------------------------------------------------------------------------------------
DESCRIPTIONS = {
    "base_lin_vel": "root linear velocity in the base frame [m/s]",
    "base_ang_vel": "root angular velocity in the base frame [rad/s]",
    "projected_gravity": "gravity DIRECTION in the base frame, unit length; (0,0,-1) upright",
    "target_pos_b": "chase target minus root position, rotated into the base frame, norm-clipped to "
                    f"{rig['chase']['targetObsClip']} m",
    "joint_pos": "q - q_default [rad], simulator joint order",
    "joint_vel": "qd [rad/s]",
    "actions": "the previous RAW policy output",
}
obs_rows = "\n".join(f"| {t['start']}-{t['end'] - 1} | `{t['term']}` | {t['dim']} | {DESCRIPTIONS.get(t['term'], '')} |"
                     for t in obs_layout)
joint_rows = "\n".join(
    f"| {i} | `{n}` | {rig_joints[n]['joint']['axisInChild']} | {default_joint_pos[i]:+.3f} | {lim[i][0]:+.3f} | "
    f"{lim[i][1]:+.3f} | {stiffness[i]:.0f} | {damping[i]:.1f} | {(effort[i] if effort else float('nan')):.0f} |"
    for i, n in enumerate(joint_names))
ev = evaluation
contract = f"""# Boy - contract

The exact interface between the Isaac Lab policy and the Unity rig. Every number here was
read from the LIVE simulation by `ISAAC/scripts/export_bundle.py`, not from documentation.
Checkpoint `{os.path.basename(ckpt)}`, task `{args_cli.task}`.

## 1. Policy I/O

| | |
|---|---|
| file | `IsaacBox.onnx`, single file, {onnx_meta['size_bytes']:,} bytes |
| input | `obs` `float32[1, {obs_dim}]` |
| output | `actions` `float32[1, {act_dim}]` |
| opset / IR | ai.onnx {onnx_meta['opset'][0]['version']} / IR {onnx_meta['ir_version']} |
| operators | {', '.join(op_types)} |
| normaliser | **baked into the graph** ({'yes' if onnx_meta['normalizer_baked'] else 'NO - check'}): feed RAW observations |
| onnxruntime vs PyTorch | max abs diff {max_diff:.3e} over {args_cli.ref_steps} real observations (gate 1e-4) |
| runtime | Unity Inference Engine `com.unity.ai.inference`, `BackendType.CPU`, driven directly (NOT ML-Agents) |

## 2. Observation vector - {obs_dim} floats

| idx | term | size | meaning |
|---|---|---|---|
{obs_rows}

All in Isaac's frame. `target_pos_b` is the ONLY task input: Unity computes
`PosToIsaac(inv(rootRot) * (target - rootPos))`, then scales the vector down to length
{rig['chase']['targetObsClip']} m if it is longer. In training the target was resampled on a ring of radius
{rig['chase']['targetRadiusRange']} m whenever the robot came within {rig['chase']['reachRadius']} m of it or after
{rig['chase']['resampleRangeS']} s. No velocity command exists; the policy's pace is whatever the reward
taught it (target speed {rig['chase']['targetSpeed']} m/s).

## 3. Actions -> joint targets

```
joint_position_target[i] = default_joint_pos[i] + {jsonable(env_cfg.actions.joint_pos.scale)} * action[i]   # rad
tau[i] = kp[i] * (joint_position_target[i] - q[i]) - kd[i] * qd[i]                          # clamped to effort[i]
```

Physics {1.0 / physics_dt:.0f} Hz, decimation {env_cfg.decimation}, policy {1.0 / raw.step_dt:.0f} Hz. Unity:
`Time.fixedDeltaTime = {physics_dt}`, inference every {env_cfg.decimation} fixed steps. Unity's `ArticulationDrive`
target and limits are in DEGREES, `jointPosition`/`jointVelocity` in radians; gains are radian-based
(measured by the rung-2b test).

| # | joint | axis (child frame) | default [rad] | lower | upper | kp | kd | effort |
|---|---|---|---|---|---|---|---|---|
{joint_rows}

## 4. Frames

Isaac: right-handed, Z-up, X-forward, Y-left. Unity: left-handed, Y-up, Z-forward.

```
M : (x, y, z)_isaac -> (-y, z, x)_unity           true vectors (position, velocity, gravity)
-M: (x, y, z)_isaac -> ( y,-z,-x)_unity           pseudovectors (angular velocity, rotation axes)
quaternion (x, y, z, w)_isaac -> (y, -z, -x, w)_unity
```

Every revolute anchor's local +X is built at `-M * axis`, so a positive Unity joint angle IS a
positive Isaac joint angle and no sign is flipped anywhere.

## 5. Zero pose vs default pose

The articulation's zero pose is the authored T-POSE (all link frames world-aligned). The
default pose is the joint-angle offset in the table above (arms hang, knees bent). Unity
attaches the skinned-mesh bones in the T-pose and lets the drives take the rig to the default
pose, so the mesh follows for free.

## 6. Reference recording (`isaac_reference.json`)

Env 0, target fixed {args_cli.ref_target_distance} m straight ahead, {args_cli.ref_steps} policy steps. Quaternions stored XYZW
(verified against projected_gravity: residual {quat_votes[quat_order]:.4f} vs {max(quat_votes.values()):.4f} for the other order).
Mean forward speed after the first second: **{ref_speed:.3f} m/s**, {ref_travel:.2f} m travelled.

## 7. Isaac evaluation ({ev['num_envs']} robots, {ev['sim_seconds']:.0f} s, random targets)

| metric | value |
|---|---|
| mean speed | {ev['mean_speed_m_s']:.3f} m/s |
| mean speed toward target | {ev['mean_speed_toward_target_m_s']:.3f} m/s |
| mean \\|v_along - {target_speed}\\| | {ev['mean_target_speed_error_m_s']:.3f} m/s |
| falls | {ev['falls']} ({ev['falls_per_robot_per_minute']:.3f} per robot per minute) |
| targets reached | {ev['targets_reached']} ({ev['targets_reached_per_robot_per_minute']:.2f} per robot per minute) |
"""
with open(os.path.join(OUT, "CONTRACT.md"), "w", encoding="utf-8") as f:
    f.write(contract)

# --------------------------------------------------------------------------------------------------
# 7. checkpoint + params beside the rig (NOT under Assets/)
# --------------------------------------------------------------------------------------------------
shutil.copy2(ckpt, os.path.join(RIG_OUT, "checkpoint", "checkpoint.pt"))
dump_yaml(os.path.join(RIG_OUT, "checkpoint", "params", "env.yaml"), env_cfg)
dump_yaml(os.path.join(RIG_OUT, "checkpoint", "params", "agent.yaml"), agent_cfg)
shutil.copy2(onnx_path, os.path.join(RIG_OUT, "IsaacBox.onnx"))
shutil.copy2(os.path.join(OUT, "export_report.json"), os.path.join(RIG_OUT, "export_report.json"))

print(f"[export] wrote {OUT}\n         next: Unity > Boy > Rebuild Rig Asset From JSON, Boy > Build Prefab, "
      f"then run the IsaacBox play-mode tests (rung 0 checks IsaacBox.onnx against isaac_reference.json).")

env.close()
simulation_app.close()
