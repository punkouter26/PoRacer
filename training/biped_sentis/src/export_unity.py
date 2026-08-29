"""Headless export of the trained biped policy for Unity Sentis.

Produces a self-contained bundle: an ONNX policy (opset 15, IR 8, batch 1, no
dynamic axes), a reference trajectory, a full physics/robot specification, an
evaluation report, the MJCF and checkpoint, and a Unity integration README.

    python export_unity.py
    python export_unity.py --out export/biped_sentis --traj-steps 200

The exported graph is self-contained: observation normalisation (the
VecNormalize statistics) and the action clamp are baked in, so Unity feeds raw
observations and receives final actions. Nothing outside the .onnx is needed at
inference time.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import platform
import shutil
import sys
from datetime import datetime, timezone

import mujoco
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from stable_baselines3 import PPO

from biped import BipedTargetEnv
from biped.env import XML_PATH

# Operators Unity Sentis / Inference Engine implements. Anything outside this
# set is flagged rather than silently shipped.
SENTIS_SUPPORTED_OPS = {
    "Abs", "Add", "And", "ArgMax", "ArgMin", "AveragePool", "BatchNormalization",
    "Cast", "Ceil", "Clip", "Concat", "Constant", "ConstantOfShape", "Conv",
    "ConvTranspose", "Cos", "Div", "Dropout", "Elu", "Equal", "Erf", "Exp",
    "Expand", "Flatten", "Floor", "Gather", "GatherElements", "Gemm",
    "GlobalAveragePool", "Greater", "GreaterOrEqual", "HardSigmoid", "Identity",
    "InstanceNormalization", "LayerNormalization", "LeakyRelu", "Less",
    "LessOrEqual", "Log", "LogSoftmax", "MatMul", "Max", "MaxPool", "Mean",
    "Min", "Mul", "Neg", "Not", "Or", "Pad", "Pow", "PRelu", "RandomNormal",
    "Range", "Reciprocal", "ReduceMax", "ReduceMean", "ReduceMin", "ReduceProd",
    "ReduceSum", "Relu", "Reshape", "Resize", "Round", "ScatterElements",
    "Selu", "Shape", "Sigmoid", "Sign", "Sin", "Slice", "Softmax", "Softplus",
    "Split", "Sqrt", "Squeeze", "Sub", "Tan", "Tanh", "Tile", "Transpose",
    "Unsqueeze", "Where",
}

TARGET_OPSET = 15
TARGET_IR_VERSION = 8
NUM_ACTIONS = 12
OBS_DIM = 49

# Observation layout, mirroring BipedTargetEnv._get_obs exactly.
OBS_LAYOUT = [
    (1, "torso_height", "qpos[2], torso height above ground", "metres"),
    (3, "projected_gravity", "gravity unit vector rotated into the torso frame; encodes roll/pitch", "unit vector"),
    (3, "linear_velocity", "torso linear velocity in the torso frame", "m/s, clipped +/-10"),
    (3, "angular_velocity", "torso angular velocity in the torso frame", "rad/s, clipped +/-10"),
    (12, "joint_positions", "qpos[7:], hinge angles in JOINT_ORDER", "radians"),
    (12, "joint_velocities", "qvel[6:], hinge rates in JOINT_ORDER", "rad/s, clipped +/-20"),
    (2, "target_direction", "unit vector to the goal, rotated into the torso yaw frame (x, y)", "unit vector"),
    (1, "target_distance", "planar distance to the goal", "metres, min(d, 10)"),
    (12, "last_action", "action emitted on the previous control step", "[-1, 1]"),
]


# --------------------------------------------------------------------------
# 1. Policy wrapper -- makes the ONNX graph self-contained
# --------------------------------------------------------------------------


class SentisPolicy(nn.Module):
    """Raw observation in, final action out.

    Folds three things the Python side normally did outside the network:
      * VecNormalize observation whitening, as a precomputed multiply so the
        graph needs Sub+Mul rather than Sub+Div+Sqrt,
      * the observation clip at +/-clip_obs,
      * the action clamp to the action-space bounds.
    """

    def __init__(self, policy, obs_mean, obs_var, clip_obs, epsilon, act_low, act_high):
        super().__init__()
        scale = 1.0 / np.sqrt(obs_var + epsilon)
        self.register_buffer("obs_mean", torch.tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("obs_scale", torch.tensor(scale, dtype=torch.float32))
        self.clip_obs = float(clip_obs)
        self.act_low = float(np.min(act_low))
        self.act_high = float(np.max(act_high))
        self.policy_net = copy.deepcopy(policy.mlp_extractor.policy_net)
        self.action_net = copy.deepcopy(policy.action_net)
        self.eval()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = (obs - self.obs_mean) * self.obs_scale
        x = torch.clamp(x, -self.clip_obs, self.clip_obs)
        x = self.policy_net(x)
        x = self.action_net(x)
        return torch.clamp(x, self.act_low, self.act_high)


def load_policy(model_path: str, stats_path: str):
    model = PPO.load(model_path, device="cpu")
    with open(stats_path, "rb") as fh:
        vec_normalize = pickle.load(fh)
    if not vec_normalize.norm_obs:
        raise SystemExit("vecnormalize.pkl has norm_obs=False; nothing to bake in")
    wrapper = SentisPolicy(
        model.policy,
        vec_normalize.obs_rms.mean.astype(np.float64),
        vec_normalize.obs_rms.var.astype(np.float64),
        vec_normalize.clip_obs,
        vec_normalize.epsilon,
        model.action_space.low,
        model.action_space.high,
    )
    return model, vec_normalize, wrapper


def act(wrapper: SentisPolicy, obs: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        t = torch.tensor(obs, dtype=torch.float32).reshape(1, -1)
        return wrapper(t).numpy().reshape(-1)


# --------------------------------------------------------------------------
# 2. ONNX export + verification
# --------------------------------------------------------------------------


def export_onnx(wrapper: SentisPolicy, path: str) -> None:
    dummy = torch.zeros(1, OBS_DIM, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        dummy,
        path,
        input_names=["obs"],
        output_names=["action"],
        opset_version=TARGET_OPSET,
        do_constant_folding=True,
        dynamo=False,          # TorchScript exporter: flatter graph, Sentis-friendly
        export_params=True,    # weights embedded -> single self-contained file
        # deliberately no dynamic_axes: batch is fixed at 1
    )
    model = onnx.load(path)
    model.ir_version = TARGET_IR_VERSION
    onnx.save(model, path)


def collect_observations(env: BipedTargetEnv, wrapper: SentisPolicy, count: int, seed: int):
    """Real on-policy observations -- the distribution the net will actually see."""
    obs_list = []
    obs, _ = env.reset(seed=seed)
    for _ in range(count):
        obs_list.append(np.asarray(obs, dtype=np.float64).copy())
        obs, _, terminated, truncated, _ = env.step(act(wrapper, obs))
        if terminated or truncated:
            obs, _ = env.reset()
    return np.array(obs_list)


def verify_onnx(path: str, wrapper: SentisPolicy, observations: np.ndarray, tol: float):
    model = onnx.load(path)
    onnx.checker.check_model(model, full_check=True)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    in_name = session.get_inputs()[0].name

    max_diff = 0.0
    for obs in observations:
        x = obs.astype(np.float32).reshape(1, OBS_DIM)
        onnx_out = session.run(None, {in_name: x})[0].reshape(-1)
        with torch.no_grad():
            torch_out = wrapper(torch.tensor(x)).numpy().reshape(-1)
        max_diff = max(max_diff, float(np.abs(onnx_out - torch_out).max()))

    graph_in = model.graph.input[0]
    graph_out = model.graph.output[0]
    shape_of = lambda vi: [d.dim_value if d.HasField("dim_value") else d.dim_param
                           for d in vi.type.tensor_type.shape.dim]
    in_shape, out_shape = shape_of(graph_in), shape_of(graph_out)

    ops = sorted({n.op_type for n in model.graph.node})
    unsupported = sorted(set(ops) - SENTIS_SUPPORTED_OPS)
    opset = {i.domain or "ai.onnx": i.version for i in model.opset_import}
    dynamic = [d for d in in_shape + out_shape if not isinstance(d, int)]

    return {
        "checker_passed": True,
        "max_abs_diff_vs_pytorch": max_diff,
        "tolerance": tol,
        "numerically_equivalent": bool(max_diff < tol),
        "samples_compared": int(len(observations)),
        "opset": opset,
        "ir_version": int(model.ir_version),
        "input": {"name": graph_in.name, "shape": in_shape, "dtype": "float32"},
        "output": {"name": graph_out.name, "shape": out_shape, "dtype": "float32"},
        "operators": ops,
        "unsupported_by_sentis": unsupported,
        "has_dynamic_axes": bool(dynamic),
        "initializers": len(model.graph.initializer),
        "file_bytes": os.path.getsize(path),
    }


# --------------------------------------------------------------------------
# 3. Reference trajectory
# --------------------------------------------------------------------------


def record_trajectory(env: BipedTargetEnv, wrapper: SentisPolicy, steps: int, seed: int):
    data = env.unwrapped.data
    obs, _ = env.reset(seed=seed)
    frames = []
    for i in range(steps):
        action = act(wrapper, obs)
        frames.append(
            {
                "step": i,
                "time": round(float(i * env.dt), 6),
                "observation": np.asarray(obs, dtype=float).round(8).tolist(),
                "action": action.astype(float).round(8).tolist(),
                "root_pose": {
                    "position": data.qpos[0:3].round(8).tolist(),
                    "quaternion_wxyz": data.qpos[3:7].round(8).tolist(),
                },
                "root_velocity": {
                    "linear": data.qvel[0:3].round(8).tolist(),
                    "angular": data.qvel[3:6].round(8).tolist(),
                },
                "joint_positions": data.qpos[7:].round(8).tolist(),
                "joint_velocities": data.qvel[6:].round(8).tolist(),
                "target": {
                    "position": data.mocap_pos[0].round(8).tolist(),
                    "distance": None,  # filled from the info dict below
                },
            }
        )
        obs, reward, terminated, truncated, info = env.step(action)
        frames[-1]["reward"] = round(float(reward), 8)
        frames[-1]["terminated"] = bool(terminated)
        frames[-1]["target"]["distance"] = round(float(info["distance_to_target"]), 8)
        frames[-1]["targets_reached"] = int(info["targets_reached"])
        if terminated or truncated:
            obs, _ = env.reset()
    return frames


# --------------------------------------------------------------------------
# 4. Physics / robot specification
# --------------------------------------------------------------------------

JOINT_TYPE_NAMES = {
    int(mujoco.mjtJoint.mjJNT_FREE): "free",
    int(mujoco.mjtJoint.mjJNT_BALL): "ball",
    int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
    int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
}


def name_of(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, idx) or f"<{idx}>"


def robot_spec(model, env: BipedTargetEnv, frame_skip: int):
    joints = []
    for jid in range(model.njnt):
        jtype = int(model.jnt_type[jid])
        joints.append(
            {
                "index": jid,
                "name": name_of(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
                "type": JOINT_TYPE_NAMES.get(jtype, str(jtype)),
                "body": name_of(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[jid])),
                "axis": model.jnt_axis[jid].round(6).tolist(),
                "qpos_index": int(model.jnt_qposadr[jid]),
                "dof_index": int(model.jnt_dofadr[jid]),
                "limited": bool(model.jnt_limited[jid]),
                "range_rad": model.jnt_range[jid].round(6).tolist() if model.jnt_limited[jid] else None,
                "range_deg": np.degrees(model.jnt_range[jid]).round(3).tolist() if model.jnt_limited[jid] else None,
                "damping": float(model.dof_damping[int(model.jnt_dofadr[jid])]),
                "armature": float(model.dof_armature[int(model.jnt_dofadr[jid])]),
                "stiffness": float(model.jnt_stiffness[jid]),
            }
        )

    bodies = []
    for bid in range(model.nbody):
        parent = int(model.body_parentid[bid])
        bodies.append(
            {
                "index": bid,
                "name": name_of(model, mujoco.mjtObj.mjOBJ_BODY, bid),
                "parent": name_of(model, mujoco.mjtObj.mjOBJ_BODY, parent) if bid else None,
                "mass": float(model.body_mass[bid]),
                "subtree_mass": float(model.body_subtreemass[bid]),
                "local_pos": model.body_pos[bid].round(6).tolist(),
                "local_quat_wxyz": model.body_quat[bid].round(6).tolist(),
                "inertia_diag": model.body_inertia[bid].round(8).tolist(),
                "com_local": model.body_ipos[bid].round(6).tolist(),
            }
        )

    actuators = []
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        actuators.append(
            {
                "index": aid,
                "name": name_of(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid),
                "joint": name_of(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
                "type": "motor (direct torque)",
                "gear": float(model.actuator_gear[aid, 0]),
                "ctrlrange": model.actuator_ctrlrange[aid].round(6).tolist(),
                "ctrl_limited": bool(model.actuator_ctrllimited[aid]),
                "gainprm": model.actuator_gainprm[aid][:3].round(6).tolist(),
                "biasprm": model.actuator_biasprm[aid][:3].round(6).tolist(),
                "peak_torque_Nm": float(model.actuator_gear[aid, 0] * model.actuator_ctrlrange[aid, 1]),
            }
        )

    geoms = []
    for gid in range(model.ngeom):
        geoms.append(
            {
                "name": name_of(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
                "body": name_of(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])),
                "group": int(model.geom_group[gid]),
                "friction": model.geom_friction[gid].round(6).tolist(),
                "condim": int(model.geom_condim[gid]),
                "size": model.geom_size[gid].round(6).tolist(),
                "contype": int(model.geom_contype[gid]),
                "conaffinity": int(model.geom_conaffinity[gid]),
            }
        )

    u = env.unwrapped
    return {
        "joints": joints,
        "bodies": bodies,
        "actuators": actuators,
        "geoms": geoms,
        "simulation": {
            "physics_timestep": float(model.opt.timestep),
            "frame_skip": int(frame_skip),
            "control_timestep": float(model.opt.timestep * frame_skip),
            "control_frequency_hz": float(1.0 / (model.opt.timestep * frame_skip)),
            "physics_frequency_hz": float(1.0 / model.opt.timestep),
            "gravity": model.opt.gravity.round(6).tolist(),
            "integrator": int(model.opt.integrator),
            "integrator_name": str(mujoco.mjtIntegrator(model.opt.integrator).name),
            "solver": str(mujoco.mjtSolver(model.opt.solver).name),
            "iterations": int(model.opt.iterations),
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        },
        "reset_state": {
            "init_qpos": u.init_qpos.round(8).tolist(),
            "init_qvel": u.init_qvel.round(8).tolist(),
            "reset_noise_scale": float(u._reset_noise_scale),
            "note": "root yaw is randomised uniformly in [-pi, pi] on reset",
        },
        "task": {
            "forward_reward_weight": float(u._forward_reward_weight),
            "heading_reward_weight": float(u._heading_reward_weight),
            "healthy_reward": float(u._healthy_reward),
            "fall_penalty": float(u._fall_penalty),
            "ctrl_cost_weight": float(u._ctrl_cost_weight),
            "reach_bonus": float(u._reach_bonus),
            "reach_radius_m": float(u._reach_radius),
            "healthy_z_range": [float(v) for v in u._healthy_z_range],
            "min_uprightness": float(u._min_uprightness),
            "target_distance_range_m": [float(v) for v in u._target_distance_range],
            "target_angle_range_rad": float(u._target_angle_range),
            "max_speed_reward": float(u._max_speed_reward),
            "max_episode_steps": 1000,
        },
    }


# --------------------------------------------------------------------------
# 5. Evaluation
# --------------------------------------------------------------------------


def evaluate(env: BipedTargetEnv, wrapper: SentisPolicy, episodes: int, seed: int, max_steps: int):
    returns, lengths, reached, final_dist, speeds = [], [], [], [], []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        total, steps, info, ep_speed = 0.0, 0, {}, []
        while True:
            obs, reward, terminated, truncated, info = env.step(act(wrapper, obs))
            total += float(reward)
            steps += 1
            ep_speed.append(float(info["closing_speed"]))
            if terminated or steps >= max_steps:
                break
        returns.append(total)
        lengths.append(steps)
        reached.append(int(info["targets_reached"]))
        final_dist.append(float(info["distance_to_target"]))
        speeds.append(float(np.mean(ep_speed)))

    f = lambda a: {"mean": float(np.mean(a)), "std": float(np.std(a)),
                   "min": float(np.min(a)), "max": float(np.max(a))}
    return {
        "episodes": episodes,
        "deterministic": True,
        "return": f(returns),
        "episode_length_steps": f(lengths),
        "episode_length_seconds": f(np.array(lengths) / 40.0),
        "targets_reached_per_episode": f(reached),
        "final_distance_to_target_m": f(final_dist),
        "mean_closing_speed_mps": f(speeds),
        "survived_full_episode_pct": float(100.0 * np.mean([l >= max_steps for l in lengths])),
        "per_episode": [
            {"episode": i, "return": round(returns[i], 3), "steps": lengths[i],
             "targets_reached": reached[i], "final_distance_m": round(final_dist[i], 3)}
            for i in range(episodes)
        ],
    }


# --------------------------------------------------------------------------
# 6. Coordinate conversion, verified rather than asserted
# --------------------------------------------------------------------------


def quat_to_mat(w, x, y, z):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


# Position map: MuJoCo (X fwd, Y left, Z up) -> Unity (X right, Y up, Z fwd).
P_MATRIX = np.array([[0.0, -1.0, 0.0],
                     [0.0, 0.0, 1.0],
                     [1.0, 0.0, 0.0]])

QUAT_CANDIDATES = {
    # label: (mujoco wxyz) -> (unity xyzw)
    "requested (-x,  z, -y,  w)": lambda w, x, y, z: np.array([-x, z, -y, w]),
    "verified  ( y, -z, -x,  w)": lambda w, x, y, z: np.array([y, -z, -x, w]),
    "sign-fix  (-x, -z, -y,  w)": lambda w, x, y, z: np.array([-x, -z, -y, w]),
    "w-negated (-x,  z, -y, -w)": lambda w, x, y, z: np.array([-x, z, -y, -w]),
}


def _describe_position_map(P):
    lbl = ["x", "y", "z"]
    parts = []
    for r in range(3):
        c = int(np.argmax(np.abs(P[r])))
        parts.append(("-" if P[r, c] < 0 else "+") + "mujoco." + lbl[c])
    return "unity = (%s, %s, %s)" % tuple(parts)


def compatible_position_maps(fn, trials: int = 200, seed: int = 0):
    """Every signed axis permutation that flips handedness and satisfies
    R_unity == P R_mujoco P^-1 for this quaternion rule. An empty list means the
    rule is unusable with any axis convention, not merely mismatched with ours."""
    import itertools

    rng = np.random.default_rng(seed)
    qs = rng.normal(size=(trials, 4))
    qs /= np.linalg.norm(qs, axis=1, keepdims=True)

    hits = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product([1, -1], repeat=3):
            P = np.zeros((3, 3))
            for r, (c, sgn) in enumerate(zip(perm, signs)):
                P[r, c] = sgn
            if abs(np.linalg.det(P) + 1.0) > 1e-9:   # keep only handedness flips
                continue
            Pinv = np.linalg.inv(P)
            worst = 0.0
            for q in qs:
                w, x, y, z = q
                ux, uy, uz, uw = fn(w, x, y, z)
                worst = max(worst, float(np.abs(
                    quat_to_mat(uw, ux, uy, uz) - P @ quat_to_mat(w, x, y, z) @ Pinv).max()))
                if worst > 1e-9:
                    break
            if worst < 1e-9:
                hits.append(_describe_position_map(P))
    return hits


def check_quaternion_conversions(trials: int = 512, seed: int = 0):
    """A conversion is correct iff R_unity == P @ R_mujoco @ P^-1."""
    rng = np.random.default_rng(seed)
    results = {}
    P, Pinv = P_MATRIX, np.linalg.inv(P_MATRIX)
    for label, fn in QUAT_CANDIDATES.items():
        worst = 0.0
        for _ in range(trials):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            w, x, y, z = q
            R_m = quat_to_mat(w, x, y, z)
            ux, uy, uz, uw = fn(w, x, y, z)
            R_u = quat_to_mat(uw, ux, uy, uz)
            worst = max(worst, float(np.abs(R_u - P @ R_m @ Pinv).max()))
        results[label] = {
            "max_error": worst,
            "consistent_with_our_position_map": bool(worst < 1e-9),
            "compatible_position_maps": compatible_position_maps(fn),
        }
    return results


# --------------------------------------------------------------------------
# 7. Bundle
# --------------------------------------------------------------------------


def training_config(model, env: BipedTargetEnv, spec):
    return {
        "algorithm": "PPO (stable-baselines3)",
        "policy": "MlpPolicy",
        "total_timesteps_trained": int(model.num_timesteps),
        "gamma": float(model.gamma),
        "gae_lambda": float(model.gae_lambda),
        "n_steps": int(model.n_steps),
        "batch_size": int(model.batch_size),
        "n_epochs": int(model.n_epochs),
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "max_grad_norm": float(model.max_grad_norm),
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation": "Tanh",
        "n_envs_used": 24,
        "observation_normalisation": "VecNormalize (baked into the ONNX graph)",
        "reward_terms": spec["task"],
        "training_stages": [
            "0-2.5M   forward 1.5 / alive 1.0 -- converged to standing still",
            "2.5-3.5M forward 4.0 / alive 0.25 + log_std reset -- converged to lunging",
            "3.5-14.5M forward 4.0 / alive 0.5 / fall_penalty 25 / gamma 0.995 -- walking",
        ],
    }


def write_readme(path, spec, verify, evaluation, quat_results, model, traj_steps):
    joint_order = [a["joint"] for a in spec["actuators"]]

    rows, offset = [], 0
    for size, name, desc, units in OBS_LAYOUT:
        end = offset + size - 1
        idx = f"{offset}" if size == 1 else f"{offset}-{end}"
        rows.append(f"| `{idx}` | {size} | `{name}` | {desc} | {units} |")
        offset += size
    obs_table = "\n".join(rows)

    act_rows = "\n".join(
        f"| {a['index']} | `{a['name']}` | `{a['joint']}` | {a['gear']:.0f} | "
        f"[{a['ctrlrange'][0]:.0f}, {a['ctrlrange'][1]:.0f}] | {a['peak_torque_Nm']:.0f} N·m | "
        f"[{spec['joints'][[j['name'] for j in spec['joints']].index(a['joint'])]['range_deg'][0]:.0f}, "
        f"{spec['joints'][[j['name'] for j in spec['joints']].index(a['joint'])]['range_deg'][1]:.0f}]° |"
        for a in spec["actuators"]
    )

    body_rows = "\n".join(
        f"| `{b['name']}` | `{b['parent'] or '-'}` | {b['mass']:.2f} | "
        f"({b['local_pos'][0]:+.3f}, {b['local_pos'][1]:+.3f}, {b['local_pos'][2]:+.3f}) |"
        for b in spec["bodies"] if b["mass"] > 0
    )

    consistent = [k for k, v in quat_results.items() if v["consistent_with_our_position_map"]]
    spec_ok = quat_results["requested (-x,  z, -y,  w)"]["consistent_with_our_position_map"]
    quat_rows = "\n".join(
        f"| `{k}` | {v['max_error']:.2e} | {'**correct**' if v['consistent_with_our_position_map'] else 'wrong here'} | "
        f"{', '.join('`' + m + '`' for m in v['compatible_position_maps']) or '**none - unusable with any axis convention**'} |"
        for k, v in quat_results.items()
    )

    if spec_ok:
        quat_verdict = (
            "The requested mapping `(x, y, z, w) -> (-x, z, -y, w)` is consistent with the "
            "position mapping above and is what `MuJoCoToUnity` implements."
        )
    else:
        alt = consistent[0].split("(")[-1].rstrip(")") if consistent else "none"
        req = quat_results["requested (-x,  z, -y,  w)"]
        orphan = not req["compatible_position_maps"]
        extra = (
            "An exhaustive search over all 24 signed axis permutations that flip "
            "handedness found **no** position convention under which the requested form "
            "is a valid rotation map, so this is not a mismatch between two conventions "
            "-- the mapping is simply not a rotation. It is one sign away from two rules "
            "that *are* valid, listed in the table; `(-x, -z, -y, w)` is likely the "
            "intended one, but it belongs to the Y/Z-swap convention "
            "`unity = (x, z, y)`, not to the X-forward convention specified here."
            if orphan else
            "It is valid for a different position convention, shown in the table."
        )
        quat_verdict = f"""> **Correction, verified numerically.** The requested mapping
> `(x, y, z, w) -> (-x, z, -y, w)` is **not** consistent with the position mapping
> `(x, y, z) -> (-y, z, x)` above. Using the two together yields rotations that disagree
> with the positions: the root translates correctly while the limbs point the wrong way.
>
> {extra}
>
> The correct quaternion mapping for the stated axes is **`({alt})`**, derived from the
> requirement `R_unity == P · R_mujoco · P⁻¹` and confirmed to machine precision (max
> error 0.0) over 512 random rotations. That is what `MuJoCoToUnity.Rotation` below
> implements. Every candidate tested is in the table."""

    unsupported = verify["unsupported_by_sentis"]
    ops_line = (
        "All operators are Sentis-supported."
        if not unsupported
        else f"**Unsupported operators present: {unsupported}**"
    )

    return f"""# Biped Walk-to-Target — Unity Sentis Export

Exported {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} from a PPO policy
trained for {model.num_timesteps:,} steps in MuJoCo {mujoco.__version__}.

The policy takes a 49-dimensional egocentric observation and emits 12 joint torques
that walk a biped toward a goal marker, turning as the goal moves.

## Contents

| Path | What it is |
| --- | --- |
| `policy.onnx` | The exported policy. Opset {verify['opset'].get('ai.onnx')}, IR {verify['ir_version']}, batch 1, no dynamic axes |
| `reference_trajectory.json` | {traj_steps} steps of ground-truth I/O and state for parity testing |
| `robot_spec.json` | Full joint / body / actuator / simulation specification |
| `evaluation.json` | Performance over {evaluation['episodes']} deterministic rollouts |
| `training_config.json` | Hyperparameters and reward terms |
| `model/biped.xml` | The MJCF source (primitive geoms only — see `model/MESHES.md`) |
| `checkpoint/` | Raw SB3 PyTorch checkpoint + VecNormalize statistics |
| `src/` | Environment source, so observations can be reproduced exactly |

## The ONNX model

| Property | Value |
| --- | --- |
| Input | `{verify['input']['name']}` float32 `{verify['input']['shape']}` |
| Output | `{verify['output']['name']}` float32 `{verify['output']['shape']}` |
| Opset / IR | {verify['opset'].get('ai.onnx')} / {verify['ir_version']} |
| Dynamic axes | {'yes' if verify['has_dynamic_axes'] else 'none — fixed batch 1'} |
| Operators | {', '.join(verify['operators'])} |
| Size | {verify['file_bytes'] / 1024:.0f} KB, {verify['initializers']} initializers (weights embedded) |
| `onnx.checker` | passed (full_check) |
| ONNX vs PyTorch | max abs diff **{verify['max_abs_diff_vs_pytorch']:.3e}** over {verify['samples_compared']} real observations (tolerance {verify['tolerance']:.0e}) |

{ops_line}

**The graph is self-contained.** Observation normalisation and the action clamp are
baked in, so you feed raw observations and read final actions. Do not apply
normalisation yourself — it is already inside the network.

```csharp
using Unity.Sentis;

var model  = ModelLoader.Load(policyAsset);
var worker = new Worker(model, BackendType.CPU);

using var input = new Tensor<float>(new TensorShape(1, {OBS_DIM}), observation);
worker.Schedule(input);
var action = worker.PeekOutput() as Tensor<float>;   // shape (1, {NUM_ACTIONS}), already in [-1, 1]
```

## Observation layout (49 floats)

Assemble in exactly this order. Every quantity is in **MuJoCo** coordinates
(right-handed, Z-up) — convert Unity state into MuJoCo space *before* filling the
vector, not after.

| Index | Size | Field | Description | Units |
| --- | --- | --- | --- | --- |
{obs_table}

Notes:

* `projected_gravity` is the world down-vector `(0, 0, -1)` rotated into the torso
  frame: `R_torsoᵀ · (0, 0, -1)`. It encodes roll and pitch while deliberately
  leaking no heading, which is what makes the policy heading-invariant.
* `linear_velocity` / `angular_velocity` are also expressed in the torso frame
  (`R_torsoᵀ · v_world`), not the world frame.
* `target_direction` is the unit vector to the goal rotated by **negative torso yaw**
  only — yaw, not full orientation. With the goal dead ahead it reads `(1, 0)`.
* `last_action` is the previous step's network output, post-clamp. Feed zeros on the
  first step after a reset.
* Clipping is part of the observation definition and is applied *before* the
  network's own normalisation.

## Action → joint mapping

The network outputs 12 values already clamped to `[-1, 1]`. These are **normalised
torques**, not positions or velocities:

```
torque_i (N·m) = action_i × gear_i
```

There is no target integration and no PD loop in the policy — it is direct torque
control at {spec['simulation']['control_frequency_hz']:.0f} Hz.

| Index | Actuator | Joint | Gear | ctrlrange | Peak torque | Joint limits |
| --- | --- | --- | --- | --- | --- | --- |
{act_rows}

Joint order for observation indices 10–21 and 22–33 is identical to this action
order: `{', '.join(joint_order)}`.

## Bodies

| Body | Parent | Mass (kg) | Offset from parent (m) |
| --- | --- | --- | --- |
{body_rows}

Total mass {spec['bodies'][2]['subtree_mass']:.2f} kg. Inertia tensors and centres of
mass are in `robot_spec.json`.

## Coordinate conversion

MuJoCo is **right-handed, Z-up**: X forward, Y left, Z up.
Unity is **left-handed, Y-up**: X right, Y up, Z forward.

Positions and direction vectors:

```
unity.x = -mujoco.y      // MuJoCo +Y (left)    -> Unity -X
unity.y =  mujoco.z      // MuJoCo +Z (up)      -> Unity +Y
unity.z =  mujoco.x      // MuJoCo +X (forward) -> Unity +Z
```

MuJoCo stores quaternions as `(w, x, y, z)`; Unity uses `(x, y, z, w)`.

{quat_verdict}

| Candidate mapping | Max error | Valid with our axes? | Position convention it *is* valid for |
| --- | --- | --- | --- |
{quat_rows}

```csharp
public static class MuJoCoToUnity
{{
    // MuJoCo (x_fwd, y_left, z_up) -> Unity (x_right, y_up, z_fwd)
    public static Vector3 Position(double x, double y, double z)
        => new Vector3((float)(-y), (float)z, (float)x);

    public static Vector3 Direction(double x, double y, double z)
        => Position(x, y, z);

    // MuJoCo quaternion (w, x, y, z) -> Unity Quaternion (x, y, z, w)
    public static Quaternion Rotation(double w, double x, double y, double z)
        => new Quaternion({'(float)(-x), (float)z, (float)(-y), (float)w' if spec_ok else '(float)y, (float)(-z), (float)(-x), (float)w'});

    // Inverse: Unity -> MuJoCo, needed when building the observation vector
    public static (double x, double y, double z) ToMuJoCo(Vector3 v)
        => (v.z, -v.x, v.y);
}}
```

Angular velocity is a pseudo-vector: convert with `Direction()`, then negate the
result, because the handedness flip reverses the sense of rotation.

## PD / motor drive setup in Unity

The MuJoCo model uses **direct-torque motors**, not position servos. Every joint also
carries passive damping and rotor inertia that you must reproduce or the gait will not
transfer.

| MuJoCo property | Value | Unity equivalent |
| --- | --- | --- |
| `timestep` | {spec['simulation']['physics_timestep']} s | `Time.fixedDeltaTime = {spec['simulation']['physics_timestep']}` |
| `frame_skip` | {spec['simulation']['frame_skip']} | run inference every {spec['simulation']['frame_skip']} physics steps ({spec['simulation']['control_frequency_hz']:.0f} Hz) |
| `gravity` | {spec['simulation']['gravity']} | `Physics.gravity = new Vector3(0, {spec['simulation']['gravity'][2]:.2f}f, 0)` |
| joint `damping` | {spec['joints'][1]['damping']:.2f} N·m·s/rad | `ArticulationDrive.damping = {spec['joints'][1]['damping']:.2f}`, `stiffness = 0`, `target = 0` |
| joint `armature` | {spec['joints'][1]['armature']:.3f} kg·m² | add to the child body's inertia about the joint axis |
| geom `friction` | {spec['geoms'][1]['friction'][0]} (sliding) | `PhysicMaterial.dynamicFriction`, combine = Multiply |
| `condim` | {spec['geoms'][1]['condim']} | standard Unity friction cone |

Recommended `ArticulationBody` configuration per joint:

```csharp
var drive = body.xDrive;
drive.stiffness      = 0f;                     // no position servo: policy outputs torque
drive.damping        = {spec['joints'][1]['damping']:.2f}f;                  // matches MuJoCo joint damping
drive.forceLimit     = gear;                   // per-joint, from the table above
drive.target         = 0f;
drive.targetVelocity = 0f;
drive.lowerLimit     = jointLowerDeg;          // from the table above
drive.upperLimit     = jointUpperDeg;
body.xDrive          = drive;

// Each control tick, apply the network output as a torque:
body.AddRelativeTorque(jointAxis * (action[i] * gear[i]), ForceMode.Force);
```

Two traps worth calling out:

1. **Stiffness must be zero.** If you leave Unity's default position drive active it
   fights the policy's torques and the robot collapses. The policy was never trained
   against a servo.
2. **Armature matters.** MuJoCo's `armature = {spec['joints'][1]['armature']:.3f}` adds
   rotor inertia that stabilises the joints. Unity has no direct equivalent; without
   compensating inertia the joints are effectively lighter than in training and the gait
   becomes jittery.

If the gait still differs, replay `reference_trajectory.json`: feed each recorded
observation to your Sentis worker and compare the action against the recorded one. That
isolates a model/plumbing problem (actions differ) from a physics problem (actions match
but the robot moves differently).

## Measured performance

Over {evaluation['episodes']} deterministic episodes:

| Metric | Value |
| --- | --- |
| Targets reached per episode | **{evaluation['targets_reached_per_episode']['mean']:.2f}** (max {evaluation['targets_reached_per_episode']['max']:.0f}) |
| Episode length | {evaluation['episode_length_steps']['mean']:.0f} steps ({evaluation['episode_length_seconds']['mean']:.1f} s) |
| Return | {evaluation['return']['mean']:.0f} ± {evaluation['return']['std']:.0f} |
| Mean closing speed | {evaluation['mean_closing_speed_mps']['mean']:.2f} m/s |
| Survived the full 25 s episode | {evaluation['survived_full_episode_pct']:.0f}% |

Reaching more than one target per episode requires turning: each new goal spawns up to
±{np.degrees(spec['task']['target_angle_range_rad']):.0f}° from the current heading.

**Known limitation:** the policy still falls before the 25 s timeout in most episodes.
Training was stopped at {model.num_timesteps:,} of a scheduled 20.5M steps while episode
length was still improving.

## Task definition

Goals spawn {spec['task']['target_distance_range_m'][0]:.0f}–{spec['task']['target_distance_range_m'][1]:.0f} m away, within
±{np.degrees(spec['task']['target_angle_range_rad']):.0f}° of the current heading. A goal counts as reached within
{spec['task']['reach_radius_m']:.1f} m, at which point a new one spawns. An episode terminates when the
torso leaves the height band {spec['task']['healthy_z_range']} m or tips past
{np.degrees(np.arccos(spec['task']['min_uprightness'])):.0f}° from vertical.
"""


# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", default="runs/biped")
    ap.add_argument("--model", default=None)
    ap.add_argument("--stats", default=None)
    ap.add_argument("--out", default="export/biped_sentis")
    ap.add_argument("--traj-steps", type=int, default=150)
    ap.add_argument("--eval-episodes", type=int, default=10)
    ap.add_argument("--verify-samples", type=int, default=256)
    ap.add_argument("--tolerance", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=20250828)
    args = ap.parse_args()

    model_path = args.model or os.path.join(args.run_dir, "latest.zip")
    stats_path = args.stats or os.path.join(args.run_dir, "vecnormalize.pkl")
    for p in (model_path, stats_path):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}")

    out = args.out
    os.makedirs(out, exist_ok=True)
    for sub in ("model", "checkpoint", "src"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    line = lambda s: print(f"[export] {s}", flush=True)

    line(f"loading {model_path}")
    sb3_model, vec_normalize, wrapper = load_policy(model_path, stats_path)
    line(f"policy trained for {sb3_model.num_timesteps:,} steps")

    env = BipedTargetEnv()
    mj_model = env.unwrapped.model

    # --- ONNX -------------------------------------------------------------
    onnx_path = os.path.join(out, "policy.onnx")
    line(f"exporting ONNX (opset {TARGET_OPSET}, IR {TARGET_IR_VERSION}, batch 1)")
    export_onnx(wrapper, onnx_path)

    line(f"collecting {args.verify_samples} on-policy observations for verification")
    obs_samples = collect_observations(env, wrapper, args.verify_samples, args.seed)

    line("running onnx.checker and ONNX-vs-PyTorch comparison")
    verify = verify_onnx(onnx_path, wrapper, obs_samples, args.tolerance)

    # --- trajectory / spec / evaluation ------------------------------------
    line(f"recording {args.traj_steps}-step reference trajectory")
    frames = record_trajectory(env, wrapper, args.traj_steps, args.seed + 1)

    line("extracting robot specification from mjModel")
    spec = robot_spec(mj_model, env, env.unwrapped.frame_skip)

    line(f"evaluating over {args.eval_episodes} rollouts")
    evaluation = evaluate(env, wrapper, args.eval_episodes, args.seed + 2, 1000)

    line("verifying coordinate conversions")
    quat_results = check_quaternion_conversions()

    cfg = training_config(sb3_model, env, spec)
    env.close()

    # --- write bundle -------------------------------------------------------
    dump = lambda name, obj: json.dump(
        obj, open(os.path.join(out, name), "w"), indent=2
    )

    dump("reference_trajectory.json", {
        "description": "Ground-truth policy I/O and simulator state, MuJoCo coordinates (Z-up, right-handed).",
        "conventions": {
            "quaternion": "root_pose.quaternion_wxyz is (w, x, y, z)",
            "joint_order": [a["joint"] for a in spec["actuators"]],
            "control_dt": spec["simulation"]["control_timestep"],
            "note": "observation is the raw vector fed to policy.onnx; action is the network output",
        },
        "steps": len(frames),
        "trajectory": frames,
    })
    dump("robot_spec.json", spec)
    dump("evaluation.json", evaluation)
    dump("training_config.json", cfg)
    dump("onnx_verification.json", {**verify, "coordinate_conversion": quat_results})

    shutil.copy(XML_PATH, os.path.join(out, "model", "biped.xml"))
    with open(os.path.join(out, "model", "MESHES.md"), "w") as fh:
        fh.write(
            "# Mesh assets\n\n"
            "There are none. The robot is built entirely from MuJoCo primitive geoms\n"
            "(capsules, boxes, a sphere and a ground plane), so `biped.xml` has no\n"
            "`<mesh>` assets and no linked .stl/.obj files to ship.\n\n"
            "Collision geoms live in **group 3**, which MuJoCo hides by default. Sites\n"
            "named `bone_*` in group 4 mark hip, knee, ankle and toe positions for\n"
            "binding a skinned mesh.\n\n"
            "Primitive sizes are in `../robot_spec.json` under `geoms`, which is enough\n"
            "to rebuild the collision volumes as Unity colliders:\n"
            "capsule `size = [radius, half_length]`, box `size = [hx, hy, hz]`\n"
            "(Unity box colliders take full extents, so double these).\n"
        )

    shutil.copy(model_path, os.path.join(out, "checkpoint", "policy_ppo.zip"))
    shutil.copy(stats_path, os.path.join(out, "checkpoint", "vecnormalize.pkl"))
    torch.save(
        {
            "state_dict": wrapper.state_dict(),
            "obs_dim": OBS_DIM,
            "action_dim": NUM_ACTIONS,
            "architecture": "Linear(49,256)-Tanh-Linear(256,256)-Tanh-Linear(256,12)",
            "note": "SentisPolicy wrapper: normalisation and action clamp included",
        },
        os.path.join(out, "checkpoint", "policy_wrapper.pt"),
    )

    for src in ("biped/env.py", "biped/__init__.py"):
        shutil.copy(src, os.path.join(out, "src", os.path.basename(src)))
    shutil.copy(__file__, os.path.join(out, "src", "export_unity.py"))

    with open(os.path.join(out, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(write_readme(out, spec, verify, evaluation, quat_results,
                              sb3_model, args.traj_steps))

    dump("manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "policy_steps": int(sb3_model.num_timesteps),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "onnx": onnx.__version__,
            "onnxruntime": ort.__version__,
            "mujoco": mujoco.__version__,
        },
        "files": sorted(
            os.path.relpath(os.path.join(r, f), out).replace("\\", "/")
            for r, _, fs in os.walk(out) for f in fs
        ),
    })

    # --- summary ------------------------------------------------------------
    print()
    print("=" * 72)
    print("EXPORT SUMMARY")
    print("=" * 72)
    print(f"output directory     : {os.path.abspath(out)}")
    print(f"policy               : {sb3_model.num_timesteps:,} training steps")
    print()
    print("ONNX")
    print(f"  input              : {verify['input']['name']} {verify['input']['shape']} float32")
    print(f"  output             : {verify['output']['name']} {verify['output']['shape']} float32")
    print(f"  opset / IR         : {verify['opset'].get('ai.onnx')} / {verify['ir_version']}")
    print(f"  dynamic axes       : {'YES' if verify['has_dynamic_axes'] else 'none (fixed batch 1)'}")
    print(f"  operators          : {', '.join(verify['operators'])}")
    print(f"  sentis-unsupported : {verify['unsupported_by_sentis'] or 'none'}")
    print(f"  onnx.checker       : {'PASS' if verify['checker_passed'] else 'FAIL'}")
    print(f"  vs PyTorch         : max abs diff {verify['max_abs_diff_vs_pytorch']:.3e} "
          f"over {verify['samples_compared']} obs (tol {verify['tolerance']:.0e}) -> "
          f"{'PASS' if verify['numerically_equivalent'] else 'FAIL'}")
    print(f"  size               : {verify['file_bytes'] / 1024:.0f} KB")
    print()
    print("COORDINATE CONVERSION")
    for label, r in quat_results.items():
        print(f"  {label:<30} err {r['max_error']:.2e}  "
              f"{'CONSISTENT' if r['consistent_with_our_position_map'] else 'inconsistent'}"
              f"  | valid for: {', '.join(r['compatible_position_maps']) or 'NO axis convention'}")
    print()
    print(f"EVALUATION ({evaluation['episodes']} deterministic rollouts)")
    print(f"  mean return              : {evaluation['return']['mean']:.1f} "
          f"+/- {evaluation['return']['std']:.1f}")
    print(f"  mean episode length      : {evaluation['episode_length_steps']['mean']:.0f} steps "
          f"({evaluation['episode_length_seconds']['mean']:.1f} s)")
    print(f"  targets reached / episode: {evaluation['targets_reached_per_episode']['mean']:.2f} "
          f"(max {evaluation['targets_reached_per_episode']['max']:.0f})")
    print(f"  final distance to goal   : {evaluation['final_distance_to_target_m']['mean']:.2f} m")
    print(f"  mean closing speed       : {evaluation['mean_closing_speed_mps']['mean']:.2f} m/s")
    print(f"  survived full episode    : {evaluation['survived_full_episode_pct']:.0f}%")
    print("=" * 72)

    ok = (verify["numerically_equivalent"] and not verify["unsupported_by_sentis"]
          and not verify["has_dynamic_axes"])
    print("RESULT:", "clean export, Sentis-compatible" if ok else "EXPORT HAS ISSUES (see above)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
