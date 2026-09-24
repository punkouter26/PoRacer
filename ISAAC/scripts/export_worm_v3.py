#!/usr/bin/env python3
"""Export + evaluate the Isaac Lab 3 / Newton (MuJoCo-Warp) Worm5 policy (WORM_SPEC.md "Export").

    ISAAC\\isaaclab3\\.venv\\Scripts\\python.exe ISAAC\\scripts\\export_worm_v3.py [--checkpoint <model_N.pt>]

1. Loads the newest checkpoint under ISAAC/logs/rsl_rl_v3/worm5_newton (or --checkpoint).
2. ONNX: input ``obs`` float[1, 35], output ``actions`` float[1, 8] = the deterministic mean, NOT
   clipped, observation normaliser baked in; opset 15, IR 8, fixed batch 1.
   -> training/worm/export/worm_isaaclab3.onnx
3. Evaluates --episodes deterministic 20 s episodes in Newton/MuJoCo-Warp (Isaac-Worm5-Flat-Newton-Play-v0:
   spec reset, no randomisation, seed 12345), one episode per env, all in parallel, driven by the ONNX
   itself: actions = clip(ONNX mean, -1, 1) (WORM_SPEC.md item 15). Speed of an episode = (seg2 x at
   step 1000 - seg2 x at reset) / 20 s. Worlds ended by the health guard (item 12) are counted and left
   out of the speed statistics.
4. onnxruntime vs torch parity on every observation of that rollout (max abs error).
5. Writes worm_isaaclab3_report.json and worm_isaaclab3_rig_order.json next to the ONNX.
"""

import argparse
import glob
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

from isaaclab_tasks.utils.sim_launcher import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description="Export and evaluate the Worm5 Isaac Lab 3 / Newton policy.")
parser.add_argument("--task", default="Isaac-Worm5-Flat-Newton-Play-v0")
parser.add_argument("--checkpoint", default=None, help="model_N.pt (default: newest run, newest model).")
parser.add_argument("--log_root", default=os.path.join(REPO, "ISAAC", "logs", "rsl_rl_v3"))
parser.add_argument("--experiment", default="worm5_newton")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--out_dir", default=os.path.join(REPO, "training", "worm", "export"))
parser.add_argument("--onnx_name", default="worm_isaaclab3.onnx")
parser.add_argument("--report_name", default="worm_isaaclab3_report.json")
parser.add_argument("--rig_order_name", default="worm_isaaclab3_rig_order.json")
parser.add_argument("--seed", type=int, default=None, help="Default: WORM_SPEC.md item 15 (12345).")
add_launcher_args(parser)
args = parser.parse_args()

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import worm_tasks_v3  # noqa: E402,F401
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from worm_tasks_v3 import spec  # noqa: E402

EPISODE_STEPS = int(round(spec.EPISODE_LENGTH_S / spec.STEP_DT))  # 1000


def newest_checkpoint(exp_dir: str) -> str:
    runs = sorted((d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)), key=os.path.getmtime)
    for run in reversed(runs):
        models = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(os.path.basename(p)[6:-3]))
        if models:
            return models[-1]
    raise FileNotFoundError(f"no model_*.pt under {exp_dir}")


def ver(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return None


class _OnnxPolicy(torch.nn.Module):
    """obs[1, 35] -> actions[1, 8]: baked normaliser + MLP + Gaussian mean (no clip)."""

    def __init__(self, actor):
        super().__init__()
        self.inner = actor.as_onnx(verbose=False)

    def forward(self, obs):
        return self.inner(obs)


def main():
    os.makedirs(args.out_dir, exist_ok=True)
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = handle_deprecated_rsl_rl_cfg(
        load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point"), metadata.version("rsl-rl-lib")
    )
    env_cfg.scene.num_envs = args.episodes
    env_cfg.seed = spec.EVAL_SEED if args.seed is None else args.seed
    # one spare second so no env auto-resets inside the measured 1000 steps
    env_cfg.episode_length_s = spec.EPISODE_LENGTH_S + 1.0
    with launch_simulation(env_cfg, args):
        run(env_cfg, agent_cfg)


def run(env_cfg, agent_cfg):
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    assert abs(raw.step_dt - spec.STEP_DT) < 1e-9

    ckpt = os.path.abspath(args.checkpoint) if args.checkpoint else newest_checkpoint(
        os.path.join(args.log_root, args.experiment))
    print(f"[export_worm_v3] checkpoint: {ckpt}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=raw.device)
    runner.load(ckpt)
    actor = runner.get_inference_policy(device=raw.device)

    # ---------------------------------------------------------------------------- ONNX --
    onnx_path = os.path.join(args.out_dir, args.onnx_name)
    module = _OnnxPolicy(actor).to("cpu").eval()
    for p in module.parameters():
        p.requires_grad_(False)
    with torch.inference_mode():
        torch.onnx.export(
            module, (torch.zeros(1, spec.NUM_OBS),), onnx_path, export_params=True, opset_version=15,
            do_constant_folding=True, input_names=["obs"], output_names=["actions"], dynamic_axes=None, dynamo=False,
        )
    model = onnx.load(onnx_path)
    model.ir_version = 8
    onnx.save_model(model, onnx_path, save_as_external_data=False)
    onnx.checker.check_model(onnx_path)
    opset = max(o.version for o in model.opset_import if o.domain in ("", "ai.onnx"))
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    io = {"inputs": [[i.name, i.shape, i.type] for i in sess.get_inputs()],
          "outputs": [[o.name, o.shape, o.type] for o in sess.get_outputs()]}
    assert io["inputs"] == [["obs", [1, spec.NUM_OBS], "tensor(float)"]], io
    assert io["outputs"] == [["actions", [1, spec.NUM_ACTIONS], "tensor(float)"]], io
    assert opset == 15 and model.ir_version == 8, (opset, model.ir_version)
    print(f"[export_worm_v3] wrote {onnx_path}  io={io}  opset {opset}, IR {model.ir_version}")

    # ---------------------------------------------------------------------- evaluation --
    seg2 = robot.body_names.index(spec.REF_BODY)
    seg_ids = [robot.body_names.index(n) for n in spec.SEGMENT_NAMES]
    origins = raw.scene.env_origins
    tm = raw.termination_manager
    n_env = raw.num_envs
    dev = raw.device
    sat_level = spec.FORCE_LIMIT * (1.0 - 1e-4)
    max_err = 0.0
    with torch.inference_mode():
        env.reset()
        obs = env.get_observations()
        d = robot.data
        x0 = (d.body_link_pos_w.torch[:, seg2, :2] - origins[:, :2]).clone()
        vx_sum = torch.zeros(n_env, device=dev)
        roll_sum = torch.zeros(n_env, device=dev)
        roll_max = torch.zeros(n_env, device=dev)
        qd_max = torch.zeros(n_env, device=dev)
        sat_steps = torch.zeros(n_env, device=dev)
        sat_joint = torch.zeros(n_env, robot.num_joints, device=dev)
        height_sum = torch.zeros(n_env, device=dev)
        diverged = torch.zeros(n_env, dtype=torch.bool, device=dev)
        for step in range(EPISODE_STEPS):
            o = obs["policy"]
            o_np = o.detach().cpu().numpy().astype(np.float32)
            onnx_mean = np.concatenate([sess.run(None, {"obs": o_np[i:i + 1]})[0] for i in range(n_env)], axis=0)
            torch_mean = actor(obs).detach().cpu().numpy()
            max_err = max(max_err, float(np.abs(onnx_mean - torch_mean).max()))
            # clip(ONNX mean, -1, 1): the RSL-RL wrapper clips with clip_actions = 1.0
            obs, _, dones, _ = env.step(torch.from_numpy(onnx_mean).to(dev))
            div_now = tm.get_term("diverged")
            diverged |= div_now
            if bool((dones & ~div_now).any()):
                raise RuntimeError(f"a healthy env reset inside the measured window at step {step}")
            quat = d.body_link_quat_w.torch
            vx_sum += d.body_link_lin_vel_w.torch[:, seg2, 0]
            w_b = quat_apply_inverse(quat[:, seg2], d.body_link_ang_vel_w.torch[:, seg2])
            roll = w_b[:, 0].abs()
            roll_sum += roll
            roll_max = torch.maximum(roll_max, roll)
            qd_max = torch.maximum(qd_max, d.joint_vel.torch.abs().max(dim=1).values)
            sat = d.applied_torque.torch.abs() >= sat_level
            sat_steps += sat.any(dim=1).float()
            sat_joint += sat.float()
            height_sum += (d.body_link_pos_w.torch[:, seg_ids, 2] - origins[:, 2:3]).mean(dim=1)
        x1 = d.body_link_pos_w.torch[:, seg2, :2] - origins[:, :2]  # read at step 1000
    ok = (~diverged).cpu().numpy()
    n_div = int(diverged.sum())

    def good(t):
        return t.cpu().numpy()[ok]

    dist = good(x1[:, 0] - x0[:, 0])
    lateral = good(x1[:, 1] - x0[:, 1])
    speed = dist / spec.EPISODE_LENGTH_S
    mean_vx = good(vx_sum / EPISODE_STEPS)
    health = {
        "divergedWorlds": n_div,
        "meanAbsRollRateSeg2": float(good(roll_sum / EPISODE_STEPS).mean()),
        "peakAbsRollRateSeg2": float(good(roll_max).max()),
        "peakJointSpeedRadS": float(good(qd_max).max()),
        "fractionTimeAnyActuatorSaturated": float(good(sat_steps / EPISODE_STEPS).mean()),
        "fractionTimeSaturatedPerJoint": {
            n: float(good(sat_joint[:, robot.joint_names.index(n)] / EPISODE_STEPS).mean()) for n in spec.ACTION_ORDER
        },
        "meanSegmentHeightM": float(good(height_sum / EPISODE_STEPS).mean()),
        "meanAbsLateralDrift20sM": float(np.abs(lateral).mean()),
        "meanLateralDrift20sM": float(lateral.mean()),
    }
    print(f"[export_worm_v3] {ok.sum()}/{n_env} healthy episodes (diverged {n_div}): "
          f"speed {speed.mean():+.4f} +/- {speed.std():.4f} m/s, distance {dist.mean():+.3f} m in 20 s "
          f"(min {dist.min():+.3f}, max {dist.max():+.3f}); mean instantaneous vx {mean_vx.mean():+.4f} m/s")
    print(f"[export_worm_v3] health: {json.dumps(health)}")
    print(f"[export_worm_v3] onnxruntime vs torch (GPU policy) on {EPISODE_STEPS * n_env} real observations: "
          f"max|diff| = {max_err:.3e}")

    # ------------------------------------------------------------------------ reports --
    run_dir = os.path.dirname(ckpt)
    summary = {}
    if os.path.isfile(os.path.join(run_dir, "train_summary.json")):
        with open(os.path.join(run_dir, "train_summary.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
    ckpt_iter = int(torch.load(ckpt, map_location="cpu", weights_only=False).get("iter", -1))
    report = {
        "tool": "isaaclab3-newton-mjwarp-rsl_rl",
        "meanSpeed": float(speed.mean()),
        "stdSpeed": float(speed.std()),
        "meanDistance20s": float(dist.mean()),
        "stepsTrained": summary.get("stepsTrained"),
        "wallMinutes": summary.get("wallMinutes"),
        "onnxMaxAbsErr": max_err,
        "notes": (
            f"Isaac Lab {ver('isaaclab')} (tag v3.0.0-beta2.patch1), kit-less: Newton {ver('newton')} with the "
            f"MuJoCo-Warp solver (mujoco-warp {ver('mujoco-warp')}, warp {ver('warp-lang')}; Newton method, 10 iterations, "
            "implicitfast, pyramidal cone, dt 0.005 s), no Isaac Sim / PhysX. "
            f"rsl-rl-lib {ver('rsl-rl-lib')}, torch {ver('torch')}. {args.episodes} deterministic episodes of 1000 policy "
            "steps (20 s), seed 12345, spec reset (yaw U(+/-45 deg) about the head, joint noise +/-0.05 rad), no "
            "randomisation, actions = clip(ONNX mean, -1, 1). Speed = seg2 displacement along +x (read at step "
            "1000) / 20 s; std is over episodes (population std); health-guard (diverged) episodes excluded. "
            f"Joint damping {spec.JOINT_DAMPING} is MuJoCo's own passive dof_damping (as worm.xml); servo kp "
            f"{spec.KP}, force limit {spec.FORCE_LIMIT} N*m. wallMinutes = training loop wall time (excludes start-up). "
            "Parity: onnxruntime (CPU) vs the torch policy on GPU, on every rollout observation."
        ),
        "details": {
            "checkpoint": os.path.relpath(ckpt, REPO),
            "checkpointIteration": ckpt_iter,
            "episodes": args.episodes,
            "healthyEpisodes": int(ok.sum()),
            "seed": env_cfg.seed,
            "meanInstantaneousVx": float(mean_vx.mean()),
            "health": health,
            "minDistance20s": float(dist.min()),
            "maxDistance20s": float(dist.max()),
            "onnx": {"path": os.path.relpath(onnx_path, REPO), "opset": opset, "irVersion": model.ir_version, "io": io},
            "training": summary,
        },
    }
    with open(os.path.join(args.out_dir, args.report_name), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    om = raw.observation_manager
    layout, start = [], 0
    for name, dim in zip(om.active_terms["policy"], om.group_obs_term_dim["policy"]):
        layout.append({"term": name, "start": start, "size": int(np.prod(dim))})
        start += int(np.prod(dim))
    act_ids = raw.action_manager.get_term("joint_pos")._joint_ids
    act_ids = list(range(robot.num_joints)) if isinstance(act_ids, slice) else [int(i) for i in act_ids]
    rig_order = {
        "tool": "isaaclab3-newton",
        "actionOrder": spec.ACTION_ORDER,
        "simJointOrder": list(robot.joint_names),
        "actionToSimJointIndex": act_ids,
        "simBodyOrder": list(robot.body_names),
        "referenceBody": spec.REF_BODY,
        "actionScaleRad": spec.ACTION_SCALE,
        "obsLayout": layout,
        "note": "Observation joint terms and actions are in actionOrder; simJointOrder is Newton's internal order.",
    }
    with open(os.path.join(args.out_dir, args.rig_order_name), "w", encoding="utf-8") as f:
        json.dump(rig_order, f, indent=2)
    print(f"[export_worm_v3] wrote {args.report_name} and {args.rig_order_name} in {args.out_dir}")
    print(f"[export_worm_v3] report: {json.dumps({k: v for k, v in report.items() if k != 'details'})}")
    env.close()


if __name__ == "__main__":
    code = 0
    try:
        main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
