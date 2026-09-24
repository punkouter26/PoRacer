#!/usr/bin/env python3
"""Export + evaluate the Isaac Lab Worm5 policy (WORM_SPEC.md "Export").

    set OMNI_KIT_ACCEPT_EULA=YES
    ISAAC\\isaaclab\\.venv\\Scripts\\python.exe ISAAC\\scripts\\export_worm.py [--checkpoint <model_N.pt>]

1. Loads the newest checkpoint under ISAAC/logs/rsl_rl/worm5 (or --checkpoint).
2. ONNX: input ``obs`` float[1, 35], output ``actions`` float[1, 8] = the deterministic mean,
   NOT clipped, observation normaliser baked in; opset 15, IR 8, fixed batch 1.
   -> training/worm/export/worm_isaac.onnx
3. Evaluates --episodes deterministic 20 s episodes in Isaac (Isaac-Worm5-Flat-Play-v0: spec
   reset, no randomisation), one episode per env, all in parallel. Speed of an episode =
   (seg2 x after 1000 policy steps - seg2 x at reset) / 20 s, i.e. progress along the goal (+x).
4. onnxruntime vs torch parity on real observations from that rollout (max abs error).
5. Writes worm_isaac_report.json and worm_isaac_rig_order.json next to the ONNX.
"""

import argparse
import glob
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# onnx / onnxruntime BEFORE the SimulationApp: kit ships its own onnxruntime DLLs and the pip one
# cannot initialise on top of them (see export_bundle.py).
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

from isaaclab.app import AppLauncher  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

parser = argparse.ArgumentParser(description="Export and evaluate the Worm5 Isaac policy.")
parser.add_argument("--task", default="Isaac-Worm5-Flat-Play-v0")
parser.add_argument("--checkpoint", default=None, help="model_N.pt (default: newest run, newest model).")
parser.add_argument("--log_root", default=os.path.join(REPO, "ISAAC", "logs", "rsl_rl"))
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--parity_samples", type=int, default=2000)
parser.add_argument("--out_dir", default=os.path.join(REPO, "training", "worm", "export"))
parser.add_argument("--onnx_name", default="worm_isaac.onnx")
parser.add_argument("--report_name", default="worm_isaac_report.json")
parser.add_argument("--rig_order_name", default="worm_isaac_rig_order.json")
parser.add_argument("--seed", type=int, default=7)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import worm_tasks  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from worm_tasks import spec  # noqa: E402
from worm_tasks.worm_cfg import DAMPING_MODE  # noqa: E402

EPISODE_STEPS = int(round(spec.EPISODE_LENGTH_S / spec.STEP_DT))  # 1000


def newest_checkpoint(exp_dir: str) -> str:
    runs = sorted((d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d)), key=os.path.getmtime)
    for run in reversed(runs):
        models = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(os.path.basename(p)[6:-3]))
        if models:
            return models[-1]
    raise FileNotFoundError(f"no model_*.pt under {exp_dir}")


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
    env_cfg.seed = args.seed
    # one spare second so no env auto-resets inside the measured 1000 steps
    env_cfg.episode_length_s = spec.EPISODE_LENGTH_S + 1.0
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    assert abs(raw.step_dt - spec.STEP_DT) < 1e-9

    ckpt = os.path.abspath(args.checkpoint) if args.checkpoint else newest_checkpoint(
        os.path.join(args.log_root, agent_cfg.experiment_name))
    print(f"[export_worm] checkpoint: {ckpt}")
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
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    io = {"inputs": [[i.name, i.shape, i.type] for i in sess.get_inputs()],
          "outputs": [[o.name, o.shape, o.type] for o in sess.get_outputs()]}
    assert io["inputs"] == [["obs", [1, spec.NUM_OBS], "tensor(float)"]], io
    assert io["outputs"] == [["actions", [1, spec.NUM_ACTIONS], "tensor(float)"]], io
    print(f"[export_worm] wrote {onnx_path}  io={io}  opset 15, IR {model.ir_version}")

    # ---------------------------------------------------------------------- evaluation --
    seg2 = robot.body_names.index(spec.REF_BODY)
    origins = raw.scene.env_origins
    obs_log = []
    per_step = min(raw.num_envs, -(-args.parity_samples // EPISODE_STEPS))  # ceil
    with torch.inference_mode():
        env.reset()
        obs = env.get_observations()
        x0 = (robot.data.body_link_pos_w[:, seg2, :2] - origins[:, :2]).clone()
        vx_sum = torch.zeros(raw.num_envs, device=raw.device)
        seg_ids = [robot.body_names.index(n) for n in spec.SEGMENT_NAMES]
        z_max = torch.zeros(raw.num_envs, device=raw.device)
        grounded_sum = torch.zeros(raw.num_envs, device=raw.device)
        qd_max = torch.zeros(raw.num_envs, device=raw.device)
        for step in range(EPISODE_STEPS):
            actions = actor(obs)  # deterministic mean, unclipped (the wrapper clips before the env)
            obs_log.append(torch.cat([obs["policy"][:per_step].cpu(), actions[:per_step].cpu()], dim=1))
            obs, _, dones, _ = env.step(actions)
            if bool(dones.any()):
                raise RuntimeError(f"an env reset inside the measured window at step {step}")
            vx_sum += robot.data.body_link_lin_vel_w[:, seg2, 0]
            seg_z = robot.data.body_link_pos_w[:, seg_ids, 2] - origins[:, 2:3]
            z_max = torch.maximum(z_max, seg_z.max(dim=1).values)
            grounded_sum += (seg_z < spec.RIG["radius"] + 0.01).float().sum(dim=1)
            qd_max = torch.maximum(qd_max, robot.data.joint_vel.abs().max(dim=1).values)
        x1 = robot.data.body_link_pos_w[:, seg2, :2] - origins[:, :2]
    dist = (x1[:, 0] - x0[:, 0]).cpu().numpy()
    lateral = (x1[:, 1] - x0[:, 1]).cpu().numpy()
    speed = dist / spec.EPISODE_LENGTH_S
    mean_vx = (vx_sum / EPISODE_STEPS).cpu().numpy()
    print(f"[export_worm] {args.episodes} episodes: speed {speed.mean():+.4f} +/- {speed.std():.4f} m/s, "
          f"distance {dist.mean():+.3f} m in 20 s (min {dist.min():+.3f}, max {dist.max():+.3f}); "
          f"mean instantaneous vx {mean_vx.mean():+.4f} m/s; |lateral| {np.abs(lateral).mean():.3f} m; "
          f"max segment height {float(z_max.max()):.3f} m, segments grounded {float((grounded_sum / EPISODE_STEPS).mean()):.2f}/5, "
          f"max |qdot| {float(qd_max.max()):.1f} rad/s")

    # ------------------------------------------------------------------------ parity --
    pairs = torch.cat(obs_log, dim=0)[: args.parity_samples]
    obs_np = pairs[:, : spec.NUM_OBS].numpy().astype(np.float32)
    torch_act = pairs[:, spec.NUM_OBS:].numpy()
    ort_act = np.concatenate([sess.run(None, {"obs": o[None]})[0] for o in obs_np], axis=0)
    max_err = float(np.abs(ort_act - torch_act).max())
    print(f"[export_worm] onnxruntime vs torch (GPU policy) on {len(obs_np)} real observations: max|diff| = {max_err:.3e}")

    # ------------------------------------------------------------------------ reports --
    run_dir = os.path.dirname(ckpt)
    summary = {}
    if os.path.isfile(os.path.join(run_dir, "train_summary.json")):
        with open(os.path.join(run_dir, "train_summary.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
    ckpt_iter = int(torch.load(ckpt, map_location="cpu", weights_only=False).get("iter", -1))
    report = {
        "tool": "isaaclab-rsl_rl",
        "meanSpeed": float(speed.mean()),
        "stdSpeed": float(speed.std()),
        "meanDistance20s": float(dist.mean()),
        "stepsTrained": summary.get("stepsTrained"),
        "wallMinutes": summary.get("wallMinutes"),
        "onnxMaxAbsErr": max_err,
        "notes": (
            f"Isaac Lab {metadata.version('isaaclab')} / Isaac Sim {metadata.version('isaacsim')} (PhysX 5, TGS), "
            f"rsl-rl-lib {metadata.version('rsl-rl-lib')}. {args.episodes} deterministic episodes of 1000 policy steps "
            "(20 s), spec reset (yaw U(+/-45 deg) about the head, joint noise +/-0.05 rad), no randomisation. "
            "Speed = seg2 displacement along +x / 20 s; std is over episodes (population std). "
            f"Joint damping mode '{DAMPING_MODE}' ('joint' = PhysX joint viscous friction 1.0, drive damping 0). "
            "wallMinutes = training loop wall time (excludes simulator start-up). Parity: onnxruntime (CPU) vs the "
            "torch policy on GPU, on real rollout observations."
        ),
        "details": {
            "checkpoint": os.path.relpath(ckpt, REPO),
            "checkpointIteration": ckpt_iter,
            "episodes": args.episodes,
            "meanInstantaneousVx": float(mean_vx.mean()),
            "meanAbsLateral20s": float(np.abs(lateral).mean()),
            "plausibility": {
                "maxSegmentHeightM": float(z_max.max()),
                "meanSegmentsGrounded": float((grounded_sum / EPISODE_STEPS).mean()),
                "maxJointSpeedRadS": float(qd_max.max()),
            },
            "minDistance20s": float(dist.min()),
            "maxDistance20s": float(dist.max()),
            "onnx": {"path": os.path.relpath(onnx_path, REPO), "opset": 15, "irVersion": model.ir_version, "io": io},
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
        "tool": "isaaclab",
        "actionOrder": spec.ACTION_ORDER,
        "simJointOrder": list(robot.joint_names),
        "actionToSimJointIndex": act_ids,
        "simBodyOrder": list(robot.body_names),
        "referenceBody": spec.REF_BODY,
        "actionScaleRad": spec.ACTION_SCALE,
        "obsLayout": layout,
        "note": "Observation joint terms and actions are in actionOrder; simJointOrder is PhysX's internal order.",
    }
    with open(os.path.join(args.out_dir, args.rig_order_name), "w", encoding="utf-8") as f:
        json.dump(rig_order, f, indent=2)
    print(f"[export_worm] wrote {args.report_name} and {args.rig_order_name} in {args.out_dir}")
    print(f"[export_worm] report: {json.dumps({k: v for k, v in report.items() if k != 'details'})}")
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
        app.close()
        os._exit(code)
