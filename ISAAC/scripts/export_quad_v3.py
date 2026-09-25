#!/usr/bin/env python3
"""Export + evaluate the Isaac Lab 3 / Newton (MuJoCo-Warp) quad policy (QUAD_SPEC.md "Export and tests").

    ISAAC\\isaaclab3\\.venv\\Scripts\\python.exe ISAAC\\scripts\\export_quad_v3.py [--checkpoint <model_N.pt> | --best]

1. Checkpoint: --checkpoint <path>, or --best (the newest run's train_summary.json bestCheckpoint: the
   best one before a collapse), or by default the newest model of the newest run under
   ISAAC/logs/rsl_rl_v3/quad_newton.
2. ONNX: input ``obs`` float[1, 36], output ``actions`` float[1, 8] = the deterministic mean, NOT clipped,
   observation normaliser baked in; opset 15, IR 8, fixed batch 1 -> training/quad/export/quad_isaaclab3.onnx
3. Evaluation, the same protocol as training/creature/evaluate.py (the MuJoCo side): --episodes (100)
   deterministic 20 s episodes in Newton/MuJoCo-Warp (Isaac-Quad-Flat-Newton-Play-v0: spec reset, seed
   12345, no randomisation, no pushes), one episode per env, all in parallel, driven by the ONNX itself:
   actions = clip(ONNX mean, -1, 1). Each env's FIRST episode counts. A fall (torso up.z < 0.5 or torso
   z < 0.45 m on the post-step state) ends that racer's run: its distance stops there and it still counts
   in meanSpeed = torso x displacement until the end / 20 s; fallRate is reported beside it. The fall
   test includes rounds 5-6's torso / upper-leg floor contact at any substep. (The Play
   env's fall termination is switched off here and the same test is applied by this script, so the
   post-step state of the falling step is still readable before any reset.) Worlds ended by the health
   guard are excluded from every statistic and counted.
4. Plausibility (rule I), the MuJoCo side's QuadPlausibility keys, over each episode while it runs. Contact
   metrics are at SUBSTEP resolution (5 ms, the graphed FootContactTracker): feet down, airborne fraction,
   slip in stance, stance duration per footfall (closed at lift-off), footfalls/s, peak single-foot impact
   (after the first 0.5 s, and including the spawn drop), mean per-footfall peak force. Also
   meanActionRate50HzEquivalent = the 20 Hz action rate x (0.02 / 0.05)^2 (same slew spread over 2.5x more steps).
5. onnxruntime vs torch parity on every observation of the rollout (max abs error).
6. Writes quad_isaaclab3_report.json and quad_isaaclab3_rig_order.json next to the ONNX.
"""

import argparse
import glob
import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

from isaaclab_tasks.utils.sim_launcher import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description="Export and evaluate the quad Isaac Lab 3 / Newton policy.")
parser.add_argument("--task", default="Isaac-Quad-Flat-Newton-Play-v0")
parser.add_argument("--checkpoint", default=None, help="model_N.pt (default: newest run, newest model).")
parser.add_argument("--best", action="store_true", help="Use the newest run's bestCheckpoint (train_summary.json).")
parser.add_argument("--log_root", default=os.path.join(REPO, "ISAAC", "logs", "rsl_rl_v3"))
parser.add_argument("--experiment", default="quad_newton")
parser.add_argument("--episodes", type=int, default=100)
parser.add_argument("--out_dir", default=os.path.join(REPO, "training", "quad", "export"))
parser.add_argument("--onnx_name", default="quad_isaaclab3.onnx")
parser.add_argument("--report_name", default="quad_isaaclab3_report.json")
parser.add_argument("--rig_order_name", default="quad_isaaclab3_rig_order.json")
parser.add_argument("--seed", type=int, default=None, help="Default: 12345 (WORM_SPEC item 15).")
add_launcher_args(parser)
args = parser.parse_args()

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import quad_tasks_v3  # noqa: E402,F401
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from quad_tasks_v3 import spec  # noqa: E402
from quad_tasks_v3.contact_tracker import FootContactTracker  # noqa: E402
from quad_tasks_v3.mdp import fallen_mask, touch_this_step  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

EPISODE_STEPS = spec.EPISODE_STEPS  # 400 at 20 Hz
SETTLE_SECONDS = 0.5  # peak foot impact excludes the spawn drop (as the MuJoCo side)


def newest_run(exp_dir: str) -> str:
    runs = sorted((d for d in glob.glob(os.path.join(exp_dir, "*")) if os.path.isdir(d) and glob.glob(os.path.join(d, "model_*.pt"))),
                  key=os.path.getmtime)
    if not runs:
        raise FileNotFoundError(f"no run with model_*.pt under {exp_dir}")
    return runs[-1]


def pick_checkpoint() -> str:
    if args.checkpoint:
        return os.path.abspath(args.checkpoint)
    run = newest_run(os.path.join(args.log_root, args.experiment))
    if args.best:
        with open(os.path.join(run, "train_summary.json"), "r", encoding="utf-8") as f:
            best = json.load(f)["bestCheckpoint"]["checkpoint"]
        return os.path.join(REPO, best)
    models = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(os.path.basename(p)[6:-3]))
    return models[-1]


def ver(p):
    try:
        return metadata.version(p)
    except metadata.PackageNotFoundError:
        return None


class _OnnxPolicy(torch.nn.Module):
    """obs[1, 36] -> actions[1, 8]: baked normaliser + MLP + Gaussian mean (no clip)."""

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
    env_cfg.episode_length_s = spec.EPISODE_LENGTH_S + 1.0  # no auto-reset inside the measured 400 steps
    env_cfg.terminations.fall = None  # applied below on the post-step state (same test), see docstring
    FootContactTracker.metrics_enabled = True  # substep contact metrics, built into the graph
    with launch_simulation(env_cfg, args):
        run(env_cfg, agent_cfg)


def run(env_cfg, agent_cfg):
    env = RslRlVecEnvWrapper(gym.make(args.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    raw = env.unwrapped
    robot = raw.scene["robot"]
    assert abs(raw.step_dt - spec.STEP_DT) < 1e-9

    ckpt = pick_checkpoint()
    print(f"[export_quad_v3] checkpoint: {ckpt}")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=raw.device)
    runner.load(ckpt)
    actor = runner.get_inference_policy(device=raw.device)

    # ---------------------------------------------------------------------------- ONNX --
    onnx_path = os.path.join(args.out_dir, args.onnx_name)
    module = _OnnxPolicy(actor).to("cpu").eval()
    for p in module.parameters():
        p.requires_grad_(False)
    with torch.inference_mode():
        torch.onnx.export(module, (torch.zeros(1, spec.NUM_OBS),), onnx_path, export_params=True, opset_version=15,
                          do_constant_folding=True, input_names=["obs"], output_names=["actions"], dynamic_axes=None,
                          dynamo=False)
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
    print(f"[export_quad_v3] wrote {onnx_path}  io={io}  opset {opset}, IR {model.ir_version}")

    # ---------------------------------------------------------------------- evaluation --
    torso = robot.body_names.index(spec.TORSO)
    joint_ids = [robot.joint_names.index(n) for n in spec.ACTION_ORDER]
    tcfg = SceneEntityCfg("robot", body_names=[spec.TORSO])
    tcfg.resolve(raw.scene)
    tracker = robot.contact_tracker  # substep contact metrics (metrics_enabled was set before gym.make)
    assert tracker is not None and tracker.metrics
    origins = raw.scene.env_origins
    tm = raw.termination_manager
    am = raw.action_manager
    n = raw.num_envs
    dev = raw.device
    mass, g = spec.TOTAL_MASS, 9.81
    settle_steps = int(round(SETTLE_SECONDS / spec.STEP_DT))
    z = lambda: torch.zeros(n, device=dev)  # noqa: E731
    steps, jsum, jpeak, sat, hsum, hpeak = z(), z(), z(), z(), z(), z()
    hmin = torch.full((n,), 1e9, device=dev)
    upsum, upmin = z(), torch.ones(n, device=dev)
    angpeak, rollsum, pitchsum, vzsq = z(), z(), z(), z()
    powsum, vxsum, latsum, arsum = z(), z(), z(), z()
    flightsum = z()
    active = torch.ones(n, dtype=torch.bool, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    fell_touch = torch.zeros(n, dtype=torch.bool, device=dev)
    diverged = torch.zeros(n, dtype=torch.bool, device=dev)
    distance, length = z(), z()
    max_err = 0.0
    t0 = time.time()
    with torch.inference_mode():
        env.reset()
        for v_ in tracker.t_m.values():
            v_.zero_()
        tracker.t_active.fill_(1)
        tracker.t_settled.zero_()
        obs = env.get_observations()
        d = robot.data
        x0 = (d.body_link_pos_w.torch[:, torso, 0] - origins[:, 0]).clone()
        for step in range(EPISODE_STEPS):
            o_np = obs["policy"].detach().cpu().numpy().astype(np.float32)
            onnx_mean = np.concatenate([sess.run(None, {"obs": o_np[i:i + 1]})[0] for i in range(n)], axis=0)
            max_err = max(max_err, float(np.abs(onnx_mean - actor(obs).detach().cpu().numpy()).max()))
            obs, _, dones, _ = env.step(torch.from_numpy(onnx_mean).to(dev))  # the wrapper clips to [-1, 1]
            div_now = tm.get_term("diverged")
            if bool((dones & ~div_now).any()):
                raise RuntimeError(f"a healthy env reset inside the measured window at step {step}")
            a = active.float()
            q = d.body_link_quat_w.torch[:, torso]
            v = d.body_link_lin_vel_w.torch[:, torso]
            w = d.body_link_ang_vel_w.torch[:, torso]
            pos_z = d.body_link_pos_w.torch[:, torso, 2] - origins[:, 2]
            ez = torch.zeros(n, 3, device=dev)
            ez[:, 2] = 1.0
            up = quat_apply(q, ez)[:, 2]
            w_b = quat_apply_inverse(q, w)
            qd = d.joint_vel.torch[:, joint_ids]
            tau = d.applied_torque.torch[:, joint_ids]
            steps += a
            jsum += a * qd.abs().mean(dim=1)
            jpeak = torch.maximum(jpeak, a * qd.abs().amax(dim=1))
            sat += a * (tau.abs() >= spec.FORCE_LIMIT * 0.999).any(dim=1).float()
            hsum += a * pos_z
            hpeak = torch.maximum(hpeak, a * pos_z)
            hmin = torch.where(active, torch.minimum(hmin, pos_z), hmin)
            upsum += a * up
            upmin = torch.where(active, torch.minimum(upmin, up), upmin)
            angpeak = torch.maximum(angpeak, a * torch.linalg.norm(w, dim=1))
            rollsum += a * w_b[:, 0].abs()
            pitchsum += a * w_b[:, 1].abs()
            vzsq += a * v[:, 2] * v[:, 2]
            powsum += a * (tau * qd).abs().sum(dim=1)
            vxsum += a * v[:, 0]
            latsum += a * v[:, 1].abs()
            arsum += a * torch.mean(torch.square(am.action - am.prev_action), dim=1)
            flightsum += a * tracker.flight_step / float(spec.DECIMATION)  # debounced, as the flight reward

            touch_now = touch_this_step(raw)
            fall_now = fallen_mask(raw, tcfg, spec.FALL_UP_DOT, spec.FALL_HEIGHT) & ~div_now
            ending = active & (fall_now | div_now | torch.full_like(active, step == EPISODE_STEPS - 1))
            x_now = d.body_link_pos_w.torch[:, torso, 0] - origins[:, 0]
            distance = torch.where(ending, torch.nan_to_num(x_now - x0), distance)
            length = torch.where(ending, torch.full_like(length, step + 1), length)
            fell |= ending & fall_now
            fell_touch |= ending & fall_now & touch_now
            diverged |= ending & div_now
            active &= ~ending
            # the substeps of the ending step were counted; the tracker stops for worlds that ended
            tracker.t_active.copy_(active.int())
            if step + 1 == settle_steps:
                tracker.t_settled.fill_(1)
    eval_s = time.time() - t0
    keep = (~diverged).cpu().numpy()
    k = ~diverged
    total = steps[k].sum().clamp_min(1.0)
    tmx = tracker.t_m
    sub_total = tmx["sub"][k].sum().clamp_min(1.0)

    def per_step(x):
        return float(x[k].sum() / total)

    def per_sub(x):
        return float(x[k].sum() / sub_total)

    dist = distance.cpu().numpy().astype(np.float64)[keep]
    fell_np = fell.cpu().numpy()[keep]
    len_s = length.cpu().numpy().astype(np.float64)[keep] * spec.STEP_DT
    speed = dist / spec.EPISODE_LENGTH_S
    full = ~fell_np
    mean_vx = per_step(vxsum)
    power = per_step(powsum)
    footfalls = float(tmx["stance_cnt"][k].sum())
    seconds = float(sub_total) * spec.PHYSICS_DT
    impact = float(tmx["peak_force"][k].max())
    action_rate = per_step(arsum)
    plaus = {
        "meanJointSpeed": per_step(jsum),
        "peakJointSpeed": float(jpeak[k].max()),
        "fractionTimeAnyActuatorSaturated": per_step(sat),
        "meanTorsoHeight": per_step(hsum),
        "minTorsoHeight": float(hmin[k].min()),
        "peakTorsoHeight": float(hpeak[k].max()),
        "meanTorsoUp": per_step(upsum),
        "minTorsoUp": float(upmin[k].min()),
        "peakTorsoAngularSpeed": float(angpeak[k].max()),
        "meanAbsRollRate": per_step(rollsum),
        "meanAbsPitchRate": per_step(pitchsum),
        "rmsTorsoVerticalSpeed": per_step(vzsq) ** 0.5,
        "meanFootSlipSpeedInStance": float(tmx["slip"][k].sum() / tmx["feet"][k].sum().clamp_min(1.0)),
        "meanFeetInContact": per_sub(tmx["feet"]),
        "airborneFraction": per_sub(tmx["air"]),
        "flightFractionDebounced": per_step(flightsum),
        "meanStanceDurationS": float(tmx["stance_sum"][k].sum()) / footfalls * spec.PHYSICS_DT if footfalls > 0 else None,
        "footfallsPerSecond": footfalls / seconds if seconds > 0 else None,
        "paidTouchdownsPerSecond": float(tmx["touchdowns"][k].sum()) / seconds if seconds > 0 else None,
        "peakFootImpactN": impact,
        "peakFootImpactBodyWeights": impact / (mass * g),
        "peakFootImpactNote": f"max over feet and 5 ms substeps of one foot's floor normal force, excluding the first "
                              f"{SETTLE_SECONDS:g} s after reset (spawn drop); standing = 0.25",
        "peakFootImpactBodyWeightsInclSpawn": float(tmx["peak_spawn"][k].max()) / (mass * g),
        "meanPeakForcePerFootfallBodyWeights": (float(tmx["footfall_peak_sum"][k].sum()) / footfalls / (mass * g)
                                                if footfalls > 0 else None),
        "contactSampling": "every physics substep (5 ms), graphed tracker (quad_tasks_v3/contact_tracker.py)",
        "fallsByTorsoOrThighTouch": int(fell_touch[k].sum()),
        "meanForwardSpeed": mean_vx,
        "meanLateralSpeed": per_step(latsum),
        "meanActionRate": action_rate,
        "meanActionRate50HzEquivalent": action_rate * (0.02 / spec.STEP_DT) ** 2,
        "meanAbsMechanicalPowerW": power,
        "costOfTransport": power / (mass * g * mean_vx) if mean_vx > 0.05 else None,
        "meanEpisodeSeconds": float(steps[k].mean() * spec.STEP_DT),
        "forceLimit": spec.FORCE_LIMIT,
    }
    print(f"[export_quad_v3] {int(keep.sum())}/{n} healthy episodes: speed {speed.mean():+.4f} +/- {speed.std():.4f} m/s, "
          f"distance {dist.mean():+.3f} m, fall rate {fell_np.mean():.0%}, eval {eval_s:.0f} s")
    print("[export_quad_v3] plausibility: " + ", ".join(f"{a} {b:.4g}" if isinstance(b, float) else f"{a} {b}" for a, b in plaus.items()))
    print(f"[export_quad_v3] onnxruntime vs torch on {EPISODE_STEPS * n} observations: max|diff| = {max_err:.3e}")

    # ------------------------------------------------------------------------ reports --
    run_dir = os.path.dirname(ckpt)
    summary = {}
    if os.path.isfile(os.path.join(run_dir, "train_summary.json")):
        with open(os.path.join(run_dir, "train_summary.json"), "r", encoding="utf-8") as f:
            summary = json.load(f)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    ckpt_iter = int(ck.get("iter", -1))
    steps_at_ckpt = (ckpt_iter + 1) * agent_cfg.num_steps_per_env * summary["numEnvs"] if summary.get("numEnvs") else None
    report = {
        "tool": "isaaclab3-newton-mjwarp-rsl_rl",
        "meanSpeed": float(speed.mean()),
        "stdSpeed": float(speed.std()),
        "meanDistance20s": float(dist.mean()),
        "fallRate": float(fell_np.mean()),
        "stepsTrained": steps_at_ckpt,
        "wallMinutes": summary.get("wallMinutes"),
        "onnxMaxAbsErr": max_err,
        "notes": (
            f"Isaac Lab {ver('isaaclab')} (v3.0.0-beta2.patch1), kit-less: Newton {ver('newton')} with the MuJoCo-Warp solver "
            f"(mujoco-warp {ver('mujoco-warp')}, warp {ver('warp-lang')}; Newton method, 10/8 iterations, implicitfast, "
            f"pyramidal, dt 0.005 s), rsl-rl-lib {ver('rsl-rl-lib')}, torch {ver('torch')}. {n} deterministic 20 s episodes, "
            f"seed {env_cfg.seed}, spec reset, no randomisation, no pushes, actions = clip(ONNX mean, -1, 1); speed = torso x "
            "displacement until the episode ends (timeout or fall) / 20 s; health-guard episodes excluded. stepsTrained = "
            "environment steps up to this checkpoint; wallMinutes = the whole training run."
        ),
        "simulator": "newton-mujoco_warp",
        "creature": "quad",
        "episodes": n,
        "episodesDiverged": int((~keep).sum()),
        "episodesFell": int(fell_np.sum()),
        "meanSpeedNoFall": float(speed[full].mean()) if full.any() else None,
        "meanTimeToFall": float(len_s[fell_np].mean()) if fell_np.any() else None,
        "stdDistance20s": float(dist.std()),
        "meanInstantaneousSpeedX": mean_vx,
        "minSpeed": float(speed.min()),
        "maxSpeed": float(speed.max()),
        "onnx": os.path.relpath(onnx_path, REPO).replace("\\", "/"),
        "checkpoint": os.path.relpath(ckpt, REPO).replace("\\", "/"),
        "checkpointIteration": ckpt_iter,
        "evaluatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plausibility": plaus,
        "details": {"onnx": {"opset": opset, "irVersion": model.ir_version, "io": io}, "training": summary},
    }
    with open(os.path.join(args.out_dir, args.report_name), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    om = raw.observation_manager
    layout, start = [], 0
    for name, dim in zip(om.active_terms["policy"], om.group_obs_term_dim["policy"]):
        layout.append({"term": name, "start": start, "size": int(np.prod(dim))})
        start += int(np.prod(dim))
    act_ids = am.get_term("joint_pos")._joint_ids
    act_ids = list(range(robot.num_joints)) if isinstance(act_ids, slice) else [int(i) for i in act_ids]
    rig_order = {
        "tool": "isaaclab3-newton",
        "actionOrder": spec.ACTION_ORDER_MJCF,
        "actionOrderUsd": spec.ACTION_ORDER,
        "simJointOrder": list(robot.joint_names),
        "actionToSimJointIndex": act_ids,
        "simBodyOrder": list(robot.body_names),
        "referenceBody": spec.TORSO_MJCF,
        "actionScaleRad": spec.ACTION_SCALE,
        "obsLayout": layout,
        "note": "Observation joint terms and actions are in actionOrder (quad_rig.json). USD names replace '-' by 'm'.",
    }
    with open(os.path.join(args.out_dir, args.rig_order_name), "w", encoding="utf-8") as f:
        json.dump(rig_order, f, indent=2)
    print(f"[export_quad_v3] wrote {args.report_name} and {args.rig_order_name} in {args.out_dir}")
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
