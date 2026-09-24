"""Evaluate ANY Worm5 ONNX policy in MuJoCo (the WORM_SPEC.md interface).

  # the MuJoCo-trained worm (defaults)
  .venv-mjwarp\\Scripts\\python.exe training/worm/mujoco/eval_worm_mujoco.py

  # cross-simulator check: the Isaac-trained worm, driven in MuJoCo
  .venv-mjwarp\\Scripts\\python.exe training/worm/mujoco/eval_worm_mujoco.py ^
      --onnx training/worm/export/worm_isaac.onnx --tag isaac_in_mujoco --tool isaac

Protocol: N deterministic 20 s episodes (1000 policy steps, the actor MEAN clipped
to [-1, 1]) in MJWarp, all run in parallel, with the spec's reset (head at the
origin, yaw U(+-45 deg), joint noise U(+-0.05 rad), fixed seed) and NO domain
randomisation (nominal friction, masses and kp).

Speed of one episode = x displacement of segment 2 over the 20 s / 20 s.
meanSpeed / stdSpeed are over episodes. meanDistance20s is the mean displacement.

ONNX-vs-torch parity: needs the torch checkpoint the ONNX was exported from. It is
found automatically from the ONNX metadata written by train_worm_mujoco.py, or
given with --checkpoint. For a foreign ONNX (e.g. Isaac) without a checkpoint in
this trainer's format, onnxMaxAbsErr is taken from --onnx-max-abs-err or null.

Writes training/worm/export/worm_<tag>_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from worm_env import ACTION_SIZE, EPISODE_STEPS, OBS_SIZE, WormEnv  # noqa: E402

EXPORT_DIR = HERE.parent / "export"


def resolve(path: str | Path) -> Path:
    """Absolute as given; relative to the cwd if that exists, else to the repo root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return p.resolve() if p.exists() else REPO / p


def mjwarp_version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("mujoco-warp")
    except PackageNotFoundError:
        return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a Worm5 ONNX policy in MuJoCo Warp")
    parser.add_argument("--onnx", type=str, default=str(EXPORT_DIR / "worm_mujoco.onnx"))
    parser.add_argument("--tag", type=str, default="mujoco",
                        help="names the report: export/worm_<tag>_report.json")
    parser.add_argument("--tool", type=str, default=None,
                        help="the TRAINING tool recorded in the report (default: ONNX metadata, "
                             "else the tag)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="policy.pt the ONNX came from, for the parity check")
    parser.add_argument("--steps-trained", type=int, default=None)
    parser.add_argument("--wall-minutes", type=float, default=None)
    parser.add_argument("--onnx-max-abs-err", type=float, default=None,
                        help="parity figure to record when no torch checkpoint is available")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    import onnx
    import onnxruntime as ort

    onnx_path = resolve(args.onnx)
    model = onnx.load(str(onnx_path))
    meta = {p.key: p.value for p in model.metadata_props}
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]
    notes = []
    if inp.name != "obs" or out.name != "actions":
        notes.append(f"ONNX names are {inp.name!r} -> {out.name!r}, spec says 'obs' -> 'actions'")
    in_shape, out_shape = list(inp.shape), list(out.shape)
    if in_shape[-1] != OBS_SIZE or out_shape[-1] != ACTION_SIZE:
        print(f"ERROR: ONNX shape {in_shape} -> {out_shape}, expected [1, {OBS_SIZE}] -> "
              f"[1, {ACTION_SIZE}]")
        return 2
    batch_one = isinstance(in_shape[0], int) and in_shape[0] == 1

    n = args.episodes
    env = WormEnv(n, seed=args.seed, randomize=False, random_initial_episode=False)
    obs = env.observation()
    all_obs, all_act = [], []
    diverged = torch.zeros(n, dtype=torch.bool, device=env.device)
    # Plausibility metrics (AGENTS rule I), accumulated on the GPU per world.
    roll_sum = torch.zeros(n, device=env.device)
    roll_peak = torch.zeros(n, device=env.device)
    joint_peak = torch.zeros(n, device=env.device)
    saturated = torch.zeros(n, device=env.device)
    height_sum = torch.zeros(n, device=env.device)
    started = time.perf_counter()
    for step in range(EPISODE_STEPS):
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        if batch_one:
            act_np = np.concatenate([session.run([out.name], {inp.name: obs_np[i:i + 1]})[0]
                                     for i in range(n)], axis=0)
        else:
            act_np = session.run([out.name], {inp.name: obs_np})[0]
        if step % 10 == 0:
            all_obs.append(obs_np)
            all_act.append(act_np)
        obs, _, done, timeouts, info = env.step(torch.as_tensor(act_np, device=env.device))
        diverged |= info["diverged"]
        roll_sum += info["roll_rate"]
        roll_peak = torch.maximum(roll_peak, info["roll_rate"])
        joint_peak = torch.maximum(joint_peak, env.qvel[:, env.dof_addr].abs().amax(dim=1))
        saturated += (env.actuator_force.abs() >= env.force_limit * 0.999).any(dim=1).float()
        height_sum += env.xpos[:, env.segment_bodies, 2].mean(dim=1)
        if step == EPISODE_STEPS - 1:
            distance = info["ep_distance"].cpu().numpy().astype(np.float64)
            speed_inst = info["ep_speed"].cpu().numpy().astype(np.float64)
    seconds = time.perf_counter() - started
    duration = EPISODE_STEPS * env.dt                                    # 20 s
    keep = ~diverged.cpu().numpy()
    if not keep.any():
        print("ERROR: every episode tripped the simulator-health guard")
        return 1
    if not keep.all():
        notes.append(f"{int((~keep).sum())} of {n} episodes tripped the simulator-health guard "
                     f"(non-finite or |qvel| > 500) and are excluded from the statistics")
    distance, speed_inst = distance[keep], speed_inst[keep]
    kept = torch.as_tensor(keep, device=env.device)
    plaus = {
        "meanAbsRollRateSeg2": float((roll_sum[kept] / EPISODE_STEPS).mean()),
        "peakAbsRollRateSeg2": float(roll_peak[kept].max()),
        "peakJointSpeed": float(joint_peak[kept].max()),
        "fractionTimeAnyActuatorSaturated": float((saturated[kept] / EPISODE_STEPS).mean()),
        "meanSegmentHeight": float((height_sum[kept] / EPISODE_STEPS).mean()),
        "forceLimit": env.force_limit,
    }
    speed = distance / duration

    # ---- parity against torch
    err = args.onnx_max_abs_err
    checkpoint = args.checkpoint or meta.get("checkpoint")
    parity_note = "no torch checkpoint available; onnxMaxAbsErr as supplied" \
        if err is not None else "no torch checkpoint available; parity not measured"
    if checkpoint:
        ck_path = resolve(checkpoint)
        if ck_path.exists():
            from train_worm_mujoco import load_policy
            policy = load_policy(ck_path)
            obs_all = np.concatenate(all_obs, axis=0)
            with torch.no_grad():
                ref = policy(torch.from_numpy(obs_all)).numpy()
            err = float(np.abs(ref - np.concatenate(all_act, axis=0)).max())
            parity_note = f"parity over {obs_all.shape[0]} visited observations vs {ck_path.name}"
        else:
            parity_note = f"checkpoint {checkpoint} not found; parity not measured"
    notes.append(parity_note)

    def meta_num(key, cast):
        try:
            return cast(meta[key])
        except (KeyError, ValueError):
            return None

    steps_trained = args.steps_trained if args.steps_trained is not None \
        else meta_num("stepsTrained", int)
    wall_minutes = args.wall_minutes if args.wall_minutes is not None \
        else meta_num("wallMinutes", float)
    tool = args.tool or meta.get("tool") or args.tag
    notes.insert(0, f"{n} deterministic {duration:.0f} s episodes in MuJoCo Warp "
                    f"(mujoco-warp {mjwarp_version()}), "
                    f"spec reset, seed {args.seed}, no domain randomisation; speed = segment-2 "
                    f"x displacement / {duration:.0f} s")

    report = {
        "tool": tool,
        "meanSpeed": float(speed.mean()),
        "stdSpeed": float(speed.std()),
        "meanDistance20s": float(distance.mean()),
        "stepsTrained": steps_trained,
        "wallMinutes": wall_minutes,
        "onnxMaxAbsErr": err,
        "notes": "; ".join(notes),
        # extras, not in the shared schema
        "simulator": "mujoco_warp",
        "episodes": n,
        "episodesDiverged": int((~keep).sum()),
        "stdDistance20s": float(distance.std()),
        "meanInstantaneousSpeedX": float(speed_inst.mean()),
        "minSpeed": float(speed.min()),
        "maxSpeed": float(speed.max()),
        "onnx": onnx_path.relative_to(REPO).as_posix() if onnx_path.is_relative_to(REPO)
        else str(onnx_path),
        "evaluatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plausibility": plaus,
    }
    out_path = resolve(args.out) if args.out else EXPORT_DIR / f"worm_{args.tag}_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"{n} episodes x {duration:.0f} s in {seconds:.1f} s wall")
    print(f"mean speed {report['meanSpeed']:.4f} +- {report['stdSpeed']:.4f} m/s   "
          f"distance {report['meanDistance20s']:.3f} m in 20 s   "
          f"onnx max|err| {err if err is None else f'{err:.2e}'}")
    print("plausibility: " + ", ".join(f"{k} {v:.4g}" for k, v in plaus.items()))
    print(f"report: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
