"""Evaluate ANY ONNX policy with a creature's interface in MuJoCo Warp (lifted from
training/worm/mujoco/eval_worm_mujoco.py, generalised to creatures that can fall).

Protocol (WORM_SPEC items 10 and 15): N deterministic episodes of episode_steps policy
steps, all in parallel, actions = clip(ONNX mean, -1, 1), the spec reset with a fixed
seed (12345), NO domain randomisation and NO pushes (unless --pushes).

Each world's FIRST episode is what counts. An episode ends at the timeout, at a task
termination (a fall: the racer is out, like a DNF, and its distance stops there) or at
the simulator-health guard (excluded from every statistic and reported). Speed of an
episode = reference-body x displacement until its end / the full episode duration, so a
fall costs the distance it did not run. meanSpeed/stdSpeed are over non-diverged
episodes; meanSpeedNoFall is over the episodes that ran the full duration.

The plausibility block (rule I) comes from a creature plug-in with
    reset(env) / accumulate(env, info, active) / summary(env, kept) -> dict,
where `active` masks the worlds whose first episode is still running, and optionally
    substep(env)
called after every physics substep (the env then steps eagerly instead of replaying
its CUDA graph), for contact metrics that a once-per-policy-step sample would alias
(stance duration, airtime, peak impacts).

Writes <export_dir>/<name>_<tag>_report.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]


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


def main(name: str, make_env: Callable[..., object], make_plausibility: Callable[[object], object],
         export_dir: Path, argv=None) -> int:
    parser = argparse.ArgumentParser(description=f"Evaluate a {name} ONNX policy in MuJoCo Warp")
    parser.add_argument("--onnx", type=str, default=str(export_dir / f"{name}_mujoco.onnx"))
    parser.add_argument("--tag", type=str, default="mujoco",
                        help=f"names the report: {name}_<tag>_report.json")
    parser.add_argument("--tool", type=str, default=None,
                        help="the TRAINING tool recorded in the report (default: ONNX metadata, "
                             "else the tag)")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--pushes", action="store_true", help="apply the training pushes too")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="the torch checkpoint the ONNX came from, for the parity check")
    parser.add_argument("--steps-trained", type=int, default=None)
    parser.add_argument("--wall-minutes", type=float, default=None)
    parser.add_argument("--onnx-max-abs-err", type=float, default=None,
                        help="parity figure to record when no torch checkpoint is available")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args(argv)

    import onnx
    import onnxruntime as ort

    onnx_path = resolve(args.onnx)
    model = onnx.load(str(onnx_path))
    meta = {p.key: p.value for p in model.metadata_props}
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inp, out = session.get_inputs()[0], session.get_outputs()[0]

    n = args.episodes
    env = make_env(n, args.seed, pushes=args.pushes)
    notes = []
    if inp.name != "obs" or out.name != "actions":
        notes.append(f"ONNX names are {inp.name!r} -> {out.name!r}, spec says 'obs' -> 'actions'")
    in_shape, out_shape = list(inp.shape), list(out.shape)
    if in_shape[-1] != env.obs_size or out_shape[-1] != env.action_size:
        print(f"ERROR: ONNX shape {in_shape} -> {out_shape}, expected [1, {env.obs_size}] -> "
              f"[1, {env.action_size}]")
        return 2
    batch_one = isinstance(in_shape[0], int) and in_shape[0] == 1

    plaus = make_plausibility(env)
    plaus.reset(env)
    if hasattr(plaus, "substep"):                  # contact metrics sampled every substep
        env.substep_callback = plaus.substep
    dev = env.device
    active = torch.ones(n, dtype=torch.bool, device=dev)
    distance = torch.zeros(n, device=dev)
    length = torch.zeros(n, device=dev)
    fell = torch.zeros(n, dtype=torch.bool, device=dev)
    diverged = torch.zeros(n, dtype=torch.bool, device=dev)
    speed_inst = torch.zeros(n, device=dev)
    obs = env.observation()
    all_obs, all_act = [], []
    started = time.perf_counter()
    for step in range(env.episode_steps):
        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        if batch_one:
            act_np = np.concatenate([session.run([out.name], {inp.name: obs_np[i:i + 1]})[0]
                                     for i in range(n)], axis=0)
        else:
            act_np = session.run([out.name], {inp.name: obs_np})[0]
        if step % 10 == 0:
            all_obs.append(obs_np)
            all_act.append(act_np)
        obs, _, done, timeouts, info = env.step(torch.as_tensor(act_np, device=dev))
        plaus.accumulate(env, info, active)
        ending = done & active
        distance = torch.where(ending, info["ep_distance"], distance)
        length = torch.where(ending, info["ep_length"], length)
        speed_inst = torch.where(ending, info["ep_speed"], speed_inst)
        fell |= ending & info["terminated"]
        diverged |= ending & info["diverged"]
        active &= ~done
    seconds = time.perf_counter() - started
    duration = env.episode_steps * env.dt
    if active.any():                                   # cannot happen: every world times out
        notes.append(f"{int(active.sum())} episodes had not ended after {duration:.0f} s")

    keep_t = ~diverged
    keep = keep_t.cpu().numpy()
    if not keep.any():
        print("ERROR: every episode tripped the simulator-health guard")
        return 1
    if not keep.all():
        notes.append(f"{int((~keep).sum())} of {n} episodes tripped the simulator-health guard "
                     f"(non-finite or |qvel| > {env.cfg.divergence_qvel:g}) and are excluded")
    dist_np = distance.cpu().numpy().astype(np.float64)[keep]
    fell_np = fell.cpu().numpy()[keep]
    len_np = length.cpu().numpy().astype(np.float64)[keep] * env.dt
    speed = dist_np / duration
    full = ~fell_np
    plausibility = plaus.summary(env, keep_t)

    # ---- parity against torch
    err = args.onnx_max_abs_err
    checkpoint = args.checkpoint or meta.get("checkpoint")
    parity_note = ("no torch checkpoint available; onnxMaxAbsErr as supplied" if err is not None
                   else "no torch checkpoint available; parity not measured")
    if checkpoint:
        ck_path = resolve(checkpoint)
        if ck_path.exists():
            from .export import load_policy
            policy, _ = load_policy(ck_path, env.obs_size, env.action_size)
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
                    f"(mujoco-warp {mjwarp_version()}), spec reset, seed {args.seed}, no domain "
                    f"randomisation, {'with' if args.pushes else 'no'} pushes; speed = "
                    f"{env.cfg.reference_body} x displacement until the episode ends (timeout or "
                    f"fall) / {duration:.0f} s")

    report = {
        "tool": tool,
        "meanSpeed": float(speed.mean()),
        "stdSpeed": float(speed.std()),
        "meanDistance20s": float(dist_np.mean()),
        "fallRate": float(fell_np.mean()),
        "stepsTrained": steps_trained,
        "wallMinutes": wall_minutes,
        "onnxMaxAbsErr": err,
        "notes": "; ".join(notes),
        "simulator": "mujoco_warp",
        "creature": name,
        "episodes": n,
        "episodesDiverged": int((~keep).sum()),
        "episodesFell": int(fell_np.sum()),
        "meanSpeedNoFall": float(speed[full].mean()) if full.any() else None,
        "meanTimeToFall": float(len_np[fell_np].mean()) if fell_np.any() else None,
        "stdDistance20s": float(dist_np.std()),
        "meanInstantaneousSpeedX": float(speed_inst.cpu().numpy()[keep].mean()),
        "minSpeed": float(speed.min()),
        "maxSpeed": float(speed.max()),
        "onnx": onnx_path.relative_to(REPO).as_posix() if onnx_path.is_relative_to(REPO)
        else str(onnx_path),
        "onnxMetadata": meta,
        "evaluatedAt": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "plausibility": plausibility,
    }
    out_path = resolve(args.out) if args.out else export_dir / f"{name}_{args.tag}_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))
    print(f"{n} episodes x {duration:.0f} s in {seconds:.1f} s wall")
    print(f"mean speed {report['meanSpeed']:.4f} +- {report['stdSpeed']:.4f} m/s   "
          f"distance {report['meanDistance20s']:.3f} m   fall rate {report['fallRate']:.0%}   "
          f"onnx max|err| {err if err is None else f'{err:.2e}'}")
    print("plausibility: " + ", ".join(f"{k} {v:.4g}" if isinstance(v, float) else f"{k} {v}"
                                       for k, v in plausibility.items()))
    print(f"report: {out_path}")
    return 0
