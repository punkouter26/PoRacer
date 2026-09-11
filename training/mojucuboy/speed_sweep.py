"""K2 across the commanded speed RANGE, not at a single point.

  .venv-mjwarp\\Scripts\\python.exe training/mojucuboy/speed_sweep.py --run e15_speedcmd

WHY THIS EXISTS. K2 ("velocity tracking within +/-10 %") has been evaluated at a
single commanded speed of 1.5 m/s for this project's whole history, which is also
the only speed any policy was ever trained at. That makes the KPI unfalsifiable in
the direction that matters: a policy which ignores the command entirely and simply
runs at 1.5 m/s scores a perfect 0 % error, and one that genuinely tracks scores
the same. M11 measured the difference -- E12 reads -1.1 % at 1.5 m/s and collapses
to 0.325 m/s when commanded 2.0, a -84 % error that the single-point KPI cannot see.

So the honest form of K2 is the error across the commanded range, and the honest
summary is the WORST point in it, not the mean: a racer that tracks three speeds
and falls over at the fourth is not a racer that tracks.

Reports per-command achieved speed, signed error, uptime, and the aggregate
worst-case, which is the number a command-following claim should be judged on.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import mojucuboy_env  # noqa: E402
from gate4_eval import evaluate  # noqa: E402
from train_mojucuboy import RESULTS, ActorCritic  # noqa: E402

TRACKING_GATE = 0.10          # K2: +/-10 %
UPTIME_GATE = 0.90            # K1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--onnx", type=str, default=None,
                        help="Evaluate an .onnx instead of policy.pt (for shipped brains).")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint filename within the run dir, e.g. policy_best.pt or "
                             "a snapshot. Defaults to policy.pt -- which is the LAST "
                             "iteration, not the best (M15).")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--speeds", type=float, nargs="+",
                        default=[1.0, 1.5, 2.0, 2.4])
    parser.add_argument("--reset-fallen", type=float, default=0.0,
                        help="Default 0: racers spawn upright in the game, so that is "
                             "the condition a shipping comparison belongs in.")
    args = parser.parse_args()

    run_dir = RESULTS / args.run
    if args.onnx or not (run_dir / (args.checkpoint or "policy.pt")).exists():
        from gate4_eval_onnx import OnnxPolicy
        onnx_path = Path(args.onnx) if args.onnx else sorted(run_dir.glob("*.onnx"))[0]
        policy = OnnxPolicy(onnx_path)
        source = onnx_path.name
    else:
        ck_path = run_dir / (args.checkpoint or "policy.pt")
        checkpoint = torch.load(ck_path, map_location="cuda:0",
                                weights_only=False)
        policy = ActorCritic().to("cuda:0")
        policy.load_state_dict(checkpoint["model"])
        policy.eval()
        source = f"{ck_path.name} @ iter {checkpoint.get('iteration', '?')}"

    print(f"run {args.run} ({source}), {args.episodes} episodes per speed, "
          f"reset_fallen={args.reset_fallen}\n")
    print(f"{'cmd m/s':>8} {'achieved':>9} {'error':>8} {'uptime':>7} {'fallen':>7}  K2")
    rows = []
    worst = 0.0
    for speed in args.speeds:
        stats = evaluate(policy, args.episodes, args.seed, True,
                         reset_fallen=args.reset_fallen, command_speed=speed)
        achieved = stats["mean_forward_speed"]
        error = (achieved - speed) / speed
        worst = max(worst, abs(error))
        ok = "pass" if abs(error) <= TRACKING_GATE else "FAIL"
        print(f"{speed:8.2f} {achieved:9.3f} {error:+7.1%} "
              f"{stats['mean_uptime']:7.3f} {stats['mean_fallen']:7.3f}  {ok}")
        rows.append({"command": speed, "achieved": achieved, "error": error,
                     "uptime": stats["mean_uptime"], "fallen": stats["mean_fallen"]})

    mean_uptime = sum(r["uptime"] for r in rows) / len(rows)
    print(f"\n  worst |error| across range : {worst:.1%}   (K2 gate {TRACKING_GATE:.0%})")
    print(f"  mean uptime across range  : {mean_uptime:.3f}   (K1 gate {UPTIME_GATE})")
    verdict = "PASS" if worst <= TRACKING_GATE and mean_uptime >= UPTIME_GATE else "FAIL"
    print(f"  VERDICT: {verdict}  -- a command-following policy must hold BOTH at every "
          f"commanded speed, not on average")

    # Named after the checkpoint, so sweeping a snapshot and the final policy of the
    # same run does not clobber one with the other -- which is the whole point of
    # being able to compare last against best (M15).
    stem = Path(source.split(" @ ")[0]).stem
    out = run_dir / f"speed_sweep_{stem}.json"
    out.write_text(json.dumps(
        {"rows": rows, "worst_abs_error": worst, "mean_uptime": mean_uptime,
         "verdict": verdict}, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
