"""Gate 4 / K10 evaluation of an ONNX policy, for brains with no `policy.pt`.

  .venv-mjwarp\\Scripts\\python.exe training/mojucuboy/gate4_eval_onnx.py --run boy_chase01

WHY THIS EXISTS. `boy_chase01` -- the brain that actually ships in the game -- has
no torch checkpoint. Its `.onnx` IS the artifact (see PROTECTED_GLOBS in
train_mojucuboy.py, and the note there about a training run having once deleted
it). So `gate4_eval.py`, which loads `policy.pt`, cannot measure the one policy
whose behaviour the project's shipping decision rests on.

That matters right now because session 1's summary asserts the shipped brain
"can get up from a sprawl: yes (races unaided)" while every post-M9 policy
cannot -- and that claim was never produced by an instrument. It came from
watching races. K10 exists precisely because watching could not separate "landed
upright" from "stood back up", so the shipped brain deserves the same scrutiny
as the challengers before any of them is judged against it.

The ONNX graph is the deterministic forward pass WITH observation normalisation
baked in (see ActorCritic.forward), so it is a drop-in for the `policy` callable
`evaluate()` expects. The only cost is a host round-trip per step, which is fine
for a 100-episode evaluation and irrelevant to the result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mojucuboy_env  # noqa: E402
from gate4_eval import evaluate  # noqa: E402
from train_mojucuboy import RESULTS  # noqa: E402


class OnnxPolicy:
    """Adapts an ONNX session to the `policy(obs) -> action` callable."""

    def __init__(self, path: Path):
        import onnx
        import onnxruntime as ort
        # The shipped graph is exported with a FIXED batch of 1, which is right for
        # Unity -- it runs one racer -- and wrong for a 100-world evaluation. The
        # network is a plain MLP and therefore batch-agnostic, so relax the leading
        # dimension to a symbolic one rather than calling the session 100,000 times.
        # Nothing about the weights or the computation changes.
        model = onnx.load(str(path))
        for tensor in list(model.graph.input) + list(model.graph.output):
            dim0 = tensor.type.tensor_type.shape.dim[0]
            dim0.ClearField("dim_value")
            dim0.dim_param = "batch"
        # CPU is deliberate. The GPU is busy integrating 100 worlds of MuJoCo and
        # the policy is a 75->128x3->21 MLP; contending for the device to save
        # microseconds on a 1000-step evaluation would be a poor trade.
        self.session = ort.InferenceSession(model.SerializeToString(),
                                            providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        raw = obs.detach().cpu().numpy().astype(np.float32)
        out = self.session.run([self.output_name], {self.input_name: raw})[0]
        return torch.from_numpy(out).to(obs.device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--onnx", type=str, default=None,
                        help="Explicit .onnx path; defaults to the only one in the run dir.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--sprawl-tilt", type=float, default=180.0)
    parser.add_argument("--reset-fallen", type=float, default=None)
    parser.add_argument("--command-speed", type=float, default=None,
                        help="Override the commanded speed in m/s.")
    args = parser.parse_args()

    run_dir = RESULTS / args.run
    if args.onnx:
        onnx_path = Path(args.onnx)
    else:
        found = sorted(run_dir.glob("*.onnx"))
        if not found:
            print(f"no .onnx in {run_dir}")
            return 2
        onnx_path = found[0]

    print(f"run {args.run}, ONNX {onnx_path.name}, {args.episodes} deterministic episodes, "
          f"sprawl tilt {args.sprawl_tilt:.0f} deg\n")
    policy = OnnxPolicy(onnx_path)

    results = {}
    for label, randomise in (("randomised", True), ("nominal", False)):
        stats = evaluate(policy, args.episodes, args.seed, randomise,
                         sprawl_tilt_deg=args.sprawl_tilt, reset_fallen=args.reset_fallen,
                         command_speed=args.command_speed)
        results[label] = stats
        print(f"=== {label.upper()} MODEL ===")
        print(f"  mean forward speed  : {stats['mean_forward_speed']:7.3f} m/s")
        print(f"  mean uptime         : {stats['mean_uptime']:7.3f}")
        print(f"  mean fallen         : {stats['mean_fallen']:7.3f}")
        print(f"  episodes >90% up    : {stats['uptime_over_90pct']:7.1%}")
        print(f"  mean return         : {stats['mean_return']:7.2f}")
        # Same K4-K8 block as gate4_eval, because a shipped brain has to be
        # judged on the same dashboard as its challengers or the comparison is
        # not one. These were computed here all along and simply not printed.
        print(f"  -- K4-K8 --")
        print(f"  K4 heading err      : {stats['heading_err_deg']:7.1f} deg   (< 15)")
        print(f"  K5 lateral drift    : {stats['lateral_drift']:7.3f} m/s   (< 0.25)")
        print(f"  K6 control effort   : {stats['ctrl_effort']:7.4f}  (commanded angle)")
        print(f"  K6b applied torque  : {stats['torque_abs']:7.3f} N.m mean abs")
        print(f"  K6b torque squared  : {stats['torque_sq']:7.4g}")
        print(f"  K7 joint accel      : {stats['joint_accel']:7.3e}")
        print(f"  K8 uprightness      : {stats['uprightness']:7.3f}       (> 0.90)")
        print(f"  -- K10 recovery --")
        print(f"  started down        : {stats['started_down_count']:4d} / {stats['episodes']}"
              f"   ({stats['started_down_frac']:.1%})")
        print(f"  recovery rate       : {stats['recovery_rate']:7.1%}")
        print(f"  ever stood again    : {stats['ever_stood_again'] if 'ever_stood_again' in stats else stats['ever_stood_after_down']:7.1%}")
        print(f"  median steps to up  : {stats['median_steps_to_stand']:7.1f}\n")

    out = run_dir / f"gate4_k10_onnx_tilt{int(args.sprawl_tilt)}.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
