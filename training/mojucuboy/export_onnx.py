"""Export the trained policy to ONNX, and emit the reference trajectory.

  .venv-mjwarp\\Scripts\\python.exe training/mojucuboy/export_onnx.py --run boy_chase01

Two artefacts:

  MojucuBoy_v01.onnx        opset 15, fixed batch 1, observation normaliser baked
                         INTO the graph. Unity feeds raw observations and gets
                         actions back -- there are no normalisation statistics
                         for the C# side to hold, get wrong, or forget to update.

  mujoco_reference.json  a deterministic execution trajectory: observation and
                         action at each of N policy steps from a fixed seed. The
                         Unity controller replays the same observations through
                         Inference Engine and must reproduce the same actions.
                         This is what turns "the ONNX loaded" into "the ONNX
                         computes what Python computed".

The exported graph is the DETERMINISTIC policy (tanh of the actor mean), not a
sample from the distribution: the racer must behave the same way every run.
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

from mojucuboy_env import ACTION_SIZE, OBS_SIZE  # noqa: E402
from train_mojucuboy import RESULTS, ActorCritic  # noqa: E402

REFERENCE_STEPS = 128


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--opset", type=int, default=15)
    args = parser.parse_args()

    run_dir = RESULTS / args.run
    checkpoint = torch.load(run_dir / "policy.pt", map_location="cpu", weights_only=False)
    policy = ActorCritic()
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    onnx_path = run_dir / "MojucuBoy_v01.onnx"
    dummy = torch.zeros(1, OBS_SIZE)
    torch.onnx.export(
        policy, (dummy,), str(onnx_path),
        input_names=["obs"], output_names=["action"],
        opset_version=args.opset,
        dynamo=False,
        # Fixed batch 1 on purpose: Unity evaluates one racer per call, and a
        # static shape lets Inference Engine plan the whole graph up front.
        dynamic_axes=None,
    )
    print(f"wrote {onnx_path}  (opset {args.opset}, batch 1)")

    # Prove the normaliser really is inside the graph: a graph without it would
    # give identical outputs for raw and pre-normalised inputs.
    import onnx
    model = onnx.load(str(onnx_path))
    initialiser_names = {i.name for i in model.graph.initializer}
    ops = [n.op_type for n in model.graph.node]
    print(f"  nodes: {len(ops)}  ops: {sorted(set(ops))}")
    print(f"  initialisers: {len(initialiser_names)}")

    mean = policy.norm.mean.detach().numpy()
    var = policy.norm.var.detach().numpy()
    baked = any(np.allclose(onnx.numpy_helper.to_array(i), mean, atol=1e-6)
                for i in model.graph.initializer
                if onnx.numpy_helper.to_array(i).shape == mean.shape)
    print(f"  normaliser mean baked into graph: {baked}")
    print(f"  obs mean range [{mean.min():+.3f}, {mean.max():+.3f}]  "
          f"var range [{var.min():.4f}, {var.max():.4f}]")

    # Reference trajectory: closed loop in the real environment so the states are
    # ones the policy actually visits, not synthetic noise.
    import onnxruntime as ort
    from mojucuboy_env import MojucuBoyEnv

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    env = MojucuBoyEnv(1, seed=12345)
    obs = env.observation()

    records, max_delta = [], 0.0
    with torch.no_grad():
        for _ in range(REFERENCE_STEPS):
            obs_np = obs.detach().cpu().numpy().astype(np.float32)
            torch_action = policy(obs.cpu()).numpy()
            onnx_action = session.run(["action"], {"obs": obs_np})[0]
            max_delta = max(max_delta, float(np.max(np.abs(torch_action - onnx_action))))
            records.append({
                "obs": [float(v) for v in obs_np[0]],
                "action": [float(v) for v in onnx_action[0]],
            })
            obs, _, done, _ = env.step(torch.as_tensor(onnx_action, device=obs.device))
            if bool(done[0]):
                env.reset(torch.zeros(1, dtype=torch.long, device=obs.device))
                obs = env.observation()

    reference = {
        "run": args.run,
        "iteration": int(checkpoint.get("iteration", -1)),
        "obs_size": OBS_SIZE,
        "action_size": ACTION_SIZE,
        "opset": args.opset,
        "policy_dt": env.dt,
        "decimation": 4,
        "torch_vs_onnx_max_delta": max_delta,
        "steps": records,
    }
    ref_path = run_dir / "mujoco_reference.json"
    ref_path.write_text(json.dumps(reference, indent=1))
    print(f"wrote {ref_path}  ({REFERENCE_STEPS} steps)")
    print(f"  torch vs onnxruntime max |delta| = {max_delta:.3e}")
    return 0 if max_delta < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())
