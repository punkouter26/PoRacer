#!/usr/bin/env python3
"""
check_onnx.py - prove IsaacH1.onnx reproduces the recorded Isaac actions.

Feeds every recorded observation from isaac_reference.json through the shipped
ONNX with onnxruntime and compares against the recorded action. This isolates the
INFERENCE path: if this passes and Unity's in-engine rung 0 also passes, any
remaining divergence is physics, not the network.

Also verifies the claims the Unity side depends on:
  * single file, no external data
  * exactly one input [1, 69] and one output [1, 19]
  * no normalisation node - obs_normalization is false, so the policy must be a
    bare MLP and raw observations are fed straight in
  * operator set is within what Inference Engine runs on the CPU backend

Gate: max abs difference < 1e-4.

Usage: python check_onnx.py [--onnx IsaacH1.onnx] [--ref isaac_reference.json]
                            [--tol 1e-4]
Requires: onnxruntime, numpy, onnx (optional, for the graph report).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXPORT = os.path.normpath(os.path.join(HERE, "..", "..", "h1"))

# Ops Inference Engine runs on BackendType.CPU. Anything outside this set would
# silently fall back or fail to import, so it is worth asserting.
SUPPORTED_OPS = {
    "Gemm", "MatMul", "Add", "Mul", "Sub", "Div", "Relu", "Elu", "Tanh",
    "Sigmoid", "LeakyRelu", "Identity", "Reshape", "Concat", "Slice", "Clip",
    "Constant", "Flatten", "Softplus", "Erf", "Pow", "Sqrt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=os.path.join(HERE, "IsaacH1.onnx"))
    ap.add_argument("--ref", default=os.path.join(HERE, "isaac_reference.json"))
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args()

    ref_path = args.ref
    if not os.path.exists(ref_path):
        ref_path = os.path.join(DEFAULT_EXPORT, "isaac_reference.json")
    ref = json.load(open(ref_path))
    # The raw export is a bare JSON array; the copy in this folder wraps it in an
    # object under "steps" (Unity's JsonUtility cannot parse a top-level array).
    if isinstance(ref, dict):
        ref = ref["steps"]
    obs = np.array([s["obs"] for s in ref], dtype=np.float32)
    act = np.array([s["action"] for s in ref], dtype=np.float32)
    print(f"reference : {ref_path}")
    print(f"            {len(ref)} steps, obs {obs.shape}, action {act.shape}")

    fails = []

    # ------------------------------------------------------------- graph ----
    try:
        import onnx
        m = onnx.load(args.onnx)
        onnx.checker.check_model(m)
        ops = sorted({n.op_type for n in m.graph.node})
        ext = [i for i in m.graph.initializer if i.HasField("data_location") and i.data_location == 1]
        print(f"onnx      : ir={m.ir_version} opset="
              f"{ {o.domain or 'ai.onnx': o.version for o in m.opset_import} }")
        print(f"            ops={ops}")
        print(f"            initializers={len(m.graph.initializer)} external={len(ext)}")
        if ext:
            fails.append(f"model uses {len(ext)} external data tensors; must be a single file")
        bad = set(ops) - SUPPORTED_OPS
        if bad:
            fails.append(f"operators outside the Inference Engine CPU set: {sorted(bad)}")
        norm_ops = [n.op_type for n in m.graph.node
                    if n.op_type in ("BatchNormalization", "LayerNormalization", "InstanceNormalization")]
        if norm_ops:
            fails.append(f"unexpected normalisation layers {norm_ops}; export says obs_normalization=false")
        print(f"            normalisation nodes: none (raw observations fed directly) "
              f"- {'OK' if not norm_ops else 'MISMATCH'}")
    except ImportError:
        print("onnx      : package not installed, skipping graph report "
              "(numeric check below is the gate)")

    # --------------------------------------------------------- onnxruntime --
    try:
        import onnxruntime as ort
    except ImportError:
        sys.exit("ERROR: onnxruntime is required.  pip install onnxruntime")

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(args.onnx, so, providers=["CPUExecutionProvider"])
    ins = [(i.name, i.shape, i.type) for i in sess.get_inputs()]
    outs = [(o.name, o.shape, o.type) for o in sess.get_outputs()]
    print(f"inputs    : {ins}")
    print(f"outputs   : {outs}")
    if len(ins) != 1 or len(outs) != 1:
        fails.append(f"expected exactly 1 input and 1 output, got {len(ins)}/{len(outs)}")
    else:
        if list(ins[0][1]) != [1, obs.shape[1]]:
            fails.append(f"input shape {ins[0][1]} != [1, {obs.shape[1]}]")
        if list(outs[0][1]) != [1, act.shape[1]]:
            fails.append(f"output shape {outs[0][1]} != [1, {act.shape[1]}]")

    in_name = ins[0][0]
    pred = np.zeros_like(act)
    for i in range(len(obs)):
        pred[i] = sess.run(None, {in_name: obs[i : i + 1]})[0][0]

    diff = np.abs(pred - act)
    mx, mean = float(diff.max()), float(diff.mean())
    arg = np.unravel_index(int(diff.argmax()), diff.shape)
    print()
    print(f"max  abs diff : {mx:.6e}   (step {arg[0]}, action index {arg[1]})")
    print(f"mean abs diff : {mean:.6e}")
    print(f"tolerance     : {args.tol:g}")
    per_step = diff.max(axis=1)
    print(f"worst 5 steps : "
          f"{[(int(i), float(per_step[i])) for i in np.argsort(per_step)[-5:][::-1]]}")

    # determinism: the same input twice must give bit-identical output
    a = sess.run(None, {in_name: obs[0:1]})[0]
    b = sess.run(None, {in_name: obs[0:1]})[0]
    if not np.array_equal(a, b):
        fails.append("model is not deterministic across identical runs")
    print(f"deterministic : {np.array_equal(a, b)}")

    if mx >= args.tol:
        fails.append(f"max abs diff {mx:.3e} >= tolerance {args.tol:g}")

    print()
    if fails:
        for f in fails:
            print(f"FAIL: {f}")
        sys.exit(1)
    print(f"PASS - ONNX reproduces all {len(ref)} recorded actions to {mx:.3e} "
          f"(< {args.tol:g})")


if __name__ == "__main__":
    main()
