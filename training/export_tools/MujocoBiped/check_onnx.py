"""
Validates MujocoBiped.onnx against the recorded MuJoCo actions with onnxruntime.

This is the "is the model itself intact" gate. It touches no physics: it feeds the
150 recorded observations straight in and compares the output against the actions
MuJoCo actually applied. If this passes and the creature still walks wrong, the
problem is in the physics, not in the model or the plumbing - which is exactly the
split the triage ladder in CONTRACT.md depends on.

The in-engine twin of this check is rung 0 of MujocoBipedPlayModeTests, which runs
the same observations through Unity's Inference Engine worker. The two should agree
to about 1e-6; a gap between them is an Inference Engine problem, not a model one.

    python check_onnx.py          # exits non-zero if max abs diff > 1e-4
"""
import json
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "MujocoBiped.onnx")
REFERENCE = os.path.join(HERE, "mujoco_reference.json")
TOLERANCE = 1e-4

OBS_DIM = 49
ACT_DIM = 12


def main():
    if not os.path.exists(REFERENCE):
        sys.exit("%s not found - run make_reference.py first." % REFERENCE)

    model = onnx.load(MODEL)
    onnx.checker.check_model(model, full_check=True)
    ops = sorted({n.op_type for n in model.graph.node})
    ins = [(i.name, [d.dim_value or d.dim_param for d in i.type.tensor_type.shape.dim])
           for i in model.graph.input]
    outs = [(o.name, [d.dim_value or d.dim_param for d in o.type.tensor_type.shape.dim])
            for o in model.graph.output]

    print("model    %s (%.1f KB)" % (os.path.basename(MODEL), os.path.getsize(MODEL) / 1024.0))
    print("opset    %s   ir %s" % (model.opset_import[0].version, model.ir_version))
    print("input    %s" % ins)
    print("output   %s" % outs)
    print("ops      %s" % ", ".join(ops))
    print("initialisers %d (normalisation and the action clamp are baked in)"
          % len(model.graph.initializer))

    if ins[0][1] != [1, OBS_DIM]:
        sys.exit("expected input shape [1, %d], got %s" % (OBS_DIM, ins[0][1]))
    if outs[0][1] != [1, ACT_DIM]:
        sys.exit("expected output shape [1, %d], got %s" % (ACT_DIM, outs[0][1]))

    ref = json.load(open(REFERENCE))
    steps = ref["trajectory"]
    obs = np.array([s["observation"] for s in steps], dtype=np.float32)
    want = np.array([s["action"] for s in steps], dtype=np.float32)

    sess = ort.InferenceSession(MODEL, providers=["CPUExecutionProvider"])
    iname = sess.get_inputs()[0].name

    got = np.empty_like(want)
    for i in range(len(obs)):
        got[i] = sess.run(None, {iname: obs[i: i + 1]})[0][0]

    diff = np.abs(got - want)
    worst_step = int(diff.max(axis=1).argmax())
    worst_joint = int(diff[worst_step].argmax())

    print()
    print("checked  %d recorded steps" % len(obs))
    print("max abs  %.3e  (step %d, action index %d = %s)"
          % (diff.max(), worst_step, worst_joint, ref["conventions"]["jointOrder"][worst_joint]))
    print("mean abs %.3e" % diff.mean())
    print("out of range [-1, 1]: %d of %d values" % (int((np.abs(got) > 1.0 + 1e-6).sum()), got.size))

    if diff.max() > TOLERANCE:
        sys.exit("FAIL: max abs diff %.3e exceeds tolerance %.0e" % (diff.max(), TOLERANCE))
    print("PASS: within %.0e" % TOLERANCE)


if __name__ == "__main__":
    main()
