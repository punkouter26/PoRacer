"""ONNX vs recorded Isaac actions.  Run:  python check_onnx.py  (needs onnxruntime, numpy)
PASS when max |onnx(obs) - recorded_action| < 1e-4 over all 200 reference steps."""
import json
import os
import sys

import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
sess = ort.InferenceSession(os.path.join(HERE, "spider.onnx"), providers=["CPUExecutionProvider"])
inp, out = sess.get_inputs()[0], sess.get_outputs()[0]
print(f"input  {inp.name} {inp.shape} {inp.type}")
print(f"output {out.name} {out.shape} {out.type}")
assert inp.name == "obs" and list(inp.shape) == [1, 59], inp
assert out.name == "actions" and list(out.shape) == [1, 16], out

ref = json.load(open(os.path.join(HERE, "isaac_reference.json")))
steps = ref["steps"] if isinstance(ref, dict) else ref
worst = 0.0
for step in steps:
    obs = np.asarray(step["obs"], dtype=np.float32)[None]
    act = sess.run(None, {"obs": obs})[0][0]
    worst = max(worst, float(np.abs(act - np.asarray(step["action"], dtype=np.float32)).max()))
print(f"{len(steps)} steps, max |onnx - isaac| = {worst:.3e}  ->  {'PASS' if worst < 1e-4 else 'FAIL'}")
sys.exit(0 if worst < 1e-4 else 1)
