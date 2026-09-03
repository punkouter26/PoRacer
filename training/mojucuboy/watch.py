"""Read the live TFEvents for a run and print the latest metrics.

The trainer's stdout is block-buffered when piped, so the event file is the
reliable progress channel -- and it is the one CLAUDE.md wants watched anyway.
"""
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

run = sys.argv[1] if len(sys.argv) > 1 else "boy_chase01"
d = Path(__file__).resolve().parent / "runs" / run
acc = EventAccumulator(str(d), size_guidance={"scalars": 0})
acc.Reload()
tags = acc.Tags().get("scalars", [])
if not tags:
    print(f"{run}: no scalars yet")
    raise SystemExit(0)
series = {t: acc.Scalars(t) for t in tags}
steps = series[tags[0]][-1].step
print(f"{run}: {len(series[tags[0]])} logged points, {steps/1e6:.2f}M samples")
print(f"{'metric':<32}{'first':>10}{'latest':>10}{'best':>10}")
for t in sorted(tags):
    v = [s.value for s in series[t]]
    best = max(v) if ("length" in t or "speed" in t or "return" in t) else min(v)
    print(f"{t:<32}{v[0]:>10.3f}{v[-1]:>10.3f}{best:>10.3f}")
