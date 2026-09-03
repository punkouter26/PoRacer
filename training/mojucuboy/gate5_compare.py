"""Gate 5: compare the in-Unity evaluation against the Phase 4 training numbers.

  1. python gate5_compare.py --run boy_chase01          (after gate4_eval.py)
     reads runs/<run>/gate4_eval.json  and  runs/<run>/gate5_unity.json

The gate asks for performance parity within 10%. Two things are worth being
explicit about, because a naive comparison would either flatter or unfairly
punish the Unity side:

  * The Unity harness runs the NOMINAL model -- Unity has one model, not a
    randomised ensemble -- so the honest comparison is against the Python
    "nominal" evaluation, not the randomised one. Both are reported; the verdict
    uses nominal.

  * Survival rate is a proportion, so a relative error on a number near 1.0 is
    misleading. It is compared as an absolute percentage-point difference, and
    episode length and speed are compared relatively.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "runs"
TOLERANCE = 0.10


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    args = parser.parse_args()

    run_dir = RESULTS / args.run
    python_path = run_dir / "gate4_eval.json"
    unity_path = run_dir / "gate5_unity.json"
    for path in (python_path, unity_path):
        if not path.exists():
            print(f"ERROR: {path} missing")
            return 2

    python_all = json.loads(python_path.read_text())
    unity = json.loads(unity_path.read_text())
    nominal = python_all["nominal"]
    randomised = python_all["randomised"]

    print(f"run {args.run}\n")
    print(f"{'metric':<24}{'PY random':>12}{'PY nominal':>12}{'UNITY':>12}"
          f"{'diff vs nominal':>18}")

    rows = [
        ("mean episode length", "mean_episode_length", "rel"),
        ("median episode length", "median_episode_length", "rel"),
        ("mean forward speed", "mean_forward_speed", "rel"),
        ("survival rate", "survival_rate", "abs"),
    ]
    failures = []
    for label, key, kind in rows:
        r, n, u = randomised[key], nominal[key], unity[key]
        if kind == "rel":
            denom = max(abs(n), 1e-9)
            diff = abs(u - n) / denom
            shown = f"{diff:+.1%}" if n else "n/a"
            bad = diff > TOLERANCE
        else:
            diff = abs(u - n)
            shown = f"{diff:+.1%} pts"
            bad = diff > TOLERANCE
        flag = "  FAIL" if bad else ""
        if bad:
            failures.append((label, shown))
        print(f"{label:<24}{r:>12.3f}{n:>12.3f}{u:>12.3f}{shown:>18}{flag}")

    print("\n=== GATE 5 VERDICT ===")
    print(f"  tolerance: {TOLERANCE:.0%} (relative for lengths and speed, "
          f"absolute for survival)")
    if failures:
        for label, shown in failures:
            print(f"  FAIL  {label} differs by {shown}")
        print("  FAIL: Unity does not match training within tolerance.")
        return 1
    print("  PASS: in-Unity performance matches the training numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
