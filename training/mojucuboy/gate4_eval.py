"""Gate 4: deterministic evaluation of a trained policy.

  .venv-mjwarp\\Scripts\\python.exe training/mojucuboy/gate4_eval.py --run boy_chase01

Runs the DETERMINISTIC policy (no sampling) over a fixed set of episodes from a
fixed seed, and reports the agreed convergence metrics:

  mean episode length  >= 900 / 1000 steps   (>= 18 s of a 20 s episode)
  mean forward speed   >= 1.2 m/s            along the COMMANDED heading
  survival             >= 90 / 100 episodes

Domain randomisation stays ON. A policy that only survives the nominal model is
not the thing being shipped -- Unity's model is nominal but its solver, contacts
and timing are not bit-identical, so the margin randomisation buys is the margin
that carries the policy across.

The same routine is what Phase 5 compares against in Unity, so keep the metric
definitions here and nowhere else.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mojucuboy_env  # noqa: E402
from mojucuboy_env import ACTION_SIZE, MojucuBoyEnv  # noqa: E402
from train_mojucuboy import RESULTS, ActorCritic  # noqa: E402

TARGET_EP_LEN = 900
TARGET_SPEED = 1.2
TARGET_SURVIVAL = 0.90
# Uptime replaces episode length and survival as the stability gate. Both of
# those are degenerate while `done` is timeout-only: no world ever terminates
# early, so length is always EPISODE_STEPS and survival always 1.0, for any
# policy including an untrained one.
TARGET_UPTIME = 0.90
TARGET_FALLEN = 0.10


def evaluate(policy, episodes: int, seed: int, randomise: bool = True,
             sprawl_tilt_deg: float = 180.0, reset_fallen: float = None,
             command_speed: float = None):
    """One deterministic pass. Each world runs exactly one episode; a world that
    terminates early is frozen rather than reset, so every episode is independent
    and the mean is not biased toward short ones."""
    # Evaluation condition is explicit, because "can it get up" is meaningless
    # without saying "from what". The default reproduces the shipped gate (fully
    # random sprawl); --sprawl-tilt lets a policy be measured at the difficulty it
    # was actually trained on, which is the only way to tell a curriculum that is
    # working at its current rung from one that is not working at all.
    # The speed command is observation index 11 -- the policy SEES it -- so a policy
    # that genuinely tracks its command should follow a different one. Overriding the
    # module constant (not just the tensor) makes reset() honour it too, otherwise
    # every episode boundary would silently restore 1.5 m/s.
    if command_speed is not None:
        mojucuboy_env.TARGET_SPEED = float(command_speed)
    env_kwargs = {"sprawl_max_tilt": math.radians(sprawl_tilt_deg)}
    # The evaluator must NOT inherit a run's reward-shaping overrides: K5 and K6 are
    # measured quantities, not reward terms, so every policy is scored on the same
    # physics regardless of what reward trained it. Only the config recorded in the
    # run directory says what shaping was used.
    if reset_fallen is not None:
        env_kwargs["reset_fallen_fraction"] = reset_fallen
    env = MojucuBoyEnv(episodes, seed=seed, **env_kwargs)
    # STATE THE CONDITION, ALWAYS. This module's `reset_fallen` defaults to None,
    # which inherits the env's RESET_FALLEN_FRACTION = 0.30 -- while speed_sweep.py
    # defaults the same argument to 0.0, because upright is how racers spawn in the
    # game. So the SAME KPI name can be reported under two different conditions by
    # two instruments, and K6 in particular is very sensitive to it: standing up from
    # supine needs peak torque, so any fallen fraction inflates control effort.
    # Changing the default is not the fix -- K10 recovery is meaningless without
    # fallen starts -- so the condition is printed and recorded instead, and no
    # number from here should be compared against one whose condition differs.
    print(f"[eval condition] reset_fallen_fraction={env.reset_fallen_fraction:.2f}  "
          f"sprawl_tilt={sprawl_tilt_deg:.0f}deg  randomise={randomise}  "
          f"command_speed={mojucuboy_env.TARGET_SPEED:.2f} m/s  episodes={episodes}")
    if not randomise:
        import warp as wp
        wp.to_torch(env.wm.actuator_gainprm).copy_(env.nominal_gain)
        wp.to_torch(env.wm.actuator_biasprm).copy_(env.nominal_bias)
        wp.to_torch(env.wm.body_mass).copy_(env.nominal_mass)
        wp.to_torch(env.wm.geom_friction).copy_(env.nominal_friction)

    device = torch.device("cuda:0")
    alive = torch.ones(episodes, dtype=torch.bool, device=device)
    length = torch.zeros(episodes, device=device)
    speed_sum = torch.zeros(episodes, device=device)
    ret = torch.zeros(episodes, device=device)
    # Uptime, not episode length, is what survival means in this env. `done` is
    # timeout-only -- a fall no longer ends the episode -- so `length` is always
    # EPISODE_STEPS and `survival_rate` below is always exactly 1.0, for any
    # policy including an untrained one. These two accumulate the per-step
    # standing/fallen fractions, which a policy lying on the floor genuinely fails.
    standing_sum = torch.zeros(episodes, device=device)
    fallen_sum = torch.zeros(episodes, device=device)
    # K4-K8 accumulators. The trainer has logged these per iteration since M1, but
    # the EVALUATOR reported only K1/K2/K3 -- so the brief's exit criterion ("all
    # Phase 1 KPIs satisfied across 10 consecutive evaluation episodes") could not
    # actually be checked against a policy, only watched during training. Same
    # definitions as KPI_TERMS in train_mojucuboy.py so the two are comparable.
    kpi_sums = {name: torch.zeros(episodes, device=device)
                for name in ("facing", "lateral", "ctrl", "accel", "upright",
                             # APPLIED TORQUE -- the quantity K6 was always meant to
                             # be. `ctrl` is the commanded joint angle of a position
                             # servo; holding the stance with a ZERO action measures
                             # ctrl 0.000 while the actuators apply up to 75 N.m.
                             "torque", "torque_abs")}
    # K10 recovery instrumentation. `uptime_over_90pct` cannot answer the question
    # two experiments were run to answer: with `reset_fallen_fraction` set, an
    # episode may start in a random attitude dropped from clear air, and one that
    # LANDS nearly upright and simply stabilises scores identically to one that
    # genuinely pushes up off its back. E12 and E13 were both judged on that proxy.
    #
    # So classify each episode by the state it was actually in once it had settled,
    # and track whether it later got up and stayed up:
    #   started_down  - genuinely down at the settle mark, not merely mid-air
    #   recovered     - started down AND spent the last quarter of the episode up
    #   steps_to_up   - first step at which a down-starter reached standing
    SETTLE_STEP = 50            # 1 s: long enough for a clear-air drop to land
    DOWN_THRESHOLD = 0.30       # `standing` below this at settle = genuinely down
    UP_THRESHOLD = 0.80         # `standing` above this = back on its feet
    started_down = torch.zeros(episodes, dtype=torch.bool, device=device)
    ever_up_after_down = torch.zeros(episodes, dtype=torch.bool, device=device)
    steps_to_up = torch.full((episodes,), float("nan"), device=device)
    tail_standing_sum = torch.zeros(episodes, device=device)
    tail_steps = 0
    tail_from = int(mojucuboy_env.EPISODE_STEPS * 0.75)
    obs = env.observation()

    with torch.no_grad():
        for step_index in range(mojucuboy_env.EPISODE_STEPS):
            action = policy(obs)
            obs, reward, done, terms = env.step(action)
            length += alive.float()
            speed_sum += terms["speed_along"] * alive.float()
            standing_sum += terms["standing"] * alive.float()
            fallen_sum += terms["fallen"] * alive.float()
            ret += reward * alive.float()

            for name, accumulator in kpi_sums.items():
                accumulator += terms[name] * alive.float()

            standing = terms["standing"]
            if step_index == SETTLE_STEP:
                started_down = (standing < DOWN_THRESHOLD) & alive
            if step_index > SETTLE_STEP:
                now_up = standing > UP_THRESHOLD
                first_up = started_down & now_up & torch.isnan(steps_to_up)
                if first_up.any():
                    steps_to_up = torch.where(first_up,
                                              torch.full_like(steps_to_up, float(step_index - SETTLE_STEP)),
                                              steps_to_up)
                ever_up_after_down |= started_down & now_up
            if step_index >= tail_from:
                tail_standing_sum += standing * alive.float()
                tail_steps += 1

            # Freeze on first termination; env.step keeps integrating those worlds
            # but their statistics stop accumulating.
            alive &= ~done
            if not alive.any():
                break

    survived = length >= mojucuboy_env.EPISODE_STEPS
    mean_speed = speed_sum / length.clamp(min=1)
    uptime = standing_sum / length.clamp(min=1)
    fallen = fallen_sum / length.clamp(min=1)
    return {
        "episodes": episodes,
        # Travels with every result, so a KPI read back out of a JSON months later
        # carries the condition it was measured under instead of an assumption.
        "eval_reset_fallen_fraction": float(env.reset_fallen_fraction),
        "eval_sprawl_tilt_deg": float(sprawl_tilt_deg),
        "eval_command_speed": float(mojucuboy_env.TARGET_SPEED),
        "eval_randomise": bool(randomise),
        # Kept for continuity, but see the note above: both are degenerate while
        # `done` is timeout-only. Read mean_uptime / mean_fallen instead.
        "mean_episode_length": float(length.mean()),
        "median_episode_length": float(length.median()),
        "survival_rate": float(survived.float().mean()),
        "survivors": int(survived.sum()),
        "mean_forward_speed": float(mean_speed.mean()),
        "mean_return": float(ret.mean()),
        "mean_uptime": float(uptime.mean()),
        "mean_fallen": float(fallen.mean()),
        "uptime_over_90pct": float((uptime > 0.90).float().mean()),
        # K10 — recovery, measured rather than inferred.
        # `started_down_frac` is what fraction of episodes were genuinely on the
        # floor once settled, which is NOT the same as reset_fallen_fraction: a
        # clear-air drop in a random attitude sometimes lands on its feet.
        # `recovery_rate` is the number that answers "can it get up": of the
        # episodes that WERE down, the fraction that finished the episode up.
        # K4-K8. `facing` is cos(heading error), so the angle is its arccos -- the
        # same derivation train_mojucuboy.py uses for heading_err_deg, kept identical
        # on purpose so a training curve and an evaluation can be read against each
        # other rather than merely resembling one another.
        "heading_err_deg": float(torch.rad2deg(torch.acos(
            (kpi_sums["facing"] / length.clamp(min=1)).clamp(-1.0, 1.0))).mean()),
        "lateral_drift": float((kpi_sums["lateral"] / length.clamp(min=1)).mean()),
        "ctrl_effort": float((kpi_sums["ctrl"] / length.clamp(min=1)).mean()),
        "torque_sq": float((kpi_sums["torque"] / length.clamp(min=1)).mean()),
        "torque_abs": float((kpi_sums["torque_abs"] / length.clamp(min=1)).mean()),
        "joint_accel": float((kpi_sums["accel"] / length.clamp(min=1)).mean()),
        "uprightness": float((kpi_sums["upright"] / length.clamp(min=1)).mean()),
        "started_down_frac": float(started_down.float().mean()),
        "started_down_count": int(started_down.sum()),
        "recovery_rate": float(
            (started_down & (tail_standing_sum / max(tail_steps, 1) > UP_THRESHOLD)).sum()
            / started_down.sum().clamp(min=1)),
        "ever_stood_after_down": float(
            ever_up_after_down.sum() / started_down.sum().clamp(min=1)),
        "median_steps_to_stand": (
            float(torch.nanmedian(steps_to_up)) if bool((~torch.isnan(steps_to_up)).any()) else float("nan")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=4242)
    parser.add_argument("--sprawl-tilt", type=float, default=180.0,
                        help="Evaluation sprawl difficulty in degrees off vertical. "
                             "180 (default) reproduces the shipped gate; lower values "
                             "measure a curriculum policy at the rung it trained on.")
    parser.add_argument("--reset-fallen", type=float, default=None,
                        help="Override the fraction of episodes that start sprawled.")
    parser.add_argument("--command-speed", type=float, default=None,
                        help="Override the commanded speed in m/s. A policy that tracks "
                             "its command should follow this; one that ignores it will not.")
    args = parser.parse_args()

    run_dir = RESULTS / args.run
    checkpoint = torch.load(run_dir / "policy.pt", map_location="cuda:0", weights_only=False)
    policy = ActorCritic().to("cuda:0")
    policy.load_state_dict(checkpoint["model"])
    policy.eval()

    print(f"run {args.run}, iteration {checkpoint.get('iteration', '?')}, "
          f"{args.episodes} deterministic episodes of {mojucuboy_env.EPISODE_STEPS} steps "
          f"({mojucuboy_env.EPISODE_STEPS * 0.02:.0f} s)\n")

    results = {}
    for label, randomise in (("randomised", True), ("nominal", False)):
        stats = evaluate(policy, args.episodes, args.seed, randomise,
                         sprawl_tilt_deg=args.sprawl_tilt, reset_fallen=args.reset_fallen,
                         command_speed=args.command_speed)
        results[label] = stats
        print(f"=== {label.upper()} MODEL ===")
        print(f"  mean episode length : {stats['mean_episode_length']:7.1f} / "
              f"{mojucuboy_env.EPISODE_STEPS}   (target >= {TARGET_EP_LEN})")
        print(f"  median ep length    : {stats['median_episode_length']:7.1f}")
        print(f"  mean forward speed  : {stats['mean_forward_speed']:7.3f} m/s"
              f"        (target >= {TARGET_SPEED})")
        print(f"  survival            : {stats['survivors']:4d} / {stats['episodes']}"
              f"          (degenerate: always 100%, see mean uptime)")
        print(f"  mean uptime         : {stats['mean_uptime']:7.3f}"
              f"            (target >= {TARGET_UPTIME})")
        print(f"  mean fallen         : {stats['mean_fallen']:7.3f}"
              f"            (target <= {TARGET_FALLEN})")
        print(f"  episodes >90% up    : {stats['uptime_over_90pct']:7.1%}")
        print(f"  mean return         : {stats['mean_return']:7.2f}")
        # The full K1-K8 dashboard, so a policy can be checked against the exit
        # criteria directly instead of by reading training curves and hoping.
        print(f"  -- K4-K8 --")
        print(f"  K4 heading err      : {stats['heading_err_deg']:7.1f} deg   (< 15)")
        print(f"  K5 lateral drift    : {stats['lateral_drift']:7.3f} m/s   (< 0.25)")
        print(f"  K6 control effort   : {stats['ctrl_effort']:7.4f}  (commanded angle)")
        print(f"  K6b applied torque  : {stats['torque_abs']:7.3f} N.m mean abs")
        print(f"  K6b torque squared  : {stats['torque_sq']:7.4g}")
        print(f"  K7 joint accel      : {stats['joint_accel']:7.3e}")
        print(f"  K8 uprightness      : {stats['uprightness']:7.3f}       (> 0.90)")
        # K10 — the get-up question, measured on the episodes it actually applies
        # to. started_down is counted at the settle mark, so a clear-air drop that
        # lands on its feet is NOT counted as a recovery.
        print(f"  -- K10 recovery --")
        print(f"  started down        : {stats['started_down_count']:4d} / {stats['episodes']}"
              f"   ({stats['started_down_frac']:.1%} of episodes, settled)")
        print(f"  recovery rate       : {stats['recovery_rate']:7.1%}"
              f"   (of those, up for the final quarter)")
        print(f"  ever stood again    : {stats['ever_stood_after_down']:7.1%}")
        print(f"  median steps to up  : {stats['median_steps_to_stand']:7.1f}\n")

    (run_dir / "gate4_eval.json").write_text(json.dumps(results, indent=2))

    primary = results["randomised"]
    # "mean episode length" and "survival rate" are deliberately NOT gated on:
    # both are constant by construction while `done` is timeout-only, so a gate
    # on them passes an untrained policy. Uptime and fallen replace them.
    checks = [
        ("mean forward speed", primary["mean_forward_speed"], TARGET_SPEED, "ge"),
        ("mean uptime", primary["mean_uptime"], TARGET_UPTIME, "ge"),
        ("mean fallen", primary["mean_fallen"], TARGET_FALLEN, "le"),
    ]
    print("=== GATE 4 VERDICT (randomised model) ===")
    passed = True
    for name, value, target, direction in checks:
        ok = value >= target if direction == "ge" else value <= target
        passed &= ok
        arrow = ">=" if direction == "ge" else "<="
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<22} {value:8.3f}  vs target {arrow} {target}")
    print(f"  {'PASS' if passed else 'FAIL'}: overall")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
