"""Validate the MuJoCo bug twins produced by prefab_to_mjcf.py.

For each of Quad_v01, Hexapod_v01, Crab_v01:
  1. body / hinge / actuator counts vs the Unity prefab (and the expected 9/8, 13/12, 13/12)
  2. total mass vs the sum of the prefab's ArticulationBody m_Mass
  3. rest root height (feet exactly on the floor) vs CreatureCatalog spawnHeight
  4. self-penetration in the rest pose (creature geom vs creature geom)
  5. 5 s drop-and-hold from the Unity spawn height, every actuator holding target 0:
     final root height, uprightness (root up . world up), worst floor/self penetration,
     worst joint-limit overshoot, NaN check
  6. mujoco_warp: put_model / put_data on cuda, 10 steps, finite and matching CPU

Usage:  .venv-mjwarp\\Scripts\\python.exe training\\bugs\\check_bug_mjcf.py [--no-warp] [names...]
Exit code 0 only if every check passes.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prefab_to_mjcf as conv  # noqa: E402

EXPECTED = {"Quad_v01": (9, 8), "Hexapod_v01": (13, 12), "Crab_v01": (13, 12)}
HOLD_SECONDS = 5.0
PEN_TOL = 0.01          # m: worst allowed floor penetration during the hold
SELF_PEN_TOL = 0.002    # m: worst allowed self penetration (rest pose and during hold)
LIMIT_TOL = math.radians(2.0)
UPRIGHT_MIN = 0.95
WARP_STEPS = 10
UNITY_DT = 0.005        # Unity Fixed Timestep (ProjectSettings/TimeManager.asset)
GAIT_PEN_TOL = 0.01     # m: self penetration allowed transiently while legs clash in the gait
GAIT_LIMIT_TOL = math.radians(5.0)
REF_DT = 0.0005         # converged step for the literal (kv-in-clamp) reference
FIDELITY_SECONDS = 1.0  # compared window (the gait is chaotic, trajectories diverge later)
FID_JOINT_TOL = 5.0     # deg joint RMS vs the converged literal reference


def prefab_truth(name: str) -> dict:
    docs = conv.parse_unity_yaml(conv.CREATURES[name])
    abs_ = [d["data"] for d in docs.values() if d["cid"] == conv.CID_ARTICULATIONBODY]
    hinges = sum(1 for a in abs_ if int(a["m_ArticulationJointType"]) == 2 and int(a["m_Twist"]) != 0)
    return {"bodies": len(abs_), "hinges": hinges, "mass": sum(float(a["m_Mass"]) for a in abs_)}


def creature_geom_ids(model) -> set[int]:
    return {g for g in range(model.ngeom) if model.geom_bodyid[g] != 0}


def contact_extremes(model, data, creature: set[int]) -> tuple[float, float, list[str]]:
    """(worst floor penetration, worst self penetration, self-contact pair names). Positive = overlap."""
    floor_pen, self_pen, pairs = 0.0, 0.0, []
    for i in range(data.ncon):
        c = data.contact[i]
        pen = max(0.0, -float(c.dist))
        if c.geom1 in creature and c.geom2 in creature:
            self_pen = max(self_pen, pen)
            pairs.append(f"{model.geom(c.geom1).name}~{model.geom(c.geom2).name}")
        else:
            floor_pen = max(floor_pen, pen)
    return floor_pen, self_pen, pairs


def limit_overshoot(model, data) -> float:
    worst = 0.0
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE or not model.jnt_limited[j]:
            continue
        q = data.qpos[model.jnt_qposadr[j]]
        lo, hi = model.jnt_range[j]
        worst = max(worst, lo - q, q - hi)
    return worst


def root_up(data) -> float:
    w, x, y, z = data.qpos[3:7]
    return float(1.0 - 2.0 * (x * x + y * y))  # R[2,2] = body z-axis . world z


def gait_replay(model, rig: dict, creature: set[int], seconds: float, sample_every: int = 1) -> dict:
    """Replay Agent_Creature.Heuristic from the rest pose. Actions are sampled every
    DecisionPeriod Unity steps (0.005 s each) and held, exactly as ML-Agents does."""
    data = mujoco.MjData(model)
    agent = rig["agent"]
    n = model.nu
    amps = np.array(agent["gaitAmplitudes"] or [0.0] * n)
    phases = np.array(agent["gaitPhases"] or [0.0] * n)
    offsets = np.array(agent["gaitOffsets"] or [0.0] * n)
    hold = max(1, int(agent["decisionPeriod"] or 1)) * UNITY_DT
    dt = model.opt.timestep
    mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
    mujoco.mj_forward(model, data)
    start = data.qpos[:2].copy()
    out = {"min_up": math.inf, "self_pen": 0.0, "nan": None, "pairs": set(), "limit_over": 0.0, "max_qd": 0.0}
    traj, next_decision = [], 0.0
    for step in range(int(round(seconds / dt))):
        t = step * dt
        if t >= next_decision - 1e-9:
            action = np.clip(amps * np.sin(2 * math.pi * agent["gaitFrequency_hz"] * t + phases) + offsets, -1, 1)
            data.ctrl[:] = action * rig["action_scale_rad"]
            next_decision += hold
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            out["nan"] = step
            break
        _, sp, pairs = contact_extremes(model, data, creature)
        out["self_pen"] = max(out["self_pen"], sp)
        out["pairs"].update(pairs)
        out["min_up"] = min(out["min_up"], root_up(data))
        out["limit_over"] = max(out["limit_over"], limit_overshoot(model, data))
        out["max_qd"] = max(out["max_qd"], float(np.max(np.abs(data.qvel[6:]))))
        if (step + 1) % sample_every == 0:
            traj.append(np.concatenate([data.qpos[:3], data.qpos[7:]]))
    out["forward"], out["lateral"] = (float(v) for v in data.qpos[:2] - start)
    out["final_z"] = float(data.qpos[2])
    out["pairs"] = sorted(out["pairs"])
    out["traj"] = np.array(traj)
    return out


def check_one(name: str, use_warp: bool) -> tuple[dict, list[str]]:
    fails: list[str] = []
    rig = json.loads((HERE / f"{name}_rig.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(HERE / f"{name}.xml"))
    data = mujoco.MjData(model)
    truth = prefab_truth(name)
    creature = creature_geom_ids(model)

    nbody = model.nbody - 1
    nhinge = int(np.sum(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE))
    exp_b, exp_h = EXPECTED[name]
    r = {"name": name, "bodies": nbody, "hinges": nhinge, "actuators": model.nu,
         "prefab_bodies": truth["bodies"], "prefab_hinges": truth["hinges"]}
    if not (nbody == truth["bodies"] == exp_b):
        fails.append(f"body count {nbody} (prefab {truth['bodies']}, expected {exp_b})")
    if not (nhinge == truth["hinges"] == exp_h == model.nu):
        fails.append(f"hinge/actuator count {nhinge}/{model.nu} (prefab {truth['hinges']}, expected {exp_h})")

    mass = float(np.sum(model.body_mass))
    r["mass"], r["prefab_mass"] = mass, truth["mass"]
    if abs(mass - truth["mass"]) > 1e-4:
        fails.append(f"mass {mass:.4f} != prefab {truth['mass']:.4f}")

    # Rest pose: feet exactly on the floor.
    mujoco.mj_resetDataKeyframe(model, data, model.key("rest").id)
    mujoco.mj_forward(model, data)
    r["rest_height"] = float(data.qpos[2])
    r["catalog_height"] = rig["catalog_spawn_height"]
    r["spawn_height"] = rig["unity_spawn_root_height"]
    _, rest_self_pen, rest_pairs = contact_extremes(model, data, creature)
    r["rest_self_pen"] = rest_self_pen
    if rest_self_pen > SELF_PEN_TOL:
        fails.append(f"self-penetration {rest_self_pen * 1000:.1f} mm in rest pose: {rest_pairs}")

    # Drop and hold from the Unity spawn height.
    mujoco.mj_resetDataKeyframe(model, data, model.key("spawn").id)
    data.ctrl[:] = 0.0
    mujoco.mj_forward(model, data)
    steps = int(round(HOLD_SECONDS / model.opt.timestep))
    worst_floor, worst_self, worst_limit, self_pairs = 0.0, 0.0, 0.0, set()
    min_z, min_up, nan_at = math.inf, math.inf, None
    for step in range(steps):
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)):
            nan_at = step
            break
        fp, sp, pairs = contact_extremes(model, data, creature)
        worst_floor, worst_self = max(worst_floor, fp), max(worst_self, sp)
        self_pairs.update(pairs)
        worst_limit = max(worst_limit, limit_overshoot(model, data))
        min_z, min_up = min(min_z, data.qpos[2]), min(min_up, root_up(data))
    r.update(final_height=float(data.qpos[2]), final_up=root_up(data), min_up=min_up, min_height=min_z,
             floor_pen=worst_floor, self_pen=worst_self, self_pairs=sorted(self_pairs),
             limit_over_deg=math.degrees(worst_limit), nan_at=nan_at,
             final_joint_dev_deg=float(np.degrees(np.max(np.abs(data.qpos[7:])))) if nan_at is None else math.nan,
             final_speed=float(np.linalg.norm(data.qvel[:3])) if nan_at is None else math.nan)
    if nan_at is not None:
        fails.append(f"NaN at step {nan_at}")
    else:
        if r["final_up"] < UPRIGHT_MIN:
            fails.append(f"not upright after hold (up={r['final_up']:.3f})")
        if abs(r["final_height"] - r["rest_height"]) > 0.05:
            fails.append(f"hold height {r['final_height']:.3f} vs rest {r['rest_height']:.3f}")
        if worst_floor > PEN_TOL:
            fails.append(f"floor penetration {worst_floor * 1000:.1f} mm")
        if worst_self > SELF_PEN_TOL:
            fails.append(f"self penetration {worst_self * 1000:.1f} mm ({sorted(self_pairs)})")
        if worst_limit > LIMIT_TOL:
            fails.append(f"joint limit overshoot {math.degrees(worst_limit):.2f} deg")
        if r["final_speed"] > 0.01:
            fails.append(f"still moving after hold ({r['final_speed']:.3f} m/s)")

    # Gait replay: the prefab's own coded gait (Agent_Creature.Heuristic), sampled
    # every DecisionPeriod physics steps and held in between, as Unity does it.
    # Fails only on NaN / flip / self overlap / limit blow-through; travel is info.
    g = gait_replay(model, rig, creature, HOLD_SECONDS)
    r.update({f"gait_{k}": v for k, v in g.items() if k != "traj"})
    if g["nan"] is not None:
        fails.append(f"gait replay NaN at step {g['nan']}")
    elif g["min_up"] < 0.0:
        fails.append(f"gait replay flipped over (min up {g['min_up']:.2f})")
    if g["self_pen"] > GAIT_PEN_TOL:
        fails.append(f"gait replay self penetration {g['self_pen'] * 1000:.1f} mm ({g['pairs']})")
    if g["limit_over"] > GAIT_LIMIT_TOL:
        fails.append(f"gait replay joint limit overshoot {math.degrees(g['limit_over']):.1f} deg")

    # Fidelity of the damping placement: default model at 5 ms vs the literal
    # (kv inside the force clamp) model at 0.5 ms, where it has converged.
    literal = mujoco.MjModel.from_xml_path(str(HERE / f"{name}_literal.xml"))
    literal.opt.timestep = REF_DT
    ref = gait_replay(literal, rig, creature_geom_ids(literal), FIDELITY_SECONDS,
                      sample_every=int(round(model.opt.timestep / REF_DT)))
    lit5 = gait_replay(mujoco.MjModel.from_xml_path(str(HERE / f"{name}_literal.xml")), rig,
                       creature_geom_ids(literal), FIDELITY_SECONDS)
    n = min(len(ref["traj"]), len(g["traj"]))
    def rms(tr):
        return (float(np.degrees(np.sqrt(np.mean((tr[:n, 3:] - ref["traj"][:n, 3:]) ** 2)))),
                float(np.sqrt(np.mean((tr[:n, 2] - ref["traj"][:n, 2]) ** 2))))
    r["fid_joint_deg"], r["fid_z_m"] = rms(g["traj"])
    r["lit5_joint_deg"], r["lit5_z_m"] = rms(lit5["traj"])
    r["ref_max_qd"], r["lit5_max_qd"] = ref["max_qd"], lit5["max_qd"]
    if r["fid_joint_deg"] > FID_JOINT_TOL:
        fails.append(f"gait fidelity {r['fid_joint_deg']:.1f} deg joint RMS vs converged literal reference")

    # mujoco_warp on cuda.
    r["warp"] = "skipped"
    if use_warp:
        try:
            import mujoco_warp as mjw
            import warp as wp

            wp.config.log_level = wp.LOG_WARNING if hasattr(wp, 'LOG_WARNING') else None
            seed = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, seed, model.key("spawn").id)
            mujoco.mj_forward(model, seed)
            cpu = mujoco.MjData(model)
            mujoco.mj_resetDataKeyframe(model, cpu, model.key("spawn").id)
            with wp.ScopedDevice("cuda:0"):
                wm = mjw.put_model(model)
                wd = mjw.put_data(model, seed, nworld=4)
                for _ in range(WARP_STEPS):
                    mjw.step(wm, wd)
                wp.synchronize()
                qpos = wd.qpos.numpy()
            for _ in range(WARP_STEPS):
                mujoco.mj_step(model, cpu)
            diff = float(np.max(np.abs(qpos[0] - cpu.qpos)))
            finite = bool(np.all(np.isfinite(qpos)))
            r["warp"] = f"ok on {wp.get_device('cuda:0').name}, {WARP_STEPS} steps, max |qpos-cpu| {diff:.2e}"
            if not finite:
                fails.append("warp produced non-finite qpos")
                r["warp"] = "NaN"
            elif diff > 1e-3:
                fails.append(f"warp vs CPU qpos differ by {diff:.2e} after {WARP_STEPS} steps")
        except Exception as exc:  # report, do not hide
            r["warp"] = f"FAILED: {type(exc).__name__}: {exc}"
            fails.append(r["warp"])
    return r, fails


def main(argv: list[str]) -> int:
    use_warp = "--no-warp" not in argv
    names = [a for a in argv if not a.startswith("--")] or list(EXPECTED)
    all_ok = True
    rows = []
    for name in names:
        r, fails = check_one(name, use_warp)
        rows.append((r, fails))
        all_ok &= not fails
        print(f"\n=== {name} ===")
        print(f"  counts      bodies {r['bodies']} (prefab {r['prefab_bodies']})  hinges {r['hinges']} "
              f"(prefab {r['prefab_hinges']})  actuators {r['actuators']}")
        print(f"  mass        {r['mass']:.3f} kg (prefab sum {r['prefab_mass']:.3f})")
        print(f"  root height rest {r['rest_height']:.4f} m | catalog spawnHeight {r['catalog_height']:.3f} "
              f"| Unity spawn (catalog+0.05) {r['spawn_height']:.3f} | rest self-pen {r['rest_self_pen'] * 1000:.2f} mm")
        print(f"  5 s hold    final z {r['final_height']:.4f} m  up {r['final_up']:.4f} (min {r['min_up']:.4f})  "
              f"min z {r['min_height']:.4f}  speed {r['final_speed']:.4f} m/s  max |q| {r['final_joint_dev_deg']:.2f} deg")
        print(f"              floor pen {r['floor_pen'] * 1000:.2f} mm  self pen {r['self_pen'] * 1000:.2f} mm "
              f"{r['self_pairs'] if r['self_pairs'] else ''} limit overshoot {r['limit_over_deg']:.2f} deg"
              f"{'  NaN at ' + str(r['nan_at']) if r['nan_at'] is not None else ''}")
        print(f"  gait 5 s    forward {r['gait_forward']:+.3f} m  lateral {r['gait_lateral']:+.3f} m  "
              f"final z {r['gait_final_z']:.3f}  min up {r['gait_min_up']:.3f}  max qd {r['gait_max_qd']:.1f} rad/s  "
              f"self pen {r['gait_self_pen'] * 1000:.2f} mm {r['gait_pairs'] if r['gait_pairs'] else ''} "
              f"limit overshoot {math.degrees(r['gait_limit_over']):.2f} deg")
        print(f"  fidelity    1 s gait vs literal model @ {REF_DT * 1000:g} ms (max qd {r['ref_max_qd']:.1f}): "
              f"this model {r['fid_joint_deg']:.2f} deg / {r['fid_z_m'] * 1000:.1f} mm RMS; literal @ 5 ms "
              f"{r['lit5_joint_deg']:.2f} deg / {r['lit5_z_m'] * 1000:.1f} mm RMS (max qd {r['lit5_max_qd']:.1f})")
        print(f"  warp        {r['warp']}")
        print(f"  RESULT      {'PASS' if not fails else 'FAIL: ' + '; '.join(fails)}")

    print("\n| creature | bodies | hinges | act | mass kg | rest z | catalog | hold z | up | floor pen mm "
          "| self pen mm | gait fwd m | gait fidelity deg | warp | result |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r, fails in rows:
        print(f"| {r['name']} | {r['bodies']} | {r['hinges']} | {r['actuators']} | {r['mass']:.1f} | "
              f"{r['rest_height']:.3f} | {r['catalog_height']:.2f} | {r['final_height']:.3f} | {r['final_up']:.3f} | "
              f"{r['floor_pen'] * 1000:.1f} | {r['self_pen'] * 1000:.1f} | {r['gait_forward']:+.2f} | {r['fid_joint_deg']:.2f} | "
              f"{'ok' if r['warp'].startswith('ok') else r['warp']} | {'PASS' if not fails else 'FAIL'} |")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
