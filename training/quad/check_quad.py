"""Validates quad.xml (QUAD_SPEC.md): loads in MuJoCo and MuJoCo Warp, stands for 5 s
with zero action, and does not self-penetrate, also under random actions.

  .venv-mjwarp\\Scripts\\python.exe training/quad/check_quad.py          # add --no-warp to skip the GPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "quad.xml"
RIG = HERE / "quad_rig.json"
PEN_TOL = 0.01       # m; a body-body contact deeper than this counts as passing through.
                     # Soft contacts (solref 0.01) legitimately sink a few mm when two legs
                     # driven at 300 N*m collide; that is a detected, resolved contact.


def torso_state(m, d, torso):
    up = d.xmat[torso].reshape(3, 3)[2, 2]
    return float(d.xpos[torso, 2]), float(up)


def contact_stats(m, d, floor):
    """(deepest body-body penetration, deepest floor penetration, contact count)."""
    self_pen, floor_pen = 0.0, 0.0
    for i in range(d.ncon):
        c = d.contact[i]
        depth = max(0.0, -float(c.dist))
        if floor in (c.geom1, c.geom2):
            floor_pen = max(floor_pen, depth)
        else:
            self_pen = max(self_pen, depth)
    return self_pen, floor_pen, int(d.ncon)


def stand(m, key: str, seconds: float = 5.0) -> dict:
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, key))
    d.ctrl[:] = 0.0
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Quad_v01")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    worst_self, worst_floor, peak_con, min_up = 0.0, 0.0, 0, 1.0
    for _ in range(int(round(seconds / m.opt.timestep))):
        mujoco.mj_step(m, d)
        s, f, n = contact_stats(m, d, floor)
        worst_self, worst_floor, peak_con = max(worst_self, s), max(worst_floor, f), max(peak_con, n)
        min_up = min(min_up, torso_state(m, d, torso)[1])
    z, up = torso_state(m, d, torso)
    return {"start": key, "finalHeight": z, "finalUp": up, "minUp": min_up,
            "finalMaxAbsQvel": float(np.abs(d.qvel).max()),
            "finalXYDrift": float(np.hypot(d.qpos[0], d.qpos[1])),
            "maxSelfPenetration": worst_self, "maxFloorPenetration": worst_floor,
            "peakContacts": peak_con, "finite": bool(np.isfinite(d.qpos).all())}


def random_actions(m, episodes: int = 16, seconds: float = 10.0, seed: int = 0) -> dict:
    """Policy-like random actions (held 4 substeps), to provoke leg-leg and leg-torso contact."""
    rng = np.random.default_rng(seed)
    torso = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "Quad_v01")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    worst_self, peak_con, peak_qvel, falls, ok = 0.0, 0, 0.0, 0, True
    self_pairs = set()
    for _ in range(episodes):
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, 0)
        action = np.zeros(m.nu)
        fell = False
        for step in range(int(round(seconds / m.opt.timestep))):
            if step % 4 == 0:
                # smooth-ish random walk in [-1, 1], the kind of thing an untrained policy does
                action = np.clip(action + rng.normal(0, 0.5, m.nu), -1, 1)
                d.ctrl[:] = action * 0.785398
            mujoco.mj_step(m, d)
            s, _, n = contact_stats(m, d, floor)
            if s > PEN_TOL:
                for i in range(d.ncon):
                    c = d.contact[i]
                    if floor not in (c.geom1, c.geom2) and -c.dist > PEN_TOL:
                        self_pairs.add(tuple(sorted((mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1),
                                                     mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)))))
            worst_self, peak_con = max(worst_self, s), max(peak_con, n)
            peak_qvel = max(peak_qvel, float(np.abs(d.qvel).max()))
            z, up = torso_state(m, d, torso)
            fell |= up < 0.5 or z < 0.45
        falls += fell
        ok &= bool(np.isfinite(d.qpos).all())
    return {"episodes": episodes, "seconds": seconds, "maxSelfPenetration": worst_self,
            "selfPenetratingPairs": sorted(self_pairs), "peakContacts": peak_con,
            "peakAbsQvel": peak_qvel, "episodesThatFell": falls, "finite": ok}


def warp_stand(m, worlds: int = 64, seconds: float = 5.0) -> dict:
    import mujoco_warp as mjw
    import warp as wp
    wp.config.quiet = True
    wp.init()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "spawn"))
    mujoco.mj_forward(m, d)
    wm = mjw.put_model(m)
    wm.opt.warn_overflow = 0   # float32 linesearch at a converged rest state; benign
    wd = mjw.put_data(m, d, nworld=worlds, nconmax=48, njmax=224)
    peak = 0
    for _ in range(int(round(seconds / m.opt.timestep))):
        mjw.step(wm, wd)
        peak = max(peak, int(wd.nacon.numpy()[0]))
    mjw.kinematics(wm, wd)
    qpos = wd.qpos.numpy()
    xmat = wd.xmat.numpy()[:, 1].reshape(worlds, 3, 3)
    return {"worlds": worlds, "finalHeightMean": float(qpos[:, 2].mean()),
            "finalHeightSpread": float(qpos[:, 2].max() - qpos[:, 2].min()),
            "finalUpMin": float(xmat[:, 2, 2].min()), "peakContactsAllWorlds": peak,
            "finite": bool(np.isfinite(qpos).all())}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-warp", action="store_true")
    args = parser.parse_args()
    m = mujoco.MjModel.from_xml_path(str(MODEL))
    rig = json.loads(RIG.read_text())
    report = {
        "load": {"nbody": m.nbody - 1, "njnt_hinge": int((m.jnt_type == 3).sum()), "nu": m.nu,
                 "mass": float(m.body_subtreemass[1]), "forcerange": m.actuator_forcerange[0].tolist(),
                 "friction": sorted(set(float(f) for f in m.geom_friction[:, 0])),
                 "timestep": float(m.opt.timestep), "rigActionOrderMatches":
                 [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)] == rig["actionOrder"]},
        "standFromSpawn": stand(m, "spawn"),
        "standFromRest": stand(m, "rest"),
        "randomActions": random_actions(m),
    }
    if not args.no_warp:
        report["warpStand"] = warp_stand(m)
    print(json.dumps(report, indent=1))
    s = report["standFromSpawn"]
    passed = (report["load"]["rigActionOrderMatches"] and s["finite"]
              and abs(s["finalHeight"] - 0.90) < 0.02 and s["finalUp"] > 0.99
              and s["maxSelfPenetration"] < PEN_TOL
              and report["randomActions"]["maxSelfPenetration"] < PEN_TOL
              and report["randomActions"]["finite"]
              and (args.no_warp or (report["warpStand"]["finite"]
                                    and abs(report["warpStand"]["finalHeightMean"] - s["finalHeight"]) < 0.005)))
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
