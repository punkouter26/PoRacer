"""Gate 1 verification for mojucuboy.xml.

Compiles the generated MJCF, reports the model census against the authored plan,
solves for the standing root height, and runs the passive zero-action drop test
the gate asks for. Actuation is DISABLED for the drop: with position actuators a
zero ctrl means "hold the rest pose", which is active control, not a passive
fall.
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent


def load() -> tuple[mujoco.MjModel, dict]:
    meta = json.loads((HERE / "mojucuboy_rig.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(HERE / "mojucuboy.xml"))
    return model, meta


def stance_qpos(model: mujoco.MjModel, meta: dict) -> np.ndarray:
    """Standing pose: the authored stance angles, with the root raised until the
    lowest contact point of the collision geometry sits exactly on the floor."""
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    qpos = data.qpos.copy()
    qpos[:3] = model.body("hips").pos
    qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    for index, spec in enumerate(meta["joints"]):
        qpos[7 + index] = spec["stance_rad"]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    lowest = np.inf
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        centre = data.geom_xpos[geom_id]
        rot = data.geom_xmat[geom_id].reshape(3, 3)
        size = model.geom_size[geom_id]
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            corners = np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3) * size
            lowest = min(lowest, float((corners @ rot.T + centre)[:, 2].min()))
        elif model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            axis = rot[:, 2] * size[1]
            for end in (centre - axis, centre + axis):
                lowest = min(lowest, float(end[2] - size[0]))
        else:  # sphere
            lowest = min(lowest, float(centre[2] - size[0]))
    qpos[2] -= lowest
    return qpos


def passive_fall(model: mujoco.MjModel, qpos: np.ndarray) -> tuple[float, float, float]:
    """Drop the model with actuation disabled. Returns (fall time, start head
    height, end head height). Fall time is when the head first drops below half
    its standing height -- an unambiguous collapse, not a wobble."""
    model = mujoco.MjModel.from_xml_path(str(HERE / "mojucuboy.xml"))
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    head_id = model.body("head").id
    start = float(data.xipos[head_id][2])
    threshold = 0.5 * start
    fall_time = float("nan")
    steps = int(10.0 / model.opt.timestep)
    for step in range(steps):
        mujoco.mj_step(model, data)
        if np.isnan(fall_time) and data.xipos[head_id][2] < threshold:
            fall_time = (step + 1) * model.opt.timestep
    return fall_time, start, float(data.xipos[head_id][2])


def main() -> None:
    model, meta = load()
    print("=== MODEL CENSUS ===")
    print(f"  bodies (excl. world) : {model.nbody - 1}")
    print(f"  joints               : {model.njnt}  (1 free + {model.njnt - 1} hinge)")
    print(f"  actuators            : {model.nu}")
    print(f"  nq / nv              : {model.nq} / {model.nv}")
    print(f"  geoms (excl. floor)  : {model.ngeom - 1}")
    print(f"  total mass           : {model.body_mass.sum():.4f} kg")
    print(f"  timestep             : {model.opt.timestep} s   decimation {meta['decimation']}"
          f"  -> policy {meta['policy_dt']} s")

    obs = 3 + 3 + 3 + 3 + 3 * model.nu
    print(f"  observation size     : {obs}   (gravity 3 + linvel 3 + angvel 3 + cmd 3"
          f" + qpos {model.nu} + qvel {model.nu} + last_action {model.nu})")
    print(f"  action size          : {model.nu}")

    bad = [model.body(i).name for i in range(1, model.nbody)
           if np.any(model.body_inertia[i] <= 0)]
    print(f"  non-positive inertia : {bad if bad else 'none'}")

    qpos = stance_qpos(model, meta)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    print("\n=== STANDING STANCE ===")
    print(f"  root (hips) height   : {qpos[2]:.4f} m")
    print(f"  head height          : {data.xipos[model.body('head').id][2]:.4f} m")
    print(f"  CoM height           : {data.subtree_com[0][2]:.4f} m")
    lo = np.array([s["range_rad"][0] for s in meta["joints"]])
    hi = np.array([s["range_rad"][1] for s in meta["joints"]])
    inside = bool(np.all(qpos[7:] >= lo - 1e-9) and np.all(qpos[7:] <= hi + 1e-9))
    print(f"  stance within limits : {inside}")
    if not inside:
        for index, spec in enumerate(meta["joints"]):
            value = qpos[7 + index]
            if value < lo[index] - 1e-9 or value > hi[index] + 1e-9:
                print(f"    OUT OF RANGE {spec['name']}: {np.degrees(value):.1f} deg"
                      f" not in {spec['range_deg']}")

    # Foot flatness: the sole should be parallel to the floor at the stance
    # angle, otherwise the model stands on an edge and the ankle target is wrong.
    for side in ("L", "R"):
        geom_id = model.geom(f"foot_{side}").id
        rot = data.geom_xmat[geom_id].reshape(3, 3)
        size = model.geom_size[geom_id]
        corners = np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3) * size
        z = (corners @ rot.T + data.geom_xpos[geom_id])[:, 2]
        sole = np.sort(z)[:4]
        tilt = np.degrees(np.arccos(np.clip(abs(rot[2, 2]), -1.0, 1.0)))
        print(f"  foot {side} sole spread : {sole.max() - sole.min():.5f} m"
              f"   tilt {tilt:.2f} deg   lowest {sole.min():.5f} m")

    fall_time, start, end = passive_fall(model, qpos)
    print("\n=== PASSIVE ZERO-ACTION DROP (actuation disabled) ===")
    print(f"  head height at t=0   : {start:.4f} m")
    print(f"  head height at t=10  : {end:.4f} m")
    print(f"  PASSIVE FALL TIME    : {fall_time:.3f} s   (head below 50% of standing)")
    verdict = "COLLAPSES (correct - no active control)" if np.isfinite(fall_time) \
        else "DID NOT COLLAPSE (wrong - model is self-supporting without actuation)"
    print(f"  verdict              : {verdict}")

    # Supplementary, not part of the gate: does the stance actually hold under
    # the authored gains? A stance that cannot be held means the kp values are
    # wrong and Phase 4 would spend its budget learning to stand up.
    hold = mujoco.MjData(model)
    hold.qpos[:] = qpos
    hold.ctrl[:] = qpos[7:]
    mujoco.mj_forward(model, hold)
    contacts_at_rest = int(hold.ncon)
    self_contacts = sum(
        1 for c in range(hold.ncon)
        if model.geom_type[hold.contact[c].geom1] != mujoco.mjtGeom.mjGEOM_PLANE
        and model.geom_type[hold.contact[c].geom2] != mujoco.mjtGeom.mjGEOM_PLANE
    )
    head_id = model.body("head").id
    peak = 0.0
    for _ in range(int(3.0 / model.opt.timestep)):
        mujoco.mj_step(model, hold)
        peak = max(peak, float(np.abs(hold.actuator_force).max()))
    drift = float(np.linalg.norm(hold.qpos[:2] - qpos[:2]))
    print("\n=== ACTUATED HOLD, 3 s (supplementary) ===")
    print(f"  contacts at rest     : {contacts_at_rest} ({self_contacts} self-collisions)")
    print(f"  head height after 3 s: {hold.xipos[head_id][2]:.4f} m"
          f"   (standing {data.xipos[head_id][2]:.4f} m)")
    print(f"  horizontal drift     : {drift:.4f} m")
    print(f"  peak actuator torque : {peak:.1f} N.m")

    meta["stance_qpos"] = qpos.tolist()
    meta["stance_root_height"] = float(qpos[2])
    meta["obs_size"] = obs
    meta["action_size"] = int(model.nu)
    (HERE / "mojucuboy_rig.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
