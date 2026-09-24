#!/usr/bin/env python3
"""Runtime checks of the Isaac Lab 3 / Newton (MuJoCo-Warp) worm task against WORM_SPEC.md.

    ISAAC\\isaaclab3\\.venv\\Scripts\\python.exe ISAAC\\scripts\\check_worm_physics_v3.py

The v3 counterpart of check_worm_physics.py (2.3 / PhysX). Newton's SolverMuJoCo builds a real
mjModel from the Newton model, so on top of the Isaac Lab-level checks this compares that model
field by field with training/worm/worm.xml compiled by MuJoCo itself (the MuJoCo trainer's model):

* joint/body order, action map, observation size
* masses (7.08 kg) and inertias, per body, vs worm.xml
* per joint: range +/-45 deg, kp 30, drive damping 0, force limit 6, armature 0.01, PASSIVE joint
  damping 2.0 (MuJoCo dof_damping), frictionloss 0, limit solref
* contacts: friction 0.9 * s (floor below it; MuJoCo max()), solref/solimp/condim as worm.xml,
  adjacent-segment exclusion == worm.xml <contact><exclude>, non-adjacent segments collide
* solver options vs worm.xml <option>, physics dt 0.005 s
* sign test (WORM_SPEC "Unity mapping"): j0_yaw +0.5 -> seg1 toward -y; j0_pitch +0.5 -> seg1 up
* free-decay damping test vs CPU MuJoCo on worm.xml (+ a no-damping control world)
* effort: Isaac Lab applied_torque == MuJoCo actuator_force (clip(kp*(target-q), +/-6))
* rest test, per-env friction slide test, one observation vector
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from isaaclab_tasks.utils.sim_launcher import add_launcher_args, launch_simulation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Worm5-Flat-Newton-Play-v0")
add_launcher_args(parser)
args = parser.parse_args()  # no --viz: headless (the env cfg defines no visualizers)

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import worm_tasks_v3  # noqa: E402,F401
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from worm_tasks_v3 import spec  # noqa: E402

FAILS = []
TILT_DEG = 50.0  # informative slide: must exceed atan(1.035) = 46 deg so every s slides


def check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}  {detail}")
    if not ok:
        FAILS.append(label)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    # this script calls sim.step() substep by substep; with the task's graphed decimation one sim.step()
    # would run all 4 substeps
    cfg.sim.use_newton_actuators = False
    with launch_simulation(cfg, args):
        return run(cfg)


def run(cfg):
    from isaaclab_newton.physics import NewtonManager
    from newton.solvers import SolverNotifyFlags

    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    sim = env.sim
    dt = sim.get_physics_dt()
    d = robot.data
    dev = env.device
    solver = NewtonManager._solver
    mm = solver.mj_model  # the mjModel Newton built (world 0 template)
    mw = solver.mjw_model  # MuJoCo-Warp model, batched per world
    ref = mujoco.MjModel.from_xml_path(os.path.join(REPO, "training", "worm", "worm.xml"))

    def mj_body(name):  # Newton names bodies by prim path; map the worm names
        for i in range(mm.nbody):
            if mm.body(i).name.endswith("_" + name):
                return i
        raise KeyError(name)

    def mj_joint(name):
        for i in range(mm.njnt):
            if mm.joint(i).name.endswith("_" + name):
                return i
        raise KeyError(name)

    print(f"\n== backend: {type(solver).__name__} (Newton) on {dev}")
    print(f"  Lab joints: {robot.joint_names}")
    print(f"  Lab bodies: {robot.body_names}")
    act_ids = env.action_manager.get_term("joint_pos")._joint_ids
    act_ids = list(range(robot.num_joints)) if isinstance(act_ids, slice) else [int(i) for i in act_ids]
    mapped = [robot.joint_names[i] for i in act_ids]
    check("action term joint order == spec ACTION_ORDER", mapped == spec.ACTION_ORDER, str(mapped))
    obs_dims = env.observation_manager.group_obs_dim["policy"]
    check("observation size 35", tuple(obs_dims) == (35,), str(obs_dims))

    print("\n== solver options vs worm.xml <option>")
    o, r = mm.opt, ref.opt
    for k in ("solver", "iterations", "ls_iterations", "integrator", "cone", "impratio", "tolerance"):
        a, b = getattr(o, k), getattr(r, k)
        check(f"opt.{k} = {a}", abs(float(a) - float(b)) < 1e-12, f"(worm.xml {b})")
    check("gravity (0, 0, -9.81)", np.allclose(mw.opt.gravity.numpy().reshape(-1, 3)[0], [0, 0, -9.81], atol=1e-5))
    check("Isaac physics dt 0.005 s, decimation 4", abs(dt - spec.PHYSICS_DT) < 1e-12 and env.cfg.decimation == 4)

    print("\n== mass / inertia (Newton-built mjModel vs worm.xml)")
    total = 0.0
    for name in robot.body_names:
        i, j = mj_body(name), ref.body(name).id
        total += float(mm.body_mass[i])
        ok = abs(mm.body_mass[i] - ref.body_mass[j]) < 1e-5 and np.allclose(mm.body_inertia[i], ref.body_inertia[j], atol=1e-8)
        ok &= np.allclose(mm.body_ipos[i], ref.body_ipos[j], atol=1e-7)
        check(f"{name:6s} m={mm.body_mass[i]:.6f} I={np.round(mm.body_inertia[i], 8).tolist()}", ok)
    lab_mass = d.body_mass.torch[0]
    check("total mass 7.08 kg (MuJoCo and Isaac Lab)", abs(total - spec.RIG["totalMass"]) < 1e-4
          and abs(float(lab_mass.sum()) - spec.RIG["totalMass"]) < 1e-4, f"{total:.5f} / {float(lab_mass.sum()):.5f}")

    print("\n== joints (Isaac Lab data + MuJoCo model)")
    lim = d.joint_pos_limits.torch[0]
    kp, kd = d.joint_stiffness.torch[0], d.joint_damping.torch[0]
    eff, arm = d.joint_effort_limits.torch[0], d.joint_armature.torch[0]
    dof_damp = mw.dof_damping.numpy()[0]
    for jn in spec.ACTION_ORDER:
        li = robot.joint_names.index(jn)
        mi, ri = mj_joint(jn), ref.joint(jn).id
        mdof, rdof = mm.jnt_dofadr[mi], ref.jnt_dofadr[ri]
        # actuator on this joint
        acts = [a for a in range(mm.nu) if mm.actuator_trnid[a][0] == mi]
        a = acts[0]
        row = (f"range [{float(lim[li, 0]):+.5f}, {float(lim[li, 1]):+.5f}] kp {float(kp[li]):.1f} drive-kd {float(kd[li]):.1f} "
               f"limit {float(eff[li]):.1f} armature {float(arm[li]):.3f} | mj: dof_damping {dof_damp[mdof]:.2f} "
               f"armature {mm.dof_armature[mdof]:.3f} frictionloss {mm.dof_frictionloss[mdof]:.1f} "
               f"solreflimit {mm.jnt_solref[mi].tolist()} actfrcrange {mm.jnt_actfrcrange[mi].tolist()} "
               f"gain {mm.actuator_gainprm[a][0]:.1f} bias {mm.actuator_biasprm[a][:3].tolist()}")
        ok = abs(float(lim[li, 1]) - 0.785398) < 1e-4 and abs(float(lim[li, 0]) + 0.785398) < 1e-4
        ok &= abs(float(kp[li]) - spec.KP) < 1e-5 and abs(float(kd[li])) < 1e-9
        ok &= abs(float(eff[li]) - spec.FORCE_LIMIT) < 1e-5 and abs(float(arm[li]) - spec.ARMATURE) < 1e-7
        ok &= abs(dof_damp[mdof] - ref.dof_damping[rdof]) < 1e-6 and abs(mm.dof_armature[mdof] - ref.dof_armature[rdof]) < 1e-7
        ok &= mm.dof_frictionloss[mdof] == 0.0 and np.allclose(mm.jnt_range[mi], ref.jnt_range[ri], atol=1e-6)
        ok &= np.allclose(mm.jnt_solref[mi], ref.jnt_solref[ri]) and np.allclose(mm.jnt_solimp[mi], ref.jnt_solimp[ri])
        ok &= bool(mm.jnt_actfrclimited[mi]) and np.allclose(mm.jnt_actfrcrange[mi], [-6, 6])
        ok &= abs(mm.actuator_gainprm[a][0] - spec.KP) < 1e-5 and np.allclose(mm.actuator_biasprm[a][:3], [0, -spec.KP, 0])
        ok &= np.allclose(mm.jnt_axis[mi], ref.jnt_axis[ri])
        check(f"{jn:9s}", ok, row)

    print("\n== contacts (geoms)")
    gf = mw.geom_friction.numpy()[0]
    seg_geoms = {}
    ground = None
    for g in range(mm.ngeom):
        b = mm.geom_bodyid[g]
        if b == 0:
            ground = g
        else:
            seg_geoms[mm.body(b).name.split("_")[-1]] = g
    rg = ref.geom("seg0").id
    for name, g in sorted(seg_geoms.items()):
        ok = abs(gf[g][0] - spec.FRICTION) < 1e-6 and np.allclose(mm.geom_solref[g], ref.geom_solref[rg])
        ok &= np.allclose(mm.geom_solimp[g], ref.geom_solimp[rg]) and mm.geom_condim[g] == ref.geom_condim[rg]
        ok &= np.allclose(mm.geom_size[g][:2], ref.geom_size[rg][:2]) and mm.geom_margin[g] == ref.geom_margin[rg]
        check(f"{name} capsule r {mm.geom_size[g][0]:.3f} half {mm.geom_size[g][1]:.3f} friction {gf[g].round(5).tolist()} "
              f"solref {mm.geom_solref[g].tolist()} condim {mm.geom_condim[g]}", ok)
    fl = ref.geom("floor").id
    check(f"floor friction {gf[ground][0]:.3f} < 0.765 (MuJoCo max() -> worm's 0.9*s); solref {mm.geom_solref[ground].tolist()}",
          gf[ground][0] < 0.9 * spec.FRICTION_SCALE[0] and np.allclose(mm.geom_solref[ground], ref.geom_solref[fl])
          and np.allclose(mm.geom_solimp[ground], ref.geom_solimp[fl]))
    excl = {(int(s) >> 16, int(s) & 0xFFFF) for s in mm.exclude_signature}
    segs = spec.SEGMENT_NAMES

    def excluded(a, b):
        ia, ib = mj_body(a), mj_body(b)
        return (min(ia, ib), max(ia, ib)) in excl

    def can_collide(ga, gb):
        return bool((mm.geom_contype[ga] & mm.geom_conaffinity[gb]) or (mm.geom_contype[gb] & mm.geom_conaffinity[ga]))

    adj = [(segs[i], segs[i + 1]) for i in range(4)]
    non_adj = [(segs[i], segs[j]) for i in range(5) for j in range(i + 2, 5)]
    ref_excl = sorted((ref.body(int(s) >> 16).name, ref.body(int(s) & 0xFFFF).name) for s in ref.exclude_signature)
    check("adjacent segment pairs excluded (== worm.xml <exclude>)", all(excluded(a, b) for a, b in adj) and sorted(adj) == ref_excl,
          str(adj))
    check("non-adjacent segment pairs collide", all(not excluded(a, b) and can_collide(seg_geoms[a], seg_geoms[b]) for a, b in non_adj),
          str(non_adj))
    check("every segment collides with the floor", all(can_collide(g, ground) for g in seg_geoms.values()))

    ids = {n: i for i, n in enumerate(robot.joint_names)}
    bid = {n: i for i, n in enumerate(robot.body_names)}
    nj = robot.num_joints

    def place(env_ids, joint_pos, joint_vel=None, z=1.0, vel=None):
        e = torch.tensor(env_ids, device=dev, dtype=torch.int32)
        n = len(env_ids)
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :3] = env.scene.env_origins[e.long()]
        pose[:, 2] += z
        pose[:, 6] = 1.0
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=e)
        v = torch.zeros(n, 6, device=dev) if vel is None else vel
        robot.write_root_velocity_to_sim_index(root_velocity=v, env_ids=e)
        jv = torch.zeros_like(joint_pos) if joint_vel is None else joint_vel
        robot.write_joint_state_to_sim_index(position=joint_pos, velocity=jv, env_ids=e)
        robot.set_joint_position_target_index(target=joint_pos, env_ids=e)

    def step(n):
        for _ in range(n):
            robot.write_data_to_sim()
            sim.step(render=False)
            env.scene.update(dt)

    step(1)
    check("MuJoCo-Warp timestep == 0.005 s during stepping", abs(float(mw.opt.timestep.numpy().reshape(-1)[0]) - 0.005) < 1e-9,
          f"{float(mw.opt.timestep.numpy().reshape(-1)[0])}")

    print("\n== sign test (WORM_SPEC 'Unity mapping'): +0.5 rad, seg1 relative to seg0's frame")
    for jn, axis, sign in (("j0_yaw", 1, -1), ("j0_pitch", 2, +1)):
        q = torch.zeros(1, nj, device=dev)
        q[0, ids[jn]] = 0.5
        place([0], q, z=1.0)
        step(1)
        pw, qw = d.body_link_pos_w.torch, d.body_link_quat_w.torch
        rel = quat_apply_inverse(qw[0, bid["seg0"]].unsqueeze(0), (pw[0, bid["seg1"]] - pw[0, bid["seg0"]]).unsqueeze(0))[0]
        want = "-y" if sign < 0 else "+z"
        check(f"{jn}=+0.5 swings seg1 toward MuJoCo {want}", float(rel[axis]) * sign > 0.03,
              f"seg1 in seg0 frame = ({float(rel[0]):+.4f}, {float(rel[1]):+.4f}, {float(rel[2]):+.4f}); "
              f"q={float(d.joint_pos.torch[0, ids[jn]]):+.4f}")

    print("\n== effort term: Isaac Lab applied_torque vs MuJoCo actuator_force (saturating targets)")
    q = torch.zeros(1, nj, device=dev)
    place([0], q, z=1.0)
    tgt = torch.tensor([[0.7, -0.7, 0.05, -0.02, 0.3, 0.1, -0.5, 0.01]], device=dev)
    robot.set_joint_position_target_index(target=tgt, env_ids=torch.tensor([0], device=dev, dtype=torch.int32))
    step(1)
    lab_tau = d.applied_torque.torch[0].cpu().numpy()
    mj_force = solver.mjw_data.actuator_force.numpy()[0]
    qfrc = solver.mjw_data.qfrc_actuator.numpy()[0][6:]
    act_order = [robot.joint_names.index(mm.joint(mm.actuator_trnid[a][0]).name.split("_joints_")[-1]) for a in range(mm.nu)]
    mj_clip = np.clip(mj_force, -6, 6)
    print(f"  Lab applied_torque      : {np.round(lab_tau, 4).tolist()}")
    print(f"  MuJoCo actuator_force   : {np.round(mj_force[np.argsort(act_order)], 4).tolist()}")
    print(f"  MuJoCo qfrc_actuator    : {np.round(qfrc, 4).tolist()}  (after jnt_actfrcrange +/-6)")
    check("applied_torque == clip(actuator_force, +/-6) == qfrc_actuator (first substep of the step)",
          np.allclose(lab_tau, mj_clip[np.argsort(act_order)], atol=1e-3) and np.allclose(lab_tau, qfrc, atol=1e-3))

    print("\n== damping test: free fall, kp 0, j3_yaw qdot0 = 5 rad/s  (vs CPU MuJoCo on worm.xml)")
    e3 = torch.tensor([0, 1, 2], device=dev, dtype=torch.int32)
    robot.write_joint_stiffness_to_sim_index(stiffness=0.0, env_ids=e3)
    # world 1: no passive damping; world 2: no passive damping, drive damping 2.0 instead
    kd = torch.zeros(3, nj, device=dev)
    kd[2] = spec.JOINT_DAMPING
    robot.write_joint_damping_to_sim_index(damping=kd, env_ids=e3)
    pdamp = wp.to_torch(NewtonManager.get_model().mujoco.dof_passive_damping)
    dofs_per_world = pdamp.shape[0] // env.num_envs
    saved = pdamp.clone()
    for w in (1, 2):
        pdamp[w * dofs_per_world + 6: (w + 1) * dofs_per_world] = 0.0
    NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
    for a in robot.actuators.values():
        a.stiffness[e3.long()] = 0.0
        a.damping[e3.long()] = kd
    q = torch.zeros(3, nj, device=dev)
    qd = torch.zeros_like(q)
    qd[:, ids["j3_yaw"]] = 5.0
    place([0, 1, 2], q, qd, z=2.0)
    trace = []
    for _ in range(10):
        step(1)
        trace.append([float(v) for v in d.joint_vel.torch[:3, ids["j3_yaw"]]])
    print(f"  mjw dof_damping world0/1/2 (j3_yaw): {[float(mw.dof_damping.numpy()[w][6 + ids['j3_yaw']]) for w in range(3)]}")
    # CPU MuJoCo reference on worm.xml: same free fall, kp 0 (actuators off), same initial qvel
    m2 = mujoco.MjModel.from_xml_path(os.path.join(REPO, "training", "worm", "worm.xml"))
    m2.actuator_gainprm[:, 0] = 0.0
    m2.actuator_biasprm[:, 1] = 0.0
    dd = mujoco.MjData(m2)
    dd.qpos[2] = 2.0
    dd.qvel[ref.joint("j3_yaw").dofadr[0]] = 5.0
    cpu = []
    for _ in range(10):
        mujoco.mj_step(m2, dd)
        cpu.append(float(dd.qvel[ref.joint("j3_yaw").dofadr[0]]))
    for lbl, i in (("passive damping 2.0 (task)", 0), ("no damping           ", 1), ("drive damping 2.0    ", 2)):
        print(f"  {lbl}: qdot after 1/4/10 substeps = {trace[0][i]:.4f} / {trace[3][i]:.4f} / {trace[9][i]:.4f}")
    print(f"  CPU MuJoCo worm.xml       : qdot after 1/4/10 substeps = {cpu[0]:.4f} / {cpu[3]:.4f} / {cpu[9]:.4f}")
    check("passive joint damping == CPU MuJoCo on worm.xml (10 substeps)",
          max(abs(trace[k][0] - cpu[k]) for k in range(10)) < 0.02, f"max |diff| {max(abs(trace[k][0] - cpu[k]) for k in range(10)):.4f}")
    check("damping acts (well below the undamped world)", trace[9][0] < 0.8 * trace[9][1])
    # restore
    pdamp.copy_(saved)
    NewtonManager.add_model_change(SolverNotifyFlags.JOINT_DOF_PROPERTIES)
    robot.write_joint_stiffness_to_sim_index(stiffness=spec.KP, env_ids=e3)
    robot.write_joint_damping_to_sim_index(damping=0.0, env_ids=e3)
    for a in robot.actuators.values():
        a.stiffness[e3.long()] = spec.KP
        a.damping[e3.long()] = 0.0

    print("\n== randomisation: lean per-world mass / kp writers vs Newton's own model-change path")
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    import worm_tasks_v3.mdp as wmdp

    seg_cfg = SceneEntityCfg("robot", body_names=spec.SEGMENT_NAMES, preserve_order=True)
    jnt_cfg = SceneEntityCfg("robot", joint_names=spec.ACTION_ORDER, preserve_order=True)
    seg_cfg.resolve(env.scene)
    jnt_cfg.resolve(env.scene)
    step(1)  # flush the damping test's pending kp restore into MuJoCo-Warp
    mterm = wmdp.randomize_segment_mass_lean(EventTermCfg(func=wmdp.randomize_segment_mass_lean, mode="reset",
                                             params={"asset_cfg": seg_cfg, "scale_range": spec.MASS_SCALE}), env)
    kterm = wmdp.randomize_kp_lean(EventTermCfg(func=wmdp.randomize_kp_lean, mode="reset",
                                   params={"asset_cfg": jnt_cfg, "scale_range": spec.KP_SCALE}), env)
    torch.manual_seed(3)
    ids02 = torch.tensor([0, 2], device=dev)
    mterm(env, ids02, seg_cfg, spec.MASS_SCALE)
    kterm(env, ids02, jnt_cfg, spec.KP_SCALE)
    mjm = wp.to_torch(mw.body_mass).clone()
    mji = wp.to_torch(mw.body_inertia).clone()
    gain = wp.to_torch(mw.actuator_gainprm)[..., 0].clone()
    bias = wp.to_torch(mw.actuator_biasprm)[..., 1].clone()
    lab_m = d.body_mass.torch.clone()
    lab_kp = d.joint_stiffness.torch.clone()
    nominal_m = torch.tensor([float(ref.body(n).mass[0]) for n in robot.body_names], device=dev)
    seg_idx = [bid[n] for n in spec.SEGMENT_NAMES]
    link_idx = [bid[n] for n in robot.body_names if n.startswith("link")]
    ratio = lab_m[:, seg_idx] / nominal_m[seg_idx]
    print(f"  mass scale world0 {ratio[0].cpu().numpy().round(4).tolist()}  world2 {ratio[2].cpu().numpy().round(4).tolist()}")
    print(f"  kp world0 {lab_kp[0].cpu().numpy().round(3).tolist()}")
    check("segment masses x U(0.9, 1.1) in the reset worlds only, links untouched",
          bool(torch.all((ratio[[0, 2]] >= 0.9) & (ratio[[0, 2]] <= 1.1))) and bool(torch.allclose(ratio[[1, 3]], torch.ones_like(ratio[[1, 3]])))
          and bool(torch.allclose(lab_m[:, link_idx], nominal_m[link_idx].expand(4, -1))) and len(set(ratio[0].tolist())) == 5)
    check("kp x U(0.8, 1.2) per joint in the reset worlds only",
          bool(torch.all((lab_kp[[0, 2]] >= 24.0) & (lab_kp[[0, 2]] <= 36.0))) and bool(torch.allclose(lab_kp[[1, 3]], torch.full_like(lab_kp[[1, 3]], 30.0))))
    # now let Newton rebuild every per-world MuJoCo field from its own model (the slow official path)
    NewtonManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES | SolverNotifyFlags.JOINT_DOF_PROPERTIES)
    step(1)
    same = (torch.allclose(wp.to_torch(mw.body_mass), mjm, rtol=1e-5) and torch.allclose(wp.to_torch(mw.body_inertia), mji, rtol=1e-4)
            and torch.allclose(wp.to_torch(mw.actuator_gainprm)[..., 0], gain) and torch.allclose(wp.to_torch(mw.actuator_biasprm)[..., 1], bias))
    check("MuJoCo-Warp body_mass / body_inertia / actuator gain+bias unchanged after Newton's full notify", same,
          f"(max |d mass| {float((wp.to_torch(mw.body_mass) - mjm).abs().max()):.2e}, "
          f"|d inertia| {float((wp.to_torch(mw.body_inertia) - mji).abs().max()):.2e})")
    # effect on the servo: applied torque in world 0 uses the randomised kp
    q = torch.zeros(1, nj, device=dev)
    place([0], q, z=1.0)
    tgt = torch.full((1, nj), 0.05, device=dev)
    robot.set_joint_position_target_index(target=tgt, env_ids=torch.tensor([0], device=dev, dtype=torch.int32))
    step(1)
    qfrc = solver.mjw_data.qfrc_actuator.numpy()[0][6:]
    lab_tau = d.applied_torque.torch[0].cpu().numpy()
    q_before = 0.0  # joints started at 0 for this single substep; the servo force is kp * 0.05
    check("servo torque uses the randomised kp (Lab applied_torque == MuJoCo qfrc_actuator == kp*0.05)",
          np.allclose(lab_tau, qfrc, atol=1e-4) and np.allclose(lab_tau, lab_kp[0].cpu().numpy() * (0.05 - q_before), rtol=1e-4),
          f"{np.round(lab_tau, 4).tolist()}")
    # back to nominal for the remaining tests
    all4 = torch.arange(4, device=dev)
    mterm(env, all4, seg_cfg, (1.0, 1.0))
    kterm(env, all4, jnt_cfg, (1.0, 1.0))

    print("\n== rest test: spec reset, zero action, 2 s")
    obs, _ = env.reset()
    ob = obs["policy"][0]
    print(f"  obs[0] after reset: {[round(float(v), 4) for v in ob]}")
    check("gravity_b ~ (0, 0, -1)", float(ob[2]) < -0.99)
    rq = d.root_link_quat_w.torch[0]  # xyzw
    yaw_head = 2 * math.atan2(float(rq[2]), float(rq[3]))
    check("goal_dir_b ~ (cos yaw, -sin yaw)", abs(float(ob[33]) - math.cos(yaw_head)) < 0.1 and abs(float(ob[34]) + math.sin(yaw_head)) < 0.1,
          f"yaw(head) {math.degrees(yaw_head):+.1f} deg")
    zero = torch.zeros(env.num_envs, 8, device=dev)
    for _ in range(100):
        obs, rew, term, trunc, extras = env.step(zero)
    z = d.body_link_pos_w.torch[:, [bid[f"seg{i}"] for i in range(5)], 2] - env.scene.env_origins[:, 2:3]
    v = d.body_link_lin_vel_w.torch[:, bid["seg2"]]
    print(f"  segment heights env0: {[round(float(x), 4) for x in z[0]]}   seg2 |v| max over envs {float(v.norm(dim=-1).max()):.4f}")
    print(f"  per-step reward env0 (zero action, at rest): {float(rew[0]):.6f}")
    check("worm rests on the floor (z ~ radius 0.045)", bool(torch.all((z - 0.045).abs() < 0.01)))
    check("worm at rest", float(v.norm(dim=-1).max()) < 0.02)

    # Friction. The 2.3 test (kick the worm to 1 m/s, time the deceleration, solve for mu) does not carry
    # over to MuJoCo contacts: an impulsive start against the PYRAMIDAL cone gives a matching normal
    # impulse (each pyramid edge pushes along n +/- mu*t) and the worm hops; even a slow slide on MuJoCo's
    # soft contacts hops, in CPU MuJoCo on worm.xml as much as here, so a closed-form mu is not a valid
    # pass criterion on EITHER side. The exact check is the pair friction MuJoCo-Warp's constraint solver
    # actually uses, per contact (d.contact.friction[0], after max() combination), which must be 0.9*s.
    print("\n== friction: per-env randomised scale s, pair friction in every live worm-floor contact")
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    import worm_tasks_v3.mdp as wmdp

    fcfg = EventTermCfg(func=wmdp.randomize_friction_scale, mode="reset",
                        params={"asset_cfg": SceneEntityCfg("robot"), "base": spec.FRICTION, "scale_range": spec.FRICTION_SCALE})
    fterm = wmdp.randomize_friction_scale(fcfg, env)
    all_ids = list(range(env.num_envs))
    e_all = torch.arange(env.num_envs, device=dev, dtype=torch.int32)
    # 1) the event term itself: draw, then read back what reached MuJoCo-Warp
    fterm(env, e_all, **fcfg.params)
    q = torch.zeros(env.num_envs, nj, device=dev)
    place(all_ids, q, z=spec.RIG["radius"] + 0.0005)
    step(40)  # settle; also flushes the SHAPE_PROPERTIES notification into mjw
    s_drawn = fterm.scale.cpu().numpy()
    con = solver.mjw_data.contact
    n = int(solver.mjw_data.nacon.numpy()[0])
    cw = con.worldid.numpy()[:n]
    cg = con.geom.numpy()[:n]
    cf = con.friction.numpy()[:n]
    cs = con.solref.numpy()[:n]
    cdim = con.dim.numpy()[:n]
    for w in all_ids:
        sel = (cw == w) & ((cg[:, 0] == ground) | (cg[:, 1] == ground))
        want = spec.FRICTION * float(s_drawn[w])
        mus = cf[sel, 0]
        ok = sel.sum() >= 5 and np.allclose(mus, want, atol=1e-5) and np.allclose(cs[sel], [0.01, 1.0], atol=1e-6)
        # contact friction = (tangent1, tangent2, spin, roll1, roll2); condim 3 uses only the two tangents
        ok &= bool(np.all(cdim[sel] == 3)) and np.allclose(cf[sel, 1], mus, atol=1e-7)
        ok &= np.allclose(cf[sel, 2], 0.005, atol=1e-7)
        check(f"env{w}: s={s_drawn[w]:.4f} -> {int(sel.sum())} worm-floor contacts, mu {np.unique(mus.round(5)).tolist()} "
              f"== 0.9*s = {want:.5f}; solref (0.01, 1); condim 3", ok)
    check("friction scale drawn in U(0.85, 1.15), continuous", bool(np.all((s_drawn >= 0.85) & (s_drawn <= 1.15)))
          and len(set(np.round(s_drawn, 6))) == env.num_envs, str(np.round(s_drawn, 4).tolist()))

    # 2) informative: the same tilted slide in Newton and in CPU MuJoCo on worm.xml (trainer semantics)
    th = math.radians(TILT_DEG)
    print(f"  informative: joints locked (+/-1e-4 rad), gravity tilted {TILT_DEG:.0f} deg, 0.3 s slide from rest")
    scales = torch.tensor([0.85, 1.0, 1.15, 0.9], device=dev)[: env.num_envs]
    mu = wp.to_torch(fterm._mu)
    mu[:] = (spec.FRICTION * scales).unsqueeze(-1)
    NewtonManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)
    LOCK = 1e-4
    lim_saved = d.joint_pos_limits.torch.clone()
    lock = torch.zeros(env.num_envs, nj, 2, device=dev)
    lock[..., 0], lock[..., 1] = -LOCK, LOCK
    robot.write_joint_position_limit_to_sim_index(limits=lock, env_ids=e_all, warn_limit_violation=False)
    place(all_ids, q, z=spec.RIG["radius"] + 0.0005)
    step(40)
    x0 = (d.body_link_pos_w.torch[:, bid["seg2"], 0] - env.scene.env_origins[:, 0]).clone()
    grav = wp.to_torch(NewtonManager.get_model().gravity)
    g_saved = grav.clone()
    grav[:] = torch.tensor([9.81 * math.sin(th), 0.0, -9.81 * math.cos(th)], device=grav.device)
    NewtonManager.add_model_change(SolverNotifyFlags.MODEL_PROPERTIES)
    step(60)
    x1 = (d.body_link_pos_w.torch[:, bid["seg2"], 0] - env.scene.env_origins[:, 0]).clone()
    for i in all_ids:
        s = float(scales[i])
        m3 = mujoco.MjModel.from_xml_path(os.path.join(REPO, "training", "worm", "worm.xml"))
        m3.geom_friction[:, 0] = spec.FRICTION * s  # every geom incl. the floor (MuJoCo trainer, spec item 14)
        m3.jnt_range[1:] = [-LOCK, LOCK]
        d3 = mujoco.MjData(m3)
        d3.qpos[2] = spec.RIG["radius"] + 0.0005
        for _ in range(40):
            mujoco.mj_step(m3, d3)
        cx0 = float(d3.xpos[ref.body("seg2").id][0])
        m3.opt.gravity[:] = [9.81 * math.sin(th), 0.0, -9.81 * math.cos(th)]
        for _ in range(60):
            mujoco.mj_step(m3, d3)
        cx1 = float(d3.xpos[ref.body("seg2").id][0])
        print(f"    s={s:.2f} (mu {spec.FRICTION * s:.3f}): slide Newton {float(x1[i] - x0[i]):.4f} m, "
              f"CPU MuJoCo worm.xml {cx1 - cx0:.4f} m")
    grav.copy_(g_saved)
    NewtonManager.add_model_change(SolverNotifyFlags.MODEL_PROPERTIES)
    robot.write_joint_position_limit_to_sim_index(limits=lim_saved, env_ids=e_all, warn_limit_violation=False)


    print("\n" + ("ALL RUNTIME CHECKS PASSED" if not FAILS else f"FAILED: {FAILS}"))
    env.close()
    return 0 if not FAILS else 1


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(code)
