#!/usr/bin/env python3
"""Runtime checks of the Isaac Lab 3 / Newton (MuJoCo-Warp) quad task against QUAD_SPEC.md and quad.xml.

    ISAAC\\isaaclab3\\.venv\\Scripts\\python.exe ISAAC\\scripts\\check_quad_physics_v3.py

The quad counterpart of check_worm_physics_v3.py. Newton's SolverMuJoCo builds a real mjModel from the
Newton model, so besides the Isaac Lab-level checks this compares that model field by field with
training/quad/quad.xml compiled by MuJoCo itself (what the MuJoCo trainer simulates), and runs the same
motions in both:

* action order (quad_rig.json actuator order), observation size 36
* solver options vs quad.xml <option>, gravity, dt 0.005 s, decimation 4
* bodies (9), masses (90 kg), inertias, centres of mass
* per hinge: range, kp 1500, drive damping 0, force limit 300, armature 0, PASSIVE damping 100,
  frictionloss 0, limit solref as MuJoCo K/B (Newton writes the direct form), solimp, actuator law
* geoms: type, size, friction 0.9 (floor below it; MuJoCo max()), solref, solimp, condim, margin
* self-collision rules: the set of body pairs MuJoCo's filter lets collide == quad.xml's
  (every pair except parent-child), and a random-action run with no deep self-penetration
* kinematics: random joint angles -> every body pose and foot point == CPU MuJoCo on quad.xml
* sign tests: one hip and one knee at +0.5 rad (foot swings toward -x), vs CPU MuJoCo
* effort term: applied_torque == MuJoCo actuator_force (clip(kp * (target - q), +/-300))
* passive damping free decay and a joint-limit push, each vs CPU MuJoCo
* rest stand: spec reset, zero action, 5 s: torso ~0.90 m, upright, all feet in contact, no slip, no
  termination; same stand in CPU MuJoCo
* per-env friction 0.9 * s in every live foot-floor contact; mass / kp randomisers; the push event;
  the fall termination
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
sys.path.insert(0, HERE)

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Quad-Flat-Newton-Play-v0")
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402

import quad_tasks_v3  # noqa: E402,F401
from convert_mjcf_v3 import collidable_body_pairs  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from quad_tasks_v3 import spec  # noqa: E402

FAILS = []
QUAD_XML = spec.QUAD_XML


def check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}  {detail}")
    if not ok:
        FAILS.append(label)


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    cfg.sim.use_newton_actuators = False  # this script steps substep by substep
    with launch_simulation(cfg, args):
        return run(cfg)


def kb(solref, dmax, dt, refsafe=True):
    """MuJoCo's reference-acceleration gains (K, B) for a solref at impedance dmax."""
    a, b = float(solref[0]), float(solref[1])
    if a > 0:
        tc = max(a, 2 * dt) if refsafe else a
        return 1.0 / (dmax * dmax * tc * tc * b * b), 2.0 / (dmax * tc)
    return -a / (dmax * dmax), -b / dmax


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
    mm = solver.mj_model   # the mjModel Newton built (world-0 template)
    mw = solver.mjw_model  # MuJoCo-Warp model, batched per world
    ref = mujoco.MjModel.from_xml_path(QUAD_XML)
    usd2mj_body = {v: k for k, v in spec.NAMES["bodies"].items()}
    usd2mj_joint = {v: k for k, v in spec.NAMES["joints"].items()}

    def mj_body(usd_name):
        hits = [i for i in range(mm.nbody) if mm.body(i).name.endswith("_" + usd_name)]
        assert len(hits) == 1, (usd_name, hits)
        return hits[0]

    def mj_joint(usd_name):
        hits = [i for i in range(mm.njnt) if mm.joint(i).name.endswith("_" + usd_name)]
        assert len(hits) == 1, (usd_name, hits)
        return hits[0]

    print(f"\n== backend: {type(solver).__name__} (Newton) on {dev}")
    print(f"  Lab joints: {robot.joint_names}")
    print(f"  Lab bodies: {robot.body_names}")
    act_ids = env.action_manager.get_term("joint_pos")._joint_ids
    act_ids = list(range(robot.num_joints)) if isinstance(act_ids, slice) else [int(i) for i in act_ids]
    mapped = [robot.joint_names[i] for i in act_ids]
    check("action term joint order == quad_rig.json actionOrder", mapped == spec.ACTION_ORDER
          and [usd2mj_joint[n] for n in mapped] == [a["joint"] for a in spec.RIG["actuators"]], str(mapped))
    obs_dims = env.observation_manager.group_obs_dim["policy"]
    check(f"observation size {spec.NUM_OBS} (round {spec.ROUND})", tuple(obs_dims) == (spec.NUM_OBS,), str(obs_dims))

    print("\n== solver options vs quad.xml <option>")
    o, r = mm.opt, ref.opt
    # (the timestep is checked on the MuJoCo-Warp model below: Newton's CPU template mjModel keeps MuJoCo's
    #  0.002 default, but the batched model it steps with carries sim.dt = 0.005)
    for k in ("solver", "iterations", "ls_iterations", "integrator", "cone", "impratio", "tolerance"):
        a, b = getattr(o, k), getattr(r, k)
        check(f"opt.{k} = {a}", abs(float(a) - float(b)) < 1e-12, f"(quad.xml {b})")
    fp = mujoco.mjtDisableBit.mjDSBL_FILTERPARENT
    check("filterparent enabled in both", not (mm.opt.disableflags & fp) and not (ref.opt.disableflags & fp),
          f"disableflags newton {mm.opt.disableflags} / quad.xml {ref.opt.disableflags}")
    check("gravity (0, 0, -9.81)", np.allclose(mw.opt.gravity.numpy().reshape(-1, 3)[0], [0, 0, -9.81], atol=1e-5))
    check("Isaac physics dt 0.005 s, decimation 10, step (reward) dt 0.05 s, 400-step episodes (QUAD_SPEC round 3)",
          abs(dt - spec.PHYSICS_DT) < 1e-12 and env.cfg.decimation == 10 and abs(env.step_dt - 0.05) < 1e-12
          and env.max_episode_length == 400, f"step_dt {env.step_dt}, {env.max_episode_length} steps")
    rw = {n: env.reward_manager.get_term_cfg(n).weight for n in env.reward_manager.active_terms}
    want_w = {"alive": 1.0, "speed": 2.0, "progress": 1.0, "heading": 0.1, "upright": 0.2, "lateral_drift": -0.5,
              "effort": -0.2, "action_rate": -0.2, "feet_air_time": 0.5, "flight": -1.0, "vertical_bounce": -2.0,
              "foot_slip": -1.0}
    if spec.GAIT:
        want_w.update({"gait_ref": 2.0, "contact_phase": 2.0 if spec.ROUND >= 9 else 0.5})
    sig = env.reward_manager.get_term_cfg("speed").params["sigma"]
    gsig = env.reward_manager.get_term_cfg("gait_ref").params["sigma"] if spec.GAIT else None
    check(f"reward terms and weights == QUAD_SPEC round {spec.ROUND} (speed sigma 1.0"
          + (f", gait_ref sigma {0.2 if spec.ROUND >= 9 else 0.3})" if spec.GAIT else ")"),
          rw == want_w and sig == 1.0 and (not spec.GAIT or gsig == (0.2 if spec.ROUND >= 9 else 0.3)), f"{rw}, sigma {sig}, {gsig}")
    from quad_tasks_v3.agents.capped_ppo import CappedPPO
    from quad_tasks_v3.agents.rsl_rl_ppo_cfg import QuadFlatPPORunnerCfg

    alg = QuadFlatPPORunnerCfg().algorithm
    probe = CappedPPO.__new__(CappedPPO)
    probe.learning_rate = 1e-2
    check("PPO: adaptive KL 0.01, lr 3e-4, CappedPPO clamps RSL-RL's raise to 3e-4",
          alg.schedule == "adaptive" and alg.desired_kl == 0.01 and alg.learning_rate == 3e-4
          and alg.class_name.endswith(":CappedPPO") and probe.learning_rate == 3e-4)

    print("\n== bodies: mass / inertia / com (Newton-built mjModel vs quad.xml)")
    total = 0.0
    for name in robot.body_names:
        i, j = mj_body(name), ref.body(usd2mj_body[name]).id
        total += float(mm.body_mass[i])
        ok = abs(mm.body_mass[i] - ref.body_mass[j]) < 1e-5 and np.allclose(mm.body_inertia[i], ref.body_inertia[j], rtol=1e-5)
        ok &= np.allclose(mm.body_ipos[i], ref.body_ipos[j], atol=1e-6)
        check(f"{name:12s} m={mm.body_mass[i]:7.3f} I={np.round(mm.body_inertia[i], 6).tolist()}", ok)
    lab_total = float(d.body_mass.torch[0].sum())
    check("total mass 90 kg (MuJoCo and Isaac Lab)", abs(total - spec.TOTAL_MASS) < 1e-3 and abs(lab_total - spec.TOTAL_MASS) < 1e-3,
          f"{total:.4f} / {lab_total:.4f}")

    print("\n== hinges (Isaac Lab data + Newton-built mjModel vs quad.xml)")
    lim = d.joint_pos_limits.torch[0]
    kp, kd = d.joint_stiffness.torch[0], d.joint_damping.torch[0]
    eff, arm = d.joint_effort_limits.torch[0], d.joint_armature.torch[0]
    dof_damp = mw.dof_damping.numpy()[0]
    for jn in spec.ACTION_ORDER:
        li = robot.joint_names.index(jn)
        mi, ri = mj_joint(jn), ref.joint(usd2mj_joint[jn]).id
        mdof, rdof = mm.jnt_dofadr[mi], ref.jnt_dofadr[ri]
        a = [x for x in range(mm.nu) if mm.actuator_trnid[x][0] == mi][0]
        ra = [x for x in range(ref.nu) if ref.actuator_trnid[x][0] == ri][0]
        dmax = float(ref.jnt_solimp[ri][1])
        k_new, b_new = kb(mm.jnt_solref[mi], float(mm.jnt_solimp[mi][1]), mm.opt.timestep)
        k_ref, b_ref = kb(ref.jnt_solref[ri], dmax, ref.opt.timestep)
        row = (f"range [{float(lim[li, 0]):+.4f}, {float(lim[li, 1]):+.4f}] kp {float(kp[li]):.0f} drive-kd {float(kd[li]):.1f} "
               f"limit {float(eff[li]):.0f} armature {float(arm[li]):.3f} | mj: damping {dof_damp[mdof]:.0f} "
               f"frictionloss {mm.dof_frictionloss[mdof]:.1f} solreflimit {np.round(mm.jnt_solref[mi], 4).tolist()} "
               f"(K,B)=({k_new:.4g},{b_new:.4g}) vs quad.xml {ref.jnt_solref[ri].tolist()} ({k_ref:.4g},{b_ref:.4g}) "
               f"actfrcrange {mm.jnt_actfrcrange[mi].tolist()} gain {mm.actuator_gainprm[a][0]:.0f}")
        ok = np.allclose([float(lim[li, 0]), float(lim[li, 1])], ref.jnt_range[ri], atol=1e-5)
        ok &= np.allclose(mm.jnt_range[mi], ref.jnt_range[ri], atol=1e-6)
        ok &= abs(float(kp[li]) - spec.KP) < 1e-3 and abs(float(kd[li])) < 1e-9
        ok &= abs(float(eff[li]) - spec.FORCE_LIMIT) < 1e-4 and abs(float(arm[li]) - spec.ARMATURE) < 1e-9
        ok &= abs(dof_damp[mdof] - ref.dof_damping[rdof]) < 1e-4 and abs(mm.dof_armature[mdof] - ref.dof_armature[rdof]) < 1e-9
        ok &= mm.dof_frictionloss[mdof] == ref.dof_frictionloss[rdof] == 0.0
        ok &= abs(k_new - k_ref) < 1e-6 * k_ref and abs(b_new - b_ref) < 1e-6 * b_ref
        ok &= np.allclose(mm.jnt_solimp[mi], ref.jnt_solimp[ri]) and np.allclose(mm.jnt_axis[mi], ref.jnt_axis[ri])
        ok &= bool(mm.jnt_actfrclimited[mi]) and np.allclose(mm.jnt_actfrcrange[mi], ref.actuator_forcerange[ra])
        ok &= abs(mm.actuator_gainprm[a][0] - ref.actuator_gainprm[ra][0]) < 1e-3
        ok &= np.allclose(mm.actuator_biasprm[a][:3], ref.actuator_biasprm[ra][:3], atol=1e-3)
        check(f"{jn:11s}", ok, row)

    print("\n== geoms")
    gf = mw.geom_friction.numpy()[0]
    ground = [g for g in range(mm.ngeom) if mm.geom_bodyid[g] == 0]
    check("one world geom (the ground plane)", len(ground) == 1)
    ground = ground[0]
    body_geoms = {}
    for g in range(mm.ngeom):
        if g == ground:
            continue
        usd = [n for n in robot.body_names if mm.body(int(mm.geom_bodyid[g])).name.endswith("_" + n)][0]
        rg = [x for x in range(ref.ngeom) if ref.geom_bodyid[x] == ref.body(usd2mj_body[usd]).id][0]
        body_geoms[usd] = g
        nsz = {int(mujoco.mjtGeom.mjGEOM_BOX): 3, int(mujoco.mjtGeom.mjGEOM_SPHERE): 1}.get(int(ref.geom_type[rg]), 2)
        ok = mm.geom_type[g] == ref.geom_type[rg] and np.allclose(mm.geom_size[g][:nsz], ref.geom_size[rg][:nsz], atol=1e-6)
        ok &= abs(gf[g][0] - spec.FRICTION) < 1e-6 and np.allclose(gf[g][1:], ref.geom_friction[rg][1:], atol=1e-7)
        ok &= np.allclose(mm.geom_solref[g], ref.geom_solref[rg]) and np.allclose(mm.geom_solimp[g], ref.geom_solimp[rg])
        ok &= mm.geom_condim[g] == ref.geom_condim[rg] and mm.geom_margin[g] == ref.geom_margin[rg] == 0.0
        ok &= mm.geom_gap[g] == ref.geom_gap[rg] == 0.0
        ok &= int(mm.geom_priority[g]) == int(ref.geom_priority[rg]) and abs(mm.geom_solmix[g] - ref.geom_solmix[rg]) < 1e-9
        ok &= np.allclose(mw.geom_solref.numpy()[0][g], ref.geom_solref[rg], atol=1e-6)  # the batched model too
        check(f"{usd:12s} {['plane', 'hfield', 'sphere', 'capsule', 'ellipsoid', 'cylinder', 'box'][mm.geom_type[g]]:7s} "
              f"size {np.round(mm.geom_size[g], 4).tolist()} friction {gf[g].round(5).tolist()} solref {mm.geom_solref[g].tolist()} "
              f"condim {mm.geom_condim[g]} priority {mm.geom_priority[g]}", ok)
    fl = ref.geom("floor").id
    if spec.FLOOR_PRIORITY:  # round 9 final: the floor geom itself is soft with priority 1
        fg = spec.FLOOR_GEOM
        check(f"quad.xml floor: solref {ref.geom_solref[fl].tolist()}, priority {ref.geom_priority[fl]}, friction "
              f"{ref.geom_friction[fl].tolist()} == sidecar floorGeom; every body geom priority 0",
              np.allclose(ref.geom_solref[fl], fg["solref"]) and int(ref.geom_priority[fl]) == fg["priority"] >= 1
              and np.allclose(ref.geom_friction[fl], fg["friction"])
              and all(int(ref.geom_priority[g]) == 0 for g in range(ref.ngeom) if ref.geom_bodyid[g] != 0))
        check(f"ground: priority {mm.geom_priority[ground]}, solref {np.round(mm.geom_solref[ground], 4).tolist()}, "
              f"friction {np.round(gf[ground], 5).tolist()} == quad.xml floor",
              int(mm.geom_priority[ground]) == fg["priority"] and np.allclose(mm.geom_solref[ground], fg["solref"], atol=1e-6)
              and np.allclose(mw.geom_solref.numpy()[0][ground], fg["solref"], atol=1e-6)
              and np.allclose(gf[ground], fg["friction"], atol=1e-6) and np.allclose(mm.geom_solimp[ground], fg["solimp"])
              and mm.geom_condim[ground] == fg["condim"])
    elif ref.npair:  # round 9 (superseded): floor <pair>s (feet) reproduced as ground parameters + priority 1
        fp = spec.FLOOR_CONTACT
        excl_w = {int(x) & 0xFFFF for x in ref.exclude_signature if (int(x) >> 16) == 0}
        pair_geoms = sorted(int(ref.pair_geom2[i]) if ref.pair_geom1[i] == fl else int(ref.pair_geom1[i]) for i in range(ref.npair))
        check(f"quad.xml floor pairs {[ref.geom(g).name for g in pair_geoms]} (solref {ref.pair_solref[0].tolist()}, friction "
              f"{ref.pair_friction[0].tolist()}) + world excludes == the sidecar's floorContact",
              [ref.geom(g).name for g in pair_geoms] == sorted(spec.NAMES["floorContactGeoms"])
              and excl_w == {int(ref.geom_bodyid[g]) for g in pair_geoms}
              and all(np.allclose(ref.pair_solref[i], fp["solref"]) and np.allclose(ref.pair_friction[i], fp["friction"])
                      and np.allclose(ref.pair_solimp[i], fp["solimp"]) and ref.pair_dim[i] == fp["condim"] for i in range(ref.npair)))
        check(f"ground: priority {mm.geom_priority[ground]}, solref {np.round(mm.geom_solref[ground], 4).tolist()}, friction "
              f"{np.round(gf[ground], 5).tolist()} == the floor pair's (priority 1 > every body geom's 0)",
              int(mm.geom_priority[ground]) == 1 and all(int(mm.geom_priority[g]) == 0 for g in body_geoms.values())
              and np.allclose(mm.geom_solref[ground], fp["solref"], atol=1e-6)
              and np.allclose(mw.geom_solref.numpy()[0][ground], fp["solref"], atol=1e-6)
              and abs(gf[ground][0] - fp["friction"][0]) < 1e-6 and np.allclose(gf[ground][1:], fp["friction"][2:4], atol=1e-7)
              and np.allclose(mm.geom_solimp[ground], fp["solimp"]) and mm.geom_condim[ground] == fp["condim"])
    else:
        check(f"floor friction {gf[ground][0]:.3f} < 0.765 (max() -> body 0.9*s); solref {mm.geom_solref[ground].tolist()} "
              f"== quad.xml {ref.geom_solref[fl].tolist()}",
              gf[ground][0] < 0.9 * spec.FRICTION_SCALE[0] and np.allclose(mm.geom_solref[ground], ref.geom_solref[fl])
              and np.allclose(mm.geom_solimp[ground], ref.geom_solimp[fl]) and mm.geom_condim[ground] == ref.geom_condim[fl])

    print("\n== self-collision rules (rule M): collidable body pairs, Newton-built mjModel vs quad.xml")

    def to_mj(pairs_newton):
        out = set()
        for a, b in pairs_newton:
            ua = [n for n in robot.body_names if a.endswith("_" + n)][0]
            ub = [n for n in robot.body_names if b.endswith("_" + n)][0]
            out.add(tuple(sorted((usd2mj_body[ua], usd2mj_body[ub]))))
        return out

    pn, pr = to_mj(collidable_body_pairs(mm)), collidable_body_pairs(ref)
    n_all = len(robot.body_names) * (len(robot.body_names) - 1) // 2
    print(f"  quad.xml: {len(pr)} of {n_all} body pairs collide; Newton: {len(pn)}; not colliding (parent-child): "
          f"{sorted(set(tuple(sorted((ref.body(b).name, ref.body(ref.body_parentid[b]).name))) for b in range(2, ref.nbody)))}")
    check("collidable body pairs identical", pn == pr, f"only Newton {sorted(pn - pr)}, only quad.xml {sorted(pr - pn)}")
    check("every body collides with the floor",
          all((mm.geom_contype[g] & mm.geom_conaffinity[ground]) or (mm.geom_contype[ground] & mm.geom_conaffinity[g])
              for g in body_geoms.values()))

    ids = {n: i for i, n in enumerate(robot.joint_names)}
    bid = {n: i for i, n in enumerate(robot.body_names)}
    nj = robot.num_joints
    all_ids = list(range(env.num_envs))

    def place(env_ids, joint_pos, joint_vel=None, z=1.5, vel=None, quat=None):
        e = torch.tensor(env_ids, device=dev, dtype=torch.int32)
        n = len(env_ids)
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :3] = env.scene.env_origins[e.long()]
        pose[:, 2] += z
        pose[:, 6] = 1.0
        if quat is not None:
            pose[:, 3:] = quat
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=e)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev) if vel is None else vel, env_ids=e)
        robot.write_joint_state_to_sim_index(position=joint_pos, velocity=torch.zeros_like(joint_pos) if joint_vel is None else joint_vel,
                                             env_ids=e)
        robot.set_joint_position_target_index(target=joint_pos, env_ids=e)

    def step(n):
        for _ in range(n):
            robot.write_data_to_sim()
            sim.step(render=False)
            env.scene.update(dt)

    def cpu_model(kp=None):
        m2 = mujoco.MjModel.from_xml_path(QUAD_XML)
        if kp is not None:
            m2.actuator_gainprm[:, 0] = kp
            m2.actuator_biasprm[:, 1] = -kp
        return m2

    def cpu_set(m2, q_lab, z=1.5):
        """CPU MuJoCo state matching place(): root at (0, 0, z), identity, joints q (Lab order)."""
        d2 = mujoco.MjData(m2)
        d2.qpos[:3] = [0, 0, z]
        d2.qpos[3:7] = [1, 0, 0, 0]
        for jn, v in zip(robot.joint_names, q_lab):
            d2.qpos[m2.jnt_qposadr[m2.joint(usd2mj_joint[jn]).id]] = v
            act = [a for a in range(m2.nu) if m2.actuator_trnid[a][0] == m2.joint(usd2mj_joint[jn]).id][0]
            d2.ctrl[act] = v
        return d2

    step(1)
    check("MuJoCo-Warp timestep == 0.005 s during stepping", abs(float(mw.opt.timestep.numpy().reshape(-1)[0]) - 0.005) < 1e-9)

    print("\n== soft feet: 2 cm spawn drop (0.92 m, rest pose, zero targets), peak single-foot floor normal force, 1 s")
    foot_sr = ([float(spec.FLOOR_GEOM['solref'][0])] if spec.FLOOR_PRIORITY
               else [float(spec.FLOOR_CONTACT['solref'][0])] if ref.npair
               else [float(ref.geom_solref[ref.geom(g).id][0]) for g in spec.RIG['footGeoms']])
    import mujoco_warp as mjw

    naconmax = int(solver.mjw_data.contact.dist.shape[0])
    cf_ids = wp.array(np.arange(naconmax, dtype=np.int32), dtype=wp.int32, device=str(dev))
    cf_out = wp.zeros(naconmax, dtype=wp.spatial_vector, device=str(dev))
    feet_mj = [[g for g in range(mm.ngeom) if mm.geom_bodyid[g] == mj_body(n)][0] for n in spec.LOWER_LEGS]
    place(all_ids, torch.zeros(env.num_envs, nj, device=dev), z=spec.SPAWN_Z)
    peak_n = np.zeros(env.num_envs)
    for _ in range(200):
        step(1)
        mjw.contact_force(mw, solver.mjw_data, cf_ids, False, cf_out)
        k_ = int(solver.mjw_data.nacon.numpy()[0])
        g_, w_, f_ = solver.mjw_data.contact.geom.numpy()[:k_], solver.mjw_data.contact.worldid.numpy()[:k_], cf_out.numpy()[:k_, 0]
        per = np.zeros((env.num_envs, 4))
        for (a_, b_), ww, ff in zip(g_, w_, f_):
            o_ = b_ if a_ == ground else (a_ if b_ == ground else -1)
            if o_ in feet_mj:
                per[ww, feet_mj.index(o_)] += ff
        peak_n = np.maximum(peak_n, per.max(axis=1))
    m2 = cpu_model()
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d2, m2.key("spawn").id)
    fl2 = m2.geom("floor").id
    feet2 = [m2.geom(g).id for g in spec.RIG["footGeoms"]]
    cpu_peak, f6 = 0.0, np.zeros(6)
    for _ in range(200):
        mujoco.mj_step(m2, d2)
        per = np.zeros(4)
        for ci in range(d2.ncon):
            c = d2.contact[ci]
            o_ = c.geom[1] if c.geom[0] == fl2 else (c.geom[0] if c.geom[1] == fl2 else -1)
            if o_ in feet2:
                mujoco.mj_contactForce(m2, d2, ci, f6)
                per[feet2.index(o_)] += f6[0]
        cpu_peak = max(cpu_peak, per.max())
    bw = spec.TOTAL_MASS * 9.81
    check(f"spawn-drop peak foot force == CPU MuJoCo on quad.xml (foot solref {foot_sr[0]})",
          bool(np.all(np.abs(peak_n - cpu_peak) < 0.02 * cpu_peak)),
          f"Newton {np.round(peak_n / bw, 3).tolist()} BW, CPU MuJoCo {cpu_peak / bw:.3f} BW")

    print("\n== kinematics: random joint angles, one substep in free fall, every body pose vs CPU MuJoCo quad.xml")
    rng = np.random.default_rng(0)
    worst_p, worst_q = 0.0, 0.0
    for trial in range(3):
        q = np.array([rng.uniform(*ref.jnt_range[ref.joint(usd2mj_joint[n]).id]) * 0.9 for n in robot.joint_names], dtype=np.float32)
        place([0], torch.tensor(q, device=dev).unsqueeze(0))
        step(1)
        m2 = cpu_model()
        d2 = cpu_set(m2, q)
        mujoco.mj_step(m2, d2)
        mujoco.mj_kinematics(m2, d2)
        o = env.scene.env_origins[0]
        for n in robot.body_names:
            pl = (d.body_link_pos_w.torch[0, bid[n]] - o).cpu().numpy()
            ql = d.body_link_quat_w.torch[0, bid[n]].cpu().numpy()  # xyzw
            j = m2.body(usd2mj_body[n]).id
            worst_p = max(worst_p, float(np.abs(pl - d2.xpos[j]).max()))
            qm = d2.xquat[j]  # wxyz
            worst_q = max(worst_q, 1.0 - abs(float(np.dot([ql[3], *ql[:3]], qm))))
    check("body positions / orientations == CPU MuJoCo (3 random poses)", worst_p < 2e-3 and worst_q < 1e-5,
          f"max |dp| {worst_p * 1000:.3f} mm, max 1-|<q,q'>| {worst_q:.2e}")

    print("\n== sign tests: +0.5 rad on one hip and one knee, foot point in the torso frame (vs CPU MuJoCo)")
    foot = torch.tensor(spec.FOOT_POINTS[0], device=dev)
    for jn in (spec.ACTION_ORDER[0], spec.ACTION_ORDER[1]):  # rear-left hip, rear-left knee
        leg = spec.LOWER_LEGS[0]
        res = {}
        for val in (0.0, 0.5):
            q = torch.zeros(1, nj, device=dev)
            q[0, ids[jn]] = val
            place([0], q)
            step(1)
            tq, tp = d.body_link_quat_w.torch[0, bid[spec.TORSO]], d.body_link_pos_w.torch[0, bid[spec.TORSO]]
            lq, lp = d.body_link_quat_w.torch[0, bid[leg]], d.body_link_pos_w.torch[0, bid[leg]]
            fw = lp + quat_apply(lq.unsqueeze(0), foot.unsqueeze(0))[0]
            res[val] = quat_apply_inverse(tq.unsqueeze(0), (fw - tp).unsqueeze(0))[0].cpu().numpy()
        m2 = cpu_model()
        d2 = cpu_set(m2, [0.5 if n == jn else 0.0 for n in robot.joint_names])
        mujoco.mj_step(m2, d2)
        mujoco.mj_kinematics(m2, d2)
        lj, tj = m2.body(usd2mj_body[leg]).id, m2.body(spec.TORSO_MJCF).id
        fw2 = d2.xpos[lj] + d2.xmat[lj].reshape(3, 3) @ np.array(spec.FOOT_POINTS[0])
        cpu = d2.xmat[tj].reshape(3, 3).T @ (fw2 - d2.xpos[tj])
        dx = res[0.5][0] - res[0.0][0]
        check(f"{jn} ({spec.RIG['legNames'][usd2mj_joint[jn]]}) = +0.5 moves the foot toward -x",
              dx < -0.05 and np.allclose(res[0.5], cpu, atol=2e-3),
              f"foot in torso frame {np.round(res[0.5], 4).tolist()} (dx {dx:+.4f}); CPU MuJoCo {np.round(cpu, 4).tolist()}")

    print("\n== effort term: Isaac Lab applied_torque vs MuJoCo actuator_force (targets partly saturating)")
    q = torch.zeros(1, nj, device=dev)
    place([0], q)
    tgt = torch.tensor([[0.2, -0.3, 0.01, -0.02, 0.05, 0.1, -0.5, 0.03]], device=dev)
    robot.set_joint_position_target_index(target=tgt, env_ids=torch.tensor([0], device=dev, dtype=torch.int32))
    step(1)
    lab_tau = d.applied_torque.torch[0].cpu().numpy()
    mj_force = solver.mjw_data.actuator_force.numpy()[0]
    qfrc = solver.mjw_data.qfrc_actuator.numpy()[0][6:]
    act_order = [robot.joint_names.index([n for n in robot.joint_names if mm.joint(mm.actuator_trnid[a][0]).name.endswith("_" + n)][0])
                 for a in range(mm.nu)]
    inv = np.argsort(act_order)
    mj_clip = np.clip(mj_force, -spec.FORCE_LIMIT, spec.FORCE_LIMIT)
    print(f"  Lab applied_torque    : {np.round(lab_tau, 2).tolist()}")
    print(f"  MuJoCo actuator_force : {np.round(mj_force[inv], 2).tolist()}")
    print(f"  MuJoCo qfrc_actuator  : {np.round(qfrc, 2).tolist()}")
    check("applied_torque == clip(actuator_force, +/-300) == qfrc_actuator (first substep)",
          np.allclose(lab_tau, mj_clip[inv], atol=1e-2) and np.allclose(lab_tau, qfrc, atol=1e-2)
          and float(np.abs(lab_tau).max()) >= spec.FORCE_LIMIT - 1e-3)

    # ---- passive damping + joint limit, both against CPU MuJoCo with kp 0 --------------------------
    e0 = torch.tensor([0], device=dev, dtype=torch.int32)
    robot.write_joint_stiffness_to_sim_index(stiffness=0.0, env_ids=e0)
    for a in robot.actuators.values():
        a.stiffness[0] = 0.0
    knee = spec.ACTION_ORDER[1]
    print(f"\n== passive damping: free fall, kp 0, {knee} qdot0 = 5 rad/s (vs CPU MuJoCo on quad.xml)")
    q = torch.zeros(1, nj, device=dev)
    qd = torch.zeros_like(q)
    qd[0, ids[knee]] = 5.0
    place([0], q, qd)
    lab = []
    for _ in range(10):
        step(1)
        lab.append(float(d.joint_vel.torch[0, ids[knee]]))
    m2 = cpu_model(kp=0.0)
    d2 = cpu_set(m2, np.zeros(nj))
    d2.ctrl[:] = 0
    d2.qvel[m2.jnt_dofadr[m2.joint(usd2mj_joint[knee]).id]] = 5.0
    cpu = []
    for _ in range(10):
        mujoco.mj_step(m2, d2)
        cpu.append(float(d2.qvel[m2.jnt_dofadr[m2.joint(usd2mj_joint[knee]).id]]))
    print(f"  Newton     qdot after 1/2/4/10 substeps: {lab[0]:.4f} / {lab[1]:.4f} / {lab[3]:.4f} / {lab[9]:.4f}")
    print(f"  CPU MuJoCo qdot after 1/2/4/10 substeps: {cpu[0]:.4f} / {cpu[1]:.4f} / {cpu[3]:.4f} / {cpu[9]:.4f}")
    check(f"passive damping {spec.JOINT_DAMPING:g} == CPU MuJoCo (10 substeps)", max(abs(a - b) for a, b in zip(lab, cpu)) < 0.05,
          f"max |diff| {max(abs(a - b) for a, b in zip(lab, cpu)):.4f} rad/s")

    lim_hi = float(ref.jnt_range[ref.joint(usd2mj_joint[knee]).id][1])
    print(f"\n== joint limit: kp 0, {knee} started 0.1 rad beyond its +{math.degrees(lim_hi):.0f} deg limit (vs CPU MuJoCo)")
    q = torch.zeros(1, nj, device=dev)
    q[0, ids[knee]] = lim_hi + 0.1
    place([0], q)
    robot.write_joint_position_limit_to_sim_index(limits=d.joint_pos_limits.torch[:1].clone(), env_ids=e0, warn_limit_violation=False)
    lab = []
    for _ in range(20):
        step(1)
        lab.append(float(d.joint_pos.torch[0, ids[knee]]))
    m2 = cpu_model(kp=0.0)
    d2 = cpu_set(m2, [lim_hi + 0.1 if n == knee else 0.0 for n in robot.joint_names])
    d2.ctrl[:] = 0
    cpu = []
    for _ in range(20):
        mujoco.mj_step(m2, d2)
        cpu.append(float(d2.qpos[m2.jnt_qposadr[m2.joint(usd2mj_joint[knee]).id]]))
    print(f"  Newton     q - limit after 1/5/10/20 substeps: {[round(lab[k] - lim_hi, 5) for k in (0, 4, 9, 19)]}")
    print(f"  CPU MuJoCo q - limit after 1/5/10/20 substeps: {[round(cpu[k] - lim_hi, 5) for k in (0, 4, 9, 19)]}")
    check("limit solref (0.01, 1): recovery trajectory == CPU MuJoCo", max(abs(a - b) for a, b in zip(lab, cpu)) < 2e-3,
          f"max |diff| {max(abs(a - b) for a, b in zip(lab, cpu)):.2e} rad")
    robot.write_joint_stiffness_to_sim_index(stiffness=spec.KP, env_ids=e0)
    for a in robot.actuators.values():
        a.stiffness[0] = spec.KP

    print("\n== randomisers: per-world body mass (every body) and kp, vs Newton's own model-change path")
    from isaaclab.managers import EventTermCfg, SceneEntityCfg

    import quad_tasks_v3.mdp as qmdp

    body_cfg = SceneEntityCfg("robot", body_names=spec.BODY_NAMES, preserve_order=True)
    jnt_cfg = SceneEntityCfg("robot", joint_names=spec.ACTION_ORDER, preserve_order=True)
    body_cfg.resolve(env.scene)
    jnt_cfg.resolve(env.scene)
    step(1)
    mterm = qmdp.randomize_body_mass_lean(EventTermCfg(func=qmdp.randomize_body_mass_lean, mode="reset",
                                          params={"asset_cfg": body_cfg, "scale_range": spec.MASS_SCALE}), env)
    kterm = qmdp.randomize_kp_lean(EventTermCfg(func=qmdp.randomize_kp_lean, mode="reset",
                                   params={"asset_cfg": jnt_cfg, "scale_range": spec.KP_SCALE}), env)
    torch.manual_seed(3)
    ids02 = torch.tensor([0, 2], device=dev)
    mterm(env, ids02, body_cfg, spec.MASS_SCALE)
    kterm(env, ids02, jnt_cfg, spec.KP_SCALE)
    mjm, mji = wp.to_torch(mw.body_mass).clone(), wp.to_torch(mw.body_inertia).clone()
    gain, bias = wp.to_torch(mw.actuator_gainprm)[..., 0].clone(), wp.to_torch(mw.actuator_biasprm)[..., 1].clone()
    lab_m, lab_kp = d.body_mass.torch.clone(), d.joint_stiffness.torch.clone()
    nominal = torch.tensor([float(ref.body(usd2mj_body[n]).mass[0]) for n in robot.body_names], device=dev)
    ratio = lab_m / nominal
    print(f"  mass scale world0 {ratio[0].cpu().numpy().round(4).tolist()}")
    check("every body mass x U(0.9, 1.1) (torso included) in the reset worlds only",
          bool(torch.all((ratio[[0, 2]] >= 0.9) & (ratio[[0, 2]] <= 1.1))) and bool(torch.allclose(ratio[[1, 3]], torch.ones_like(ratio[[1, 3]])))
          and len(set(ratio[0].tolist())) == len(robot.body_names))
    check("kp x U(0.8, 1.2) per joint in the reset worlds only",
          bool(torch.all((lab_kp[[0, 2]] >= 0.8 * spec.KP) & (lab_kp[[0, 2]] <= 1.2 * spec.KP)))
          and bool(torch.allclose(lab_kp[[1, 3]], torch.full_like(lab_kp[[1, 3]], spec.KP))))
    NewtonManager.add_model_change(SolverNotifyFlags.BODY_INERTIAL_PROPERTIES | SolverNotifyFlags.JOINT_DOF_PROPERTIES)
    step(1)
    same = (torch.allclose(wp.to_torch(mw.body_mass), mjm, rtol=1e-5) and torch.allclose(wp.to_torch(mw.body_inertia), mji, rtol=1e-4)
            and torch.allclose(wp.to_torch(mw.actuator_gainprm)[..., 0], gain) and torch.allclose(wp.to_torch(mw.actuator_biasprm)[..., 1], bias))
    check("MuJoCo-Warp mass / inertia / gains unchanged after Newton's full notify (lean writers == official path)", same)
    all4 = torch.arange(4, device=dev)
    mterm(env, all4, body_cfg, (1.0, 1.0))
    kterm(env, all4, jnt_cfg, (1.0, 1.0))

    print("\n== rest stand: spec reset, zero action, 5 s (100 policy steps at 20 Hz)")
    obs, _ = env.reset()
    ob = obs["policy"][0]
    print(f"  obs[0] after reset: {[round(float(v), 4) for v in ob]}")
    rq = d.root_link_quat_w.torch[0]
    yaw = 2 * math.atan2(float(rq[2]), float(rq[3]))
    check("obs: gravity_b ~ (0,0,-1), goal_dir_b ~ (cos yaw, -sin yaw), target speed 0.745",
          float(ob[2]) < -0.99 and abs(float(ob[33]) - math.cos(yaw)) < 0.05 and abs(float(ob[34]) + math.sin(yaw)) < 0.05
          and abs(float(ob[35]) - 0.745) < 1e-6, f"yaw {math.degrees(yaw):+.1f} deg")
    if spec.GAIT:
        ph = env._quad_gait_clock.phase
        check("obs[36:38] = (sin phi, cos phi) of a per-env random clock",
              torch.allclose(obs["policy"][:, 36], torch.sin(ph), atol=1e-6) and torch.allclose(obs["policy"][:, 37], torch.cos(ph), atol=1e-6)
              and len(set(np.round(ph.cpu().numpy(), 5))) == env.num_envs, str(np.round(ph.cpu().numpy(), 3).tolist()))
    check("reset: torso at 0.92 m, joints within +/-0.05 rad of rest",
          bool(torch.all((d.root_link_pos_w.torch[:, 2] - spec.SPAWN_Z).abs() < 1e-3))
          and bool(torch.all(d.joint_pos.torch.abs() <= spec.RESET_JOINT_NOISE + 1e-5)),
          f"z {d.root_link_pos_w.torch[:, 2].cpu().numpy().round(4).tolist()}")
    zero = torch.zeros(env.num_envs, 8, device=dev)
    fs = env.reward_manager.get_term_cfg("foot_slip").func
    min_up, n_done = 1.0, 0
    for _ in range(int(round(5.0 / env.step_dt))):
        obs, rew, term, trunc, extras = env.step(zero)
        n_done += int((term | trunc).sum())
        tq = d.body_link_quat_w.torch[:, bid[spec.TORSO]]
        zz = torch.zeros_like(tq[:, :3])
        zz[:, 2] = 1
        min_up = min(min_up, float(quat_apply(tq, zz)[:, 2].min()))
    tz = (d.body_link_pos_w.torch[:, bid[spec.TORSO], 2] - env.scene.env_origins[:, 2]).cpu().numpy()
    m2 = cpu_model()
    d2 = mujoco.MjData(m2)
    mujoco.mj_resetDataKeyframe(m2, d2, m2.key("spawn").id)
    for _ in range(1000):
        mujoco.mj_step(m2, d2)
    cz = float(d2.xpos[m2.body(spec.TORSO_MJCF).id][2])
    print(f"  torso height after 5 s: {tz.round(4).tolist()} (CPU MuJoCo quad.xml, spawn keyframe: {cz:.4f}); "
          f"min up.z {min_up:.4f}; feet in contact {fs.contact.sum(dim=1).cpu().tolist()}; "
          f"reward/step env0 {float(rew[0]):.5f}")
    check("stands: torso ~0.90 m (+/-1 cm, and within 5 mm of CPU MuJoCo), upright > 0.99 throughout, no termination",
          bool(np.all(np.abs(tz - spec.REST_Z) < 0.01)) and bool(np.all(np.abs(tz - cz) < 0.005)) and min_up > 0.99 and n_done == 0)
    check("all 4 feet in floor contact at rest; foot speed ~0",
          bool(fs.contact.all()) and float(torch.linalg.norm(fs.foot_velocity_w()[..., :2], dim=-1).max()) < 0.01)

    print("\n== friction: per-env randomised scale s, pair friction in every live foot-floor contact")
    if spec.FLOOR_PRIORITY:  # the floor's friction is randomised per world (lean writer, no notify)
        fcfg = EventTermCfg(func=qmdp.randomize_floor_friction, mode="reset",
                            params={"base": spec.GROUND_FRICTION, "scale_range": spec.FRICTION_SCALE})
        fterm = qmdp.randomize_floor_friction(fcfg, env)
    else:
        fcfg = EventTermCfg(func=qmdp.randomize_friction_scale, mode="reset",
                            params={"asset_cfg": SceneEntityCfg("robot"), "base": spec.FRICTION, "scale_range": spec.FRICTION_SCALE})
        fterm = qmdp.randomize_friction_scale(fcfg, env)
    fterm(env, torch.arange(env.num_envs, device=dev, dtype=torch.int32), **fcfg.params)
    step(8)  # flush any notification into mjw; still standing
    s_drawn = fterm.scale.cpu().numpy()
    con = solver.mjw_data.contact
    n = int(solver.mjw_data.nacon.numpy()[0])
    cw, cg, cf, cdim = con.worldid.numpy()[:n], con.geom.numpy()[:n], con.friction.numpy()[:n], con.dim.numpy()[:n]
    csr = con.solref.numpy()[:n]
    foot_sr = ([float(spec.FLOOR_GEOM["solref"][0])] if spec.FLOOR_PRIORITY
               else [float(spec.FLOOR_CONTACT["solref"][0])] if ref.npair
               else [float(ref.geom_solref[ref.geom(g).id][0]) for g in spec.RIG["footGeoms"]])
    gfw = mw.geom_friction.numpy()
    for w in all_ids:
        sel = (cw == w) & ((cg[:, 0] == ground) | (cg[:, 1] == ground))
        # floor pairs: foot-floor friction is the pair's (fixed, as quad.xml); body geoms still carry 0.9 * s
        if spec.FLOOR_PRIORITY:  # floor 0.9 * s per world (priority 1 -> used by every floor contact); body fixed 0.9
            want = spec.GROUND_FRICTION * float(s_drawn[w])
            body_want = spec.FRICTION
        else:
            want = float(spec.FLOOR_CONTACT["friction"][0]) if ref.npair else spec.FRICTION * float(s_drawn[w])
            body_want = spec.FRICTION * float(s_drawn[w])
        body_mu = np.array([gfw[w][g][0] for g in body_geoms.values()])
        ok_body = np.allclose(body_mu, body_want, atol=1e-5)
        mus = cf[sel, 0]
        ok = sel.sum() >= 4 and np.allclose(mus, want, atol=1e-5) and bool(np.all(cdim[sel] == 3)) and np.allclose(cf[sel, 1], mus)
        # priority: a foot-floor contact takes the FOOT's solref (0.03, 1 in round 9), not MuJoCo's mix
        ok &= np.allclose(csr[sel][:, 0], foot_sr[0], atol=1e-6) and np.allclose(csr[sel][:, 1], 1.0, atol=1e-6)
        ok &= bool(ok_body) and int(sel.sum()) == 4  # one contact per foot, no duplicate automatic contact
        check(f"env{w}: s={s_drawn[w]:.4f} -> {int(sel.sum())} floor contacts, mu {np.unique(mus.round(5)).tolist()} == {want:.5f}, "
              f"contact solref {np.unique(csr[sel].round(4), axis=0).tolist()} (expected {foot_sr[0]}), body geoms mu "
              f"{np.unique(body_mu.round(5)).tolist()}", ok)
    if spec.FLOOR_PRIORITY:
        # the per-world floor values must survive env.reset / env.step (the Play env has no friction event, so a
        # reset must leave them alone), and a redraw of some worlds must touch only those worlds
        env.reset()
        for _ in range(3):
            env.step(torch.zeros(env.num_envs, 8, device=dev))
        gf_now = mw.geom_friction.numpy()[:, ground, 0]
        fterm(env, torch.tensor([1, 3], device=dev), spec.GROUND_FRICTION, spec.FRICTION_SCALE)
        step(1)
        gf_re = mw.geom_friction.numpy()[:, ground, 0]
        s_re = fterm.scale.cpu().numpy()
        tcfg_f = load_cfg_from_registry("Isaac-Quad-Flat-Newton-v0", "env_cfg_entry_point").events.friction
        check("floor friction per world persists through env.reset/env.step; a redraw touches only its worlds; the "
              "training env uses randomize_floor_friction",
              np.allclose(gf_now, spec.GROUND_FRICTION * s_drawn, atol=1e-6)
              and np.allclose(gf_re, spec.GROUND_FRICTION * s_re, atol=1e-6) and np.allclose(gf_re[[0, 2]], gf_now[[0, 2]])
              and not np.allclose(gf_re[[1, 3]], gf_now[[1, 3]]) and tcfg_f.func is qmdp.randomize_floor_friction
              and tcfg_f.params["base"] == 0.9,
              f"after reset+steps {np.round(gf_now, 4).tolist()}, after redraw of worlds 1,3 {np.round(gf_re, 4).tolist()}")
        fterm(env, None, spec.GROUND_FRICTION, (1.0, 1.0))  # back to nominal 0.9 for the remaining tests
    else:
        mu = wp.to_torch(fterm._mu)
        mu[:] = spec.FRICTION
        NewtonManager.add_model_change(SolverNotifyFlags.SHAPE_PROPERTIES)

    print("\n== push event: +0.5 m/s horizontal, random direction, whole body (in the air: no contact eats it)")
    place(all_ids, torch.zeros(env.num_envs, nj, device=dev), z=1.5)
    step(1)
    v0 =d.body_link_lin_vel_w.torch[:, :, :].clone()
    qmdp.push_horizontal(env, torch.tensor([1, 3], device=dev), spec.PUSH_SPEED)
    step(1)
    v1 = d.body_link_lin_vel_w.torch
    dv = (v1[:, bid[spec.TORSO]] - v0[:, bid[spec.TORSO]]).cpu().numpy()
    print(f"  torso dv after one substep: {np.round(dv, 3).tolist()}")
    hor = np.linalg.norm(dv[:, :2], axis=1)
    check("pushed worlds gain ~0.5 m/s horizontally, others ~0", bool(np.all(np.abs(hor[[1, 3]] - 0.5) < 0.05)) and bool(np.all(hor[[0, 2]] < 0.05)),
          f"|dv_xy| {np.round(hor, 3).tolist()}")
    legs_dv = (v1[[1, 3]][:, :, :2] - v0[[1, 3]][:, :, :2]).cpu().numpy()
    check("every body of a pushed world gains the same horizontal dv", float(np.abs(legs_dv - legs_dv[:, :1]).max()) < 0.02)
    tcfg_train = load_cfg_from_registry("Isaac-Quad-Flat-Newton-v0", "env_cfg_entry_point")
    pe = tcfg_train.events.push
    check("training cfg: push is an interval event, per-env timer U(10, 15) s (redrawn at reset), speed 0.5",
          pe.mode == "interval" and tuple(pe.interval_range_s) == spec.PUSH_INTERVAL_S and not pe.is_global_time
          and pe.params["speed"] == 0.5)

    print("\n== feet air time (debounced substep touchdowns, cap 0.5 s; step sum paid if post-step v_x > 0.3) vs CPU MuJoCo")
    tracker = robot.contact_tracker
    air_col = env.reward_manager.active_terms.index("feet_air_time")
    for h, vx, tag in ((0.12, 1.0, "12 cm drop at 1.0 m/s"), (0.12, 0.2, "12 cm drop at 0.2 m/s (gated)"),
                       (0.30, 1.0, "30 cm drop at 1.0 m/s")):
        vel = torch.zeros(env.num_envs, 6, device=dev)
        vel[:, 0] = vx
        place(all_ids, torch.zeros(env.num_envs, nj, device=dev), z=spec.REST_Z + h, vel=vel)
        tracker.reset(slice(None))
        tracker.t_counter.zero_()
        total = torch.zeros(env.num_envs, device=dev)
        fl_total = torch.zeros(env.num_envs, device=dev)
        fl_col = env.reward_manager.active_terms.index("flight")
        zero = torch.zeros(env.num_envs, 8, device=dev)
        ended = 0
        for _ in range(int(round(0.6 / env.step_dt))):
            _, _, term_, trunc_, _ = env.step(zero)
            ended += int((term_ | trunc_).sum())
            # _step_reward = weight * value, value = sum / step_dt  ->  raw touchdown sum
            total += env.reward_manager._step_reward[:, air_col] * env.step_dt / spec.W_AIR_TIME
            fl_total += env.reward_manager._step_reward[:, fl_col] / spec.W_FLIGHT * env.step_dt / spec.PHYSICS_DT  # substeps
        qv = float(solver.mjw_data.qvel.numpy()[0][0])
        tv = float(d.body_link_lin_vel_w.torch[0, bid[spec.TORSO], 0])
        m2 = cpu_model()
        d2 = cpu_set(m2, np.zeros(nj), z=spec.REST_Z + h)
        d2.qvel[0] = vx
        feet_g = [m2.geom(g).id for g in spec.RIG["footGeoms"]]
        floor_g = m2.geom("floor").id
        db, run, air_t, cpu_total, n_td = [1] * 4, [0] * 4, [0.0] * 4, 0.0, 0
        step_sum, cpu_flight = 0.0, 0
        for sub in range(int(round(0.6 / m2.opt.timestep))):
            mujoco.mj_step(m2, d2)
            down = [0] * 4
            for ci in range(d2.ncon):
                c = d2.contact[ci]
                if c.dist < 0 and floor_g in (c.geom[0], c.geom[1]):
                    o = c.geom[1] if c.geom[0] == floor_g else c.geom[0]
                    if o in feet_g:
                        down[feet_g.index(o)] = 1
            for k in range(4):
                run[k] = run[k] + 1 if down[k] != db[k] else 0
                if run[k] >= 3:
                    db[k], run[k] = down[k], 0
                    if down[k]:
                        step_sum += min(air_t[k], 0.5) - spec.AIR_TIME_OFFSET
                        n_td += 1
                        air_t[k] = 0.0
                if db[k] == 0:
                    air_t[k] += m2.opt.timestep
            cpu_flight += int(all(x == 0 for x in db))
            if (sub + 1) % spec.DECIMATION == 0:  # end of a policy step: gate on the post-step torso v_x
                if d2.qvel[0] > spec.AIR_VX_MIN:
                    cpu_total += step_sum
                step_sum = 0.0
        lab = total.cpu().numpy()
        check(f"{tag}: raw touchdown sum over 0.6 s == CPU MuJoCo (debounce 3, cap 0.5, v_x gate); no resets",
              ended == 0 and bool(np.all(np.abs(lab - cpu_total) < 0.0051 * 4))
              and bool(np.all(np.abs(fl_total.cpu().numpy() - cpu_flight) < 0.5)),
              f"Newton {np.round(lab, 4).tolist()}, CPU MuJoCo {cpu_total:.4f} ({n_td} touchdowns); flight substeps "
              f"Newton {fl_total.cpu().numpy().round(2).tolist()} CPU {cpu_flight}; "
              f"qvel[0] {qv:.4f} == torso v_x {tv:.4f}")

    print("\n== fall termination + random-action self-penetration (rule M)")
    env.reset()
    tilt = torch.tensor([[math.sin(math.radians(35)), 0.0, 0.0, math.cos(math.radians(35))]], device=dev)  # 70 deg roll
    q0 = torch.zeros(1, nj, device=dev)
    place([2], q0, z=1.0, quat=tilt)
    step(1)
    from quad_tasks_v3.mdp import fallen_mask
    from isaaclab.managers import SceneEntityCfg as SEC
    tcfg = SEC("robot", body_names=[spec.TORSO])
    tcfg.resolve(env.scene)
    fm = fallen_mask(env, tcfg, spec.FALL_UP_DOT, spec.FALL_HEIGHT)
    check("70 deg roll -> fall; upright worlds -> no fall", bool(fm[2]) and not bool(fm[0]), str(fm.cpu().tolist()))
    place([3], q0, z=1.0)
    step(1)
    h3 = float(d.body_link_pos_w.torch[3, bid[spec.TORSO], 2] - env.scene.env_origins[3, 2])
    check("height branch: an upright torso at height h is a fall iff h < z_min",
          not bool(fallen_mask(env, tcfg, spec.FALL_UP_DOT, spec.FALL_HEIGHT)[3])
          and bool(fallen_mask(env, tcfg, spec.FALL_UP_DOT, h3 + 0.01)[3])
          and not bool(fallen_mask(env, tcfg, spec.FALL_UP_DOT, h3 - 0.01)[3]), f"h {h3:.4f}")
    from quad_tasks_v3.mdp import floor_touch_mask
    trk = robot.contact_tracker
    tgeoms = [trk.touch_torso] + [int(trk.touch_upper[i]) for i in range(4)]
    tbodies = [mm.body(int(mm.geom_bodyid[g])).name for g in tgeoms]
    lut_names = sorted(n for n in robot.body_names if any(x.endswith('_' + n) for x in tbodies))
    check("touch-fall geoms (substep tracker) = torso + the 4 upper legs", lut_names == sorted(spec.FALL_TOUCH_BODIES),
          str(lut_names))
    side = torch.tensor([[math.sin(math.radians(45)), 0.0, 0.0, math.cos(math.radians(45))]], device=dev)  # 90 deg roll
    env.reset()
    place([1], q0, z=0.60, quat=side)  # lies on the legs of one side (they stick out past the box)
    for _ in range(80):
        robot.write_data_to_sim()
        sim.step(render=False)
        env.scene.update(dt)
    tm_ = floor_touch_mask(env)
    tu_ = floor_touch_mask(env, spec.UPPER_LEGS)
    env.step(torch.zeros(env.num_envs, 8, device=dev))
    ts_ = robot.contact_tracker.touch_step.clone()
    check("touch_this_step (any of the 10 substeps) after an env.step: side-lying world only",
          bool(ts_[1]) and not bool(ts_[0]) and not bool(ts_[2]) and bool(env.termination_manager.get_term("fall")[1]),
          f"{ts_.cpu().tolist()}")
    check("lying on its side: torso / upper legs touch the floor -> fall; standing worlds -> no touch",
          bool(tm_[1]) and bool(tu_[1]) and not bool(tm_[0]) and not bool(tm_[2]),
          f"touch {tm_.cpu().tolist()}, upper legs only {tu_.cpu().tolist()}")
    env.reset()
    torch.manual_seed(0)
    a = torch.zeros(env.num_envs, 8, device=dev)
    worst_self, worst_foot, worst_body, falls_terminal = 0.0, 0.0, 0.0, 0
    foot_ids = [[g for g in range(mm.ngeom) if mm.geom_bodyid[g] == mj_body(n_)][0] for n_ in spec.LOWER_LEGS]
    for k in range(int(round(10.0 / env.step_dt))):
        a = torch.clamp(a + 0.5 * torch.randn_like(a), -1, 1)
        obs, rew, term, trunc, extras = env.step(a)
        falls_terminal += int(env.termination_manager.get_term("fall").sum())
        n = int(solver.mjw_data.nacon.numpy()[0])
        cg, cd = con.geom.numpy()[:n], con.dist.numpy()[:n]
        fl_ = (cg[:, 0] == ground) | (cg[:, 1] == ground)
        ft_ = fl_ & (np.isin(cg[:, 0], foot_ids) | np.isin(cg[:, 1], foot_ids))
        if (~fl_).any():
            worst_self = max(worst_self, float(-cd[~fl_].min()))
        if ft_.any():
            worst_foot = max(worst_foot, float(-cd[ft_].min()))
        if (fl_ & ~ft_).any():
            worst_body = max(worst_body, float(-cd[fl_ & ~ft_].min()))
        bad = ~torch.isfinite(obs["policy"]).all()
        if bad:
            break
    # Soft contacts sink a little when legs driven at 300 N*m hit each other or a 60 kg torso lands: CPU MuJoCo on
    # quad.xml itself reaches ~14 mm leg-leg (training/quad/check_quad.py). Passing THROUGH would mean depths of the
    # order of a capsule radius (72-90 mm) or no contact. Round 9: the feet's floor contacts are soft (0.03 s), and so
    # are the torso / upper-leg floor contacts here (ground priority; in quad.xml they stay 0.01, but such a contact
    # is a fall that ends the episode on that step), so the non-foot floor bound is looser.
    # Round 9 per-geom soft feet (solref 0.03, priority 1) are soft in ALL their contacts, foot-foot included: CPU
    # MuJoCo on this quad.xml reaches 65 mm lower-leg/lower-leg and 71 mm foot-floor overlap under the same kind of
    # random actions (combined capsule radii 144 mm), so the self bound follows it.
    soft = bool(ref.npair == 0 and any(ref.geom_solref[ref.geom(g).id][0] > 0.011 for g in spec.RIG["footGeoms"]))
    self_max = 0.07 if soft else 0.03
    check(f"random actions 10 s x 4 envs: finite; self-penetration < {self_max * 100:.0f} cm; foot-floor < 8 cm; "
          "torso/thigh-floor < 12 cm",
          not bool(bad) and worst_self < self_max and worst_foot < 0.08 and worst_body < 0.12,
          f"max self {worst_self * 1000:.1f} mm, foot-floor {worst_foot * 1000:.1f} mm, torso/thigh-floor "
          f"{worst_body * 1000:.1f} mm (terminal contacts), falls (terminal) {falls_terminal}")

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
