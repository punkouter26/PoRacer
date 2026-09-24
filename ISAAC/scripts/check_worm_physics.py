#!/usr/bin/env python3
"""Runtime checks of the Worm5 Isaac task against WORM_SPEC.md (what PhysX actually built).

    set OMNI_KIT_ACCEPT_EULA=YES
    ISAAC\\isaaclab\\.venv\\Scripts\\python.exe ISAAC\\scripts\\check_worm_physics.py

Prints: PhysX joint/body order and the action->sim joint map, masses and inertias (vs the
analytic solid capsule MuJoCo uses), drive/armature/viscous-friction/limit values, materials,
the joint sign test from WORM_SPEC.md "Unity mapping", a damping decay test (joint viscous
friction vs drive damping vs none, in free fall), a 1 s rest test and one observation vector.
"""

import argparse
import math
import os
import sys

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Isaac-Worm5-Flat-Play-v0")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import worm_tasks  # noqa: E402,F401
from isaaclab.utils.math import quat_apply_inverse  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from worm_tasks import spec  # noqa: E402
from worm_tasks.worm_cfg import DAMPING_MODE  # noqa: E402

FAILS = []


def check(label, ok, detail=""):
    print(f"  [{'OK ' if ok else 'FAIL'}] {label}  {detail}")
    if not ok:
        FAILS.append(label)


def capsule_inertia(m, r, half):
    H = 2 * half
    vc, vs = math.pi * r * r * H, 4.0 / 3.0 * math.pi * r ** 3
    mc, ms = m * vc / (vc + vs), m * vs / (vc + vs)
    axial = mc * r * r / 2 + ms * 2 * r * r / 5
    perp = mc * (3 * r * r + H * H) / 12 + ms * (2 * r * r / 5 + H * H / 4 + 3 * H * r / 8)
    return axial, perp


def main():
    cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    cfg.scene.num_envs = 4
    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset()
    robot = env.scene["robot"]
    sim = env.sim
    dt = sim.get_physics_dt()
    view = robot.root_physx_view

    print(f"\n== order (damping mode: {DAMPING_MODE})")
    print(f"  PhysX joints: {robot.joint_names}")
    print(f"  PhysX bodies: {robot.body_names}")
    act_ids = env.action_manager.get_term("joint_pos")._joint_ids
    act_ids = list(range(robot.num_joints)) if isinstance(act_ids, slice) else list(act_ids)
    mapped = [robot.joint_names[i] for i in act_ids]
    check("action term joint order == spec ACTION_ORDER", mapped == spec.ACTION_ORDER, str(mapped))
    obs_dims = env.observation_manager.group_obs_dim["policy"]
    check("observation size 35", obs_dims == (35,), str(obs_dims))
    print(f"  obs terms: {list(zip(env.observation_manager.active_terms['policy'], env.observation_manager.group_obs_term_dim['policy']))}")

    print("\n== mass / inertia")
    masses = view.get_masses()[0]
    inertias = view.get_inertias()[0].reshape(-1, 3, 3)
    ax_ref, perp_ref = capsule_inertia(spec.RIG["segmentMass"], spec.RIG["radius"], spec.RIG["capsuleHalfLength"])
    for b, name in enumerate(robot.body_names):
        I = inertias[b]
        print(f"  {name:6s} m={float(masses[b]):.4f}  I_diag=({float(I[0,0]):.6f}, {float(I[1,1]):.6f}, {float(I[2,2]):.6f})")
        if name.startswith("seg"):
            check(f"{name} mass", abs(float(masses[b]) - spec.RIG["segmentMass"]) < 1e-4)
            check(f"{name} inertia = solid capsule", abs(float(I[0, 0]) - ax_ref) < 2e-5 and abs(float(I[1, 1]) - perp_ref) < 2e-5
                  and abs(float(I[2, 2]) - perp_ref) < 2e-5, f"ref axial {ax_ref:.6f} perp {perp_ref:.6f}")
        else:
            check(f"{name} mass", abs(float(masses[b]) - 0.1) < 1e-5)
    check("total mass 7.08 kg", abs(float(masses.sum()) - spec.RIG["totalMass"]) < 1e-3, f"{float(masses.sum()):.4f}")

    print("\n== joints")
    d = robot.data
    lim = d.joint_pos_limits[0]
    fr = view.get_dof_friction_properties()[0]
    for j, name in enumerate(robot.joint_names):
        print(f"  {name:9s} limits [{float(lim[j,0]):+.5f}, {float(lim[j,1]):+.5f}]  kp {float(d.joint_stiffness[0,j]):.2f}"
              f"  kd {float(d.joint_damping[0,j]):.2f}  effort {float(d.joint_effort_limits[0,j]):.2f}"
              f"  armature {float(d.joint_armature[0,j]):.4f}  friction(s,d,visc) {[round(float(v), 4) for v in fr[j]]}")
    check("limits +/-0.785398", bool(torch.all((lim[:, 1] - 0.785398).abs() < 1e-4) and torch.all((lim[:, 0] + 0.785398).abs() < 1e-4)))
    check("kp 30", bool(torch.all((d.joint_stiffness[0] - 30).abs() < 1e-5)))
    check("effort limit 12", bool(torch.all((d.joint_effort_limits[0] - 12).abs() < 1e-5)))
    check("armature 0.01", bool(torch.all((d.joint_armature[0] - 0.01).abs() < 1e-6)))
    if DAMPING_MODE == "joint":
        check("drive damping 0 + joint viscous friction 1.0",
              bool(torch.all(d.joint_damping[0].abs() < 1e-6) and torch.all((fr[:, 2] - 1.0).abs() < 1e-5)))
    else:
        check("drive damping 1.0, no viscous friction", bool(torch.all((d.joint_damping[0] - 1).abs() < 1e-6)))
    mats = view.get_material_properties()[0]
    print(f"  worm shape materials (static, dynamic, restitution): {mats.tolist()}")
    check("worm material 0.9/0.9/0", bool(torch.all((mats[:, :2] - 0.9).abs() < 1e-5) and torch.all(mats[:, 2] == 0)))

    ids = {n: i for i, n in enumerate(robot.joint_names)}
    bid = {n: i for i, n in enumerate(robot.body_names)}

    def place(env_ids, joint_pos, joint_vel=None, z=1.0):
        env_ids_t = torch.tensor(env_ids, device=env.device)
        root = d.default_root_state[env_ids_t].clone()
        root[:, :3] = env.scene.env_origins[env_ids_t]
        root[:, 2] += z
        robot.write_root_state_to_sim(root, env_ids=env_ids_t)
        jv = torch.zeros_like(joint_pos) if joint_vel is None else joint_vel
        robot.write_joint_state_to_sim(joint_pos, jv, env_ids=env_ids_t)
        robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)

    def step(n):
        for _ in range(n):
            robot.write_data_to_sim()
            sim.step(render=False)
            env.scene.update(dt)

    print("\n== sign test (WORM_SPEC 'Unity mapping'): +0.5 rad, seg1 relative to seg0's frame")
    for jn, axis, sign in (("j0_yaw", 1, -1), ("j0_pitch", 2, +1)):
        q = torch.zeros(1, robot.num_joints, device=env.device)
        q[0, ids[jn]] = 0.5
        place([0], q, z=1.0)
        step(1)
        p0, q0 = d.body_link_pos_w[0, bid["seg0"]], d.body_link_quat_w[0, bid["seg0"]]
        rel = quat_apply_inverse(q0.unsqueeze(0), (d.body_link_pos_w[0, bid["seg1"]] - p0).unsqueeze(0))[0]
        want = "-y" if sign < 0 else "+z"
        check(f"{jn}=+0.5 swings seg1 toward MuJoCo {want}", float(rel[axis]) * sign > 0.03,
              f"seg1 in seg0 frame = ({float(rel[0]):+.4f}, {float(rel[1]):+.4f}, {float(rel[2]):+.4f}); "
              f"q={float(d.joint_pos[0, ids[jn]]):+.4f}")

    print("\n== damping test: free fall, kp 0, j3_yaw qdot0 = 5 rad/s")
    kp0 = d.joint_stiffness.clone()
    kd0 = d.joint_damping.clone()
    visc0 = d.joint_viscous_friction_coeff.clone()
    e = torch.tensor([0, 1, 2], device=env.device)
    robot.write_joint_stiffness_to_sim(0.0, env_ids=e)
    robot.write_joint_damping_to_sim(torch.tensor([[0.0], [0.0], [1.0]], device=env.device).expand(3, robot.num_joints).contiguous(), env_ids=e)
    robot.write_joint_viscous_friction_coefficient_to_sim(
        torch.tensor([[1.0], [0.0], [0.0]], device=env.device).expand(3, robot.num_joints).contiguous(), env_ids=e)
    for a in robot.actuators.values():
        a.stiffness[e] = 0.0
        a.damping[e] = d.joint_damping[e]
    q = torch.zeros(3, robot.num_joints, device=env.device)
    qd = torch.zeros_like(q)
    qd[:, ids["j3_yaw"]] = 5.0
    place([0, 1, 2], q, qd, z=2.0)
    trace = []
    for k in range(10):
        step(1)
        trace.append([float(v) for v in d.joint_vel[:3, ids["j3_yaw"]]])
    for lbl, i in (("joint viscous 1.0", 0), ("none           ", 1), ("drive damping 1", 2)):
        print(f"  {lbl}: qdot after 1/4/10 substeps = {trace[0][i]:.3f} / {trace[3][i]:.3f} / {trace[9][i]:.3f}")
    check("viscous friction damps like drive damping", abs(trace[9][0] - trace[9][2]) < 0.15 * abs(trace[9][1]) and trace[9][0] < 0.8 * trace[9][1],
          "(env0 ~ env2, both well below env1)")
    robot.write_joint_stiffness_to_sim(kp0[e], env_ids=e)
    robot.write_joint_damping_to_sim(kd0[e], env_ids=e)
    robot.write_joint_viscous_friction_coefficient_to_sim(visc0[e], env_ids=e)
    for a in robot.actuators.values():
        a.stiffness[e] = kp0[e]
        a.damping[e] = kd0[e]

    print("\n== rest test: spec reset, zero action, 1 s")
    obs, _ = env.reset()
    o = obs["policy"][0]
    print(f"  obs[0] after reset: {[round(float(v), 4) for v in o]}")
    check("gravity_b ~ (0, 0, -1)", float(o[2]) < -0.99)
    yaw_head = 2 * math.atan2(float(d.root_quat_w[0, 3]), float(d.root_quat_w[0, 0]))
    check("goal_dir_b ~ (cos yaw, -sin yaw)", abs(float(o[33]) - math.cos(yaw_head)) < 0.1 and abs(float(o[34]) + math.sin(yaw_head)) < 0.1,
          f"yaw(head) {math.degrees(yaw_head):+.1f} deg")
    zero = torch.zeros(env.num_envs, 8, device=env.device)
    for _ in range(50):
        obs, rew, term, trunc, extras = env.step(zero)
    z = d.body_link_pos_w[:, [bid[f"seg{i}"] for i in range(5)], 2] - env.scene.env_origins[:, 2:3]
    v = d.body_link_lin_vel_w[:, bid["seg2"]]
    print(f"  segment heights env0: {[round(float(x), 4) for x in z[0]]}   seg2 |v| max over envs {float(v.norm(dim=-1).max()):.4f}")
    print(f"  per-step reward env0 (zero action, at rest): {float(rew[0]):.6f}")
    check("worm rests on the floor (z ~ radius 0.045)", bool(torch.all((z - 0.045).abs() < 0.01)))
    check("worm at rest", float(v.norm(dim=-1).max()) < 0.02)

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
        app.close()
        os._exit(code)
