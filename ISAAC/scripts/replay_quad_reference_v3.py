#!/usr/bin/env python3
"""Open-loop replay of the quad's reference trot (QUAD_SPEC rounds 8-9) in Isaac Lab 3 / Newton AND CPU MuJoCo.

    ISAAC\\isaaclab3\\.venv\\Scripts\\python.exe ISAAC\\scripts\\replay_quad_reference_v3.py [--envs 16] [--seconds 20]

Checks the Isaac port of the reference gait (quad_tasks_v3/mdp/gait.py) against an independent CPU MuJoCo
re-implementation on training/quad/quad.xml:

* both start from the spec reset without noise (torso at 0.92 m, rest pose, yaw 0), clock phases
  phi0 = 2 pi k / envs;
* every policy step the action is the reference at the clock phase the observation shows (pre-step phase):
  a = [0.3 sin(psi), 0.5 max(0, -cos(psi))] per leg, psi = phi + (0, pi, pi, 0) for RL, FL, RR, FR;
* after the 10 physics steps the clock advances by 2 pi 1.5 Hz 0.05 s; the rewards use the post-step clock:
  gait_ref = exp(-mean (q - q_ref)^2 / sigma^2), contact_phase = share of feet whose DEBOUNCED state (3 substeps)
  equals "stance expected" (-cos psi_hip < 0).
Reports mean gait_ref (sigma 0.2 and 0.3), mean contact_phase, speed (torso x displacement / time) and falls, per
simulator. No training; nothing is written.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

from isaaclab_tasks.utils.sim_launcher import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--envs", type=int, default=16)
parser.add_argument("--seconds", type=float, default=20.0)
add_launcher_args(parser)
args = parser.parse_args()

import gymnasium as gym  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import quad_tasks_v3  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from quad_tasks_v3 import spec  # noqa: E402
from quad_tasks_v3.mdp.gait import HIP_PHASES, get_clock  # noqa: E402

TASK = "Isaac-Quad-Flat-Newton-Play-v0"


def cpu_replay(phi0: float, steps: int) -> dict:
    m = mujoco.MjModel.from_xml_path(spec.QUAD_XML)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("spawn").id)
    act_ids = [m.actuator(n).id for n in spec.ACTION_ORDER_MJCF]
    qadr = [m.jnt_qposadr[m.actuator_trnid[a][0]] for a in act_ids]
    floor = m.geom("floor").id
    feet = [m.geom(g).id for g in spec.RIG["footGeoms"]]
    torso = m.body(spec.TORSO_MJCF).id
    offs = np.array([ph for ph in HIP_PHASES for _ in (0, 1)])
    amps = np.array([spec.GAIT_AMP_HIP, spec.GAIT_AMP_KNEE] * 4)
    lift = np.array([False, True] * 4)

    def ref(phi):
        psi = phi + offs
        return amps * np.where(lift, spec.GAIT_KNEE_SIGN * np.maximum(0.0, -np.cos(psi)), np.sin(psi))

    phi = phi0
    db, run = [1] * 4, [0] * 4
    x0 = d.xpos[torso][0]
    g2, g3, cp, fell = 0.0, 0.0, 0.0, None
    for k in range(steps):
        d.ctrl[act_ids] = np.clip(ref(phi), -1, 1) * spec.ACTION_SCALE
        for _ in range(spec.DECIMATION):
            mujoco.mj_step(m, d)
            down = [0] * 4
            for ci in range(d.ncon):
                c = d.contact[ci]
                o = c.geom[1] if c.geom[0] == floor else (c.geom[0] if c.geom[1] == floor else -1)
                if o in feet and c.dist < 0:
                    down[feet.index(o)] = 1
            for f in range(4):
                run[f] = run[f] + 1 if down[f] != db[f] else 0
                if run[f] >= 3:
                    db[f], run[f] = down[f], 0
        phi = (phi + 2 * math.pi * spec.GAIT_FREQ * spec.STEP_DT) % (2 * math.pi)
        q = d.qpos[qadr]
        err = np.mean((q - (np.array(spec.REST_POSE) + spec.ACTION_SCALE * ref(phi))) ** 2)
        g2 += math.exp(-err / 0.2 ** 2)
        g3 += math.exp(-err / 0.3 ** 2)
        stance = -np.cos(phi + np.array(HIP_PHASES)) < 0.0
        cp += float(np.mean(np.array(db, bool) == stance))
        up = d.xmat[torso].reshape(3, 3)[2, 2]
        if up < spec.FALL_UP_DOT or d.xpos[torso][2] < spec.FALL_HEIGHT:
            fell = k + 1
            break
    n = k + 1
    return {"gait_ref_s02": g2 / n, "gait_ref_s03": g3 / n, "contact_phase": cp / n,
            "speed": (d.xpos[torso][0] - x0) / (steps * spec.STEP_DT), "fell_at": fell}


def main():
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = args.envs
    cfg.episode_length_s = args.seconds + 1.0
    with launch_simulation(cfg, args):
        env = gym.make(TASK, cfg=cfg).unwrapped
        assert spec.GAIT, "set QUAD_V3_ROUND >= 8"
        env.reset()
        robot, dev, n = env.scene["robot"], env.device, env.num_envs
        e = torch.arange(n, device=dev, dtype=torch.int32)
        pose = torch.zeros(n, 7, device=dev)
        pose[:, :3] = env.scene.env_origins
        pose[:, 2] += spec.SPAWN_Z
        pose[:, 6] = 1.0
        robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=e)
        robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=e)
        q0 = torch.zeros(n, 8, device=dev)
        robot.write_joint_state_to_sim_index(position=q0, velocity=q0, env_ids=e)
        robot.set_joint_position_target_index(target=q0, env_ids=e)
        tracker = robot.contact_tracker
        tracker.reset(slice(None))
        tracker.t_counter.zero_()
        clock = get_clock(env)
        phi0 = torch.arange(n, device=dev, dtype=torch.float32) * (2 * math.pi / n)
        clock.phase = phi0.clone()
        clock.last_step = int(env.common_step_counter)
        rm = env.reward_manager
        cg, cc = rm.active_terms.index("gait_ref"), rm.active_terms.index("contact_phase")
        torso = robot.body_names.index(spec.TORSO)
        x0 = (robot.data.body_link_pos_w.torch[:, torso, 0] - env.scene.env_origins[:, 0]).clone()
        steps = int(round(args.seconds / env.step_dt))
        gsum, csum = torch.zeros(n, device=dev), torch.zeros(n, device=dev)
        ended = torch.zeros(n, dtype=torch.bool, device=dev)
        live = torch.zeros(n, device=dev)
        x_end = torch.zeros(n, device=dev)
        for _ in range(steps):
            a = clock.reference_action()  # the phase the observation shows (pre-step)
            _, _, term, trunc, _ = env.step(a)
            alive = (~ended).float()
            gsum += alive * rm._step_reward[:, cg] / spec.W_GAIT_REF
            csum += alive * rm._step_reward[:, cc] / spec.W_CONTACT_PHASE
            live += alive
            now_end = (term | trunc) & ~ended
            x_end = torch.where(now_end, robot.data.body_link_pos_w.torch[:, torso, 0] - env.scene.env_origins[:, 0], x_end)
            ended |= term | trunc
        x_end = torch.where(ended, x_end, robot.data.body_link_pos_w.torch[:, torso, 0] - env.scene.env_origins[:, 0])
        isaac = {"gait_ref": (gsum / live).cpu().numpy(), "contact_phase": (csum / live).cpu().numpy(),
                 "speed": ((x_end - x0) / args.seconds).cpu().numpy(), "fell": ended.cpu().numpy()}
        cpu = [cpu_replay(float(p), steps) for p in phi0.cpu().numpy()]
        print(f"\nround {spec.ROUND}, gait_ref sigma {spec.GAIT_REF_SIGMA}; {n} phases, {args.seconds:g} s, open-loop reference")
        print(" phi0   | Isaac gait_ref contact speed fell | CPU MuJoCo gait_ref(s0.2) (s0.3) contact speed fell")
        for i in range(n):
            c = cpu[i]
            print(f" {float(phi0[i]):5.2f}  |  {isaac['gait_ref'][i]:.3f}  {isaac['contact_phase'][i]:.3f}  {isaac['speed'][i]:+.3f}  "
                  f"{int(isaac['fell'][i])}   |  {c['gait_ref_s02']:.3f}  {c['gait_ref_s03']:.3f}  {c['contact_phase']:.3f}  "
                  f"{c['speed']:+.3f}  {c['fell_at']}")
        key = "gait_ref_s02" if spec.GAIT_REF_SIGMA == 0.2 else "gait_ref_s03"
        print(f"MEAN Isaac: gait_ref {isaac['gait_ref'].mean():.3f}, contact_phase {isaac['contact_phase'].mean():.3f}, "
              f"speed {isaac['speed'].mean():+.3f} m/s, falls {int(isaac['fell'].sum())}/{n}")
        print(f"MEAN CPU MuJoCo: gait_ref {np.mean([c[key] for c in cpu]):.3f} (sigma 0.3: {np.mean([c['gait_ref_s03'] for c in cpu]):.3f}), "
              f"contact_phase {np.mean([c['contact_phase'] for c in cpu]):.3f}, speed {np.mean([c['speed'] for c in cpu]):+.3f} m/s, "
              f"falls {sum(c['fell_at'] is not None for c in cpu)}/{n}")
        env.close()


if __name__ == "__main__":
    code = 1
    try:
        main()
        code = 0
    except BaseException:  # noqa: BLE001
        import traceback

        traceback.print_exc()
    finally:
        sys.stdout.flush()
        os._exit(code)
