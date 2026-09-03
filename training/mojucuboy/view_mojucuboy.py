"""Watch MojucuBoy walk in MuJoCo's native viewer.

  .venv-mjwarp\\Scripts\\python.exe training/mojucuboy/view_mojucuboy.py

Opens the MuJoCo UI and drives the trained policy in real time. A red marker shows
the current target; when he gets within ARRIVE_RADIUS a new one is placed, so he
keeps walking from goal to goal.

Note what "walking to a target" means here: the policy is HEADING-tracked, not
target-seeking. Its observation carries cos/sin of the heading error and a target
speed -- never a position or a distance -- so this script converts the target into
a heading each step. He therefore walks toward a point but has no notion of
arriving and stopping; the arrival check is done here, outside the policy.

Keys: the usual MuJoCo viewer bindings. Space pauses.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import mojucuboy_obs  # noqa: E402

DECIMATION = 4
ARRIVE_RADIUS = 1.5
TARGET_RANGE = 8.0
TARGET_SPEED = 1.5
MIN_HEIGHT = 0.45
MIN_UPRIGHT = 0.30
ACTION_SCALE = 0.6
# Trained heading band is spawn_yaw +/- 0.6 rad; stay inside it.
TURN_LIMIT = 0.5


def load_policy(run: str):
    """Rebuild the network on CPU. train_mojucuboy imports the Warp env, which would
    initialise CUDA and compile kernels for nothing, so the classes are rebuilt
    here instead of imported."""
    import torch.nn as nn

    checkpoint = torch.load(HERE / "runs" / run / "policy.pt",
                            map_location="cpu", weights_only=False)
    state = checkpoint["model"]
    hidden = (128, 128, 128)

    def mlp(in_size, out_size):
        layers, last = [], in_size
        for size in hidden:
            layers += [nn.Linear(last, size), nn.ELU()]
            last = size
        return nn.Sequential(*layers, nn.Linear(last, out_size))

    class Policy(nn.Module):
        def __init__(self):
            super().__init__()
            self.actor = mlp(mojucuboy_obs.OBS_SIZE, mojucuboy_obs.ACTION_SIZE)
            self.register_buffer("mean", torch.zeros(mojucuboy_obs.OBS_SIZE))
            self.register_buffer("var", torch.ones(mojucuboy_obs.OBS_SIZE))

        def forward(self, obs):
            normed = torch.clamp((obs - self.mean) / torch.sqrt(self.var + 1e-8), -10, 10)
            return torch.tanh(self.actor(normed))

    policy = Policy()
    policy.mean.copy_(state["norm.mean"])
    policy.var.copy_(state["norm.var"])
    policy.actor.load_state_dict(
        {k[len("actor."):]: v for k, v in state.items() if k.startswith("actor.")})
    policy.eval()
    print(f"policy: {run}, iteration {checkpoint.get('iteration', '?')}")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default="boy_chase01")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="playback rate; 1.0 is real time")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(HERE / "mojucuboy_roundtrip.xml"))
    data = mujoco.MjData(model)
    policy = load_policy(args.run)

    import json
    rig = json.loads((HERE / "mojucuboy_rig.json").read_text())
    order = rig["actuator_order"]
    stance = np.array(rig["stance_qpos"])
    joint_stance = np.array([j["stance_rad"] for j in rig["joints"]])
    lo = np.array([j["range_rad"][0] for j in rig["joints"]])
    hi = np.array([j["range_rad"][1] for j in rig["joints"]])
    half = 0.5 * (hi - lo)

    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hips")
    qpos_addr, dof_addr = mojucuboy_obs.addresses(model, order)
    act_addr = np.array([mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "act_" + n)
                         for n in order])

    rng = np.random.default_rng(0)
    action = np.zeros(mojucuboy_obs.ACTION_SIZE, dtype=np.float32)
    target = np.array([0.0, TARGET_RANGE])

    def reset():
        nonlocal action, target
        mujoco.mj_resetData(model, data)
        data.qpos[:] = stance
        data.qvel[:] = 0
        action = np.zeros(mojucuboy_obs.ACTION_SIZE, dtype=np.float32)
        # He spawns facing +y and was only ever trained to steer within about
        # +/-34 degrees of his own heading, so the first target goes straight
        # ahead rather than anywhere on the circle.
        target = np.array([0.0, TARGET_RANGE])
        mujoco.mj_forward(model, data)

    reset()
    step_dt = model.opt.timestep * DECIMATION
    print(f"policy {1/step_dt:.0f} Hz, physics {1/model.opt.timestep:.0f} Hz")
    print("red marker = target. Space pauses. Close the window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Track the racer rather than leaving the camera at the origin.
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = root
        viewer.cam.distance = 4.0
        viewer.cam.elevation = -12.0

        falls = 0
        wall = time.perf_counter()
        while viewer.is_running():
            here = data.qpos[0:2].copy()
            delta = target - here
            if np.linalg.norm(delta) < ARRIVE_RADIUS:
                # Place the next target within TURN_LIMIT of where he is already
                # facing. Asking for more is extrapolation beyond the trained
                # heading band and just makes him fall over.
                forward = data.xmat[root].reshape(3, 3)[:, mojucuboy_obs.FORWARD_AXIS]
                facing = np.arctan2(forward[1], forward[0])
                angle = facing + rng.uniform(-TURN_LIMIT, TURN_LIMIT)
                target = here + np.array([np.cos(angle), np.sin(angle)]) * TARGET_RANGE
                delta = target - here
            heading = float(np.arctan2(delta[1], delta[0]))

            obs = mojucuboy_obs.build(data, root, qpos_addr, dof_addr,
                                heading, TARGET_SPEED, action)
            with torch.no_grad():
                action = policy(torch.from_numpy(obs)).numpy()
            data.ctrl[act_addr] = np.clip(joint_stance + ACTION_SCALE * half * action, lo, hi)

            for _ in range(DECIMATION):
                mujoco.mj_step(model, data)

            rot = data.xmat[root].reshape(3, 3)
            # rot[2][2] is +1 when upright. The env derives this as -obs[2],
            # and obs[2] is itself -rot[2][2] -- so the double negation cancels.
            # Writing -rot[2][2] here reads -1 when upright and terminates instantly.
            if data.qpos[2] < MIN_HEIGHT or rot[2, 2] < MIN_UPRIGHT:
                falls += 1
                print(f"fell ({falls}); resetting")
                reset()

            # Draw the target as a marker in the viewer's user scene.
            viewer.user_scn.ngeom = 0
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[0], mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.25, 0.0, 0.0]),
                np.array([target[0], target[1], 0.25]),
                np.eye(3).ravel(), np.array([0.9, 0.2, 0.2, 0.85], dtype=np.float32))
            viewer.user_scn.ngeom = 1
            viewer.sync()

            wall += step_dt / max(args.speed, 1e-3)
            remaining = wall - time.perf_counter()
            if remaining > 0:
                time.sleep(remaining)
            else:
                wall = time.perf_counter()
    return 0


if __name__ == "__main__":
    sys.exit(main())
