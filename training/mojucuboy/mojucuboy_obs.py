"""MojucuBoy's 75-element observation vector, Python side.

CONTRACT with Assets/Creature/Scripts/Boy/MojucuBoyObservation.cs. The two build the
same vector in the same order; change one, change both, or the policy is fed
garbage that looks plausible.

  0.. 2  gravity_local  R^T @ (0,0,-1), R = xmat of the root body
  3.. 5  linvel_local   R^T @ qvel[0:3]  (a free joint stores linear velocity in WORLD)
  6.. 8  angvel_local   qvel[3:6]        (a free joint stores angular velocity in BODY)
  9..11  command        cos(heading error), sin(heading error), target speed
 12..32  joint_pos      qpos, in actuator order
 33..53  joint_vel      qvel, in actuator order
 54..74  last_action    previous clipped action

Forward is body-local +Y: the rig is authored facing MuJoCo +Y so org.mujoco maps
it onto Unity +Z, the race direction.
"""

from __future__ import annotations

import numpy as np

JOINT_COUNT = 21
OBS_SIZE = 9 + 3 + 3 * JOINT_COUNT  # 75
ACTION_SIZE = JOINT_COUNT
FORWARD_AXIS = 1
ROOT_BODY = "hips"


def heading_error(rot: np.ndarray, command_heading: float) -> float:
    """Signed angle from the racer's world heading to the commanded one, in (-pi, pi]."""
    forward = rot[:, FORWARD_AXIS]
    heading = np.arctan2(forward[1], forward[0])
    error = command_heading - heading
    return float((error + np.pi) % (2.0 * np.pi) - np.pi)


def build(data, root_body_id: int, qpos_addr, dof_addr,
          command_heading: float, command_speed: float, last_action) -> np.ndarray:
    """Assemble the observation from a stepped mjData."""
    obs = np.zeros(OBS_SIZE, dtype=np.float32)
    rot = data.xmat[root_body_id].reshape(3, 3)  # body -> world

    obs[0:3] = rot.T @ np.array([0.0, 0.0, -1.0])
    obs[3:6] = rot.T @ data.qvel[0:3]
    obs[6:9] = data.qvel[3:6]

    error = heading_error(rot, command_heading)
    obs[9] = np.cos(error)
    obs[10] = np.sin(error)
    obs[11] = command_speed

    obs[12:12 + JOINT_COUNT] = data.qpos[qpos_addr]
    obs[12 + JOINT_COUNT:12 + 2 * JOINT_COUNT] = data.qvel[dof_addr]
    obs[12 + 2 * JOINT_COUNT:] = last_action
    return obs


def addresses(model, actuator_order):
    """qpos and qvel addresses for each actuated joint, resolved by NAME.

    Never by index: org.mujoco renames every element on export, so the actuator
    order in mojucuboy_rig.json is the contract and the index is not.
    """
    import mujoco

    qpos_addr, dof_addr = [], []
    for name in actuator_order:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise KeyError(f"joint {name!r} not in model")
        qpos_addr.append(int(model.jnt_qposadr[jid]))
        dof_addr.append(int(model.jnt_dofadr[jid]))
    return np.array(qpos_addr), np.array(dof_addr)
