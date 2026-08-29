"""MJX environment for the custom creature in Assets/Creature/creature.xml.

Built on MuJoCo Playground's `MjxEnv` (the actively maintained MJX env base --
brax's own System/pipeline is deprecated), trained with Brax PPO.

The observation layout below is a CONTRACT with the Unity runtime
(Assets/Creature/Scripts/CreatureAgent.cs). Change one, change both, and
re-export the policy.
"""
from __future__ import annotations

import jax
import jax.numpy as jp
import mujoco
from etils import epath
from ml_collections import config_dict
from mujoco import mjx
from mujoco_playground import MjxEnv, State
from mujoco_playground._src import mjx_env

XML_PATH = (
    epath.Path(__file__).resolve().parent.parent / "Assets" / "Creature" / "creature.xml"
)

# --- observation contract (33 floats, in this exact order) ---------------
OBS_LAYOUT = (
    ("gravity_local", 3),  # world -Z rotated into the torso frame (upright sensing)
    ("linvel_local", 3),   # qvel[0:3] (world frame) rotated into the torso frame
    ("angvel_local", 3),   # qvel[3:6] -- already torso-local for a MuJoCo free joint
    ("joint_pos", 8),      # qpos[7:] in actuator order
    ("joint_vel", 8),      # qvel[6:] in actuator order
    ("last_action", 8),    # previous clipped action
)
OBS_SIZE = sum(n for _, n in OBS_LAYOUT)  # 33
ACTION_SIZE = 8

_GRAVITY_DIR = jp.array([0.0, 0.0, -1.0])


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,        # 50 Hz policy
      sim_dt=0.004,        # 5 physics substeps per control step
      episode_length=1000,  # 20 s
      action_repeat=1,
      # Reward shaping. The survival terms must stay SMALL relative to the
      # forward term, or standing still becomes the optimal policy: an earlier
      # run converged to 0.04 m/s because healthy+upright paid 0.8/step while
      # creeping forward paid ~0.07/step. Standing now earns 0.15/step against
      # ~3.0/step for walking at the target speed.
      target_speed=1.5,
      forward_weight=2.0,
      upright_weight=0.05,
      healthy_reward=0.10,
      ctrl_cost=0.005,
      joint_vel_cost=1e-4,
      # termination / reset
      min_height=0.16,
      max_height=0.60,
      min_uprightness=0.2,
      reset_noise=0.10,
      # MJX physics backend: 'jax' (default) or 'warp' (MuJoCo Warp).
      # Warp needs its contact buffers sized up front; see _make_data.
      impl='jax',
      warp_naconmax=16,
      warp_njmax=64,
  )


class Creature(MjxEnv):
  """Forward-locomotion task for the quadruped creature."""

  def __init__(self, config: config_dict.ConfigDict | None = None,
               config_overrides: dict | None = None):
    super().__init__(config or default_config(), config_overrides)

    mj_model = mujoco.MjModel.from_xml_path(XML_PATH.as_posix())
    mj_model.opt.timestep = self.sim_dt
    # MJX wants few, cheap solver iterations.
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    mj_model.opt.iterations = 4
    mj_model.opt.ls_iterations = 8

    self._mj_model = mj_model
    self._mjx_model = mjx.put_model(mj_model, impl=self._config.impl)
    self._xml_path = XML_PATH.as_posix()

    self._torso_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    self._home_qpos = jp.array(mj_model.key_qpos[0])
    self._impl = getattr(self._mjx_model, "impl", None)

  # -- observation -------------------------------------------------------
  def _get_obs(self, data: mjx.Data, last_action: jax.Array) -> jax.Array:
    xmat = data.xmat[self._torso_id].reshape(3, 3)  # torso -> world
    return jp.concatenate([
        xmat.T @ _GRAVITY_DIR,   # gravity_local
        xmat.T @ data.qvel[0:3],  # linvel_local
        data.qvel[3:6],          # angvel_local
        data.qpos[7:],           # joint_pos
        data.qvel[6:],           # joint_vel
        last_action,
    ])

  def _make_data(self, qpos, qvel) -> mjx.Data:
    kwargs = {"qpos": qpos, "qvel": qvel}
    if self._impl is not None:
      kwargs["impl"] = self._impl
    if str(self._config.impl) == "warp":
      # Unlike the JAX backend, Warp will not size these itself. Too small and
      # the broadphase overflows and silently drops contacts.
      kwargs["naconmax"] = self._config.warp_naconmax
      kwargs["njmax"] = self._config.warp_njmax
    data = mjx_env.make_data(self._mj_model, **kwargs)
    return mjx.forward(self._mjx_model, data)

  # -- rollout -----------------------------------------------------------
  def reset(self, rng: jax.Array) -> State:
    rng, k_pos, k_vel = jax.random.split(rng, 3)
    noise = self._config.reset_noise
    qpos = self._home_qpos.at[7:].add(
        jax.random.uniform(k_pos, (ACTION_SIZE,), minval=-noise, maxval=noise)
    )
    qvel = jax.random.normal(k_vel, (self._mj_model.nv,)) * 0.05

    data = self._make_data(qpos, qvel)
    obs = self._get_obs(data, jp.zeros(ACTION_SIZE))

    metrics = {
        "reward/forward": jp.zeros(()),
        "reward/healthy": jp.zeros(()),
        "reward/upright": jp.zeros(()),
        "cost/ctrl": jp.zeros(()),
        "x_velocity": jp.zeros(()),
        "torso_height": jp.zeros(()),
    }
    info = {"rng": rng, "last_action": jp.zeros(ACTION_SIZE)}
    return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

  def step(self, state: State, action: jax.Array) -> State:
    cfg = self._config
    action = jp.clip(action, -1.0, 1.0)

    prev_x = state.data.qpos[0]
    data = mjx_env.step(self._mjx_model, state.data, action, self.n_substeps)

    x_velocity = (data.qpos[0] - prev_x) / self.dt
    height = data.qpos[2]
    xmat = data.xmat[self._torso_id].reshape(3, 3)
    uprightness = -(xmat.T @ _GRAVITY_DIR)[2]  # 1.0 when torso +Z == world +Z

    is_healthy = jp.where(
        (height > cfg.min_height)
        & (height < cfg.max_height)
        & (uprightness > cfg.min_uprightness),
        1.0, 0.0)

    # Pay for forward progress only up to the target speed.
    reward_forward = cfg.forward_weight * jp.minimum(x_velocity, cfg.target_speed)
    reward_healthy = cfg.healthy_reward * is_healthy
    reward_upright = cfg.upright_weight * jp.clip(uprightness, 0.0, 1.0)
    cost_ctrl = cfg.ctrl_cost * jp.sum(jp.square(action))
    cost_jvel = cfg.joint_vel_cost * jp.sum(jp.square(data.qvel[6:]))

    reward = reward_forward + reward_healthy + reward_upright - cost_ctrl - cost_jvel
    done = 1.0 - is_healthy

    obs = self._get_obs(data, action)
    state.metrics.update(**{
        "reward/forward": reward_forward,
        "reward/healthy": reward_healthy,
        "reward/upright": reward_upright,
        "cost/ctrl": cost_ctrl,
        "x_velocity": x_velocity,
        "torso_height": height,
    })
    state.info["last_action"] = action
    return state.replace(data=data, obs=obs, reward=reward, done=done)

  # -- required MjxEnv properties ---------------------------------------
  @property
  def xml_path(self) -> str:
    return self._xml_path

  @property
  def action_size(self) -> int:
    return ACTION_SIZE

  @property
  def mj_model(self) -> mujoco.MjModel:
    return self._mj_model

  @property
  def mjx_model(self) -> mjx.Model:
    return self._mjx_model
