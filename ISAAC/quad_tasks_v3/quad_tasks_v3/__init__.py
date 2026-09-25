"""Isaac Lab 3 tasks for the quad pilot (training/quad/QUAD_SPEC.md) on the Newton backend with the
MuJoCo-Warp solver. The MuJoCo Warp trainer lives in training/quad/mujoco.

Importing this package registers:

* ``Isaac-Quad-Flat-Newton-v0``       training: 4096 envs, per-reset friction/mass/kp randomisation, pushes
* ``Isaac-Quad-Flat-Newton-Play-v0``  evaluation/export: 100 envs, spec reset, no randomisation, no pushes

Both use the RSL-RL PPO cfg in :mod:`quad_tasks_v3.agents.rsl_rl_ppo_cfg` (QUAD_SPEC "PPO and budget").
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Quad-Flat-Newton-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quad_env_cfg:QuadFlatNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Quad-Flat-Newton-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.quad_env_cfg:QuadFlatNewtonEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:QuadFlatPPORunnerCfg",
    },
)
