"""Isaac Lab tasks for the PoRacer Boy character.

Importing this package registers:

* ``Isaac-Chase-Flat-Boy-v0``       training: 4096 envs, domain randomisation, pushes
* ``Isaac-Chase-Flat-Boy-Play-v0``  evaluation/export: 64 envs, no noise, no pushes

Both use :class:`boy_tasks.chase_env_cfg.BoyChaseFlatEnvCfg` and the RSL-RL PPO runner
config in :mod:`boy_tasks.agents.rsl_rl_ppo_cfg`.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Chase-Flat-Boy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chase_env_cfg:BoyChaseFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoyChaseFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Chase-Flat-Boy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.chase_env_cfg:BoyChaseFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BoyChaseFlatPPORunnerCfg",
    },
)
