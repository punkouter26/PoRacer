"""Isaac Lab tasks for Worm5 - method B of the two-trainer comparison (training/worm/WORM_SPEC.md).

Importing this package registers:

* ``Isaac-Worm5-Flat-v0``       training: 4096 envs, per-episode friction/mass/kp randomisation
* ``Isaac-Worm5-Flat-Play-v0``  evaluation/export: 100 envs, spec reset, no randomisation

Both use the RSL-RL PPO cfg in :mod:`worm_tasks.agents.rsl_rl_ppo_cfg`.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Worm5-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.worm_env_cfg:WormFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WormFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Worm5-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.worm_env_cfg:WormFlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WormFlatPPORunnerCfg",
    },
)
