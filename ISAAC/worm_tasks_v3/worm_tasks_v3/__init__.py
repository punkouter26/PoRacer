"""Isaac Lab 3 tasks for Worm5 on the Newton backend with the MuJoCo-Warp solver (third trainer of
training/worm/WORM_SPEC.md; the 2.3 / PhysX task lives in ISAAC/worm_tasks and is untouched).

Importing this package registers:

* ``Isaac-Worm5-Flat-Newton-v0``       training: 4096 envs, per-episode friction/mass/kp randomisation
* ``Isaac-Worm5-Flat-Newton-Play-v0``  evaluation/export: 100 envs, spec reset, no randomisation

Both use the RSL-RL PPO cfg in :mod:`worm_tasks_v3.agents.rsl_rl_ppo_cfg` (identical to 2.3's).
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Worm5-Flat-Newton-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.worm_env_cfg:WormFlatNewtonEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WormFlatPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Worm5-Flat-Newton-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.worm_env_cfg:WormFlatNewtonEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:WormFlatPPORunnerCfg",
    },
)
