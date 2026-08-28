"""Biped-to-target task. Registers ``Isaac-Biped-Direct-v0`` with gymnasium.

Use with Isaac Lab's stock trainers via the external-callback hook:
    python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Biped-Direct-v0 --external_callback biped.register
"""

import sys

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Biped-Direct-v0",
    entry_point=f"{__name__}.biped_env:BipedEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.biped_env_cfg:BipedEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:BipedPPORunnerCfg",
    },
)


def register() -> list[str]:
    """External callback for train.py/play.py: importing this module already registered the task."""
    return sys.argv[1:]
