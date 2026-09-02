"""MDP terms for the Boy target-chasing task.

Everything generic (joint penalties, feet air time, resets, pushes, terminations) is
re-exported from Isaac Lab's velocity locomotion MDP so the config reads like the H1 one.
The chase-specific pieces live in the sibling modules.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab_tasks.manager_based.locomotion.velocity.mdp import *  # noqa: F401, F403

from .commands import TargetPositionCommand, TargetPositionCommandCfg  # noqa: F401
from .observations import target_pos_b  # noqa: F401
from .rewards import heading_to_target, target_progress, target_speed_exp, targets_reached  # noqa: F401
