"""MDP terms for the quad (Isaac Lab 3, Newton / MuJoCo-Warp).

Generic Isaac Lab terms are re-exported so the env cfg reads like a stock task. The torso-frame
observations, the heading / lateral / effort / action-rate rewards, the health guard and the lean
per-world randomisers are the worm port's (ISAAC/worm_tasks_v3), reused unchanged: QUAD_SPEC.md defines
them exactly as WORM_SPEC.md with "segment 2" read as "torso". The quad-only terms live here.
"""

from isaaclab.envs.mdp import *  # noqa: F401, F403

from worm_tasks_v3.mdp.events import (  # noqa: F401
    randomize_friction_scale,
    randomize_kp_lean,
    randomize_segment_mass_lean as randomize_body_mass_lean,
)
from worm_tasks_v3.mdp.observations import (  # noqa: F401
    goal_dir_b,
    joint_vel_ordered,
    ref_ang_vel_b,
    ref_gravity_b,
    ref_lin_vel_b,
)
from worm_tasks_v3.mdp.rewards import action_rate_mean, effort_mean, heading, lateral_drift  # noqa: F401
from worm_tasks_v3.mdp.terminations import diverged_mask, sim_diverged  # noqa: F401

from .events import push_horizontal  # noqa: F401
from .observations import joint_pos_rel_ordered, target_speed_obs  # noqa: F401
from .rewards import alive, feet_air_time, flight, foot_slip, progress, speed_tracking, upright, vertical_bounce  # noqa: F401
from .terminations import fallen, fallen_mask, floor_touch_mask, time_out_healthy_standing, touch_this_step  # noqa: F401
