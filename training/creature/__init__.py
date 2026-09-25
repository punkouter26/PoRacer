"""Reusable MuJoCo Warp creature-training template, lifted from the Worm5 trainer
(training/worm/mujoco), which stays as it was.

    env.py       CreatureEnv: MJCF loading, batched MJWarp worlds, CUDA-graph physics,
                 the WORM_SPEC step order (item 13), spec resets, domain randomisation,
                 optional pushes, the simulator-health guard, episode bookkeeping.
                 A creature subclasses it and supplies compute_observation(),
                 compute_reward() and (optionally) task_terminated().
    ppo.py       ActorCritic + RSL-RL-style PPO (WORM_SPEC "PPO" + item 11), TensorBoard
                 first (rule C), run pruning, checkpoints every N iterations, metrics log.
    export.py    ONNX export (obs -> actions, normaliser baked in), parity check, loading.
    evaluate.py  Deterministic N-episode evaluation of ANY ONNX with the creature's
                 interface, with a pluggable plausibility (rule I) block, JSON report.

A creature is: an MJCF, a CreatureConfig, a CreatureEnv subclass, a plausibility
plug-in and three thin scripts (train / export / eval). See training/quad/mujoco.
"""
