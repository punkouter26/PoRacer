"""RSL-RL PPO runner cfg for Worm5, exactly WORM_SPEC.md "PPO (both)".

actor & critic 3 x 128 ELU, initial action std 0.5, empirical observation normalisation (baked
into the ONNX at export), 4096 envs x 24 steps, 5 epochs, 4 mini-batches, lr 3e-4 constant
(schedule "fixed"), gamma 0.99, lambda 0.95, clip 0.2, entropy 0.005, value coef 1.0, max grad
norm 1.0. The wall-clock budget (30 min) is enforced by ISAAC/scripts/train_worm.py, so
max_iterations is only an upper bound.

Library defaults kept (not in the spec - listed so the MuJoCo side can match them):
    use_clipped_value_loss=True, advantage normalisation over the whole rollout
    (normalize_advantage_per_mini_batch=False), optimizer Adam, std parameterisation "scalar",
    normaliser = (x - mean) / (std + 0.01), updated on every rollout step (never frozen).
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class WormFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000  # the wall-clock budget stops the run
    save_interval = 50
    experiment_name = "worm5"
    empirical_normalization = True  # deprecated alias; the model cfgs below carry the real switch
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    # Spec: "Eight values, clipped to [-1, 1]". Clipped in RslRlVecEnvWrapper, before the env,
    # so last_action (obs) and the action-rate penalty both see the clipped action.
    clip_actions = 1.0

    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=True,
    )

    algorithm = RslRlPpoAlgorithmCfg(
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,
        schedule="fixed",
        desired_kl=None,  # unused with a fixed schedule
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        entropy_coef=0.005,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        max_grad_norm=1.0,
    )
