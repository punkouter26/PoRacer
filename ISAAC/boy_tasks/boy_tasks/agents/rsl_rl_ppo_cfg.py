"""PPO runner configuration for the Boy target-chasing task.

Mirrors the H1 flat config the IsaacH1 port was trained with (3 x 128 ELU actor), except
that observation normalisation is ON. The export pipeline bakes the fitted normaliser into
the ONNX graph, so Unity feeds raw observations exactly as it does for the H1.
"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

try:  # Isaac Lab >= 2.2 / rsl-rl-lib >= 3: model configs
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

    _NEW_STYLE = True
except ImportError:  # older: RslRlPpoActorCriticCfg
    from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

    _NEW_STYLE = False


@configclass
class BoyChaseFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 50
    experiment_name = "boy_chase_flat"
    empirical_normalization = True  # honoured by the old-style runner; new style uses actor.obs_normalization

    if _NEW_STYLE:
        actor = RslRlMLPModelCfg(
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=True,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
        critic = RslRlMLPModelCfg(
            hidden_dims=[128, 128, 128],
            activation="elu",
            obs_normalization=True,
        )
    else:
        policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[128, 128, 128],
            critic_hidden_dims=[128, 128, 128],
            activation="elu",
        )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
