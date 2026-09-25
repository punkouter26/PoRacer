"""RSL-RL PPO runner cfg for the quad (Isaac Lab 3 / Newton task): QUAD_SPEC "PPO and budget" = the worm's.

actor & critic 3 x 128 ELU, initial action std 0.5, empirical observation normalisation (baked into the
ONNX at export), 4096 envs x 24 steps, 5 epochs, 4 mini-batches, lr 3e-4 at most (adaptive KL 0.01, see below),
gamma 0.99, lambda 0.95, clip 0.2, entropy 0.005, value coef 1.0, max grad norm 1.0, checkpoint every 50
iterations. The 30-minute wall-clock budget is enforced by ISAAC/scripts/train_quad_v3.py.

Round 4: the learning rate follows RSL-RL's adaptive-KL rule (desired KL 0.01, per mini-batch x/÷ 1.5,
floor 1e-5) with a ceiling of 3e-4 (agents/capped_ppo.py), because Isaac3Worm's constant 3e-4 collapsed.

Library defaults kept (not in the spec, listed so the MuJoCo side can match them):
    use_clipped_value_loss=True, advantages normalised over the whole rollout, Adam, one
    state-independent std per action ("scalar"), normaliser (x - mean) / (std + 0.01) updated on every
    rollout step, lengths of the first episodes randomised.

"""

from isaaclab.utils.configclass import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class QuadFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 100000  # the wall-clock budget stops the run
    save_interval = 50
    experiment_name = "quad_newton"  # ISAAC/logs/rsl_rl_v3/quad_newton
    empirical_normalization = True  # deprecated alias; the model cfgs below carry the real switch
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    # "8 values, clipped to [-1, 1]": clipped in RslRlVecEnvWrapper, before the env, so last_action (obs)
    # and the action-rate penalty both see the clipped action (WORM_SPEC item 5).
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
        class_name="quad_tasks_v3.agents.capped_ppo:CappedPPO",
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,  # start and ceiling
        schedule="adaptive",
        desired_kl=0.01,
        gamma=0.99,
        lam=0.95,
        clip_param=0.2,
        entropy_coef=0.005,
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        max_grad_norm=1.0,
    )
