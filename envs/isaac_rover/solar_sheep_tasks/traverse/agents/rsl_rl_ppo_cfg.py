"""envs/isaac_rover/solar_sheep_tasks/traverse/agents/rsl_rl_ppo_cfg.py -- RSL-RL PPO for the traverse task.

Same config classes and layout as Isaac Lab's own tasks (core/velocity/config/anymal_d/agents). Numbers
from the design brief; gamma 0.995 is the MJX prototype's (25 s episodes, sparse +20 per target).
Logger: TensorBoard only (never wandb/neptune: nothing leaves the box). Actions are clipped to +-1 in
RslRlVecEnvWrapper, then scaled to +-pi rad/s by the action term.
"""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class TraverseRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "solar_sheep_traverse"
    logger = "tensorboard"
    # every env's first episode runs its full 25 s: the navigation curriculum demotes an env that
    # reached 0 targets, and a randomly shortened first episode would count as a failure (rl_cfg.py)
    init_at_random_ep_len = False
    clip_actions = 1.0
    obs_groups = {"actor": ["policy"], "critic": ["policy"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[128, 128, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.5),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[256, 256, 256],
        activation="elu",
        obs_normalization=True,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class TraverseFlatPPORunnerCfg(TraverseRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        # the flat task is pose tracking alone: it stops earlier. Its own experiment folder keeps a
        # play run without --checkpoint from picking up a rough-task checkpoint (and vice versa).
        self.max_iterations = 500
        self.experiment_name = "solar_sheep_traverse_flat"
