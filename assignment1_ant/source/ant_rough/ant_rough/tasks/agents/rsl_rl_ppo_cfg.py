"""RSL-RL PPO config for Ant rough-terrain tasks.

The actor-critic is kept *identical* to the Isaac-Ant-v0 baseline ([400, 200, 100], elu, no obs normalization,
symmetric critic on the 60-dim policy observation). ``OnPolicyRunner.load`` loads the state dict with
``strict=True`` using the network built from the *evaluating task's* config, so any structural change here would
make our checkpoint unloadable under the TA's unseen-environment task.
"""

from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg import AntPPORunnerCfg


@configclass
class AntRoughPPORunnerCfg(AntPPORunnerCfg):
    experiment_name = "ant_rough"
    max_iterations = 3000


##
# Round 2, axis 2: the policy may change because the TA loads our task config (docs/02_weekend_plan_1002.md)
##

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg, RslRlPpoActorCriticRecurrentCfg  # noqa: E402


@configclass
class AntV12PPORunnerCfg(AntRoughPPORunnerCfg):
    """Same MLP, observation normalization on (inputs now mix joint states, contact forces and a 3-step history)."""

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
    )


@configclass
class AntV13PPORunnerCfg(AntRoughPPORunnerCfg):
    """LSTM (256) in front of the same MLP heads; single-step observation, the recurrent state is the memory."""

    policy = RslRlPpoActorCriticRecurrentCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,
        critic_obs_normalization=True,
        actor_hidden_dims=[400, 200, 100],
        critic_hidden_dims=[400, 200, 100],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )


from isaaclab_rl.rsl_rl import RslRlSymmetryCfg  # noqa: E402

from ..symmetry import ant_mirror_augmentation  # noqa: E402


@configclass
class AntV8PPORunnerCfg(AntV12PPORunnerCfg):
    """V12 network + left-right mirror data augmentation in the PPO update (Mittal et al., ICRA 2024).

    Augmentation adds no parameters, so the checkpoint loads under the plain V12 agent (deploy = Deploy-V1214).
    """

    def __post_init__(self):
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=True, use_mirror_loss=False, data_augmentation_func=ant_mirror_augmentation
        )


@configclass
class AntV8MLPPORunnerCfg(AntV12PPORunnerCfg):
    """Ablation of V8: no augmented PPO samples, only a mirror loss pulling pi(mirror(o)) toward mirror(pi(o))."""

    def __post_init__(self):
        self.algorithm.symmetry_cfg = RslRlSymmetryCfg(
            use_data_augmentation=False, use_mirror_loss=True, mirror_loss_coeff=1.0,
            data_augmentation_func=ant_mirror_augmentation,
        )  # fmt: skip
