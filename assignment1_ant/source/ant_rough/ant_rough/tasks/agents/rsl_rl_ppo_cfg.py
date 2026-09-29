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
