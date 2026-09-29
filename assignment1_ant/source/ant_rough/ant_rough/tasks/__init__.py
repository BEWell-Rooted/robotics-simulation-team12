"""Gym registrations for Ant rough-terrain tasks.

All tasks share the Isaac-Ant-v0 MDP interface (obs 60, action 8) and the baseline network, so every checkpoint
trained here also loads with ``--task Isaac-Ant-v0``.
"""

import gymnasium as gym

from . import agents

gym.register(
    id="Isaac-Ant-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Rough-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
    },
)
