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

gym.register(
    id="Isaac-Ant-Rough-Curriculum-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughCurriculumEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Rough-Stable-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughStableEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Ant-Rough-V5-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ant_rough_env_cfg:AntRoughV5EnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:AntRoughPPORunnerCfg",
    },
)

##
# Self-made unseen evaluation environments: Isaac-Ant-Eval-<Name>-v0 (see eval_envs.py)
##

from .eval_envs import EVAL_SPECS, make_eval_cfg

for _name, (_split, _kwargs, _desc) in EVAL_SPECS.items():

    # a plain function (not functools.partial): the cfg loader calls inspect.getfile() on callable entry points
    def _make_cfg(_kwargs=_kwargs):
        return make_eval_cfg(**_kwargs)

    gym.register(
        id=f"Isaac-Ant-Eval-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _make_cfg,
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )


##
# Diagnosis-only: identical block strip as box primitives vs one triangle mesh (see diag_envs.py)
##

from .diag_envs import make_box_mesh_cfg, make_box_mesh_flat_cfg, make_box_prim_cfg, make_box_prim_flat_cfg

for _name, _factory in (
    ("BoxPrim", make_box_prim_cfg),
    ("BoxMesh", make_box_mesh_cfg),
    ("BoxPrimFlat", make_box_prim_flat_cfg),
    ("BoxMeshFlat", make_box_mesh_flat_cfg),
):
    gym.register(
        id=f"Isaac-Ant-Diag-{_name}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": _factory,
            "rsl_rl_cfg_entry_point": "isaaclab_tasks.manager_based.classic.ant.agents.rsl_rl_ppo_cfg:AntPPORunnerCfg",
        },
    )
