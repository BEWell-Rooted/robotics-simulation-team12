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
# Round 2 (docs/02_weekend_plan_1002.md). Axis 1: environment changes on V2, baseline network.
# Axis 2: robot-side changes; each has a TRAIN task and a DEPLOY task (Isaac-Ant-v0 + the robot change) whose
# terrain the evaluation / the TA swaps.
##

_AG = f"{agents.__name__}.rsl_rl_ppo_cfg"
_ROUND2 = {
    # id suffix: (env cfg "module:Class", agent cfg class)
    "Rough-ANoPromo": ("ant_rough_env_cfg:AntRoughANoPromoEnvCfg", "AntRoughPPORunnerCfg"),
    "Rough-ARandDiff": ("ant_rough_env_cfg:AntRoughARandDiffEnvCfg", "AntRoughPPORunnerCfg"),
    "Rough-V2S": ("ant_rough_env_cfg:AntRoughV2SEnvCfg", "AntRoughPPORunnerCfg"),
    "Rough-V2R": ("ant_rough_env_cfg:AntRoughV2REnvCfg", "AntRoughPPORunnerCfg"),
    "Rough-V2RS": ("ant_rough_env_cfg:AntRoughV2RSEnvCfg", "AntRoughPPORunnerCfg"),
    "Rough-V6c": ("ant_rough_env_cfg:AntRoughV6cEnvCfg", "AntRoughPPORunnerCfg"),
    "Rough-V10": ("ant_rough_env_cfg:AntRoughV10EnvCfg", "AntRoughPPORunnerCfg"),
    "Rough-V11": ("ant_rough_env_cfg:AntRoughV11EnvCfg", "AntRoughPPORunnerCfg"),
    "Rough-V12": ("robot_env_cfg:AntV12TrainEnvCfg", "AntV12PPORunnerCfg"),
    "Rough-V12R": ("robot_env_cfg:AntV12RTrainEnvCfg", "AntV12PPORunnerCfg"),
    "Rough-V1214": ("robot_env_cfg:AntV1214TrainEnvCfg", "AntV12PPORunnerCfg"),
    "Rough-V1214R": ("robot_env_cfg:AntV1214RTrainEnvCfg", "AntV12PPORunnerCfg"),
    "Deploy-V1214": ("robot_env_cfg:AntV1214DeployEnvCfg", "AntV12PPORunnerCfg"),
    "Rough-V1214R-Mu02": ("robot_env_cfg:AntV1214RMu02TrainEnvCfg", "AntV12PPORunnerCfg"),
    "Deploy-V1214-Mu02": ("robot_env_cfg:AntV1214Mu02DeployEnvCfg", "AntV12PPORunnerCfg"),
    "Rough-V1214R-Mu03": ("robot_env_cfg:AntV1214RMu03TrainEnvCfg", "AntV12PPORunnerCfg"),
    "Rough-V1214R-Mu06": ("robot_env_cfg:AntV1214RMu06TrainEnvCfg", "AntV12PPORunnerCfg"),
    "Deploy-V1214-Mu03": ("robot_env_cfg:AntV1214Mu03DeployEnvCfg", "AntV12PPORunnerCfg"),
    "Deploy-V1214-Mu06": ("robot_env_cfg:AntV1214Mu06DeployEnvCfg", "AntV12PPORunnerCfg"),
    "Rough-V13": ("robot_env_cfg:AntV13TrainEnvCfg", "AntV13PPORunnerCfg"),
    "Rough-V14": ("robot_env_cfg:AntV14TrainEnvCfg", "AntRoughPPORunnerCfg"),
    "Deploy-V12": ("robot_env_cfg:AntV12DeployEnvCfg", "AntV12PPORunnerCfg"),
    "Deploy-V13": ("robot_env_cfg:AntV13DeployEnvCfg", "AntV13PPORunnerCfg"),
    "Deploy-V14": ("robot_env_cfg:AntV14DeployEnvCfg", "AntRoughPPORunnerCfg"),
}
for _suffix, (_env, _agent) in _ROUND2.items():
    gym.register(
        id=f"Isaac-Ant-{_suffix}-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={"env_cfg_entry_point": f"{__name__}.{_env}", "rsl_rl_cfg_entry_point": f"{_AG}:{_agent}"},
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


##
# LOCKED final test environment, one task per deploy config (Isaac-Ant-v0 and the axis-2 deploy configs)
##

from .eval_envs import TEST_SPECS, apply_eval_terrain  # noqa: E402

_DEPLOYS = {"": "isaaclab_tasks.manager_based.classic.ant.ant_env_cfg:AntEnvCfg"}
_DEPLOYS.update({f"-{v}": f"{__name__}.robot_env_cfg:Ant{v}DeployEnvCfg" for v in ("V12", "V13", "V14", "V1214")})
_DEPLOY_AGENTS = {"": "AntRoughPPORunnerCfg", "-V12": "AntV12PPORunnerCfg", "-V13": "AntV13PPORunnerCfg",
                  "-V14": "AntRoughPPORunnerCfg", "-V1214": "AntV12PPORunnerCfg"}  # fmt: skip

for _tname, _tkw in TEST_SPECS.items():
    for _dsuffix, _dpath in _DEPLOYS.items():

        def _make_test_cfg(_dpath=_dpath, _tkw=_tkw):
            import importlib

            mod, cls = _dpath.split(":")
            return apply_eval_terrain(getattr(importlib.import_module(mod), cls)(), **_tkw)

        gym.register(
            id=f"Isaac-Ant-Test-{_tname}{_dsuffix}-v0",
            entry_point="isaaclab.envs:ManagerBasedRLEnv",
            disable_env_checker=True,
            kwargs={"env_cfg_entry_point": _make_test_cfg, "rsl_rl_cfg_entry_point": f"{_AG}:{_DEPLOY_AGENTS[_dsuffix]}"},
        )
