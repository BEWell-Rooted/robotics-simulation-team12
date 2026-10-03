"""Submission task ``Isaac-Ant-Team12-v0`` (team 12): the deploy configuration of our final model.

For the TA's unseen-terrain evaluation: only ``scene.terrain`` below should change. Everything else -- robot (foot
contact sensors, skin material), observations (60 Isaac-Ant-v0 terms + 8 foot-contact terms, 3-step history),
actions, and the ORIGINAL Isaac-Ant-v0 rewards and terminations -- must stay as is for the checkpoint to load
(the network is built from this task's agent config and loaded with ``strict=True``).

    ./isaaclab.sh -p <repo>/assignment1_ant/scripts/rsl_rl/play_one_episode.py --task Isaac-Ant-Team12-v0 \
        --seed 24 --num_envs 100 --checkpoint <repo>/assignment1_ant/checkpoints/final/model.pt
"""

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .robot_env_cfg import AntV1214DeployEnvCfg

# final model family (see docs/log_1004.md); its agent config is registered with the task in __init__.py
FinalDeployEnvCfg = AntV1214DeployEnvCfg


@configclass
class AntTeam12EnvCfg(FinalDeployEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # ------------------------------------------------------------------------------------------------------
        # TERRAIN: replace this block with the evaluation terrain (shape and terrain parameters such as friction).
        # Default = Isaac-Ant-v0's flat plane. For generated terrains also keep gpu_max_rigid_patch_count.
        # ------------------------------------------------------------------------------------------------------
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="average",
                restitution_combine_mode="average",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
