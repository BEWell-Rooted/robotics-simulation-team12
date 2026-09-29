"""Isaac-Ant-v0 on randomized rough terrain with randomized friction.

Only the *environment* differs from Isaac-Ant-v0: terrain, contact friction and physics buffers. Observations
(60-dim, including the absolute torso height), actions (8), rewards and terminations (absolute torso height < 0.31)
are inherited unchanged, because the TA evaluates our checkpoint in an Isaac-Ant-v0 variant where only terrain and
terrain parameters change. Training under the same absolute-z semantics lets the policy learn to keep its torso
high enough on uneven ground instead of learning a relative-height signal it will not get at evaluation time.
"""

import isaaclab.sim as sim_utils
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, EventCfg

from .lane_terrain import LaneTerrainGeneratorCfg, terrain_levels_progress
from .terrains import ANT_ROUGH_TERRAINS_CFG


@configclass
class AntRoughEventCfg(EventCfg):
    """Baseline reset events plus per-env foot/body friction randomization."""

    # The ground uses friction_combine_mode="multiply" with friction 1.0, which takes precedence over the robot's
    # "average" mode in PhysX, so the effective contact friction equals the robot material sampled here.
    # This covers the TA's likely settings: ground mu in [0.1, 2.0] under "average" with robot mu 1.0 gives an
    # effective mu of [0.55, 1.5]; under "min"/"multiply" it gives the ground mu itself.
    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.5),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class AntRoughEnvCfg(AntEnvCfg):
    """V1: plain domain randomization over a mixed rough-terrain patchwork and contact friction."""

    events: AntRoughEventCfg = AntRoughEventCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="generator",
            terrain_generator=ANT_ROUGH_TERRAINS_CFG.replace(),  # own copy; PLAY cfg mutates it
            # spawn in the first 8 of 20 rows so a ~120 m run stays on the patchwork / flat border
            max_init_terrain_level=7,
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            debug_vis=False,
        )
        # more contact patches on meshes (value used by Isaac Lab's rough-terrain locomotion tasks)
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15


@configclass
class AntRoughEnvCfg_PLAY(AntRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        # smaller terrain for quick visual checks
        self.scene.terrain.terrain_generator.num_rows = 20
        self.scene.terrain.terrain_generator.num_cols = 4


@configclass
class AntRoughCurriculumCfg:
    terrain_levels = CurrTerm(func=terrain_levels_progress, params={"up_distance": 60.0, "down_distance": 20.0})


@configclass
class AntRoughCurriculumEnvCfg(AntRoughEnvCfg):
    """V2: same terrain mix and friction randomization as V1, arranged as difficulty lanes with a progress curriculum.

    10 lanes (levels) of 20 x 8 m segments each; a robot starts in lanes 0-2 and moves one lane up after running
    > 60 m toward +x in an episode, one lane down after < 20 m.
    """

    curriculum: AntRoughCurriculumCfg = AntRoughCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        gen = ANT_ROUGH_TERRAINS_CFG
        self.scene.terrain.terrain_generator = LaneTerrainGeneratorCfg(
            size=gen.size,
            border_width=gen.border_width,
            num_rows=10,
            num_cols=20,
            num_spawn_cols=4,
            horizontal_scale=gen.horizontal_scale,
            vertical_scale=gen.vertical_scale,
            slope_threshold=gen.slope_threshold,
            difficulty_range=gen.difficulty_range,
            use_cache=False,
            sub_terrains=gen.sub_terrains,
        )
        self.scene.terrain.max_init_terrain_level = 2
