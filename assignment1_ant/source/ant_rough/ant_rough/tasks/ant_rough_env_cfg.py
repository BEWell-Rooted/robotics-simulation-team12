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
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.classic.humanoid.mdp as mdp
from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg, EventCfg, RewardsCfg, TerminationsCfg

from .lane_terrain import LaneTerrainGeneratorCfg, terrain_levels_progress
from .mdp_extra import cat_orientation_termination, level_friction
from .terrains import ANT_ROUGH_TERRAINS_CFG, ANT_ROUGH_V5_SUB_TERRAINS


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


@configclass
class AntRoughStableRewardsCfg(RewardsCfg):
    """Baseline reward terms + two stability penalties (training only; evaluation uses the baseline terms).

    V1 learned a hopping gait that scores well on blocks but lands badly and flips on long flat stretches
    (Eval-Flat fall rate 89%, torso z mean 0.71 / min 0.32). These penalize the vertical bounce and the roll/pitch
    rates that precede a flip.
    """

    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.1)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.02)


@configclass
class AntRoughStableEnvCfg(AntRoughEnvCfg):
    """V4: V1 (same terrain mix and friction randomization) + anti-hop stability penalties."""

    rewards: AntRoughStableRewardsCfg = AntRoughStableRewardsCfg()


@configclass
class AntRoughV5EventCfg(EventCfg):
    """Wider friction than V1/V2: an evaluation-style ground (mu 1.0, static = dynamic) sat near the top of the
    old range, where make_consistent (dynamic <= static) made high dynamic friction rare."""

    robot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.2, 2.0),
            "dynamic_friction_range": (0.2, 2.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )


@configclass
class AntRoughV5EnvCfg(AntRoughCurriculumEnvCfg):
    """V5: V2's lane curriculum + per-tile random terrain types + stairs + more flat + wider friction."""

    events: AntRoughV5EventCfg = AntRoughV5EventCfg()

    def __post_init__(self):
        super().__post_init__()
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains = ANT_ROUGH_V5_SUB_TERRAINS
        gen.random_tile_types = True


##
# Round 2, axis 1: structure fixed, environment changes on top of V2 (docs/02_weekend_plan_1002.md)
##


@configclass
class AntRoughANoPromoEnvCfg(AntRoughCurriculumEnvCfg):
    """Ablation A-noPromo: V2's lanes, no promotion/demotion; robots start in uniformly random lanes and stay."""

    curriculum = None

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.max_init_terrain_level = None


@configclass
class AntRoughARandDiffEnvCfg(AntRoughANoPromoEnvCfg):
    """Ablation A-randDiff: lanes and column type order kept, but tile difficulty is uniform random (no lane level)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.random_difficulty = True


@configclass
class AntRoughV2SEnvCfg(AntRoughCurriculumEnvCfg):
    """V2S (not run): V2 + pyramid stairs. With V2's column-wise type order the stair columns land at the far end of
    every lane and are rarely reached -- superseded by V2RS."""

    def __post_init__(self):
        super().__post_init__()
        subs = dict(self.scene.terrain.terrain_generator.sub_terrains)
        subs["stairs"] = ANT_ROUGH_V5_SUB_TERRAINS["stairs"]
        subs["stairs_narrow"] = ANT_ROUGH_V5_SUB_TERRAINS["stairs_narrow"]
        self.scene.terrain.terrain_generator.sub_terrains = subs


@configclass
class AntRoughV2REnvCfg(AntRoughCurriculumEnvCfg):
    """V2R: V2 with per-tile random terrain types (lanes and their difficulty curriculum kept).

    V2's column-wise type order puts slopes (cols 15-16) and flat (17-19) at the far end of every lane, ~100 m from
    the spawn columns 0-3, so V2 barely trains on them (SlopedGrid 6-10 vs 50+ for patchwork-trained V1/V5).
    """

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator.random_tile_types = True


@configclass
class AntRoughV2RSEnvCfg(AntRoughV2REnvCfg):
    """V2RS: V2R + pyramid stairs (replaces V2S, whose stair columns would sit unreached at the lane ends)."""

    def __post_init__(self):
        super().__post_init__()
        subs = dict(self.scene.terrain.terrain_generator.sub_terrains)
        subs["stairs"] = ANT_ROUGH_V5_SUB_TERRAINS["stairs"]
        subs["stairs_narrow"] = ANT_ROUGH_V5_SUB_TERRAINS["stairs_narrow"]
        self.scene.terrain.terrain_generator.sub_terrains = subs


@configclass
class AntRoughV6cTerminationsCfg(TerminationsCfg):
    # Constraints-as-Terminations on the posture that precedes a flip; the absolute-z termination is kept as is
    cat_orientation = DoneTerm(func=cat_orientation_termination, time_out=False)


@configclass
class AntRoughV6cEnvCfg(AntRoughCurriculumEnvCfg):
    """V6': V2 + CaT stochastic termination on torso tilt (> 35 deg) and roll/pitch rate (> 3 rad/s)."""

    terminations: AntRoughV6cTerminationsCfg = AntRoughV6cTerminationsCfg()


@configclass
class AntRoughV10EventCfg(AntRoughEventCfg):
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(4.0, 8.0),
        params={"velocity_range": {"x": (-0.6, 0.6), "y": (-0.6, 0.6)}},
    )

    def __post_init__(self):
        # start from a tilted torso so that recovering from roll/pitch is part of the data (yaw kept at 0)
        self.reset_base.params = {
            "pose_range": {"roll": (-0.3, 0.3), "pitch": (-0.3, 0.3)},
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
        }


@configclass
class AntRoughV10RewardsCfg(RewardsCfg):
    # milder version of the V6' signal, as a penalty (training only)
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.5)


@configclass
class AntRoughV10EnvCfg(AntRoughCurriculumEnvCfg):
    """V10: V2 + horizontal pushes + tilted initial pose + mild torso-tilt penalty."""

    events: AntRoughV10EventCfg = AntRoughV10EventCfg()
    rewards: AntRoughV10RewardsCfg = AntRoughV10RewardsCfg()


@configclass
class AntRoughV11EventCfg(EventCfg):
    """No startup friction draw; friction is re-drawn at every reset from a range set by the env's lane level."""

    level_friction = EventTerm(func=level_friction, mode="reset", params={"w_min": 0.1, "w_max": 0.8})


@configclass
class AntRoughV11EnvCfg(AntRoughCurriculumEnvCfg):
    """V11: V2 with friction in the curriculum: robot mu ~ U(1 - w, 1 + w), w 0.1 on lane 0 -> 0.8 on the top lane."""

    events: AntRoughV11EventCfg = AntRoughV11EventCfg()
