"""Diagnosis-only environments (not part of the evaluation suite; see docs/01_improvement_plan_0930.md, section 2).

Isaac-Ant-v0 with a random-height block strip whose cells are either USD box primitives or the identical cells as
one triangle mesh, so "primitive vs mesh" contact is the only difference between the two.
"""

import isaaclab.sim as sim_utils

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg

from .box_terrain import BoxGridTerrainImporterCfg


def make_box_grid_cfg(as_mesh: bool, grid_width: float = 0.45, grid_height: float = 0.08) -> AntEnvCfg:
    cfg = AntEnvCfg()
    cfg.scene.terrain = BoxGridTerrainImporterCfg(
        prim_path="/World/ground",
        as_mesh=as_mesh,
        grid_width=grid_width,
        grid_height=grid_height,
        max_init_terrain_level=0,
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
    cfg.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
    # ~20k static box shapes overflow the default broad-phase pair buffers (PhysX then silently misses contacts and
    # robots fall through the cells). With 256 robots PhysX asked for 5.3M found-lost pairs and 69M aggregate pairs,
    # and raising the aggregate buffer to 2**27 made GPU kernels fail to launch -> evaluate with <= 64 envs, where
    # the aggregate demand (~17M) fits the default 2**25, and raise only the (cheap) non-aggregate buffers.
    cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**24
    cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**24
    return cfg


def make_box_prim_cfg() -> AntEnvCfg:
    return make_box_grid_cfg(as_mesh=False)


def make_box_mesh_cfg() -> AntEnvCfg:
    return make_box_grid_cfg(as_mesh=True)


def make_box_prim_flat_cfg() -> AntEnvCfg:
    """Flat floor made of box primitives (2 m cells, all tops at z = 0): plane vs box-primitive vs mesh flat."""
    return make_box_grid_cfg(as_mesh=False, grid_width=2.0, grid_height=0.0)


def make_box_mesh_flat_cfg() -> AntEnvCfg:
    return make_box_grid_cfg(as_mesh=True, grid_width=2.0, grid_height=0.0)
