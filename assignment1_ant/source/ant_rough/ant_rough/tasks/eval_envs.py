"""Self-made "unseen" evaluation environments (D3).

Each one mimics how the TA's unseen environment is described: Isaac-Ant-v0 unchanged except for the terrain shape
and terrain parameters (ground friction). So these start from the *baseline* ``AntEnvCfg`` (no robot friction
randomization, ground friction combine mode "average"), swap in a single-type terrain at a fixed difficulty, and
spawn every robot in row 0 so that the whole ~120 m run stays on the evaluated terrain.

Terrains are grouped by how they relate to our training distribution (``terrains.ANT_ROUGH_TERRAINS_CFG``):
  * ID  (in-distribution): terrain family and parameters seen in training
  * OOD (out-of-distribution): taller/wider/narrower blocks, extreme friction, or terrain types never trained on
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, TerrainImporterCfg

from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg

from .custom_terrains import MeshSlopedGridTerrainCfg
from .eval_names import EVAL_SPLITS


def make_eval_terrain(
    sub_terrain: SubTerrainBaseCfg | None = None,
    friction: float = 1.0,
    combine_mode: str = "average",
    num_cols: int = 4,
    sub_terrains: dict[str, SubTerrainBaseCfg] | None = None,
) -> TerrainImporterCfg:
    """Evaluation terrain: one terrain type (``sub_terrain``) or a per-tile random mix (``sub_terrains``)."""
    return TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            size=(8.0, 8.0),
            border_width=40.0,
            num_rows=20,
            num_cols=num_cols,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            difficulty_range=(1.0, 1.0),  # parameters below are given as exact values
            curriculum=False,
            use_cache=False,
            seed=0,  # identical terrain for every checkpoint
            sub_terrains=sub_terrains if sub_terrains is not None else {"eval": sub_terrain},
        ),
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode=combine_mode,
            restitution_combine_mode="average",
            static_friction=friction,
            dynamic_friction=friction,
            restitution=0.0,
        ),
        debug_vis=False,
    )


def apply_eval_terrain(cfg, **terrain_kwargs):
    """Swap only the terrain of an Isaac-Ant-v0-style (deploy) config, as the TA does with our task."""
    cfg.scene.terrain = make_eval_terrain(**terrain_kwargs)
    cfg.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
    return cfg


def make_eval_cfg(base_cfg_cls=AntEnvCfg, **terrain_kwargs) -> AntEnvCfg:
    """Isaac-Ant-v0 (or another deploy config) with an evaluation terrain."""
    return apply_eval_terrain(base_cfg_cls(), **terrain_kwargs)


def _grid(width: float, height: float) -> SubTerrainBaseCfg:
    return terrain_gen.MeshRandomGridTerrainCfg(
        proportion=1.0, grid_width=width, grid_height_range=(height, height), platform_width=1.5
    )


# name -> (split, factory kwargs, description)
EVAL_SPECS: dict[str, tuple[str, dict, str]] = {
    "Flat": ("ID", dict(sub_terrain=terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)), "flat, mu 1.0"),
    "Grid": ("ID", dict(sub_terrain=_grid(0.45, 0.08)), "blocks w0.45 h±0.08, mu 1.0"),
    "GridTall": ("OOD", dict(sub_terrain=_grid(0.45, 0.15)), "blocks w0.45 h±0.15 (taller than train 0.12)"),
    "GridWide": ("OOD", dict(sub_terrain=_grid(1.5, 0.10)), "blocks w1.5 h±0.10 (wider than train 0.95)"),
    "GridNarrow": ("OOD", dict(sub_terrain=_grid(0.15, 0.05)), "blocks w0.15 h±0.05 (narrower than train 0.3)"),
    "GridLowMu": ("OOD", dict(sub_terrain=_grid(0.45, 0.08), friction=0.2), "blocks w0.45 h±0.08, ground mu 0.2"),
    "GridHighMu": ("OOD", dict(sub_terrain=_grid(0.45, 0.08), friction=2.0), "blocks w0.45 h±0.08, ground mu 2.0"),
    "FlatIce": (
        "OOD",
        dict(sub_terrain=terrain_gen.MeshPlaneTerrainCfg(proportion=1.0), friction=0.2, combine_mode="min"),
        "flat, mu 0.2 with combine 'min' (effective 0.2)",
    ),
    "Stairs": (
        "OOD",
        dict(
            sub_terrain=terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=1.0, step_height_range=(0.06, 0.06), step_width=0.4, platform_width=2.0
            )
        ),
        "pyramid stairs, step 0.06 m (type never trained)",
    ),
    "Wave": (
        "OOD",
        dict(sub_terrain=terrain_gen.HfWaveTerrainCfg(proportion=1.0, amplitude_range=(0.1, 0.1), num_waves=4)),
        "sinusoidal waves, amplitude 0.1 m (type never trained)",
    ),
    # held out from every training mix, including V5 (which adds stairs)
    "Boxes": (
        "OOD",
        dict(
            sub_terrain=terrain_gen.MeshRepeatedBoxesTerrainCfg(
                proportion=1.0,
                platform_width=1.5,
                object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=60, height=0.08, size=(0.5, 0.5), max_yx_angle=30.0, degrees=True
                ),
                object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                    num_objects=60, height=0.08, size=(0.5, 0.5), max_yx_angle=30.0, degrees=True
                ),
            )
        ),
        "60 scattered boxes 0.5 m, h 0.08, tilted up to 30 deg (type never trained)",
    ),
    "Rails": (
        "OOD",
        dict(
            sub_terrain=terrain_gen.MeshRailsTerrainCfg(
                proportion=1.0, rail_thickness_range=(0.3, 0.3), rail_height_range=(0.08, 0.08), platform_width=2.0
            )
        ),
        "square rails h 0.08 around each tile center (type never trained)",
    ),
    # report-only held-out (never used for checkpoint selection, see scripts/select_checkpoint.py)
    "SlopedGrid": (
        "REPORT",
        dict(
            sub_terrain=MeshSlopedGridTerrainCfg(
                proportion=1.0, grid_width=0.45, grid_height_range=(0.06, 0.06), platform_width=1.5, slope=0.08
            )
        ),
        "blocks w0.45 h+-0.06 on a pyramid slope 0.08 (blocks + slope mixed; never trained)",
    ),
    "Pyramids": (
        "REPORT",
        dict(
            sub_terrain=terrain_gen.MeshRepeatedPyramidsTerrainCfg(
                proportion=1.0,
                platform_width=1.5,
                object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                    num_objects=50, height=0.10, radius=0.5, max_yx_angle=20.0, degrees=True
                ),
                object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                    num_objects=50, height=0.10, radius=0.5, max_yx_angle=20.0, degrees=True
                ),
            )
        ),
        "50 scattered pyramids r0.5 h0.10, tilted up to 20 deg (never trained)",
    ),
}

assert {k: v[0] for k, v in EVAL_SPECS.items()} == EVAL_SPLITS, "eval_names.EVAL_SPLITS out of sync with EVAL_SPECS"


##
# LOCKED final test environment (docs/log_1002.md): never used for training, selection or variant
# comparison; evaluated once with the official protocol (play_one_episode.py --seed 24 --num_envs 100).
##

TEST_SPECS: dict[str, dict] = {
    "UnseenMix": dict(
        sub_terrains={
            # blocks beyond the training range (train: w 0.3-0.95, h <= 0.12)
            "blocks_w06_h013": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.2, grid_width=0.6, grid_height_range=(0.13, 0.13), platform_width=1.5
            ),
            "pyramids": terrain_gen.MeshRepeatedPyramidsTerrainCfg(
                proportion=0.2,
                platform_width=1.5,
                object_params_start=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                    num_objects=50, height=0.10, radius=0.5, max_yx_angle=20.0, degrees=True
                ),
                object_params_end=terrain_gen.MeshRepeatedPyramidsTerrainCfg.ObjectCfg(
                    num_objects=50, height=0.10, radius=0.5, max_yx_angle=20.0, degrees=True
                ),
            ),
            "sloped_grid": MeshSlopedGridTerrainCfg(
                proportion=0.2, grid_width=0.45, grid_height_range=(0.06, 0.06), platform_width=1.5, slope=0.08
            ),
            # never evaluated before this test
            "cylinders": terrain_gen.MeshRepeatedCylindersTerrainCfg(
                proportion=0.2,
                platform_width=1.5,
                object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                    num_objects=40, height=0.08, radius=0.3, max_yx_angle=15.0, degrees=True
                ),
                object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                    num_objects=40, height=0.08, radius=0.3, max_yx_angle=15.0, degrees=True
                ),
            ),
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
        },
        friction=1.0,
        combine_mode="average",
    ),
}
