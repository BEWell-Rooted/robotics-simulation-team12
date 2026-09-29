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

from .eval_names import EVAL_SPLITS


def make_eval_cfg(
    sub_terrain: SubTerrainBaseCfg,
    friction: float = 1.0,
    combine_mode: str = "average",
    num_cols: int = 4,
) -> AntEnvCfg:
    """Isaac-Ant-v0 with one terrain type at fixed difficulty and a given ground friction."""
    cfg = AntEnvCfg()
    cfg.scene.terrain = TerrainImporterCfg(
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
            sub_terrains={"eval": sub_terrain},
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
    cfg.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
    return cfg


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
}

assert {k: v[0] for k, v in EVAL_SPECS.items()} == EVAL_SPLITS, "eval_names.EVAL_SPLITS out of sync with EVAL_SPECS"
