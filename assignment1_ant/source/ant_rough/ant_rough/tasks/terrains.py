"""Terrain generator configs for Ant rough-terrain training.

Layout: the terrain generator places row ``i`` at x = (i + 0.5) * size[0], i.e. rows run along +x, which is the
Ant's walking direction (target at (1000, 0, 0)). The Ant covers ~120 m in a 16 s episode on flat ground, so the
track is long in x (20 rows x 8 m = 160 m) and robots spawn only in the first rows (``max_init_terrain_level``).
A wide flat border surrounds the patchwork so fast robots run out onto flat ground instead of off a ledge.

Heights are chosen with the *absolute-z* termination of Isaac-Ant-v0 (torso z < 0.31) in mind: the lowest point of
any sub-terrain stays within ~0.12 m below z = 0.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

TILE_SIZE = (8.0, 8.0)

ANT_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=TILE_SIZE,
    border_width=40.0,
    num_rows=20,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    curriculum=False,
    use_cache=False,
    sub_terrains={
        # square blocks of random height -- the terrain family shown in the assignment.
        # grid_width must not divide the 8 m tile evenly (random_grid_terrain needs a border > 0).
        "grid_fine": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.3, grid_height_range=(0.01, 0.07), platform_width=1.5
        ),
        "grid_mid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20, grid_width=0.45, grid_height_range=(0.02, 0.10), platform_width=1.5
        ),
        "grid_coarse": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.95, grid_height_range=(0.02, 0.12), platform_width=1.5
        ),
        # small-scale bumps
        "uniform_noise": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15, noise_range=(0.0, 0.06), noise_step=0.01, downsampled_scale=0.2
        ),
        # sparse rectangular obstacles
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.10,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.3, 1.0),
            obstacle_height_range=(0.03, 0.12),
            num_obstacles=40,
            platform_width=1.5,
        ),
        # gentle pyramid (up then down); no inverted pits, which would trip the absolute-z termination
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.10, slope_range=(0.0, 0.25), platform_width=2.0),
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.15),
    },
)
"""Mixed rough terrain for V1 (plain domain randomization): every tile draws a random type and difficulty."""


ANT_ROUGH_V5_SUB_TERRAINS = {
    "grid_fine": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=0.12, grid_width=0.3, grid_height_range=(0.01, 0.07), platform_width=1.5
    ),
    "grid_mid": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=0.15, grid_width=0.45, grid_height_range=(0.02, 0.10), platform_width=1.5
    ),
    "grid_coarse": terrain_gen.MeshRandomGridTerrainCfg(
        proportion=0.12, grid_width=0.95, grid_height_range=(0.02, 0.12), platform_width=1.5
    ),
    "uniform_noise": terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.10, noise_range=(0.0, 0.06), noise_step=0.01, downsampled_scale=0.2
    ),
    "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
        proportion=0.08,
        obstacle_height_mode="choice",
        obstacle_width_range=(0.3, 1.0),
        obstacle_height_range=(0.03, 0.12),
        num_obstacles=40,
        platform_width=1.5,
    ),
    "slope": terrain_gen.HfPyramidSlopedTerrainCfg(proportion=0.08, slope_range=(0.0, 0.25), platform_width=2.0),
    # new in V5: stairs (up toward the tile center, then down), two tread widths
    "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
        proportion=0.08, step_height_range=(0.02, 0.08), step_width=0.4, platform_width=2.0
    ),
    "stairs_narrow": terrain_gen.MeshPyramidStairsTerrainCfg(
        proportion=0.07, step_height_range=(0.02, 0.07), step_width=0.3, platform_width=2.0
    ),
    # more flat than V1/V2 (0.15 -> 0.20): long flat mesh stretches were the main failure mode
    "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.20),
}
"""V5 mix: V1/V2 terrain families + pyramid stairs, more flat. Tile types are drawn per tile (see lane_terrain)."""
