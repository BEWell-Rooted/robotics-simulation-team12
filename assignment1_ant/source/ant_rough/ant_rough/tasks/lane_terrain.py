"""Lane terrain for the V2 curriculum: difficulty varies across y, each lane is a long constant-difficulty track in x.

Isaac Lab's ``TerrainGenerator`` puts row ``i`` (= difficulty level in curriculum mode) at x = (i + 0.5) * size[0].
For the Ant, which always walks toward +x and covers ~120 m per episode, that would make every episode sweep
through all difficulty levels and run off the far end. ``LaneTerrainGenerator`` transposes the layout:

    level (row)  -> y   : lane index, one difficulty per lane
    column       -> x   : segments along the lane (sub-terrain type varies per segment)

and restricts spawning to the first ``num_spawn_cols`` segments of every lane so the whole run stays on the lane.
The terrain importer's standard level bookkeeping (``terrain_levels``, ``update_env_origins``) is reused as is.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh

from isaaclab.terrains import TerrainGenerator, TerrainGeneratorCfg
from isaaclab.terrains.trimesh.utils import make_border
from isaaclab.utils import configclass


class LaneTerrainGenerator(TerrainGenerator):
    cfg: LaneTerrainGeneratorCfg

    def __init__(self, cfg: LaneTerrainGeneratorCfg, device: str = "cpu"):
        if not cfg.curriculum:
            raise ValueError("LaneTerrainGenerator is meant for curriculum mode (levels = lanes).")
        super().__init__(cfg, device)
        # the base class centered the terrain assuming rows along x; re-center for rows along y
        rows_x = cfg.size[0] * cfg.num_rows * 0.5
        cols_y = cfg.size[1] * cfg.num_cols * 0.5
        lane_x = cfg.size[0] * cfg.num_cols * 0.5
        lane_y = cfg.size[1] * cfg.num_rows * 0.5
        shift = np.array([rows_x - lane_x, cols_y - lane_y, 0.0])
        transform = np.eye(4)
        transform[:3, -1] = shift
        self.terrain_mesh.apply_transform(transform)
        self.terrain_origins += shift
        # spawn only in the first segments of each lane: column c spawns at column (c mod num_spawn_cols)
        spawn_cols = np.arange(cfg.num_cols) % cfg.num_spawn_cols
        self.terrain_origins = self.terrain_origins[:, spawn_cols].copy()

    def _generate_curriculum_terrains(self):
        if not self.cfg.random_tile_types:
            return super()._generate_curriculum_terrains()
        # difficulty per row (lane) as in the base class, but the sub-terrain type is drawn per tile by proportion.
        # The base class assigns one type per column in proportion order, which along an x-lane puts the same type
        # sequence in every lane (e.g. all flat tiles at the far end, rarely reached from the spawn segments).
        cfgs = list(self.cfg.sub_terrains.values())
        proportions = np.array([c.proportion for c in cfgs], dtype=float)
        proportions /= proportions.sum()
        lower, upper = self.cfg.difficulty_range
        for sub_col in range(self.cfg.num_cols):
            for sub_row in range(self.cfg.num_rows):
                difficulty = lower + (upper - lower) * (sub_row + self.np_rng.uniform()) / self.cfg.num_rows
                sub_cfg = cfgs[self.np_rng.choice(len(cfgs), p=proportions)]
                mesh, origin = self._get_terrain_mesh(difficulty, sub_cfg)
                self._add_sub_terrain(mesh, origin, sub_row, sub_col, sub_cfg)

    def _add_sub_terrain(self, mesh: trimesh.Trimesh, origin: np.ndarray, row: int, col: int, sub_terrain_cfg):
        # same as the base class with the placement axes swapped (column along x, row/level along y)
        transform = np.eye(4)
        transform[0:2, -1] = (col + 0.5) * self.cfg.size[0], (row + 0.5) * self.cfg.size[1]
        mesh.apply_transform(transform)
        self.terrain_meshes.append(mesh)
        self.terrain_origins[row, col] = origin + transform[:3, -1]

    def _add_terrain_border(self):
        inner_size = (self.cfg.num_cols * self.cfg.size[0], self.cfg.num_rows * self.cfg.size[1])
        border_size = (inner_size[0] + 2 * self.cfg.border_width, inner_size[1] + 2 * self.cfg.border_width)
        border_center = (inner_size[0] / 2, inner_size[1] / 2, -self.cfg.border_height / 2)
        border = trimesh.util.concatenate(
            make_border(border_size, inner_size, height=abs(self.cfg.border_height), position=border_center)
        )
        border.update_faces(~(np.asarray(border.triangles)[:, :, 2] < -0.1).any(1))
        self.terrain_meshes.append(border)


@configclass
class LaneTerrainGeneratorCfg(TerrainGeneratorCfg):
    class_type: type = LaneTerrainGenerator
    curriculum: bool = True
    num_spawn_cols: int = 4
    """Robots spawn in the first ``num_spawn_cols`` segments (x) of their lane."""
    random_tile_types: bool = False
    """Draw the sub-terrain type per tile (True) instead of one type per column in proportion order (base class)."""


def terrain_levels_progress(
    env, env_ids: torch.Tensor, up_distance: float = 60.0, down_distance: float = 20.0
) -> torch.Tensor:
    """Curriculum term: move to a harder lane after a long run toward +x, to an easier one after a short run.

    Called on reset before the robot state is reset, so ``root_pos_w`` is still the terminal position.
    For reference, the flat-ground baseline covers ~135 m in a 16 s episode.
    """
    terrain = env.scene.terrain
    asset = env.scene["robot"]
    progress_x = asset.data.root_pos_w[env_ids, 0] - env.scene.env_origins[env_ids, 0]
    move_up = progress_x > up_distance
    move_down = (progress_x < down_distance) & ~move_up
    terrain.update_env_origins(env_ids, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
