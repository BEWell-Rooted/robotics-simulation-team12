"""Custom sub-terrains used only for evaluation."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains import MeshRandomGridTerrainCfg
from isaaclab.terrains.trimesh.mesh_terrains import random_grid_terrain
from isaaclab.utils import configclass


def sloped_random_grid_terrain(difficulty: float, cfg: MeshSlopedGridTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Random-height blocks laid on a pyramid slope (peak at the tile center, 0 at the tile edge).

    Every vertex of the ``random_grid_terrain`` meshes is raised by ``slope * (half - max(|dx|, |dy|))``, so block
    tops follow the slope while keeping their random offsets -- a "slope + blocks" mix none of the training mixes
    contains (they have blocks and pyramid slopes only as separate tiles).
    """
    meshes, origin = random_grid_terrain(difficulty, cfg)
    half = 0.5 * cfg.size[0]
    out = []
    for m in meshes:
        v = np.asarray(m.vertices).copy()
        dist = np.maximum(np.abs(v[:, 0] - half), np.abs(v[:, 1] - half))
        v[:, 2] += cfg.slope * np.clip(half - dist, 0.0, None)
        out.append(trimesh.Trimesh(vertices=v, faces=m.faces, process=False))
    origin = np.asarray(origin, dtype=float).copy()
    origin[2] += cfg.slope * half
    return out, origin


@configclass
class MeshSlopedGridTerrainCfg(MeshRandomGridTerrainCfg):
    function = sloped_random_grid_terrain
    slope: float = 0.08
    """Rise per meter toward the tile center (0.08 -> +0.32 m at the center of an 8 m tile)."""
