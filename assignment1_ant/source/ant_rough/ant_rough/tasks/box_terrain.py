"""Random-height block terrain built from USD box primitives instead of one triangle mesh (diagnosis only).

All Isaac Lab generator terrains are imported as a single triangle mesh. If the TA builds the unseen block terrain
from box prims (e.g. cubes placed in Isaac Sim), contacts are box-vs-capsule primitives rather than
mesh-vs-capsule, which can behave differently. This importer reproduces ``MeshRandomGridTerrainCfg`` geometry
(square cells, top heights ~ U(-h, +h), a raised spawn platform) with one static ``UsdGeom.Cube`` collider per cell,
over a finite strip in front of the spawn points, on top of an analytic ground plane lowered below the lowest cell.

Env origins follow the evaluation layout (``eval_envs.make_eval_cfg``): ``num_spawn`` spawn platforms at
x = spawn_x, spread in y; every robot runs toward +x across the strip.
"""

from __future__ import annotations

import numpy as np
import torch
import trimesh
from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, UsdGeom, UsdPhysics, UsdShade

import isaaclab.sim as sim_utils
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.utils import configclass


class BoxGridTerrainImporter(TerrainImporter):
    cfg: BoxGridTerrainImporterCfg

    def __init__(self, cfg: BoxGridTerrainImporterCfg):
        # replicate the base-class bookkeeping; geometry is built here instead of via terrain_type branches
        self.cfg = cfg
        self.device = sim_utils.SimulationContext.instance().device  # type: ignore
        self.terrain_prim_paths = list()
        self.terrain_origins = None
        self.env_origins = None
        self._terrain_flat_patches = dict()

        stage = get_current_stage()
        rng = np.random.default_rng(cfg.seed)
        h, w = cfg.grid_height, cfg.grid_width
        root = f"{cfg.prim_path}/boxes"
        self.terrain_prim_paths.append(cfg.prim_path)

        # analytic ground plane below the lowest cell top, so cells lower than z = 0 stay exposed
        ground_cfg = sim_utils.GroundPlaneCfg(physics_material=cfg.physics_material, size=(2.0e4, 2.0e4))
        ground_cfg.func(f"{cfg.prim_path}/plane", ground_cfg, translation=(0.0, 0.0, -h - cfg.box_depth))

        # one physics material shared by all cells, bound per cell (Isaac Lab's helper only binds to collision prims)
        mat_path = f"{cfg.prim_path}/box_material"
        cfg.physics_material.func(mat_path, cfg.physics_material)

        xs = np.arange(cfg.x_range[0], cfg.x_range[1], w) + 0.5 * w
        ys = np.arange(cfg.y_range[0], cfg.y_range[1], w) + 0.5 * w
        spawn_y = np.linspace(cfg.y_range[0], cfg.y_range[1], cfg.num_spawn + 2)[1:-1]
        tops = rng.uniform(-h, h, size=(len(xs), len(ys)))
        # flatten cells under each spawn platform to +h (same as the mesh terrain's center platform)
        half_p = 0.5 * cfg.platform_width
        for sy in spawn_y:
            mask_x = np.abs(xs - cfg.spawn_x) <= half_p
            mask_y = np.abs(ys - sy) <= half_p
            tops[np.ix_(mask_x, mask_y)] = h

        if cfg.as_mesh:
            # identical cells as one triangle mesh: isolates "primitive vs mesh" from any layout difference
            boxes = []
            for i, x in enumerate(xs):
                for j, y in enumerate(ys):
                    height = tops[i, j] + h + cfg.box_depth
                    boxes.append(
                        trimesh.creation.box(
                            (w, w, height),
                            trimesh.transformations.translation_matrix((x, y, tops[i, j] - 0.5 * height)),
                        )
                    )
            self.import_mesh("terrain", trimesh.util.concatenate(boxes))
        else:
            self._spawn_box_prims(stage, root, xs, ys, tops, mat_path)
        self.num_boxes = len(xs) * len(ys)

        origins = np.zeros((1, cfg.num_spawn, 3))
        origins[0, :, 0] = cfg.spawn_x
        origins[0, :, 1] = spawn_y
        origins[0, :, 2] = h
        self.configure_env_origins(torch.tensor(origins, dtype=torch.float, device=self.device))
        self.set_debug_vis(self.cfg.debug_vis)

    def _spawn_box_prims(self, stage, root, xs, ys, tops, mat_path):
        h, w = self.cfg.grid_height, self.cfg.grid_width
        material = UsdShade.Material(stage.GetPrimAtPath(mat_path))
        UsdGeom.Xform.Define(stage, root)
        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                path = f"{root}/c_{i}_{j}"
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr(1.0)
                height = tops[i, j] + h + self.cfg.box_depth  # bottom sits on the lowered ground plane
                xf = UsdGeom.Xformable(cube)
                xf.AddTranslateOp().Set(Gf.Vec3d(float(x), float(y), float(tops[i, j] - 0.5 * height)))
                xf.AddScaleOp().Set(Gf.Vec3f(float(w), float(w), float(height)))
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(
                    material, UsdShade.Tokens.weakerThanDescendants, "physics"
                )


@configclass
class BoxGridTerrainImporterCfg(TerrainImporterCfg):
    class_type: type = BoxGridTerrainImporter
    terrain_type: str = "generator"  # unused by this importer; set so the base-class validation passes
    grid_width: float = 0.45
    grid_height: float = 0.08
    box_depth: float = 0.2
    """How far each cell extends below the lowest cell top (onto the lowered ground plane)."""
    platform_width: float = 1.5
    x_range: tuple[float, float] = (-80.0, 50.0)
    y_range: tuple[float, float] = (-16.0, 16.0)
    spawn_x: float = -76.0
    num_spawn: int = 4
    seed: int = 0
    as_mesh: bool = False
    """Build the identical cells as one triangle mesh (the mesh twin) instead of box prims."""
