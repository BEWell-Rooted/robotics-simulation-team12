"""Extra MDP terms for the second round of variants (training only; evaluation always uses Isaac-Ant-v0's terms).

- ``cat_orientation_termination``: Constraints-as-Terminations (Chane-Sane et al., IROS 2024) on the posture that
  precedes a flip. Each step the episode ends with probability ``p_max * clip(violation, 0, 1)``, where violation is
  the larger normalized excess of torso tilt and roll/pitch rate. Used as a ``DoneTerm(time_out=False)`` so the value
  is not bootstrapped past the cut.
- ``level_friction``: per-env robot friction whose sampling range widens with the env's curriculum level
  (narrow around mu 1.0 on easy lanes, wide on hard ones), re-drawn at every reset after the curriculum update.
"""

from __future__ import annotations

import math

import torch

from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg


def cat_orientation_termination(
    env,
    min_up: float = math.cos(math.radians(35.0)),
    max_ang_vel_xy: float = 3.0,
    p_max: float = 0.05,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    up = -asset.data.projected_gravity_b[:, 2]  # 1 upright, 0 on its side, -1 upside down
    v_tilt = torch.clamp((min_up - up) / min_up, min=0.0)  # 0 inside the cone, 1 at 90 deg
    w_xy = torch.linalg.norm(asset.data.root_ang_vel_b[:, :2], dim=1)
    v_rate = torch.clamp((w_xy - max_ang_vel_xy) / max_ang_vel_xy, min=0.0)  # 1 at twice the limit
    p = p_max * torch.clamp(torch.maximum(v_tilt, v_rate), max=1.0)
    return torch.rand_like(p) < p


class level_friction(ManagerTermBase):
    """Reset event: robot static = dynamic friction ~ U(1 - w, 1 + w), w growing linearly with the terrain level."""

    def __init__(self, cfg: EventTermCfg, env):
        super().__init__(cfg, env)
        self.asset = env.scene[cfg.params.get("asset_cfg", SceneEntityCfg("robot")).name]

    def __call__(self, env, env_ids, w_min: float = 0.1, w_max: float = 0.8, asset_cfg: SceneEntityCfg | None = None):
        terrain = env.scene.terrain
        if env_ids is None:
            env_ids = torch.arange(env.num_envs, device=env.device)
        levels = terrain.terrain_levels[env_ids].float()
        frac = levels / max(int(terrain.max_terrain_level) - 1, 1)
        w = w_min + (w_max - w_min) * frac.clamp(0.0, 1.0)
        mu = (1.0 + (2.0 * torch.rand_like(w) - 1.0) * w).cpu()
        ids = env_ids.cpu()
        materials = self.asset.root_physx_view.get_material_properties()  # (num_envs, num_shapes, 3) on CPU
        materials[ids, :, 0] = mu[:, None]
        materials[ids, :, 1] = mu[:, None]
        self.asset.root_physx_view.set_material_properties(materials, ids)
