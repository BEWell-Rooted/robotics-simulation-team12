"""Left-right (y -> -y) mirror symmetry of the Ant for symmetric data augmentation (V8; Mittal et al., ICRA 2024).

The target is +x, so reflecting the world across the xz-plane maps a valid state/action to another valid one.
Ant layout (joint order = body order after the torso): front_left (+x,+y), front_right (-x,+y), left_back (-x,-y),
right_back (+x,-y) -- so the mirror pairs are front_left <-> right_back and front_right <-> left_back.

- hip joints (``*_leg``): all rotate counter-clockwise about +z for positive angles -> mirrored angle = -partner
- ankle joints (``*_foot``): partners share the same limits -> mirrored angle = +partner
- body-frame linear velocity (vx, -vy, vz); angular velocity (pseudo-vector) (-wx, wy, -wz); yaw, roll and the
  heading angle to the target flip sign; height and up/heading projections are unchanged
- feet wrench: per-component partner and sign measured in simulation (``scripts/check_symmetry.py``), see WRENCH_MAP

Each observation term keeps its own (history x dim) block, so the per-frame mapping is applied per term and repeated
over the history steps (Isaac Lab flattens history per term).
"""

from __future__ import annotations

import torch

# joint / action index order: [FL_leg, FR_leg, LB_leg, RB_leg, FL_foot, FR_foot, LB_foot, RB_foot]
FOOT_PARTNER = [3, 2, 1, 0]  # FL<->RB, FR<->LB
JOINT_IDX = [3, 2, 1, 0, 4 + 3, 4 + 2, 4 + 1, 4 + 0]
JOINT_SIGN = [-1.0] * 4 + [1.0] * 4

# feet wrench (6 per foot: fx fy fz tx ty tz in each foot's parent frame): for a mirrored foot, component c takes the
# partner's component WRENCH_MAP[c][0] times WRENCH_MAP[c][1]. Filled from scripts/check_symmetry.py.
WRENCH_MAP = [(0, -1.0), (1, 1.0), (2, 1.0), (3, 1.0), (4, -1.0), (5, -1.0)]  # measured: corr 0.81-0.99, same on all 4 feet

# per-term (dim, per-dim (source index within term, sign)) for one frame, in policy-group order
def _term_maps():
    wrench = []
    for f in range(4):
        p = FOOT_PARTNER[f]
        wrench += [(6 * p + src, s) for src, s in WRENCH_MAP]
    contact = [(FOOT_PARTNER[f], 1.0) for f in range(4)] + [(4 + FOOT_PARTNER[f], 1.0) for f in range(4)]
    joints = list(zip(JOINT_IDX, JOINT_SIGN))
    return {
        "base_height": [(0, 1.0)],
        "base_lin_vel": [(0, 1.0), (1, -1.0), (2, 1.0)],
        "base_ang_vel": [(0, -1.0), (1, 1.0), (2, -1.0)],
        "base_yaw_roll": [(0, -1.0), (1, -1.0)],
        "base_angle_to_target": [(0, -1.0)],
        "base_up_proj": [(0, 1.0)],
        "base_heading_proj": [(0, 1.0)],
        "joint_pos_norm": joints,
        "joint_vel_rel": joints,
        "feet_body_forces": wrench,
        "actions": joints,
        "feet_contact": contact,
    }


def build_obs_map(term_names: list[str], term_dims: list[int], history: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Index and sign vectors so that mirrored_obs = obs[:, idx] * sign for the flattened policy observation."""
    maps = _term_maps()
    idx, sign, offset = [], [], 0
    for name, dim in zip(term_names, term_dims):
        frame = maps[name]
        d = len(frame)
        h = max(history, 1)
        assert dim == d * h, f"term {name}: dim {dim} != {d} x {h}"
        for step in range(h):
            for src, s in frame:
                idx.append(offset + step * d + src)
                sign.append(s)
        offset += dim
    return torch.tensor(idx), torch.tensor(sign)


_CACHE = {}


def _maps_for(env, device):
    key = (id(env), device)
    if key not in _CACHE:
        om = env.unwrapped.observation_manager
        names = om.active_terms["policy"]
        dims = [int(torch.tensor(d).prod()) for d in om.group_obs_term_dim["policy"]]
        history = om.cfg.policy.history_length or 0
        idx, sign = build_obs_map(names, dims, history)
        _CACHE[key] = (idx.to(device), sign.to(device), torch.tensor(JOINT_IDX, device=device),
                       torch.tensor(JOINT_SIGN, device=device))  # fmt: skip
    return _CACHE[key]


def mirror_obs(obs: torch.Tensor, env) -> torch.Tensor:
    idx, sign, _, _ = _maps_for(env, obs.device)
    assert obs.shape[1] == idx.numel(), f"obs dim {obs.shape[1]} != policy map {idx.numel()} (critic group differs?)"
    return obs[:, idx] * sign


def mirror_actions(actions: torch.Tensor, env) -> torch.Tensor:
    _, _, jidx, jsign = _maps_for(env, actions.device)
    return actions[:, jidx] * jsign


def ant_mirror_augmentation(env, obs=None, actions=None):
    """rsl-rl data_augmentation_func: returns (original + mirrored) batches."""
    obs_out = None
    if obs is not None:
        obs_out = obs.clone()
        mirrored = obs.clone()
        for key in mirrored.keys():
            mirrored[key] = mirror_obs(obs[key], env)
        obs_out = torch.cat([obs_out, mirrored], dim=0)
    act_out = None
    if actions is not None:
        act_out = torch.cat([actions, mirror_actions(actions, env)], dim=0)
    return obs_out, act_out
