"""Verify the Ant mirror map (tasks/symmetry.py) in simulation and measure the feet-wrench component map.

Two envs on a flat plane. Env 1 is set to the mirror image of env 0's state, then both are stepped with mirrored
actions. If the map is right, obs(env1) == mirror_obs(obs(env0)) for every term (up to simulation noise). For the
feet wrench (whose per-foot frames make signs hard to derive by hand) every mirrored component is correlated with the
6 components of the partner foot to find its source and sign.

    ~/IsaacLab_RS/isaaclab.sh -p scripts/check_symmetry.py --headless
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import ant_rough.tasks  # noqa: E402, F401
from ant_rough.tasks.symmetry import FOOT_PARTNER, JOINT_IDX, JOINT_SIGN, mirror_actions, mirror_obs  # noqa: E402

OUT = open("logs/check_symmetry.txt", "w")


def log(*a):
    OUT.write(" ".join(str(x) for x in a) + "\n")
    OUT.flush()


TASK = "Isaac-Ant-Deploy-V1214-v0"
cfg = parse_env_cfg(TASK, num_envs=2)
cfg.terminations.torso_height = None  # keep the pair alive (no resets while comparing)
env = gym.make(TASK, cfg=cfg)
base = env.unwrapped
robot = base.scene["robot"]
om = base.observation_manager
names = om.active_terms["policy"]
dims = [int(np.prod(d)) for d in om.group_obs_term_dim["policy"]]
H = om.cfg.policy.history_length
log("terms", list(zip(names, dims)), "history", H)
dev = base.device
jidx = torch.tensor(JOINT_IDX, device=dev)
jsign = torch.tensor(JOINT_SIGN, device=dev)


def mirror_state():
    o = base.scene.env_origins
    rs = robot.data.root_state_w.clone()
    s = rs[0].clone()
    p = s[:3] - o[0]
    s[:3] = o[1] + torch.stack([p[0], -p[1], p[2]])
    s[3:7] = torch.stack([s[3], -s[4], s[5], -s[6]])  # quaternion (w, x, y, z) reflected across xz
    s[7:10] = torch.stack([s[7], -s[8], s[9]])
    s[10:13] = torch.stack([-s[10], s[11], -s[12]])
    robot.write_root_state_to_sim(s[None], env_ids=torch.tensor([1], device=dev))
    q = (robot.data.joint_pos[0, jidx] * jsign)[None]
    qd = (robot.data.joint_vel[0, jidx] * jsign)[None]
    robot.write_joint_state_to_sim(q, qd, env_ids=torch.tensor([1], device=dev))


pairs = []
torch.manual_seed(0)
for rollout in range(60):
    base.reset()
    for _ in range(15):  # wander off the default pose
        base.step(torch.rand(2, 8, device=dev) * 2 - 1)
    mirror_state()
    for t in range(40):
        a0 = torch.rand(1, 8, device=dev) * 2 - 1
        a = torch.cat([a0, mirror_actions(a0, env)], dim=0)
        obs, *_ = base.step(a)
        if H + 1 <= t <= H + 4:  # refreshed history, before the two trajectories drift apart
            pairs.append((obs["policy"][0:1].clone(), obs["policy"][1:2].clone()))

o0 = torch.cat([p[0] for p in pairs])
o1 = torch.cat([p[1] for p in pairs])
pred = mirror_obs(o0, env)
off = 0
for n, d in zip(names, dims):
    err = (o1[:, off:off + d] - pred[:, off:off + d]).abs()
    scale = o1[:, off:off + d].abs().mean().item() + 1e-6
    # per-dim correlation of prediction and actual on the newest frame: +1 = right source and sign
    fd = d // H
    a = o1[:, off + d - fd: off + d].cpu().numpy()
    b = pred[:, off + d - fd: off + d].cpu().numpy()
    cors = [np.corrcoef(a[:, i], b[:, i])[0, 1] if a[:, i].std() > 1e-6 and b[:, i].std() > 1e-6 else float("nan")
            for i in range(fd)]
    log(f"{n:22s} dim {d:3d}  mean|err| {err.mean().item():.4f}  mean|obs| {scale:.4f}  corr(newest) {np.round(cors, 2).tolist()}")
    if n == "feet_body_forces":
        # current frame = last 24 values of the term block (history oldest -> newest)
        cur0 = o0[:, off + d - 24: off + d].cpu().numpy()
        cur1 = o1[:, off + d - 24: off + d].cpu().numpy()
        for f in range(4):
            p = FOOT_PARTNER[f]
            for c in range(6):
                y = cur1[:, 6 * f + c]
                cors = [np.corrcoef(y, cur0[:, 6 * p + k])[0, 1] if cur0[:, 6 * p + k].std() > 1e-6 else 0 for k in range(6)]
                k = int(np.argmax(np.abs(cors)))
                log(f"  wrench foot{f} comp{c} <- partner foot{p} comp{k} corr {cors[k]:+.3f}   all {np.round(cors, 2).tolist()}")
    off += d
log("pairs", len(pairs))
env.close()
simulation_app.close()
