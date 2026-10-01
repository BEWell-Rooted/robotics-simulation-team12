"""Render one still image of a task's terrain (one robot at its spawn point for scale) -> PNG.

The camera is fixed relative to env 0's origin (not following the robot), and the terrain gets a light grey visual
material for the still only (the task default is black, which hides relief in a single frame). Physics, friction
and geometry are untouched.

    ~/IsaacLab_RS/isaaclab.sh -p scripts/capture_terrains.py --task Isaac-Ant-Eval-Grid-v0 --out docs/figures/terrains/Grid.png --headless
    # --view overview: higher, wider camera for the training patchworks
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Render a still image of a task's terrain.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--view", choices=["close", "overview"], default="close")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import numpy as np
import torch
from PIL import Image

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnvCfg

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import ant_rough.tasks  # noqa: F401

VIEWS = {
    # (eye, lookat) relative to env 0's origin; the robot runs toward +x
    "close": ((-3.2, -4.6, 2.6), (3.0, 0.6, 0.0)),
    # high oblique view across the track: tiles (V1/V5) and y-lanes (V2) both visible
    "overview": ((20.0, -42.0, 46.0), (24.0, 14.0, 0.0)),
}


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    env_cfg.viewer.eye, env_cfg.viewer.lookat = VIEWS[args_cli.view]
    env_cfg.viewer.resolution = (1600, 900)
    terrain = env_cfg.scene.terrain
    if getattr(terrain, "terrain_type", None) == "plane":
        terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.48, 0.45))
    elif args_cli.view == "overview":
        # from far away shading alone washes out: color the generated mesh by height instead (vertex colors only
        # show when no visual material is bound)
        terrain.visual_material = None
        terrain.terrain_generator.color_scheme = "height"
    else:
        terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.50, 0.48, 0.45), roughness=0.8)

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    env.reset()
    zero = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    frame = None
    with torch.inference_mode():
        for _ in range(30):  # let the robot settle and the renderer converge
            env.step(zero)
            frame = env.render()
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    Image.fromarray(np.asarray(frame)[..., :3]).save(args_cli.out)
    print(f"[TERRAIN] {args_cli.task} -> {args_cli.out}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
