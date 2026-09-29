"""Evaluate several RSL-RL checkpoints on ONE task inside a single Isaac Sim process.

For every (checkpoint, seed) all environments are reset and run until each has finished its *first* episode
(the same statistic as ``play_one_episode.py``, which the TA uses). One CSV row per (task, checkpoint, seed):

    task, split, checkpoint, seed, num_envs, reward_mean, reward_std, steps_mean, fall_rate, x_dist_mean, x_dist_std

The network is built from the evaluated task's ``rsl_rl_cfg_entry_point`` (the baseline Ant config for the
``Isaac-Ant-Eval-*`` tasks), exactly as the TA's play script would, so a checkpoint that fails to load here would
also fail for the TA.

Usage (from assignment1_ant/):
    ~/IsaacLab_RS/isaaclab.sh -p scripts/eval_task.py --task Isaac-Ant-Eval-Grid-v0 --headless \
        --ckpt baseline=checkpoints/baseline_flat/model_999.pt --ckpt v1=logs/.../model_2999.pt \
        --seeds 24 25 26 --num_envs 256 --out results/eval.csv
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsl_rl"))
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate checkpoints on one task (first-episode statistics).")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--agent", type=str, default="rsl_rl_cfg_entry_point")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--ckpt", action="append", required=True, help="name=path, repeatable")
parser.add_argument("--seeds", type=int, nargs="+", default=[24])
parser.add_argument("--split", type=str, default="", help="ID / OOD label written to the CSV")
parser.add_argument("--out", type=str, required=True, help="CSV file to append to")
parser.add_argument("--dump_dir", type=str, default=None, help="also save per-env arrays (.npz) here")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import csv

import numpy as np

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

import ant_rough.tasks  # noqa: F401

CSV_FIELDS = [
    "task", "split", "checkpoint", "seed", "num_envs",
    "reward_mean", "reward_std", "steps_mean", "fall_rate", "x_dist_mean", "x_dist_std",
]  # fmt: skip


def run_first_episodes(env: RslRlVecEnvWrapper, policy) -> tuple[dict, dict[str, float]]:
    """Reset all envs and roll out until every env has finished its first episode."""
    # everything (including reset) in inference mode: env buffers touched by step() become inference tensors
    with torch.inference_mode():
        return _rollout(env, policy)


def _rollout(env: RslRlVecEnvWrapper, policy) -> tuple[dict, dict[str, float]]:
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    obs, _ = env.reset()
    n = env.num_envs
    rewards_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    steps = torch.zeros(n, dtype=torch.long, device=env.device)
    fell = torch.zeros(n, dtype=torch.bool, device=env.device)
    finished = torch.zeros(n, dtype=torch.bool, device=env.device)
    x_start = robot.data.root_pos_w[:, 0].clone()
    x_last = x_start.clone()
    y_last = robot.data.root_pos_w[:, 1].clone()
    z_last = robot.data.root_pos_w[:, 2].clone()
    z_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    z_min = torch.full((n,), float("inf"), device=env.device)
    speed_sum = torch.zeros(n, dtype=torch.float64, device=env.device)
    for _ in range(int(base_env.max_episode_length) + 1):
        actions = policy(obs)
        obs, rewards, dones, extras = env.step(actions)
        active = ~finished
        rewards_sum[active] += rewards[active].double()
        steps[active] += 1
        done = dones.bool()
        time_outs = extras.get("time_outs", torch.zeros_like(done)).bool()
        # position is already auto-reset for envs that just finished, so only envs still running update x_last
        still = active & ~done
        x_last[still] = robot.data.root_pos_w[still, 0]
        y_last[still] = robot.data.root_pos_w[still, 1]
        z_last[still] = robot.data.root_pos_w[still, 2]
        # torso height (world z) and forward speed statistics over the steps each env is still running
        z_sum[still] += robot.data.root_pos_w[still, 2].double()
        z_min[still] = torch.minimum(z_min[still], robot.data.root_pos_w[still, 2])
        speed_sum[still] += robot.data.root_lin_vel_w[still, 0].double()
        fell |= active & done & ~time_outs
        finished |= done
        if finished.all():
            break
    x_dist = (x_last - x_start).double()
    per_env = {
        "reward": rewards_sum.cpu().numpy(),
        "steps": steps.cpu().numpy(),
        "fell": fell.cpu().numpy(),
        "x_start": x_start.cpu().numpy(),
        "xyz_last": torch.stack([x_last, y_last, z_last], dim=-1).cpu().numpy(),
        "z_mean": (z_sum / steps.clamp(min=1)).cpu().numpy(),
        "z_min": z_min.cpu().numpy(),
        "vx_mean": (speed_sum / steps.clamp(min=1)).cpu().numpy(),
    }
    return per_env, {
        "reward_mean": rewards_sum.mean().item(),
        "reward_std": rewards_sum.std(unbiased=False).item(),
        "steps_mean": steps.double().mean().item(),
        "fall_rate": fell.double().mean().item(),
        "x_dist_mean": x_dist.mean().item(),
        "x_dist_std": x_dist.std(unbiased=False).item(),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seeds[0]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    write_header = not os.path.exists(args_cli.out)
    with open(args_cli.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()
        for spec in args_cli.ckpt:
            name, path = spec.split("=", 1)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(os.path.abspath(path))  # strict: fails if the network differs from the task's config
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            for seed in args_cli.seeds:
                env.unwrapped.seed(seed)
                per_env, stats = run_first_episodes(env, policy)
                if args_cli.dump_dir:
                    os.makedirs(args_cli.dump_dir, exist_ok=True)
                    np.savez(os.path.join(args_cli.dump_dir, f"{args_cli.task}__{name}__s{seed}.npz"), **per_env)
                row = {"task": args_cli.task, "split": args_cli.split, "checkpoint": name, "seed": seed,
                       "num_envs": env.num_envs, **{k: round(v, 4) for k, v in stats.items()}}  # fmt: skip
                writer.writerow(row)
                f.flush()
                print(f"[EVAL] {row}", flush=True)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
