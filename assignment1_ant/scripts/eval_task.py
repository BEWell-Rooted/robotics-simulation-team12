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
# --- diagnosis options (docs/01_improvement_plan_0930.md, section 2); defaults leave the task unchanged ---
parser.add_argument("--cond", type=str, default="", help="free-form condition label written to the CSV")
parser.add_argument("--eval_terrain", type=str, default=None,
                    help="swap the --task (deploy) config's terrain for this Isaac-Ant-Eval-* terrain (axis-2 variants)")  # fmt: skip
parser.add_argument("--min_height", type=float, default=None, help="override torso_height termination threshold")
parser.add_argument("--ground_friction", type=float, default=None, help="override ground static=dynamic friction")
parser.add_argument("--ground_combine", type=str, default=None, help="override ground friction combine mode")
parser.add_argument("--force_obs", choices=["none", "clip", "zero"], default="none",
                    help="modify the 24-dim feet_body_forces observation before the policy sees it")  # fmt: skip
parser.add_argument("--force_clip_dir", type=str, default=None, help="dir with <ckpt>.npz 'p99' bounds for clip")
parser.add_argument("--obs_stats_dir", type=str, default=None, help="save feet_body_forces statistics per ckpt")
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
    "cond", "fall_upright_rate",
]  # fmt: skip
FORCE_SLICE = slice(28, 52)  # feet_body_forces inside the 60-dim policy obs (see docs/00_codebase_notes.md, 2)
UPRIGHT_COS = 0.7  # a fall with torso up-projection >= this counts as "upright but too low", else "tipped over"


def run_first_episodes(env: RslRlVecEnvWrapper, policy, force_bound=None) -> tuple[dict, dict[str, float]]:
    """Reset all envs and roll out until every env has finished its first episode.

    ``force_bound``: None (unchanged), 0 (zero the force obs) or a (24,) tensor of |x| bounds to clip to.
    """
    # everything (including reset) in inference mode: env buffers touched by step() become inference tensors
    with torch.inference_mode():
        return _rollout(env, policy, force_bound)


def _rollout(env: RslRlVecEnvWrapper, policy, force_bound=None) -> tuple[dict, dict[str, float]]:
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
    up_last = torch.ones(n, device=env.device)
    # generated terrains are centred at the world origin and surrounded by a flat border: count steps spent outside
    tg = getattr(base_env.cfg.scene.terrain, "terrain_generator", None)
    half = (tg.num_rows * tg.size[0] / 2, tg.num_cols * tg.size[1] / 2) if tg is not None else (float("inf"),) * 2
    off_steps = torch.zeros(n, dtype=torch.long, device=env.device)
    y_absmax = robot.data.root_pos_w[:, 1].abs().clone()
    force_samples = []
    for step in range(int(base_env.max_episode_length) + 1):
        active_now = ~finished
        if step % 4 == 0 and active_now.any():
            force_samples.append(obs["policy"][active_now, FORCE_SLICE].clone())
        if force_bound is not None:
            f = obs["policy"][:, FORCE_SLICE]
            obs["policy"][:, FORCE_SLICE] = torch.zeros_like(f) if isinstance(force_bound, int) else f.clamp(
                -force_bound, force_bound
            )
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
        up_last[still] = -robot.data.projected_gravity_b[still, 2]
        pos = robot.data.root_pos_w
        off = (pos[:, 0].abs() > half[0]) | (pos[:, 1].abs() > half[1])
        off_steps[still & off] += 1
        y_absmax[still] = torch.maximum(y_absmax[still], pos[still, 1].abs())
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
        "up_last": up_last.cpu().numpy(),
        "off_steps": off_steps.cpu().numpy(),
        "y_absmax": y_absmax.cpu().numpy(),
        "terrain_half": np.array(half),
        "force_samples": torch.cat(force_samples).cpu().numpy() if force_samples else np.zeros((0, 24)),
    }
    return per_env, {
        "reward_mean": rewards_sum.mean().item(),
        "reward_std": rewards_sum.std(unbiased=False).item(),
        "steps_mean": steps.double().mean().item(),
        "fall_rate": fell.double().mean().item(),
        "x_dist_mean": x_dist.mean().item(),
        "x_dist_std": x_dist.std(unbiased=False).item(),
        "fall_upright_rate": (fell & (up_last >= UPRIGHT_COS)).double().mean().item(),
        "off_terrain_frac": (off_steps.double().sum() / steps.double().sum().clamp(min=1)).item(),
        "off_terrain_envs": (off_steps > 0).double().mean().item(),
    }


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seeds[0]
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    task_label = args_cli.task
    if args_cli.eval_terrain is not None:
        from ant_rough.tasks.eval_envs import EVAL_SPECS, apply_eval_terrain

        apply_eval_terrain(env_cfg, **EVAL_SPECS[args_cli.eval_terrain][1])
        task_label = f"Isaac-Ant-Eval-{args_cli.eval_terrain}-v0"
    if args_cli.min_height is not None:
        env_cfg.terminations.torso_height.params["minimum_height"] = args_cli.min_height
    mat = env_cfg.scene.terrain.physics_material
    if args_cli.ground_friction is not None:
        mat.static_friction = mat.dynamic_friction = args_cli.ground_friction
    if args_cli.ground_combine is not None:
        mat.friction_combine_mode = args_cli.ground_combine

    env = RslRlVecEnvWrapper(gym.make(args_cli.task, cfg=env_cfg), clip_actions=agent_cfg.clip_actions)
    # robot shape materials (static, dynamic, restitution) of env 0: needed to read the effective contact friction
    robot_mats = env.unwrapped.scene["robot"].root_physx_view.get_material_properties()
    print(f"[EVAL] ground material: {mat.static_friction}/{mat.dynamic_friction} ({mat.friction_combine_mode}), "
          f"robot shape materials env0 (first 2): {robot_mats[0, :2].tolist()}, "
          f"env{robot_mats.shape[0] - 1} (first 2): {robot_mats[-1, :2].tolist()}", flush=True)  # fmt: skip

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    write_header = not os.path.exists(args_cli.out)
    with open(args_cli.out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")  # off-terrain stats: printed only
        if write_header:
            writer.writeheader()
        for spec in args_cli.ckpt:
            name, path = spec.split("=", 1)
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(os.path.abspath(path))  # strict: fails if the network differs from the task's config
            policy = runner.get_inference_policy(device=env.unwrapped.device)
            force_bound = None
            if args_cli.force_obs == "zero":
                force_bound = 0
            elif args_cli.force_obs == "clip":
                bound = np.load(os.path.join(args_cli.force_clip_dir, f"{name}.npz"))["p99"]
                force_bound = torch.tensor(bound, dtype=torch.float, device=env.unwrapped.device)
            all_forces = []
            for seed in args_cli.seeds:
                env.unwrapped.seed(seed)
                if getattr(runner.alg.policy, "is_recurrent", False):
                    runner.alg.policy.reset()  # fresh LSTM state per rollout (first episode starts from zeros)
                per_env, stats = run_first_episodes(env, policy, force_bound)
                all_forces.append(per_env.pop("force_samples"))
                if args_cli.dump_dir:
                    os.makedirs(args_cli.dump_dir, exist_ok=True)
                    np.savez(os.path.join(args_cli.dump_dir, f"{task_label}__{name}__s{seed}.npz"), **per_env)
                row = {"task": task_label, "split": args_cli.split, "checkpoint": name, "seed": seed,
                       "num_envs": env.num_envs, "cond": args_cli.cond,
                       **{k: round(v, 4) for k, v in stats.items()}}  # fmt: skip
                writer.writerow(row)
                f.flush()
                print(f"[EVAL] {row}", flush=True)
            if args_cli.obs_stats_dir:
                # statistics of the observation the policy would receive (before any clip/zero)
                x = np.abs(np.concatenate(all_forces))
                os.makedirs(args_cli.obs_stats_dir, exist_ok=True)
                idx = np.random.default_rng(0).choice(len(x), size=min(len(x), 20000), replace=False)
                np.savez(os.path.join(args_cli.obs_stats_dir, f"{name}.npz"), mean=x.mean(0),
                         p99=np.percentile(x, 99, axis=0), max=x.max(0), sample=x[idx])  # fmt: skip

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
