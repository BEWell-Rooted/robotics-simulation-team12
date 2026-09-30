"""Mesh-flat fall diagnosis (docs/01_improvement_plan_0930.md, section 2) with existing checkpoints only.

    H3  friction grid : {analytic plane, mesh flat} x effective mu in MU_GRID (ground combine "multiply";
                        robot shapes are mu 1.0, so effective contact friction = ground mu)
    H1  threshold     : mesh flat (mu 1.0) with torso termination at 0.28 / 0.25 / 0.20 (0.31 = H3 mesh mu 1.0)
    H2  force obs     : feet_body_forces statistics on plane vs mesh (mu 1.0); mesh flat with that 24-dim
                        observation clipped to each checkpoint's own plane p99, or zeroed
    BOX primitives    : identical random block strip (w 0.45, h +-0.08) as USD box prims vs one triangle mesh

Two queues run in parallel (queue A holds the plane runs that H2-clip depends on). All rows go to one CSV with a
``cond`` label. Plain Python; each run is an ``eval_task.py`` subprocess.

    python scripts/diag_meshflat.py --out results/diag_meshflat.csv
"""

import argparse
import os
import subprocess
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
ISAACLAB_SH = os.path.expanduser("~/IsaacLab_RS/isaaclab.sh")
R = "logs/rsl_rl/ant_rough"
CKPTS = {
    "baseline": "checkpoints/baseline_flat/model_999.pt",
    "v1": f"{R}/2026-09-29_16-51-24_v1_dr_mix_s42/model_2999.pt",
    "v1_s43": f"{R}/2026-09-29_18-23-21_v1_dr_mix_s43/model_2999.pt",
    "v2": f"{R}/2026-09-29_17-29-32_v2_curr_mix_s42/model_2999.pt",
    "v2_s43": f"{R}/2026-09-29_19-13-05_v2_curr_mix_s43/model_2999.pt",
    "v2_s44": f"{R}/2026-09-29_19-13-05_v2_curr_mix_s44/model_2999.pt",
}
# must match eval_task.CSV_FIELDS
CSV_FIELDS = ["task", "split", "checkpoint", "seed", "num_envs", "reward_mean", "reward_std", "steps_mean",
              "fall_rate", "x_dist_mean", "x_dist_std", "cond", "fall_upright_rate"]  # fmt: skip
MU_GRID = [0.2, 0.5, 0.8, 1.0, 1.5]
PLANE, MESH = "Isaac-Ant-v0", "Isaac-Ant-Eval-Flat-v0"


def runs(stats_dir):
    queue_a, queue_b = [], []
    for mu in MU_GRID:
        common = ["--ground_friction", str(mu), "--ground_combine", "multiply"]
        queue_a.append((f"H3|plane|mu={mu}", PLANE, common + ["--obs_stats_dir", f"{stats_dir}/plane_mu{mu}"]))
        queue_b.append((f"H3|mesh|mu={mu}", MESH, common + ["--obs_stats_dir", f"{stats_dir}/mesh_mu{mu}"]))
    mu1 = ["--ground_friction", "1.0", "--ground_combine", "multiply"]
    for h in (0.28, 0.25, 0.20):
        queue_b.append((f"H1|mesh|h={h}", MESH, mu1 + ["--min_height", str(h)]))
    queue_a.append(("H2|mesh|clip_plane_p99", MESH, mu1 + ["--force_obs", "clip", "--force_clip_dir", f"{stats_dir}/plane_mu1.0"]))
    queue_a.append(("H2|mesh|zero", MESH, mu1 + ["--force_obs", "zero"]))
    queue_a.append(("BOX|prim", "Isaac-Ant-Diag-BoxPrim-v0", []))
    queue_b.append(("BOX|mesh", "Isaac-Ant-Diag-BoxMesh-v0", []))
    return queue_a, queue_b


def run_queue(queue, args, log_dir):
    for cond, task, extra in queue:
        cmd = [ISAACLAB_SH, "-p", os.path.join(HERE, "eval_task.py"), "--task", task, "--headless",
               "--num_envs", str(args.num_envs), "--seeds", *map(str, args.seeds), "--out", args.out,
               "--cond", cond, "--split", "diag", "--dump_dir", args.dump_dir, *extra]  # fmt: skip
        for name, path in CKPTS.items():
            cmd += ["--ckpt", f"{name}={os.path.join(PROJECT, path)}"]
        log = os.path.join(log_dir, cond.replace("|", "__").replace("=", "") + ".log")
        print(f"[DIAG] {cond} -> {log}", flush=True)
        with open(log, "w") as f:
            ret = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=PROJECT)
        if ret.returncode != 0:
            print(f"[DIAG] !! {cond} exited with {ret.returncode}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="results/diag_meshflat.csv")
    parser.add_argument("--seeds", type=int, nargs="+", default=[24, 25])
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--stats_dir", default="results/diag_force_stats")
    parser.add_argument("--dump_dir", default="logs/eval_dump_diag")
    args = parser.parse_args()
    args.out = os.path.abspath(args.out)
    args.dump_dir = os.path.abspath(args.dump_dir)
    stats_dir = os.path.abspath(args.stats_dir)
    log_dir = os.path.join(PROJECT, "logs", "diag")
    os.makedirs(log_dir, exist_ok=True)
    # write the header once up front: both queues append to the same CSV concurrently
    if not os.path.exists(args.out):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(",".join(CSV_FIELDS) + "\n")
    queue_a, queue_b = runs(stats_dir)
    threads = [threading.Thread(target=run_queue, args=(q, args, log_dir)) for q in (queue_a, queue_b)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main()
