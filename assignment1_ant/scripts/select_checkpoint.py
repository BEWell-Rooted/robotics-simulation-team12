"""Held-out checkpoint selection (docs/01_improvement_plan_0930.md, 3.6; Cobbe et al., ICML 2019).

1. SELECT: evaluate every candidate run x iteration on the selection held-out terrains (Boxes, Rails) and rank by
   mean first-episode reward over terrains and eval seeds (ties: lower fall rate).
2. REPORT: evaluate the selected checkpoint -- and, for reference, each run's last iteration -- on the report-only
   held-out terrains (SlopedGrid, Pyramids), which never influence the choice, so the reported number carries no
   selection bias.

Writes <out_prefix>_select.csv, <out_prefix>_ranking.csv, <out_prefix>_report.csv and prints both tables.

    python scripts/select_checkpoint.py --out_prefix results/select_v2 \\
        --run v2_s42=logs/rsl_rl/ant_rough/<run> --run v2_s43=... --iters 1500 2000 2500 2999
"""

import argparse
import csv
import importlib.util
import os
import subprocess
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
_spec = importlib.util.spec_from_file_location(
    "eval_names", os.path.join(PROJECT, "source", "ant_rough", "ant_rough", "tasks", "eval_names.py")
)
_names = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_names)


def run_suite(tasks, ckpts, out, seeds, num_envs, log_dir, deploy_task=None):
    cmd = ["python", os.path.join(HERE, "eval_suite.py"), "--out", out, "--tasks", *tasks, "--seeds", *map(str, seeds),
           "--num_envs", str(num_envs), "--log_dir", log_dir]  # fmt: skip
    if deploy_task:
        cmd += ["--deploy_task", deploy_task]
    for name, path in ckpts:
        cmd += ["--ckpt", f"{name}={path}"]
    subprocess.run(cmd, check=True, cwd=PROJECT)


def summarize(csv_path):
    acc = defaultdict(list)  # ckpt -> [(task, reward, fall)]
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            acc[r["checkpoint"]].append((r["task"], float(r["reward_mean"]), float(r["fall_rate"])))
    rows = []
    for ckpt, xs in acc.items():
        per_task = defaultdict(list)
        for t, rew, _ in xs:
            per_task[t].append(rew)
        rows.append({
            "checkpoint": ckpt,
            "reward_mean": sum(r for _, r, _ in xs) / len(xs),
            "fall_rate": sum(fr for _, _, fr in xs) / len(xs),
            **{t.replace("Isaac-Ant-Eval-", "").replace("-v0", ""): sum(v) / len(v) for t, v in per_task.items()},
        })  # fmt: skip
    rows.sort(key=lambda r: (-r["reward_mean"], r["fall_rate"]))
    return rows


def print_table(title, rows):
    keys = [k for k in rows[0] if k != "checkpoint"]
    print(f"\n{title}\n{'checkpoint':28s}" + "".join(f"{k:>13s}" for k in keys))
    for r in rows:
        print(f"{r['checkpoint']:28s}" + "".join(f"{r[k]:13.3f}" if k == "fall_rate" else f"{r[k]:13.1f}" for k in keys))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="name=run_dir, repeatable")
    parser.add_argument("--iters", type=int, nargs="+", default=[1500, 2000, 2500, 2999])
    parser.add_argument("--seeds", type=int, nargs="+", default=[24, 25, 26])
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--out_prefix", required=True)
    parser.add_argument("--skip_eval", action="store_true", help="only re-rank existing CSVs")
    parser.add_argument("--deploy_task", default=None, help="deploy task for axis-2 runs (terrain swapped in)")
    args = parser.parse_args()
    prefix = os.path.abspath(args.out_prefix)
    log_dir = os.path.join(PROJECT, "logs", "select")

    runs = [r.split("=", 1) for r in args.run]
    candidates = []
    for name, run_dir in runs:
        for it in args.iters:
            path = os.path.abspath(os.path.join(run_dir, f"model_{it}.pt"))
            if os.path.exists(path):
                candidates.append((f"{name}@{it}", path))
            else:
                print(f"[SELECT] skip missing {path}")

    select_csv = f"{prefix}_select.csv"
    if not args.skip_eval:
        run_suite(_names.SELECTION_TASKS, candidates, select_csv, args.seeds, args.num_envs, log_dir, args.deploy_task)
    ranking = summarize(select_csv)
    with open(f"{prefix}_ranking.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(ranking[0]))
        w.writeheader()
        w.writerows(ranking)
    print_table(f"SELECTION held-out {_names.SELECTION_TASKS} (ranked)", ranking)

    best = ranking[0]["checkpoint"]
    last = [f"{name}@{max(args.iters)}" for name, _ in runs]
    report_names = [best] + [c for c in last if c != best]
    paths = dict(candidates)
    report_csv = f"{prefix}_report.csv"
    if not args.skip_eval:
        run_suite(_names.REPORT_TASKS, [(n, paths[n]) for n in report_names if n in paths], report_csv, args.seeds,
                  args.num_envs, log_dir, args.deploy_task)  # fmt: skip
    print_table(f"REPORT held-out {_names.REPORT_TASKS} (selected = {best})", summarize(report_csv))
    print(f"\n[SELECT] selected checkpoint: {best} -> {paths[best]}")


if __name__ == "__main__":
    main()
