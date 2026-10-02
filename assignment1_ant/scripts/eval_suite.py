"""Run the self-made unseen evaluation suite: every Isaac-Ant-Eval-* task x every checkpoint x seeds.

Launches one Isaac Sim process per task (``eval_task.py``) and appends all rows to one CSV, then prints a
pivot (mean over seeds) of reward / fall rate per task and checkpoint. Plain Python: no Isaac Sim import here.

Usage (from assignment1_ant/, conda env lerobot-arena):
    python scripts/eval_suite.py --out results/eval_v1.csv \
        --ckpt baseline=checkpoints/baseline_flat/model_999.pt \
        --ckpt v1=logs/rsl_rl/ant_rough/<run>/model_2999.pt
    # subsets: --tasks Grid GridTall   |   more seeds: --seeds 24 25 26
"""

import argparse
import csv
import importlib.util
import os
import subprocess
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
# load the pure-Python name list by file path: importing the ant_rough package would require Isaac Sim
_spec = importlib.util.spec_from_file_location(
    "eval_names", os.path.join(PROJECT, "source", "ant_rough", "ant_rough", "tasks", "eval_names.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
EVAL_SPLITS: dict[str, str] = _mod.EVAL_SPLITS

ISAACLAB_SH = os.path.expanduser("~/IsaacLab_RS/isaaclab.sh")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", action="append", required=True, help="name=path, repeatable")
    parser.add_argument("--tasks", nargs="+", default=list(EVAL_SPLITS), help=f"subset of {list(EVAL_SPLITS)}")
    parser.add_argument("--seeds", type=int, nargs="+", default=[24, 25, 26])
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dump_dir", type=str, default=None)
    parser.add_argument("--log_dir", type=str, default="logs/eval_suite")
    parser.add_argument("--deploy_task", type=str, default=None,
                        help="evaluate in this deploy task with each eval terrain swapped in (axis-2 variants)")  # fmt: skip
    args = parser.parse_args()
    # child processes run with cwd=PROJECT: make every user path absolute w.r.t. the caller's cwd
    args.out = os.path.abspath(args.out)
    args.dump_dir = os.path.abspath(args.dump_dir) if args.dump_dir else None
    args.log_dir = os.path.abspath(args.log_dir)
    args.ckpt = [f"{n}={os.path.abspath(p)}" for n, p in (c.split("=", 1) for c in args.ckpt)]

    os.makedirs(args.log_dir, exist_ok=True)
    for name in args.tasks:
        split = EVAL_SPLITS[name]
        task = f"Isaac-Ant-Eval-{name}-v0"
        task_arg = ["--task", args.deploy_task, "--eval_terrain", name] if args.deploy_task else ["--task", task]
        cmd = [ISAACLAB_SH, "-p", os.path.join(HERE, "eval_task.py"), *task_arg, "--headless",
               "--num_envs", str(args.num_envs), "--split", split, "--out", args.out,
               "--seeds", *map(str, args.seeds)]  # fmt: skip
        for c in args.ckpt:
            cmd += ["--ckpt", c]
        if args.dump_dir:
            cmd += ["--dump_dir", args.dump_dir]
        log_path = os.path.join(args.log_dir, f"{task}{'__' + args.deploy_task if args.deploy_task else ''}.log")
        print(f"[SUITE] {task} ({split}) -> {log_path}", flush=True)
        with open(log_path, "w") as log:
            ret = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=PROJECT)
        if ret.returncode != 0:
            print(f"[SUITE] !! {task} exited with {ret.returncode}, see {log_path}", flush=True)

    print_pivot(args.out, [c.split("=", 1)[0] for c in args.ckpt])


def print_pivot(csv_path: str, ckpt_names: list[str]):
    acc = defaultdict(list)
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            acc[(row["task"], row["split"], row["checkpoint"])].append(row)
    tasks = sorted({(t, s) for t, s, _ in acc}, key=lambda ts: (ts[1] != "ID", ts[0]))
    header = f"{'task':34s} {'split':5s} " + " ".join(f"{n:>20s}" for n in ckpt_names)
    print("\nreward_mean (fall_rate) — mean over seeds\n" + header)
    for t, s in tasks:
        cells = []
        for n in ckpt_names:
            rows = acc.get((t, s, n), [])
            if not rows:
                cells.append(f"{'-':>20s}")
                continue
            r = sum(float(x["reward_mean"]) for x in rows) / len(rows)
            fr = sum(float(x["fall_rate"]) for x in rows) / len(rows)
            cells.append(f"{r:12.1f} ({fr:4.0%})")
        print(f"{t:34s} {s:5s} " + " ".join(cells))


if __name__ == "__main__":
    main()
