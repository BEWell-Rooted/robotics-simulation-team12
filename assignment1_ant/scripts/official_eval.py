"""Official self-evaluation protocol (TA notice): play_one_episode.py --seed 24 --num_envs 100 -> mean / std.

Runs ``scripts/rsl_rl/play_one_episode.py`` -- the course script copied from IsaacLab_RS with one added line,
``import ant_rough.tasks`` (registers our tasks) -- once per (task, checkpoint), parses the two [RESULT] lines and
appends a CSV row. Each checkpoint is copied to its own folder first because the script exports the policy next to
the checkpoint.

    python scripts/official_eval.py --out results/official_test.csv \\
        --run name=Isaac-Ant-Test-UnseenMix-v0=logs/rsl_rl/ant_rough/<run>/model_2999.pt
"""

import argparse
import csv
import os
import re
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
ISAACLAB_SH = os.path.expanduser("~/IsaacLab_RS/isaaclab.sh")
FIELDS = ["name", "task", "checkpoint", "seed", "num_envs", "reward_mean", "reward_std", "steps_mean", "steps_std",
          "completed"]  # fmt: skip
RE_R = re.compile(r"Episode reward total: mean=([-0-9.e]+), std=([-0-9.e]+)")
RE_S = re.compile(r"Episode steps: mean=([-0-9.e]+), std=([-0-9.e]+)")
RE_C = re.compile(r"Completed first episodes: (\d+)/(\d+)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", action="append", required=True, help="name=task=checkpoint, repeatable")
    parser.add_argument("--seed", type=int, default=24)
    parser.add_argument("--num_envs", type=int, default=100)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if not os.path.exists(out):
        with open(out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    for spec in args.run:
        name, task, ckpt = spec.split("=", 2)
        work = os.path.join(PROJECT, "logs", "official_ckpt", name)
        os.makedirs(work, exist_ok=True)
        local = os.path.join(work, "model.pt")
        shutil.copy(os.path.join(PROJECT, ckpt) if not os.path.isabs(ckpt) else ckpt, local)
        cmd = [ISAACLAB_SH, "-p", os.path.join(HERE, "rsl_rl", "play_one_episode.py"), "--task", task,
               "--seed", str(args.seed), "--num_envs", str(args.num_envs), "--checkpoint", local, "--headless"]  # fmt: skip
        log = os.path.join(work, f"{task}.log")
        with open(log, "w") as f:
            try:
                subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=PROJECT,
                               timeout=1800)  # fmt: skip
            except subprocess.TimeoutExpired:
                pass  # results are printed before the (occasionally hanging) shutdown
        text = open(log, errors="ignore").read()
        r, s, c = RE_R.search(text), RE_S.search(text), RE_C.search(text)
        if not r:
            print(f"[OFFICIAL] !! {name} {task}: no result, see {log}", flush=True)
            continue
        row = {"name": name, "task": task, "checkpoint": ckpt, "seed": args.seed, "num_envs": args.num_envs,
               "reward_mean": r.group(1), "reward_std": r.group(2), "steps_mean": s.group(1) if s else "",
               "steps_std": s.group(2) if s else "", "completed": f"{c.group(1)}/{c.group(2)}" if c else ""}  # fmt: skip
        with open(out, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS).writerow(row)
        print(f"[OFFICIAL] {name} {task}: reward {float(r.group(1)):.2f} ± {float(r.group(2)):.2f}", flush=True)


if __name__ == "__main__":
    main()
