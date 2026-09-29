"""Figures for the report: eval-suite bars (reward, fall rate) and training curves.

Plain Python (matplotlib + tensorboard), no Isaac Sim. Run from assignment1_ant/:
    python scripts/plot_results.py --csv results/eval_full_b_v1_v3.csv results/eval_full_v2_v4.csv \
        --runs v1=logs/rsl_rl/ant_rough/<v1_run> v2=... --out docs/figures

Colors follow the entity (checkpoint), in a fixed categorical order, so a checkpoint keeps its color in every
figure regardless of which subset is plotted.
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# reference categorical palette (light), fixed order: slot i always belongs to the same checkpoint
ENTITY_COLORS = {
    "baseline": "#2a78d6",
    "v1": "#eb6834",
    "v2": "#1baf7a",
    "v3": "#eda100",
    "v4": "#e87ba4",
    "v5": "#008300",
    "final": "#4a3aa7",
}
ENTITY_LABELS = {
    "baseline": "Baseline (flat)",
    "v1": "V1 DR",
    "v2": "V2 Curriculum",
    "v3": "V3 Fine-tune",
    "v4": "V4 DR + stability",
    "v5": "V5",
    "final": "Final",
}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

# the 7 Isaac-Ant-v0 reward terms: training curves sum only these, so V4's extra penalties don't skew comparison
BASE_TERMS = ["progress", "alive", "upright", "move_to_target", "action_l2", "energy", "joint_pos_limits"]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def load_eval(csv_paths):
    acc = defaultdict(list)  # (task, split, ckpt) -> rows
    for p in csv_paths:
        with open(p) as f:
            for row in csv.DictReader(f):
                acc[(row["task"], row["split"], row["checkpoint"])].append(row)
    return acc


def grouped_bars(acc, metric, ylabel, title, out_path, ckpts, percent=False):
    tasks = sorted({(t, s) for t, s, _ in acc}, key=lambda ts: (ts[1] != "ID", ts[0]))
    names = [t.replace("Isaac-Ant-Eval-", "").replace("-v0", "") for t, _ in tasks]
    n, k = len(tasks), len(ckpts)
    width = 0.8 / k
    fig, ax = plt.subplots(figsize=(12, 4.2), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    for i, c in enumerate(ckpts):
        means, errs = [], []
        for t, s in tasks:
            vals = [float(r[metric]) for r in acc.get((t, s, c), [])]
            means.append(np.mean(vals) * (100 if percent else 1) if vals else np.nan)
            errs.append(np.std(vals) * (100 if percent else 1) if len(vals) > 1 else 0)
        x = np.arange(n) + (i - (k - 1) / 2) * width
        ax.bar(x, means, width * 0.9, color=ENTITY_COLORS[c], label=ENTITY_LABELS[c],
               edgecolor=SURFACE, linewidth=1)  # fmt: skip
        ax.errorbar(x, means, yerr=errs, fmt="none", ecolor=TEXT_2, elinewidth=0.8, capsize=0)
    # ID / OOD separator
    n_id = sum(1 for _, s in tasks if s == "ID")
    ax.axvline(n_id - 0.5, color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.text(n_id / 2 - 0.5, 1.02, "in-distribution", transform=ax.get_xaxis_transform(), ha="center",
            color=TEXT_2, fontsize=9)  # fmt: skip
    ax.text((n_id + n) / 2 - 0.5, 1.02, "unseen (out-of-distribution)", transform=ax.get_xaxis_transform(),
            ha="center", color=TEXT_2, fontsize=9)  # fmt: skip
    ax.set_xticks(np.arange(n), names, rotation=0, fontsize=9, color=TEXT)
    ax.set_ylabel(ylabel, color=TEXT_2, fontsize=10)
    ax.set_title(title, color=TEXT, fontsize=12, loc="left", pad=18)
    ax.legend(frameon=False, fontsize=9, ncol=k, loc="upper left", bbox_to_anchor=(0, -0.08), labelcolor=TEXT)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def load_curve(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ev = sorted(glob.glob(os.path.join(run_dir, "events.out.tfevents.*")))
    ea = EventAccumulator(ev[-1], size_guidance={"scalars": 0})
    ea.Reload()
    steps = None
    total = None
    for term in BASE_TERMS:
        sc = ea.Scalars(f"Episode_Reward/{term}")
        s = np.array([e.step for e in sc])
        v = np.array([e.value for e in sc])
        if total is None:
            steps, total = s, v
        else:
            total = total + np.interp(steps, s, v)
    ep_len = ea.Scalars("Train/mean_episode_length")
    return steps, total, np.array([e.step for e in ep_len]), np.array([e.value for e in ep_len])


def smooth(y, k=25):
    if len(y) < k:
        return y
    kernel = np.ones(k) / k
    return np.convolve(np.pad(y, (k // 2, k - 1 - k // 2), mode="edge"), kernel, mode="valid")


def training_curves(runs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        style_axes(ax)
    for name, run_dir in runs:
        steps, total, ls, lv = load_curve(run_dir)
        c = ENTITY_COLORS[name]
        axes[0].plot(steps, smooth(total), color=c, linewidth=2, label=ENTITY_LABELS[name])
        axes[1].plot(ls, smooth(lv), color=c, linewidth=2, label=ENTITY_LABELS[name])
    axes[0].set_title("Baseline reward terms per episode (sum of 7 terms, / s)", color=TEXT, fontsize=11, loc="left")
    axes[1].set_title("Mean episode length (steps, max 960)", color=TEXT, fontsize=11, loc="left")
    for ax in axes:
        ax.set_xlabel("iteration", color=TEXT_2, fontsize=9)
    # end-of-line direct labels collide when curves converge, so both panels carry a legend instead
    for ax in axes:
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower right")
    fig.text(0.01, -0.02, "Training-terrain metrics: V2 is measured on its own curriculum lanes, so its level is not"
             " directly comparable. V3 resumes from the flat baseline at iteration 999.", color=TEXT_2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print(f"[PLOT] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--ckpts", nargs="+", default=None, help="checkpoint names to plot, in order")
    parser.add_argument("--runs", nargs="*", default=[], help="name=run_dir for training curves")
    parser.add_argument("--out", default="docs/figures")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    acc = load_eval(args.csv)
    present = {c for _, _, c in acc}
    ckpts = args.ckpts or [c for c in ENTITY_COLORS if c in present]
    grouped_bars(acc, "reward_mean", "episode reward (first episode)",
                 "Episode reward on self-made evaluation terrains (mean ± sd over 3 seeds, 256 envs each)",
                 os.path.join(args.out, "eval_reward.png"), ckpts)  # fmt: skip
    grouped_bars(acc, "fall_rate", "fall rate (%)", "Early termination (fall) rate on evaluation terrains",
                 os.path.join(args.out, "eval_fall_rate.png"), ckpts, percent=True)  # fmt: skip
    if args.runs:
        training_curves([r.split("=", 1) for r in args.runs], os.path.join(args.out, "training_curves.png"))


if __name__ == "__main__":
    main()
