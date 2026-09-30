"""Figures for the mesh-flat diagnosis (results/diag_meshflat.csv -> docs/figures/diag_*.png).

Checkpoints are grouped into variants (training-seed replicates averaged): Baseline, V1 (2 seeds), V2 (3 seeds).
Color = variant (fixed order from plot_results.py); line style = terrain (solid plane, dashed mesh).
"""

import argparse
import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from plot_results import ENTITY_COLORS, ENTITY_LABELS, GRID, SURFACE, TEXT, TEXT_2, style_axes  # noqa: E402

VARIANTS = ["baseline", "v1", "v2"]


def load(path):
    acc = defaultdict(list)  # (cond, variant) -> per-checkpoint means
    per_ckpt = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            per_ckpt[(r["cond"], r["checkpoint"])].append(r)
    for (cond, ckpt), rows in per_ckpt.items():
        m = {k: np.mean([float(r[k]) for r in rows]) for k in ("reward_mean", "fall_rate", "x_dist_mean")}
        acc[(cond, ckpt.split("_s")[0])].append(m)
    return acc


def agg(acc, cond, variant, key):
    xs = [m[key] for m in acc.get((cond, variant), [])]
    return (np.mean(xs), np.std(xs)) if xs else (np.nan, 0.0)


def mu_grid(acc, out):
    mus = sorted({float(c.split("mu=")[1]) for c, _ in acc if c.startswith("H3|")})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, ylabel, scale in ((axes[0], "fall_rate", "fall rate (%)", 100), (axes[1], "reward_mean", "episode reward", 1)):
        style_axes(ax)
        for v in VARIANTS:
            for terrain, ls, marker in (("plane", "-", "o"), ("mesh", (0, (4, 2)), "s")):
                ys = np.array([agg(acc, f"H3|{terrain}|mu={mu}", v, key) for mu in mus]) * [scale, scale]
                ax.plot(mus, ys[:, 0], color=ENTITY_COLORS[v], linestyle=ls, linewidth=2, marker=marker, markersize=6,
                        label=f"{ENTITY_LABELS[v]} — {'analytic plane' if terrain == 'plane' else 'mesh flat'}")  # fmt: skip
        ax.set_xlabel("effective contact friction μ (ground μ × robot μ 1.0)", color=TEXT_2, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
        ax.set_xticks(mus)
    axes[0].set_title("H3 — fall rate: analytic plane vs mesh flat", color=TEXT, fontsize=11, loc="left")
    axes[1].set_title("H3 — first-episode reward", color=TEXT, fontsize=11, loc="left")
    axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")


def threshold(acc, out):
    hs = [0.31, 0.28, 0.25, 0.20]
    cond = lambda h: "H3|mesh|mu=1.0" if h == 0.31 else f"H1|mesh|h={h}"  # noqa: E731
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, ylabel, scale in ((axes[0], "fall_rate", "fall rate (%)", 100), (axes[1], "x_dist_mean", "forward distance (m)", 1)):
        style_axes(ax)
        for v in VARIANTS:
            ys = np.array([agg(acc, cond(h), v, key)[0] for h in hs]) * scale
            ax.plot(hs, ys, color=ENTITY_COLORS[v], linewidth=2, marker="o", markersize=6, label=ENTITY_LABELS[v])
        ax.set_xlabel("torso-height termination threshold (m)  — task default 0.31", color=TEXT_2, fontsize=9)
        ax.set_ylabel(ylabel, color=TEXT_2, fontsize=9)
        ax.set_xticks(hs)
        ax.invert_xaxis()
    axes[0].set_title("H1 — lowering the threshold removes 'falls'...", color=TEXT, fontsize=11, loc="left")
    axes[1].set_title("...but not the lost distance: robots tip over", color=TEXT, fontsize=11, loc="left")
    axes[1].legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="center right")
    fig.text(0.01, -0.03, "Mesh flat, μ 1.0. V1 = mean of 2 training seeds, V2 = mean of 3; 2 eval seeds × 256 envs each.",
             color=TEXT_2, fontsize=8)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")


def surface(acc, out):
    """Same policies on different contact surfaces (all mu 1.0): the analytic plane is the outlier."""
    groups = [
        ("analytic plane", "H3|plane|mu=1.0"),
        ("box prims, flat", "BOXFLAT|primflat"),
        ("mesh twin, flat", "BOXFLAT|meshflat"),
        ("mesh flat\n(generator)", "H3|mesh|mu=1.0"),
        ("box prims,\nblocks ±0.08", "BOX|prim|n64"),
        ("mesh twin,\nblocks ±0.08", "BOX|mesh|n64"),
    ]
    fig, ax = plt.subplots(figsize=(12, 3.8), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    k = len(VARIANTS)
    width = 0.8 / k
    for i, v in enumerate(VARIANTS):
        ys = [agg(acc, c, v, "fall_rate")[0] * 100 for _, c in groups]
        x = np.arange(len(groups)) + (i - (k - 1) / 2) * width
        ax.bar(x, ys, width * 0.9, color=ENTITY_COLORS[v], label=ENTITY_LABELS[v], edgecolor=SURFACE, linewidth=1)
    ax.axvline(3.5, color=TEXT_2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_xticks(np.arange(len(groups)), [g for g, _ in groups], fontsize=9, color=TEXT)
    ax.set_ylabel("fall rate (%)", color=TEXT_2, fontsize=9)
    ax.set_title("Fall rate by contact surface (μ 1.0): box primitives behave like meshes — the analytic plane is the outlier",
                 color=TEXT, fontsize=11, loc="left")  # fmt: skip
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper right")
    fig.text(0.01, -0.03, "Flat: 256 envs × 2 eval seeds. Blocks: 64 envs × 4 eval seeds (PhysX pair-buffer limit with ~20k box prims).",
             color=TEXT_2, fontsize=8)  # fmt: skip
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="results/diag_meshflat.csv")
    parser.add_argument("--out", default="docs/figures")
    args = parser.parse_args()
    acc = load(args.csv)
    mu_grid(acc, os.path.join(args.out, "diag_mu_grid.png"))
    threshold(acc, os.path.join(args.out, "diag_threshold.png"))
    surface(acc, os.path.join(args.out, "diag_surface.png"))


if __name__ == "__main__":
    main()
