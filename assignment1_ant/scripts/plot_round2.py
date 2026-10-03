"""Round-2 figures (docs/figures/r2_*.png) from all evaluation CSVs (eval seeds 24·25, final models).

  r2_ladder.png   : 12-terrain mean per variant, grouped as the design story (environment → robot → combination),
                    one dot per training seed + mean bar
  r2_mu_sweep.png : skin friction mu vs 12-terrain mean (V1214R), single-seed points + 3-seed means
  r2_heatmap.png  : terrain x finalist reward

    python scripts/plot_round2.py
"""

import csv
import glob
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "docs", "figures")
KO = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
KO_B = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc")
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ANALYSIS = ["Flat", "Grid", "FlatIce", "GridHighMu", "GridLowMu", "GridNarrow", "GridTall", "GridWide", "Stairs",
            "Wave", "Boxes", "Rails"]  # fmt: skip
HEAT_TERRAINS = ANALYSIS + ["SlopedGrid", "Pyramids"]

# design-story groups: (label, variant key, color)
LADDER = [
    ("기준", [("baseline", "baseline\n1000 iter"), ("baseline3000", "baseline\n3000 iter")], "#9a8877"),
    ("지형 배치·커리큘럼", [("v1", "V1\nDR"), ("v2r", "V2R\n타일 랜덤"), ("arand", "A-rand\nDiff"),
                         ("anopromo", "A-no\nPromo"), ("v2", "V2\n커리큘럼")], "#2a78d6"),
    ("V2 + 자세·마찰 (축 1)", [("v6c", "V6′\nCaT"), ("v10", "V10\npush"), ("v11", "V11\n레벨 마찰")], "#1baf7a"),
    ("V2 + 로봇 (축 2)", [("v12", "V12\n접촉+\n히스토리"), ("v14", "V14\n외피 μ0.4")], "#eb6834"),
    ("결합", [("v12r", "V12R"), ("v1214nohist", "V1214\n−히스토리"), ("v1214r", "V1214R"), ("v1214b", "V1214B"), ("v1214bmu03", "V1214B\nμ0.3"),
            ("v1214", "V1214")], "#7a2e2e"),
    ("V1214B + 학습", [("v8", "V8\n좌우 대칭\n증강")], "#6b4bc4"),
]  # fmt: skip


def load():
    """(variant, seed) -> terrain -> mean reward over eval seeds 24/25 for the FINAL model of each run."""
    acc = defaultdict(lambda: defaultdict(list))
    for path in glob.glob(os.path.join(PROJECT, "results", "eval_*.csv")):
        name = os.path.basename(path)
        if any(k in name for k in ("plane", "quick", "budget1000")):
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("seed") not in ("24", "25"):
                    continue
                m = re.match(r"^([a-z0-9]+?)(?:_s(\d+))?(?:@(\d+))?$", r["checkpoint"])
                variant, seed, it = m.group(1), m.group(2) or "42", m.group(3)
                if it == "1000":
                    continue
                t = r["task"].replace("Isaac-Ant-Eval-", "").replace("-v0", "")
                acc[(variant, seed)][t].append(float(r["reward_mean"]))
    return acc


def run_mean(terrains, names=ANALYSIS):
    vals = [np.mean(terrains[t]) for t in names if terrains.get(t)]
    return np.mean(vals) if len(vals) == len(names) else np.nan


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def ladder(acc):
    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    x, ticks, labels = 0, [], []
    for group, items, color in LADDER:
        start = x
        for key, label in items:
            seeds = [run_mean(t) for (v, s), t in acc.items() if v == key]
            seeds = [v for v in seeds if not np.isnan(v)]
            if not seeds:
                continue
            ax.bar(x, np.mean(seeds), 0.7, color=color, alpha=0.85, edgecolor=SURFACE)
            ax.scatter([x] * len(seeds), seeds, color=TEXT, s=10, zorder=3)
            ax.text(x, max(seeds) + 2, f"{np.mean(seeds):.1f}", ha="center", fontsize=8, color=TEXT)
            ticks.append(x)
            labels.append(f"{label}\n(n={len(seeds)})")
            x += 1
        ax.text((start + x - 1) / 2, -22, group, ha="center", fontproperties=KO_B, fontsize=9, color=color)
        x += 0.6
    ax.set_xticks(ticks, labels, fontproperties=KO, fontsize=7.5)
    ax.set_ylabel("12종 평균 reward (최종 모델)", fontproperties=KO, color=TEXT_2)
    ax.set_ylim(0, max(ax.get_ylim()[1], 90))
    ax.set_title("변형별 자체 평가 12종 평균 (막대 = 학습 seed 평균, 점 = seed별)", fontproperties=KO_B, fontsize=12,
                 loc="left", color=TEXT)  # fmt: skip
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    fig.savefig(os.path.join(OUT, "r2_ladder.png"), facecolor=SURFACE)
    plt.close(fig)


def mu_sweep(acc):
    keys = {0.2: "v1214rmu02", 0.3: "v1214rmu03", 0.4: "v1214r", 0.6: "v1214rmu06"}
    fig, ax = plt.subplots(figsize=(6.5, 4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    style(ax)
    xs, means = [], []
    for mu, key in keys.items():
        vals = [run_mean(t) for (v, s), t in acc.items() if v == key]
        vals = [v for v in vals if not np.isnan(v)]
        if not vals:
            continue
        ax.scatter([mu] * len(vals), vals, color="#7a2e2e", s=18, alpha=0.6, zorder=3)
        xs.append(mu)
        means.append(np.mean(vals))
        ax.text(mu, np.mean(vals) + 1.2, f"{np.mean(vals):.1f} (n={len(vals)})", ha="center", fontsize=8, color=TEXT)
    ax.plot(xs, means, color="#7a2e2e", linewidth=2, marker="o")
    ax.set_xlabel("로봇 외피 마찰 μ (combine = min)", fontproperties=KO, color=TEXT_2)
    ax.set_ylabel("12종 평균 reward", fontproperties=KO, color=TEXT_2)
    ax.set_title("외피 마찰 sweep (V1214R)", fontproperties=KO_B, fontsize=12, loc="left", color=TEXT)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "r2_mu_sweep.png"), facecolor=SURFACE)
    plt.close(fig)


def heatmap(acc):
    rows = [("baseline", "baseline"), ("v1", "V1 DR"), ("v2", "V2 커리큘럼"), ("v12", "V12"), ("v14", "V14"),
            ("v1214r", "V1214R"), ("v1214b", "V1214B"), ("v1214bmu03", "V1214B\nμ0.3"), ("v1214", "V1214"), ("v8", "V8 대칭")]  # fmt: skip
    data, labels = [], []
    for key, label in rows:
        runs = [t for (v, s), t in acc.items() if v == key]
        if not runs:
            continue
        data.append([np.mean([np.mean(r[t]) for r in runs if r.get(t)]) if any(r.get(t) for r in runs) else np.nan
                     for t in HEAT_TERRAINS])  # fmt: skip
        labels.append(f"{label} (n={len(runs)})")
    data = np.array(data)
    fig, ax = plt.subplots(figsize=(13, 0.55 * len(labels) + 1.6), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    im = ax.imshow(data, cmap="YlOrBr", vmin=0, vmax=110, aspect="auto")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center", fontsize=8,
                        color="#ffffff" if data[i, j] > 70 else TEXT)  # fmt: skip
    ax.set_xticks(range(len(HEAT_TERRAINS)), HEAT_TERRAINS, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontproperties=KO, fontsize=9)
    ax.set_title("지형별 reward (최종 모델, 학습 seed 평균) — 오른쪽 2열은 보고용 held-out", fontproperties=KO_B,
                 fontsize=12, loc="left")  # fmt: skip
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "r2_heatmap.png"), facecolor=SURFACE)
    plt.close(fig)


def main():
    acc = load()
    ladder(acc)
    mu_sweep(acc)
    heatmap(acc)
    print("[PLOT] r2_ladder.png r2_mu_sweep.png r2_heatmap.png")


if __name__ == "__main__":
    main()
