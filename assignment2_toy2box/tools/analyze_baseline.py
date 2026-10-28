"""classify_failures.py 의 CSV 여러 개(학습 seed 별 평가)를 모아 진단 그림·표를 만든다.

  python -I analyze_baseline.py --runs <csv> [<csv> ...] --demos <source annotated.hdf5> --out <폴더>
산출물:
  funnel.png        단계 도달 깔때기 (접근 → 파지 → 들기 → 상자 위 → 성공), run 별
  categories.png    실패 범주·세부 유형 막대
  init_scatter.png  시작 위치(트럭 xy)별 결과 + 원본 사람 시범 시작 위치
  yaw.png           트럭 yaw 구간별 성공률
  summary.md        표 (run 별 성공률, 평균 ± 표준편차, 범주 분포, 세부 유형)
"""
import argparse
import csv
import os
from collections import Counter, OrderedDict

import h5py
import numpy as np

CATS = ["success", "approach", "grasp", "transport", "place"]
KO = {"success": "성공", "approach": "접근", "grasp": "파지", "transport": "운반", "place": "배치"}
# 기본 팔레트 (dataviz reference, light): 실패 4단계 = 범주 슬롯 1–4, 성공 = 중립 회색
COL = {"approach": "#2a78d6", "grasp": "#eb6834", "transport": "#1baf7a", "place": "#eda100", "success": "#8a8984"}
MARK = {"approach": "o", "grasp": "s", "transport": "^", "place": "D", "success": "."}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def read_csv(p):
    with open(p) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v) if ("." in v or "e" in v.lower() or v == "nan") else int(v)
            except (ValueError, AttributeError):
                pass
    return rows


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.yaxis.label.set_color(INK2)
    ax.xaxis.label.set_color(INK2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--demos", nargs="*", default=[])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": ["Noto Sans CJK JP", "Noto Sans CJK KR", "DejaVu Sans"], "axes.unicode_minus": False})

    runs = OrderedDict()
    for i, p in enumerate(args.runs):
        lab = args.labels[i] if args.labels else os.path.splitext(os.path.basename(p))[0]
        runs[lab] = read_csv(p)

    md = ["| run | n | 성공 | 접근 | 파지 | 운반 | 배치 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    rates = []
    for lab, rows in runs.items():
        c = Counter(r["category"] for r in rows)
        n = len(rows)
        rates.append(c.get("success", 0) / n)
        md.append(f"| {lab} | {n} | " + " | ".join(f"{c.get(k, 0)}" for k in CATS) + " |")
    if len(rates) > 1:
        md.append("")
        md.append(f"성공률 평균 ± 표준편차(모표준편차, n={len(rates)}): {np.mean(rates):.1%} ± {np.std(rates):.1%}"
                  f" · 표본표준편차 {np.std(rates, ddof=1):.1%}")
    md.append("")
    md.append("| run | 범주 | 세부 유형 | 건수 |")
    md.append("| --- | --- | --- | --- |")
    for lab, rows in runs.items():
        sub = Counter((r["category"], r["subtype"]) for r in rows if r["category"] != "success")
        for (cat, s), v in sorted(sub.items(), key=lambda x: (CATS.index(x[0][0]), -x[1])):
            md.append(f"| {lab} | {KO[cat]} | {s} | {v} |")

    # 1. 깔때기
    stages = [("t_near", "손끝이 트럭 5 cm 안"), ("t_grasp", "grasp 신호"), ("t_lift", "들어 올림"),
              ("t_over", "상자 위"), ("t_success", "성공")]
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 2.8), squeeze=False)
    ramp = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
    for ax, (lab, rows) in zip(axes[0], runs.items()):
        n = len(rows)
        # 뒤 단계에 도달했으면 앞 단계도 지난 것으로 센다 (분류 규칙과 같게)
        def reached(r, i):
            return any((r[k] >= 0 and (k != "t_success" or r["success"])) for k, _ in stages[i:])
        vals = [sum(1 for r in rows if reached(r, i)) for i in range(len(stages))]
        y = np.arange(len(stages))[::-1]
        ax.barh(y, vals, color=ramp, height=0.62)
        for yy, v in zip(y, vals):
            ax.text(v + n * 0.01, yy, f"{v}", va="center", fontsize=8, color=INK)
        ax.set_yticks(y, [s for _, s in stages], fontsize=8)
        ax.set_xlim(0, n * 1.12)
        ax.set_title(lab, fontsize=9, color=INK)
        style(ax)
    fig.suptitle(f"단계 도달 rollout 수 (run 당 {len(next(iter(runs.values())))}회 중)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "funnel.png"), dpi=130)
    plt.close(fig)

    # 2. 범주 막대 (run 별 묶음)
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    w = 0.8 / len(runs)
    for i, (lab, rows) in enumerate(runs.items()):
        c = Counter(r["category"] for r in rows)
        xs = np.arange(len(CATS)) + i * w - 0.4 + w / 2
        bars = ax.bar(xs, [c.get(k, 0) for k in CATS], width=w * 0.92, color=[COL[k] for k in CATS],
                      alpha=1.0 if i == 0 else 0.6, edgecolor="white", linewidth=1)
        for x, k in zip(xs, CATS):
            ax.text(x, c.get(k, 0) + 0.8, f"{c.get(k, 0)}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(np.arange(len(CATS)), [KO[k] for k in CATS])
    ax.set_ylabel("rollout 수")
    ax.set_title("결과 범주 — 진한 막대 " + list(runs)[0] + (", 옅은 막대 " + ", ".join(list(runs)[1:]) if len(runs) > 1 else ""),
                 fontsize=8, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "categories.png"), dpi=130)
    plt.close(fig)

    # 3. 시작 위치 산점도
    demo_xy = []
    for p in args.demos:
        with h5py.File(p, "r") as f:
            for k in f["data"]:
                demo_xy.append(f["data"][k]["obs/toy_truck_positions"][0, :2])
    demo_xy = np.array(demo_xy) if demo_xy else None
    fig, axes = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 3.8), squeeze=False)
    for ax, (lab, rows) in zip(axes[0], runs.items()):
        for k in CATS:
            pts = [(r["init_truck_x"], r["init_truck_y"]) for r in rows if r["category"] == k]
            if pts:
                pts = np.array(pts)
                ax.scatter(pts[:, 1], pts[:, 0], s=40 if k != "success" else 30, marker=MARK[k] if k != "success" else "o",
                           facecolor=COL[k] if k != "success" else "none", edgecolor="white" if k != "success" else COL[k],
                           linewidth=1.0, label=f"{KO[k]} {len(pts)}", zorder=3)
        if demo_xy is not None:
            ax.scatter(demo_xy[:, 1], demo_xy[:, 0], marker="x", s=28, c=INK, linewidth=1.2, label="원본 시범", zorder=4)
        ax.set_xlabel("트럭 시작 y (base, m)")
        ax.set_ylabel("트럭 시작 x (base, m)")
        ax.invert_xaxis()
        ax.set_title(lab, fontsize=9, color=INK)
        ax.legend(fontsize=7, frameon=False, loc="best")
        style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "init_scatter.png"), dpi=130)
    plt.close(fig)

    # 4. yaw 구간별 성공률 (모든 run 합침)
    allr = [r for rows in runs.values() for r in rows]
    yaw = np.array([r["init_truck_yaw"] for r in allr])
    succ = np.array([r["success"] for r in allr])
    bins = np.arange(-180, 181, 45)
    idx = np.digitize(yaw, bins) - 1
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    rate = [succ[idx == i].mean() if (idx == i).any() else np.nan for i in range(len(bins) - 1)]
    cnt = [(idx == i).sum() for i in range(len(bins) - 1)]
    xs = (bins[:-1] + bins[1:]) / 2
    ax.bar(xs, rate, width=40, color="#2a78d6")
    for x, r_, c_ in zip(xs, rate, cnt):
        ax.text(x, (0 if np.isnan(r_) else r_) + 0.02, f"n={c_}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(bins)
    ax.set_xlabel("트럭 시작 yaw (deg, base)")
    ax.set_ylabel("성공률")
    ax.set_ylim(0, 1)
    ax.set_title("트럭 방향별 성공률 (모든 run 합침)", fontsize=9, color=INK)
    style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "yaw.png"), dpi=130)
    plt.close(fig)

    # 5. 같은 시작 상태(rollout 번호)에서 seed 간 결과 짝짓기 — 실패가 상태에 묶여 있는가
    labs = list(runs)
    if len(labs) >= 2:
        A_, B_ = {r["id"]: r for r in runs[labs[0]]}, {r["id"]: r for r in runs[labs[1]]}
        ids = [k for k in A_ if k in B_]
        both_f = sum(1 for k in ids if not A_[k]["success"] and not B_[k]["success"])
        one_f = sum(1 for k in ids if A_[k]["success"] != B_[k]["success"])
        both_s = sum(1 for k in ids if A_[k]["success"] and B_[k]["success"])
        pa = 1 - np.mean([A_[k]["success"] for k in ids]); pb = 1 - np.mean([B_[k]["success"] for k in ids])
        md += ["", f"### 같은 시작 상태 짝짓기 ({labs[0]} × {labs[1]}, n={len(ids)})", "",
               f"- 둘 다 실패 {both_f} · 한쪽만 실패 {one_f} · 둘 다 성공 {both_s}",
               f"- 실패가 서로 독립이라면 둘 다 실패 기대값 {pa * pb * len(ids):.1f} (실패율 {pa:.0%} × {pb:.0%})",
               "- 둘 다 실패한 rollout: " + ", ".join(f"{k}({A_[k]['category']}/{B_[k]['category']})" for k in ids
                                                if not A_[k]["success"] and not B_[k]["success"])]

    # 6. 시작 상태 구간별 성공률 (모든 run 합침) + 데이터 커버리지
    def feat(r):
        dx, dy = r["init_truck_x"] - r["init_box_x"], r["init_truck_y"] - r["init_box_y"]
        return dict(dist=float(np.hypot(dx, dy)), tx=r["init_truck_x"], ty=r["init_truck_y"],
                    yaw=r["init_truck_yaw"], yaw180=((r["init_truck_yaw"] + 90) % 180) - 90)
    F = [feat(r) for r in allr]
    def binned(name, edges):
        v = np.array([f[name] for f in F])
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (v >= lo) & (v < hi)
            if m.any():
                out.append(f"{lo:g}–{hi:g}: {succ[m].mean():.0%} (n={m.sum()})")
        return out
    md += ["", "### 시작 상태 구간별 성공률 (모든 run 합침)", ""]
    md.append("- 트럭–상자 xy 거리 (m): " + " · ".join(binned("dist", [0.15, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.65])))
    md.append("- 트럭 x (base, m): " + " · ".join(binned("tx", [0.4, 0.5, 0.6, 0.7, 0.8, 0.9])))
    md.append("- 트럭 y (base, m): " + " · ".join(binned("ty", [-0.32, -0.28, -0.24, -0.20, -0.16, -0.12])))
    md.append("- 트럭 yaw mod 180 (deg): " + " · ".join(binned("yaw180", [-90, -60, -30, 0, 30, 60, 90.1])))

    # 원본 시범과의 거리 (x, y 는 m, yaw 는 180° 대칭으로 0.1 m = 30° 환산)
    if args.demos:
        src = []
        for p in args.demos:
            with h5py.File(p, "r") as f:
                for k in f["data"]:
                    o = f["data"][k]["obs"]
                    t0, b0, q = o["toy_truck_positions"][0], o["box_positions"][0], o["toy_truck_orientations"][0]
                    w, x, y, z = q
                    yw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                    src.append((t0[0], t0[1], b0[0], b0[1], yw))
        src = np.array(src)
        def cov(r):
            dyaw = np.abs(((src[:, 4] - r["init_truck_yaw"] + 90) % 180) - 90) / 300.0
            d = np.sqrt((src[:, 0] - r["init_truck_x"]) ** 2 + (src[:, 1] - r["init_truck_y"]) ** 2
                        + (src[:, 2] - r["init_box_x"]) ** 2 + (src[:, 3] - r["init_box_y"]) ** 2 + dyaw ** 2)
            return float(d.min())
        cv = np.array([cov(r) for r in allr])
        q = np.quantile(cv, [0, 1 / 3, 2 / 3, 1])
        md.append("- 가장 가까운 원본 시범까지 거리 (트럭·상자 xy + yaw/180대칭, 3분위): " + " · ".join(
            f"{lo:.3f}–{hi:.3f}: {succ[(cv >= lo) & (cv <= hi)].mean():.0%} (n={((cv >= lo) & (cv <= hi)).sum()})"
            for lo, hi in zip(q[:-1], q[1:])))

    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
