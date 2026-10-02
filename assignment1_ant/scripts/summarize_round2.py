"""Collect every evaluation CSV into one variant x {budget-matched 1000, final} table (results/summary_round2.md).

Only eval seeds 24 and 25 are used so round-1 (seeds 24-26) and round-2 (24-25) rows are comparable. A row's value
is the mean over training seeds (± sd across them) of the per-checkpoint mean over terrains.

    python scripts/summarize_round2.py
"""

import csv
import glob
import os
import re
from collections import defaultdict

import numpy as np

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANALYSIS = ["Flat", "Grid", "FlatIce", "GridHighMu", "GridLowMu", "GridNarrow", "GridTall", "GridWide", "Stairs",
            "Wave", "Boxes", "Rails"]  # fmt: skip
BLOCKS = ["Grid", "GridHighMu", "GridLowMu", "GridNarrow", "GridTall", "GridWide"]
REPORT = ["SlopedGrid", "Pyramids"]
ORDER = ["baseline", "baseline3000", "v1", "v2", "v3", "v4", "v5", "anopromo", "arand", "v2s", "v6c", "v10", "v11",
         "v12", "v13", "v14"]  # fmt: skip
LABEL = {"baseline": "baseline (평지)", "baseline3000": "baseline 3000", "v1": "V1 DR", "v2": "V2 차선 커리큘럼",
         "v3": "V3 파인튜닝", "v4": "V4 안정성 페널티", "v5": "V5", "anopromo": "A-noPromo", "arand": "A-randDiff",
         "v2s": "V2S (+계단)", "v6c": "V6′ CaT 자세", "v10": "V10 push+기울기", "v11": "V11 레벨별 마찰",
         "v12": "V12 접촉+히스토리", "v13": "V13 LSTM", "v14": "V14 외피 μ0.4"}  # fmt: skip


def parse_ckpt(name, source):
    """-> (variant, seed, stage) with stage '1000' or 'final'."""
    m = re.match(r"^([a-z0-9]+?)(?:_s(\d+))?(?:@(\d+))?$", name)
    variant, seed, it = m.group(1), m.group(2) or "42", m.group(3)
    if variant == "baseline":
        return variant, seed, "1000"  # the 1000-iteration baseline is both the budget-matched and the "final" point
    if it is not None:
        return variant, seed, "1000" if it == "1000" else "final"
    return variant, seed, "1000" if "budget1000" in source else "final"


def main():
    per = defaultdict(lambda: defaultdict(list))  # (variant, seed, stage) -> terrain -> [(reward, fall)]
    for path in glob.glob(os.path.join(PROJECT, "results", "eval_*.csv")):
        if "plane" in path or "quick" in path:
            continue
        with open(path) as f:
            for r in csv.DictReader(f):
                if r.get("seed") not in ("24", "25"):
                    continue
                key = parse_ckpt(r["checkpoint"], os.path.basename(path))
                terrain = r["task"].replace("Isaac-Ant-Eval-", "").replace("-v0", "")
                per[key][terrain].append((float(r["reward_mean"]), float(r["fall_rate"])))

    def ckpt_stats(terrains):
        def mean_of(names, idx):
            vals = [np.mean([x[idx] for x in terrains[t]]) for t in names if terrains.get(t)]
            return np.mean(vals) if len(vals) == len(names) else np.nan
        return {"all12": mean_of(ANALYSIS, 0), "blocks": mean_of(BLOCKS, 0), "flat": mean_of(["Flat"], 0),
                "flat_fall": mean_of(["Flat"], 1) * 100, "stairs": mean_of(["Stairs"], 0),
                "report": mean_of(REPORT, 0)}  # fmt: skip

    rows = []
    for variant in ORDER:
        for stage in ("1000", "final"):
            seeds = sorted({s for v, s, st in per if v == variant and st == stage})
            if not seeds:
                continue
            stats = [ckpt_stats(per[(variant, s, stage)]) for s in seeds]
            cell = {}
            for k in stats[0]:
                vals = np.array([st[k] for st in stats])
                vals = vals[~np.isnan(vals)]
                cell[k] = (f"{vals.mean():.1f}" + (f" ± {vals.std():.1f}" if len(vals) > 1 else "")) if len(vals) else "-"
            rows.append((LABEL.get(variant, variant), stage, ",".join(seeds), cell))

    head = "| 변형 | 예산 | 학습 seed | 12종 평균 | 블록 6종 | 평지 | 평지 낙상% | 계단 | 보고용 2종 |\n|---|---|---|---|---|---|---|---|---|\n"
    body = "".join(f"| {lbl} | {'1000' if st == '1000' else '최종'} | {sd} | {c['all12']} | {c['blocks']} | {c['flat']} | "
                   f"{c['flat_fall']} | {c['stairs']} | {c['report']} |\n" for lbl, st, sd, c in rows)  # fmt: skip
    text = ("# 2차 결과 요약 (eval seed 24·25, 256 envs, 원래 보상)\n\n"
            "값 = 학습 seed 평균 ± 표준편차. 12종 = 분석용 평가 지형, 블록 6종 = Grid 계열, 보고용 2종 = SlopedGrid·Pyramids.\n\n"
            + head + body)  # fmt: skip
    out = os.path.join(PROJECT, "results", "summary_round2.md")
    with open(out, "w") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    main()
