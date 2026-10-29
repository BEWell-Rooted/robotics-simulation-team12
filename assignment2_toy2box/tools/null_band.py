"""null 띠 계산 (사전등록 docs/prereg_1029.md 5절).

같은 기계·같은 run 의 이웃 체크포인트 비교(paired_compare.py --json 출력)를 여러 개 받아,
지표마다 |차| 를 rollout 100개당 값으로 바꾼 뒤 최댓값을 null 띠로 삼는다.
처치 팔은 같은 지표의 차(100개당)가 이 띠 밖이고 예측 방향일 때만 "띠 밖" 으로 본다.

  python -I null_band.py null_a6000.json null_spark.json --ref machine_a6000_vs_spark.json --md out.md

--ref 로 준 비교(예: 기계 간)는 표에만 싣고 띠 계산에는 넣지 않는다.
"""
import argparse
import json
import os

ROWS = [("success_pp", "성공률 Δ (%p)"), ("bundle", "표적 묶음(운반+배치) 건수"), ("pre_lift_fail", "들기 전 실패 건수"),
        ("push_carry", "push_carry 건수"), ("collision", "귀속 충돌 건수"), ("self_release", "귀속 스스로 놓음 건수 (C3 판정 묶음)"), ("hold", "HOLD 건수"), ("low_approach", "낮은 접근(approach_z < 0.15) 건수"), ("box_push", "상자 밂 > 5 cm 건수"),
        ("rot60_state", "rot60_state 중앙값 (°)"), ("carry_yaw_net", "carry_yaw_net 중앙값 (°)"), ("xy30", "xy30 중앙값 (m)"),
        ("approach_z", "approach_z 중앙값 (m)"), ("carry_speed", "carry_speed 중앙값 (m/step)"), ("carry_steps", "carry_steps 중앙값 (step)")]


def diffs(res):
    """지표별 차. 건수는 rollout 100개당으로 바꾼다."""
    n = res["n"]
    out = {"success_pp": res["delta_pp"]}
    if "bundle_a" in res:
        out["bundle"] = 100.0 * (res["bundle_b"] - res["bundle_a"]) / n
    for k, v in res.get("mediators", {}).items():
        out[k] = 100.0 * v["diff"] / n if v["kind"] != "median" else v["diff"]
    return out


def summary(D, band):
    """지표별 |차| 분포와, null 비교를 처치로 가정했을 때 규칙 통과 수 (200쌍 기준 근사)."""
    import numpy as np
    use = [(n, r, d) for n, r, d, u in D if u]
    ns = [r["n"] for _, r, _ in use]
    lines = [f"null 비교 {len(use)}개 (각 {min(ns)}–{max(ns)}쌍, 시작 상태 짝). 건수 지표는 짝 100개당 차 (100 × 차 / n).", "",
             "| 지표 | **null 띠 = 최대 \\|차\\|** | 90% | 중앙 | 최대가 나온 비교 |", "| --- | --- | --- | --- | --- |"]
    for key, label in ROWS:
        vals = [(abs(d[key]), n) for n, _, d in use if key in d]
        if not vals:
            continue
        a = np.array([v for v, _ in vals])
        top = max(vals)[1]
        lines.append(f"| {label} | **{band[key]:.3g}** | {np.percentile(a, 90):.3g} | {np.median(a):.3g} | {top} |")
    lines += ["", "**null 비교를 처치로 가정한 규칙 점검 (짝 약 200개, 사전등록 5절 문턱):**", ""]
    floor = max(6.0, band["success_pp"])
    r1 = [n for n, r, d in use if r["p_two"] < 0.05 and r["delta_pp"] > floor and all(v > 0 for v in r["delta_by_train_seed"].values())]
    r3a = [n for n, r, d in use if r["delta_pp"] >= 5 and r.get("bundle_p_one", 1) < 0.10]
    lines.append(f"- ① (보정 전 양측 p < 0.05 · Δ > {floor:.2f} · seed 별 Δ > 0; 문턱이 이 표본의 최댓값이라 구성상 0): {len(r1)}/{len(use)}")
    lines.append(f"- ③ 앞부분 (Δ ≥ +5 · 표적 묶음 단측 p < 0.10): {len(r3a)}/{len(use)}" + (f" — {', '.join(r3a)}" if r3a else ""))
    for name, mp in (("A-T", 4.0), ("B2", 4.0)):
        r2 = [n for n, r, d in use if r["ci95"][1] < mp]
        lines.append(f"- 참 Δ = 0 일 때 ② 첫 조건(95% 상한 < +{mp:g})으로 {name} 가 기각되는 비율 (바라는 방향, 200쌍): {len(r2)}/{len(use)}")
    deltas = np.array([r["delta_pp"] for _, r, _ in use])
    lines.append(f"- 성공률 Δ 분포: 최소 {deltas.min():+.1f} · 중앙 {np.median(deltas):+.1f} · 최대 {deltas.max():+.1f} (%p); "
                 f"90% CI 폭 {min(r['ci90'][1] - r['ci90'][0] for _, r, _ in use):.1f}–{max(r['ci90'][1] - r['ci90'][0] for _, r, _ in use):.1f}%p; "
                 f"불일치 쌍 {min(r['only_a'] + r['only_b'] for _, r, _ in use)}–{max(r['only_a'] + r['only_b'] for _, r, _ in use)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nulls", nargs="+", help="null 비교 json (paired_compare.py --json)")
    ap.add_argument("--ref", nargs="*", default=[], help="표에만 싣는 참고 비교 json")
    ap.add_argument("--md", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--summary", action="store_true", help="비교가 많을 때: 지표별 최대·90%·중앙 |차| 와 규칙 점검만")
    args = ap.parse_args()
    cols = [(p, json.load(open(p)), True) for p in args.nulls] + [(p, json.load(open(p)), False) for p in args.ref]
    D = [(os.path.basename(p).replace(".json", ""), r, diffs(r), use) for p, r, use in cols]
    band = {}
    for key, _ in ROWS:
        vals = [abs(d[key]) for _, _, d, use in D if use and key in d]
        if vals:
            band[key] = max(vals)
    if args.summary:
        txt = summary(D, band)
        print(txt)
        if args.md:
            with open(args.md, "w") as f:
                f.write(txt + "\n")
        if args.json:
            with open(args.json, "w") as f:
                json.dump(dict(band=band, n_comparisons=len(D), nulls=args.nulls, ref=args.ref), f, ensure_ascii=False, indent=2)
        return
    lines = ["| 지표 (건수는 rollout 100개당) | " + " | ".join(f"{name} (n={r['n']}){'' if use else ' · 참고'}" for name, r, _, use in D)
             + " | **null 띠 = 최대 \\|차\\|** |",
             "| --- |" + " --- |" * (len(D) + 1)]
    for key, label in ROWS:
        cells = []
        for _, _, d, _ in D:
            cells.append(f"{d[key]:+.3g}" if key in d else "—")
        b = f"**{band[key]:.3g}**" if key in band else "—"
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {b} |")
    lines.append("")
    lines.append("성공 수: " + " · ".join(f"{name} {r['label_a']} {r['succ_a']} → {r['label_b']} {r['succ_b']} "
                                       f"(McNemar 양측 p = {r['p_two']:.3f})" for name, r, _, _ in D))
    txt = "\n".join(lines)
    print(txt)
    if args.md:
        with open(args.md, "w") as f:
            f.write(txt + "\n")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(band=band, nulls=args.nulls, ref=args.ref), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
