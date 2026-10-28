"""두 평가(같은 평가 seed 42 → rollout 번호별 시작 상태가 같음)를 rollout 단위로 짝지어 비교한다.

classify_failures.py 가 만든 CSV 두 개(또는 두 묶음)를 받아:
  - 성공 수, 불일치 쌍 (A만 성공 : B만 성공), 정확 McNemar 단측·양측 p
  - 범주 이동표 (A 범주 → B 범주)
  - 진단 지표 중앙값 비교 (rot60_after_lift, box_disp_max 등)
seed 101·111 을 함께 보려면 각 쪽에 CSV 두 개씩 주면 (seed, rollout) 200쌍으로 묶는다(순서를 맞출 것).

  python -I paired_compare.py --a base_s101.csv base_s111.csv --b treat_s101.csv treat_s111.csv --label_a baseline --label_b A1

주의: 같은 체크포인트도 기계가 다르면 25/100 rollout 결과가 갈렸고(A6000 63 vs Spark 74, p 0.022),
같은 run 의 이웃 체크포인트끼리도 p < 0.05 가 나온다. 판정에는 같은 기계에서 잰 잡음 기준선(docs/diagnosis_1028.md 7절)을 함께 쓴다.
"""
import argparse
import csv
from collections import Counter
from math import comb

import numpy as np

CATS = ["success", "approach", "grasp", "transport", "place"]


def load(paths):
    rows = []
    for k, p in enumerate(paths):
        with open(p) as f:
            for r in csv.DictReader(f):
                r["_key"] = (k, r["id"])
                rows.append(r)
    return {r["_key"]: r for r in rows}


def mcnemar(b, c):
    n = b + c
    if n == 0:
        return 1.0, 1.0
    lo = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return lo, min(1.0, 2 * lo)


def med(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    args = ap.parse_args()
    A, B = load(args.a), load(args.b)
    keys = [k for k in A if k in B]
    sa = sum(int(A[k]["success"]) for k in keys)
    sb = sum(int(B[k]["success"]) for k in keys)
    only_a = sum(1 for k in keys if A[k]["success"] == "1" and B[k]["success"] == "0")
    only_b = sum(1 for k in keys if A[k]["success"] == "0" and B[k]["success"] == "1")
    p1, p2 = mcnemar(only_a, only_b)
    n = len(keys)
    print(f"짝 {n}개: {args.label_a} {sa} ({sa / n:.1%}) vs {args.label_b} {sb} ({sb / n:.1%}), 차 {100 * (sb - sa) / n:+.1f}%p")
    print(f"불일치 {args.label_a}만 성공 {only_a} : {args.label_b}만 성공 {only_b} · 정확 McNemar 단측 p = {p1:.3f}, 양측 p = {p2:.3f}")
    print()
    print("범주 이동 (행 = " + args.label_a + ", 열 = " + args.label_b + ")")
    m = Counter((A[k]["category"], B[k]["category"]) for k in keys)
    print("            " + "".join(f"{c:>10s}" for c in CATS))
    for ca in CATS:
        print(f"{ca:>10s}  " + "".join(f"{m.get((ca, cb), 0):10d}" for cb in CATS))
    print()
    fa = [A[k] for k in keys]
    fb = [B[k] for k in keys]
    for key in ("rot60_after_lift", "box_disp_max", "min_cheb_box_after_lift"):
        if key in fa[0] and key in fb[0]:
            print(f"{key:>26s} 중앙값: {med(fa, key):.3f} → {med(fb, key):.3f}")
    ca, cb = Counter(r["category"] for r in fa), Counter(r["category"] for r in fb)
    print("범주 합계: " + " · ".join(f"{c} {ca.get(c, 0)}→{cb.get(c, 0)}" for c in CATS))


if __name__ == "__main__":
    main()
