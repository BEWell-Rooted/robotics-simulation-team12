"""판정 규칙의 검정력 모의 (사전등록 docs/prereg_1029.md 5·8절).

짝 비교 한 번 = n 쌍(학습 seed 2개 × 평가 seed 수 × 100). 쌍마다 (둘 다 성공, A만, B만, 둘 다 실패)를
다항분포로 뽑는다: P(B만) = (d + Δ)/2, P(A만) = (d − Δ)/2, d = 불일치 비율(시작 상태로 짝지은 null 비교 32개 실측 0.227–0.299, 중앙 0.269 → 기본 0.27).
학습 seed 별 Δ 는 쌍을 반으로 나눠 계산. CI 는 짝 차이의 정규 근사(군집 부트스트랩 대신).
Holm 은 확증 팔 3개(A-T, B2, C-1mode). C-1mode 를 안 돌리면 p = 1, B2 는 따로 준 참 Δ 로 함께 뽑는다.

  python -I power_sim.py                    # 표 출력
  python -I power_sim.py --d 0.25 --n 400
"""
import argparse
from math import comb

import numpy as np


def mcnemar_two(b, c):
    n = b + c
    if n == 0:
        return 1.0
    lo = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * lo)


def draw(rng, n, d, delta):
    pb, pa = (d + delta) / 2, (d - delta) / 2
    pa, pb = max(pa, 0.0), max(pb, 0.0)
    # 쌍 순서: 앞 절반 = 학습 seed 101, 뒤 절반 = 111
    u = rng.random(n)
    only_a = u < pa
    only_b = (u >= pa) & (u < pa + pb)
    return only_a, only_b


def judge(only_a, only_b, n, min_pred, delta_floor=6.0):
    b, c = int(only_a.sum()), int(only_b.sum())
    dhat = 100.0 * (c - b) / n
    diff = only_b.astype(float) - only_a.astype(float)
    se = 100.0 * diff.std(ddof=1) / np.sqrt(n)
    h = n // 2
    seeds = [100.0 * (only_b[:h].sum() - only_a[:h].sum()) / h, 100.0 * (only_b[h:].sum() - only_a[h:].sum()) / h]
    return dict(p=mcnemar_two(b, c), dhat=dhat, up95=dhat + 1.96 * se, lo90=dhat - 1.645 * se, up90=dhat + 1.645 * se,
                seeds_pos=all(s > 0 for s in seeds), b=b, c=c)


def holm(ps, alpha=0.05):
    order = np.argsort(ps)
    m = len(ps)
    ok = [False] * m
    for rank, i in enumerate(order):
        if ps[i] < alpha / (m - rank):
            ok[i] = True
        else:
            break
    return ok


def simulate(delta_at, delta_b2, n, d, reps, seed, min_at, min_b2, floor):
    rng = np.random.default_rng(seed)
    cnt = dict(one=0, two=0, three_part=0, four=0)
    for _ in range(reps):
        ja = judge(*draw(rng, n, d, delta_at / 100), n, min_at)
        jb = judge(*draw(rng, n, d, delta_b2 / 100), n, min_b2)
        ok = holm([ja["p"], jb["p"], 1.0])
        if ok[0] and ja["dhat"] > floor and ja["seeds_pos"]:
            cnt["one"] += 1
        elif ja["up95"] < min_at or (ja["dhat"] <= -5 and ja["p"] < 0.05):
            cnt["two"] += 1
        elif ja["dhat"] >= 5:
            cnt["three_part"] += 1  # ③ 의 Δ 조건만 (묶음·상태 지표 조건은 더 엄격)
        else:
            cnt["four"] += 1
    return {k: v / reps for k, v in cnt.items()}


def equivalence(delta, n, d, reps, seed, margin):
    rng = np.random.default_rng(seed)
    eq = diff = 0
    for _ in range(reps):
        j = judge(*draw(rng, n, d, delta / 100), n, 0)
        if -margin < j["lo90"] and j["up90"] < margin:
            eq += 1
        elif j["p"] < 0.05:
            diff += 1
    return eq / reps, diff / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--d", type=float, default=0.27)
    ap.add_argument("--reps", type=int, default=4000)
    ap.add_argument("--min_at", type=float, default=4.0)
    ap.add_argument("--min_b2", type=float, default=4.0)
    ap.add_argument("--floor", type=float, default=6.09)
    ap.add_argument("--margin", type=float, default=7.0)
    args = ap.parse_args()
    print(f"n = {args.n} 쌍, 불일치 d = {args.d}, 반복 {args.reps}, A-T 최소 예측 +{args.min_at:g}, ① Δ 하한 {args.floor:g}, B2 참 Δ = 0 으로 함께 뽑음")
    print("| A-T 참 Δ (%p) | ① 효과 | ② 기각 | ③ 의 Δ 조건만 (③ 의 상한) | ④ 보류 (하한) |")
    print("| --- | --- | --- | --- | --- |")
    for da in (0, 3.5, 4, 5, 6, 8, 9.5):
        r = simulate(da, 0.0, args.n, args.d, args.reps, 1, args.min_at, args.min_b2, args.floor)
        print(f"| {da:+g} | {r['one']:.2f} | {r['two']:.2f} | {r['three_part']:.2f} | {r['four']:.2f} |")
    print()
    print(f"귀속 비교(짝 팔끼리, Holm 없음) — 동등 = 90% CI 가 ±{args.margin:g}%p 안, 차이 = 양측 p < 0.05")
    print("| 참 차이 (%p) | n = 200: 동등 · 차이 | n = 400: 동등 · 차이 |")
    print("| --- | --- | --- |")
    for dd in (0, 3, 5, 8):
        e2 = equivalence(dd, 200, args.d, args.reps, 2, args.margin)
        e4 = equivalence(dd, 400, args.d, args.reps, 2, args.margin)
        print(f"| {dd:g} | {e2[0]:.2f} · {e2[1]:.2f} | {e4[0]:.2f} · {e4[1]:.2f} |")


if __name__ == "__main__":
    main()
