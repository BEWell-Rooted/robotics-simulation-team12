"""두 평가를 시작 상태 단위로 짝지어 비교한다 (사전등록 docs/prereg_1029.md 의 판정 도구).

같은 평가 seed 에서 시작 상태(트럭·상자 자세)의 순서는 정책과 무관하다. 다만 정책이 트럭을 떨어뜨려
환경이 step 안에서 끝나면(end_reason = dropped) 자동 reset 이 난수를 하나 더 써서, 그 뒤 rollout 은
시작 상태가 한 칸씩 밀린다(rollout 번호가 같아도 시작 상태가 다르다). 그래서 **(학습 seed, 평가 seed,
시작 상태 6값 init_truck_x·y·yaw·init_box_x·y·yaw) 로 짝짓는다**. 학습·평가 seed 는 파일 이름의
`_s<학습seed>_` 와 `seed<평가seed>` 에서 읽는다(없으면 인자 순서 번호).
짝 없는 rollout 은 (학습 seed, 평가 seed) 묶음마다 양쪽 dropped 수의 합까지만 허용하고, 넘으면 멈춘다.

출력:
  - 성공 수, Δ(%p), 불일치 쌍(A만 성공 : B만 성공), 정확 McNemar 양측 p 와 사전 방향(B > A) 단측 p
  - Δ 의 95%·90% 신뢰구간: 시작 상태(평가 seed, 시작 상태) 군집 부트스트랩 — 같은 시작 상태의 두 학습 seed 결과를 함께 뽑는다
  - 학습 seed 별 Δ
  - --bundle transport,place : 표적 실패 묶음(범주 목록)에 대한 rollout 별 0/1 McNemar (사전 방향 = B 에서 줄어듦)
    --bundle_col 로 묶음을 정할 열을 바꾼다 (기본 category; 분류 CSV 에 없으면 .carry.csv 에서 찾음). 예:
      C3 판정 묶음  --bundle_col attr_ii --bundle self_release          (들기 이후 실패 중 비충돌 스스로 놓음)
      HOLD          --bundle_col subtype --bundle no_release,in_box_not_open,stall
      음성 대조     --bundle approach,grasp
  - --min_d0 0.35 : 시작 트럭–상자 거리 ≥ 값인 짝만 (하위 분석; 짝은 시작 상태가 같으므로 양쪽 값이 같다)
  - 범주 이동표, 기전 지표 중앙값
  - 상태 매개 지표(prereg 4절): 각 CSV 옆에 <이름>.carry.csv(carry_metrics.py 출력)가 있으면
    push_carry 건수 · 낮은 접근(approach_z < 0.15 m) 건수 · rot60_state 등 중앙값 · 들기 전 실패 수의 A → B 차
  - --json : 위 값을 json 으로 (holm 보정·보고서용)

  python -I paired_compare.py --a base_kgh500_s101_e2000_seed42.csv base_kgh500_s111_e2000_seed42.csv \
      --b base_translate500_s101_e2000_seed42.csv base_translate500_s111_e2000_seed42.csv \
      --label_a baseline --label_b A-T --bundle transport,place --json out.json
"""
import argparse
import csv
import json
import os
import re
from collections import Counter
from math import comb

import numpy as np

CATS = ["success", "approach", "grasp", "transport", "place"]


def seeds_from_name(path, k):
    base = os.path.basename(path)
    m_t = re.search(r"_s(\d+)_", base)
    m_e = re.search(r"seed(\d+)", base)
    return (int(m_t.group(1)) if m_t else k, int(m_e.group(1)) if m_e else -1)


STATE_COLS = ("init_truck_x", "init_truck_y", "init_truck_yaw", "init_box_x", "init_box_y", "init_box_yaw")


def load(paths):
    """분류 CSV 들을 (학습 seed, 평가 seed, 시작 상태) 키로 읽는다.
    반환: rows{키: 행}, idmap{(학습 seed, 평가 seed, rollout id): 키}, drops{(학습 seed, 평가 seed): dropped 수}"""
    out, idmap, drops = {}, {}, Counter()
    for k, p in enumerate(paths):
        ts, es = seeds_from_name(p, k)
        with open(p) as f:
            for r in csv.DictReader(f):
                key = (ts, es, tuple(r[c] for c in STATE_COLS))
                if key in out:
                    raise SystemExit(f"같은 파일 묶음에 같은 시작 상태가 두 번: {key} ({p})")
                out[key] = r
                idmap[(ts, es, r["id"])] = key
                drops[(ts, es)] += int(r.get("end_reason") == "dropped")
    return out, idmap, drops


def mcnemar(b, c):
    """정확 McNemar. b = A만 성공, c = B만 성공. (양측, 단측 B>A)"""
    n = b + c
    if n == 0:
        return 1.0, 1.0
    lo = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    two = min(1.0, 2 * lo)
    one = sum(comb(n, i) for i in range(c, n + 1)) / 2 ** n  # P(X >= c), X ~ Bin(n, 0.5)
    return two, one


def cluster_ci(A, B, keys, n_boot=10000, seed=0):
    clusters = sorted({(k[1], k[2]) for k in keys})
    idx = {c: [k for k in keys if (k[1], k[2]) == c] for c in clusters}
    diff = {c: sum(int(B[k]["success"]) - int(A[k]["success"]) for k in idx[c]) for c in clusters}
    cnt = {c: len(idx[c]) for c in clusters}
    rng = np.random.default_rng(seed)
    cl = np.array(clusters, dtype=object)
    d = np.array([diff[c] for c in clusters], float)
    m = np.array([cnt[c] for c in clusters], float)
    picks = rng.integers(0, len(cl), size=(n_boot, len(cl)))
    stats = 100.0 * d[picks].sum(1) / m[picks].sum(1)
    return {q: float(np.percentile(stats, q)) for q in (2.5, 5, 95, 97.5)}


def med(rows, key):
    v = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
    return float(np.median(v)) if v else float("nan")


# (지표, 종류): count = 200(400) rollout 중 1 인 건수, median = 값이 있는 rollout 의 중앙값
# low_approach = approach_z < 0.15 m 인 rollout 수 (baseline, approach_z 가 있는 rollout 중: 성공 9% · 들기 이후 실패 38% · 충돌 65%
#   — 중앙값은 성공이 대부분이라 둔하다)
# collision·self_release = 귀속 (ii) 건수 (들기 이후 실패만), hold = 안 놓음류 세부 유형 건수 (C3 판정용)
MEDIATORS = [("push_carry", "count"), ("low_approach", "low"), ("collision", "attr"), ("self_release", "attr"), ("hold", "hold"),
             ("approach_z", "median"), ("rot60_state", "median"),
             ("carry_yaw_net", "median"), ("xy30", "median"), ("carry_speed", "median"), ("carry_steps", "median"),
             ("box_push", "count")]


LOW_APPROACH_Z = 0.15
HOLD_SUBTYPES = ("no_release", "in_box_not_open", "stall")  # 안 놓음류 (트랙 C)


def load_carry(paths, idmap):
    """각 분류 CSV 의 짝 <이름>.carry.csv 를 분류 CSV 와 같은 (시작 상태) 키로 읽는다. 하나라도 없으면 None."""
    out = {}
    for k, p in enumerate(paths):
        cp = p[:-4] + ".carry.csv" if p.endswith(".csv") else p + ".carry.csv"
        if not os.path.exists(cp):
            return None
        ts, es = seeds_from_name(p, k)
        with open(cp) as f:
            for r in csv.DictReader(f):
                key = idmap.get((ts, es, r["id"]))
                if key is not None:
                    out[key] = r
    return out


def mediator(rows, key, kind):
    if kind == "count":
        return float(sum(1 for r in rows if r.get(key) in ("1", 1)))
    if kind == "attr":
        return float(sum(1 for r in rows if r.get("attr_ii") == key))
    if kind == "hold":
        return float(sum(1 for r in rows if r.get("subtype") in HOLD_SUBTYPES))
    if kind == "low":
        return float(sum(1 for r in rows if r.get("approach_z") not in (None, "", "nan", "-")
                         and float(r["approach_z"]) < LOW_APPROACH_Z))
    v = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan", "-")]
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", nargs="+", required=True)
    ap.add_argument("--b", nargs="+", required=True)
    ap.add_argument("--label_a", default="A")
    ap.add_argument("--label_b", default="B")
    ap.add_argument("--bundle", default="transport,place", help="표적 실패 묶음 범주(쉼표). 빈 문자열이면 생략")
    ap.add_argument("--json", default=None)
    ap.add_argument("--allow_partial", action="store_true", help="짝이 다 맞지 않아도 계속(기본은 멈춤)")
    ap.add_argument("--bundle_col", default="category", help="묶음을 정할 열 (분류 CSV, 없으면 .carry.csv)")
    ap.add_argument("--min_d0", type=float, default=None, help="시작 트럭–상자 거리 ≥ 값인 rollout 만 비교")
    ap.add_argument("--require_carry", action="store_true", help=".carry.csv 가 없으면 멈춤 (판정용)")
    args = ap.parse_args()
    (A, ida, dra), (B, idb, drb) = load(args.a), load(args.b)
    keys = sorted(k for k in A if k in B)
    groups = sorted({k[:2] for k in A} | {k[:2] for k in B})
    unmatched = {}
    for g in groups:
        ua = sum(1 for k in A if k[:2] == g and k not in B)
        ub = sum(1 for k in B if k[:2] == g and k not in A)
        unmatched[g] = (ua, ub, dra[g], drb[g])
        allow = dra[g] + drb[g]
        if (ua > allow or ub > allow or not any(k[:2] == g for k in keys)) and not args.allow_partial:
            raise SystemExit(f"짝이 맞지 않는다 (학습 seed {g[0]}, 평가 seed {g[1]}): A 짝 없음 {ua} · B 짝 없음 {ub} · "
                             f"dropped A {dra[g]} + B {drb[g]} — 학습·평가 seed 와 rollout 수를 확인")
    if any(u[0] or u[1] for u in unmatched.values()):
        print("[짝] 시작 상태가 밀려 짝이 없는 rollout: " + " · ".join(
            f"s{g[0]}/seed{g[1]} A {u[0]} · B {u[1]} (dropped {u[2]}·{u[3]})" for g, u in unmatched.items() if u[0] or u[1]))
    if args.min_d0 is not None:
        def d0(r):
            return float(np.hypot(float(r["init_truck_x"]) - float(r["init_box_x"]), float(r["init_truck_y"]) - float(r["init_box_y"])))
        keys = [k for k in keys if d0(A[k]) >= args.min_d0]
        print(f"[하위 분석] 시작 트럭–상자 거리 ≥ {args.min_d0} m 인 rollout {len(keys)}개만")
    n = len(keys)
    sa = sum(int(A[k]["success"]) for k in keys)
    sb = sum(int(B[k]["success"]) for k in keys)
    only_a = sum(1 for k in keys if A[k]["success"] == "1" and B[k]["success"] == "0")
    only_b = sum(1 for k in keys if A[k]["success"] == "0" and B[k]["success"] == "1")
    p2, p1 = mcnemar(only_a, only_b)
    ci = cluster_ci(A, B, keys)
    delta = 100.0 * (sb - sa) / n
    seeds = sorted({(k[0]) for k in keys})
    per_seed = {}
    for s in seeds:
        ks = [k for k in keys if k[0] == s]
        per_seed[s] = 100.0 * (sum(int(B[k]["success"]) for k in ks) - sum(int(A[k]["success"]) for k in ks)) / len(ks)
    evs = sorted({k[1] for k in keys})
    print(f"짝 {n}개 (학습 seed {seeds}, 평가 seed {evs}): {args.label_a} {sa} ({sa / n:.1%}) vs {args.label_b} {sb} ({sb / n:.1%}), Δ {delta:+.1f}%p")
    print(f"불일치 {args.label_a}만 성공 {only_a} : {args.label_b}만 성공 {only_b} · 정확 McNemar 양측 p = {p2:.4f}, 단측(B>A) p = {p1:.4f}")
    print(f"Δ 군집 부트스트랩(시작 상태 군집 {len({(k[1], k[2]) for k in keys})}개): 95% [{ci[2.5]:+.1f}, {ci[97.5]:+.1f}] · 90% [{ci[5]:+.1f}, {ci[95]:+.1f}]")
    print("학습 seed 별 Δ: " + " · ".join(f"s{s} {v:+.1f}" for s, v in per_seed.items()))
    res = dict(label_a=args.label_a, label_b=args.label_b, n=n, succ_a=sa, succ_b=sb, delta_pp=delta,
               only_a=only_a, only_b=only_b, p_two=p2, p_one_b_gt_a=p1, ci95=[ci[2.5], ci[97.5]], ci90=[ci[5], ci[95]],
               delta_by_train_seed=per_seed, files_a=args.a, files_b=args.b, min_d0=args.min_d0,
               unmatched={f"s{g[0]}_seed{g[1]}": dict(a=u[0], b=u[1], dropped_a=u[2], dropped_b=u[3]) for g, u in unmatched.items()})
    ma, mb = load_carry(args.a, ida), load_carry(args.b, idb)
    if args.require_carry:
        if ma is None or mb is None:
            raise SystemExit("--require_carry: 각 CSV 옆 <이름>.carry.csv 가 없다 (carry_metrics.py 로 만든다)")
        miss = [k for k in keys if k not in ma or k not in mb]
        bad = [k for k in keys if k in ma and k in mb and (ma[k]["success"] != A[k]["success"] or mb[k]["success"] != B[k]["success"])]
        if miss or bad:
            raise SystemExit(f"--require_carry: .carry.csv 에 짝 {len(miss)}개가 없거나 성공 값이 분류 CSV 와 {len(bad)}개 다르다")
    if args.bundle:
        bund = set(args.bundle.split(","))
        col = args.bundle_col

        def member(R, M, k):
            if col in R[k]:
                return int(R[k][col] in bund)
            if M is None or k not in M or col not in M[k]:
                raise SystemExit(f"묶음 열 {col} 이 분류 CSV 에도 .carry.csv 에도 없다")
            return int(M[k][col] in bund)

        ia = {k: member(A, ma, k) for k in keys}
        ib = {k: member(B, mb, k) for k in keys}
        b_only_a = sum(1 for k in keys if ia[k] == 1 and ib[k] == 0)  # A 에서만 묶음 실패 (B 가 고침)
        b_only_b = sum(1 for k in keys if ia[k] == 0 and ib[k] == 1)
        q2, _ = mcnemar(b_only_b, b_only_a)
        q1 = sum(comb(b_only_a + b_only_b, i) for i in range(b_only_a, b_only_a + b_only_b + 1)) / 2 ** (b_only_a + b_only_b) if (b_only_a + b_only_b) else 1.0
        print(f"표적 묶음 {col} ∈ {sorted(bund)}: {args.label_a} {sum(ia.values())} → {args.label_b} {sum(ib.values())} · "
              f"A에서만 {b_only_a} : B에서만 {b_only_b} · 단측(B에서 줄어듦) p = {q1:.4f}, 양측 p = {q2:.4f}")
        res.update(bundle=sorted(bund), bundle_col=col, bundle_a=sum(ia.values()), bundle_b=sum(ib.values()), bundle_only_a=b_only_a,
                   bundle_only_b=b_only_b, bundle_p_one=q1, bundle_p_two=q2)
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
    ca_, cb_ = Counter(r["category"] for r in fa), Counter(r["category"] for r in fb)
    print("범주 합계: " + " · ".join(f"{c} {ca_.get(c, 0)}→{cb_.get(c, 0)}" for c in CATS))
    res["categories_a"], res["categories_b"] = dict(ca_), dict(cb_)
    res["pre_lift_fail_a"] = ca_.get("approach", 0) + ca_.get("grasp", 0)
    res["pre_lift_fail_b"] = cb_.get("approach", 0) + cb_.get("grasp", 0)
    # 상태 매개 지표: 각 CSV 옆의 <이름>.carry.csv (carry_metrics.py 출력) 가 있으면 같은 키로 읽는다
    if ma is not None and mb is not None and all(k in ma for k in keys) and all(k in mb for k in keys):
        print()
        print("상태 매개 지표 (carry_metrics, prereg 4절): A → B (차)")
        med_ = {}
        for key, kind in MEDIATORS:
            va, vb = mediator([ma[k] for k in keys], key, kind), mediator([mb[k] for k in keys], key, kind)
            med_[key] = dict(a=va, b=vb, diff=vb - va, kind=kind)
            unit = "중앙값" if kind == "median" else "건"
            print(f"{key:>14s} ({unit}): {va:.4g} → {vb:.4g} ({vb - va:+.4g})")
        med_["pre_lift_fail"] = dict(a=res["pre_lift_fail_a"], b=res["pre_lift_fail_b"],
                                     diff=res["pre_lift_fail_b"] - res["pre_lift_fail_a"], kind="count")
        print(f"{'pre_lift_fail':>14s} (건): {res['pre_lift_fail_a']} → {res['pre_lift_fail_b']} "
              f"({res['pre_lift_fail_b'] - res['pre_lift_fail_a']:+d})")
        res["mediators"] = med_
    else:
        print("(상태 매개 지표 생략: .carry.csv 가 없거나 짝이 다름)")
    if args.json:
        os.makedirs(os.path.dirname(os.path.abspath(args.json)), exist_ok=True)
        with open(args.json, "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
