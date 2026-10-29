"""A-T 성공 시범 묶음(N≈650)에서 baseline Mimic 의 시작 상태 분포에 맞춰 500개를 결정적으로 층화 추출한다. (트랙 A, 1029)

이유(교차 점검 #9): A-T 는 생성률이 높아(44% → 89%) 성공 필터를 통과하는 장면 분포가 baseline 과 달라진다.
baseline 생성률은 시작 트럭–상자 거리와 상관이 있다(< 0.30 m 28% · 0.30–0.35 32% · 0.35–0.40 45% · ≥ 0.40 53%).
그래서 처치 데이터의 (거리 4구간 × 상대 yaw 4구간) 히스토그램을 baseline 500 과 같게 맞춘 뒤 학습에 쓴다.

구간
  거리 d = |트럭 xy − 상자 xy| (첫 step, 로봇 base 기준): [0, 0.30) [0.30, 0.35) [0.35, 0.40) [0.40, ∞)
  상대 yaw = |wrap(트럭 yaw − 상자 yaw)| (deg, 0–180): [0, 45) [45, 90) [90, 135) [135, 180]
추출
  목표 개수 = baseline 칸 비율 × n (최대 나머지 반올림). 칸마다 demo 이름 정렬 뒤 numpy Generator(seed) 로 비복원 추출.
  칸이 모자라면 부족분을 같은 거리 구간의 다른 yaw 칸(가까운 yaw 칸 먼저) → 이웃 거리 구간 순서로 메우고 표에 표시한다.
  (거리 주변분포를 먼저 맞추는 규칙 — 생성률과 상관된 축이 거리이기 때문)

  python -I plan_A_stratify.py --ref ./datasets/team12_kgh/kgh_mimic_500.hdf5 --out <층화_500.hdf5> --n 500 --seed 0 \
      <A-T 샤드1.hdf5> [<샤드2.hdf5> ...]
  --dry_run : 표만 출력하고 파일은 쓰지 않음
"""
import argparse
import json
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_failures as CF  # noqa: E402

D_EDGES = [0.0, 0.30, 0.35, 0.40, np.inf]
Y_EDGES = [0.0, 45.0, 90.0, 135.0, 180.0001]


def start_cell(demo):
    o = demo["obs"]
    t, b = o["toy_truck_positions"][0], o["box_positions"][0]
    d = float(np.hypot(*(t - b)[:2]))
    ty = float(np.degrees(CF.quat_yaw(o["toy_truck_orientations"][0])))
    by = float(np.degrees(CF.quat_yaw(o["box_orientations"][0])))
    rel = abs(((ty - by) + 180.0) % 360.0 - 180.0)
    i = int(np.searchsorted(D_EDGES, d, side="right") - 1)
    j = int(np.searchsorted(Y_EDGES, rel, side="right") - 1)
    return min(i, 3), min(j, 3), d, rel


def histogram(files):
    """[(파일, demo 이름, 칸)] 목록과 4×4 개수."""
    items = []
    H = np.zeros((4, 4), int)
    for p in files:
        with h5py.File(p, "r") as f:
            for k in sorted(f["data"], key=lambda s: int(s.split("_")[-1])):
                if not bool(f["data"][k].attrs.get("success", True)):
                    continue
                i, j, d, rel = start_cell(f["data"][k])
                items.append((p, k, i, j))
                H[i, j] += 1
    return items, H


def targets(Href, n):
    w = Href / Href.sum() * n
    T = np.floor(w).astype(int)
    rem = n - T.sum()
    order = np.argsort(-(w - T), axis=None, kind="stable")
    for idx in order[:rem]:
        T.flat[idx] += 1
    return T


def fill_order(i, j):
    """부족분을 메울 칸 순서: 같은 거리 구간의 가까운 yaw 칸 → 이웃 거리 구간(가까운 순) 의 칸."""
    same = sorted([(i, jj) for jj in range(4) if jj != j], key=lambda c: abs(c[1] - j))
    others = sorted([(ii, jj) for ii in range(4) if ii != i for jj in range(4)], key=lambda c: (abs(c[0] - i), abs(c[1] - j)))
    return same + others


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pool", nargs="+")
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true")
    a = ap.parse_args()

    _, Href = histogram([a.ref])
    items, Hpool = histogram(a.pool)
    if len(items) < a.n:
        print(f"묶음이 {len(items)}개라 n={a.n} 을 채울 수 없다")
        if not a.dry_run:
            raise SystemExit(2)  # 사전등록: 학습 데이터는 정확히 500 — 줄여서 쓰지 않는다
        a.n = len(items)
    T = targets(Href, a.n)
    rng = np.random.default_rng(a.seed)
    cells = {(i, j): [it for it in items if it[2] == i and it[3] == j] for i in range(4) for j in range(4)}
    for c in cells:
        cells[c] = [cells[c][k] for k in rng.permutation(len(cells[c]))]  # 칸 안 순서를 seed 로 섞는다
    take = {c: min(T[c], len(cells[c])) for c in cells}
    deficit = {c: T[c] - take[c] for c in cells if T[c] > take[c]}
    borrowed = {c: 0 for c in cells}
    for c, need in sorted(deficit.items()):
        for c2 in fill_order(*c):
            spare = len(cells[c2]) - take[c2]
            if spare <= 0:
                continue
            k = min(spare, need)
            take[c2] += k
            borrowed[c2] += k
            need -= k
            if need == 0:
                break
        if need:
            print(f"경고: 칸 {c} 부족분 {need} 을 메우지 못함")
    sel = [it for c in sorted(cells) for it in cells[c][: take[c]]]

    dl = ["<0.30", "0.30–0.35", "0.35–0.40", "≥0.40"]
    yl = ["0–45°", "45–90°", "90–135°", "135–180°"]
    print(f"baseline {Href.sum()} · 묶음 {len(items)} · 추출 {len(sel)} (seed {a.seed})")
    print("| 거리 \\ 상대 yaw | " + " | ".join(yl) + " | 합 (baseline 비율 · 묶음 비율 · 추출 비율) |")
    print("| --- |" + " --- |" * 5)
    for i in range(4):
        cells_txt = []
        for j in range(4):
            c = (i, j)
            mark = f" (+{borrowed[c]})" if borrowed[c] else (f" (−{T[c] - take[c]})" if T[c] > take[c] else "")
            cells_txt.append(f"{Href[i, j]} → 목표 {T[i, j]} / 있음 {Hpool[i, j]} / 뽑음 {take[c]}{mark}")
        sr = sum(take[(i, j)] for j in range(4))
        print(f"| {dl[i]} | " + " | ".join(cells_txt) +
              f" | {Href[i].sum() / Href.sum():.1%} · {Hpool[i].sum() / max(Hpool.sum(), 1):.1%} · {sr / max(len(sel), 1):.1%} |")
    col = [sum(take[(i, j)] for i in range(4)) for j in range(4)]
    print("| yaw 비율 (baseline · 추출) | " + " | ".join(
        f"{Href[:, j].sum() / Href.sum():.1%} · {col[j] / max(len(sel), 1):.1%}" for j in range(4)) + " | |")

    # 사전등록 3절 게이트: 칸마다 |뽑음 − 목표| ≤ 2 (메움 포함). 넘으면 쓰지 않고 종료 코드 2
    worst = max(abs(take[c] - T[c]) for c in cells)
    print(f"층화 게이트 (칸마다 |뽑음 − 목표| ≤ 2): 최대 {worst} → {'통과' if worst <= 2 else '미통과'}")
    if worst > 2 and not a.dry_run:
        raise SystemExit(2)
    if a.dry_run or not a.out:
        return
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} 가 이미 있다")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    total = 0
    with h5py.File(a.out, "w") as out:
        out.create_group("data")
        env_args = None
        for n, (p, k, i, j) in enumerate(sel):
            with h5py.File(p, "r") as f:
                f.copy(f"data/{k}", out, f"data/demo_{n}")
                total += int(f["data"][k].attrs["num_samples"])
                env_args = env_args or f["data"].attrs["env_args"]
        out["data"].attrs["env_args"] = env_args
        out["data"].attrs["total"] = total
        out["data"].attrs["stratify"] = json.dumps(dict(
            ref=os.path.abspath(a.ref), pool=[os.path.abspath(p) for p in a.pool], seed=a.seed, n=len(sel),
            d_edges=[x if np.isfinite(x) else "inf" for x in D_EDGES], y_edges=Y_EDGES[:-1] + [180],
            ref_hist=Href.tolist(), pool_hist=Hpool.tolist(), target=T.tolist(),
            selected=[[os.path.basename(p), k] for p, k, _, _ in sel]))
    print(f"wrote {len(sel)} demos, total {total} steps -> {a.out}")


if __name__ == "__main__":
    main()
