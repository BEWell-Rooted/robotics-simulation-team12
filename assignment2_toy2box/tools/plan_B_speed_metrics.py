"""트랙 B 속도·접근 지표와 open-loop 재생 게이트 (교차 점검 1029 #11).

B2(reach k=4) 는 라벨 크기를 줄여 운반을 느리게 할 수 있다. 운반이 느려지기만 해도 흔들림·충돌이 줄 수 있으므로
"방향·일관성" 효과와 섞이지 않게 속도·시간 초과·접근 단계를 따로 잰다. 이정표는 classify_failures.py v3.

궤적별 열 (평가 기록 rollouts/ · 데이터셋 data/ 모두):
  approach_steps   0 → t_near (손끝–트럭 < 5 cm 첫 step). 없으면 빈칸
  approach_speed   [0, t_near) 손끝 평균 속도 (m/step)
  close_steps      0 → t_first_close (트럭 10 cm 안 첫 닫기 명령)
  lift_to_over     t_lift → t_over (들기 → 상자 입구 위) 소요 step. 둘 중 없으면 빈칸
  carry_speed      [t_lift, 끝) 손끝 평균 속도 (m/step). 끝 = t_over, 없으면 마지막 '들고 있음' step
  carry_path       같은 구간 손끝 경로 길이 (m)
  timeout          평가 기록에서 성공 없이 horizon 까지 간 rollout (end_reason == horizon) → 1
요약: 묶음(성공 / 실패 범주 / 들기 이후 실패 / 전체)별 중앙값, timeout 수.

재생 게이트 (--replay_src <재생한 데이터셋>): replay_check.py 기록을 원본 시범과 같은 시점으로 맞춰
  dev50 (t=50 손끝 위치 편차), dev_lift (원본 시범의 t_lift 시점 편차), devmax, 재생 성공 을 낸다.
  (1029 이전 replay_check 기록은 행동 열이 덮여 있으므로 행동은 --replay_src 의 데이터셋 값으로 바꿔 쓴다.)

  python -I plan_B_speed_metrics.py <기록/데이터셋.hdf5> [...] --csv out.csv
  python -I plan_B_speed_metrics.py <replay_record.hdf5> --replay_src <sub10_X.hdf5> --orig <sub10_orig.hdf5> --csv out.csv
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict, defaultdict

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_failures as CF  # noqa: E402

NUM = ["approach_steps", "approach_speed", "close_steps", "lift_to_over", "carry_speed", "carry_path"]


def speed_row(tr, row=None):
    row = row or CF.classify(tr)
    eef, truck = tr["eef"], tr["truck"]
    T = len(eef)
    v = np.r_[np.linalg.norm(np.diff(eef, axis=0), axis=1), np.nan]
    out = OrderedDict(src=row["src"], id=row["id"], success=row["success"], category=row["category"], subtype=row["subtype"],
                      T=T, end_reason=row["end_reason"])
    nan = float("nan")
    tn, tc, tl, to = row["t_near"], row["t_first_close"], row["t_lift"], row["t_over"]
    out["approach_steps"] = tn if tn >= 0 else nan
    out["approach_speed"] = round(float(np.nanmean(v[:tn])), 5) if tn > 0 else nan
    out["close_steps"] = tc if tc >= 0 else nan
    out["lift_to_over"] = (to - tl) if (tl >= 0 and to >= 0) else nan
    if tl >= 0:
        d3 = np.linalg.norm(eef - truck, axis=1)
        held = (truck[:, 2] > truck[0, 2] + CF.LIFT_DZ) & (d3 < CF.D_HOLD)
        hs = np.flatnonzero(held[tl:]) + tl
        te = to if to >= 0 else (int(hs[-1]) if len(hs) else tl)
        te = max(te, tl + 1)
        out["carry_speed"] = round(float(np.nanmean(v[tl:te])), 5)
        out["carry_path"] = round(float(np.nansum(v[tl:te])), 4)
    else:
        out["carry_speed"] = out["carry_path"] = nan
    out["timeout"] = int(tr["kind"] == "rollout" and not row["success"] and row["end_reason"] == "horizon")
    return out


def summarize(rows):
    g = defaultdict(list)
    for r in rows:
        g["success" if r["success"] else "fail_" + r["category"]].append(r)
        if not r["success"] and r["category"] in ("transport", "place"):
            g["fail_post_lift"].append(r)
    g["ALL"] = rows
    lines = [f"{'group':<16s} {'n':>4s} " + " ".join(f"{k:>14s}" for k in NUM) + f" {'timeout':>8s}"]
    for key in sorted(g, key=lambda k: (k != "ALL", k)):
        R = g[key]
        vals = []
        for k in NUM:
            x = [float(r[k]) for r in R if r[k] == r[k]]
            vals.append(f"{np.median(x):14.4f}" if x else f"{'—':>14s}")
        lines.append(f"{key:<16s} {len(R):4d} " + " ".join(vals) + f" {sum(r['timeout'] for r in R):8d}")
    return "\n".join(lines)


def replay_rows(path, src, orig):
    trajs = CF.load_trajs(path)
    out = []
    with h5py.File(path, "r") as f, h5py.File(src, "r") as fs, h5py.File(orig, "r") as fo:
        otrajs = {t["id"]: t for t in CF.load_trajs(orig)}
        for tr in trajs:
            g = f["rollouts"][tr["id"]]
            demo = g.attrs["demo"]
            tr["act"] = fs["data"][demo]["actions"][: len(tr["truck"]) - 1]  # 실제 실행된 행동
            r = CF.classify(tr)
            ot = otrajs[demo]
            orow = CF.classify(ot)
            eo, er = ot["eef"], tr["eef"]
            n = min(len(eo), len(er))
            dev = np.linalg.norm(er[:n] - eo[:n], axis=1)
            tl_o = orow["t_lift"]
            sp = speed_row(tr, r)
            out.append(OrderedDict(variant=os.path.basename(src), demo=demo, replay_success=int(bool(g.attrs["success"])),
                                   category=r["category"], subtype=r["subtype"],
                                   dev50_cm=round(100 * float(dev[min(50, n - 1)]), 2),
                                   dev_lift_cm=round(100 * float(dev[min(tl_o, n - 1)]), 2) if tl_o >= 0 else float("nan"),
                                   devmax_cm=round(100 * float(dev.max()), 2), orig_t_lift=tl_o,
                                   approach_steps=sp["approach_steps"], lift_to_over=sp["lift_to_over"],
                                   carry_speed=sp["carry_speed"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--replay_src", default=None, help="재생한 데이터셋 (재생 게이트 모드)")
    ap.add_argument("--orig", default=None, help="원본 데이터셋 (재생 게이트 기준 궤적)")
    args = ap.parse_args()
    rows = []
    if args.replay_src:
        for p in args.files:
            R = replay_rows(p, args.replay_src, args.orig or args.replay_src)
            rows += R
            d50 = [r["dev50_cm"] for r in R]
            dl = [r["dev_lift_cm"] for r in R if r["dev_lift_cm"] == r["dev_lift_cm"]]
            print(f"## {os.path.basename(p)} vs {os.path.basename(args.orig or '')}: 성공 {sum(r['replay_success'] for r in R)}/{len(R)} · "
                  f"dev50 중앙값 {np.median(d50):.2f} cm (최대 {max(d50):.2f}) · dev_lift 중앙값 {np.median(dl):.2f} cm (최대 {max(dl):.2f}) · "
                  f"devmax 중앙값 {np.median([r['devmax_cm'] for r in R]):.1f} cm")
    else:
        for p in args.files:
            R = [speed_row(tr) for tr in CF.load_trajs(p)]
            rows += R
            print(f"## {os.path.basename(p)} (중앙값; timeout 은 개수)")
            print(summarize(R))
            print()
    if args.csv and rows:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
