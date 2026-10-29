"""운반 구간 기전 지표 — 데이터셋(사람 시범·Mimic 생성물)과 평가 기록(play_record.py)에 같은 정의를 쓴다.

모든 실험의 "기전 판정"은 이 스크립트 값으로 한다 (서베이 점검 1028: 지표 정의를 하나로 고정).
궤적 하나마다 classify_failures.classify() 의 이정표(t_lift, t_over …)를 쓰고, 그 위에 아래 값을 계산한다.

  rot60            들기(t_lift) 직후 60 step 회전 행동 크기 |Δroll|+|Δpitch|+|Δyaw| 의 평균 (classify 의 rot60_after_lift 와 같음)
  carry_rot_cmd    들기 ~ 상자 위(t_over, 없으면 마지막 '들고 있음' step) 구간 회전 행동 크기 평균
  carry_pos_cmd    같은 구간 위치 행동 크기 |Δx|+|Δy|+|Δz| 평균
  carry_yaw_net    같은 구간 손목 yaw 순변화 |yaw(끝) − yaw(시작)| (deg, unwrap)
  carry_yaw_max    같은 구간 손목 yaw 최대 이탈 max|yaw(t) − yaw(시작)| (deg)
  yaw30            들기 뒤 30 step 손목 yaw 변화 (deg)
  xy30             들기 뒤 30 step 손끝 xy 이동 (m)
  approach_z       들기 뒤 트럭이 상자 몸체 좌표 max(|x|,|y|) < 0.20 m 에 처음 들어올 때 트럭 높이(상자 중심 기준 z, m).
                   상자 외벽 반폭 약 0.127 m — 이 값이 낮으면 벽 높이로 들어와 부딪힌다
  box_push         상자가 처음 위치에서 5 cm 넘게 밀렸는가 (0/1)
  empty_at_box     트럭 없이 손이 상자 0.12 m 안까지 갔는가 (0/1, classify 와 같음)
  rot60_state      들기 뒤 60 step 손목(eef_quat) 각변위 합 (deg) — 상태 기반. 팔 사이 기전 비교는 이것으로 한다
                   (행동 기반 rot60·carry_*_cmd 는 라벨을 바꾸는 처치에서 저절로 줄어드는 '조작 확인' 지표)
  carry_steps      들기 → 상자 위(t_over) 소요 step (상자 위에 못 가면 nan)
  carry_speed      운반 구간 손끝 평균 속도 (m/step)
  timeout          horizon 끝까지 갔는가 (기록 파일에서 T = horizon + 1, 0/1)
  ── 실패 원인 귀속 (ii) (사전등록 prereg_1029.md, 들기 이후 실패 = 운반·배치 범주에만) ──
  t_loss           마지막 '들고 있음' step (끝까지 들고 있으면 -1)
  push_carry       운반 창 [t_lift, t_loss + 5] 에서 상자 xy 이동 증가 > 0.05 m 또는 상자 기울기 > 70° (0/1)
  release          놓친 직후 그리퍼 명령: open(스스로 놓음) / closed(미끄러짐) / hold(끝까지 들고 있음)
  attr_ii          collision → self_release → slip → other 순서로 처음 맞는 것 (성공·들기 전 실패는 '-')
                   민감도: attr_i (창 = t_loss ± 5), attr_iii (rollout 전체 box_disp_max > 0.05)

  python -I carry_metrics.py <파일.hdf5> [...] --csv out.csv            # 궤적별 행 + 묶음별 중앙값 요약
  python -I carry_metrics.py ... --group category                      # 범주별 요약 (기본: success/실패 범주)
"""
import argparse
import csv
import os
import sys
from collections import OrderedDict, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import classify_failures as CF  # noqa: E402

KEYS = ["rot60", "rot60_state", "carry_rot_cmd", "carry_pos_cmd", "carry_yaw_net", "carry_yaw_max", "yaw30", "xy30",
        "approach_z", "carry_steps", "carry_speed", "box_push", "empty_at_box"]
ATTR_KEYS = ["t_loss", "push_carry", "release", "attr_ii", "attr_i", "attr_iii", "timeout"]


def _yaw_deg(q):
    return np.degrees(np.unwrap(np.radians(np.degrees(CF.quat_yaw(q)))))


def metrics(tr, row=None):
    row = row or CF.classify(tr)
    act, eef, truck, box = tr["act"], tr["eef"], tr["truck"], tr["box"]
    T, A = len(truck), len(act)
    out = OrderedDict(src=row["src"], id=row["id"], success=row["success"], category=row["category"], subtype=row["subtype"])
    tl = row["t_lift"]
    nan = float("nan")
    for k in KEYS:
        out[k] = nan
    out["box_push"] = int(row["box_disp_max"] > 0.05)
    out["empty_at_box"] = row["empty_at_box"]
    out["timeout"] = int(tr.get("horizon") is not None and T >= tr["horizon"] + 1 and not row["success"])
    for k in ("t_loss", "push_carry", "release", "attr_ii", "attr_i", "attr_iii"):
        out[k] = "-"
    if tl < 0:
        return out
    # 운반 구간 끝: 상자 위 도달, 없으면 마지막 '들고 있음' step (d3 < 0.08 & 트럭 공중)
    d3 = np.linalg.norm(eef - truck, axis=1)
    z0 = float(truck[0, 2])
    held = (truck[:, 2] > z0 + CF.LIFT_DZ) & (d3 < CF.D_HOLD)
    hs = np.flatnonzero(held[tl:]) + tl
    te = row["t_over"] if row["t_over"] >= 0 else (int(hs[-1]) if len(hs) else tl)
    te = max(te, tl + 1)
    a_seg = act[tl:min(te, A)]
    if len(a_seg):
        out["carry_rot_cmd"] = round(float(np.abs(a_seg[:, 3:6]).sum(1).mean()), 4)
        out["carry_pos_cmd"] = round(float(np.abs(a_seg[:, :3]).sum(1).mean()), 4)
    out["rot60"] = round(float(np.abs(act[tl:tl + 60, 3:6]).sum(1).mean()), 4) if tl < A else nan
    yw = _yaw_deg(tr["eef_q"])
    seg = yw[tl:te + 1]
    if len(seg) > 1:
        out["carry_yaw_net"] = round(float(abs(seg[-1] - seg[0])), 2)
        out["carry_yaw_max"] = round(float(np.abs(seg - seg[0]).max()), 2)
    e30 = min(tl + 30, T - 1)
    out["yaw30"] = round(float(abs(yw[e30] - yw[tl])), 2)
    out["xy30"] = round(float(np.linalg.norm(eef[e30, :2] - eef[tl, :2])), 4)
    loc = CF.qrot_inv(tr["box_q"], truck - box)
    cheb = np.maximum(np.abs(loc[:, 0]), np.abs(loc[:, 1]))
    idx = np.flatnonzero(cheb[tl:] < 0.20)
    if len(idx):
        out["approach_z"] = round(float(loc[tl + idx[0], 2]), 4)
    # 상태 기반 회전: 들기 뒤 60 step 손목 각변위 합 (deg)
    q = tr["eef_q"][tl:min(tl + 61, T)]
    if len(q) > 1:
        dots = np.abs(np.sum(q[1:] * q[:-1], axis=1)).clip(0, 1)
        out["rot60_state"] = round(float(np.degrees(2 * np.arccos(dots)).sum()), 2)
    if row["t_over"] >= 0:
        out["carry_steps"] = int(row["t_over"] - tl)
    seg_e = eef[tl:te + 1]
    if len(seg_e) > 1:
        out["carry_speed"] = round(float(np.linalg.norm(np.diff(seg_e, axis=0), axis=1).mean()), 5)
    # ── 실패 원인 귀속 (들기 이후 실패만) ──
    if not row["success"] and row["category"] in ("transport", "place"):
        cmd_close, _ = CF.debounce(act[:, 6] < 0)
        cmd_s = np.r_[False, cmd_close] if A == T - 1 else (cmd_close[:T] if A >= T else np.r_[cmd_close, np.zeros(T - A, bool)])
        hold = (truck[:, 2] > z0 + CF.LIFT_DZ) & cmd_s & (d3 < CF.D_HOLD)
        hs2 = np.flatnonzero(hold[tl:]) + tl
        t_loss = int(hs2[-1]) if len(hs2) else tl
        holding_end = bool(hold[-1])
        box_disp = np.linalg.norm((box - box[0])[:, :2], axis=1)
        box_tilt = CF.quat_tilt(tr["box_q"])

        def pushed(a0, a1):
            a0, a1 = max(a0, 0), min(a1, T)
            if a1 <= a0:
                return False
            return bool((box_disp[a0:a1].max() - box_disp[a0]) > 0.05 or (box_tilt[a0:a1] > np.deg2rad(70)).any())

        if holding_end:
            rel = "hold"
        else:
            rel = "open" if not bool(cmd_s[min(t_loss + 1, T - 1)]) else "closed"

        def attribute(coll):
            if coll:
                return "collision"
            if rel == "open":
                return "self_release"
            if rel == "closed":
                return "slip"
            return "other"

        out["t_loss"] = -1 if holding_end else t_loss
        pc = pushed(tl, (T if holding_end else t_loss + 6))
        out["push_carry"] = int(pc)
        out["release"] = rel
        out["attr_ii"] = attribute(pc)
        out["attr_i"] = attribute(pushed(t_loss - 5, t_loss + 6) if not holding_end else False)
        out["attr_iii"] = attribute(bool(box_disp.max() > 0.05))
    return out


def summarize(rows, group):
    g = defaultdict(list)
    for r in rows:
        key = r[group] if group != "outcome" else ("success" if r["success"] else "fail_" + r["category"])
        g[key].append(r)
        if group == "outcome" and not r["success"] and r["category"] in ("transport", "place"):
            g["fail_post_lift"].append(r)
    g["ALL"] = rows
    lines = []
    hdr = f"{'group':<18s} {'n':>4s} " + " ".join(f"{k:>13s}" for k in KEYS)
    lines.append(hdr)
    for key in sorted(g, key=lambda k: (k != "ALL", k)):
        R = g[key]
        vals = []
        for k in KEYS:
            v = [float(r[k]) for r in R if r[k] == r[k]]
            if k in ("box_push", "empty_at_box"):
                vals.append(f"{np.mean(v):13.0%}" if v else f"{'—':>13s}")
            else:
                vals.append(f"{np.median(v):13.3f}" if v else f"{'—':>13s}")
        lines.append(f"{key:<18s} {len(R):4d} " + " ".join(vals))
    post = [r for r in rows if r["attr_ii"] not in ("-", None)]
    if post:
        from collections import Counter
        for k in ("attr_ii", "attr_i", "attr_iii"):
            c = Counter(r[k] for r in post)
            lines.append(f"귀속 {k} (들기 이후 실패 {len(post)}): " + " · ".join(f"{a} {c.get(a, 0)}" for a in ("collision", "self_release", "slip", "other")))
        lines.append(f"시간 초과(horizon) 실패: {sum(int(r['timeout']) for r in rows)}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--group", default="outcome", help="outcome(기본) | category | subtype | src")
    args = ap.parse_args()
    all_rows = []
    for p in args.files:
        rows = [metrics(tr) for tr in CF.load_trajs(p)]
        all_rows += rows
        print(f"## {os.path.basename(p)} (중앙값; box_push·empty_at_box 는 비율)")
        print(summarize(rows, args.group))
        print()
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {len(all_rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
