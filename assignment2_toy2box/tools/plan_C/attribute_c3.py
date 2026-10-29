"""C3 표적 묶음: 들기 이후 실패를 귀속 규칙 (ii) 계층형으로 나누고, '비충돌 스스로 놓음 중 소수 열기' 몫을 센다.

귀속 규칙 (ii) — 교차 점검 1029 §1 #8, 위에서부터 처음 맞는 하나:
  1. 충돌      운반 창 [t_lift, t_loss+5] 에서 상자가 t_lift 때 위치보다 xy 5 cm 넘게 밀렸거나 70° 넘게 기울었다
  2. 스스로 놓음  놓친 직후 명령(행동 t_loss)이 '열기'. 그 열기 구간의 첫 step t_rel 에서 혼합 P(open) < 0.5 면 '소수 열기', 아니면 '다수 열기'
  3. 미끄러짐   놓친 직후 명령이 '닫기'
  4. 기타      끝까지 들고 있음(놓침 없음) 등
  ('들고 있음' = 트럭 높이 > 처음 높이 + 0.03 m 이고 손끝–트럭 < 0.08 m. t_loss = t_lift 뒤 마지막 '들고 있음' 상태.
   기록 파일은 상태 t+1 이 행동 t 의 결과이므로 '놓친 직후 명령' = 행동 t_loss.)
민감도: (i) 상자 밂을 t_loss ±5 창에서만, (iii) rollout 전체 상자 최대 변위로 본 같은 분류도 함께 낸다.

P(open) 은 gmm_offline.py 출력(npz, 키 "<rollout>__popen", 행동 step 색인)을 붙인다.
공용 carry_metrics.py 에 t_loss·push_carry 열이 생기면 그 값과 이 스크립트 값을 대조할 것(정의는 위와 같게 맞춤).

  python -I attribute_c3.py --rec <record.hdf5> --gmm <gmm_offline.npz> [--rec ... --gmm ...] --csv out.csv
"""
import argparse
import csv
import os
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import classify_failures as CF  # noqa: E402

PUSH = 0.05
KNOCK = np.deg2rad(70)


def attribute(tr, row, popen):
    truck, eef, box, act = tr["truck"], tr["eef"], tr["box"], tr["act"]
    T, A = len(truck), len(act)
    tl = row["t_lift"]
    out = dict(src=row["src"], id=row["id"], success=row["success"], category=row["category"], subtype=row["subtype"],
               t_lift=tl, t_loss=-1, t_rel=-1, popen_rel=float("nan"), push_ii=0, push_i=0, push_iii=int(row["box_disp_max"] > PUSH),
               cls_ii="", cls_i="", cls_iii="")
    if tl < 0:
        out.update(cls_ii="no_lift", cls_i="no_lift", cls_iii="no_lift")
        return out
    z0 = float(truck[0, 2])
    d3 = np.linalg.norm(eef - truck, axis=1)
    held = (truck[:, 2] > z0 + CF.LIFT_DZ) & (d3 < CF.D_HOLD)
    hs = np.flatnonzero(held[tl:]) + tl
    t_loss = int(hs[-1]) if len(hs) else tl
    out["t_loss"] = t_loss
    lost = t_loss < T - 1
    bxy = box[:, :2]
    btilt = CF.quat_tilt(tr["box_q"])

    def pushed(a, b):
        a, b = max(a, 0), min(b, T - 1)
        if b < a:
            return False
        disp = np.linalg.norm(bxy[a:b + 1] - bxy[a], axis=1).max()
        return bool(disp > PUSH or (btilt[a:b + 1] - btilt[a]).max() > KNOCK)

    out["push_ii"] = int(pushed(tl, t_loss + 5))
    out["push_i"] = int(pushed(t_loss - 5, t_loss + 5))
    # 놓친 직후 명령과 그 열기 구간의 첫 step
    rel_open = lost and t_loss < A and act[t_loss, 6] >= 0
    if rel_open:
        t = t_loss
        while t - 1 >= tl and act[t - 1, 6] >= 0:
            t -= 1
        out["t_rel"] = t
        out["popen_rel"] = round(float(popen[t]), 4) if popen is not None and t < len(popen) else float("nan")

    def cls(push):
        if not lost:
            return "other_no_loss"
        if push:
            return "collision"
        if rel_open:
            p = out["popen_rel"]
            if p != p:
                return "self_release_unknown"
            return "self_release_minority" if p < 0.5 else "self_release_majority"
        return "slip"

    out["cls_ii"], out["cls_i"], out["cls_iii"] = cls(out["push_ii"]), cls(out["push_i"]), cls(out["push_iii"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", action="append", required=True)
    ap.add_argument("--gmm", action="append", default=[])
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = []
    for k, rec in enumerate(args.rec):
        Z = np.load(args.gmm[k]) if k < len(args.gmm) and args.gmm[k] else None
        for tr in CF.load_trajs(rec):
            r = CF.classify(tr)
            po = Z[f"{tr['id']}__popen"] if Z is not None else None
            rows.append(attribute(tr, r, po))
    post = [r for r in rows if not r["success"] and r["category"] in ("transport", "place")]
    succ = [r for r in rows if r["success"] and r["t_lift"] >= 0]
    print(f"들기 이후 실패 {len(post)}건 — 귀속 (ii) 계층형 / 민감도 (i) ±5 step / (iii) rollout 전체")
    order = ["collision", "self_release_minority", "self_release_majority", "slip", "other_no_loss", "self_release_unknown"]
    for key in ("cls_ii", "cls_i", "cls_iii"):
        c = Counter(r[key] for r in post)
        print(f"  {key}: " + " · ".join(f"{o} {c.get(o, 0)}" for o in order if c.get(o, 0) or o != 'self_release_unknown'))
    tgt = [r for r in post if r["cls_ii"] == "self_release_minority"]
    print(f"C3 표적 묶음 = (ii) 비충돌 스스로 놓음 중 소수 열기: {len(tgt)}건 → " + ", ".join(f"{r['src'][:15]}:{r['id']}({r['subtype']}, P={r['popen_rel']})" for r in tgt))
    print("  v3 세부 유형별 (ii) 분포:")
    for st, n in Counter((r["category"] + "/" + r["subtype"]) for r in post).most_common():
        c = Counter(r["cls_ii"] for r in post if r["category"] + "/" + r["subtype"] == st)
        print(f"    {st:36s} {n:3d}  " + " · ".join(f"{o} {c[o]}" for o in order if c.get(o)))
    # 성공 rollout 의 정상 놓기(같은 정의, 참고): 비충돌 스스로 놓음 중 소수 열기 비율
    cs = Counter(r["cls_ii"] for r in succ)
    print(f"참고 — 성공 {len(succ)}건의 놓기(같은 규칙): " + " · ".join(f"{o} {cs.get(o, 0)}" for o in order if cs.get(o, 0)))
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
