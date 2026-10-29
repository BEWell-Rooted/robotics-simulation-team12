"""트랙 A 스모크·게이트 점검 (1029): generate_yawhold.py 산출물을 baseline Mimic 과 비교한다.

  python -I plan_A_smoke_check.py passivity <yawhold0.hdf5> <baseline_shard.hdf5>
      demo 순서대로 시작 상태·actions 를 비교 (bit 동일 여부, 최대 차이)
  python -I plan_A_smoke_check.py gate <처리.hdf5> <대조.hdf5> [--log 로그 ...] [--log_ref 로그 ...]
      carry_metrics 정의로 데이터 게이트 값(중앙값·p90), 생성 성공률(로그 여러 개면 샤드별 마지막 'N/M' 을 합산),
      시작 상태 분포(트럭–상자 거리, 상대 yaw)
  python -I plan_A_smoke_check.py table --set 이름 데이터.hdf5 [로그,로그,...] [--set ...]
      여러 데이터셋의 게이트 지표를 한 표(markdown)로 나란히
"""
import argparse
import os
import re
import sys

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import carry_metrics as CM  # noqa: E402
import classify_failures as CF  # noqa: E402


def passivity(a, b, n=None):
    fa, fb = h5py.File(a, "r"), h5py.File(b, "r")
    ka = sorted(fa["data"], key=lambda s: int(s.split("_")[-1]))
    kb = sorted(fb["data"], key=lambda s: int(s.split("_")[-1]))
    n = min(len(ka), len(kb)) if n is None else n
    same_len = same_act = same_init = 0
    rows = []
    for i in range(n):
        da, db = fa["data"][ka[i]], fb["data"][kb[i]]
        aa, ab = da["actions"][:], db["actions"][:]
        ia = da["initial_state/rigid_object/toy_truck/root_pose"][:]
        ib = db["initial_state/rigid_object/toy_truck/root_pose"][:]
        init_d = float(np.abs(ia - ib).max())
        same_init += init_d == 0
        same_len += len(aa) == len(ab)
        m = min(len(aa), len(ab))
        d = np.abs(aa[:m] - ab[:m]).max(axis=1)
        first = int(np.argmax(d > 0)) if (d > 0).any() else -1
        same_act += (len(aa) == len(ab)) and first == -1
        rows.append((ka[i], len(aa), len(ab), init_d, float(d.max()) if m else float("nan"), first))
    print(f"비교 {n}개: 시작 상태 동일 {same_init} · 길이 동일 {same_len} · actions bit 동일 {same_act}")
    for r in rows[:n]:
        print(f"  {r[0]:>8s} len {r[1]:4d} vs {r[2]:4d}  init max|Δ| {r[3]:.2e}  actions max|Δ| {r[4]:.2e}  first diff step {r[5]}")
    # 시작 상태가 다르면 짝 순서가 다를 수 있으므로, 시작 트럭 자세로 짝을 다시 찾는다
    if same_init < n:
        init_b = {kb[j]: fb["data"][kb[j]]["initial_state/rigid_object/toy_truck/root_pose"][0] for j in range(len(kb))}
        matched = 0
        for i in range(n):
            ia = fa["data"][ka[i]]["initial_state/rigid_object/toy_truck/root_pose"][0]
            best = min(init_b, key=lambda k: np.abs(init_b[k] - ia).max())
            if np.abs(init_b[best] - ia).max() < 1e-5:
                matched += 1
        print(f"  시작 트럭 자세로 다시 짝지은 결과: {matched}/{n} 이 대조 파일 안에 같은 시작 상태를 가진 demo 가 있다")


def gen_rate(logs):
    """샤드 로그 여러 개의 마지막 'N/M (x%)' 를 합산한다. 없으면 None."""
    if not logs:
        return None
    if isinstance(logs, str):
        logs = [x for x in logs.split(",") if x]
    n = m_ = 0
    found = False
    for log in logs:
        if not os.path.exists(log):
            continue
        txt = open(log, errors="ignore").read()
        m = re.findall(r"(\d+)/(\d+) \(([\d.]+)%\)", txt)
        if m:
            n += int(m[-1][0])
            m_ += int(m[-1][1])
            found = True
    return (n, m_) if found else None


def grasp_over(path):
    """파지→상자 위 손목 yaw 순변화(deg)와 손끝 경로 길이(m) — 이정표는 classify v3."""
    yaw, plen = [], []
    for tr in CF.load_trajs(path):
        r = CF.classify(tr)
        tg, to = r["t_grasp"], r["t_over"]
        if tg < 0 or to < 0 or to <= tg:
            continue
        y = np.degrees(np.unwrap(CF.quat_yaw(tr["eef_q"])))
        yaw.append(abs(y[to] - y[tg]))
        plen.append(float(np.linalg.norm(np.diff(tr["eef"][tg:to], axis=0), axis=1).sum()))
    return np.array(yaw), np.array(plen)


def row_stats(path, logs=None):
    rows = [CM.metrics(tr) for tr in CF.load_trajs(path)]
    out = {"n": len(rows)}
    r = gen_rate(logs)
    out["gen"] = r
    for k in ("carry_yaw_net", "yaw30", "rot60", "carry_pos_cmd", "xy30", "approach_z"):
        v = np.array([x[k] for x in rows if x[k] == x[k]], float)
        out[k] = (np.median(v), np.percentile(v, 10), np.percentile(v, 90)) if len(v) else (np.nan,) * 3
    out["box_push"] = float(np.mean([x["box_push"] for x in rows]))
    yaw, plen = grasp_over(path)
    out["yaw_go"] = np.median(yaw) if len(yaw) else np.nan
    out["path_go"] = np.median(plen) if len(plen) else np.nan
    d, rel = start_dist(path)
    out["near"] = float(np.mean(d < 0.30))
    out["rel90"] = float(np.mean(rel > 90))
    return out


def table(sets):
    cols = [("생성 성공", lambda o: f"{o['gen'][0]}/{o['gen'][1]} ({o['gen'][0] / o['gen'][1]:.0%})" if o["gen"] else "—"),
            ("n", lambda o: str(o["n"])),
            ("손목 yaw 파지→상자 위 (°)", lambda o: f"{o['yaw_go']:.1f}"),
            ("carry_yaw_net (°)", lambda o: f"{o['carry_yaw_net'][0]:.1f}"),
            ("yaw30 (°)", lambda o: f"{o['yaw30'][0]:.1f}"),
            ("xy30 (m)", lambda o: f"{o['xy30'][0]:.3f}"),
            ("approach_z 중앙 / p10 (m)", lambda o: f"{o['approach_z'][0]:.3f} / {o['approach_z'][1]:.3f}"),
            ("box_push", lambda o: f"{o['box_push']:.0%}"),
            ("rot60 (행동)", lambda o: f"{o['rot60'][0]:.3f}"),
            ("carry_pos_cmd (행동)", lambda o: f"{o['carry_pos_cmd'][0]:.3f}"),
            ("파지→상자 위 경로 (m)", lambda o: f"{o['path_go']:.2f}"),
            ("시작 < 0.30 m", lambda o: f"{o['near']:.0%}"),
            ("상대 yaw > 90°", lambda o: f"{o['rel90']:.0%}")]
    stats = [(name, row_stats(path, logs)) for name, path, logs in sets]
    print("| 지표 | " + " | ".join(n for n, _ in stats) + " |")
    print("| --- |" + " --- |" * len(stats))
    for cname, f in cols:
        print(f"| {cname} | " + " | ".join(f(o) for _, o in stats) + " |")


def start_dist(path):
    d, rel = [], []
    with h5py.File(path, "r") as f:
        for k in f["data"]:
            o = f["data"][k]["obs"]
            t, b = o["toy_truck_positions"][0], o["box_positions"][0]
            d.append(float(np.hypot(*(t - b)[:2])))
            ty = np.degrees(CF.quat_yaw(o["toy_truck_orientations"][0]))
            by = np.degrees(CF.quat_yaw(o["box_orientations"][0]))
            rel.append(abs(((ty - by) + 180) % 360 - 180))
    return np.array(d), np.array(rel)


def gate(tre, ref, log=None, log_ref=None):
    for name, p, lg in (("처리", tre, log), ("대조", ref, log_ref)):
        rows = [CM.metrics(tr) for tr in CF.load_trajs(p)]
        print(f"## {name}: {os.path.basename(p)}  n={len(rows)}")
        r = gen_rate(lg)
        if r:
            print(f"  생성 성공 {r[0]}/{r[1]} = {r[0] / r[1]:.1%}")
        for k in ("carry_yaw_net", "carry_yaw_max", "yaw30", "rot60", "carry_rot_cmd", "carry_pos_cmd", "xy30", "approach_z"):
            v = np.array([x[k] for x in rows if x[k] == x[k]], float)
            if len(v):
                print(f"  {k:>14s} 중앙값 {np.median(v):8.3f}  p10 {np.percentile(v, 10):8.3f}  p90 {np.percentile(v, 90):8.3f}")
        bp = np.mean([x["box_push"] for x in rows])
        print(f"  {'box_push':>14s} {bp:.0%}")
        d, rel = start_dist(p)
        print(f"  시작 트럭–상자 거리 중앙값 {np.median(d):.3f} (p10 {np.percentile(d, 10):.3f}, < 0.30 m {np.mean(d < 0.30):.0%})"
              f" · 상대 yaw(360) 중앙값 {np.median(rel):.0f}° (> 90° {np.mean(rel > 90):.0%})")


# 사전등록 docs/prereg_1029.md 3절 데이터 게이트 (이 표가 유일한 기준; plan_A 4절은 이것을 가리킨다)
GATES = {
    "AT": [("gen", ">=", 0.70), ("yaw_go", "<=", 20.0), ("carry_yaw_net", "<=", 15.0), ("xy30", "<=", 0.13),
           ("approach_z_p10", ">=", 0.15), ("rot60", "<=", 0.08), ("carry_pos_cmd", "<=", 0.08), ("end_xy", "<=", 0.07)],
    "AR": [("gen", ">=", 0.45), ("yaw_go", "<=", 20.0), ("carry_yaw_net", "<=", 15.0), ("xy30", ">=", 0.15),
           ("path_go", ">=", 1.1), ("end_xy", "<=", 0.07)],
}


def end_xy(path):
    """성공 시점(시범 끝) 트럭–상자 xy 거리 중앙값 (m)."""
    v = [CF.classify(tr)["end_xy_box"] for tr in CF.load_trajs(path)]
    v = [x for x in v if x == x]
    return float(np.median(v)) if v else float("nan")


def check(arm, path, logs=None, at_yaw=None, n_expect=500, n_logs=5):
    """게이트를 문턱과 비교해 PASS/FAIL 을 찍고, 하나라도 FAIL 이면 종료 코드 1.
    생성 로그는 샤드 n_logs 개가 모두 있어야 하고(없는 파일이 있으면 FAIL), 데이터는 demo n_expect 개여야 한다."""
    if arm == "AR" and at_yaw is None:
        print("FAIL  A-R 게이트에는 --at_yaw (A-T 의 손목 yaw 파지→상자 위 중앙값)가 필요하다")
        return False
    logs = logs or []
    missing = [lg for lg in logs if not os.path.exists(lg)]
    o = row_stats(path, logs)
    vals = {"gen": (o["gen"][0] / o["gen"][1]) if o["gen"] else float("nan"), "yaw_go": o["yaw_go"], "path_go": o["path_go"],
            "carry_yaw_net": o["carry_yaw_net"][0], "xy30": o["xy30"][0], "approach_z_p10": o["approach_z"][1],
            "rot60": o["rot60"][0], "carry_pos_cmd": o["carry_pos_cmd"][0], "end_xy": end_xy(path)}
    ok = True
    print(f"## 게이트 {arm}: {os.path.basename(path)} (n = {o['n']})")
    for name, good, msg in (("logs", len(logs) == n_logs and not missing, f"생성 로그 {len(logs)}개, 없는 파일 {missing}"),
                            ("n", n_expect is None or o["n"] == n_expect, f"demo {o['n']}개 (기대 {n_expect})")):
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {name:>14s} : {msg}")
    gates = list(GATES[arm])
    if arm == "AR" and at_yaw is not None:
        gates.append(("yaw_go_vs_AT", "<=", 6.0))
        vals["yaw_go_vs_AT"] = abs(o["yaw_go"] - at_yaw)
    for k, op, th in gates:
        v = vals[k]
        good = (v >= th) if op == ">=" else (v <= th)
        good = bool(good) and v == v
        ok &= good
        print(f"  {'PASS' if good else 'FAIL'}  {k:>14s} = {v:.3f}  (기준 {op} {th})")
    print(f"  box_push {o['box_push']:.0%} · 시작 < 0.30 m {o['near']:.0%} · 상대 yaw > 90° {o['rel90']:.0%} (보고만)")
    print("게이트 " + ("통과" if ok else "미통과"))
    return ok


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("passivity")
    p1.add_argument("a")
    p1.add_argument("b")
    p1.add_argument("--n", type=int, default=None)
    p2 = sub.add_parser("gate")
    p2.add_argument("treat")
    p2.add_argument("ref")
    p2.add_argument("--log", nargs="*", default=None, help="처리 생성 로그 (샤드 여러 개면 모두)")
    p2.add_argument("--log_ref", nargs="*", default=None)
    p4 = sub.add_parser("check", help="사전등록 3절 게이트를 문턱과 비교 (FAIL 이면 종료 코드 1)")
    p4.add_argument("arm", choices=sorted(GATES))
    p4.add_argument("data")
    p4.add_argument("--log", nargs="*", default=None, help="생성 로그 (샤드 여러 개면 모두)")
    p4.add_argument("--at_yaw", type=float, default=None, help="A-R 일 때 필수: A-T 의 손목 yaw 파지→상자 위 중앙값 (± 6° 비교)")
    p4.add_argument("--n_expect", type=int, default=500, help="기대 demo 수 (스모크 확인 때만 바꾼다)")
    p4.add_argument("--n_logs", type=int, default=5, help="기대 샤드 로그 수 (스모크 확인 때만 바꾼다)")
    p3 = sub.add_parser("table")
    p3.add_argument("--set", nargs="+", action="append", required=True, metavar=("NAME", "DATA"),
                    help="이름 데이터.hdf5 [로그1,로그2,...]")
    a = ap.parse_args()
    if a.cmd == "passivity":
        passivity(a.a, a.b, a.n)
    elif a.cmd == "gate":
        gate(a.treat, a.ref, a.log, a.log_ref)
    elif a.cmd == "check":
        sys.exit(0 if check(a.arm, a.data, a.log, a.at_yaw, a.n_expect, a.n_logs) else 1)
    else:
        table([(x[0], x[1], x[2] if len(x) > 2 else None) for x in a.set])


if __name__ == "__main__":
    main()
