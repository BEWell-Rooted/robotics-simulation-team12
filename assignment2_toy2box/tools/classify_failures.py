"""Toy2Box 궤적을 접근 · 파지 · 운반 · 배치 단계로 나눠 실패를 분류한다 (Isaac Sim 불필요).

입력 (형식 자동 판별):
  - play_record.py 기록 파일 (rollouts/rollout_<k>)  ← BC 정책 평가 rollout
  - robomimic/Isaac Lab 데이터셋 (data/demo_<k>)      ← 사람 시범, Mimic 생성물, Mimic _failed
모든 값은 로봇 base 기준 관측(eef_pos, toy_truck_positions, box_positions, gripper_pos, actions)으로 계산하므로
세 종류에 같은 기준이 적용된다.

단계 이정표 (궤적마다 처음 도달한 step):
  t_near    손끝-트럭 3D 거리 < D_NEAR (0.05 m, 환경 grasp 신호와 같은 거리)
  t_grasp   grasp 가 GRASP_K step 이상 연속 (기록·annotate 파일은 환경 신호, 없으면 대용 신호)
  t_lift    들고 있음(트럭 높이 > 처음 높이 + LIFT_DZ, 닫기 명령, 손끝-트럭 < D_HOLD)이 LIFT_K step 이상 연속
  t_over    들린 트럭이 상자 중심 xy < OVER_XY (0.10 m, 성공 조건과 같은 값)
  t_success 성공 (기록 파일은 환경 성공 함수, 데이터셋은 attrs success + 마지막 step 조건)
분류 = 가장 멀리 간 이정표의 다음 단계:
  (접근·파지에서 트럭이 테이블 밖으로 떨어져 종료되면 pushed_off_table)
  접근 approach  t_near 없음            세부: knocked(먼저 트럭을 밀거나 넘어뜨림) / close_misaligned(0.10 m 안에서 닫았지만 0.05 m 안엔 못 감)
                                          / to_box_without_truck(트럭 없이 손이 상자로 감) / hover(0.10 m 안까지만, 닫기 없음) / far
  파지 grasp     t_near 있음, t_lift 없음 세부: no_close(닫기 명령 없음) / miss(닫았지만 grasp 안 됨) / slip(잡았지만 못 듦)
  운반 transport t_lift 있음, t_over 없음 세부: dropped(닫은 채 미끄러져 놓침) / released_early(상자 위에 가기 전에 열어서 놓음)
                                          / dropped_near_box · released_near_box(상자 20 cm 안에서 놓침) / box_collision(상자를 밀거나 넘어뜨림)
                                          / off_table(테이블 밖 낙하 종료) / stall(들고 있다가 시간 초과)
  배치 place     t_over 있음(또는 들었던 트럭이 상자 안 구석에 있음), 성공 없음
                                          세부: in_box_off_center(상자 안에 들어갔지만 중심 0.10 m 밖) / in_box_not_open(상자 안인데 그리퍼 닫힘)
                                          / no_release(안 놓음) / released_outside(상자 밖에서 놓음) / other
  v3 (1028 시각 검증 워크플로 반영 — 실패 68건을 장면 띠·시간축으로 판정: 범주 일치 63/68, 재판정 패치 채택):
  - t_over: 상자 몸체 좌표에서 트럭 중심이 입구(반폭 0.11 m) 안에 들기 뒤 3 step 연속 (기울어진 상자에서도 맞음)
  - t_lift 대안: grasp 신호 + 정지 높이 + 5 cm + 손끝-트럭 < 0.08 m 가 3 step (낮게 들고 휩쓴 경우)
  - 배치 세부 in_box_offcenter(상자 안에 들어갔지만 중심 0.10 m 밖), dropped_outside / 판정 순서 in_box_not_open → in_box_offcenter → no_release → released_outside
  - 운반 세부 box_knocked_over(들고 가다 상자를 70° 넘게 넘어뜨림), box_collision_released/dropped(놓치기 직전 10 step 창에서 상자가 5 cm 넘게 밀림)
    · 상자를 미는 것만으로는 범주를 바꾸지 않는다(성공 시범 81/540 도 5 cm 넘게 밂) — box_disp_max 열로만 기록
  - '상자 가까이' 반경 0.20 m(마지막 20 step 최소, 처음 상자 위치 기준 포함), 그리퍼 1-step 스파이크 제거,
    접근 세부 wrong_object(상자를 듦)·freeze(마지막 300 step 멈춤)
  - 표시 열: empty_at_box(트럭 없이 손이 상자 0.12 m 안) · shallow_grasp · rot60_after_lift(들기 직후 60 step 회전 명령 크기) · box_moved

사용:
  python -I classify_failures.py <파일.hdf5> [<파일2> ...] --csv out.csv [--plots 폴더] [--max_plots 30]
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter, OrderedDict

import h5py
import numpy as np

# ---- 기준값 (사람 시범 40개에서 확인: grasp 순간 손끝-트럭 거리 0.006–0.045 m, 들 때 트럭 상승 ≥ 0.21 m,
#      성공 순간 트럭-상자 xy 0.001–0.099 m · z 차 -0.091 ~ -0.100 m)
D_NEAR = 0.05
D_HOVER = 0.10
GRASP_K = 3
LIFT_DZ = 0.03
LIFT_K = 5
OVER_XY = 0.10
SUCC_XY = 0.10
SUCC_DZ = 0.10  # |z 차| - height_diff(0.06) < height_threshold(0.04)
KNOCK_XY = 0.05  # 첫 닫기 전에 트럭이 이만큼 밀리면 knocked 표시 (조교 성공 시범도 최대 8 cm 밀었으므로 표시일 뿐)
KNOCK_TILT = np.deg2rad(30)
D_HOLD = 0.08  # 들고 있음 판정: 손끝-트럭 거리 (Mimic 생성물은 운반 중 최대 0.055 m)
D_PROXY = 0.07  # grasp 신호가 없는 파일의 대용 grasp 거리
OPEN_ABS = 0.794  # |finger| > 0.994 - 0.2 이면 열림 (성공 조건과 같은 판정)
IDLE_EPS = 0.01
OVER_K = 3  # 상자 위 판정 연속 step (1 step 스침 제외, 검증 1028)
NEAR_BOX = 0.20  # 운반 실패 중 '상자 가까이' 반경 (림 바깥·벽 옆 놓기 포함, 검증 1028: 0.16 은 좁음)
OPEN_HALF = 0.11  # 상자 입구 반폭 (상자 몸체 좌표). 외곽 반폭 약 0.127 (상자가 넘어질 때 중심 상승으로 추정)
LIFT_DZ_REST = 0.05  # 정지 높이 기준 들기 높이 (스폰 z0 + 0.03 ≈ 정지 + 0.049 와 같은 값)
LIFT_K2 = 3
BOX_KNOCK_TILT = np.deg2rad(70)  # 성공 시범 540개의 상자 기울기 최대 56°
BOX_PUSH = 0.05  # 놓치기 직전 창에서 상자가 이만큼 밀리면 상자 충돌 (성공 시범 81/540 도 상자를 5 cm 넘게 밀므로 창 안에서만 봄)
PUSH_WIN = 10
BOX_MOVE_XY = 0.03  # 표시 열 box_moved 용
BOX_MOVE_TILT = np.deg2rad(10)


def qrot_inv(q, v):
    """(w,x,y,z) 쿼터니언의 역회전을 벡터 v 에 적용 (상자 몸체 좌표로 옮길 때)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    u = np.stack([-x, -y, -z], -1)
    t = 2 * np.cross(u, v)
    return v + w[..., None] * t + np.cross(u, t)


CAT_KO = OrderedDict(
    [("success", "성공"), ("approach", "접근"), ("grasp", "파지"), ("transport", "운반"), ("place", "배치")]
)


def quat_tilt(q):
    """(w,x,y,z) → 몸체 z축이 세계 z축과 이루는 각 (rad)."""
    x, y = q[..., 1], q[..., 2]
    return np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))


def quat_yaw(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def first_run(mask, k):
    """mask 가 k step 이상 연속 True 가 되는 첫 시작 index, 없으면 -1."""
    if k <= 1:
        idx = np.flatnonzero(mask)
        return int(idx[0]) if len(idx) else -1
    run = 0
    for t, m in enumerate(mask):
        run = run + 1 if m else 0
        if run >= k:
            return t - k + 1
    return -1


def load_trajs(path):
    """파일 하나에서 궤적 dict 목록을 읽는다."""
    out = []
    with h5py.File(path, "r") as f:
        if "rollouts" in f:
            kind = "rollout"
            g0 = f["rollouts"]
            names = sorted(g0, key=lambda s: int(s.split("_")[-1]))
            for n in names:
                g = g0[n]
                o = g["obs"]
                tr = dict(
                    src=os.path.basename(path), kind=kind, id=n,
                    eef=o["eef_pos"][:], eef_q=o["eef_quat"][:], truck=o["toy_truck_positions"][:], truck_q=o["toy_truck_orientations"][:],
                    box=o["box_positions"][:], box_q=o["box_orientations"][:], gp=o["gripper_pos"][:],
                    act=g["actions"][:], grasp=g["grasp"][:].astype(bool), success_t=g["success"][:].astype(bool),
                    grip_open=g["grip_open"][:].astype(bool) if "grip_open" in g else None,
                    end_reason=str(g.attrs.get("end_reason", "")), success=bool(g.attrs.get("success", False)),
                    horizon=int(f.attrs.get("horizon", 800)),
                )
                out.append(tr)
        elif "data" in f:
            D = f["data"]
            names = sorted(D, key=lambda s: int(s.split("_")[-1]))
            base = os.path.basename(path)
            kind = "failed_gen" if "_failed" in base else ("demo")
            for n in names:
                g = D[n]
                o = g["obs"]
                grasp = None
                if "datagen_info" in o and "subtask_term_signals" in o["datagen_info"]:
                    grasp = o["datagen_info/subtask_term_signals/grasp"][:].astype(bool)
                succ = bool(g.attrs.get("success", False))
                tr = dict(
                    src=base, kind=kind, id=n,
                    eef=o["eef_pos"][:], eef_q=o["eef_quat"][:], truck=o["toy_truck_positions"][:], truck_q=o["toy_truck_orientations"][:],
                    box=o["box_positions"][:], box_q=o["box_orientations"][:], gp=o["gripper_pos"][:],
                    act=g["actions"][:], grasp=grasp, success_t=None, grip_open=None,
                    end_reason="success" if succ else "", success=succ, horizon=None,
                )
                out.append(tr)
        else:
            raise ValueError(f"{path}: rollouts/ 도 data/ 도 없다")
    return out


def debounce(cmd):
    """그리퍼 명령의 1-step 스파이크(앞뒤가 같은데 혼자 다른 step)를 없앤다. (값, 스파이크 수)"""
    c = cmd.copy()
    n = 0
    for t in range(1, len(cmd) - 1):
        if cmd[t - 1] == cmd[t + 1] != cmd[t]:
            c[t] = cmd[t - 1]
            n += 1
    return c, n


def classify(tr):
    eef, truck, box, gp, act = tr["eef"], tr["truck"], tr["box"], tr["gp"], tr["act"]
    T = len(truck)
    z0 = float(truck[0, 2])  # 스폰 높이 (reset 직후). 정지 높이는 약 1.9 cm 아래
    d3 = np.linalg.norm(eef - truck, axis=1)
    _pre = np.flatnonzero(d3 < D_HOVER)
    z_rest = float(truck[: max(int(_pre[0]) if len(_pre) else T, 1), 2].min())  # 손이 0.10 m 안에 오기 전 최저 높이
    xy_tb = np.linalg.norm((truck - box)[:, :2], axis=1)
    xy_tb0 = np.linalg.norm(truck[:, :2] - box[0, :2], axis=1)  # 처음 상자 위치 기준 (상자가 밀린 경우 대비)
    dz_tb = truck[:, 2] - box[:, 2]
    tilt = quat_tilt(tr["truck_q"])
    loc = qrot_inv(tr["box_q"], truck - box)  # 상자 몸체 좌표의 트럭 위치 (상자가 기울어도 맞음)
    cheb_loc = np.maximum(np.abs(loc[:, 0]), np.abs(loc[:, 1]))
    box_tilt = quat_tilt(tr["box_q"])
    box_disp = np.linalg.norm((box - box[0])[:, :2], axis=1)
    box_moved_t = (box_disp > BOX_MOVE_XY) | (np.abs(box_tilt - box_tilt[0]) > BOX_MOVE_TILT)
    box_lift = box[:, 2] - box[0, 2]
    fingers_open = (np.abs(gp) > OPEN_ABS).all(axis=1)
    if tr.get("grip_open") is not None:
        fingers_open = tr["grip_open"].all(axis=1)
    # 행동 → 상태 정렬: 기록 파일은 obs 가 T+1, actions 가 T. 데이터셋은 둘 다 T.
    A = len(act)
    cmd_raw = act[:, 6] < 0  # BinaryJointPositionAction: 음수면 닫기
    cmd_close, n_spike = debounce(cmd_raw)  # 1-step 스파이크 제거
    # 상태 index 기준 그리퍼 명령: 기록 파일(obs T+1)은 상태 t 가 행동 t-1 의 결과, 데이터셋(obs T)은 상태 t 에서 행동 t
    if A == T - 1:
        cmd_close_s = np.r_[False, cmd_close]
    else:
        cmd_close_s = cmd_close[:T] if A >= T else np.r_[cmd_close, np.zeros(T - A, bool)]
    # 닫기 명령 시작 시점 (열림→닫힘 전환). 첫 step 이 닫힘이면 그것도 1회.
    close_starts = np.flatnonzero(cmd_close & ~np.r_[False, cmd_close[:-1]])
    n_switch = int((np.diff(np.sign(act[:, 6])) != 0).sum())

    # grasp 신호 (없으면 대용: 손끝이 트럭 가까이 + 손가락이 열림도 완전 닫힘(빈손 0.20)도 아님)
    if tr["grasp"] is not None:
        grasp = tr["grasp"][:T]
        grasp_src = "env"
    else:
        closed_on_obj = (np.abs(gp) < OPEN_ABS).all(axis=1) & (np.abs(gp) > 0.215).all(axis=1)
        grasp = (d3 < D_PROXY) & closed_on_obj
        grasp_src = "proxy"

    t_near = first_run(d3 < D_NEAR, 1)
    t_grasp = first_run(grasp, GRASP_K)
    air = truck[:, 2] > z0 + LIFT_DZ
    held = air & cmd_close_s & (d3 < D_HOLD)
    t_lift = first_run(held, LIFT_K)
    # 대안: grasp 신호가 켜진 채 정지 높이 + 5 cm 이상 3 step (낮게 들고 바로 휩쓴 경우)
    held2 = (truck[:, 2] > z_rest + LIFT_DZ_REST) & grasp & (d3 < D_HOLD)
    t_lift2 = first_run(held2, LIFT_K2)
    if t_lift2 >= 0 and (t_lift < 0 or t_lift2 < t_lift):
        t_lift = t_lift2
    held = held | held2
    # 상자 위 = 상자 몸체 좌표에서 트럭 중심이 입구(반폭 OPEN_HALF) 안, 들기 뒤 OVER_K step 연속
    in_open = (cheb_loc < OPEN_HALF) & (loc[:, 2] > -0.11)
    t_over = first_run(in_open & (np.arange(T) >= max(t_lift, 0)), OVER_K) if t_lift >= 0 else -1
    end_inside = bool(cheb_loc[-1] < OPEN_HALF and -0.11 < loc[-1, 2] < 0.12 and d3[-1] > D_HOLD)
    _hs = np.flatnonzero(held)
    t_h_last = int(_hs[-1]) if len(_hs) else -1
    _e = min(T, (t_h_last if t_h_last >= 0 else T - 1) + PUSH_WIN)
    box_knocked = bool(t_lift >= 0 and (box_tilt[t_lift:_e] > BOX_KNOCK_TILT).any())
    _a = max(t_lift, t_h_last - PUSH_WIN, 0)
    box_push_rel = bool(t_lift >= 0 and t_h_last >= 0 and (box_disp[_a: t_h_last + 5].max() - box_disp[_a]) > BOX_PUSH)

    if tr["success_t"] is not None:
        st = tr["success_t"][:T]
    else:
        st = (xy_tb < SUCC_XY) & (np.abs(dz_tb) < SUCC_DZ) & fingers_open
        if not tr["success"]:
            st = np.zeros(T, bool)
    t_success = first_run(st, 1)
    # 데이터셋은 attrs success 를 정답으로 쓴다 (마지막 obs 는 성공 직전 step 이라 재계산이 한 step 모자랄 수 있음)
    success = bool(tr["success"])

    # 첫 닫기(손끝이 트럭 근처일 때) 전에 트럭이 밀리거나 넘어졌는가
    near_close = [int(c) for c in close_starts if c < T and d3[min(c, T - 1)] < D_HOVER]
    t_first_close = near_close[0] if near_close else -1
    t_ref = t_first_close if t_first_close >= 0 else (t_grasp if t_grasp >= 0 else T)
    disp = np.linalg.norm(truck[: max(t_ref, 1), :2] - truck[0, :2], axis=1)
    dtilt = np.abs(tilt[: max(t_ref, 1)] - tilt[0])
    knocked_before_close = bool((disp > KNOCK_XY).any() or (dtilt > KNOCK_TILT).any())
    t_knock = int(np.flatnonzero((disp > KNOCK_XY) | (dtilt > KNOCK_TILT))[0]) if knocked_before_close else -1

    # 표시 열 (범주를 바꾸지 않음)
    eef_box_xy = np.linalg.norm((eef - box)[:, :2], axis=1)
    # 트럭 없이 손이 상자 0.12 m 안 — 트럭이 상자 밖(xy > 0.15 m)에 있을 때만 (상자에 넣은 뒤의 빈손은 제외)
    t_empty_box = first_run((~air) & (eef_box_xy < 0.12) & (d3 > D_HOLD) & (xy_tb > 0.15), 5)
    shallow = bool(held.any() and np.median(d3[held]) > 0.05)  # 들고 있는 동안 손끝-트럭 중앙값 > 5 cm
    tail_idle = float((np.abs(act[-300:, :6]) < IDLE_EPS).all(axis=1).mean()) if A >= 50 else 0.0
    rot60 = float(np.abs(act[t_lift:t_lift + 60, 3:6]).sum(1).mean()) if 0 <= t_lift < A else float("nan")

    # 최종 상태
    end_in_box = bool(xy_tb[-1] < SUCC_XY and abs(dz_tb[-1]) < SUCC_DZ)
    end_in_hand = bool(grasp[-1] or (air[-1] and cmd_close_s[-1] and d3[-1] < D_HOLD))
    end_tilt = float(np.rad2deg(tilt[-1]))

    # 범주 = 가장 멀리 간 단계. 뒤 단계에 도달했으면 앞 단계는 지난 것으로 본다
    # (예: 손끝-트럭 최소 거리가 0.0502 m 라 t_near 는 없지만 트럭을 들어 올린 rollout → 운반 단계)
    reached_near = t_near >= 0 or t_grasp >= 0 or t_lift >= 0
    cat, sub = "success", ""
    if success:
        cat = "success"
    elif t_over >= 0 or (t_lift >= 0 and end_inside):
        cat = "place"
        opened_after = np.flatnonzero((np.arange(A) >= max(t_over, t_lift)) & ~cmd_close)
        if tr["end_reason"] == "dropped":
            sub = "off_table"
        elif end_in_box and not fingers_open[-1]:
            sub = "in_box_not_open"
        elif end_inside:
            sub = "in_box_offcenter"  # 상자 안에 들어갔지만 중심 0.10 m 밖 (안쪽 구석·벽에 기댐)
        elif len(opened_after) == 0:
            sub = "no_release"
        else:
            t_rel = int(opened_after[0])
            if xy_tb[min(t_rel, T - 1)] >= SUCC_XY:
                sub = "released_outside" if not bool(cmd_close_s[min(t_h_last + 1, T - 1)]) or t_h_last < 0 else "dropped_outside"
            else:
                sub = "other"
    elif t_lift >= 0:
        cat = "transport"
        hs = np.flatnonzero(held[t_lift:]) + t_lift
        t_h = int(hs[-1]) if len(hs) else t_lift
        opened = not bool(cmd_close_s[min(t_h + 1, T - 1)])  # 놓친 직후 명령이 '열기'면 스스로 놓은 것
        last = hs[-20:] if len(hs) else np.array([t_lift])
        near_box = float(np.minimum(xy_tb[last], xy_tb0[last]).min()) < NEAR_BOX
        if tr["end_reason"] == "dropped":
            sub = "off_table"
        elif box_knocked:
            sub = "box_knocked_over"
        elif held[-1]:
            sub = "stall"
        elif box_push_rel:
            sub = "box_collision_" + ("released" if opened else "dropped")
        elif near_box:
            sub = "released_near_box" if opened else "dropped_near_box"
        else:
            sub = "released_early" if opened else "dropped"
    elif reached_near:
        cat = "grasp"
        if tr["end_reason"] == "dropped":
            sub = "pushed_off_table"
        elif t_grasp >= 0:
            sub = "slip"
        elif t_first_close < 0 and not any(c < T and d3[c] < D_NEAR * 1.5 for c in close_starts):
            sub = "no_close"
        else:
            sub = "miss"
    else:
        cat = "approach"
        if tr["end_reason"] == "dropped":
            sub = "pushed_off_table"
        elif float(box_lift.max()) > 0.05:
            sub = "wrong_object"  # 트럭 대신 상자를 잡아 듦
        elif knocked_before_close:
            sub = "knocked"
        elif t_empty_box >= 0:
            sub = "to_box_without_truck"  # 트럭을 잡지 않고(공중에서 닫는 등) 손이 상자로 감
        elif t_first_close >= 0:
            sub = "close_misaligned"
        elif tail_idle > 0.8:
            sub = "freeze"
        elif d3.min() < D_HOVER:
            sub = "hover"
        else:
            sub = "far"

    # 놓기 시점: 들기 뒤 첫 열기 명령 (운반 실패에서도 기록)
    t_release = -1
    if t_lift >= 0:
        op = np.flatnonzero((np.arange(A) >= max(t_over, t_lift)) & ~cmd_close)
        t_release = int(op[0]) if len(op) else -1

    idle = float((np.abs(act[:, :6]) < IDLE_EPS).all(axis=1).mean())
    path_len = float(np.linalg.norm(np.diff(eef, axis=0), axis=1).sum())
    n_close_near = len(near_close)
    g_int = grasp.astype(int)
    n_grasp_lost = int(((g_int[:-1] == 1) & (g_int[1:] == 0)).sum()) if T > 1 else 0

    row = OrderedDict(
        src=tr["src"], kind=tr["kind"], id=tr["id"], T=T, success=int(success),
        category=cat, category_ko=CAT_KO[cat], subtype=sub, end_reason=tr["end_reason"],
        t_near=t_near, t_first_close=t_first_close, t_grasp=t_grasp, t_lift=t_lift, t_over=t_over,
        t_release=t_release, t_success=t_success, t_knock=t_knock, t_empty_at_box=t_empty_box,
        n_close=len(close_starts), n_close_near=n_close_near, n_grip_switch=n_switch, n_grip_spike=n_spike,
        n_grasp_lost=n_grasp_lost, knocked_before_close=int(knocked_before_close),
        box_moved=int(bool(box_moved_t.any())), box_disp_max=round(float(box_disp.max()), 4),
        box_tilt_max_deg=round(float(np.rad2deg(box_tilt.max())), 1),
        empty_at_box=int(t_empty_box >= 0), shallow_grasp=int(shallow),
        rot60_after_lift=round(rot60, 4) if t_lift >= 0 else float("nan"),
        min_d3=round(float(d3.min()), 4), near_borderline=int(0.045 <= float(d3.min()) <= 0.06),
        max_lift=round(float(truck[:, 2].max() - z_rest), 4),
        min_xy_box_air=round(float(xy_tb[air].min()) if air.any() else float("nan"), 4),
        min_cheb_box_after_lift=round(float(cheb_loc[t_lift:].min()), 4) if t_lift >= 0 else float("nan"),
        end_xy_box=round(float(xy_tb[-1]), 4), end_xy_box0=round(float(xy_tb0[-1]), 4), end_dz_box=round(float(dz_tb[-1]), 4),
        end_in_box=int(end_in_box), end_inside_box=int(end_inside), end_in_hand=int(end_in_hand),
        end_tilt_deg=round(end_tilt, 1), end_fingers_open=int(bool(fingers_open[-1])), idle=round(idle, 3),
        tail_idle=round(tail_idle, 3), eef_path=round(path_len, 3),
        init_truck_x=round(float(truck[0, 0]), 4), init_truck_y=round(float(truck[0, 1]), 4),
        init_truck_yaw=round(float(np.rad2deg(quat_yaw(tr["truck_q"][0]))), 1),
        init_box_x=round(float(box[0, 0]), 4), init_box_y=round(float(box[0, 1]), 4),
        init_box_yaw=round(float(np.rad2deg(quat_yaw(tr["box_q"][0]))), 1),
        grasp_src=grasp_src,
    )
    return row


def plot_traj(tr, row, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eef, truck, box, act = tr["eef"], tr["truck"], tr["box"], tr["act"]
    T = len(truck)
    t = np.arange(T)
    d3 = np.linalg.norm(eef - truck, axis=1)
    xy_tb = np.linalg.norm((truck - box)[:, :2], axis=1)
    fig, ax = plt.subplots(4, 1, figsize=(9, 7.5), sharex=True)
    ax[0].plot(t, d3, label="eef–truck 3D")
    ax[0].plot(t, xy_tb, label="truck–box xy")
    ax[0].axhline(D_NEAR, ls=":", c="gray")
    ax[0].axhline(OVER_XY, ls="--", c="gray")
    ax[0].set_ylim(0, max(0.5, float(np.nanmax(xy_tb[: min(T, 5)]) + 0.05)))
    ax[0].set_ylabel("m")
    ax[0].legend(loc="upper right", fontsize=8)
    ax[1].plot(t, truck[:, 2] - truck[0, 2], label="truck z − z0")
    ax[1].plot(t, eef[:, 2] - truck[0, 2], label="eef z − z0", alpha=0.7)
    ax[1].axhline(LIFT_DZ, ls=":", c="gray")
    ax[1].set_ylabel("m")
    ax[1].legend(loc="upper right", fontsize=8)
    A = len(act)
    ax[2].plot(np.arange(A), act[:, 6], label="gripper cmd (+open/−close)")
    g = tr["grasp"] if tr["grasp"] is not None else None
    if g is not None:
        ax[2].plot(t, g[:T].astype(float) * 0.8, label="grasp signal", alpha=0.8)
    ax[2].legend(loc="upper right", fontsize=8)
    ax[3].plot(np.arange(A), np.abs(act[:, :3]).sum(1), label="|Δpos| sum")
    ax[3].plot(np.arange(A), np.abs(act[:, 3:6]).sum(1), label="|Δrot| sum", alpha=0.7)
    ax[3].legend(loc="upper right", fontsize=8)
    ax[3].set_xlabel("step")
    for k, c in (("t_near", "C0"), ("t_grasp", "C1"), ("t_lift", "C2"), ("t_over", "C3"), ("t_success", "C4")):
        if row[k] >= 0:
            for a in ax:
                a.axvline(row[k], c=c, lw=0.8, alpha=0.6)
    fig.suptitle(f"{row['src']} {row['id']}  →  {row['category']}/{row['subtype']}  (T={T}, end={row['end_reason']})", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def summarize(rows, title):
    n = len(rows)
    c = Counter(r["category"] for r in rows)
    lines = [f"## {title}  (n={n})"]
    for cat in CAT_KO:
        k = c.get(cat, 0)
        subs = Counter(r["subtype"] for r in rows if r["category"] == cat and r["subtype"])
        sub_s = ", ".join(f"{s} {v}" for s, v in subs.most_common())
        lines.append(f"  {CAT_KO[cat]:<3s} {cat:<9s} {k:4d} ({k / max(n, 1):5.1%})  {sub_s}")
    fails = [r for r in rows if not r["success"]]
    if fails:
        kb = sum(r["knocked_before_close"] for r in fails)
        lines.append(f"  실패 {len(fails)}건 중 닫기 전에 트럭을 밀거나 넘어뜨림: {kb}")
        lines.append(
            "  실패 중앙값: 닫기 명령(근처) {:.0f}회, 그리퍼 전환 {:.0f}회, idle {:.0%}, 손끝 이동 {:.2f} m".format(
                np.median([r["n_close_near"] for r in fails]), np.median([r["n_grip_switch"] for r in fails]),
                np.median([r["idle"] for r in fails]), np.median([r["eef_path"] for r in fails]),
            )
        )
    succ = [r for r in rows if r["success"]]
    if succ:
        lines.append(
            "  성공 중앙값: T {:.0f}, t_grasp {:.0f}, t_lift {:.0f}, t_over {:.0f}, 그리퍼 전환 {:.0f}회".format(
                np.median([r["T"] for r in succ]), np.median([r["t_grasp"] for r in succ]),
                np.median([r["t_lift"] for r in succ]), np.median([r["t_over"] for r in succ]),
                np.median([r["n_grip_switch"] for r in succ]),
            )
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--json", default=None, help="범주별 개수 요약 json")
    ap.add_argument("--plots", default=None, help="궤적별 시간축 그림 폴더")
    ap.add_argument("--max_plots", type=int, default=40)
    ap.add_argument("--plot_success", action="store_true", help="성공 궤적도 그림")
    args = ap.parse_args()

    all_rows, summ = [], {}
    for p in args.files:
        trajs = load_trajs(p)
        rows = [classify(tr) for tr in trajs]
        all_rows.extend(rows)
        print(summarize(rows, os.path.basename(p)))
        summ[os.path.basename(p)] = dict(
            n=len(rows), counts=dict(Counter(r["category"] for r in rows)),
            subtypes=dict(Counter(f"{r['category']}/{r['subtype']}" for r in rows if r["subtype"])),
        )
        if args.plots:
            os.makedirs(args.plots, exist_ok=True)
            k = 0
            for tr, r in zip(trajs, rows):
                if r["success"] and not args.plot_success:
                    continue
                if k >= args.max_plots:
                    break
                stem = os.path.splitext(os.path.basename(p))[0]
                plot_traj(tr, r, os.path.join(args.plots, f"{stem}__{r['id']}__{r['category']}_{r['subtype']}.png"))
                k += 1
    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {len(all_rows)} rows -> {args.csv}")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summ, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    sys.exit(main())
