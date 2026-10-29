"""Mimic 생성 데이터셋의 행동 라벨만 바꾼 사본을 만든다 (관측·상태·초기 상태는 그대로). 트랙 B (연구 축 C).

모드
  carry_norot   B1: 운반 구간 step 의 회전 라벨(actions[:,3:6])만 0. 위치·그리퍼 라벨은 bit 동일.
                운반 마스크 = 그리퍼 닫기 명령(a6<0) · 트럭 z > 처음 z + LIFT_DZ(0.03) · 손끝–트럭 < D_HOLD(0.08)
                (classify_failures.py 의 held 정의와 같은 상수).
  reach --k 4   B2: 위치·회전 라벨을 "k step 뒤 실제 도달 손끝 자세 − 현재" 로 바꾼다 (base 좌표, 회전은
                axis-angle(R_{t+k}·R_tᵀ) — Mimic 라벨과 같은 규약), ±1 로 자름. 그리퍼는 bit 동일.
                k = 4 는 사람 시범으로 잰 RMPFlow 정상 이득(위치 0.22–0.23, 회전 0.20–0.22)의 역수로 학습 전에 고정.
  cap           B3(구판, 교차 점검 #12 로 대체): 위치 노름 ≤ cap_pos(0.05), 회전 노름 ≤ cap_rot(0.20), 방향 유지.
  dir_orig_mag_b2  B3': step 마다 위치·회전을 따로, 방향은 원본·크기는 B2(reach k) — a' = a_orig/|a_orig| · |a_B2|
                (|a_orig| < 1e-6 이면 0). 크기 분포가 B2 와 같고 방향만 원본이라 B2 와 직접 비교해 "크기 vs 방향"을 가른다.
  reach_carry   B2-carry(해석용 예비): 들기가 들어 있는 '그리퍼 닫힘 구간'(파지 뒤 ~ 다시 열기 전)만 reach k 로,
                그 밖은 원본 bit 동일. 운반 감속·명령 일관성 효과를 접근 단계 변화와 떼어 본다.

data.attrs 에 relabel_* (모드·상수·바뀐 step 수·원본 md5)를, demo attrs 에 relabel_n_changed 를 남긴다.
끝에 검사: obs/·states/·initial_state/ 는 원본과 bit 동일, 바뀌지 말아야 할 행동 열도 bit 동일.

  python -I relabel_actions.py --mode carry_norot --src <kgh_mimic_500.hdf5> --out <..._carrynorot.hdf5>
  python -I relabel_actions.py --mode reach --k 4 --src ... --out ..._reach4.hdf5
  python -I relabel_actions.py --mode cap --src ... --out ..._cap.hdf5
  python -I relabel_actions.py --mode dir_orig_mag_b2 --k 4 --src ... --out ..._dirorig_magb2.hdf5
  python -I relabel_actions.py --mode reach_carry --k 4 --src ... --out ..._reach4carry.hdf5
"""
import argparse
import hashlib
import os
import shutil

import h5py
import numpy as np
from scipy.spatial.transform import Rotation as Rot

LIFT_DZ = 0.03
D_HOLD = 0.08


def md5(path, chunk=1 << 22):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def carry_mask(g):
    o = g["obs"]
    a = g["actions"][:]
    truck = o["toy_truck_positions"][:]
    eef = o["eef_pos"][:]
    d3 = np.linalg.norm(eef - truck, axis=1)
    return (a[:, 6] < 0) & (truck[:, 2] > truck[0, 2] + LIFT_DZ) & (d3 < D_HOLD)


def reach_labels(g, k):
    o = g["obs"]
    p = o["eef_pos"][:].astype(np.float64)
    q = o["eef_quat"][:].astype(np.float64)  # (w,x,y,z)
    T = len(p)
    j = np.minimum(np.arange(T) + k, T - 1)
    R = Rot.from_quat(q[:, [1, 2, 3, 0]])
    dpos = p[j] - p
    drot = (R[j] * R.inv()).as_rotvec()  # base 좌표 R_{t+k} R_t^T
    return np.clip(np.concatenate([dpos, drot], 1), -1.0, 1.0).astype(np.float32)


def cap_labels(a, cap_pos, cap_rot):
    a = a.copy()
    for sl, c in ((slice(0, 3), cap_pos), (slice(3, 6), cap_rot)):
        v = a[:, sl]
        n = np.linalg.norm(v, axis=1, keepdims=True)
        s = np.minimum(1.0, c / np.maximum(n, 1e-12))
        a[:, sl] = (v * s).astype(np.float32)
    return a


def dir_orig_mag(a, mag_src, eps=1e-6):
    """방향은 a(원본), 크기는 mag_src(B2) — 위치(0:3)·회전(3:6) 따로."""
    out = a.copy()
    for sl in (slice(0, 3), slice(3, 6)):
        v = a[:, sl].astype(np.float64)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        m = np.linalg.norm(mag_src[:, sl].astype(np.float64), axis=1, keepdims=True)
        unit = np.where(n > eps, v / np.maximum(n, eps), 0.0)
        out[:, sl] = (unit * m).astype(np.float32)
    return out


def closed_run_with_lift(g):
    """들기(운반 마스크 첫 step)가 들어 있는 '그리퍼 닫기 명령' 연속 구간 [s, e) — 없으면 None."""
    a = g["actions"][:]
    m = carry_mask(g)
    idx = np.flatnonzero(m)
    if not len(idx):
        return None
    t = int(idx[0])
    closed = a[:, 6] < 0
    s = t
    while s > 0 and closed[s - 1]:
        s -= 1
    e = t
    while e < len(a) and closed[e]:
        e += 1
    return s, e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["carry_norot", "reach", "cap", "dir_orig_mag_b2", "reach_carry"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cap_pos", type=float, default=0.05)
    ap.add_argument("--cap_rot", type=float, default=0.20)
    args = ap.parse_args()
    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} 가 이미 있다")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    src_md5 = md5(args.src)
    shutil.copyfile(args.src, args.out)
    n_tot = n_chg = 0
    with h5py.File(args.out, "r+") as f:
        D = f["data"]
        names = sorted(D, key=lambda s: int(s.split("_")[-1]))
        for n in names:
            g = D[n]
            a = g["actions"][:]
            new = a.copy()
            if args.mode == "carry_norot":
                m = carry_mask(g)
                new[m, 3:6] = 0.0
                chg = int(m.sum())
            elif args.mode == "reach":
                new[:, :6] = reach_labels(g, args.k)
                chg = int((np.abs(new[:, :6] - a[:, :6]) > 1e-7).any(1).sum())
            elif args.mode == "cap":
                new = cap_labels(a, args.cap_pos, args.cap_rot)
                chg = int((np.abs(new[:, :6] - a[:, :6]) > 1e-7).any(1).sum())
            elif args.mode == "dir_orig_mag_b2":
                new = dir_orig_mag(a, reach_labels(g, args.k))
                chg = int((np.abs(new[:, :6] - a[:, :6]) > 1e-7).any(1).sum())
            else:  # reach_carry
                rng = closed_run_with_lift(g)
                if rng is not None:
                    s0, e0 = rng
                    new[s0:e0, :6] = reach_labels(g, args.k)[s0:e0]
                    g.attrs["relabel_range"] = np.array([s0, e0])
                chg = int((np.abs(new[:, :6] - a[:, :6]) > 1e-7).any(1).sum())
            del g["actions"]
            g.create_dataset("actions", data=new)
            g.attrs["relabel_n_changed"] = chg
            n_tot += len(a)
            n_chg += chg
        D.attrs["relabel_mode"] = args.mode
        D.attrs["relabel_src"] = os.path.abspath(args.src)
        D.attrs["relabel_src_md5"] = src_md5
        D.attrs["relabel_steps_changed"] = n_chg
        D.attrs["relabel_steps_total"] = n_tot
        if args.mode == "carry_norot":
            D.attrs["relabel_mask"] = f"a6<0 & truck_z > truck_z[0]+{LIFT_DZ} & |eef-truck| < {D_HOLD}; actions[:,3:6]=0"
        elif args.mode == "reach":
            D.attrs["relabel_k"] = args.k
            D.attrs["relabel_def"] = "actions[:,0:3]=p[t+k]-p[t]; actions[:,3:6]=rotvec(R[t+k] R[t]^T) (base frame); clip +-1"
        elif args.mode == "cap":
            D.attrs["relabel_cap"] = f"pos norm <= {args.cap_pos}, rot norm <= {args.cap_rot} (direction kept)"
        elif args.mode == "dir_orig_mag_b2":
            D.attrs["relabel_k"] = args.k
            D.attrs["relabel_def"] = "per step, pos and rot separately: a' = a_orig/|a_orig| * |reach_k| (0 if |a_orig|<1e-6)"
        else:
            D.attrs["relabel_k"] = args.k
            D.attrs["relabel_def"] = "reach_k labels only on the closed-gripper run containing the lift; elsewhere original"

    # 검사: 관측·상태는 bit 동일, 바뀌지 말아야 할 행동 열도 bit 동일
    keep_cols = {"carry_norot": [0, 1, 2, 6], "reach": [6], "cap": [6], "dir_orig_mag_b2": [6], "reach_carry": [6]}[args.mode]
    bad = 0
    with h5py.File(args.src, "r") as fs, h5py.File(args.out, "r") as fo:
        for n in fs["data"]:
            gs, go = fs["data"][n], fo["data"][n]

            def walk(a, b, pre=""):
                nonlocal bad
                for k in a:
                    if isinstance(a[k], h5py.Dataset):
                        if not np.array_equal(a[k][:], b[k][:]):
                            bad += 1
                            print("MISMATCH", n, pre + k)
                    else:
                        walk(a[k], b[k], pre + k + "/")

            for grp in ("obs", "states", "initial_state"):
                if grp in gs:
                    walk(gs[grp], go[grp], grp + "/")
            if not np.array_equal(gs["actions"][:, keep_cols], go["actions"][:, keep_cols]):
                bad += 1
                print("MISMATCH actions cols", keep_cols, n)
    print(f"[relabel] mode={args.mode} steps changed {n_chg}/{n_tot} ({n_chg / max(n_tot, 1):.1%}) -> {args.out}")
    print(f"[relabel] 검사: {'통과' if bad == 0 else f'불일치 {bad}건'} (obs/states/initial_state 와 열 {keep_cols} bit 동일)")


if __name__ == "__main__":
    main()
