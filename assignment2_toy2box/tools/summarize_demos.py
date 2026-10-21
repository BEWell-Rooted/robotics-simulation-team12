"""Toy2Box 시범 HDF5 요약 — Week 08 강의자료 summarize_demos.py 출력 형식.
사용: ./isaaclab.sh -p <이 파일> [HDF5 경로]  (기본 ./datasets/ta_ref/toy2box_agibot.hdf5)
len  = 시범 step 수, 괄호는 20 Hz 기준 초 (sim dt 1/60 x decimation 3)
sw   = 그리퍼 명령(actions 마지막 열, ±1) 값이 바뀐 횟수
idle = 팔 명령(actions 앞 6열)이 전부 0인 step 비율
ok   = 에피소드 attrs['success']
"""
import sys
import h5py
import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "./datasets/ta_ref/toy2box_agibot.hdf5"
HZ = 20.0
with h5py.File(path, "r") as f:
    d = f["data"]
    keys = sorted(d.keys(), key=lambda s: int(s.split("_")[1]))
    total = sum(int(d[k].attrs["num_samples"]) for k in keys)
    print(f"episodes: {len(keys)} | total: {total}")
    for k in keys:
        a = d[k]["actions"][:]
        n = a.shape[0]
        g = a[:, -1]
        sw = int(np.count_nonzero(np.diff(g) != 0))
        idle = float(np.mean(np.all(a[:, :6] == 0, axis=1)))
        ok = int(bool(d[k].attrs.get("success", False)))
        print(f"{k:<8} len={n:>4} ({n / HZ:.1f}s) sw={sw:>2} idle={idle * 100:.0f}% ok={ok}")
