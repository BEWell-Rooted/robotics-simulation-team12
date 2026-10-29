"""데이터셋(data/demo_k: 사람 시범·Mimic 생성물)의 demo 몇 개를 play_record.py 기록 형식(rollouts/rollout_k)으로 옮긴다.
render_record.py 가 기록된 장면 상태를 그대로 다시 세워 렌더하므로, 생성 시범을 정확한 상태로 볼 수 있다 (트랙 A, 1029).

  python -I plan_A_demo_to_record.py <dataset.hdf5> <out_record.hdf5> demo_0 demo_3 ...   (없으면 앞 5개)
"""
import sys

import h5py
import numpy as np

KEYS = ["eef_pos", "eef_quat", "gripper_pos", "toy_truck_positions", "toy_truck_orientations", "box_positions", "box_orientations"]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    with h5py.File(src, "r") as f, h5py.File(dst, "w") as g:
        D = f["data"]
        names = sys.argv[3:] or sorted(D, key=lambda s: int(s.split("_")[-1]))[:5]
        g.attrs["task"] = "Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0"
        g.attrs["horizon"] = 10000
        g.attrs["source"] = src
        for k, n in enumerate(names):
            d = D[n]
            T = d["actions"].shape[0]
            r = g.create_group(f"rollouts/rollout_{k}")
            for key in KEYS:
                r.create_dataset(f"obs/{key}", data=d[f"obs/{key}"][:])
            r.create_dataset("actions", data=d["actions"][:])
            if "datagen_info" in d["obs"]:
                r.create_dataset("grasp", data=d["obs/datagen_info/subtask_term_signals/grasp"][:])
            else:
                r.create_dataset("grasp", data=np.zeros(T, bool))
            succ = np.zeros(T, bool)
            succ[-1] = bool(d.attrs.get("success", False))
            r.create_dataset("success", data=succ)
            def cp(name, obj):
                if isinstance(obj, h5py.Dataset):
                    r.create_dataset(f"states/{name}", data=obj[:])
            d["states"].visititems(cp)
            r.attrs.update(dict(trial=k, length=T, success=bool(d.attrs.get("success", False)),
                                end_reason="success" if d.attrs.get("success", False) else "other", source_demo=n))
            print(f"rollout_{k} <- {n} (T={T})")


if __name__ == "__main__":
    main()
