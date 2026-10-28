"""Mimic 샤드(seed 별 generate_seed.py 출력)를 하나로 합친다. scripts/tools/merge_hdf5_datasets.py 와 같은 방식
(data/demo_k 를 순서대로 복사, env_args 복사)에 더해, robomimic 이 쓰는 data.attrs['total'] 을 채우고
샤드별 demo 수·성공 여부·seed 를 출력한다. 성공 시범만 합친다 (_failed 파일은 분석용).

  python -I merge_shards.py --out <합친.hdf5> [--limit 500] <shard1.hdf5> <shard2.hdf5> ...
"""
import argparse
import json
import os

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None, help="합칠 최대 demo 수 (샤드 순서대로 앞에서부터)")
    ap.add_argument("shards", nargs="+")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit(f"{a.out} 가 이미 있다")
    n = total = 0
    src = []
    with h5py.File(a.out, "w") as out:
        for p in a.shards:
            with h5py.File(p, "r") as f:
                D = f["data"]
                names = sorted(D, key=lambda s: int(s.split("_")[-1]))
                k_used = 0
                for k in names:
                    if a.limit is not None and n >= a.limit:
                        break
                    if not bool(D[k].attrs.get("success", True)):
                        continue
                    f.copy(f"data/{k}", out, f"data/demo_{n}")
                    total += int(D[k].attrs["num_samples"])
                    n += 1
                    k_used += 1
                if "env_args" not in out["data"].attrs:
                    out["data"].attrs["env_args"] = D.attrs["env_args"]
                src.append(dict(file=os.path.basename(p), demos=len(names), used=k_used))
                print(f"{os.path.basename(p)}: {len(names)} demos, used {k_used}")
        out["data"].attrs["total"] = total
        out["data"].attrs["merged_from"] = json.dumps(src)
    print(f"merged {n} demos, total {total} steps -> {a.out}")


if __name__ == "__main__":
    main()
