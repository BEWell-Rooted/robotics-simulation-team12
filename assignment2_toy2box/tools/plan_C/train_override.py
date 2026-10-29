"""robomimic/train.py 를 고치지 않고 학습 seed 와 config 값 몇 개를 CLI 로 바꾸는 래퍼 (train_seed.py 확장).

  ./isaaclab.sh -p <A2>/tools/plan_C/train_override.py --seed 101 --set algo.gmm.num_modes=1 \
      --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --algo bc \
      --dataset ./datasets/team12_kgh/kgh_mimic_500.hdf5 --name C1mode_kgh500_s101

--set 은 여러 번 줄 수 있다. 값은 JSON 으로 읽는다(1, 0.5, true, [1000,1500], "str"). 경로가 json 에 없으면 멈춘다
(오타로 아무것도 안 바뀌는 일을 막음). 바뀐 값은 run 폴더 config.json 과 체크포인트 안 config 에 그대로 남으므로,
조교 play.py 가 체크포인트를 읽을 때 같은 정책 구조로 다시 만든다.
"""
import json
import os
import runpy
import sys

argv = sys.argv[1:]
seed = None
sets = []
out = []
i = 0
while i < len(argv):
    if argv[i] == "--seed":
        seed = int(argv[i + 1])
        i += 2
    elif argv[i] == "--set":
        k, v = argv[i + 1].split("=", 1)
        sets.append((k, json.loads(v)))
        i += 2
    else:
        out.append(argv[i])
        i += 1

_orig_load = json.load


def _load(fp, *a, **kw):
    obj = _orig_load(fp, *a, **kw)
    if isinstance(obj, dict) and "algo_name" in obj and isinstance(obj.get("train"), dict):
        if seed is not None:
            print(f"[train_override] train.seed {obj['train'].get('seed')} -> {seed}", flush=True)
            obj["train"]["seed"] = seed
        for path, val in sets:
            node = obj
            keys = path.split(".")
            for k in keys[:-1]:
                if k not in node:
                    raise SystemExit(f"[train_override] {path}: '{k}' 가 json 에 없다")
                node = node[k]
            if keys[-1] not in node:
                raise SystemExit(f"[train_override] {path}: '{keys[-1]}' 가 json 에 없다")
            print(f"[train_override] {path} {node[keys[-1]]!r} -> {val!r}", flush=True)
            node[keys[-1]] = val
    return obj


json.load = _load
script = os.path.join(
    os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab_RS")), "scripts/imitation_learning/robomimic/train.py"
)
sys.argv = [script] + out
runpy.run_path(script, run_name="__main__")
