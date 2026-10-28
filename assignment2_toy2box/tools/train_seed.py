"""robomimic/train.py 를 고치지 않고 학습 seed 를 CLI 로 주는 래퍼.

train.py 에는 --seed 인자가 없고, seed 는 태스크에 등록된 json(bc_rnn_low_dim.json)의
train.seed(기본 101)로만 정해진다. json 은 IsaacLab_RS 추적 파일이라 고치지 않는다.
이 래퍼는 --seed 를 떼어 낸 뒤 json.load 를 감싸서, robomimic 설정 json(algo_name 과
train 키가 있는 dict)을 읽을 때만 train.seed 를 바꾸고 원래 train.py 를 그대로 실행한다.
바뀐 값은 run 폴더의 config.json 에 남으므로 거기서 확인한다.

사용 (~/IsaacLab_RS 에서):
  ./isaaclab.sh -p ~/robotics-simulation-team12/assignment2_toy2box/tools/train_seed.py \
      --seed 111 --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --algo bc \
      --dataset ./datasets/team12_kgh/kgh_mimic_500.hdf5 --name base_kgh500_s111

같은 --name 을 두 번 쓰면 robomimic 이 overwrite 를 묻고 y 면 이전 run 을 지우므로 run 마다 이름을 다르게 준다.
"""
import json
import os
import runpy
import sys

argv = sys.argv[1:]
seed = None
out = []
i = 0
while i < len(argv):
    if argv[i] == "--seed":
        seed = int(argv[i + 1])
        i += 2
    else:
        out.append(argv[i])
        i += 1

_orig_load = json.load
_patched = []


def _load_with_seed(fp, *args, **kwargs):
    obj = _orig_load(fp, *args, **kwargs)
    if seed is not None and isinstance(obj, dict) and "algo_name" in obj and isinstance(obj.get("train"), dict):
        old = obj["train"].get("seed")
        obj["train"]["seed"] = seed
        _patched.append((getattr(fp, "name", "?"), old, seed))
        print(f"[train_seed] {getattr(fp, 'name', '?')}: train.seed {old} -> {seed}", flush=True)
    return obj


json.load = _load_with_seed
if seed is None:
    print("[train_seed] --seed 없음: json 의 train.seed 그대로 사용", flush=True)

script = os.path.join(
    os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab_RS")), "scripts/imitation_learning/robomimic/train.py"
)
sys.argv = [script] + out
runpy.run_path(script, run_name="__main__")
