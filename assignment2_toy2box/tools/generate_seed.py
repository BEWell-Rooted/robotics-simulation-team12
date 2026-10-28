"""isaaclab_mimic/generate_dataset.py 를 고치지 않고 Mimic 생성 seed 를 CLI 로 주는 래퍼.

Toy2Box Mimic cfg 는 datagen_config.seed = 1 로 고정이라, 같은 입력으로 프로세스를 여러 개 돌리면
같은 시범이 중복된다. 생성은 CPU 1코어에 묶여(RMPFlow) 느리므로, seed 를 다르게 준 프로세스 여러 개로
나눠 돌린 뒤 scripts/tools/merge_hdf5_datasets.py 로 합친다.
이 래퍼는 --datagen_seed 를 떼어 내고 gymnasium.make 에 넘어가는 cfg 의 datagen_config.seed 만 바꾼다.

사용 (~/IsaacLab_RS 에서, generate_dataset.py 인자 그대로 + --datagen_seed):
  ./isaaclab.sh -p ~/robotics-simulation-team12/assignment2_toy2box/tools/generate_seed.py --datagen_seed 2 \
      --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --headless --num_envs 50 \
      --generation_num_trials 100 --input_file <annotated.hdf5> --output_file <shard_s2.hdf5>
"""
import os
import runpy
import sys

argv = sys.argv[1:]
seed = None
out = []
i = 0
while i < len(argv):
    if argv[i] == "--datagen_seed":
        seed = int(argv[i + 1])
        i += 2
    else:
        out.append(argv[i])
        i += 1

import gymnasium as gym  # noqa: E402

_orig_make = gym.make


def _make_with_seed(*args, **kwargs):
    cfg = kwargs.get("cfg")
    if seed is not None and cfg is not None and hasattr(cfg, "datagen_config"):
        old = cfg.datagen_config.seed
        cfg.datagen_config.seed = seed
        print(f"[generate_seed] datagen_config.seed {old} -> {seed}", flush=True)
    return _orig_make(*args, **kwargs)


gym.make = _make_with_seed

script = os.path.join(
    os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab_RS")),
    "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
)
sys.argv = [script] + out
runpy.run_path(script, run_name="__main__")
