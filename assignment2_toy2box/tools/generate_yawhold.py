"""Mimic 생성(generate_dataset.py)을 고치지 않고 상자 구간(subtask 2)의 손끝 목표 자세 변환만 바꾸는 래퍼. (트랙 A, 1029)

배경: Toy2Box Mimic cfg 의 subtask 2 는 object_ref="box" 라, 원본 시범의 "상자에 대한 손끝 자세" 를 새 상자 자세로
그대로 옮긴다(transform_source_data_segment_using_object_pose, SE(3)). 원본 시범은 subtask 1(트럭 기준,
nearest_neighbor_object)에서 한 번만 고르고 상자 구간에도 같은 원본을 쓴다 — cfg 의 generation_select_src_per_subtask=True
여도 generation_select_src_per_arm=False 라 이미 고른 원본을 다시 쓰기 때문이다(data_generator.py need_source_demo_selection).
그래서 baseline 의 운반 중 회전 = 새 상자 yaw − 원본 상자 yaw (상자 yaw ±180° 균일) → 손목 yaw 변화 중앙값 43°.

상자 구간 변환 모드 (하나만, 2×2 의 칸):
                         위치: SE(3)(상자 기준 회전 포함)      위치: 평행이동만
  회전: SE(3)            (기본, baseline)                       —
  회전: 원본 world       --rotworld 1  (A-R, 귀속 팔)            --translate 1  (A-T, 본 처치)
  --translate 1  위치 = 원본 + (상자_새 − 상자_원본) 위치, 회전 = 원본 world 회전.
                 남는 회전은 "원본 트럭 yaw − 새 트럭 yaw" 뿐(원본을 트럭 최근접으로 고르므로 작음).
  --rotworld 1   위치 = SE(3) 변환 그대로, 회전만 원본 world 회전. A-T 와 회전 규칙이 같고 위치 규칙만 다르다.
  --yawhold 1    (A-Y, 옛 A1·대안) waypoint 회전만 R(t)·R(0)ᵀ·R_last (R_last = subtask 1 마지막 목표 회전), 위치 SE(3).
                 손목 yaw 상태는 없애지만 roll·pitch 추적 오차 명령·상자 밂이 늘어 '회전만' 을 떼어 내지 못한다(plan_A §3).
  (모두 0)       원래 동작 그대로 (--datagen_seed 만 적용, generate_seed.py 와 bit 동일 확인: seed 1 demo 20/20)
기타:
  --datagen_seed N   datagen_config.seed (cfg 고정값 1 대신)
  --src_log FILE     고른 원본 demo 번호를 jsonl 로 남긴다 (env_id, subtask, src). 위 이유로 subtask 0 선택만 찍힌다.

원본 선택·보간 step(15)·action noise(0.01)·그리퍼·subtask 1 은 어느 모드에서도 그대로다.

사용 (~/IsaacLab_RS 에서, generate_dataset.py 인자 + 위 옵션, 프로세스마다 TMPDIR 따로):
  TMPDIR=/tmp/rs_a2_tmp_x ./isaaclab.sh -p ~/robotics-simulation-team12/assignment2_toy2box/tools/generate_yawhold.py \
      --translate 1 --datagen_seed 1 --src_log ~/IsaacLab_RS/logs/rs_a2_runs/plan_A/AT_s1_src.jsonl \
      --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-Rel-Mimic-v0 --headless --num_envs 50 --generation_num_trials 130 \
      --input_file ./datasets/team12_kgh/toy2box_kgh_annotated.hdf5 \
      --output_file ./datasets/team12_kgh/plan_A/translate_shards/translate_s1_130.hdf5
"""
import json
import os
import runpy
import sys

argv = sys.argv[1:]
opt = {"--yawhold": 0, "--translate": 0, "--rotworld": 0, "--datagen_seed": None, "--src_log": None}
out = []
i = 0
while i < len(argv):
    if argv[i] in opt:
        opt[argv[i]] = argv[i + 1]
        i += 2
    else:
        out.append(argv[i])
        i += 1
YAWHOLD = int(opt["--yawhold"])
TRANSLATE = int(opt["--translate"])
ROTWORLD = int(opt["--rotworld"])
SEED = None if opt["--datagen_seed"] is None else int(opt["--datagen_seed"])
SRC_LOG = opt["--src_log"]
if YAWHOLD + TRANSLATE + ROTWORLD > 1:
    raise SystemExit("--yawhold / --translate / --rotworld 는 하나만 쓴다 (한 요인)")
BOX_SUBTASK = 1  # subtask_configs["agibot"][1] = 상자 구간

import gymnasium as gym  # noqa: E402

_orig_make = gym.make
_patched = {"done": False}
_stats = {"merge_calls": 0, "yawhold_applied": 0, "translate_applied": 0, "rotworld_applied": 0}


def _patch_datagen():
    if _patched["done"]:
        return
    import torch

    import isaaclab_mimic.datagen.data_generator as dg

    DG = dg.DataGenerator
    orig_merge = DG.merge_eef_subtask_trajectory
    orig_gen = DG.generate_eef_subtask_trajectory
    orig_select = DG.select_source_demo
    orig_transform = dg.transform_source_data_segment_using_object_pose
    state = {"subtask": None, "env_id": None}

    def merge(self, env_id, eef_name, subtask_index, prev_executed_traj, subtask_trajectory):
        _stats["merge_calls"] += 1
        if YAWHOLD and subtask_index == BOX_SUBTASK and prev_executed_traj:
            R_last = prev_executed_traj[-1].pose[:3, :3].clone()
            wps = [wp for seq in subtask_trajectory.waypoint_sequences for wp in seq.sequence]
            R0 = wps[0].pose[:3, :3].clone()
            M = R0.transpose(0, 1) @ R_last  # R'(t) = R(t) · R0ᵀ · R_last
            for wp in wps:
                wp.pose[:3, :3] = wp.pose[:3, :3] @ M
            _stats["yawhold_applied"] += 1
        return orig_merge(self, env_id, eef_name, subtask_index, prev_executed_traj, subtask_trajectory)

    def gen(self, env_id, eef_name, subtask_ind, *a, **kw):
        state["subtask"], state["env_id"] = subtask_ind, env_id
        try:
            return orig_gen(self, env_id, eef_name, subtask_ind, *a, **kw)
        finally:
            state["subtask"] = None

    def transform(obj_pose, src_eef_poses, src_obj_pose):
        if TRANSLATE and state["subtask"] == BOX_SUBTASK:
            outp = src_eef_poses.clone()
            outp[:, :3, 3] += (obj_pose[:3, 3] - src_obj_pose[:3, 3])[None]
            _stats["translate_applied"] += 1
            return outp
        if ROTWORLD and state["subtask"] == BOX_SUBTASK:
            outp = orig_transform(obj_pose, src_eef_poses, src_obj_pose).clone()  # 위치는 SE(3) 결과
            outp[:, :3, :3] = src_eef_poses[:, :3, :3]  # 회전만 원본 world 회전 (A-T 와 같은 회전 규칙)
            _stats["rotworld_applied"] += 1
            return outp
        return orig_transform(obj_pose, src_eef_poses, src_obj_pose)

    def select(self, *a, **kw):
        idx = orig_select(self, *a, **kw)
        if SRC_LOG:
            with open(SRC_LOG, "a") as f:
                f.write(json.dumps({"env_id": int(state["env_id"]) if state["env_id"] is not None else -1,
                                    "subtask": state["subtask"], "src": int(idx)}) + "\n")
        return idx

    DG.merge_eef_subtask_trajectory = merge
    DG.generate_eef_subtask_trajectory = gen
    DG.select_source_demo = select
    dg.transform_source_data_segment_using_object_pose = transform
    _patched["done"] = True
    print(f"[generate_yawhold] patched DataGenerator: yawhold={YAWHOLD} translate={TRANSLATE} rotworld={ROTWORLD} src_log={SRC_LOG}", flush=True)
    del torch


def _make(*args, **kwargs):
    cfg = kwargs.get("cfg")
    if cfg is not None and hasattr(cfg, "datagen_config"):
        if SEED is not None:
            old = cfg.datagen_config.seed
            cfg.datagen_config.seed = SEED
            print(f"[generate_yawhold] datagen_config.seed {old} -> {SEED}", flush=True)
        sc = cfg.subtask_configs["agibot"][BOX_SUBTASK]
        print(f"[generate_yawhold] subtask {BOX_SUBTASK}: object_ref={sc.object_ref} interp={sc.num_interpolation_steps} "
              f"noise={sc.action_noise} interpolate_from_last_target_pose={cfg.datagen_config.generation_interpolate_from_last_target_pose}",
              flush=True)
        if YAWHOLD and not cfg.datagen_config.generation_interpolate_from_last_target_pose:
            print("[generate_yawhold] 경고: interpolate_from_last_target_pose=False 라 보간 시작이 실제 손끝 자세가 된다", flush=True)
    _patch_datagen()
    return _orig_make(*args, **kwargs)


gym.make = _make

import atexit  # noqa: E402

atexit.register(lambda: print(f"[generate_yawhold] stats {_stats}", flush=True))

script = os.path.join(
    os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab_RS")),
    "scripts/imitation_learning/isaaclab_mimic/generate_dataset.py",
)
sys.argv = [script] + out
runpy.run_path(script, run_name="__main__")
