"""robomimic/play.py 를 고치지 않고 rollout 마다 매 step 상태를 HDF5 로 남기는 래퍼.

평가 숫자가 조교 play.py 와 같아야 하므로 정책·환경·난수 흐름은 건드리지 않는다.
gymnasium.make 를 감싸 base env 의 reset / step / _reset_idx / close 만 가로채고,
step 뒤에 장면 데이터를 읽기만 한다(난수 소비·상태 쓰기 없음).

기록 내용 (rollout 하나 = rollouts/rollout_<k>, k = play.py 의 trial 번호):
  obs/<key>        (T+1, d)  로봇 base 기준 관측. t=0 은 reset 직후, t=i+1 은 i번째 step 뒤.
                             key: eef_pos eef_quat gripper_pos toy_truck_positions toy_truck_orientations
                                  box_positions box_orientations (정책 입력과 같은 함수로 계산)
  actions          (T, 7)    정책이 낸 행동 (play.py 가 env.step 에 넘긴 값)
  grasp            (T+1,)    환경의 subtask 신호 object_grasped (diff 0.05, 두 손가락 힘 > 1 N)
  pad_force        (T+1, 2)  손가락 패드 접촉 힘 크기 (N)
  success          (T+1,)    play.py 와 같은 성공 함수(object_a_is_into_b, 태스크 cfg 의 params)
  xy_dist, z_diff  (T+1,)    트럭-상자 xy 거리, z 차 (성공 조건 성분)
  grip_open        (T+1, 2)  성공 조건의 그리퍼 열림 판정 (손가락 2개)
  states/...       (T+1, d)  scene.get_state(is_relative=True) — 나중에 렌더링 머신에서 재생용
  attrs: trial, length(T), success, end_reason(success | dropped | horizon | other)

트럭이 떨어져 종료되는 step 은 env 가 step 안에서 자동 reset 하므로, 그 step 의 상태는
_reset_idx 직전에 찍은 값으로 남긴다.

사용 (~/IsaacLab_RS 에서, play.py 인자 그대로 + --record_file):
  ./isaaclab.sh -p ~/robotics-simulation-team12/assignment2_toy2box/tools/play_record.py \
      --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 --checkpoint <pth> \
      --num_rollouts 100 --horizon 800 --seed 42 --headless \
      --record_file ./logs/rollout_records/<이름>.hdf5
  --no_states : states/ 그룹 생략 (파일 크기 절약)
"""
import datetime
import os
import runpy
import subprocess
import sys

argv = sys.argv[1:]
record_file = None
save_states = True
out = []
i = 0
while i < len(argv):
    a = argv[i]
    if a == "--record_file":
        record_file = argv[i + 1]
        i += 2
    elif a == "--no_states":
        save_states = False
        i += 1
    else:
        out.append(a)
        i += 1


def _arg(name, default=None):
    if name in out:
        return out[out.index(name) + 1]
    return default


if record_file is None:
    ck = os.path.splitext(os.path.basename(_arg("--checkpoint", "ckpt")))[0]
    record_file = f"./logs/rollout_records/{ck}_seed{_arg('--seed', '101')}.hdf5"
record_file = os.path.abspath(record_file)
if os.path.exists(record_file):
    print(f"[play_record] {record_file} 가 이미 있다. 덮어쓰지 않고 종료한다.", flush=True)
    sys.exit(2)
os.makedirs(os.path.dirname(record_file), exist_ok=True)

import gymnasium as gym  # noqa: E402

_orig_make = gym.make
OBS_KEYS = [
    "eef_pos", "eef_quat", "gripper_pos", "toy_truck_positions", "toy_truck_orientations",
    "box_positions", "box_orientations",
]


def _git_head(path):
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "?"


def _make_with_record(*args, **kwargs):
    env = _orig_make(*args, **kwargs)
    base = env.unwrapped

    import h5py
    import numpy as np
    import torch

    from isaaclab.managers import SceneEntityCfg
    from isaaclab_tasks.manager_based.manipulation.place import mdp as place_mdp
    from isaaclab_tasks.manager_based.manipulation.stack import mdp as stack_mdp
    from isaaclab_tasks.utils import parse_env_cfg

    task = args[0] if args else kwargs.get("id")
    # play.py 는 env_cfg.terminations.success 를 None 으로 바꾼 뒤 env 를 만든다 → 태스크 cfg 를 새로 읽어 params 를 얻는다
    succ_cfg = parse_env_cfg(task, num_envs=1).terminations.success
    succ_func, succ_params = succ_cfg.func, dict(succ_cfg.params)
    robot = base.scene["robot"]
    grip_ids, _ = robot.find_joints(base.cfg.gripper_joint_names)
    grasp_kw = dict(
        robot_cfg=SceneEntityCfg("robot"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg("toy_truck"), diff_threshold=0.05,
    )
    has_contact = "contact_grasp" in base.scene.keys()

    def _np(x):
        return x[0].detach().float().cpu().numpy().copy()

    def snap():
        s = {
            "obs/eef_pos": stack_mdp.ee_frame_pose_in_base_frame(base, return_key="pos"),
            "obs/eef_quat": stack_mdp.ee_frame_pose_in_base_frame(base, return_key="quat"),
            "obs/gripper_pos": stack_mdp.gripper_pos(base),
        }
        for name, key in (("toy_truck", "toy_truck"), ("box", "box")):
            cfg = SceneEntityCfg(name)
            s[f"obs/{key}_positions"] = place_mdp.object_poses_in_base_frame(base, object_cfg=cfg, return_key="pos")
            s[f"obs/{key}_orientations"] = place_mdp.object_poses_in_base_frame(base, object_cfg=cfg, return_key="quat")
        s = {k: _np(v) for k, v in s.items()}
        s["grasp"] = np.array(bool(place_mdp.object_grasped(base, **grasp_kw)[0]))
        s["pad_force"] = (
            _np(torch.linalg.vector_norm(base.scene["contact_grasp"].data.net_forces_w, dim=-1))
            if has_contact else np.zeros(2, np.float32)
        )
        s["success"] = np.array(bool(succ_func(base, **succ_params)[0]))
        d = base.scene["toy_truck"].data.root_pos_w - base.scene["box"].data.root_pos_w
        s["xy_dist"] = np.array(float(torch.linalg.vector_norm(d[0, :2])), np.float32)
        s["z_diff"] = np.array(float(d[0, 2]), np.float32)
        q = robot.data.joint_pos[0, grip_ids]
        s["grip_open"] = (torch.abs(torch.abs(q) - base.cfg.gripper_open_val) < base.cfg.gripper_threshold).cpu().numpy()
        if save_states:
            st = base.scene.get_state(is_relative=True)
            for g1, d1 in st.items():
                for g2, d2 in d1.items():
                    for g3, v in d2.items():
                        s[f"states/{g1}/{g2}/{g3}"] = _np(v)
        return s

    h5 = h5py.File(record_file, "w")
    meta = h5.attrs
    meta["task"] = str(task)
    meta["checkpoint"] = os.path.abspath(_arg("--checkpoint", "?"))
    meta["seed"] = int(_arg("--seed", 101))
    meta["horizon"] = int(_arg("--horizon", 800))
    meta["num_rollouts"] = int(_arg("--num_rollouts", 1))
    meta["isaaclab_rs_head"] = _git_head(os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab_RS")))
    meta["started"] = datetime.datetime.now().isoformat(timespec="seconds")
    meta["success_params"] = str(succ_params)
    meta["argv"] = " ".join(out)
    rec = {"buf": None, "acts": None, "k": 0, "pre": None, "term": False}

    def flush():
        if rec["buf"] is None:
            return
        T = len(rec["acts"])
        g = h5.create_group(f"rollouts/rollout_{rec['k']}")
        for key, seq in rec["buf"].items():
            arr = np.stack(seq)
            comp = "gzip" if key.startswith("states/") else None
            g.create_dataset(key, data=arr, compression=comp)
        g.create_dataset("actions", data=np.asarray(rec["acts"], np.float32).reshape(T, -1))
        succ = bool(rec["buf"]["success"][-1]) if T > 0 else False
        if succ:
            reason = "success"
        elif rec["term"]:
            reason = "dropped"
        elif T >= meta["horizon"]:
            reason = "horizon"
        else:
            reason = "other"
        g.attrs.update(dict(trial=rec["k"], length=T, success=succ, end_reason=reason))
        h5.flush()
        print(f"[play_record] rollout {rec['k']}: T={T} success={succ} end={reason}", flush=True)
        rec["k"] += 1
        rec["buf"] = None

    orig_reset, orig_step, orig_reset_idx, orig_close = base.reset, base.step, base._reset_idx, base.close

    def append(s):
        for k, v in s.items():
            rec["buf"].setdefault(k, []).append(v)

    def reset(*a, **kw):
        flush()
        res = orig_reset(*a, **kw)
        rec["buf"], rec["acts"], rec["term"] = {}, [], False
        append(snap())
        return res

    def _reset_idx(env_ids, *a, **kw):
        if rec["buf"] is not None:
            rec["pre"] = snap()
        return orig_reset_idx(env_ids, *a, **kw)

    def step(action):
        rec["pre"] = None
        res = orig_step(action)
        if rec["buf"] is not None:
            rec["acts"].append(action.detach().float().cpu().numpy().reshape(-1))
            terminated = bool(res[2][0]) if hasattr(res[2], "__getitem__") else bool(res[2])
            if rec["pre"] is not None:  # step 안에서 자동 reset 됨 → reset 직전 상태
                append(rec["pre"])
            else:
                append(snap())
            if terminated:
                rec["term"] = True
        return res

    def close(*a, **kw):
        flush()
        meta["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
        meta["num_recorded"] = rec["k"]
        h5.close()
        print(f"[play_record] wrote {rec['k']} rollouts -> {record_file}", flush=True)
        return orig_close(*a, **kw)

    base.reset, base.step, base._reset_idx, base.close = reset, step, _reset_idx, close
    print(f"[play_record] recording -> {record_file} (states={save_states}, success params {succ_params})", flush=True)
    return env


gym.make = _make_with_record

script = os.path.join(
    os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab_RS")), "scripts/imitation_learning/robomimic/play.py"
)
sys.argv = [script] + out
runpy.run_path(script, run_name="__main__")
