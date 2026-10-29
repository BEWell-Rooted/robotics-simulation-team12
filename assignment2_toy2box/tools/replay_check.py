"""데이터셋의 저장된 행동을 open-loop 로 재생하면서 매 step 상태를 play_record.py 와 같은 형식으로 남긴다. 트랙 B.

용도: 재라벨(relabel_actions.py) 행동이 "실행 가능한 라벨"인지 본다 — 원래 시작 상태에서 재생했을 때
원래 궤적을 얼마나 따라가는지(손끝 위치 편차), 트럭이 상자에 들어가는지(성공 함수), 손목이 얼마나 도는지.
같은 기계에서 원본 행동도 함께 재생해 대조로 쓴다 (Mimic 생성은 env 50개·다른 기계에서 했으므로 원본 재생도 완벽하지 않다).

scripts/tools/replay_demos.py 를 그대로 runpy 로 실행하고, gymnasium.make 를 감싸 base env 의 reset_to/step/close 만 가로챈다.
replay_demos.py 는 성공·종료 조건을 끄므로 성공은 태스크 cfg 의 성공 함수로 따로 계산한다(종료는 하지 않음).
기록은 rollouts/rollout_<k> (k = --select_episodes 순서), attrs: episode(데이터셋 번호), demo(이름), success(재생 중 한 번이라도 성공).

  ./isaaclab.sh -p <A2>/tools/replay_check.py --task Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0 \
      --dataset_file <relabeled.hdf5> --select_episodes 0 1 2 --headless --record_file <out.hdf5>
"""
import datetime
import os
import runpy
import sys

argv = sys.argv[1:]
record_file = None
out = []
i = 0
while i < len(argv):
    if argv[i] == "--record_file":
        record_file = os.path.abspath(argv[i + 1])
        i += 2
    else:
        out.append(argv[i])
        i += 1
if record_file is None:
    raise SystemExit("--record_file 이 필요하다")
if os.path.exists(record_file):
    raise SystemExit(f"{record_file} 가 이미 있다")
os.makedirs(os.path.dirname(record_file), exist_ok=True)

sel = []
if "--select_episodes" in out:
    j = out.index("--select_episodes") + 1
    while j < len(out) and not out[j].startswith("--"):
        sel.append(int(out[j]))
        j += 1
dataset = out[out.index("--dataset_file") + 1] if "--dataset_file" in out else "?"
task = out[out.index("--task") + 1] if "--task" in out else None

import gymnasium as gym  # noqa: E402

_orig_make = gym.make


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

    succ_cfg = parse_env_cfg(task or args[0], num_envs=1).terminations.success
    succ_func, succ_params = succ_cfg.func, dict(succ_cfg.params)
    robot = base.scene["robot"]
    grip_ids, _ = robot.find_joints(base.cfg.gripper_joint_names)
    grasp_kw = dict(robot_cfg=SceneEntityCfg("robot"), ee_frame_cfg=SceneEntityCfg("ee_frame"),
                    object_cfg=SceneEntityCfg("toy_truck"), diff_threshold=0.05)

    # replay_demos.py 는 --select_episodes 번호를 HDF5 키 순서(사전순: demo_0, demo_1, demo_10, …)로 해석한다
    with h5py.File(dataset, "r") as fd:
        names = list(fd["data"].keys())
    demo_names = [names[k] if k < len(names) else "?" for k in sel] if sel else names

    def _np(x):
        return x[0].detach().float().cpu().numpy().copy()

    def snap():
        s = {
            "obs/eef_pos": stack_mdp.ee_frame_pose_in_base_frame(base, return_key="pos"),
            "obs/eef_quat": stack_mdp.ee_frame_pose_in_base_frame(base, return_key="quat"),
            "obs/gripper_pos": stack_mdp.gripper_pos(base),
        }
        for name in ("toy_truck", "box"):
            cfg = SceneEntityCfg(name)
            s[f"obs/{name}_positions"] = place_mdp.object_poses_in_base_frame(base, object_cfg=cfg, return_key="pos")
            s[f"obs/{name}_orientations"] = place_mdp.object_poses_in_base_frame(base, object_cfg=cfg, return_key="quat")
        s = {k: _np(v) for k, v in s.items()}
        s["grasp"] = np.array(bool(place_mdp.object_grasped(base, **grasp_kw)[0]))
        s["success"] = np.array(bool(succ_func(base, **succ_params)[0]))
        d = base.scene["toy_truck"].data.root_pos_w - base.scene["box"].data.root_pos_w
        s["xy_dist"] = np.array(float(torch.linalg.vector_norm(d[0, :2])), np.float32)
        s["z_diff"] = np.array(float(d[0, 2]), np.float32)
        q = robot.data.joint_pos[0, grip_ids]
        s["grip_open"] = (torch.abs(torch.abs(q) - base.cfg.gripper_open_val) < base.cfg.gripper_threshold).cpu().numpy()
        return s

    h5 = h5py.File(record_file, "w")
    h5.attrs.update(dict(task=str(task), dataset=os.path.abspath(dataset), mode="open_loop_replay",
                         started=datetime.datetime.now().isoformat(timespec="seconds"), horizon=100000,
                         success_params=str(succ_params)))
    rec = {"buf": None, "acts": None, "k": 0}

    def flush():
        if rec["buf"] is None:
            return
        g = h5.create_group(f"rollouts/rollout_{rec['k']}")
        for key, seq in rec["buf"].items():
            g.create_dataset(key, data=np.stack(seq))
        T = len(rec["acts"])
        g.create_dataset("actions", data=np.asarray(rec["acts"], np.float32).reshape(T, -1))
        succ = bool(np.any(rec["buf"]["success"]))
        g.attrs.update(dict(trial=rec["k"], length=T, success=succ, end_reason="success" if succ else "replay_end",
                            episode=sel[rec["k"]] if rec["k"] < len(sel) else rec["k"],
                            demo=demo_names[rec["k"]] if rec["k"] < len(demo_names) else "?"))
        h5.flush()
        print(f"[replay_check] episode {rec['k']} ({g.attrs['demo']}): T={T} success={succ}", flush=True)
        rec["k"] += 1
        rec["buf"] = None

    orig_reset_to, orig_step, orig_close = base.reset_to, base.step, base.close

    def append(s):
        for k, v in s.items():
            rec["buf"].setdefault(k, []).append(v)

    def reset_to(*a, **kw):
        flush()
        res = orig_reset_to(*a, **kw)
        rec["buf"], rec["acts"] = {}, []
        append(snap())
        return res

    def step(action):
        res = orig_step(action)
        if rec["buf"] is not None:
            # replay_demos.py 는 CPU 텐서 하나를 제자리에서 바꿔 넘긴다 → 반드시 복사 (안 하면 모든 행이 마지막 행동이 됨)
            rec["acts"].append(action.detach().float().cpu().numpy().reshape(-1).copy())
            append(snap())
        return res

    def close(*a, **kw):
        flush()
        h5.attrs["num_recorded"] = rec["k"]
        h5.close()
        print(f"[replay_check] wrote {rec['k']} episodes -> {record_file}", flush=True)
        return orig_close(*a, **kw)

    base.reset_to, base.step, base.close = reset_to, step, close
    print(f"[replay_check] recording open-loop replay of {dataset} -> {record_file}", flush=True)
    return env


gym.make = _make_with_record
script = os.path.join(os.environ.get("ISAACLAB_PATH", os.path.expanduser("~/IsaacLab_RS")), "scripts/tools/replay_demos.py")
sys.argv = [script] + out
runpy.run_path(script, run_name="__main__")
