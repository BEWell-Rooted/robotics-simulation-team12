"""트랙 C 사전 예측: baseline GMM 정책을 기록된 rollout 상태에 다시 넣어 step 마다
(1) 실제로 뽑혀 실행된 행동, (2) 혼합 평균 행동 Σ w_k μ_k (단일 모드 정책이 배우는 조건부 평균에 가까움),
(3) 그리퍼: 실행된 부호 vs 혼합 평균 부호 vs P(open)=Σ w_k 1[μ_k,grip ≥ 0], (4) 회전 성분의 모드 간 퍼짐을 저장한다.
Isaac 불필요 (CPU torch). 추론과 같게 10 step 마다 RNN 상태를 초기화한다.

  python -I gmm_offline.py <ckpt.pth> <record.hdf5> <out.npz>
"""
import sys

import h5py
import numpy as np
import torch

import robomimic.utils.file_utils as FileUtils

torch.set_num_threads(8)
ck, rec, out = sys.argv[1], sys.argv[2], sys.argv[3]
dev = torch.device("cpu")
pol, _ = FileUtils.policy_from_checkpoint(ckpt_path=ck, device=dev)
algo = pol.policy
algo.set_eval()
net = algo.nets["policy"]
H = algo.algo_config.rnn.horizon
KEYS = ["eef_pos", "eef_quat", "gripper_pos", "toy_truck_positions", "toy_truck_orientations", "box_positions", "box_orientations"]
res = {}
with h5py.File(rec, "r") as f, torch.no_grad():
    names = sorted(f["rollouts"], key=lambda s: int(s.split("_")[-1]))
    for name in names:
        r = f["rollouts"][name]
        A = r["actions"][()]
        T = A.shape[0]
        obs = {k: torch.from_numpy(r["obs"][k][()]).float() for k in KEYS}
        h = None
        mean_a = np.zeros((T, 7), np.float32)
        popen = np.zeros(T, np.float32)
        spread_rot = np.zeros(T, np.float32)
        wmax = np.zeros(T, np.float32)
        for t in range(T):
            if t % H == 0:
                h = net.get_rnn_init_state(batch_size=1, device=dev)
            o = {k: obs[k][t:t + 1] for k in KEYS}
            d, h = net.forward_train_step(o, rnn_state=h)
            w = torch.softmax(d.mixture_distribution.logits, -1)[0]
            mu = d.component_distribution.base_dist.loc[0]
            m = (w[:, None] * mu).sum(0)
            mean_a[t] = m.numpy()
            popen[t] = float((w * (mu[:, 6] >= 0).float()).sum())
            dev_rot = (mu[:, 3:6] - m[3:6]).norm(dim=-1)
            spread_rot[t] = float((w * dev_rot).sum())
            wmax[t] = float(w.max())
        res[name] = dict(exec_a=A, mean_a=mean_a, popen=popen, spread_rot=spread_rot, wmax=wmax)
        print(name, flush=True)
np.savez_compressed(out, **{f"{k}__{kk}": vv for k, v in res.items() for kk, vv in v.items()})
print("done", len(res))
