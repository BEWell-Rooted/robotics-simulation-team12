"""Offline recompute of GMM gripper distribution on recorded rollouts (read-only)."""
import sys, json, h5py, numpy as np, torch
torch.set_num_threads(3)
import robomimic.utils.file_utils as FileUtils
ck, rec, out = sys.argv[1], sys.argv[2], sys.argv[3]
dev = torch.device("cpu")
pol, _ = FileUtils.policy_from_checkpoint(ckpt_path=ck, device=dev)
algo = pol.policy; algo.set_eval()
net = algo.nets["policy"]
KEYS = ["eef_pos","eef_quat","gripper_pos","toy_truck_positions","toy_truck_orientations","box_positions","box_orientations"]
f = h5py.File(rec, "r")
res = {}
with torch.no_grad():
    for name in f["rollouts"]:
        r = f["rollouts"][name]
        A = r["actions"][()]; T = A.shape[0]
        obs = {k: torch.from_numpy(r["obs"][k][()]).float() for k in KEYS}
        h = None
        pw, wsel, dsel, wmax, pexec = [], [], [], [], []
        for t in range(T):
            if t % 10 == 0:
                h = net.get_rnn_init_state(batch_size=1, device=dev)
            o = {k: obs[k][t:t+1] for k in KEYS}
            d, h = net.forward_train_step(o, rnn_state=h)
            w = torch.softmax(d.mixture_distribution.logits, -1)[0]          # (5,)
            mu = d.component_distribution.base_dist.loc[0]                    # (5,7)
            a = torch.from_numpy(A[t]).float()
            dist = (mu - a).norm(dim=-1)
            k = int(dist.argmin())
            popen = float((w * (mu[:, 6] >= 0).float()).sum())
            pw.append(popen); wsel.append(float(w[k])); dsel.append(float(dist[k])); wmax.append(float(w.max()))
        print(name, flush=True)
        res[name] = dict(T=T, popen=pw, wsel=wsel, dsel=dsel, wmax=wmax, grip=A[:, 6].tolist())
json.dump(res, open(out, "w"))
print("done", len(res))
