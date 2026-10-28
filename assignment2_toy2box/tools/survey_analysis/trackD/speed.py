import time, torch, h5py
torch.set_num_threads(2)
import robomimic.utils.file_utils as FileUtils
ck='/home/cai/IsaacLab_RS/logs/robomimic/Isaac-Place-Toy2Box-Agibot-Right-Arm-RmpFlow-v0/base_kgh500_s101/20261028195320/models/model_epoch_2000.pth'
pol,_=FileUtils.policy_from_checkpoint(ckpt_path=ck, device=torch.device('cpu'))
net=pol.policy.nets['policy']; pol.policy.set_eval()
KEYS=["eef_pos","eef_quat","gripper_pos","toy_truck_positions","toy_truck_orientations","box_positions","box_orientations"]
r=h5py.File('/home/cai/IsaacLab_RS/logs/rollout_records/base_kgh500_s101_e2000_seed42.hdf5','r')['rollouts/rollout_0']
obs={k:torch.from_numpy(r['obs'][k][()]).float() for k in KEYS}
t0=time.time()
with torch.no_grad():
  h=None
  for t in range(200):
    if t%10==0: h=net.get_rnn_init_state(1,torch.device('cpu'))
    d,h=net.forward_train_step({k:obs[k][t:t+1] for k in KEYS}, rnn_state=h)
print('per step ms',(time.time()-t0)/200*1000)
