import h5py, numpy as np, csv
from scipy.spatial.transform import Rotation as R
def yaw(q): return R.from_quat(q[...,[1,2,3,0]]).as_euler('ZYX')[...,0]
B='/home/cai/robotics-simulation-team12/assignment2_toy2box/results/baseline/'
out=[]
for s in ['101','111']:
    rows={r['id']:r for r in csv.DictReader(open(B+f'base_kgh500_s{s}_e2000_seed42.csv'))}
    f=h5py.File(f'/home/cai/IsaacLab_RS/logs/rollout_records/base_kgh500_s{s}_e2000_seed42.hdf5','r')
    for rid,g in f['rollouts'].items():
        r=rows[rid]; tl=int(r['t_lift']); te=int(r['t_release']) if int(r['t_release'])>0 else int(r['T'])
        if tl<0: continue
        a=g['actions'][()]; q=g['obs/eef_quat'][()]
        te=max(min(te,len(a)),tl+1)
        wy=yaw(q[tl:te+1]); dy=np.degrees(np.abs(np.angle(np.exp(1j*(wy-wy[0])))).max())
        rc=np.linalg.norm(a[tl:te,3:6],axis=1); pc=np.linalg.norm(a[tl:te,:3],axis=1)
        out.append((int(r['success']),r['category'],dy,float(r['rot60_after_lift']),np.median(rc),np.median(pc),float(r['box_disp_max']),np.mean(pc>=0.08)))
O=np.array([(o[0],o[2],o[3],o[4],o[5],o[6],o[7]) for o in out])
cat=np.array([o[1] for o in out])
for nm,m in [('success',O[:,0]==1),('fail after lift',(O[:,0]==0)&np.isin(cat,['transport','place']))]:
    print('%-16s n=%3d  wrist yaw max change during carry med %.0f deg (IQR %.0f-%.0f) | rot60 med %.2f | carry rot cmd med %.3f | carry pos cmd med %.3f | frac carry pos cmd>=0.08 med %.2f'%(
      nm,m.sum(),np.median(O[m,1]),*np.percentile(O[m,1],[25,75]),np.median(O[m,2]),np.median(O[m,3]),np.median(O[m,4]),np.median(O[m,6])))
# success rate vs wrist-yaw-change bins
for lo,hi in [(0,15),(15,45),(45,90),(90,181)]:
    m=(O[:,1]>=lo)&(O[:,1]<hi); print('  carry wrist yaw %3d-%3d deg: n=%3d success %.2f'%(lo,hi,m.sum(),O[m,0].mean() if m.any() else np.nan))
