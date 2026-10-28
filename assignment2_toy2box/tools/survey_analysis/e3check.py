import h5py, numpy as np, csv
from scipy.spatial.transform import Rotation as R
def yaw(q): return R.from_quat(q[...,[1,2,3,0]]).as_euler('ZYX')[...,0]
f=h5py.File('/home/cai/IsaacLab_RS/datasets/team12_kgh/kgh_mimic_500.hdf5','r')
off=[]
for k in f['data']:
    g=f['data'][k]; a=g['actions'][()]; tp=g['obs/toy_truck_positions'][()]; bp=g['obs/box_positions'][()]; ep=g['obs/eef_pos'][()]
    cl=a[:,6]<0
    if not cl.any(): continue
    i0=np.argmax(cl); op=np.where(~cl[i0:])[0]
    if len(op)==0: continue
    tr=i0+op[0]
    off.append((np.linalg.norm(tp[tr,:2]-bp[tr,:2]),np.linalg.norm(ep[tr,:2]-bp[tr,:2])))
off=np.array(off); print('Mimic release: truck-box xy med %.3f p90 %.3f | eef-box xy med %.3f p90 %.3f'%(np.median(off[:,0]),np.percentile(off[:,0],90),np.median(off[:,1]),np.percentile(off[:,1],90)))
# baseline rollouts: predictability of carry wrist yaw change from start yaws
B='/home/cai/robotics-simulation-team12/assignment2_toy2box/results/baseline/'
Y=[];X=[]
for s in ['101','111']:
    rows={r['id']:r for r in csv.DictReader(open(B+f'base_kgh500_s{s}_e2000_seed42.csv'))}
    h=h5py.File(f'/home/cai/IsaacLab_RS/logs/rollout_records/base_kgh500_s{s}_e2000_seed42.hdf5','r')
    for rid,g in h['rollouts'].items():
        r=rows[rid]; tl=int(r['t_lift']); te=int(r['t_release']) if int(r['t_release'])>0 else int(r['T'])
        if tl<0: continue
        q=g['obs/eef_quat'][()]; te=max(min(te,len(q)-1),tl+1)
        Y.append(np.angle(np.exp(1j*(yaw(q[te])-yaw(q[tl])))))
        X.append((yaw(g['obs/toy_truck_orientations'][0]),yaw(g['obs/box_orientations'][0]),yaw(q[tl])))
Y=np.array(Y);X=np.array(X)
S=lambda t: np.c_[np.sin(t),np.cos(t)]
def knn(F,y,kk=5):
    D=((F[:,None]-F[None])**2).sum(-1); np.fill_diagonal(D,1e9)
    nn=np.argsort(D,1)[:,:kk]; pred=np.angle(np.exp(1j*y[nn]).mean(1))
    return 1-np.var(np.angle(np.exp(1j*(y-pred))))/np.var(y)
print('baseline rollouts n=%d signed carry wrist yaw change |med| %.0f deg'%(len(Y),np.degrees(np.median(np.abs(Y)))))
print('  kNN5 with box yaw %.2f | without box yaw %.2f'%(knn(np.c_[S(X[:,0]),S(X[:,1]),S(X[:,2])],Y),knn(np.c_[S(X[:,0]),S(X[:,2])],Y)))
