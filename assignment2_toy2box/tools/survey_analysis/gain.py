import h5py, numpy as np
from scipy.spatial.transform import Rotation as R
f=h5py.File('/home/cai/IsaacLab_RS/datasets/team12_kgh/toy2box_kgh.hdf5','r')
gp=[];gr=[]
for k in f['data']:
    g=f['data'][k]; a=g['actions'][()]; p=g['obs/eef_pos'][()]; q=g['obs/eef_quat'][()]
    T=len(a)
    for t in range(12,T-1):
        win=a[t-12:t+1,:6]
        if np.all(win==win[-1]) and np.any(win[-1]!=0):
            c=win[-1]
            if np.any(c[:3]!=0) and np.all(c[3:]==0):
                dp=p[t+1]-p[t]; u=c[:3]/np.linalg.norm(c[:3])
                gp.append(np.dot(dp,u)/np.linalg.norm(c[:3]))
            if np.any(c[3:]!=0) and np.all(c[:3]==0):
                r=(R.from_quat(q[t+1][[1,2,3,0]])*R.from_quat(q[t][[1,2,3,0]]).inv()).as_rotvec()
                u=c[3:]/np.linalg.norm(c[3:]); gr.append(np.dot(r,u)/np.linalg.norm(c[3:]))
gp=np.array(gp); gr=np.array(gr)
print('pos steady n=%d gain med %.3f (k*=%.1f) IQR %.3f-%.3f'%(len(gp),np.median(gp),1/np.median(gp),*np.percentile(gp,[25,75])))
print('rot steady n=%d gain med %.3f (k*=%.1f) IQR %.3f-%.3f'%(len(gr),np.median(gr),1/np.median(gr) if len(gr) else 0,*(np.percentile(gr,[25,75]) if len(gr) else [0,0])))
# idle in mimic with thresholds
f2=h5py.File('/home/cai/IsaacLab_RS/datasets/team12_kgh/kgh_mimic_500.hdf5','r')
A=np.concatenate([f2['data'][k]['actions'][()] for k in f2['data']])
for th in [0.005,0.01,0.02]:
    print('mimic idle max|a6|<%.3f: %.4f'%(th,np.mean(np.abs(A[:,:6]).max(1)<th)))
print('p90 max-axis |dpos| %.4f; p90 |dx| %.4f |dy| %.4f |dz| %.4f'%(np.percentile(np.abs(A[:,:3]).max(1),90),*[np.percentile(np.abs(A[:,i]),90) for i in range(3)]))
