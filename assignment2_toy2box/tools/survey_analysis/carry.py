import h5py, numpy as np
from scipy.spatial.transform import Rotation as R
def run(path):
    f=h5py.File(path,'r'); out=[]
    for k in f['data']:
        g=f['data'][k]; a=g['actions'][()]; p=g['obs/eef_pos'][()]; q=g['obs/eef_quat'][()]; tz=g['obs/toy_truck_positions'][:,2]
        T=len(a); closed=a[:,6]<0; lifted=tz>tz[0]+0.02
        m=closed&lifted; m[-2:]=False
        idx=np.where(m)[0]
        if len(idx)==0: continue
        dp=p[idx+2]-p[idx]; c=a[idx,:3]
        dr=(R.from_quat(q[idx+2][:,[1,2,3,0]])*R.from_quat(q[idx][:,[1,2,3,0]]).inv()).as_rotvec(); cr=a[idx,3:6]
        nm=np.linalg.norm(dp,axis=1); nc=np.linalg.norm(c,axis=1)
        cos=np.sum(dp*c,1)/np.maximum(nm*nc,1e-9)
        nmr=np.linalg.norm(dr,axis=1); ncr=np.linalg.norm(cr,axis=1); cosr=np.sum(dr*cr,1)/np.maximum(nmr*ncr,1e-9)
        out.append(np.c_[nm,nc,cos,nmr,ncr,cosr])
    O=np.concatenate(out)
    print(path.split('/')[-1],'carry steps',len(O))
    print('  pos: 2-step motion med %.4f | cmd med %.4f | cos(motion,cmd) med %.2f ; for cmd>=0.08: motion med %.4f cos med %.2f n=%d'%(np.median(O[:,0]),np.median(O[:,1]),np.median(O[:,2]),np.median(O[O[:,1]>=0.08,0]) if (O[:,1]>=0.08).any() else np.nan,np.median(O[O[:,1]>=0.08,2]) if (O[:,1]>=0.08).any() else np.nan,(O[:,1]>=0.08).sum()))
    s=O[:,4]>=0.2
    print('  rot: 2-step motion med %.4f | cmd med %.4f | cos med %.2f ; for cmd>=0.2: motion med %.4f cos med %.2f n=%d'%(np.median(O[:,3]),np.median(O[:,4]),np.median(O[:,5]),np.median(O[s,3]) if s.any() else np.nan,np.median(O[s,5]) if s.any() else np.nan,s.sum()))
run('/home/cai/IsaacLab_RS/datasets/team12_kgh/toy2box_kgh.hdf5')
run('/home/cai/IsaacLab_RS/datasets/team12_kgh/kgh_mimic_500.hdf5')
