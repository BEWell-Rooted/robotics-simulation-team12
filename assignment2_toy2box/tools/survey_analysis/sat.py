import h5py, numpy as np
from scipy.spatial.transform import Rotation as R
for path in ['/home/cai/IsaacLab_RS/datasets/team12_kgh/toy2box_kgh.hdf5','/home/cai/IsaacLab_RS/datasets/team12_kgh/kgh_mimic_500.hdf5']:
    f=h5py.File(path,'r'); C=[];M=[];CR=[];MR=[]
    for k in f['data']:
        g=f['data'][k]; a=g['actions'][()]; p=g['obs/eef_pos'][()]; q=g['obs/eef_quat'][()]
        dp2=p[2:]-p[:-2]   # motion over next 2 steps after a[t]
        dr2=(R.from_quat(q[2:][:,[1,2,3,0]])*R.from_quat(q[:-2][:,[1,2,3,0]]).inv()).as_rotvec()
        d=a[:-1,:3][:len(dp2)]; r=a[:-1,3:6][:len(dr2)]
        C.append(d); M.append(dp2); CR.append(r); MR.append(dr2)
    C=np.concatenate(C);M=np.concatenate(M);CR=np.concatenate(CR);MR=np.concatenate(MR)
    print(path.split('/')[-1])
    for nm,c,m,bins in [('pos',C,M,[0.01,0.03,0.05,0.08,0.12,0.2,0.4,1.01]),('rot',CR,MR,[0.01,0.03,0.06,0.1,0.2,0.4,0.7,1.01,1.8])]:
        n=np.linalg.norm(c,axis=1); proj=np.sum(m*c,axis=1)/np.maximum(n,1e-9)
        for lo,hi in zip(bins[:-1],bins[1:]):
            s=(n>=lo)&(n<hi)
            if s.sum()>30:
                print('  %s |cmd| %.2f-%.2f n=%6d  2-step motion along cmd med %.4f  ratio med %.3f'%(nm,lo,hi,s.sum(),np.median(proj[s]),np.median(proj[s]/n[s])))
