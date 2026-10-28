import h5py, numpy as np, sys
from scipy.spatial.transform import Rotation as R
def load(path):
    f=h5py.File(path,'r'); out=[]
    for k in f['data']:
        g=f['data'][k]
        out.append(dict(a=g['actions'][()], la=g['obs/actions'][()], p=g['obs/eef_pos'][()], q=g['obs/eef_quat'][()],
                        tp=g['obs/toy_truck_positions'][()], gp=g['obs/gripper_pos'][()]))
    return out
for name,path in [('mimic','/home/cai/IsaacLab_RS/datasets/team12_kgh/kgh_mimic_500.hdf5'),('human','/home/cai/IsaacLab_RS/datasets/team12_kgh/toy2box_kgh.hdf5')]:
    D=load(path)
    d0=D[0]
    # alignment: obs/actions[t] == actions[t-1] ?
    print(name, 'align pre-step:', np.abs(d0['la'][1:]-d0['a'][:-1]).max(), 'same-step:', np.abs(d0['la']-d0['a']).max())
    A=np.concatenate([d['a'] for d in D])
    pn=np.linalg.norm(A[:,:3],axis=1); rn=np.linalg.norm(A[:,3:6],axis=1)
    print(name,'N steps',len(A),'cmd |dpos| med %.4f p90 %.4f'%(np.median(pn),np.percentile(pn,90)), '|dx| p90 %.4f'%np.percentile(np.abs(A[:,0]),90),
          'cmd |rot| med %.4f p90 %.4f max %.3f'%(np.median(rn),np.percentile(rn,90),np.abs(A[:,3:6]).max()),
          'frac any |rot comp|>=0.999: %.4f'%np.mean(np.abs(A[:,3:6]).max(1)>=0.999),
          'idle(all6==0) %.4f idle(|a|<1e-3) %.4f'%(np.mean(np.all(A[:,:6]==0,1)), np.mean(np.abs(A[:,:6]).max(1)<1e-3)))
    for k in [1,2,4,5,6,8,10,12]:
        mp=[];mr=[]
        for d in D:
            p,q=d['p'],d['q']
            if len(p)<=k: continue
            mp.append(np.linalg.norm(p[k:]-p[:-k],axis=1))
            Rt=R.from_quat(q[:-k][:,[1,2,3,0]]); Rk=R.from_quat(q[k:][:,[1,2,3,0]])
            mr.append(np.linalg.norm((Rk*Rt.inv()).as_rotvec(),axis=1))
        mp=np.concatenate(mp); mr=np.concatenate(mr)
        print('  k=%2d move med %.4f p90 %.4f | rot med %.4f p90 %.4f'%(k,np.median(mp),np.percentile(mp,90),np.median(mr),np.percentile(mr,90)))
