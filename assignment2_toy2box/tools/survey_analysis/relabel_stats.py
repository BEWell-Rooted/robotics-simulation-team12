import h5py, numpy as np
from scipy.spatial.transform import Rotation as R
f=h5py.File('/home/cai/IsaacLab_RS/datasets/team12_kgh/kgh_mimic_500.hdf5','r')
K=4
st={'orig_pos':[],'orig_rot':[],'new_pos':[],'new_rot':[],'carry_orig_rot':[],'carry_new_rot':[],'carry_orig_pos':[],'carry_new_pos':[],'cos_new_vs_orig_carry':[],'zero_frac':[]}
carry_frac=[]
for k in f['data']:
    g=f['data'][k]; a=g['actions'][()]; p=g['obs/eef_pos'][()]; q=g['obs/eef_quat'][()]; tz=g['obs/toy_truck_positions'][:,2]
    T=len(a); idx=np.minimum(np.arange(T)+K,T-1)
    dp=p[idx]-p; dr=(R.from_quat(q[idx][:,[1,2,3,0]])*R.from_quat(q[:,[1,2,3,0]]).inv()).as_rotvec()
    new=np.c_[np.clip(dp,-1,1),np.clip(dr,-1,1)]
    m=(a[:,6]<0)&(tz>tz[0]+0.02); carry_frac.append(m.mean())
    st['orig_pos'].append(np.linalg.norm(a[:,:3],axis=1)); st['orig_rot'].append(np.linalg.norm(a[:,3:6],axis=1))
    st['new_pos'].append(np.linalg.norm(new[:,:3],axis=1)); st['new_rot'].append(np.linalg.norm(new[:,3:],axis=1))
    st['carry_orig_rot'].append(np.linalg.norm(a[m,3:6],axis=1)); st['carry_new_rot'].append(np.linalg.norm(new[m,3:],axis=1))
    st['carry_orig_pos'].append(np.linalg.norm(a[m,:3],axis=1)); st['carry_new_pos'].append(np.linalg.norm(new[m,:3],axis=1))
    c=np.sum(a[m,:3]*new[m,:3],1)/np.maximum(np.linalg.norm(a[m,:3],axis=1)*np.linalg.norm(new[m,:3],axis=1),1e-9); st['cos_new_vs_orig_carry'].append(c)
    st['zero_frac'].append((np.abs(new).max(1)<0.01).astype(float))
for kk,v in st.items():
    v=np.concatenate(v); print('%-24s med %.3f p90 %.3f max %.3f'%(kk,np.median(v),np.percentile(v,90),v.max()))
print('carry fraction of steps med %.2f'%np.median(carry_frac))
