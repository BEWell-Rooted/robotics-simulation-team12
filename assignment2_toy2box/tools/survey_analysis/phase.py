import h5py, numpy as np
from scipy.spatial.transform import Rotation as R
f=h5py.File('/home/cai/IsaacLab_RS/datasets/team12_kgh/kgh_mimic_500.hdf5','r')
rows=[]
print('gripper cmd values', np.unique(np.round(f['data/demo_0/actions'][:,6],2)))
for k in f['data']:
    g=f['data'][k]; a=g['actions'][()]; p=g['obs/eef_pos'][()]; q=g['obs/eef_quat'][()]; tz=g['obs/toy_truck_positions'][:,2]
    T=len(a)
    closed=a[:,6]<0
    lifted=tz>tz[0]+0.02
    first_close=np.argmax(closed) if closed.any() else T
    ph=np.zeros(T,int)  # 0 approach,1 grasp(closed not lifted),2 carry(closed lifted),3 release/after
    for t in range(T):
        if t<first_close: ph[t]=0
        elif closed[t] and lifted[t]: ph[t]=2
        elif closed[t]: ph[t]=1
        else: ph[t]=3
    dp2=np.vstack([p[2:]-p[:-2],np.zeros((2,3))])
    dr2=np.vstack([(R.from_quat(q[2:][:,[1,2,3,0]])*R.from_quat(q[:-2][:,[1,2,3,0]]).inv()).as_rotvec(),np.zeros((2,3))])
    for t in range(T-2):
        c=a[t,:3]; r=a[t,3:6]; n=np.linalg.norm(c); nr=np.linalg.norm(r)
        rows.append((ph[t],n,np.dot(dp2[t],c)/max(n,1e-9),nr,np.dot(dr2[t],r)/max(nr,1e-9),c[2]))
A=np.array(rows)
names=['approach','grasp(closed,not lifted)','carry(closed,lifted)','after open']
for i in range(4):
    s=A[:,0]==i
    big=s&(A[:,1]>=0.08); bigr=s&(A[:,3]>=0.2)
    print('%-26s steps %6d (%.1f%%) | pos>=0.08: %5.1f%% of phase, realized ratio med %.3f | rot>=0.2: %5.1f%%, realized ratio med %.3f | rot cmd med %.3f p90 %.3f'%(
        names[i],s.sum(),100*s.mean(),100*big.sum()/s.sum(),np.median(A[big,2]/A[big,1]) if big.any() else np.nan,
        100*bigr.sum()/s.sum(),np.median(A[bigr,4]/A[bigr,3]) if bigr.any() else np.nan, np.median(A[s,3]),np.percentile(A[s,3],90)))
s=(A[:,1]>=0.08); print('pos>=0.08 steps: frac with cmd z<-0.05: %.2f'%np.mean(A[s,5]<-0.05))
