import h5py, numpy as np
from scipy.spatial.transform import Rotation as R
L=15
def fit(path, maxdemo=500):
    f=h5py.File(path,'r'); X=[];Y=[];XR=[];YR=[]
    for k in list(f['data'])[:maxdemo]:
        g=f['data'][k]; a=g['actions'][()]; p=g['obs/eef_pos'][()]; q=g['obs/eef_quat'][()]
        T=len(a)
        dp=p[1:]-p[:-1]  # motion after action t (obs pre-step => p[t+1]-p[t] results from a[t])
        dr=(R.from_quat(q[1:][:,[1,2,3,0]])*R.from_quat(q[:-1][:,[1,2,3,0]]).inv()).as_rotvec()
        for t in range(L,T-1):
            for ax in range(3):
                X.append(a[t-L+1:t+1,ax][::-1]); Y.append(dp[t,ax])
                XR.append(a[t-L+1:t+1,3+ax][::-1]); YR.append(dr[t,ax])
    for nm,(X,Y) in {'pos':(X,Y),'rot':(XR,YR)}.items():
        X=np.array(X);Y=np.array(Y)
        h,*_=np.linalg.lstsq(X,Y,rcond=None)
        r2=1-np.var(Y-X@h)/np.var(Y)
        cen=np.sum(np.arange(L)*h)/np.sum(h)
        print(path.split('/')[-1],nm,'sum h (ss gain) %.3f  k*=1/gain %.1f  lag centroid %.1f steps  R2 %.3f'%(h.sum(),1/h.sum(),cen,r2))
        print('   h:',np.round(h,3))
fit('/home/cai/IsaacLab_RS/datasets/team12_kgh/toy2box_kgh.hdf5')
fit('/home/cai/IsaacLab_RS/datasets/team12_kgh/kgh_mimic_500.hdf5',150)
