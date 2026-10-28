import sys, numpy as np, h5py
def qmul(a,b):
    w1,x1,y1,z1=a.T; w2,x2,y2,z2=b.T
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2, w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],1)
f = h5py.File(sys.argv[1],'r')
ra=[];rach=[];sat=0;n=0;posk=[]
for d in f['data']:
    o=f['data'][d]['obs']; a=f['data'][d]['actions'][:]
    q=o['eef_quat'][:]; qc=q.copy(); qc[:,1:]*=-1
    dq=qmul(q[1:],qc[:-1]); ang=2*np.arccos(np.clip(np.abs(dq[:,0]),0,1))
    ra.append(np.linalg.norm(a[:-1,3:6],axis=1)); rach.append(ang)
    sat+=int(np.sum(np.abs(a[:,3:6])>=0.999)); n+=a.shape[0]*3
    p=o['eef_pos'][:]
    for k in (5,):
        if len(p)>k: posk.append(np.linalg.norm(p[k:]-p[:-k],axis=1))
ra=np.concatenate(ra); rach=np.concatenate(rach)
print(sys.argv[1].split('/')[-1], '|rot action| med %.3f p90 %.3f max %.3f; achieved rot/step med %.4f p90 %.4f; frac rot comps saturated %.4f; |p(t+5)-p(t)| med %.3f p90 %.3f'%(np.median(ra),np.percentile(ra,90),ra.max(),np.median(rach),np.percentile(rach,90),sat/n,np.median(np.concatenate(posk)),np.percentile(np.concatenate(posk),90)))
