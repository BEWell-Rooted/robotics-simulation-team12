import csv, sys, h5py, numpy as np
def yaw(q):  # w,x,y,z
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
rows=[r for r in csv.DictReader(open(sys.argv[2])) if r['src']==sys.argv[3] and r['success']=='1']
f=h5py.File(sys.argv[1],'r'); d=f['data']
out=[]
for r in rows:
    g=d[r['id']]; tg=int(r['t_grasp']); tl=int(r['t_lift']); to=int(r['t_over']); trel=int(r['t_release'])
    if tg<0 or to<0: continue
    q=g['obs/eef_quat'][:]; p=g['obs/eef_pos'][:]; tp=g['obs/toy_truck_positions'][:]; bp=g['obs/box_positions'][:]
    tq=g['obs/toy_truck_orientations'][:]; bq=g['obs/box_orientations'][:]
    a=g['actions'][:]
    yw=np.unwrap(yaw(q))
    W=30
    tot=np.abs(yw[to]-yw[tg]); first=np.abs(yw[min(tg+W,to)]-yw[tg])
    # path-length of yaw in window vs whole carry
    dy=np.abs(np.diff(yw))
    tot_path=dy[tg:to].sum(); first_path=dy[tg:min(tg+W,to)].sum()
    # xy eef displacement in window, and z rise
    dxy=np.linalg.norm(p[min(tg+W,to),:2]-p[tg,:2]); dz=p[min(tg+W,to),2]-p[tg,2]
    # truck-box rel yaw at start and at release
    rel0=np.degrees(wrap(yaw(tq[0])-yaw(bq[0]))); relr=np.degrees(wrap(yaw(tq[trel if trel>0 else to])-yaw(bq[trel if trel>0 else to])))
    # grasp offset (truck center - eef) xy at t_lift
    off=np.linalg.norm(tp[tl,:2]-p[tl,:2]) if tl>=0 else np.nan
    # rotation action magnitude in window vs rest of carry
    rw=np.abs(a[tg:min(tg+W,to),3:6]).sum(1).mean(); rr=np.abs(a[min(tg+W,to):to,3:6]).sum(1).mean() if to>tg+W else np.nan
    pw=np.abs(a[tg:min(tg+W,to),0:3]).max()
    out.append([np.degrees(tot),np.degrees(first),np.degrees(tot_path),np.degrees(first_path),dxy,dz,rel0,relr,off,rw,rr,pw,to-tg])
o=np.array(out)
names=['|dyaw| grasp->over','|dyaw| first30','yaw path grasp->over','yaw path first30','eef dxy first30','eef dz first30','rel yaw start','rel yaw release','grasp xy offset','rot act first30','rot act rest','max |dpos act| first30','steps grasp->over']
print('n',len(o))
for i,n in enumerate(names):
    c=o[:,i]; c=c[~np.isnan(c)]
    print(f'{n:28s} med {np.median(c):8.3f} p10 {np.percentile(c,10):8.3f} p90 {np.percentile(c,90):8.3f}')
print('share of yaw path in first30 (median of per-demo ratio):', np.nanmedian(o[:,3]/np.maximum(o[:,2],1e-6)))
print('|rel yaw start| median', np.median(np.abs(o[:,6])), ' |rel yaw release| median', np.median(np.abs(o[:,7])))
