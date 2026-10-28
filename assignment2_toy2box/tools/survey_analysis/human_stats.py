import sys, h5py, numpy as np
def yaw(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
f=h5py.File(sys.argv[1],'r'); d=f['data']
R=[]
for k in sorted(d.keys(), key=lambda s:int(s.split('_')[1])):
    g=d[k]; s=g['obs/datagen_info/subtask_term_signals/grasp'][:]
    tg=int(np.argmax(s>0)) if s.any() else -1
    tp=g['obs/toy_truck_positions'][:]; bp=g['obs/box_positions'][:]; q=g['obs/eef_quat'][:]; p=g['obs/eef_pos'][:]
    tq=g['obs/toy_truck_orientations'][:]; bq=g['obs/box_orientations'][:]
    xy=np.linalg.norm(tp[:,:2]-bp[:,:2],axis=1)
    idx=np.where((np.arange(len(xy))>tg)&(xy<0.10))[0]; to=int(idx[0]) if len(idx) else len(xy)-1
    yw=np.unwrap(yaw(q)); W=30
    dist0=xy[0]
    R.append([np.degrees(abs(yw[to]-yw[tg])), np.degrees(abs(yw[min(tg+W,to)]-yw[tg])), np.linalg.norm(p[min(tg+W,to),:2]-p[tg,:2]), p[min(tg+W,to),2]-p[tg,2], np.degrees(wrap(yaw(tq[0])-yaw(bq[0]))), np.degrees(wrap(yaw(tq[to])-yaw(bq[to]))), to-tg, dist0])
R=np.array(R)
for i,n in enumerate(['|dyaw| grasp->over','|dyaw| first30','eef dxy first30','eef dz first30','rel yaw start','rel yaw at over','steps grasp->over','truck-box dist start']):
    c=R[:,i]; print(f'{n:24s} med {np.median(c):8.3f} p10 {np.percentile(c,10):8.3f} p90 {np.percentile(c,90):8.3f}')
print('|rel start| med',np.median(np.abs(R[:,4])),'|rel over| med',np.median(np.abs(R[:,5])))
print('rel yaw start sorted', np.sort(np.round(R[:,4])).astype(int))
