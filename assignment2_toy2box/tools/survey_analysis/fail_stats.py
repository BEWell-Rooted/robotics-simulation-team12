import csv, h5py, numpy as np, os
def yaw(q):
    w,x,y,z=q[...,0],q[...,1],q[...,2],q[...,3]
    return np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
root='/home/cai/IsaacLab_RS/datasets/team12_kgh/'
rows=list(csv.DictReader(open('/home/cai/robotics-simulation-team12/assignment2_toy2box/results/mimic/kgh_mimic_500_and_failed.csv')))
files={}
res={'success':[], 'fail':[]}; cat={}
for r in rows:
    src=r['src']; p=root+src if src=='kgh_mimic_500.hdf5' else root+'mimic_shards/'+src
    if p not in files: files[p]=h5py.File(p,'r')
    tg=int(r['t_grasp'])
    if tg<0: continue
    g=files[p]['data'][r['id']]; q=g['obs/eef_quat'][:]
    if len(q)<tg+31: continue
    yw=np.unwrap(yaw(q)); dy=np.degrees(abs(yw[tg+30]-yw[tg]))
    tq=g['obs/toy_truck_orientations'][0]; bq=g['obs/box_orientations'][0]
    rel=np.degrees(abs(wrap(yaw(tq)-yaw(bq))))
    k='success' if r['success']=='1' else 'fail'
    res[k].append((dy,rel)); cat.setdefault(r['category'],[]).append(dy)
for k,v in res.items():
    v=np.array(v); print(k,len(v),'bridge30 |dyaw| med %.1f p90 %.1f'%(np.median(v[:,0]),np.percentile(v[:,0],90)))
for k,v in cat.items(): print(' ',k,len(v),'med %.1f'%np.median(v))
a=np.array(res['success']+res['fail']); s=np.array([1]*len(res['success'])+[0]*len(res['fail']))
for lo,hi in [(0,20),(20,45),(45,90),(90,400)]:
    m=(a[:,0]>=lo)&(a[:,0]<hi); print(f'bridge30 |dyaw| {lo}-{hi}: n={m.sum()} gen success {s[m].mean()*100:.0f}%')
