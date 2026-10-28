import json, csv, numpy as np
base='/home/cai/robotics-simulation-team12/assignment2_toy2box/results/baseline/'
EO={'released_near_box','released_early','box_collision_released'}
res={'EO':[], 'SUC':[], 'OUT':[]}
OUT={'released_outside','dropped_outside'}
for s in ['101','111']:
    G=json.load(open(f'gmm_s{s}.json'))
    rows={r['id']:r for r in csv.DictReader(open(base+f'base_kgh500_s{s}_e2000_seed42.csv'))}
    for k,g in G.items():
        r=rows[k]; tl=int(r['t_lift'])
        if tl<0: continue
        grip=np.array(g['grip']); po=np.array(g['popen'])
        idx=np.where(grip[tl:]>=0)[0]
        if len(idx)==0: continue
        t0=tl+idx[0]
        grp='EO' if r['subtype'] in EO else ('SUC' if r['success']=='1' else ('OUT' if r['subtype'] in OUT else None))
        if grp is None: continue
        carry=po[tl:t0]
        H=float(carry[carry<0.5].sum())       # expected # minority-open samples before t0
        first_maj=tl+int(np.argmax(po[tl:]>0.5)) if (po[tl:]>0.5).any() else -1
        res[grp].append(dict(s=s,k=k,t0=int(t0),p0=float(po[t0]),H=H,lag=(first_maj-t0) if first_maj>=0 else None,
                             pre=[round(float(x),2) for x in po[max(tl,t0-5):t0+1]], tover=int(r['t_over'])))
for grp,L in res.items():
    H=np.array([x['H'] for x in L]); p0=np.array([x['p0'] for x in L])
    lag=[x['lag'] for x in L]
    print(grp,'n',len(L),'H median %.3f mean %.3f'%(np.median(H),H.mean()),'| p0<0.5:',int((p0<0.5).sum()),
          '| first step P>0.5 after lift relative to t0: before', sum(1 for l in lag if l is not None and l<0),'same',sum(1 for l in lag if l==0),'after',sum(1 for l in lag if l is not None and l>0),'never',sum(1 for l in lag if l is None))
# successes with p0<0.5: trace
print('SUC p0<0.5 traces (last 6 steps up to t0), t0 - t_over:')
for x in [x for x in res['SUC'] if x['p0']<0.5][:15]: print('  ',x['s'],x['k'],x['pre'],'t0-tover',x['t0']-x['tover'] if x['tover']>=0 else None,'H %.2f'%x['H'])
print('EO traces:')
for x in res['EO']: print('  ',x['s'],x['k'],x['pre'],'H %.2f'%x['H'],'lag',x['lag'])
