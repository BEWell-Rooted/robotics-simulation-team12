import json, csv, numpy as np, collections, sys
base='/home/cai/robotics-simulation-team12/assignment2_toy2box/results/baseline/'
EO={'released_near_box','released_early','box_collision_released'}
SLIP={'dropped','dropped_near_box'}
OUT={'released_outside','dropped_outside'}
for s in ['101','111']:
    G=json.load(open(f'gmm_s{s}.json'))
    rows={r['id']:r for r in csv.DictReader(open(base+f'base_kgh500_s{s}_e2000_seed42.csv'))}
    alld=np.concatenate([np.array(G[k]['dsel']) for k in G])
    print(f'== seed {s}: fidelity dsel median {np.median(alld):.2e} p99 {np.percentile(alld,99):.2e} max {alld.max():.2e}')
    allw=np.concatenate([np.array(G[k]['wmax']) for k in G]); print('  wmax median',np.median(allw),'frac wmax>0.99',(allw>0.99).mean(), 'frac<0.9',(allw<0.9).mean())
    # minority gripper steps: executed sign != majority
    tot=0; mino=0; carry_tot=0; carry_min=0
    per=collections.defaultdict(list)
    for k,g in G.items():
        r=rows[k]; grip=np.array(g['grip']); po=np.array(g['popen'])
        ex_open=grip>=0; maj_open=po>0.5
        m=(ex_open!=maj_open)
        tot+=len(m); mino+=m.sum()
        tl=int(r['t_lift'])
        if tl>=0:
            idx=[t for t in range(tl,len(grip)) if grip[t]>=0]
            t0=idx[0] if idx else None
            seg=slice(tl, t0 if t0 is not None else len(grip))
            carry_tot+=len(m[seg]); carry_min+=m[seg].sum()
            if t0 is not None:
                grp='EO' if r['subtype'] in EO else ('SUC' if r['success']=='1' else ('OUT' if r['subtype'] in OUT else 'OTHER'))
                pre=po[max(tl,t0-10):t0]
                per[grp].append((k,t0,round(po[t0],3),round(g['wsel'][t0],3),round(float(pre.max()) if len(pre) else -1,3), int(m[seg].sum())))
    print(f'  minority-sampled gripper steps: {mino}/{tot} ({mino/tot:.4f}); in carry before first open: {carry_min}/{carry_tot}')
    for grp in ['EO','OUT','SUC','OTHER']:
        L=per[grp]
        if not L: continue
        p=np.array([x[2] for x in L])
        print(f'  {grp} n={len(L)} P(open)@first-open: <0.5 {int((p<0.5).sum())}, 0.5-0.9 {int(((p>=0.5)&(p<0.9)).sum())}, >=0.9 {int((p>=0.9).sum())}; median {np.median(p):.3f}')
        if grp in ('EO','OUT'):
            for x in L: print('     ',x)
