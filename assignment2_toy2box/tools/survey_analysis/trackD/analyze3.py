import json, csv, numpy as np
base='/home/cai/robotics-simulation-team12/assignment2_toy2box/results/baseline/'
EO={'released_near_box','released_early','box_collision_released'}
OUT={'released_outside','dropped_outside'}
pairs={'101':('gmm_s101.json','x_m111_on_r101.json','x_m101e1900_on_r101.json'),
       '111':('gmm_s111.json','x_m101_on_r111.json','x_m111e1900_on_r111.json')}
agg={}
for s,(own,other,snap) in pairs.items():
    G=json.load(open(own)); X=json.load(open(other)); S=json.load(open(snap))
    rows={r['id']:r for r in csv.DictReader(open(base+f'base_kgh500_s{s}_e2000_seed42.csv'))}
    for k,g in G.items():
        r=rows[k]; tl=int(r['t_lift'])
        if tl<0: continue
        grip=np.array(g['grip']); idx=np.where(grip[tl:]>=0)[0]
        if len(idx)==0: continue
        t0=tl+idx[0]
        grp='EO' if r['subtype'] in EO else ('SUC' if r['success']=='1' else ('OUT' if r['subtype'] in OUT else 'OTH'))
        po_own=g['popen'][t0]; po_x=X[k]['popen'][t0]; po_s=S[k]['popen'][t0]
        # other model's max P(open) over a +-2 window and own-carry hazard of other model before t0
        w=lambda P: max(P[max(tl,t0-2):t0+1])
        Hx=sum(p for p in X[k]['popen'][tl:t0] if p<0.5)
        agg.setdefault(grp,[]).append((s,k,round(po_own,3),round(po_x,3),round(w(X[k]['popen']),3),round(po_s,3),round(w(S[k]['popen']),3),round(Hx,2)))
for grp in ['EO','OUT','SUC','OTH']:
    L=agg.get(grp,[])
    if not L: continue
    own=np.array([x[2] for x in L]); ox=np.array([x[3] for x in L]); oxw=np.array([x[4] for x in L]); sn=np.array([x[5] for x in L]); snw=np.array([x[6] for x in L])
    print(f'{grp} n={len(L)} | own P>=0.5: {int((own>=0.5).sum())} | other-seed P(open)@t0 >=0.5: {int((ox>=0.5).sum())}, max(t0-2..t0)>=0.5: {int((oxw>=0.5).sum())} | ep1900 P@t0>=0.5: {int((sn>=0.5).sum())}, max win>=0.5: {int((snw>=0.5).sum())}')
    if grp in ('EO','OUT'):
        for x in L: print('   ',x)
