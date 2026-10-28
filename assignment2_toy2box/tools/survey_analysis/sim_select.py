import h5py, numpy as np
rng=np.random.default_rng(0)
f=h5py.File('/home/cai/IsaacLab_RS/datasets/team12_kgh/toy2box_kgh_annotated.hdf5','r'); d=f['data']
S=[]
for k in sorted(d.keys(), key=lambda s:int(s.split('_')[1])):
    g=d[k]['obs/datagen_info']; sig=g['subtask_term_signals/grasp'][:]; tg=int(np.argmax(sig>0))
    S.append(dict(tg=tg, eef=g['eef_pose/agibot'][:], tgt=g['target_eef_pose/agibot'][:], truck=g['object_pose/toy_truck'][:], box=g['object_pose/box'][:]))
def Rz(a):
    c,s=np.cos(a),np.sin(a); M=np.eye(4); M[:2,:2]=[[c,-s],[s,c]]; return M
def pose(x,y,z,a): M=Rz(a); M[:3,3]=[x,y,z]; return M
def rotang(R1,R2): return np.arccos(np.clip((np.trace(R1@R2.T)-1)/2,-1,1))
def yawof(M): return np.arctan2(M[1,0],M[0,0])
def wrap(a): return (a+np.pi)%(2*np.pi)-np.pi
inv=np.linalg.inv
def objdist(Tsrc,T): return np.linalg.norm(Tsrc[:3,3]-T[:3,3])+rotang(Tsrc[:3,:3],T[:3,:3])
zt=S[0]['truck'][0][2,3]; zb=S[0]['box'][0][2,3]
N=3000; out={k:[] for k in ['base','robot_k3','robot_k1','robot_rot_only_k3','translate','pseudo_yaw','yaw_hold','mininter_full']}
for n in range(N):
    Tt=pose(rng.uniform(0.45,0.80),rng.uniform(-0.30,-0.15),zt,rng.uniform(-np.pi,np.pi))
    Tb=pose(rng.uniform(0.85,0.95),rng.uniform(0.0,0.10),zb,rng.uniform(-np.pi,np.pi))
    offs=[int(rng.integers(2,11)) for _ in S]
    # subtask1 NN on truck top3
    dA=np.array([objdist(s['truck'][0],Tt) for s in S]); A=np.argsort(dA)[:3][rng.integers(3)]
    sA=S[A]; e1=sA['tg']+offs[A]
    last=Tt@inv(sA['truck'][0])@sA['tgt'][e1-1]          # last target of subtask1 (transformed)
    def first2(B, Tbox=Tb):
        sB=S[B]; s2=sB['tg']+offs[B]
        return Tbox@inv(sB['box'][s2])@sB['tgt'][s2]
    def rec(key,B,F):
        out[key].append((np.degrees(abs(wrap(yawof(F)-yawof(last)))), np.degrees(rotang(F[:3,:3],last[:3,:3])), np.linalg.norm(F[:3,3]-last[:3,3]), B))
    # baseline NN object on box top3
    dB=np.array([objdist(s['box'][s['tg']+offs[i]],Tb) for i,s in enumerate(S)]); B=np.argsort(dB)[:3][rng.integers(3)]
    rec('base',B,first2(B))
    # robot distance
    Fs=[first2(i) for i in range(len(S))]
    dr=np.array([np.linalg.norm(F[:3,3]-last[:3,3])+rotang(F[:3,:3],last[:3,:3]) for F in Fs])
    Bk=np.argsort(dr)[:3][rng.integers(3)]; rec('robot_k3',Bk,Fs[Bk])
    B1=int(np.argmin(dr)); rec('robot_k1',B1,Fs[B1])
    drr=np.array([rotang(F[:3,:3],last[:3,:3]) for F in Fs]); Br=np.argsort(drr)[:3][rng.integers(3)]; rec('robot_rot_only_k3',Br,Fs[Br])
    # translate (box NN, translation only, source world rotation)
    sB=S[B]; s2=sB['tg']+offs[B]; F=sB['tgt'][s2].copy(); F[:3,3]+=Tb[:3,3]-sB['box'][s2][:3,3]; rec('translate',B,F)
    # pseudo box yaw: rotate box frame so first-pose yaw matches current
    F0=first2(B); th=wrap(yawof(last)-yawof(F0)); Fp=first2(B,Tb@Rz(th)); rec('pseudo_yaw',B,Fp)
    # yaw hold: baseline positions, orientation = last
    Fh=F0.copy(); Fh[:3,:3]=last[:3,:3]; rec('yaw_hold',B,Fh)
    # full mininter: same source for both subtasks, w=0.5 frobenius; include initial approach from robot start pose
    best=None
    for i,s in enumerate(S):
        e1i=s['tg']+offs[i]; first1=Tt@inv(s['truck'][0])@s['eef'][0]; lasti=Tt@inv(s['truck'][0])@s['tgt'][e1i-1]
        F2=Tb@inv(s['box'][e1i])@s['tgt'][e1i]
        st=S[0]['eef'][0]
        c=lambda P,Q: 0.5*np.linalg.norm(P[:3,3]-Q[:3,3])+0.5*np.linalg.norm(P[:3,:3].T@Q[:3,:3]-np.eye(3))
        cost=c(st,first1)+c(lasti,F2)
        if best is None or cost<best[0]: best=(cost,i,lasti,F2)
    _,i,lasti,F2=best
    out['mininter_full'].append((np.degrees(abs(wrap(yawof(F2)-yawof(lasti)))), np.degrees(rotang(F2[:3,:3],lasti[:3,:3])), np.linalg.norm(F2[:3,3]-lasti[:3,3]), i))
for k,v in out.items():
    v=np.array(v); h=np.bincount(v[:,3].astype(int),minlength=20)
    print(f'{k:18s} |dyaw| med {np.median(v[:,0]):5.1f} p90 {np.percentile(v[:,0],90):5.1f} | rot med {np.median(v[:,1]):5.1f} | pos jump med {np.median(v[:,2]):.3f} p90 {np.percentile(v[:,2],90):.3f} | src max share {h.max()/len(v)*100:.0f}% top3 {np.sort(h)[-3:].sum()/len(v)*100:.0f}% unused {(h==0).sum()}')
