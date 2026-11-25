"""Same-body earned movement retrieval from public camera and encoder windows."""

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as core


SCALE=np.array([.104,.068,.50,.89,.66,.166,.080,
                .36,.42,.42,.40,.55,.19,.23,.23,.20,.32])
ADVANCE=-1.40


def cue(x,q,ref):
    y=float(x[1]); s=float(x[4])
    t=float(np.clip((ADVANCE-y)/min(-.90,s),-.086,.46))
    key=np.r_[x[0]+t*x[3],x[2]+t*x[5]-4.54*t*t,x[3],s,x[5]-9.08*t,y,x[2],
              np.asarray(q)[:5],(np.asarray(ref)-q)[:5]]/SCALE
    return key,t


def limits(a):
    axes=a._arm_ids[1][2]
    return (a._joint_increments[axes]*(a.control_dt/a.model.opt.timestep),
            a._joint_low[axes],a._joint_high[axes])


def references(actions,ready,stride,lo,hi):
    result=np.empty((len(actions)+1,6)); result[0]=ready
    for i,u in enumerate(actions):result[i+1]=np.clip(result[i]+stride*u,lo,hi)
    return result


def data(a,ready,practice):
    stride,lo,hi=limits(a); rows=[]; lost=[]; supported=set(); failure_ids=set(); paid_shots=0
    for c,r in zip(practice["cards"],practice["rows"]):
        refs=references(r["actions"],ready,stride,lo,hi)
        awards=[v["near_contact"] for v in r["placements"]["delivered"] if v["near_landed"]]
        if len(awards)!=r["score"]["learner"]:raise ValueError("A native training reward was not bound to its real return")
        for trace in r["optical_earned"]:
            leg=int(trace["tag"][-1])-1; hit=awards[leg]["step"]
            eligible=[u for u in trace["frames"] if 3<=hit-u[0]<=22 and -1.55<u[1][1]<-.49]
            added=0
            for tick,x,q,ref,issued in eligible[::3]:
                if not (np.allclose(refs[tick],ref,rtol=0.,atol=1e-10) and np.allclose(refs[tick+1],issued,rtol=0.,atol=1e-10)):
                    raise ValueError("Causal training packet and issued controls differ")
                key,clock=cue(x,q,ref)
                tail=core.tail(refs[tick+1:tick+1+core.HORIZON],core.HORIZON)
                rows.append((key,clock,ref,tail,c["id"],leg+1)); added+=1
            if added:paid_shots+=1;supported.add(c["id"])
        offers=r["placements"]["delivered"]
        if offers and offers[-1]["near_bounce"] and not offers[-1]["near_landed"]:
            bounce=offers[-1]["near_bounce"]["step"]
            for t,x,q,ref,_ in r["optical_missed"][::3]:
                if bounce-2<=t<=bounce+22 and -1.55<x[1]<-.49:
                    key,_=cue(x,q,ref);lost.append(key);failure_ids.add(c["id"])
    if not rows:raise ValueError("No actual successful movement found")
    keys,times,ref,paths,ids,legs=zip(*rows,strict=True)
    fields=dict(keys=np.stack(keys),times=np.asarray(times),ref=np.asarray(ref),paths=np.stack(paths),
                releases=np.asarray(ids,dtype=np.int64),legs=np.asarray(legs,dtype=np.int64),
                misses=np.asarray(lost,dtype=float).reshape(-1,len(SCALE)))
    evidence=dict(rows=len(rows),paid_shots=paid_shots,earned_releases=sorted(supported),missed_windows=len(lost),
                  missed_releases=sorted(failure_ids))
    return fields,evidence


class Library:
    def __init__(self,fields,*,sham=False):
        self.fields={k:v.copy() for k,v in fields.items()}
        self.tree=cKDTree(fields["keys"])
        self.sham=sham
        self.assign=np.arange(len(fields["keys"]))
        if sham:
            # Identical successful path inventory and relative horizon, but
            # the movement-to-offer association is rotated in sorted random order.
            p=np.random.default_rng(901106).permutation(len(self.assign))
            self.assign[p]=np.roll(p,len(p)//2)

    def shifted(self,idx,ref,t):
        f=self.fields; j=self.assign[idx]
        tau=float(np.clip((f["times"][idx]-t)/core.DT,-1.6,1.6))
        ticks=np.arange(core.HORIZON); tix=np.clip(ticks+tau,0,core.HORIZON-1)
        src=f["paths"][j]; path=np.array([np.interp(tix,ticks,col) for col in src.T]).T
        align=np.clip(ref-f["ref"][j],[-.28,-.23,-.30,-.24,-.28,0.],[.28,.23,.30,.24,.28,0.])
        return path+.60*np.maximum(0.,1.-ticks/8.)[:,None]*align

    def propose(self,x,q,ref):
        k,t=cue(x,q,ref); f=self.fields
        dist,idx=self.tree.query(k,k=min(42,len(self.assign)))
        best={}
        for d,j in zip(np.atleast_1d(dist),np.atleast_1d(idx),strict=True):
            release=int(f["releases"][j])
            if d>2.70:break
            if release not in best:best[release]=(float(d),int(j))
            if len(best)>=8:break
        if len(best)<2:return []
        nodes=list(best.items()); out=[]
        for i,(origin,(dist,j)) in enumerate(nodes[:2]):
            if dist>2.20:break
            path=self.shifted(j,ref,t); mix=[origin]
            if not i:
                for next_origin,(other_dist,l) in nodes[1:]:
                    alt=self.shifted(l,ref,t)
                    # Do not average forehand/backhand or unrelated joint motion.
                    if other_dist<dist+.45 and np.max(abs(path[:12,:5]-alt[:12,:5]))<.38:
                        path=.76*path+.24*alt; mix.append(next_origin);break
            y=self.assign[j]; donor=int(f["releases"][y])
            out.append((path,dict(key_release=origin,issued_example_release=donor,episode=int(f["legs"][y]),
                                  distance=dist,support_releases=len(nodes),blend=mix,sham=self.sham)))
        return out
