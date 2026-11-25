"""Test learned local contact-direction corrections at an earlier prepared receiving state."""

import numpy as np
from scipy.stats import qmc

from .tt37_motor_clinic import landing_error
from .tt38_motorcoach import AIM, SIZES, answerable, schedules
from .tt40_long_low import plain


def refine(a,engine,prepared,bases,batch,rounds,seed):
    group=[b for b in bases if b["face"]==prepared["face"] and .025<b["eta"]<.5
           and np.linalg.norm(b["q"]-prepared["q"])<.9]
    group.sort(key=lambda b: np.linalg.norm(b["q"]-a.joints[:5])+.3*abs(b["p"][1]-prepared["p"][1]))
    group=group[:12] or [prepared]
    pose=np.asarray([b["q"] for b in group])
    rate=np.asarray([b["dq"] for b in group])
    eta=np.asarray([b["eta"] for b in group])
    until=float(eta.max()+.15)
    dt=a.control_dt
    frames=round((until+.96)/dt)
    rng=np.random.default_rng(seed)
    z=(qmc.Sobol(len(SIZES),scramble=True,seed=int(rng.integers(2**31))).random_base2(
       int(np.ceil(np.log2(batch))))[:batch]*2-1)*.84
    key=np.resize(np.arange(len(eta)),batch)
    z[:len(eta)]=0.
    memory=[]; reports=[]; wins=[]
    for itr in range(rounds):
        ref=schedules(pose[key],rate[key],eta[key],z*SIZES,dt,frames)
        f=np.zeros((len(z),9))
        u,r=engine.trajectories(ref,AIM,until,feedback=f)
        legal=np.isfinite(r[:,0]);touch=f[:,1]>.5
        if touch.any():
            v=f[:,2:5]
            # Correlate with the expected speed needed to lift a ball over
            # the net from this actual contact separation point.
            v1=np.maximum(.15,v[:,1]*.89)
            tnet=np.maximum(0.,-f[:,6]/v1)
            tland=(v[:,2]+np.sqrt(np.maximum(0.,v[:,2]**2+19.62*(f[:,7]-.78))))/9.81
            net=f[:,7]+v[:,2]*tnet-4.905*tnet**2
            yland=f[:,6]+tland*v1
            xland=f[:,5]+tland*v[:,0]
            shape=.7*landing_error(f)+6*np.maximum(.985-net,0.)+1.4*abs(yland-.65)
            shape+=2*np.maximum(abs(xland)-.40,0.)+2*np.maximum(1.4-v[:,1],0.)
        else:
            shape=np.zeros(len(f))
        merit=np.where(legal,answerable(r)-12.,np.where(touch,shape,19.+34.*f[:,0]))
        memory.extend(zip(merit,key.copy(),z.copy(),f.copy(),strict=True))
        for j in np.flatnonzero(legal):
            n=min(u.shape[1],max(1,int(np.ceil((r[j,1]+.11)/dt))))
            wins.append(dict(tape=u[j,:n].tolist(),cost=float(answerable(r[j])),outcome=r[j].tolist(),
                             feedback=f[j].tolist(),variant=plain(group[key[j]]),parameters=(z[j]*SIZES).tolist()))
        reports.append(dict(round=itr,tested=len(z),touched=int(touch.sum()),private_legal=int(legal.sum()),
                            closest=float(f[:,0].min()),best_contact=f[np.argmin(merit),2:8].tolist()
                                                                   if touch.any() else None))
        if wins:
            break
        top=sorted(memory,key=lambda p:p[0])[:14]
        predicted=[]
        for _,k,center,obs in top[:8]:
            if obs[1]<.5: continue
            population=[(m,v[2:5]) for _,ki,m,v in memory if ki==k and v[1]>.5]
            if len(population)<6: continue
            m,v=map(np.asarray,zip(*population))
            delta=m-center
            w=np.exp(-.31*np.sum(delta**2,axis=1))
            force=np.linalg.solve((delta.T*w)@delta+.36*np.eye(len(SIZES)),(delta.T*w)@(v-obs[2:5]))
            dtf=(.72-obs[6])/(3.4*.89)
            wanted=np.r_[-obs[5]/max(dtf,.3),3.4,(.78-obs[7])/dtf+4.905*dtf]
            dv=np.clip(wanted-obs[2:5],-1.7,1.7)
            corr=force@np.linalg.solve(force.T@force+.42*np.eye(3),dv)
            for scale in (.52,1.0,1.52):
                predicted.append((k,np.clip(center+scale*corr,-2.5,2.5)))
        idx=rng.integers(len(top),size=batch)
        key=np.asarray([top[j][1] for j in idx])
        z=np.asarray([top[j][2] for j in idx])+rng.normal(0,.49,(batch,len(SIZES)))
        z=np.clip(z,-2.65,2.65)
        for j,(k,v) in enumerate(predicted[:batch//2]):
            key[j],z[j]=k,v
        reports[-1]["effect_guided_next"]=min(len(predicted),batch//2)
    return sorted(wins,key=lambda p:p["cost"]),reports
