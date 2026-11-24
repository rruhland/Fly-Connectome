"""A physical signed-face chart for shallow, rising returns on the grounded SO-101."""

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .tt37_bodycoach_windows import Reach
from .tt38_impact_margin import target_launch
from .tt38_motorcoach import AIM,SIZES,answerable,schedules
from .tt40_low_feasibility import Faces


DEPTHS=(-1.08,-1.18,-1.28,-1.38)
FEATURE=np.array([.16,.16,.082,1.15,1.0,1.05])
SCALE=np.r_[.14,.13,.105,.45,.45,.45]


def unit(v):
    return v/np.linalg.norm(v)


def impulse(p,v,pace,pitch):
    want=target_launch(p,pace)
    n=unit(unit(want-v)+np.array([0.,0.,pitch]))
    speed=np.clip((want+.76*v)@n/1.76,-.14,1.65)
    return n, speed*n+np.array([-.18*v[0],.06,.39])


class Chart:
    def __init__(self,a,home,seed,*,atlas=None):
        self.ik=Faces(a)
        self.guard=Reach(a,home,home)
        self.home=np.asarray(home)
        if atlas is None:
            q=qmc.scale(qmc.Sobol(5,scramble=True,seed=410272).random_base2(13),self.ik.low,self.ik.high)
            self.raw=np.r_[np.asarray(seed)[None,:],self.home[None,:5],q]
            self.poses=np.array([np.r_[*self.ik.pose(z)] for z in self.raw])
        else:
            self.raw=np.asarray(atlas['seeds'],dtype=float)
            self.poses=np.asarray(atlas['poses'],dtype=float)
        self.tree=cKDTree(self.poses/SCALE)

    def poses_at(self,p,v,pace,pitch,q0,*,quick=False):
        n,u=impulse(p,v,pace,pitch)
        desired=p-.024*n
        rows=[]
        for sign in (1,-1):
            ids=np.atleast_1d(self.tree.query(np.r_[desired,sign*n]/SCALE,k=10)[1])
            seeds=[q0,self.raw[ids[0]]]
            if not quick:
                extra=next((idx for idx in ids[1:] if np.max(abs(self.raw[idx]-seeds[-1]))>.45),None)
                if extra is not None:seeds.append(self.raw[extra])
            for s in seeds:
                q,(c,raw,J,G)=self.ik.fit(desired,sign*n,s)
                normal=sign*raw
                d=p-c; dist=float(d@normal); tangent=float(np.linalg.norm(d-dist*normal))
                if not (-.0005<dist<.062 and tangent<.073 and float(n@normal)>.90 and self.guard.clear(q)):
                    continue
                dq=np.linalg.lstsq(np.r_[J,.067*G,.022*np.eye(5)],np.r_[u,np.zeros(8)],rcond=None)[0].clip(-6.2,6.2)
                travel=float(np.max(abs(q-q0)))
                rank=3.8*abs(dist-.024)+3.0*tangent+.11*travel+.16*(1.-n@normal)
                if any(a['sign']==sign and np.max(abs(a['q']-q))<.16 for a in rows):continue
                rows.append(dict(q=q,dq=dq,normal=normal,sign=sign,tangent=tangent,distance=dist,travel=travel,
                                 rank=float(rank),pace=float(pace),pitch=float(pitch)))
        return rows

    def assess(self):
        from collections import Counter
        home=self.home[:5]; raw=self.raw[0]
        cells=[]; hubs=[]
        for x in (-.28,-.15,0.,.15,.28):
            for y in (-1.10,-1.25,-1.40):
                for z in (.90,.96,1.055):
                    p=np.array([x,y,z]); v=np.array([.30*x,-4.0,.65]); poses=[]
                    for tilt in (.03,.19):
                        poses.extend(self.poses_at(p,v,4.3,tilt,raw))
                    best={}
                    for sign in (-1,1):
                        eligible=[a for a in poses if a['sign']==sign]
                        if eligible:
                            a=min(eligible,key=lambda a:a['rank'])
                            best[str(sign)]={key:(a[key].tolist() if isinstance(a[key],np.ndarray) else a[key])
                                             for key in ('q','rank','tangent','distance','normal')}
                            if abs(x)<=.15 and z>=.90 and y<-1.09:
                                hubs.append(a['q'])
                    cells.append(dict(position=p.tolist(),solutions=best))
        rank=[]
        feasible=[p for p in cells if p['solutions']]
        for q in (hubs+[raw,.55*raw+.45*home]):
            if not self.guard.route(home,q,steps=14):continue
            here,_=self.ik.pose(q)
            if not (abs(here[0])<.24 and .86<here[2]<1.13):continue
            ds=[]
            for p in feasible:
                ds.append(min(float(np.max(abs(q-a['q']))) for a in p['solutions'].values()))
            merit=float(np.mean(np.minimum(ds,2.5))+.28*np.max(abs(q-home))+.08*abs(here[0]))
            rank.append(dict(q=q.tolist(),position=here.tolist(),score=merit,max_home_travel=float(np.max(abs(q-home)))))
        rank.sort(key=lambda a:a['score'])
        if not rank:raise ValueError('No CAD-connected shallow neutral hub')
        signs=Counter(sig for r in cells for sig in r['solutions'])
        return dict(cells=cells,signs=dict(signs),hubs=rank[:6],home=self.home.tolist(),
                    seeds=self.raw.tolist(),poses=self.poses.tolist())

    def windows(self,path,q0,*,deployment=False):
        if not len(path):return []
        # These are postbounce model samples. Directly require rising velocity;
        # a late low drop on the same plane does not invoke the shallow skill.
        h=path[(path[:,0]>.036)&(path[:,0]<.50)&(path[:,1]>-.40)&(path[:,1]<.40)
               &(path[:,3]>.835)&(path[:,3]<1.095)&(path[:,6]>.22)]
        if not len(h):return []
        rows=[]
        for depth in DEPTHS:
            if not h[:,2].min()-.012<depth<h[:,2].max()+.012:continue
            sample=h[np.argmin(abs(h[:,2]-depth))]
            time,p,v=float(sample[0]),sample[1:4],sample[4:7]
            for pace,pitch in ((4.05,.10),(4.60,.02),(3.58,.18)):
                for row in self.poses_at(p,v,pace,pitch,q0,quick=deployment):
                    if row['travel']>4.6*time+.17 or not self.guard.route(q0,row['q'],steps=7):continue
                    rows.append(row|dict(eta=time,p=p.copy(),v=v.copy()))
        groups={}
        for s in sorted(rows,key=lambda a:a['rank']):
            key=round(float(s['p'][1]),1),s['sign']
            group=groups.setdefault(key,[])
            if len(group)<3 and not any(np.max(abs(s['q']-a['q']))<.14 for a in group):group.append(s)
        return [r for entries in groups.values() for r in entries][:16]


def goals(bases,keys,offsets,dt):
    q=np.array([bases[k]['q'] for k in keys]); dq=np.array([bases[k]['dq'] for k in keys]); eta=np.array([bases[k]['eta'] for k in keys])
    cutoff=float(eta.max()+.12); n=round((cutoff+1.07)/dt)
    return schedules(q,dq,eta,offsets,dt,n),cutoff


def contact_preference(result, feedback):
    legal=np.isfinite(result[:,0]); touch=feedback[:,1]>.5
    score=np.full(len(result),-4.)-12*np.maximum(0.,feedback[:,0]-.04)
    # Keep a signed launch signal even when the candidate loses the point.
    if touch.any():
        vx,vy,vz=feedback[touch,2:5].T
        score[touch]=-1.7-.34*abs(vy-3.9)-.30*abs(vz-2.94)-.13*abs(vx)-3.*feedback[touch,8]
    if legal.any():
        q=result[legal]
        score[legal]=3.05-.23*answerable(q)-7.*np.maximum(.052-q[:,5],0.)-4.*np.maximum(.25-q[:,4],0.)
    return score,legal,touch
