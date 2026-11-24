"""Fit and query an action-value forest on measured motor alternatives without backpropagation."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt37_motor_clinic import SCALE, geometry, landing_error
from .tt37_response import ACTION_SCALE


def features(surface, state, actions):
    """Express a proposed motor change also as blade position, angle and movement."""
    nominal,swing=geometry(surface,np.r_[0,state[:5]])
    velocity=.86*swing+actions[:,5:]
    q=nominal+actions[:,:5]-.009*velocity
    # Rate-limited approximate pose at contact. The regressor learns errors from physical trials.
    q=state[5:]+np.clip(q-state[5:],-.85,.85)
    q=np.clip(q,surface.ik.low,surface.ik.high)
    context=np.tile(state,(len(actions),1))
    pose=np.asarray([np.r_[*surface.ik.pose(a)] for a in q])
    nextpose=np.asarray([np.r_[*surface.ik.pose(a+.012*b)] for a,b in zip(q,velocity,strict=True)])
    gap=pose[:,:3]-[state[0],surface.depth,state[1]]
    normal=pose[:,3:]
    speed=(nextpose[:,:3]-pose[:,:3])/.012
    vel=np.asarray(state[2:5])
    normal_speed=np.sum((speed-vel)*normal,axis=1)
    reflected=vel+1.75*normal_speed[:,None]*normal
    return np.c_[context,actions,gap,normal,speed,reflected,q-state[5:]].astype(np.float32)


def calibration(paths, out, depth):
    from .tt29_rally_physics import BladeIK
    from .tt37_posture import Surface,sample
    from .tt37_opponent import arena
    a=arena();ik=BladeIK(a);surface=Surface(ik,sample(ik),depth,.35);a.close()
    rows=np.concatenate([np.load(p)['rows'] for p in paths])
    x,ind=np.unique(rows[:,:10],axis=0,return_inverse=True)
    xx=[];ys=[];groups=[];good=[];contexts=[]
    for i,state in enumerate(x):
        r=rows[ind==i]; y=r[:,29]; hit=r[:,21]>0
        xx.append(features(surface,state,r[:,10:20]))
        error=landing_error(r[:,20:29])
        good_hits=y>=0
        value=np.where(hit,.12+.43*np.exp(-2.*np.minimum(error,7.)),.10*np.exp(-7.*r[:,20]))
        value[good_hits]=.85+.15*np.exp(-.2*y[good_hits])
        ys.append(np.c_[good_hits,hit,value])
        groups.extend([i]*len(r))
        good.extend(r[good_hits,10:20]);contexts.extend([state]*sum(good_hits))
    np.savez_compressed(out, features=np.vstack(xx),labels=np.vstack(ys),group=groups,
                        actions=np.asarray(good),contexts=np.asarray(contexts))


def train(source,output):
    from sklearn.ensemble import ExtraTreesRegressor
    d=np.load(source);x,y,groups=d['features'],d['labels'],d['group']
    def fit(mask):
        return ExtraTreesRegressor(n_estimators=56,min_samples_leaf=5,max_features=.85,
                                   max_depth=18,n_jobs=1,random_state=37867).fit(x[mask],y[mask])
    val=groups%7==3
    checked=fit(~val).predict(x[val])
    model=fit(np.ones(len(x),dtype=bool))
    offsets=[];left=[];right=[];splits=[];threshold=[];values=[];size=0
    for e in model.estimators_:
        t=e.tree_;offsets.append(size)
        left.extend(np.where(t.children_left<0,-1,t.children_left+size));right.extend(np.where(t.children_right<0,-1,t.children_right+size))
        splits.extend(t.feature);threshold.extend(t.threshold);values.extend(t.value[:,:,0]);size+=t.node_count
    np.savez_compressed(output,roots=offsets,left=left,right=right,split=splits,threshold=threshold,values=values,
                        contexts=d['contexts'],actions=d['actions'],verification=np.mean((y[val]-checked)**2,axis=0),
                        constant=np.mean((y[val]-y[~val].mean(axis=0))**2,axis=0))
    print('held Brier / MSE',np.mean((y[val]-checked)**2,axis=0),'baseline',np.mean((y[val]-y[~val].mean(axis=0))**2,axis=0),flush=True)


class Forest:
    def __init__(self,checkpoint,surface,*,adaptive=False):
        with np.load(checkpoint) as z:
            for name in ('roots','left','right','split','threshold','values','contexts','actions'):
                setattr(self,name,z[name])
        self.surface=surface
        self.tree=cKDTree(self.contexts/SCALE)
        self.adaptive=adaptive
        self.experience=[];self.last=None

    def observe(self,won):
        if self.adaptive and self.last is not None:
            self.experience.append((*self.last,float(won)))
        self.last=None

    def predict(self,x):
        r=np.broadcast_to(self.roots,(len(x),len(self.roots))).copy()
        i=np.arange(len(x))[:,None]
        for _ in range(19):
            cond=self.left[r]>=0
            if not cond.any(): break
            r=np.where(cond,np.where(x[i,np.maximum(0,self.split[r])]<=self.threshold[r],self.left[r],self.right[r]),r)
        return self.values[r].mean(axis=1)

    def __call__(self,bank,state):
        _,ix=self.tree.query(state/SCALE,k=min(60,len(self.contexts)))
        local=self.actions[np.atleast_1d(ix)]
        rng=np.random.default_rng(37731)
        candidates=np.vstack([local,.6*local,local*.5+local[::-1]*.5,local+rng.normal(0,.13,local.shape)*ACTION_SCALE,np.zeros(10)])
        estimate=self.predict(features(self.surface,state,candidates))
        quality=.75*estimate[:,0]+.25*estimate[:,2]
        if self.experience:
            x,a,y=map(np.asarray,zip(*self.experience,strict=True))
            dd=np.linalg.norm((x-state)/SCALE,axis=1)
            da=np.linalg.norm((a[:,None]-candidates)/ACTION_SCALE,axis=2)
            w=np.exp(-.5*(dd[:,None]/.65)**2-.5*(da/.6)**2)
            quality=(quality+.7*w.T@y)/(1.+.7*w.sum(axis=0))
        act=candidates[np.argmax(quality)]
        self.last=state.copy(),act.copy()
        return act


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('features','train'));p.add_argument('out',type=Path)
    p.add_argument('sources',type=Path,nargs='+');p.add_argument('--depth',type=float,default=-1.56)
    a=p.parse_args()
    if a.mode=='features': calibration(a.sources,a.out,a.depth)
    else: train(a.sources[0],a.out)
