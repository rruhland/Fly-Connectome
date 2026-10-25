"""Forward-only mutation dose on legal encountered policy inputs."""
import numpy as np


def harmonic(left,right):
    return 0. if left+right==0 else 2.*left*right/(left+right)


def disagreement(theta,direction,X,scale,weights):
    q=X@theta.T;delta=X@direction.T;a=np.argmax(q,axis=1)
    changed=.5*((np.argmax(q+scale*delta,axis=1)!=a).astype(float)+(np.argmax(q-scale*delta,axis=1)!=a).astype(float))
    return float(weights@changed)


def mutation_scale(theta,direction,X,weights,target,maximum=.25):
    assert 0<target<1 and maximum>0 and np.isfinite(X).all() and len(X)==len(weights)
    assert np.isclose(np.sum(weights),1.) and np.all(weights>=0)
    cap_fraction=disagreement(theta,direction,X,maximum,weights)
    if cap_fraction<=target:return maximum,dict(target=target,cap_fraction=cap_fraction,achieved=cap_fraction,cap_limited=True)
    low,high=0.,maximum
    for _ in range(24):
        mid=.5*(low+high)
        if disagreement(theta,direction,X,mid,weights)<target:low=mid
        else:high=mid
    return high,dict(target=target,cap_fraction=cap_fraction,achieved=disagreement(theta,direction,X,high,weights),cap_limited=False)


def balanced_buffer(cases):
    counts={p:sum(c['phase']==p for c in cases) for p in ('left','right')};xs=[];ws=[]
    assert all(counts.values())
    for c in cases:
        x=np.asarray(c['X']);assert x.ndim==2 and len(x)>0
        xs.append(x);ws.append(np.full(len(x),.5/(counts[c['phase']]*len(x))))
    return np.concatenate(xs),np.concatenate(ws)
