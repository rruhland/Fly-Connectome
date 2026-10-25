"""Forward-only local action alternatives on the learner's actual failed inputs."""
import numpy as np
from policy_search_math import function_rms


def state_indices(q,bins=4,midpoints=False):
    q=np.asarray(q);assert q.ndim==2 and q.shape[1]==3 and len(q)>0
    ordered=np.sort(q,axis=1);margin=ordered[:,-1]-ordered[:,-2]
    result=[]
    for part in np.array_split(np.arange(len(q)),min(bins,len(q))):
        result.append(int(part[np.argmin(margin[part])]))
        if midpoints:result.append(int(part[len(part)//2]))
    return list(dict.fromkeys(result))


def local_mutation(theta,x,alternative,calibration,*,kappa,maximum_rms=.25):
    theta=np.asarray(theta);x=np.asarray(x);q=theta@x;a=int(np.argmax(q))
    assert alternative in (0,1,2) and alternative!=a and kappa>0 and maximum_rms>0
    norm=float(x@x);assert norm>0
    gap=float(q[a]-q[alternative]);delta=(gap+kappa)*(np.eye(3)[alternative]-1/3)[:,None]*x[None,:]/norm
    raw_norm=function_rms(delta,calibration);alpha=min(1.,maximum_rms/raw_norm) if raw_norm>0 else 1.
    applied=alpha*delta;after=q+applied@x
    return applied,dict(chosen=a,alternative=alternative,kappa=kappa,raw_function_rms=raw_norm,alpha=alpha,
        applied_function_rms=function_rms(applied,calibration),intended_flip_achieved=int(np.argmax(after))==alternative,
        alternative_margin=float(after[alternative]-np.max(np.delete(after,alternative))),cap_limited=alpha<1.)


def drop_guard_conflicts(cases,guards):
    """Exclude only byte-identical desired actions incompatible with hard guards."""
    expected={}
    for c in guards:
        for x,a in zip(c['X'],c['actions']):
            key=np.ascontiguousarray(x,dtype=np.float64).tobytes()
            if key in expected:assert expected[key]==int(a)
            expected[key]=int(a)
    kept=[];dropped=[]
    for c in cases:
        mask=[]
        for i,(x,a) in enumerate(zip(c['X'],c['actions'])):
            other=expected.get(np.ascontiguousarray(x,dtype=np.float64).tobytes());conflict=other is not None and other!=int(a)
            mask.append(not conflict)
            if conflict:dropped.append(dict(seed=c.get('seed'),phase=c['phase'],row=i,desired_action=int(a),guard_action=other))
        mask=np.asarray(mask)
        if mask.any():kept.append(dict(c,X=c['X'][mask],actions=c['actions'][mask]))
    return kept,dict(dropped_rows=len(dropped),dropped_episodes=len(cases)-len(kept),details=dropped)
