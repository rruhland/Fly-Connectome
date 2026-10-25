"""Forward-only local action alternatives on the learner's actual failed inputs."""
import numpy as np
from policy_search_math import function_rms


def state_indices(q,bins=4):
    q=np.asarray(q);assert q.ndim==2 and q.shape[1]==3 and len(q)>0
    ordered=np.sort(q,axis=1);margin=ordered[:,-1]-ordered[:,-2]
    return [int(part[np.argmin(margin[part])]) for part in np.array_split(np.arange(len(q)),min(bins,len(q)))]


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
