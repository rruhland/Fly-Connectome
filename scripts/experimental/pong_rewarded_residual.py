"""Closed-form preference residuals from the learner's own rewarded actions.

These are logit preference proposals, not Q targets, unbiased policy gradients,
backpropagation, or teacher/oracle actions. Actual closed-loop reward decides.
"""
import numpy as np
from policy_search_math import function_rms


def preference_targets(q,actions,kappa):
    q=np.asarray(q,dtype=float);a=np.asarray(actions,dtype=np.int64)
    assert q.ndim==2 and q.shape[1]==3 and a.shape==(len(q),) and np.isfinite(q).all()
    assert np.all((a>=0)&(a<3)) and kappa>0
    gap=q.max(1)-q[np.arange(len(q)),a]
    target=(gap+kappa)[:,None]*(np.eye(3)[a]-1/3)
    return target,np.argmax(q,axis=1)!=a


def episode_weights(cases):
    present=[p for p in ('left','right') if any(c['phase']==p and len(c['X']) for c in cases)]
    assert present
    counts={p:sum(c['phase']==p and len(c['X'])>0 for c in cases) for p in present}
    return np.concatenate([np.full(len(c['X']),1/(len(present)*counts[c['phase']]*len(c['X']))) for c in cases if len(c['X'])])


def fit_residual(theta,positive,preservation,buffer_X,buffer_weights,ridge=1e-3,kappa_fraction=.05):
    positive=[c for c in positive if len(c['X'])];preservation=[c for c in preservation if len(c['X'])]
    if not positive:return np.zeros_like(theta),dict(positive_rows=0,positive_episodes=0,reason='no contrasting rewarded actions')
    X=np.concatenate([c['X'] for c in positive]);a=np.concatenate([c['actions'] for c in positive]);w=episode_weights(positive)
    q=X@theta.T;q_buffer=buffer_X@theta.T
    q_rms=max(1e-12,float(np.sqrt(np.sum(buffer_weights[:,None]*q_buffer*q_buffer)/3)))
    scale=max(1e-12,float(np.sqrt(buffer_weights@np.sum(buffer_X*buffer_X,axis=1))))
    target,changed=preference_targets(q,a,kappa_fraction*q_rms);assert changed.all(),'filter matching incumbent actions before fitting'
    U=X/scale;G=U.T@(w[:,None]*U);B=target.T@(w[:,None]*U)
    if preservation:
        P=np.concatenate([c['X'] for c in preservation])/scale;v=episode_weights(preservation);G+=P.T@(v[:,None]*P)
    G+=ridge*np.eye(theta.shape[1]);delta=np.linalg.solve(G,B.T).T/scale
    delta-=delta.mean(axis=0,keepdims=True)
    assert np.isfinite(delta).all()
    predicted=X@delta.T
    info=dict(positive_rows=len(X),positive_episodes=len(positive),preservation_rows=sum(len(c['X']) for c in preservation),preservation_episodes=len(preservation),
        input_rms=scale,incumbent_q_rms=q_rms,kappa=kappa_fraction*q_rms,ridge=ridge,weighted_target_mse=float(np.sum(w[:,None]*(predicted-target)**2)/3),
        requested_actions_before_fraction=float(np.mean(np.argmax(q,axis=1)==a)),requested_actions_after_fit_fraction=float(np.mean(np.argmax(q+predicted,axis=1)==a)),
        requested_actions_after_fit_weighted_fraction=float(w@(np.argmax(q+predicted,axis=1)==a)),diagnostic_weighting='fraction fields are unweighted row means unless marked weighted',
        maximum_action_row_sum=float(np.max(np.abs(delta.sum(0)))))
    return delta,info


def positive_disagreement(theta,delta,X,weights,alpha):
    q=X@theta.T
    return float(weights@(np.argmax(q+alpha*(X@delta.T),axis=1)!=np.argmax(q,axis=1)))


def cap_residual(theta,delta,buffer_X,buffer_weights,calibration,target=.10,maximum_rms=.10):
    norm=function_rms(delta,calibration)
    if norm<=0:return np.zeros_like(delta),dict(alpha=0.,raw_function_rms=norm,applied_function_rms=0.,achieved_disagreement=0.,target=target,cap_limited=True)
    maximum=maximum_rms/norm;at_cap=positive_disagreement(theta,delta,buffer_X,buffer_weights,maximum)
    low,high=0.,maximum
    if at_cap>target:
        for _ in range(24):
            mid=.5*(low+high)
            if positive_disagreement(theta,delta,buffer_X,buffer_weights,mid)<target:low=mid
            else:high=mid
    applied=high*delta
    return applied,dict(alpha=high,raw_function_rms=norm,applied_function_rms=function_rms(applied,calibration),target=target,
        achieved_disagreement=positive_disagreement(theta,delta,buffer_X,buffer_weights,high),cap_disagreement=at_cap,cap_limited=at_cap<=target)


def action_diagnostics(theta,delta,positive,preservation):
    result={}
    if positive:
        X=np.concatenate([c['X'] for c in positive]);a=np.concatenate([c['actions'] for c in positive]);q=X@theta.T
        matched=np.argmax(q+X@delta.T,axis=1)==a
        result['requested_actions_after_cap_fraction']=float(np.mean(matched))
        result['requested_actions_after_cap_weighted_fraction']=float(episode_weights(positive)@matched)
    if preservation:
        X=np.concatenate([c['X'] for c in preservation]);q=X@theta.T
        changed=np.argmax(q+X@delta.T,axis=1)!=np.argmax(q,axis=1)
        result['preservation_action_change_fraction']=float(np.mean(changed))
        result['preservation_action_change_weighted_fraction']=float(episode_weights(preservation)@changed)
    return result
