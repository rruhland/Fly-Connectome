"""Rewarded sequence preference ridge, with segment-balanced soft preservation.

This is self-imitation of native-return witnesses, not a policy-gradient estimator,
Q regression, oracle supervision or a claim each held action is causally necessary.
"""
import time
import numpy as np
from pong_rewarded_residual import preference_targets


def case_weights(cases):
    """Equal present groups, episodes within groups, segments within episode, rows."""
    groups=sorted({c['group'] for c in cases if len(c['X'])});assert groups
    result=[]
    for case in cases:
        n=len(case['X'])
        if not n:continue
        group=case['group'];episodes={c['seed'] for c in cases if c['group']==group and len(c['X'])}
        segments=sum(c['group']==group and c['seed']==case['seed'] and len(c['X'])>0 for c in cases)
        result.append(np.full(n,1/(len(groups)*len(episodes)*segments*n)))
    result=np.concatenate(result);assert abs(result.sum()-1)<1e-12
    return result


def diagnostics(theta,delta,cases):
    if not cases:return None
    X=np.concatenate([c['X'] for c in cases]);w=case_weights(cases);before=X@theta.T;change=X@delta.T;after=before+change
    result=dict(rows=len(X),segments=len(cases),groups=sorted({c['group'] for c in cases}),
        weighted_action_change=float(w@(np.argmax(before,1)!=np.argmax(after,1))),
        contrast_output_rms=float(np.sqrt(np.sum(w[:,None]*change*change)/3)))
    if all('actions' in c for c in cases):
        actions=np.concatenate([c['actions'] for c in cases]);match=np.argmax(after,1)==actions
        result.update(requested_action_fraction=float(np.mean(match)),weighted_requested_action_fraction=float(w@match))
    return result


def fit(theta,positive,preservation,matching,normalization):
    began=time.monotonic();scale=normalization['input_rms'];kappa=normalization['kappa'];ridge=.001
    assert scale>0 and kappa>0 and positive
    X=np.concatenate([c['X'] for c in positive]);a=np.concatenate([c['actions'] for c in positive]);w=case_weights(positive)
    q=X@theta.T;target,_=preference_targets(q,a,kappa);U=X/scale
    G=U.T@(w[:,None]*U);B=target.T@(w[:,None]*U)
    for cases,weight in ((preservation,1.),(matching,.25)):
        if cases:
            P=np.concatenate([c['X'] for c in cases])/scale;pw=case_weights(cases);G+=weight*(P.T@(pw[:,None]*P))
    G+=ridge*np.eye(theta.shape[1]);delta=np.linalg.solve(G,B.T).T/scale;delta-=delta.mean(0,keepdims=True)
    assert np.isfinite(delta).all() and np.max(abs(delta.sum(0)))<1e-10
    residual=X@delta.T-target
    report=dict(algorithm='own reward-success sequence preference ridge',positive_weight=1.,broad_success_weight=1.,matching_weight=.25,ridge=ridge,
        frozen_input_normalization=scale,kappa=kappa,radial_shrink=None,hard_action_constraints=False,
        weighting='equal present group, equal seed/episode within group, equal segment within episode, equal rows per segment',
        positive=diagnostics(theta,delta,positive),preservation=diagnostics(theta,delta,preservation),matching=diagnostics(theta,delta,matching),
        weighted_target_mse=float(np.sum(w[:,None]*residual*residual)/3),fit_seconds=time.monotonic()-began)
    report['preservation_by_group']={group:diagnostics(theta,delta,[c for c in preservation if c['group']==group]) for group in sorted({c['group'] for c in preservation})}
    return delta,report
