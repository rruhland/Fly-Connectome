"""Convex soft action-margin fit, using analytic active-set Newton steps.

No autodiff/backprop or hard action constraints. Reward-earned witness labels are
sequence self-imitation targets. All preservation is soft and confidence-capped.
"""
import time
import numpy as np
from pong_constrained_residual import CONTRAST
from pong_rally_imitation import case_weights,diagnostics


def problem(theta,positive,preservation,matching,normalization):
    scale=normalization['input_rms'];kappa=normalization['kappa'];blocks=[]
    for cases,coefficient,kind in ((positive,1.,'positive'),(preservation,1.,'preservation'),(matching,.25,'matching')):
        if not cases:continue
        X=np.concatenate([c['X'] for c in cases]);a=np.concatenate([c['actions'] for c in cases]);w=case_weights(cases)*coefficient
        other=np.array([[b for b in range(3) if b!=action] for action in a]);q=X@theta.T;rows=np.arange(len(a))
        old=q[rows,a,None]-q[rows[:,None],other]
        required=np.full_like(old,kappa) if kind=='positive' else np.minimum(kappa,np.maximum(old,0.))
        if kind!='positive':assert old.min()>-1e-10
        blocks.append((X/scale,a,other,old,required,np.repeat((w/2)[:,None],2,axis=1)))
    return tuple(np.concatenate([v[i] for v in blocks]) for i in range(6))


def objective(parameters,data,ridge=.001,need_gradient=True):
    U,a,other,old,required,w=data;d=U.shape[1];weights=CONTRAST@parameters.reshape(2,d);q=U@weights.T;rows=np.arange(len(a))
    margins=old+q[rows,a,None]-q[rows[:,None],other];hinge=np.maximum(required-margins,0.)
    value=float(np.sum(w*hinge*hinge)+ridge*(parameters@parameters))
    if not need_gradient:return value
    loss_gradient=2*w*hinge;gq=np.zeros((len(U),3));gq[rows,a]=-loss_gradient.sum(1)
    for j in range(2):np.add.at(gq,(rows,other[:,j]),loss_gradient[:,j])
    gradient=(CONTRAST.T@(gq.T@U)).reshape(-1)+2*ridge*parameters
    return value,gradient,hinge


def active_hessian(data,hinge,ridge=.001):
    U,a,other,old,required,w=data;d=U.shape[1];H=np.eye(2*d)*(2*ridge)
    low=np.minimum(a[:,None],other);high=np.maximum(a[:,None],other)
    for left,right in ((0,1),(0,2),(1,2)):
        active=(low==left)&(high==right)&(hinge>0)
        row_weight=(2*w*active).sum(1);selected=row_weight>0
        if not selected.any():continue
        x=U[selected];G=x.T@(row_weight[selected,None]*x);c=CONTRAST[left]-CONTRAST[right]
        H[:d,:d]+=c[0]*c[0]*G;H[:d,d:]+=c[0]*c[1]*G
        H[d:,:d]+=c[0]*c[1]*G;H[d:,d:]+=c[1]*c[1]*G
    return H


def fit(theta,positive,preservation,matching,normalization):
    began=time.monotonic();data=problem(theta,positive,preservation,matching,normalization);p=np.zeros(2*theta.shape[1]);value,gradient,hinge=objective(p,data);zero=value;g0=float(np.linalg.norm(gradient));history=[];status='iteration_limit'
    for iteration in range(40):
        if time.monotonic()-began>30:status='time_limit_feasible_iterate';break
        if np.linalg.norm(gradient)<=1e-6*max(1.,g0):status='converged';break
        H=active_hessian(data,hinge);direction=-np.linalg.solve(H,gradient);slope=float(gradient@direction)
        assert np.isfinite(direction).all() and slope<0
        step=1.;accepted=False
        for _ in range(24):
            trial=p+step*direction;trial_value=objective(trial,data,need_gradient=False)
            if np.isfinite(trial_value) and trial_value<=value+1e-4*step*slope:
                p=trial;accepted=True;break
            step*=.5
        if not accepted:status='line_search_stopped_finite_iterate';break
        value,gradient,hinge=objective(p,data);history.append(dict(iteration=iteration+1,objective=value,step=step,active_pairs=int(np.count_nonzero(hinge)),gradient_norm=float(np.linalg.norm(gradient))))
    assert np.isfinite(p).all() and value<=zero+1e-10
    delta=CONTRAST@p.reshape(2,theta.shape[1])/normalization['input_rms'];delta-=delta.mean(0,keepdims=True)
    report=dict(algorithm='soft squared-hinge action margins on own reward-success sequences',solver='analytic active-set Newton with Armijo line search',status=status,
        positive_weight=1.,broad_success_weight=1.,matching_weight=.25,ridge=.001,frozen_input_normalization=normalization['input_rms'],kappa=normalization['kappa'],
        preservation_margin='min(kappa, carried desired-action gap), clipped at0 for numerical near ties',pair_weighting='each row weight divided equally between its two alternative-action comparisons',
        weighting='equal present group, episode, segment, row',radial_shrink=None,hard_action_constraints=False,
        positive=diagnostics(theta,delta,positive),preservation=diagnostics(theta,delta,preservation),matching=diagnostics(theta,delta,matching),
        zero_residual_objective=zero,final_objective=value,gradient_norm=float(np.linalg.norm(gradient)),active_pairs=int(np.count_nonzero(hinge)),history=history,fit_seconds=time.monotonic()-began)
    report['preservation_by_group']={g:diagnostics(theta,delta,[c for c in preservation if c['group']==g]) for g in sorted({c['group'] for c in preservation})}
    return delta,report
