"""Analytic positive-step bound preserving the learner's own successful actions."""
import numpy as np


def required_guard_margin(old_margin,margin_cap=None):
    required=.5*np.maximum(np.asarray(old_margin),0)
    if margin_cap is not None:
        assert np.isfinite(margin_cap) and margin_cap>0
        required=np.minimum(required,margin_cap)
    return required


def half_margin_bound(theta,delta,cases,margin_cap=None):
    bound=float('inf');limiting=None;rows=0;max_q_error=0.
    for case in cases:
        X=case['X'];a=np.asarray(case['actions'],dtype=np.int64);q=np.asarray(case['q']);dq=X@delta.T
        assert np.array_equal(np.argmax(q,axis=1),a)
        max_q_error=max(max_q_error,float(np.max(np.abs(X@theta.T-q))));rows+=len(X)
        margin=q[np.arange(len(q)),a,None]-q
        slope=dq[np.arange(len(q)),a,None]-dq
        adverse=slope<0;limits=np.full_like(slope,np.inf)
        required=required_guard_margin(margin,margin_cap)
        limits[adverse]=(np.maximum(margin[adverse],0)-required[adverse])/(-slope[adverse])
        k=int(np.argmin(limits));i,b=divmod(k,3)
        if limits[i,b]<bound:
            bound=float(limits[i,b]);limiting=dict(seed=case['seed'],row=i,chosen=int(a[i]),alternative=b,margin=float(margin[i,b]),required_margin=float(required[i,b]),slope=float(slope[i,b]))
    assert max_q_error<1e-10,'preservation states must reproduce actual carry logits'
    return bound,dict(full_guard_rows=rows,full_guard_episodes=len(cases),maximum_q_reconstruction_error=max_q_error,
        half_margin_alpha_bound=None if not np.isfinite(bound) else bound,limiting_constraint=limiting,
        margin_policy='half old confidence' if margin_cap is None else 'min(half old gap, behavior-only cap)',required_margin_cap=margin_cap)


def guard_margin_distribution(delta,alpha,cases,margin_cap=None):
    old=[];new=[]
    for c in cases:
        q=c['q'];a=c['actions'];dq=c['X']@delta.T
        for b in range(3):
            ix=np.flatnonzero(a!=b);old.extend((q[ix,a[ix]]-q[ix,b]).tolist());new.extend((q[ix,a[ix]]-q[ix,b]+alpha*(dq[ix,a[ix]]-dq[ix,b])).tolist())
    old=np.array(old);new=np.array(new);required=required_guard_margin(old,margin_cap);qs=[0,.1,.5,.9,1]
    return dict(quantiles=qs,original_gap=np.quantile(old,qs).tolist(),required_gap=np.quantile(required,qs).tolist(),applied_gap=np.quantile(new,qs).tolist(),
        maximum_required_gap_violation=float(np.max(np.maximum(required-new,0))),confidence_constraints_capped=int(np.sum(.5*old>margin_cap)) if margin_cap is not None else 0)


def preserved_action_fraction(delta,alpha,cases):
    matches=rows=0
    for case in cases:
        q=case['q']+alpha*(case['X']@delta.T);matches+=int(np.count_nonzero(np.argmax(q,axis=1)==case['actions']));rows+=len(q)
    return matches/rows if rows else None
