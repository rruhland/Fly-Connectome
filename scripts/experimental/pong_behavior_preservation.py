"""Analytic positive-step bound preserving the learner's own successful actions."""
import numpy as np


def half_margin_bound(theta,delta,cases):
    bound=float('inf');limiting=None;rows=0;max_q_error=0.
    for case in cases:
        X=case['X'];a=np.asarray(case['actions'],dtype=np.int64);q=np.asarray(case['q']);dq=X@delta.T
        assert np.array_equal(np.argmax(q,axis=1),a)
        max_q_error=max(max_q_error,float(np.max(np.abs(X@theta.T-q))));rows+=len(X)
        margin=q[np.arange(len(q)),a,None]-q
        slope=dq[np.arange(len(q)),a,None]-dq
        adverse=slope<0;limits=np.full_like(slope,np.inf)
        limits[adverse]=.5*np.maximum(margin[adverse],0)/(-slope[adverse])
        k=int(np.argmin(limits));i,b=divmod(k,3)
        if limits[i,b]<bound:
            bound=float(limits[i,b]);limiting=dict(seed=case['seed'],row=i,chosen=int(a[i]),alternative=b,margin=float(margin[i,b]),slope=float(slope[i,b]))
    assert max_q_error<1e-10,'preservation states must reproduce actual carry logits'
    return bound,dict(full_guard_rows=rows,full_guard_episodes=len(cases),maximum_q_reconstruction_error=max_q_error,
        half_margin_alpha_bound=None if not np.isfinite(bound) else bound,limiting_constraint=limiting)


def preserved_action_fraction(delta,alpha,cases):
    matches=rows=0
    for case in cases:
        q=case['q']+alpha*(case['X']@delta.T);matches+=int(np.count_nonzero(np.argmax(q,axis=1)==case['actions']));rows+=len(q)
    return matches/rows if rows else None
