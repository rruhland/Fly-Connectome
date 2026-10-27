"""Sparse raw-coordinate squared-hinge residual, analytic matrix-free Newton-CG.

No autodiff, neural backprop, hidden state, or dense 8778-square Hessian. The same
three-action linear runtime is retained; raw-coordinate ridge changes geometry.
"""
import time
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, cg
from pong_constrained_residual import CONTRAST


def weights(cases):
    groups=sorted({c['group'] for c in cases if c['X'].shape[0]}); output=[]
    for c in cases:
        n=c['X'].shape[0]
        if not n:continue
        episodes={v['seed'] for v in cases if v['group']==c['group'] and v['X'].shape[0]}
        segments=sum(v['group']==c['group'] and v['seed']==c['seed'] and v['X'].shape[0]>0 for v in cases)
        output.append(np.full(n,1/(len(groups)*len(episodes)*segments*n)))
    result=np.concatenate(output);assert abs(result.sum()-1)<1e-12;return result


def problem(W, positive, preservation, matching, normal):
    blocks=[]
    for cases, coefficient, kind in ((positive,1.,'positive'),(preservation,1.,'preservation'),(matching,.25,'matching')):
        if not cases:continue
        X=sp.vstack([c['X'] for c in cases],format='csr'); a=np.concatenate([c['actions'] for c in cases]); w=weights(cases)*coefficient
        other=np.array([[j for j in range(3) if j!=v] for v in a]); q=X@W.T; row=np.arange(len(a)); old=q[row,a,None]-q[row[:,None],other]
        required=np.full_like(old,normal['kappa']) if kind=='positive' else np.minimum(normal['kappa'],np.maximum(old,0.))
        if kind!='positive':assert old.min()>-1e-10
        blocks.append((X/normal['input_rms'],CONTRAST[a,None,:]-CONTRAST[other],old,required,np.repeat((w/2)[:,None],2,axis=1)))
    return (sp.vstack([v[0] for v in blocks],format='csr'),*(np.concatenate([v[i] for v in blocks]) for i in range(1,5)))


def objective(p,data,ridge=1e-6,gradient=True):
    U,c,old,required,w=data; z=U@p.reshape(2,U.shape[1]).T
    margin=old+np.einsum('ni,nji->nj',z,c); hinge=np.maximum(required-margin,0.); value=float(np.sum(w*hinge*hinge)+ridge*(p@p))
    if not gradient:return value
    g=(-2*(U.T@np.sum((w*hinge)[:,:,None]*c,axis=1))).T.reshape(-1)+2*ridge*p
    return value,g,hinge


def hessian_operator(data,hinge,ridge):
    U,c,old,required,w=data;d=U.shape[1]
    selected=np.any(hinge>0,axis=1)
    U=U[selected];c=c[selected];w=w[selected];hinge=hinge[selected]
    active=2*w*(hinge>0)
    def product(v):
        z=U@v.reshape(2,d).T; dm=np.einsum('ni,nji->nj',z,c)
        result=(U.T@np.sum((active*dm)[:,:,None]*c,axis=1)).T.reshape(-1)
        return result+2*ridge*v
    diagonal=(U.multiply(U).T@np.sum(active[:,:,None]*c*c,axis=1)).T.reshape(-1)+2*ridge
    return LinearOperator((2*d,2*d),matvec=product,dtype=np.float64), LinearOperator((2*d,2*d),matvec=lambda v:v/diagonal,dtype=np.float64)


def diagnostics(W,delta,cases):
    if not cases:return None
    X=sp.vstack([c['X'] for c in cases],format='csr'); w=weights(cases); a=np.concatenate([c['actions'] for c in cases]); old=X@W.T; change=X@delta.T; after=old+change
    match=np.argmax(after,axis=1)==a
    return dict(rows=X.shape[0],segments=len(cases),weighted_requested_action_fraction=float(w@match),requested_action_fraction=float(match.mean()),
        weighted_action_change=float(w@(np.argmax(after,1)!=np.argmax(old,1))),contrast_output_rms=float(np.sqrt(np.sum(w[:,None]*change*change)/3)),
        raw_input_rms=float(np.sqrt(w@np.asarray(X.multiply(X).sum(axis=1)).ravel())))


def fit(W,positive,preservation,matching,normal,deadline,ridge=1e-6,initial_delta=None):
    began=time.monotonic(); data=problem(W,positive,preservation,matching,normal); d=W.shape[1];p=np.zeros(2*d)
    zero,gzero,_=objective(p,data,ridge)
    if initial_delta is not None:
        initial_delta=np.asarray(initial_delta);assert initial_delta.shape==W.shape and np.isfinite(initial_delta).all()
        centered=initial_delta-initial_delta.mean(0,keepdims=True)
        assert np.max(abs(centered-initial_delta))<1e-10
        p=(CONTRAST.T@centered*normal['input_rms']).reshape(-1)
    value,g,hinge=objective(p,data,ridge);assert value<=zero+1e-10
    initial_value=value;g0=float(np.linalg.norm(gzero));history=[];status='iteration_limit'
    for i in range(35):
        if time.monotonic()>=deadline:status='time_limit_finite_downhill_iterate';break
        if np.linalg.norm(g)<=1e-6*max(1.,g0):status='converged';break
        H,M=hessian_operator(data,hinge,ridge); iterations=[0]
        def callback(v):
            iterations[0]+=1
            if time.monotonic()>=deadline:raise TimeoutError('fit clock')
        try:direction,info=cg(H,-g,M=M,rtol=.01,atol=1e-11,maxiter=200,callback=callback)
        except TimeoutError:status='time_limit_finite_downhill_iterate';break
        slope=float(g@direction)
        if not np.isfinite(direction).all() or slope>=0:direction=-M@g;slope=float(g@direction)
        step=1.;accepted=False
        for _ in range(24):
            trial=p+step*direction;trial_value=objective(trial,data,ridge,gradient=False)
            if np.isfinite(trial_value) and trial_value<=value+1e-4*step*slope:p=trial;accepted=True;break
            step*=.5
        if not accepted:status='line_search_finite_downhill_iterate';break
        value,g,hinge=objective(p,data,ridge);history.append(dict(iteration=i+1,objective=value,step=step,cg_iterations=iterations[0],cg_status=int(info),gradient_norm=float(np.linalg.norm(g))))
    assert np.isfinite(p).all() and value<=zero+1e-10
    delta=CONTRAST@p.reshape(2,d)/normal['input_rms'];delta-=delta.mean(0,keepdims=True)
    return delta,dict(solver='analytic sparse Newton-CG, diagonal preconditioner, Armijo; no autodiff',status=status,raw_dimension=d,contrast_parameters=2*d,
        ridge=ridge,warm_start=initial_delta is not None,initial_objective=initial_value,active_row_hessian_exact=True,frozen_input_normalization=normal['input_rms'],kappa=normal['kappa'],geometry='L2 on normalized raw coefficient residual; differs from restricted DCT parameter metric',
        positive_weight=1.,preservation_weight=1.,matching_weight=.25,positive=diagnostics(W,delta,positive),preservation=diagnostics(W,delta,preservation),matching=diagnostics(W,delta,matching),
        zero_residual_objective=zero,final_objective=value,gradient_norm=float(np.linalg.norm(g)),history=history,fit_seconds=time.monotonic()-began,hard_constraints=False,radial_shrink=None)
