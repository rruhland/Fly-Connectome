"""Sparse diagonal slack QP with lazy current-success action guards.

Only requested action margins have hinge slack loss; current earned prefixes
retain choices with capped small positive margins. No confidence/logit locks,
failed suffix guards, dense raw Hessian, or autodiff.
"""
import time
import numpy as np
import scipy.sparse as sp
import osqp
from fly_connectome.control.policy import FrozenLinearPolicy
from pong_constrained_residual import CONTRAST
from pong_rally_raw_margin import weights,diagnostics


def pairs(X,actions,W):
    other=np.array([[j for j in range(3) if j!=a] for a in actions]);q=X@W.T;row=np.arange(len(actions));old=q[row,actions,None]-q[row[:,None],other]
    return other,old,CONTRAST[actions,None,:]-CONTRAST[other]


def contrast_rows(X,contrasts,flat_indices,scale):
    flat_indices=np.asarray(flat_indices,dtype=int);row=flat_indices//2;which=flat_indices%2;values=X[row]/scale;c=contrasts[row,which]
    return sp.hstack([values.multiply(c[:,0,None]),values.multiply(c[:,1,None])],format='csc')


def fit(W,positive,guards,normal,deadline,ridge=1e-6,batch=128,initial_rows=(),initial_delta=None,initial_state=None,state_path=None,initial_guard_duals=None):
    began=time.monotonic();d=W.shape[1];scale=normal['input_rms'];kappa=normal['kappa'];n=2*d
    X=sp.vstack([c['X'] for c in positive],format='csr');actions=np.concatenate([c['actions'] for c in positive]);other,old,c=pairs(X,actions,W);target=(kappa-old).reshape(-1);pair_weights=np.repeat(weights(positive)/2,2);ns=len(target)
    Xg=sp.vstack([c['X'] for c in guards],format='csr');ag=np.concatenate([c['actions'] for c in guards]);og,gap,cg=pairs(Xg,ag,W);assert gap.min()>-1e-10
    required=np.minimum(kappa,.5*np.maximum(gap,0.));lower=(required-gap).reshape(-1);assert lower.max()<=0
    Z=contrast_rows(X,c,np.arange(ns),scale);Apositive=sp.hstack([Z,sp.eye(ns,format='csc')],format='csc');Anonnegative=sp.hstack([sp.csc_matrix((ns,n)),sp.eye(ns,format='csc')],format='csc')
    diagonal=np.r_[np.full(n,2*ridge),2*pair_weights];P=sp.diags(diagonal,format='csc');q=np.zeros(n+ns);z=np.r_[np.zeros(n),np.maximum(target,0.)];zero=float(np.sum(pair_weights*np.maximum(target,0.)**2));working=sorted(set(map(int,initial_rows)));history=[];status='working_set_limit'
    def evaluate(v):
        delta=CONTRAST@v[:n].reshape(2,d)/scale;change=Xg@delta.T;gdelta=change[np.arange(len(ag)),ag,None]-change[np.arange(len(ag))[:,None],og]
        return delta,gdelta,lower-gdelta.reshape(-1)
    base_duals=np.zeros(2*ns);guard_duals={int(i):float(v) for i,v in (initial_guard_duals or {}).items()}
    if initial_delta is not None:
        assert initial_state is None and initial_delta.shape==W.shape and np.isfinite(initial_delta).all()
        z[:n]=(CONTRAST.T@initial_delta*scale).reshape(-1);z[n:]=np.maximum(target-Z@z[:n],0.)
        _,_,initial_violation=evaluate(z)
        working=sorted(set(working)|set(map(int,np.flatnonzero(initial_violation>=-1e-4))))
    if initial_state is not None:
        z=np.asarray(initial_state['primal']).copy();assert z.shape==(n+ns,) and np.isfinite(z).all()
        working=sorted(set(map(int,initial_state['working_rows'])));base_duals=np.asarray(initial_state['base_duals']).copy()
        guard_duals={int(i):float(v) for i,v in zip(initial_state['working_rows'],initial_state['guard_duals'])}
    for iteration in range(30):
        remaining=deadline-time.monotonic()
        if remaining<=0:status='wall_limit_finite_iterate';break
        G=contrast_rows(Xg,cg,working,scale);Aguard=sp.hstack([G,sp.csc_matrix((len(working),ns))],format='csc')
        A=sp.vstack([Apositive,Anonnegative,Aguard],format='csc');lo=np.r_[target,np.zeros(ns),lower[working]]
        solver=osqp.OSQP();solver.setup(P=P,q=q,A=A,l=lo,u=np.full(len(lo),np.inf),verbose=False,polishing=True,eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,time_limit=max(.01,min(remaining,12.)))
        solver.warm_start(x=z,y=np.r_[base_duals,[guard_duals.get(i,0.) for i in working]]);answer=solver.solve(raise_error=False)
        if answer.x is None or not np.isfinite(answer.x).all():status='nonfinite_solver_preserve_previous';break
        z=np.asarray(answer.x)
        if answer.y is not None and np.isfinite(answer.y).all():
            base_duals=np.asarray(answer.y[:2*ns]).copy();guard_duals={i:float(v) for i,v in zip(working,answer.y[2*ns:])}
        delta,gdelta,violation=evaluate(z);maximum=max(0.,float(violation.max()))
        history.append(dict(round=iteration+1,working_guards=len(working),maximum_full_violation=maximum,solver_status=answer.info.status,solver_iterations=int(answer.info.iter),primal_residual=float(answer.info.prim_res),dual_residual=float(answer.info.dual_res)))
        if maximum<=1e-8:status='full_guard_feasible';break
        eligible=np.flatnonzero(violation>1e-8);existing=set(working);fresh=np.array([i for i in eligible if int(i) not in existing],dtype=int)
        if not len(fresh):status='numerical_guard_residual';break
        selected=fresh[np.argsort(-violation[fresh],kind='stable')[:batch]];working=sorted(existing|set(map(int,selected)))
    if state_path is not None:
        np.savez(state_path,primal=z,working_rows=np.asarray(working,dtype=np.int64),base_duals=base_duals,guard_duals=np.asarray([guard_duals.get(i,0.) for i in working]))
    delta,gdelta,violation=evaluate(z)
    # A single analytically determined feasibility contraction, not a dose search.
    # It repairs numerical/partial working-set violation against EVERY guard.
    bad=gdelta<0;alpha=min(1.,float(np.min((gap-required)[bad]/(-gdelta[bad])))) if bad.any() else 1.
    alpha=max(0.,alpha)
    if alpha<1.:alpha*=1-1e-9
    delta*=alpha;v=z[:n]*alpha;change=Xg@delta.T;newgap=gap+change[np.arange(len(ag)),ag,None]-change[np.arange(len(ag))[:,None],og];maximum=max(0.,float(np.max(required-newgap)))
    pred=X@delta.T;margins=old+pred[np.arange(len(actions)),actions,None]-pred[np.arange(len(actions))[:,None],other];hinge=np.maximum(kappa-margins,0.)
    objective=float(np.sum(pair_weights*hinge.reshape(-1)**2)+ridge*(v@v))
    assert np.isfinite(delta).all() and maximum<1e-8 and objective<=zero+1e-8
    policy=FrozenLinearPolicy(W+delta);changed=0
    for i,a in enumerate(ag):
        lo,hi=Xg.indptr[i:i+2];got,_=policy.choose(Xg.indices[lo:hi],Xg.data[lo:hi]);changed+=int(got!=a)
    assert changed==0, ('guard argmax changed',changed)
    report=dict(solver='OSQP diagonal hinge-slack QP with lazy sparse guard generation',osqp_version=osqp.__version__,status=status,warm_start=initial_delta is not None or initial_state is not None,solver_state_saved=state_path is not None,variables=n+ns,raw_contrast_parameters=n,positive_rows=X.shape[0],hinge_slacks=ns,guard_rows=Xg.shape[0],all_guard_inequalities=len(lower),working_guard_inequalities=len(working),rounds=len(history),history=history,
        maximum_actual_logit_guard_violation=maximum,guard_action_changes=changed,required_margin='min(kappa, half carried gap)',required_margin_min=float(required.min()),required_margin_max=float(required.max()),ridge=ridge,input_rms=scale,kappa=kappa,
        feasible_contraction=alpha,zero_residual_objective=zero,final_objective=objective,positive=diagnostics(W,delta,positive),guard_outputs=diagnostics(W,delta,guards),wall_seconds=time.monotonic()-began,
        no_hard_q_or_confidence_preservation=True,failed_suffix_guarded=False,weighting='positive equal present group/episode/segment/row; each row split between two alternative comparisons')
    return delta,report
