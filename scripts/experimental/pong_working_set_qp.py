"""Exact constraint generation for the unchanged convex preference objective."""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp
from pong_constrained_residual import CONTRAST
from pong_rewarded_residual import episode_weights,preference_targets
from pong_behavior_preservation import required_guard_margin


def solve_qp(P,q,A,lower,*,initial=None,initial_rows=(),time_limit=60.,batch=128):
    began=time.monotonic();P=sp.csc_matrix(P);A=sp.csc_matrix(A);q=np.asarray(q);lower=np.asarray(lower)
    factor=la.cho_factor(P.toarray(),lower=True,check_finite=False)
    inv=lambda v:la.cho_solve(factor,v,check_finite=False)
    x=inv(-q) if initial is None else np.asarray(initial).copy();working=sorted(set(map(int,initial_rows)));history=[]
    if not working:
        violation=lower-A@x;working=np.flatnonzero(violation>1e-8)[np.argsort(-violation[violation>1e-8],kind='stable')[:batch]].tolist()
    old_y={}
    for iteration in range(40):
        remaining=time_limit-(time.monotonic()-began)
        if remaining<=0:raise RuntimeError('working-set wall budget exhausted: '+str(history))
        work=A[working];lo=lower[working];solver=osqp.OSQP()
        solver.setup(P=sp.triu(P,format='csc'),q=q,A=work,l=lo,u=np.full(len(lo),np.inf),verbose=False,polishing=True,
            eps_abs=1e-9,eps_rel=1e-9,max_iter=30000,time_limit=remaining)
        solver.warm_start(x=x,y=np.array([old_y.get(i,0.) for i in working]));answer=solver.solve(raise_error=False)
        if answer.info.status_val not in (1,2):
            candidate=np.asarray(answer.x)
            if not np.isfinite(candidate).all():raise RuntimeError('nonfinite working-set OSQP: '+str(answer.info.status))
            ax=A@candidate;bad=ax<lower;alpha=min(1.,float(np.min(lower[bad]/ax[bad]))) if np.any(bad) else 1.
            candidate*=max(0.,alpha)*(1-1e-12);violation=max(0.,float(np.max(lower-A@candidate))) if len(lower) else 0.
            objective=float(.5*candidate@(P@candidate)+q@candidate)
            if violation>1e-7 or objective>1e-7:raise RuntimeError('unusable finite working-set iterate: '+str(answer.info.status))
            return candidate,dict(status='approximate feasible',approximate=True,solver_status=answer.info.status,wall_seconds=time.monotonic()-began,
                working_rows=len(working),full_rows=len(lower),rounds=len(history)+1,maximum_full_violation=violation,duality_gap=None,
                feasible_contraction=alpha,objective_relative_to_zero=objective,history=history)
        x=np.asarray(answer.x);y=np.minimum(np.asarray(answer.y),0.)
        full_violation=lower-A@x;maximum=max(0.,float(full_violation.max())) if len(lower) else 0.
        primal=float(.5*x@(P@x)+q@x);v=q+work.T@y;dual=float(-.5*v@inv(v)-lo@y);gap=primal-dual
        history.append(dict(round=iteration,working_rows=len(working),maximum_full_violation=maximum,primal_objective=primal,dual_objective=dual,duality_gap=gap,solver_iterations=answer.info.iter))
        if maximum<=1e-7 and -1e-7<=gap<=1e-7:
            return x,dict(status='solved',wall_seconds=time.monotonic()-began,working_rows=len(working),full_rows=len(lower),rounds=len(history),
                maximum_full_violation=maximum,duality_gap=gap,history=history,certificate='restricted optimum is full-feasible; exact SPD dual lower bound with nonpositive multipliers')
        if maximum<=1e-7 and primal<=1e-7:
            return x,dict(status='approximate feasible',approximate=True,wall_seconds=time.monotonic()-began,working_rows=len(working),full_rows=len(lower),
                rounds=len(history),maximum_full_violation=maximum,duality_gap=gap,objective_relative_to_zero=primal,history=history)
        fresh=np.flatnonzero(full_violation>1e-8);existing=set(working);fresh=np.array([i for i in fresh if i not in existing],dtype=int)
        if not len(fresh):raise RuntimeError('working-set optimality tolerance not met: '+str(history[-1]))
        selected=fresh[np.argsort(-full_violation[fresh],kind='stable')[:batch]];old_y={i:float(a) for i,a in zip(working,y)};working=sorted(existing|set(map(int,selected)))
    raise RuntimeError('working-set round budget exhausted: '+str(history))


def solve_constrained(theta,positive,preservation,guards,buffer_X,buffer_weights,*,matching_preservation=None,matching_weight=.25,positive_weight=4.,ridge=.001,kappa_fraction=.05,initial_delta=None,guard_margin_cap=None):
    began=time.monotonic();X=np.concatenate([c['X'] for c in positive]);a=np.concatenate([c['actions'] for c in positive]);weights=episode_weights(positive)
    q=X@theta.T;qb=buffer_X@theta.T;q_rms=max(1e-12,float(np.sqrt(np.sum(buffer_weights[:,None]*qb*qb)/3)))
    scale=max(1e-12,float(np.sqrt(buffer_weights@np.sum(buffer_X*buffer_X,axis=1))));target,changed=preference_targets(q,a,kappa_fraction*q_rms);assert changed.all()
    U=X/scale;G=positive_weight*(U.T@(weights[:,None]*U));B=positive_weight*(target.T@(weights[:,None]*U))
    for cases,multiplier in ((preservation,1.),(matching_preservation or [],matching_weight)):
        if cases:
            M=np.concatenate([c['X'] for c in cases])/scale;w=episode_weights(cases);G+=multiplier*(M.T@(w[:,None]*M))
    G+=ridge*np.eye(theta.shape[1]);matrices=[];bounds=[]
    for c in guards:
        x=c['X']/scale;actions=c['actions'];q0=c['q'];assert np.max(abs(c['X']@theta.T-q0))<1e-10 and np.array_equal(np.argmax(q0,axis=1),actions)
        rows=np.repeat(np.arange(len(x)),2);other=np.array([[b for b in range(3) if b!=v] for v in actions]).reshape(-1)
        contrast=CONTRAST[actions[rows]]-CONTRAST[other];matrices.append(np.hstack([contrast[:,0,None]*x[rows],contrast[:,1,None]*x[rows]]))
        gap=q0[rows,actions[rows]]-q0[rows,other];bounds.append(required_guard_margin(gap,guard_margin_cap)-gap)
    A=sp.csc_matrix(np.vstack(matrices));lower=np.concatenate(bounds);assert np.all(lower<=1e-12)
    P=sp.block_diag((sp.csc_matrix(2*G),sp.csc_matrix(2*G)),format='csc');linear=(-2*(CONTRAST.T@B)).reshape(-1)
    initial=(CONTRAST.T@(initial_delta*scale)).reshape(-1) if initial_delta is not None else None
    binding=np.flatnonzero(abs(A@initial-lower)<1e-7) if initial is not None else []
    u,info=solve_qp(P,linear,A,lower,initial=initial,initial_rows=binding)
    delta=(CONTRAST@u.reshape(2,theta.shape[1]))/scale;delta-=delta.mean(0,keepdims=True)
    zero=float(positive_weight*np.sum(weights[:,None]*target*target));du=delta*scale;objective=float(zero+np.sum((du@G)*du)-2*np.sum(B*du))
    assert objective<=zero+1e-7 and info['maximum_full_violation']<=1e-7
    guard_choices=all(np.array_equal(np.argmax(c['q']+c['X']@delta.T,axis=1),c['actions']) for c in guards)
    result=dict(solver='OSQP working-set',osqp_version=osqp.__version__,status=info['status'],wall_seconds=time.monotonic()-began,
        variables=len(u),constraints=len(lower),maximum_half_margin_violation=info['maximum_full_violation'],working_set=info,
        zero_residual_objective=zero,constrained_objective=objective,input_rms=scale,incumbent_q_rms=q_rms,kappa=kappa_fraction*q_rms,ridge=ridge,
        positive_rows=len(X),positive_episodes=len(positive),preservation_rows=sum(len(c['X']) for c in preservation),guard_rows=sum(len(c['X']) for c in guards),
        matching_preservation_rows=sum(len(c['X']) for c in matching_preservation or []),matching_preservation_weight=matching_weight,positive_weight=positive_weight,guard_margin_cap=guard_margin_cap,
        requested_actions_after_fit_fraction=float(np.mean(np.argmax(q+X@delta.T,axis=1)==a)),requested_actions_after_fit_weighted_fraction=float(weights@(np.argmax(q+X@delta.T,axis=1)==a)),guard_action_fraction=float(guard_choices))
    return delta,result
