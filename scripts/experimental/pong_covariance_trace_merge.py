"""Hard earned-target choices with empirical full-score soft preservation."""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp
from fly_connectome.control.policy import FrozenLinearPolicy
from pong_constrained_residual import CONTRAST
from pong_raw_guarded_repair import pairs


def fit(reference,case,scale,cholesky,kappa,deadline):
    began=time.monotonic();cases=case if isinstance(case,list) else [case];X=sp.vstack([c['X'] for c in cases],format='csr');actions=np.concatenate([c['actions'] for c in cases]);d=reference.shape[1];required=[];forced_rows=0
    for case in cases:
        interventions={v['row']:v for v in case.get('interventions',[])};source=FrozenLinearPolicy(case['source_weights']);forced_rows+=len(interventions)
        for i,a in enumerate(case['actions']):
            lo,hi=case['X'].indptr[i:i+2];choice,q=source.choose(case['X'].indices[lo:hi],case['X'].data[lo:hi])
            if i in interventions:
                v=interventions[i];assert v['reward_verified'] and v['action']==a and v['source_action']==choice;required.append([kappa*1e-4]*2)
            else:
                assert choice==a;required.append([min(kappa*1e-4,.5*max(q[a]-q[b],0.)) for b in range(3) if b!=a])
    required=np.asarray(required);full_X=X;full_actions=actions;full_required=required;seen=set();keep=[]
    for i,a in enumerate(actions):
        lo,hi=X.indptr[i:i+2];key=(X.indices[lo:hi].tobytes(),X.data[lo:hi].tobytes(),int(a),required[i].tobytes())
        if key not in seen:seen.add(key);keep.append(i)
    X=X[keep].tocsr();actions=actions[keep];required=required[keep];_,old,contrast=pairs(X,actions,reference);lower=(required-old).reshape(-1);direction=contrast.reshape(-1,2)
    # G=L L^T, z=L^T u: target score row x u becomes (L^-1 x)^T z.
    whitened=la.solve_triangular(cholesky,X.multiply(1/scale).T.toarray(),lower=True,check_finite=False).T
    norm=np.sqrt(np.repeat(np.einsum('ij,ij->i',whitened,whitened),2)*np.einsum('ij,ij->i',direction,direction));assert (norm>0).all();normalized_lower=lower/norm
    def kernel_columns(indices):
        unique,inverse=np.unique(np.asarray(indices)//2,return_inverse=True);dots=whitened@whitened[unique].T;columns=np.repeat(dots[:,inverse],2,axis=0);columns*=direction@direction[indices].T;columns/=norm[:,None];columns/=norm[indices][None,:];return columns
    working=np.flatnonzero(lower>0).tolist();lam=np.zeros(len(lower));duals={};history=[];candidate=None;status='bounded_no_verified_covariance_merge';tight=False
    for iteration in range(30):
        remaining=deadline-time.monotonic()
        if remaining<=0:break
        tolerance=1e-11 if tight else 1e-8;solver=osqp.OSQP();columns=kernel_columns(working);K=columns[working,:];K=(K+K.T)*.5
        solver.setup(P=sp.triu(sp.csc_matrix(K),format='csc'),q=-normalized_lower[working],A=sp.eye(len(working),format='csc'),l=np.zeros(len(working)),u=np.full(len(working),np.inf),verbose=False,polishing=True,eps_abs=tolerance,eps_rel=tolerance,max_iter=30000,time_limit=max(.01,min(remaining,12.)))
        solver.warm_start(x=lam[working],y=np.array([duals.get(i,0.) for i in working]));answer=solver.solve(raise_error=False)
        if 'infeasible' in answer.info.status:status='dual_solver_'+answer.info.status;break
        if answer.x is None or not np.isfinite(answer.x).all():status='nonfinite_no_fallback';break
        lam[:]=0.;lam[working]=answer.x
        if answer.y is not None and np.isfinite(answer.y).all():duals={i:float(y) for i,y in zip(working,answer.y)}
        violation=lower-norm*(columns@lam[working]);maximum=max(0.,float(violation.max()));history.append(dict(round=iteration+1,working=len(working),maximum_original_target_logit_violation=maximum,solver_status=answer.info.status,iterations=int(answer.info.iter),tolerance=tolerance,primal_residual=float(answer.info.prim_res),dual_residual=float(answer.info.dual_res)))
        if maximum<=1e-9:
            coefficients=(lam/norm).reshape(-1,2);beta=np.sum(contrast*coefficients[:,:,None],axis=1);z=beta.T@whitened;u=la.solve_triangular(cholesky.T,z.T,lower=False,check_finite=False).T;trial=reference+CONTRAST@u/scale
            _,actual,_=pairs(full_X,full_actions,trial);actual_violation=max(0.,float(np.max(full_required-actual)));policy=FrozenLinearPolicy(trial);wrong=[]
            for i,a in enumerate(full_actions):
                lo,hi=full_X.indptr[i:i+2];got,_=policy.choose(full_X.indices[lo:hi],full_X.data[lo:hi])
                if got!=a:wrong.append(i)
            if actual_violation<=1e-9 and not wrong:candidate=trial;status='verified_full_target_covariance_merge';break
            if not tight:tight=True;continue
            status='original_target_certificate_failed_no_fallback';break
        existing=set(working);fresh=np.array([i for i in np.flatnonzero(violation>1e-9) if int(i) not in existing],dtype=int)
        if not len(fresh):
            if not tight:tight=True;continue
            status='bounded_numerical_residual_no_fallback';break
        working=sorted(existing|set(map(int,fresh[np.argsort(-violation[fresh],kind='stable')[:256]])))
    report=dict(status=status,verified=candidate is not None,target_rows=full_X.shape[0],target_inequalities=2*full_X.shape[0],unique_hard_rows=X.shape[0],unique_hard_inequalities=len(lower),hard_cases=len(cases),working_inequalities=len(working),history=history,wall_seconds=time.monotonic()-began,known_feasible_anchor=False,numerical_fallback=False,new_observations=0,objective='minimum empirical equal-episode full score change plus fixed covariance ridge',old_actions_hard_preserved=False,hard_target_required_margin='source greedy: min(kappa*1e-4, half source gap); reward-earned forced rows: fixed kappa*1e-4',forced_rows=forced_rows,verified_exact_target_greedy_rows=full_X.shape[0] if candidate is not None else 0,full_kernel_materialized=False,deduplication='exact feature/action/two-required-margins; final certificate uses every original row')
    if candidate is not None:report.update(maximum_actual_target_logit_violation=actual_violation,whitened_parameter_cost=float(np.sum(z*z)),scaled_coefficient_norm2=float(np.sum(u*u)))
    return candidate,report
