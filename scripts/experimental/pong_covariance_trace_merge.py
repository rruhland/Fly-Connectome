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
    began=time.monotonic();X=case['X'];actions=case['actions'];d=reference.shape[1];interventions={v['row']:v for v in case.get('interventions',[])};source=FrozenLinearPolicy(case['source_weights']);required=[]
    for i,a in enumerate(actions):
        lo,hi=X.indptr[i:i+2];choice,q=source.choose(X.indices[lo:hi],X.data[lo:hi])
        if i in interventions:
            v=interventions[i];assert v['reward_verified'] and v['action']==a and v['source_action']==choice;required.append([kappa*1e-4]*2)
        else:
            assert choice==a;required.append([min(kappa*1e-4,.5*max(q[a]-q[b],0.)) for b in range(3) if b!=a])
    required=np.asarray(required);_,old,contrast=pairs(X,actions,reference);lower=(required-old).reshape(-1);direction=contrast.reshape(-1,2)
    # G=L L^T, z=L^T u: target score row x u becomes (L^-1 x)^T z.
    whitened=la.solve_triangular(cholesky,X.multiply(1/scale).T.toarray(),lower=True,check_finite=False).T
    raw_gram=whitened@whitened.T;kernel=np.repeat(np.repeat(raw_gram,2,axis=0),2,axis=1)*(direction@direction.T);norm=np.sqrt(np.diag(kernel));assert (norm>0).all();kernel/=norm[:,None];kernel/=norm[None,:];kernel=(kernel+kernel.T)*.5;normalized_lower=lower/norm
    working=np.flatnonzero(lower>0).tolist();lam=np.zeros(len(lower));duals={};history=[];candidate=None;status='bounded_no_verified_covariance_merge';tight=False
    for iteration in range(30):
        remaining=deadline-time.monotonic()
        if remaining<=0:break
        tolerance=1e-11 if tight else 1e-8;solver=osqp.OSQP();K=kernel[np.ix_(working,working)]
        solver.setup(P=sp.triu(sp.csc_matrix(K),format='csc'),q=-normalized_lower[working],A=sp.eye(len(working),format='csc'),l=np.zeros(len(working)),u=np.full(len(working),np.inf),verbose=False,polishing=True,eps_abs=tolerance,eps_rel=tolerance,max_iter=30000,time_limit=max(.01,min(remaining,12.)))
        solver.warm_start(x=lam[working],y=np.array([duals.get(i,0.) for i in working]));answer=solver.solve(raise_error=False)
        if 'infeasible' in answer.info.status:status='dual_solver_'+answer.info.status;break
        if answer.x is None or not np.isfinite(answer.x).all():status='nonfinite_no_fallback';break
        lam[:]=0.;lam[working]=answer.x
        if answer.y is not None and np.isfinite(answer.y).all():duals={i:float(y) for i,y in zip(working,answer.y)}
        violation=lower-norm*(kernel@lam);maximum=max(0.,float(violation.max()));history.append(dict(round=iteration+1,working=len(working),maximum_original_target_logit_violation=maximum,solver_status=answer.info.status,iterations=int(answer.info.iter),tolerance=tolerance,primal_residual=float(answer.info.prim_res),dual_residual=float(answer.info.dual_res)))
        if maximum<=1e-9:
            coefficients=(lam/norm).reshape(-1,2);beta=np.sum(contrast*coefficients[:,:,None],axis=1);z=beta.T@whitened;u=la.solve_triangular(cholesky.T,z.T,lower=False,check_finite=False).T;trial=reference+CONTRAST@u/scale
            _,actual,_=pairs(X,actions,trial);actual_violation=max(0.,float(np.max(required-actual)));policy=FrozenLinearPolicy(trial);wrong=[]
            for i,a in enumerate(actions):
                lo,hi=X.indptr[i:i+2];got,_=policy.choose(X.indices[lo:hi],X.data[lo:hi])
                if got!=a:wrong.append(i)
            if actual_violation<=1e-9 and not wrong:candidate=trial;status='verified_full_target_covariance_merge';break
            if not tight:tight=True;continue
            status='original_target_certificate_failed_no_fallback';break
        existing=set(working);fresh=np.array([i for i in np.flatnonzero(violation>1e-9) if int(i) not in existing],dtype=int)
        if not len(fresh):
            if not tight:tight=True;continue
            status='bounded_numerical_residual_no_fallback';break
        working=sorted(existing|set(map(int,fresh[np.argsort(-violation[fresh],kind='stable')[:256]])))
    report=dict(status=status,verified=candidate is not None,target_rows=X.shape[0],target_inequalities=len(lower),working_inequalities=len(working),history=history,wall_seconds=time.monotonic()-began,known_feasible_anchor=False,numerical_fallback=False,new_observations=0,objective='minimum empirical equal-episode full score change plus fixed covariance ridge',old_actions_hard_preserved=False,hard_target_required_margin='source greedy: min(kappa*1e-4, half source gap); reward-earned forced rows: fixed kappa*1e-4',forced_rows=len(interventions),verified_exact_target_greedy_rows=X.shape[0] if candidate is not None else 0)
    if candidate is not None:report.update(maximum_actual_target_logit_violation=actual_violation,whitened_parameter_cost=float(np.sum(z*z)),scaled_coefficient_norm2=float(np.sum(u*u)))
    return candidate,report
