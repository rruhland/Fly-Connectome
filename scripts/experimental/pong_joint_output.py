"""Joint existing linear/pairwise output deltas with active-only dual solves."""
import time
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import osqp
from pong_constrained_residual import CONTRAST
from pong_fixed_pairwise import FrozenPairwiseReadout


def metric(rows,basis,scale):
    d=len(scale);m=basis.output_dim;G=np.zeros((d+m,d+m));episodes=0
    for X in rows:
        Z=sp.csr_matrix(X).multiply(1/scale).tocsr();P=basis.batch(X);n=X.shape[0];G[:d,:d]+=(Z.T@Z).toarray()/n;G[:d,d:]+=np.asarray(Z.T@P)/n;G[d:,d:]+=P.T@P/n;episodes+=1
    G/=episodes;G[d:,:d]=G[:d,d:].T;raw=float(np.trace(G[:d,:d])/d);phi=float(np.trace(G[d:,d:])/m);assert raw>0 and phi>0;units=np.r_[np.full(d,np.sqrt(raw)),np.full(m,np.sqrt(phi))];G/=units[:,None];G/=units[None,:];G.flat[::d+m+1]+=.01
    return G,np.asarray(scale)*np.sqrt(raw),np.sqrt(phi),dict(episodes=episodes,raw_mean_diagonal_before_optimizer_normalization=raw,phi_mean_diagonal_before_optimizer_normalization=phi,ridge_fraction=.01,crossblock_correlations_included=True,conditioning_is_optimizer_only_not_deployed_features=True)


def undo(reference,theta,raw_denom,phi_denom):
    d=len(raw_denom);theta=np.asarray(theta);dw=CONTRAST@theta[:,:d]/raw_denom;dr=CONTRAST@theta[:,d:]/phi_denom
    return reference.base.weights+dw,reference.residual+dr


def fit(reference,cases,raw_denom,phi_denom,cholesky,kappa,deadline,reference_sha256=None):
    began=time.monotonic();fullX=sp.vstack([c['X'] for c in cases],format='csr');fulla=np.concatenate([c['actions'] for c in cases]);required=[];refq=[];source_rows=0;different_sources=0;intervention_rows=[]
    for c in cases:
        source=c['source_readout'];same=reference_sha256 is not None and c.get('source_policy_sha256')==reference_sha256;different_sources+=int(not same);interventions={v['row']:v for v in c.get('interventions',[])};assert len(interventions)==len(c.get('interventions',[])) and all(0<=i<len(c['actions']) for i in interventions)
        for i,a in enumerate(c['actions']):
            lo,hi=c['X'].indptr[i:i+2];ix=c['X'].indices[lo:hi];xx=c['X'].data[lo:hi];got,q=source.choose(ix,xx)
            if i in interventions:
                v=interventions[i];assert c.get('causal_audit_passed') and v['reward_verified'] and v['native_total_hit_advantage']>0 and v['action']==a and v['source_action']==got and got!=a;required.append([kappa]*2);intervention_rows.append(dict(seed=c.get('seed'),row=i,action=int(a),source_action=int(got),margin=kappa,joint_burst_effect_not_individual_causality=True))
            else:assert got==a;required.append([min(kappa*1e-4,.5*max(q[a]-q[b],0)) for b in range(3) if b!=a])
            refq.append(q if same else reference.choose(ix,xx)[1]);source_rows+=1
    required=np.asarray(required);refq=np.asarray(refq);fullrequired=required.copy();keep=[];seen=set()
    for i,a in enumerate(fulla):
        lo,hi=fullX.indptr[i:i+2];key=(fullX.indices[lo:hi].tobytes(),fullX.data[lo:hi].tobytes(),int(a),required[i].tobytes())
        if key not in seen:seen.add(key);keep.append(i)
    X=fullX[keep].tocsr();actions=fulla[keep];required=required[keep];refq=refq[keep];d=len(raw_denom);Z=X.multiply(1/raw_denom).tocsr();Phi=reference.basis.batch(X)/phi_denom;other=np.array([[b for b in range(3) if b!=a] for a in actions]);row=np.arange(len(actions));old=refq[row,actions,None]-refq[row[:,None],other];lower=(required-old).ravel();direction=(CONTRAST[actions,None,:]-CONTRAST[other]).reshape(-1,2);fnorm=np.asarray(Z.power(2).sum(1)).ravel()+np.einsum('ij,ij->i',Phi,Phi);norm=np.sqrt(np.repeat(fnorm,2)*np.einsum('ij,ij->i',direction,direction));assert (norm>0).all();ln=lower/norm;working=np.flatnonzero(lower>0).tolist();lambdas={};duals={};history=[];result=None;tight=False;status='bounded_no_verified_joint_update';reserve=max(1.,len(fulla)*.00012)
    if not working:working=[int(np.argmax(lower))]  # Inactive one-row dual has exact zero-update optimum.
    for iteration in range(30):
        remaining=deadline-time.monotonic()
        if remaining<=reserve:break
        unique,inverse=np.unique(np.asarray(working)//2,return_inverse=True);F=np.vstack((Z[unique].T.toarray(),Phi[unique].T));M=la.cho_solve((cholesky,True),F,check_finite=False);S=np.asarray(Z[unique]@M[:d])+Phi[unique]@M[d:];K=S[inverse[:,None],inverse[None,:]]*(direction[working]@direction[working].T);K/=norm[working,None];K/=norm[None,working];K=(K+K.T)*.5;solver=osqp.OSQP();tol=1e-12 if tight else 1e-9;remaining=deadline-time.monotonic()
        if remaining<=reserve:break
        solver.setup(P=sp.triu(sp.csc_matrix(K),format='csc'),q=-ln[working],A=sp.eye(len(working),format='csc'),l=np.zeros(len(working)),u=np.full(len(working),np.inf),verbose=False,polishing=True,eps_abs=tol,eps_rel=tol,max_iter=30000,time_limit=max(.01,min(remaining-reserve,8.)));solver.warm_start(x=np.array([lambdas.get(i,0.) for i in working]),y=np.array([duals.get(i,0.) for i in working]));answer=solver.solve(raise_error=False)
        if answer.x is None or not np.isfinite(answer.x).all() or 'infeasible' in answer.info.status:status='unverified_dual_solver_'+answer.info.status;break
        lam=np.asarray(answer.x);lambdas={i:float(v) for i,v in zip(working,lam)};duals={i:float(v) for i,v in zip(working,answer.y)} if answer.y is not None else {};theta=((lam/norm[working])[:,None]*direction[working]).T@M[:,inverse].T;delta=(np.asarray(Z@theta[:,:d].T)+Phi@theta[:,d:].T)@CONTRAST.T;change=delta[row,actions,None]-delta[row[:,None],other];violation=lower-change.ravel();maximum=max(0.,float(violation.max()));history.append(dict(round=iteration+1,working=len(working),active_input_RHS=len(unique),maximum_original_violation=maximum,solver_status=answer.info.status,iterations=int(answer.info.iter)))
        if maximum<=1e-9:
            W,R=undo(reference,theta,raw_denom,phi_denom);policy=FrozenPairwiseReadout(W,reference.basis,R);wrong=[];actual_violation=0.;checked=0
            for i,a in enumerate(fulla):
                if i%128==0 and time.monotonic()>=deadline:status='original_certificate_budget_partial';break
                lo,hi=fullX.indptr[i:i+2];got,q=policy.choose(fullX.indices[lo:hi],fullX.data[lo:hi]);actual_violation=max(actual_violation,float(np.max(fullrequired[i]-np.array([q[a]-q[b] for b in range(3) if b!=a]))));checked+=1
                if got!=a:wrong.append(i)
            if checked==len(fulla) and not wrong and actual_violation<=1e-9:result=(W,R,theta);status='verified_all_original_joint_deployed_choices';break
            if checked<len(fulla):break
            if not tight:tight=True;continue
            status='original_deployed_certificate_failed';break
        existing=set(working);fresh=np.array([i for i in np.flatnonzero(violation>1e-9) if int(i) not in existing],dtype=int)
        if not len(fresh):
            if not tight:tight=True;continue
            status='bounded_joint_numerical_residual';break
        working=sorted(existing|set(map(int,fresh[np.argsort(-violation[fresh],kind='stable')[:256]])))
    report=dict(status=status,verified=result is not None,wall_seconds=time.monotonic()-began,original_rows=len(fulla),unique_hard_rows=len(actions),original_inequalities=2*len(fulla),hard_cases=len(cases),source_exact_choose_rows=source_rows-len(intervention_rows),source_audited_rows=source_rows,reward_verified_intervention_rows=intervention_rows,source_policies_distinct_from_reference_cases=different_sources,reference_Q_is_carried_nonlinear_Q=True,trainable_existing_coordinates=2*(d+reference.basis.output_dim),working_inequalities=len(working),history=history,all_row_whitening=False,all_row_kernel_columns=False,dense_primal_constraint_matrix=False,active_RHS_only=True,numerical_fallback=False,new_observations=0,original_deployed_rows_certified=len(fulla) if result is not None else 0,maximum_actual_logit_violation=actual_violation if result is not None else None,source_margin='greedy:min(kappa*1e-4,halfsourcegap); explicit native-reward-verified burst differences:fixed kappa')
    return result,report
