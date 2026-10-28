"""Nearest activity-metric readout satisfying one complete rewarded trajectory."""
import time
import numpy as np
import scipy.sparse as sp
import osqp
from pong_constrained_residual import CONTRAST
from pong_raw_guarded_repair import pairs,contrast_rows
from fly_connectome.control.policy import FrozenLinearPolicy


def project(reference,feasible,X,actions,scale,kappa,deadline,pair_constraints=()):
    began=time.monotonic();d=reference.shape[1];scale=np.asarray(scale);assert scale.shape==(d,)
    other,old,c=pairs(X,actions,reference);_,fgap,_=pairs(X,actions,feasible);required=np.minimum(kappa*1e-4,.5*np.maximum(fgap,0.));lower=(required-old).reshape(-1)
    Z=contrast_rows(X,c,np.arange(len(lower)),scale);blocks=[Z];bounds=[lower]
    for pair in pair_constraints:
        row=pair['X'].tocsr();assert row.shape==(1,d);good,bad=int(pair['good']),int(pair['bad']);a=row.indices;vrow=row.data
        rq=reference[:,a]@vrow;fq=feasible[:,a]@vrow;gap=float(fq[good]-fq[bad]);assert gap>=-1e-9
        contrast=CONTRAST[good]-CONTRAST[bad];u=row/scale;blocks.append(sp.hstack((u*contrast[0],u*contrast[1]),format='csc'));bounds.append(np.array([min(kappa*1e-4,.5*max(gap,0.))-(rq[good]-rq[bad])]))
    Z=sp.vstack(blocks,format='csc');lower=np.concatenate(bounds);vfeasible=(CONTRAST.T@(feasible-reference)*scale).reshape(-1);assert np.min(Z@vfeasible-lower)>-1e-9
    zero=np.zeros(2*d);v=vfeasible.copy();working=np.flatnonzero(lower>0).tolist();history=[];status='bounded_feasible_projection';P=sp.eye(2*d,format='csc')*2
    for iteration in range(30):
        remaining=deadline-time.monotonic()
        if remaining<=0:break
        solver=osqp.OSQP();solver.setup(P=P,q=zero,A=Z[working].tocsc(),l=lower[working],u=np.full(len(working),np.inf),verbose=False,polishing=True,eps_abs=1e-8,eps_rel=1e-8,max_iter=20000,time_limit=max(.01,min(12.,remaining)))
        solver.warm_start(x=v);answer=solver.solve(raise_error=False)
        if answer.x is None or not np.isfinite(answer.x).all():break
        v=np.asarray(answer.x);violation=lower-Z@v;history.append(dict(round=iteration+1,working_inequalities=len(working),maximum_full_violation=max(0.,float(violation.max())),solver_status=answer.info.status,iterations=int(answer.info.iter)))
        if violation.max()<=1e-8:status='full_witness_projection_solved';break
        fresh=np.array([i for i in np.flatnonzero(violation>1e-8) if int(i) not in set(working)],dtype=int)
        if not len(fresh):break
        working=sorted(set(working)|set(map(int,fresh[np.argsort(-violation[fresh],kind='stable')[:128]])))
    # Restore all original logit constraints toward a known feasible witness head
    # if the bounded working-set iterate has residual numerical violations.
    slack=Z@vfeasible-lower;direction=Z@(v-vfeasible);bad=direction<0;beta=min(1.,float(np.min(np.maximum(slack[bad],0.)/-direction[bad]))) if bad.any() else 1.;beta=max(0.,beta)
    if beta<1:beta*=1-1e-9
    v=vfeasible+beta*(v-vfeasible);candidate=reference+CONTRAST@v.reshape(2,d)/scale;policy=FrozenLinearPolicy(candidate);changed=[]
    for i,a in enumerate(actions):
        lo,hi=X.indptr[i:i+2];got,_=policy.choose(X.indices[lo:hi],X.data[lo:hi])
        if got!=a:changed.append(i)
    violation=max(0.,float(np.max(lower-Z@v)));cost=float(v@v);feasible_cost=float(vfeasible@vfeasible)
    assert np.isfinite(candidate).all() and violation<1e-8 and not changed and cost<=feasible_cost*(1+1e-6)+1e-9
    return candidate,dict(status=status,solver='OSQP nearest linear readout in frozen activity metric; rewarded trajectory plus explicitly measured pairs only',rows=X.shape[0],extra_measured_pairs=len(pair_constraints),all_inequalities=len(lower),working_inequalities=len(working),rounds=len(history),history=history,whole_witness_action_changes=changed,maximum_original_logit_violation=violation,feasibility_beta=beta,projection_cost=cost,known_feasible_cost=feasible_cost,reference_action_errors=int(np.sum(np.argmax(X@reference.T,axis=1)!=actions)),wall_seconds=time.monotonic()-began,new_observations=0,no_blanket_other_game_action_locks=True)
