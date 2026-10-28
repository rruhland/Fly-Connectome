"""Bounded minimum-distance merge of actual rewarded native prefixes.

No assumed feasible anchor: only a finite solution with every original margin
and exact deployed choice verified can proceed to real games.
"""
import time
import numpy as np
import scipy.sparse as sp
import osqp
from fly_connectome.control.policy import FrozenLinearPolicy
from pong_constrained_residual import CONTRAST
from pong_raw_guarded_repair import pairs,contrast_rows


def merge(reference,cases,scale,kappa,deadline):
    began=time.monotonic();d=reference.shape[1];scale=np.asarray(scale);X=sp.vstack([c['X'] for c in cases],format='csr');actions=np.concatenate([c['actions'] for c in cases]);required=[];intervention_rows=[];seen={};conflicts=[]
    for case in cases:
        policy=FrozenLinearPolicy(case['source_weights']);margins=[];interventions={v['row']:v for v in case.get('interventions',[])}
        assert len(interventions)==len(case.get('interventions',[]))
        assert all(0<=i<len(case['actions']) for i in interventions)
        for i,a in enumerate(case['actions']):
            lo,hi=case['X'].indptr[i:i+2];ix=case['X'].indices[lo:hi];val=case['X'].data[lo:hi];choice,q=policy.choose(ix,val)
            if i in interventions:
                intervention=interventions[i];assert intervention['reward_verified'] and intervention['action']==a and intervention['source_action']==choice
                margins.append([kappa*1e-4]*2);intervention_rows.append(dict(seed=case.get('seed'),row=i,action=int(a),source_action=int(choice),required_margin=kappa*1e-4))
            else:
                assert choice==a, f"undeclared non-greedy source action at seed={case.get('seed')} row={i}"
                margins.append([min(kappa*1e-4,.5*max(q[a]-q[b],0.)) for b in range(3) if b!=a])
            key=(ix.tobytes(),val.tobytes());entry=dict(seed=case.get('seed'),row=i,action=int(a))
            if key in seen and seen[key]['action']!=a:
                if len(conflicts)<12:conflicts.append(dict(first=seen[key],second=entry))
            else:seen[key]=entry
        required.append(np.asarray(margins))
    if conflicts:
        return None,dict(status='incompatible_exact_input_actions',verified=False,rows=X.shape[0],cases=len(cases),exact_input_conflicts=conflicts,known_feasible_anchor=False,numerical_fallback=False,intervention_rows=intervention_rows,wall_seconds=time.monotonic()-began,new_observations=0)
    required=np.concatenate(required);_,old,c=pairs(X,actions,reference);lower=(required-old).reshape(-1);Z=contrast_rows(X,c,np.arange(len(lower)),scale);working=np.flatnonzero(lower>0).tolist();v=np.zeros(2*d);P=2*sp.eye(2*d,format='csc');history=[];candidate=None;status='bounded_no_verified_merge'
    for iteration in range(30):
        remaining=deadline-time.monotonic()
        if remaining<=0:break
        solver=osqp.OSQP();solver.setup(P=P,q=np.zeros(2*d),A=Z[working].tocsc(),l=lower[working],u=np.full(len(working),np.inf),verbose=False,polishing=True,eps_abs=1e-9,eps_rel=1e-9,max_iter=30000,time_limit=max(.01,min(remaining,12.)))
        solver.warm_start(x=v);result=solver.solve(raise_error=False)
        if 'infeasible' in result.info.status:status=result.info.status;break
        if result.x is None or not np.isfinite(result.x).all():status='nonfinite_no_fallback';break
        v=np.asarray(result.x);violation=lower-Z@v;maximum=max(0.,float(violation.max()));history.append(dict(round=iteration+1,working=len(working),maximum_original_logit_violation=maximum,solver_status=result.info.status,iterations=int(result.info.iter)))
        if maximum<=1e-9:
            trial=reference+CONTRAST@v.reshape(2,d)/scale;policy=FrozenLinearPolicy(trial);wrong=[]
            for i,a in enumerate(actions):
                lo,hi=X.indptr[i:i+2];got,_=policy.choose(X.indices[lo:hi],X.data[lo:hi])
                if got!=a:wrong.append(i)
            if not wrong:candidate=trial;status='verified_full_earned_trace_merge';break
            status='weak_scores_feasible_but_greedy_choices_not_merged';break
        existing=set(working);fresh=np.array([i for i in np.flatnonzero(violation>1e-9) if int(i) not in existing],dtype=int)
        if not len(fresh):status='bounded_numerical_residual_no_fallback';break
        working=sorted(existing|set(map(int,fresh[np.argsort(-violation[fresh],kind='stable')[:256]])))
    report=dict(status=status,verified=candidate is not None,rows=X.shape[0],cases=len(cases),all_inequalities=len(lower),working_inequalities=len(working),history=history,projection_cost=float(v@v),wall_seconds=time.monotonic()-began,known_feasible_anchor=False,numerical_fallback=False,required_margin='source greedy: min(kappa*1e-4, half actual source-head gap); declared reward-verified override: fixed kappa*1e-4',intervention_rows=intervention_rows,verified_exact_greedy_rows=len(actions) if candidate is not None else 0,exact_input_conflicts=[],no_confidence_growth_objective=True,new_observations=0)
    return candidate,report
