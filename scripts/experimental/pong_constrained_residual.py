"""Convex preference ridge with linear own-success action-margin constraints."""
import time
import numpy as np
import scipy.sparse as sparse
import osqp
from pong_rewarded_residual import episode_weights,preference_targets

CONTRAST=np.array([[1/np.sqrt(2),1/np.sqrt(6)],[-1/np.sqrt(2),1/np.sqrt(6)],[0.,-2/np.sqrt(6)]])


def load_saved_cases(directory):
    from pathlib import Path
    import json
    directory=Path(directory);m=json.loads((directory/'manifest.json').read_text());positive=[];preserve=[];guards=[]
    for kind,target in [('repair',positive),('preservation',preserve)]:
        for item in m['reused'][kind]:
            with np.load(directory/item['feature_file']) as v:target.append(dict(phase=item['phase'],X=v['X'],actions=v['actions']))
    for p in sorted((directory/'episodes').glob('*.json')):
        r=json.loads(p.read_text());kind=r.get('capture_kind')
        if kind not in ('repair','guard'):continue
        assert r['complete'] and r['observations_exact'] and r['report']['hits']>0
        with np.load(directory/r['feature_file']) as v:case=dict(phase=r['phase'],seed=r['seed'],X=v['X'],actions=v['actions'],q=v['q'])
        if not len(case['X']):continue
        (positive if kind=='repair' else preserve).append(case)
        if kind=='guard':guards.append(case)
    return positive,preserve,guards


def solve_constrained(theta,positive,preservation,guards,buffer_X,buffer_weights,ridge=1e-3,kappa_fraction=.05):
    began=time.monotonic();X=np.concatenate([c['X'] for c in positive]);a=np.concatenate([c['actions'] for c in positive]);weights=episode_weights(positive)
    q=X@theta.T;q_buffer=buffer_X@theta.T
    q_rms=max(1e-12,float(np.sqrt(np.sum(buffer_weights[:,None]*q_buffer*q_buffer)/3)))
    scale=max(1e-12,float(np.sqrt(buffer_weights@np.sum(buffer_X*buffer_X,axis=1))))
    target,changed=preference_targets(q,a,kappa_fraction*q_rms);assert changed.all()
    U=X/scale;G=U.T@(weights[:,None]*U);B=target.T@(weights[:,None]*U)
    if preservation:
        P=np.concatenate([c['X'] for c in preservation])/scale;pw=episode_weights(preservation);G+=P.T@(pw[:,None]*P)
    G+=ridge*np.eye(theta.shape[1]);d=theta.shape[1]
    matrices=[];lower=[]
    for case in guards:
        x=case['X']/scale;actions=np.asarray(case['actions']);q0=case['q'];assert np.array_equal(np.argmax(q0,axis=1),actions)
        assert np.max(np.abs(case['X']@theta.T-q0))<1e-10
        rows=np.repeat(np.arange(len(x)),2);other=np.array([[b for b in range(3) if b!=action] for action in actions]).reshape(-1)
        contrast=CONTRAST[actions[rows]]-CONTRAST[other]
        matrices.append(np.hstack([contrast[:,0,None]*x[rows],contrast[:,1,None]*x[rows]]))
        lower.append(-.5*(q0[rows,actions[rows]]-q0[rows,other]))
    A=sparse.csc_matrix(np.vstack(matrices)) if matrices else sparse.csc_matrix((0,2*d))
    l=np.concatenate(lower) if lower else np.empty(0);assert np.all(l<=1e-12),'zero residual must be feasible'
    quadratic=sparse.block_diag((sparse.csc_matrix(2*G),sparse.csc_matrix(2*G)),format='csc');linear=(-2*(CONTRAST.T@B)).reshape(-1)
    solver=osqp.OSQP();solver.setup(P=sparse.triu(quadratic,format='csc'),q=linear,A=A,l=l,u=np.full(len(l),np.inf),
        verbose=False,polishing=True,eps_abs=1e-9,eps_rel=1e-9,max_iter=30000,time_limit=30.)
    solver.warm_start(x=np.zeros(2*d));answer=solver.solve(raise_error=False)
    assert answer.info.status_val in (1,2),answer.info.status
    u=np.asarray(answer.x);delta=(CONTRAST@u.reshape(2,d))/scale;delta-=delta.mean(0,keepdims=True)
    violation=float(np.max(np.maximum(l-A@u,0))) if len(l) else 0.
    assert violation<1e-7 and np.isfinite(delta).all(), 'infeasible/nonfinite constrained solution'
    du=delta*scale;zero=float(np.sum(weights[:,None]*target*target));objective=float(zero+np.sum((du@G)*du)-2*np.sum(B*du))
    assert objective<=zero+1e-7,'constrained objective cannot be worse than feasible zero'
    unconstrained=np.linalg.solve(G,B.T).T
    unconstrained_flat=(CONTRAST.T@unconstrained).reshape(-1)
    unconstrained_violation=float(np.max(np.maximum(l-A@unconstrained_flat,0))) if len(l) else 0.
    action_match=np.argmax(q+X@delta.T,axis=1)==a
    guard_match=sum(np.count_nonzero(np.argmax(c['q']+c['X']@delta.T,axis=1)==c['actions']) for c in guards)
    guard_rows=sum(len(c['X']) for c in guards)
    result=dict(solver='OSQP',osqp_version=osqp.__version__,status=answer.info.status,iterations=answer.info.iter,
        solver_run_seconds=answer.info.run_time,wall_seconds=time.monotonic()-began,variables=2*d,constraints=len(l),
        maximum_half_margin_violation=violation,unconstrained_maximum_half_margin_violation=unconstrained_violation,
        solver_primal_residual=answer.info.prim_res,solver_dual_residual=answer.info.dual_res,
        zero_residual_objective=zero,constrained_objective=objective,input_rms=scale,incumbent_q_rms=q_rms,kappa=kappa_fraction*q_rms,ridge=ridge,
        positive_rows=len(X),positive_episodes=len(positive),preservation_rows=sum(len(c['X']) for c in preservation),guard_rows=guard_rows,
        requested_actions_after_fit_fraction=float(np.mean(action_match)),requested_actions_after_fit_weighted_fraction=float(weights@action_match),
        guard_action_fraction=guard_match/guard_rows if guard_rows else None,maximum_action_row_sum=float(np.max(np.abs(delta.sum(0)))))
    return delta,result
