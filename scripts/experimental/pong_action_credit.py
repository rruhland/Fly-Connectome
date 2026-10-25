"""One-camera counterfactual credit over the unchanged legal Pong policy path.

Exact prefix resets are an explicit training scaffold. Only native return outcomes
supply pairwise action preferences; cloned environment state is never a feature.
"""
import copy
import time
import numpy as np
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.online_vision import _context_cache_validity,_restore_context_cache_validity
from fly_connectome.control.scaling import policy_features
from pong_constrained_residual import CONTRAST


def clone_session(session):
    valid=_context_cache_validity(session.vision)
    result=copy.deepcopy(session)
    _restore_context_cache_validity(result.vision,valid)
    return result


def step_action(session,action):
    """One physical camera decision; resume the class chooser immediately after it."""
    assert type(action) is int and 0<=action<3 and '_choose_policy' not in vars(session)
    normal,q=session._choose_policy()
    session._choose_policy=lambda:(action,q)
    try:record=session.step(check_frozen=False)
    finally:del session._choose_policy
    record.update(intervention_action=action,unmodified_greedy_action=int(normal))
    return record


def continue_session(snapshot,action,deadline,trace=False):
    began=time.monotonic();session=clone_session(snapshot)
    before=session.transitions;initial=session.initial_cameras;records=[];attempted=0
    if time.monotonic()>=deadline:raise TimeoutError('suffix deadline before dispatch')
    attempted+=1;record=step_action(session,int(action));records.append(record)
    while not session.complete:
        if time.monotonic()>=deadline:break
        attempted+=1;record=session.step(check_frozen=False)
        if trace:records.append(record)
    session.verify_frozen()
    result=dict(action=int(action),complete=session.complete,hit=bool(session.summary()['hits']>0),
        executed_suffix_cameras=session.transitions-before,initial_cameras=session.initial_cameras-initial,
        wall_seconds=time.monotonic()-began,terminal_summary=session.summary())
    assert result['initial_cameras']==0 and result['executed_suffix_cameras']==attempted
    if trace:result.update(records=records,terminal_hash=fingerprint(session))
    return result


def legal_input(session,matrix):
    indices,values=policy_features(session.latest_encoded,session.scaler)
    return matrix[:,indices]@values


def preferences(outcomes,q=None):
    """No ranking between tied successes or tied failures."""
    assert len(outcomes)==3 and all(v in (0,1,False,True) for v in outcomes)
    successful=[a for a in range(3) if outcomes[a]]
    if q is not None and successful:
        successful=[max(successful,key=lambda a:(q[a],-a))]
    return [(a,b) for a in successful for b in range(3) if not outcomes[b]]


def fit_preferences(theta,states,buffer_X,buffer_weights,ridge=.001,anchor_weight=.25,maximum_rms=.05):
    """Closed-form regularized pairwise preference residual; no BP or Q targets."""
    began=time.monotonic();d=theta.shape[1]
    bx=np.asarray(buffer_X);bw=np.asarray(buffer_weights);bw=bw/bw.sum()
    scale=max(1e-12,float(np.sqrt(bw@np.sum(bx*bx,axis=1))))
    bq=bx@theta.T;bq-=bq.mean(1,keepdims=True)
    qrms=max(1e-12,float(np.sqrt(np.sum(bw[:,None]*bq*bq)/3)));kappa=.05*qrms
    positive=[s for s in states if preferences(s['outcomes'])]
    if not positive:return None,dict(strict_preference_states=0,reason='all measured actions tied; no invented labels')
    phases=sorted(set(s['phase'] for s in positive));rows=[];targets=[];weights=[]
    for phase in phases:
        phase_states=[s for s in positive if s['phase']==phase];seeds=sorted(set(s['seed'] for s in phase_states))
        for seed in seeds:
            selected=[s for s in phase_states if s['seed']==seed]
            for s in selected:
                x=np.asarray(s['x']);q=x@theta.T;pairs=preferences(s['outcomes'],q)
                for good,bad in pairs:
                    c=CONTRAST[good]-CONTRAST[bad]
                    rows.append(np.concatenate((c[0]*x/scale,c[1]*x/scale)))
                    targets.append(max(0.,kappa-(q[good]-q[bad])))
                    weights.append(1/(len(phases)*len(seeds)*len(selected)*len(pairs)))
    Z=np.asarray(rows);target=np.asarray(targets);w=np.asarray(weights);assert abs(w.sum()-1)<1e-12
    # A zero-output penalty, not a hard action/confidence constraint. Every sampled
    # baseline episode/phase gets equal total weight, including no-contrast states.
    anchor=np.array([s['x'] for s in states])/scale;aw=[]
    for s in states:
        ps=[v for v in states if v['phase']==s['phase']];ns=len(set(v['seed'] for v in ps));nt=sum(v['seed']==s['seed'] for v in ps)
        aw.append(1/(2*ns*nt))
    G=Z.T@(w[:,None]*Z)+ridge*np.eye(2*d)
    common=anchor.T@(np.asarray(aw)[:,None]*anchor)*(anchor_weight/3)
    G[:d,:d]+=common;G[d:,d:]+=common
    rhs=Z.T@(w*target);coef=np.linalg.solve(G,rhs).reshape(2,d)
    raw=CONTRAST@coef/scale;raw-=raw.mean(0,keepdims=True)
    output=bx@raw.T;raw_rms=float(np.sqrt(np.sum(bw[:,None]*output*output)/3));alpha=min(1.,maximum_rms/max(1e-30,raw_rms));delta=alpha*raw
    def diag(update):
        pred=Z@(CONTRAST.T@(update*scale)).reshape(-1)
        q0=np.array([np.asarray(s['x'])@theta.T for s in states]);q1=q0+np.array([np.asarray(s['x'])@update.T for s in states])
        preferred=[];pw=[]
        for s in positive:
            q=np.asarray(s['x'])@(theta+update).T
            for a,b in preferences(s['outcomes'],np.asarray(s['x'])@theta.T):preferred.append(q[a]>q[b]);pw.append(q[a]-q[b]>=kappa-1e-12)
        repairs=[s for s in states if not s['outcomes'][s['base_action']] and any(s['outcomes'])]
        repaired=[bool(s['outcomes'][int(np.argmax(np.asarray(s['x'])@(theta+update).T))]) for s in repairs]
        return dict(repair_states=len(repairs),repair_states_greedy_success=sum(repaired),repair_state_success_fraction=float(np.mean(repaired)) if repaired else None,strict_pair_preference_fraction=float(np.mean(preferred)),desired_margin_fraction=float(np.mean(pw)),
            sampled_state_action_change=float(np.mean(np.argmax(q0,1)!=np.argmax(q1,1))),
            weighted_residual_loss=float(w@((pred-target)**2)))
    report=dict(strict_preference_states=len(positive),strict_pairs=len(Z),sampled_states=len(states),ridge=ridge,soft_anchor_weight=anchor_weight,
        kappa=kappa,input_rms=scale,carried_legal_contrast_q_rms=qrms,raw_legal_contrast_rms=raw_rms,applied_legal_contrast_rms=raw_rms*alpha,
        alpha=alpha,maximum_rms=maximum_rms,raw=diag(raw),applied=diag(delta),fit_seconds=time.monotonic()-began,
        phase_contrast_states={p:sum(s['phase']==p for s in positive) for p in ('left','right')})
    assert np.isfinite(delta).all() and np.max(abs(delta.sum(0)))<1e-10
    return delta,report
