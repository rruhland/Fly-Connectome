"""Native next-own-return outcomes and bounded held-command discovery.

Holding a command is TRAIN exploration only. Successful sequences are rewarded
self-imitation witnesses, not causal labels for each individual forced action.
"""
import time
import numpy as np
from pong_action_credit import clone_session,step_action,legal_input


def outcome_from_record(record):
    if record['hits']>0:return 1,'next_own_hit'
    if record['terminated']:
        if record['reward']<0:return 0,'own_miss'
        return None,'opponent_ended_censored'
    if record['truncated']:return None,'native_timeout_censored'
    return None,None


def query_positions(transitions,last_hit,leads=(40,80,120)):
    return sorted(set(max(last_hit+1,transitions-lead) for lead in leads if max(last_hit+1,transitions-lead)<transitions))


def burst_plan(transitions,positions):
    seen=set();result=[]
    for hold in (10,20,40,80):
        for t in positions:
            effective=min(hold,transitions-t)
            for action in range(3):
                key=(t,effective,action)
                if key in seen:continue
                seen.add(key);result.append(dict(transition=t,hold=hold,effective_hold=effective,action=action))
    return result


def branch(snapshot,action,hold,matrix,deadline,max_cameras):
    began=time.monotonic();session=clone_session(snapshot);start=session.transitions;before_hits=session.episode_hits
    xs=[];actions=[];qs=[];forced=[];records=[];reason='camera_cap_censored';outcome=None
    for index in range(max_cameras):
        if time.monotonic()>=deadline:reason='wall_deadline_censored';break
        x=legal_input(session,matrix);normal,q=session._choose_policy();override=index<hold
        record=step_action(session,int(action)) if override else session.step(check_frozen=False)
        xs.append(x);actions.append(record['action']);qs.append(q.tolist());forced.append(override);records.append(record)
        outcome,stop=outcome_from_record(record)
        if stop is not None:reason=stop;break
    session.verify_frozen()
    delta=session.episode_hits-before_hits
    if outcome==1:assert delta==1
    elif outcome==0:assert delta==0
    receipt=dict(action=int(action),hold=int(hold),outcome=outcome,stop_reason=reason,outcome_observed=outcome is not None,
        prefix_transitions_inherited=start,previous_own_hits=before_hits,delta_own_hits=delta,
        suffix_transition_cameras=session.transitions-start,initial_cameras=session.initial_cameras-snapshot.initial_cameras,
        forced_camera_decisions=sum(forced),native_rally_complete=session.complete,report=session.summary(),wall_seconds=time.monotonic()-began)
    assert receipt['initial_cameras']==0 and receipt['suffix_transition_cameras']==len(records)
    data=dict(X=np.asarray(xs),actions=np.asarray(actions),q=np.asarray(qs),forced=np.asarray(forced),records=records)
    return receipt,data,session
