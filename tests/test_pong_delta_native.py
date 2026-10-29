import numpy as np

from pong_delta_native import fixed_audit_indices, restore_actual_state, audit_trace


def test_spotcheck_state_uses_only_actual_past_commands_and_hits():
    class Controller:
        pass
    c=Controller(); records=[dict(action=i%3,hits=i==6) for i in range(25)]
    restore_actual_state(c,records,6)
    assert c.ordinal==6 and not c.past_hit and c.pending_action is None
    assert not c.history[:-6].any()
    np.testing.assert_array_equal(c.history[-6:],np.eye(3)[[0,1,2,0,1,2]])
    restore_actual_state(c,records,20)
    assert c.past_hit
    np.testing.assert_array_equal(c.history,np.eye(3)[[i%3 for i in range(4,20)]])


def test_fixed_checks_include_first_posthit_and_endpoints():
    records=[dict(hits=i==4) for i in range(21)]
    selected=fixed_audit_indices(records)
    assert 0 in selected and 20 in selected and 5 in selected
    assert selected==fixed_audit_indices(records) and len(selected)<=9
    assert fixed_audit_indices([])==[]


def test_full_and_reconstructed_checks_agree_on_feedback_sensitive_policy():
    import scipy.sparse as sp
    from pong_learned_delta_planner import DeltaPlannerController
    class Model:
        scales=np.ones(4)
        def plan(self,state,history,baseline):
            # Real past command order and public phase affect the next action.
            action=int((np.argmax(history[-1])+np.argmax(state[29:])+1)%3)
            return action,dict(mode='planned_primitive')
    class Baseline:
        def choose(self,indices,values):
            q=np.zeros(3);q[indices]=values
            return int(np.argmax(q)),q
    model=Model();baseline=Baseline();controller=DeltaPlannerController(model)
    raw=sp.csr_matrix(np.eye(3)[np.arange(21)%3]);records=[];decisions=[]
    scenes=[np.array([[.1,.2,0,0,1,0]]) for _ in range(21)]
    for i,scene in enumerate(scenes):
        a,q=baseline.choose(raw[i].indices,raw[i].data)
        action,decision=controller.choose(scene,a)
        r=dict(action=action,q=q.tolist(),hits=i in (3,12),terminated=i==20)
        records.append(r);decisions.append(decision);controller.observe(r['hits'],r['terminated'])
    full=audit_trace(model,baseline,records,decisions,scenes,raw,True)
    sparse=audit_trace(model,baseline,records,decisions,scenes,raw,False)
    assert full['planner_replay_rows']==21 and full['full_chronological_replay']
    assert sparse['planner_replay_rows']<=9 and not sparse['full_chronological_replay']
    assert full['changed_actions']==sparse['changed_actions']
