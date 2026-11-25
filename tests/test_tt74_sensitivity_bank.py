from copy import deepcopy

import numpy as np

from scripts.experimental import tt74_sensitivity_bank as tt
from scripts.experimental.tt4_transfer_trial import _read


def setup_bank():
    x=np.array([-.010,-.83,.885,.01,-4.3,1.15])
    q=np.zeros(6)
    teacher=dict(
        obs=dict(x=x.tolist(),q=q.tolist(),ref=q.tolist()), contact_line=tt.CONTACT_Y,
        probe=8, shape=np.zeros((tt.H,6)).tolist(), J=np.eye(5).tolist(),
        body_jac=np.zeros((5,10)).tolist(),key=tt.key(x,q,q).tolist(),
        family=tt.family(x),source="offline",authority=.12,
    )
    bank=tt.Bank(dict(complete=True,observations=[dict(entry=deepcopy(teacher)) for _ in range(34)]))
    return x,q,bank


def test_candidate_transport_conditioned_on_live_optics_not_release_label():
    x,q,bank=setup_bank()
    check=x+np.array([.031,0.,-.024,0.,0.,0.])
    paths, notes=bank.suggest(check,q,q)
    assert len(paths)==len(notes)==8
    assert paths[0][-1,0]>0. and paths[0][-1,2]<0.
    assert notes[0]["transported_m"]>.01
    miss=x+np.array([.25,0.,.0,0.,0.,0.])
    paths,_=bank.suggest(miss,q,q)
    assert not paths
    for model in bank.source:
        model["authority"]=.01
    assert not bank.suggest(check,q,q)[0]


def test_prospective_games_release_disjoint_from_teacher_roster():
    plan=_read(tt.PLAN)
    training={c["id"] for c in plan["examples"]}
    assessment={r["card"]["id"] for r in plan["releases"]}
    assert len(assessment)==tt.ASSESS and not training & assessment
    assert len(training)>=tt.TRAIN


def test_runtime_reward_retains_real_sent_shape_with_shrunken_inherited_authority():
    from types import SimpleNamespace

    x,q,bank=setup_bank()
    teacher=bank.source[0]
    player=SimpleNamespace(credit_context=(x,q,q,teacher,[]), pending_tick=100, readout=bank, learned_here=0)
    for tick in range(100,155):
        tt.Conditional.sensory_credit(player,tick,np.full(6,(tick-100)*.002),tick==154)
    assert player.learned_here==1 and len(bank.acquired)==1
    assert bank.acquired[0]["authority"]==teacher["authority"]*.80
    assert bank.acquired[0]["J"]==teacher["J"]
    assert bank.acquired[0]["source"]=="paid-online"
    assert np.allclose(bank.acquired[0]["shape"][-1],np.full(6,(tt.H-1)*.002))
    assert player.credit_context is None


def test_all_twenty_continuous_points_terminate_with_identical_measured_actions():
    from scripts.experimental import tt74_sensitivity_report as result
    from scripts.experimental.tt4_transfer_trial import _hash

    r=_read(result.REPORT)
    assert r["complete"] and r["games"]==_hash(tt.OUT)
    assert len(r["rows"])==tt.ASSESS
    assert all(row["exact_commands"] and row["exact_referee"] and row["exact_placements"] for row in r["rows"])
    assert r["summary"]["stock"]["returns"]==r["summary"]["conditional"]["returns"]==54
    assert r["summary"]["conditional"]["projection-not-robust"]==6
    assert not r["summary"]["conditional"]["learned"]
    assert sum(t["clearance"]["samples"] for row in r["rows"] for t in row["replays"].values())==24349
