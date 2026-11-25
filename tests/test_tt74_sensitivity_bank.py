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
