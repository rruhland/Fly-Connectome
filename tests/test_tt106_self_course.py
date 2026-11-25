from types import SimpleNamespace

import numpy as np

from scripts.experimental import tt106_self_course as study
from scripts.experimental.tt4_transfer_trial import _read


def test_fork_requires_real_far_separation_and_precedes_near_bounce():
    a=SimpleNamespace(rival_contacts=1,events=[dict(kind='contact',side=-1)])
    assert study.first_choice(a) is None
    a.events.append(dict(kind='separation',collider='rival_racket'))
    assert study.first_choice(a)[0]==0
    a.events.append(dict(kind='legal_return',side=-1))
    assert study.first_choice(a) is None


def test_full_reference_search_can_change_early_preparation_without_switching_endpoint():
    joint=np.array([0.,-.6,-.3,1.4,-.36,.0])
    a=SimpleNamespace(joints=joint,data=SimpleNamespace(ctrl=joint.copy(),qvel=np.zeros(6)),
                      _near_dofs=np.arange(6),control_dt=study.DT,_arm_ids={1:[None,None,np.arange(6)]})
    coach=object.__new__(study.Coach)
    coach.a=a
    seeds=[dict(q=np.array([.09,-.3,-.28,1.4,-.55]),dq=np.zeros(5),time=.38)]
    variants=np.zeros((2,17))
    variants[1,10]=1.0
    result=coach.paths(seeds,np.array([0,0]),40,variants)
    assert result.shape==(2,40,6)
    assert abs(result[0,5,0]-result[1,5,0])>.07
    np.testing.assert_allclose(result[0,-1],result[1,-1],atol=1e-8)


def test_population_uses_disjoint_own_practice_and_full_game_releases():
    p=_read(study.PLAN)
    a,b=({r['id'] for r in p[k]} for k in ('train','held'))
    assert len(a)==28 and len(b)==6
    assert a.isdisjoint(b)
    assert (a|b).isdisjoint(p['excluded'])
