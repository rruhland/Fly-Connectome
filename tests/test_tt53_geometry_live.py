from collections import Counter

import numpy as np

from scripts.experimental.tt53_geometry_live import Basis
from scripts.experimental.tt53_signed_practice import N


def make_case(monkeypatch,winner):
    def spy(self,goals,*,long=False,traces=False):
        n=len(goals)
        feedback=np.zeros((n,9))
        feedback[:,1]=1.
        legal=np.zeros(n,bool)
        if long and winner:
            legal[0]=True
        return np.arange(n),legal,None,np.zeros((n,11)),feedback
    monkeypatch.setattr(Basis.__mro__[1],"trial",spy)
    actor=Basis.__new__(Basis)
    actor.budget=28
    actor.stride=np.full(6,.06)
    actor.lo=np.full(6,-1.0)
    actor.hi=np.full(6,1.0)
    actor.diag=Counter()
    x=np.array([.06,-.75,1.04,.15,-4.0,-.9])
    q=np.zeros(6)
    actor.decision=dict(tick=10,x=x,q=q,ref=q,anchor=None,first_long=False,empty=False,inserted=None)
    from scripts.experimental.tt53_signed_practice import state
    key=state(x,q,q,np.zeros((N,6)))
    rng=np.random.default_rng(432)
    actor.library=[dict(id=i,key=key,eta=10.,difference=rng.normal(0.,.75,(N,5))) for i in range(4)]
    initial=[np.zeros((N,6)) for _ in range(28)]
    actor.trial(initial)
    actor.trial(initial[:8],long=True)
    fallback=[np.zeros((N,6)) for _ in range(12)]
    actor.trial(fallback)
    return actor,fallback


def test_replace_only_four_existing_slots_after_no_legal_incumbent(monkeypatch):
    actor,fallback=make_case(monkeypatch,False)
    assert sum(np.any(g!=0) for g in fallback)==4
    assert len(fallback)==12
    assert not any(np.any(g!=0) for g in fallback[:8])
    assert actor.diag["learned_basis_slots"]==4
    assert all(np.all(abs(g)<=1.) for g in fallback)


def test_never_reorder_fallback_after_primary_screen_finds_a_legal_stroke(monkeypatch):
    actor,fallback=make_case(monkeypatch,True)
    assert not any(np.any(g!=0) for g in fallback)
    assert actor.diag["learned_basis_slots"]==0
