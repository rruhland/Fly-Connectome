from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt54_live_wing import BANK, Wing, rp
from scripts.experimental.tt54_live_wing_v3 import Wing as CheckedWing


def player(seen=False):
    return SimpleNamespace(ready=SimpleNamespace(seen_inbound=seen,predictor=None),engine=None)


def test_reserve_requires_an_actually_observed_outward_transition_and_encoder_entry():
    packet=rp._read(BANK)
    body=packet['trajectories'][0]
    willing=SimpleNamespace(ik=SimpleNamespace(low=np.full(5,-2.0),high=np.full(5,2.0)),route=lambda *a,**k:True)
    wing=Wing(packet,willing)
    q=np.asarray(body['q'][0]);ref=np.asarray(body['ref'][0]);view=np.asarray(body['rgb'][0]);plain=np.zeros(6)
    assert wing.command(8,view,q,ref,plain,player()) is None
    assert wing.command(9,view,q+4,ref,plain,player(True)) is None
    g=wing.command(10,view,q,ref,plain,player(True))
    assert g is not None and wing.opened==10
    assert np.all(abs(g[:5]-ref[:5])<=4.16/60+1e-12) and g[-1]==ref[-1]


def test_reserve_hands_off_at_the_existing_receive_window():
    packet=rp._read(BANK);body=packet['trajectories'][1]
    willing=SimpleNamespace(ik=SimpleNamespace(low=np.full(5,-2.0),high=np.full(5,2.0)),route=lambda *a,**k:True)
    wing=Wing(packet,willing)
    q=np.asarray(body['q'][0]);ref=np.asarray(body['ref'][0]);view=np.asarray(body['rgb'][0]);plain=np.zeros(6)
    assert wing.command(10,view,q,ref,plain,player(True)) is not None
    view=view.copy();view[1]=-.51;view[4]=-3.5
    assert wing.command(11,view,q,ref,plain,player(True)) is None
    assert wing.ended and wing.logs[-1]['reason']=='ordinary stock window'


def test_checked_scout_does_not_move_without_a_strong_early_vote():
    packet=rp._read(BANK);body=packet['trajectories'][0]
    willing=SimpleNamespace(ik=SimpleNamespace(low=np.full(5,-2.0),high=np.full(5,2.0)),route=lambda *a,**k:True)
    wing=CheckedWing(packet,willing)
    q=np.asarray(body['q'][0]);ref=np.asarray(body['ref'][0]);view=np.asarray(body['rgb'][0]);plain=np.zeros(6)
    p=player(True);p.ready.predictor=SimpleNamespace(estimates=[],future=None)
    assert wing.command(10,view,q,ref,plain,p) is None
    assert wing.opened==10 and not wing.permission
    distribution=[-.07,1.,-.3,-3.7,.31,-.11,1.01,.11,.17]
    view=view.copy();view[1]=-.42
    p.ready.predictor.future=distribution
    p.ready.predictor.estimates.append(dict(tick=22,gate=-.45,distribution=distribution.copy()))
    assert wing.command(22,view,q,ref,plain,p) is None
    distribution[-1]=.33
    p.ready.predictor.estimates.append(dict(tick=30,gate=0.,distribution=distribution.copy()))
    g=wing.command(30,view,q,ref,plain,p)
    assert wing.permission and g is not None
