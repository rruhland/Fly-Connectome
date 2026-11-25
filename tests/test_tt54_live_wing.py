from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt54_live_wing import BANK, Wing, rp


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
