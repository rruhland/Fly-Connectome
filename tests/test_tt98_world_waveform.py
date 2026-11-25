import numpy as np

from scripts.experimental import tt98_world_waveform as world
from scripts.experimental.tt4_transfer_trial import _read


def test_world_gate_keeps_low_postbounce_calls_out_and_requires_observed_bounce():
    high=np.array([.08,-.92,1.15,.09,-3.30,.70])
    low=np.array([.08,-.92,.91,.09,-3.30,.70])
    assert world.gate(high,True)
    assert not world.gate(high,False)
    assert not world.gate(low,True)
    assert not world.gate(high*np.array([1,1,1,1,-1,1]),True)
    assert world.horizon_height(high)-world.horizon_height(low)>.20


def test_world_offer_projection_and_its_twist_are_in_world_axes():
    x=np.array([-.12,-1.05,1.10,.39,-3.,.20])
    p,t=world.Archive.flight(x,-1.52)
    np.testing.assert_allclose(p, x[:3]+t*x[3:]+[0,0,-4.905*t*t])
    assert .15<t<.18
    assert .058<p[0]-x[0]<.070


def test_registered_rosters_have_no_released_body_overlap():
    p=world.plan()
    train={c['id'] for c in p['practice']}
    matched={c['id'] for c in p['held']}
    q=_read(world.GAME_PLAN)
    games={c['id'] for c in q['cards']}
    assert not train&matched
    assert not games&(train|matched|set(q['excluded']))
    assert len(games)==8
    assert all(c['opening']=='random' and c['profile']=='reach' for c in q['cards'])
