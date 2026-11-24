import numpy as np

from scripts.experimental.tt29_vision import flight_terms
from scripts.experimental.tt40_free_arc import DECAY, GRAVITY, GROUND, NET, launchers, rebalance


def test_inverse_landing_round_trip_includes_legal_clearance_for_several_arcs():
    p = np.array([-.17,-1.61,.956])
    vel, arc = launchers(p)
    assert len(vel)>12
    assert len(np.unique(arc[:,1]))>=3
    t, g = flight_terms(arc[:,2],DECAY)
    predicted = p+t[:,None]*vel - np.c_[np.zeros((len(g),2)),GRAVITY*g]
    np.testing.assert_allclose(predicted[:,:2],arc[:,:2],atol=1e-11)
    np.testing.assert_allclose(predicted[:,2],GROUND,atol=1e-11)
    assert np.all(arc[:,3]>NET)


def test_achieved_face_can_choose_a_real_normal_speed_or_reject_bad_geometry():
    p = np.array([0.,-1.61,.955])
    vin = np.array([0.,-3.55,-.68])
    normal = np.array([0.,.896,.444])
    normal /= np.linalg.norm(normal)
    good = rebalance(p,vin,normal,budget=1.40)
    assert good is not None
    s, arc, outgoing, _ = good
    assert -.18 <= s <= 1.4
    assert .17 < arc[1] < 1.20
    assert arc[3] > NET
    assert outgoing[1] > 2.35
    assert rebalance(p,vin,np.array([0.,1.,0.])) is None


def test_rate_parameterization_changes_commands_that_absolute_edits_lose_to_slew():
    from scripts.experimental.tt40_free_arc import targets_and_rates

    ready, step, lo, hi = np.zeros(6), np.full(6,.10), np.full(6,-1.), np.ones(6)
    old = np.ones((1,3,6)) * .8
    _, stock = targets_and_rates(old, ready, step, lo, hi, inputs_are_rates=False)
    _, clipped = targets_and_rates(old-.12, ready, step, lo, hi, inputs_are_rates=False)
    np.testing.assert_array_equal(stock, clipped)
    pulse = stock.copy()
    pulse[:,0,2] -= .10
    planned, lowered = targets_and_rates(pulse, ready, step, lo, hi, inputs_are_rates=True)
    recomputed, executed = targets_and_rates(planned, ready, step, lo, hi, inputs_are_rates=False)
    np.testing.assert_allclose(lowered, executed)
    np.testing.assert_allclose(planned, recomputed)
    assert lowered[0,0,2] < stock[0,0,2]
