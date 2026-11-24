import numpy as np

from scripts.experimental.tt40_low_lift import LowLift, consequence, earned_candidates, swing


def test_high_arrival_does_not_consult_the_specialist_twin_or_modify_stock():
    policy = LowLift.__new__(LowLift)
    response = policy.plan(None, None, np.array([.17, .08, 1.05, 0., -3.3, .2]),
                           None, None, None, .17)
    assert response["tape"] is None
    assert not response["note"]["active"]
    assert not response["trace"]


def test_legal_lift_cost_beats_a_low_contact():
    y = np.full((2, 13), np.nan)
    y[0] = [.1, .15, .7, .1, .75, .10, .1, 1.32, 1.06, 3.5, 0, 0, .03]
    f = np.zeros((2, 9))
    f[:, 1] = 1
    f[:, 3] = 3.3
    f[0, 4] = 4.0
    f[1, 4] = .6
    f[:, 5:8] = [[0., -1.65, .96], [0., -1.65, .96]]
    cost, legal = consequence(y, f, None)
    assert list(legal) == [True, False]
    assert cost[0] < cost[1]


def test_swing_and_earned_window_emit_six_dimensional_tapes_and_clip_time():
    goals = swing(np.zeros((2, 5)), np.ones((2, 5)), np.array([.12, .20]),
                  np.zeros((2, 11)), 1/60, n=22)
    assert goals.shape == (2, 22, 6)
    assert np.all(np.ptp(goals[:,:,0], axis=-1) <= .211)
    sample = dict(id=17, x=np.zeros(10), eta=.16, seq=goals[0], tag=(17,100))
    obs = np.array([.17, .00, .98, 0., -3.3, .2])
    assert earned_candidates([sample],obs,np.zeros(6),.17,10,omit=17) == ([], [])
    tapes, labels = earned_candidates([sample],obs,np.zeros(6),.17,10)
    assert np.asarray(tapes).shape == (32, 78, 6)
    assert labels == [(17,100)]*32
