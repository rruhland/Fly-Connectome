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


def test_wrist_pulse_keeps_zero_motion_as_exact_control_and_lifts_late():
    from scripts.experimental.tt40_low_pulse import MOVES, pulse
    from scripts.experimental.tt37_motor_clinic import schedules

    q, motion, action = np.zeros(5), np.ones(5), np.zeros(10)
    home = np.zeros(6)
    args = q, motion, action, .13, 1/60, home
    control = pulse(*args, MOVES[0])
    np.testing.assert_array_equal(control, schedules(q, motion, .13, action[None], 1/60, home, 22)[0])
    corrected = pulse(*args, MOVES[5])
    assert np.all(corrected[:,3] < control[:,3])
    assert corrected[0,1] == control[0,1]
    assert corrected[-1,1] > control[-1,1]


def test_physical_learning_does_not_promote_unchanged_or_nonlegal_trials():
    from scripts.experimental.tt40_low_pulse import Pulse

    no = dict(rows=[dict(trials=[dict(index=0,legal=False),dict(index=1,legal=False)])])
    assert not Pulse(no).rank
    yes = dict(rows=[dict(trials=[dict(index=0,legal=False),dict(index=4,legal=True)])])
    assert Pulse(yes).rank == [4]
