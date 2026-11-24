import numpy as np

from scripts.experimental.tt38_joint_mpc import HORIZON, Fragments, cost, query, structured, tail


def test_warm_transport_is_bounded_on_motor_coordinates():
    seq = np.zeros((HORIZON, 6))
    shifted = structured(seq, np.ones(10))
    assert np.all(shifted[:, 5] == 0.)
    assert np.linalg.norm(shifted[0]) < np.linalg.norm(shifted[-1])
    np.testing.assert_equal(tail(shifted[:2], HORIZON)[-1], shifted[1])


def test_observed_actuator_deflection_is_encoder_relative():
    q = np.arange(6.)
    z = query(np.zeros(6), q, q)
    assert np.all(z[11:] == 0.)
    assert np.all(query(np.zeros(6), q, q + 1.)[11:] > 0.)


def test_nominal_legal_ball_preferred_to_clean_looking_miss():
    score = np.full((2, 13), np.nan)
    score[0] = [1, .12, .62, .10, .71, .06, .11, 1.30, 1.05, 3.5, 3.0, 4.0, .025]
    f = np.zeros((2, 9))
    f[1, 0] = .025
    loss, legal = cost(score, f)
    assert list(legal) == [True, False]
    assert np.all(np.isfinite(loss))
    assert loss[0] < loss[1]


def test_credit_requires_recent_visible_approach_and_retains_only_short_ref_commands():
    memory = Fragments.__new__(Fragments)
    memory.online = []
    rows = [(k, np.array([.1, -1.0, 1., 0., -3., .2]) if 0 <= k < 23 else None,
             np.zeros(6), np.zeros(6), np.ones(6) * k) for k in range(60)]
    assert memory.keep(rows[23:], "none") == 0
    assert memory.keep(rows, "physical-referee") > 0
    assert all(r[1].shape == (HORIZON, 6) and r[2] == "physical-referee" for r in memory.online)
