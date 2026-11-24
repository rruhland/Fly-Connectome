"""Validate causal preparation and its direction-neutral training objective."""

import numpy as np

from scripts.experimental.tt39_common_preparation import Preload, common_pose, onset_samples
from scripts.experimental.tt39_recovery_audit import readiness


def test_preload_does_not_reserve_motor_wing_or_roll():
    home = np.zeros(6)
    states = np.zeros((30, 5))
    states[:, 0] = np.repeat([-.4, 0., .4], 10)
    states[:, 2:4] = [.23, -.55]
    states[:, 4] = -.33
    result, agreement = common_pose(states, home)
    assert np.min(agreement[:, [2, 3]]) == 1
    assert result[2] > 0.0 and result[3] < 0.0
    np.testing.assert_array_equal(result[[0, 1, 4, 5]], 0.)


def test_only_the_actual_legal_pulse_arms_the_preparation_window():
    no_ball = None
    outgoing = np.array([0, .3, 1., 0., 1.8, 0.])
    reply = np.array([0, 1.3, 1., 0., -2.9, 0.])
    target = np.ones(6)
    q = np.zeros(6)
    active = Preload(target, enabled=True)
    assert active.change(12, no_ball, q) is None
    assert active.change(13, outgoing, q) is None
    active.scored(14, q)
    np.testing.assert_array_equal(active.change(15, no_ball, q), target)
    np.testing.assert_array_equal(active.change(16, outgoing, q), target)
    assert active.change(17, reply, q) is None
    assert active.change(18, outgoing, q) is None
    assert active.trace[0]["handoff_tick"] == 17 and active.trace[0]["issued_ticks"] == 2

    none = Preload(target, enabled=False)
    none.scored(20, q)
    assert none.change(21, outgoing, q) is None


def test_onset_has_one_causal_training_observation_per_archived_strike():
    states = np.zeros((6, 11))
    states[:, 4] = -.95
    states[:, 6] = np.arange(6)
    marks = np.array([[15, -.30], [16, -.283333333], [17, -.266666667],
                      [40, -.30], [41, -.283333333], [42, -.266666667]])
    receipt = {"practice": [{"frames": 6, "card": {"id": 51}}]}
    q, release = onset_samples({"states": states, "marks": marks}, receipt)
    np.testing.assert_equal(q[:, 0], [1, 4])
    np.testing.assert_equal(release, [51, 51])


def test_readiness_measures_a_real_encoder_crossing_before_return():
    t = np.arange(9) / 60
    q = np.tile(np.linspace(.4, .0, len(t))[:, None], (1, 6))
    r = readiness(t, q, 0., t[7], np.zeros(6))
    assert r["first_neutral_s"] == t[5]
    assert r["time_far_minus_contact_s"] == t[7]
