from collections import Counter, deque

import numpy as np

from scripts.experimental import tt38_joint_mpc as mpc
from scripts.experimental import tt43_motor_modes as modes
from scripts.experimental.tt43_motor_modes_practice import PracticingMotor


def context(x0, y=-1.06):
    return np.array([x0, y, 1.037, .13, -3.42, -.15])


def test_phase_encoding_round_trip_for_affine_references():
    start = np.array([.03, -1.70, -1.45, 1.4, -.4, .6])
    slope = np.array([.008, 0., -.004, .006, 0., 0.])
    seq = start + (modes.FRAMES[:, None] + 1.) * slope
    phase = np.array(8.)
    knots = modes.encode(phase, seq, start)
    decoded = modes.decode(knots, phase)
    i = np.arange(1, 18)
    assert np.allclose(decoded[i], seq[i])


def test_local_affine_can_predict_a_new_command_without_backprop():
    delta = np.array([[-.50], [-.24], [.1], [.42], [.8]])
    intercept = np.zeros((11, 6))
    intercept[:, 0] = .245
    outcome = intercept[None] + delta[:, None] * np.full((1, 11, 6), .44)
    w = np.ones(len(delta))
    fit, mean, residual = modes.local_fit(delta, outcome, w)
    assert np.linalg.norm(fit - intercept) < np.linalg.norm(mean - intercept)
    assert min(np.linalg.norm(fit - y) for y in outcome) > .02
    assert residual.shape == outcome.shape


def test_one_reward_credit_per_stroke_and_distinct_mode_candidates():
    q = np.array([.17, -1.72, -1.48, 1.36, -.67, .6])
    fit = []
    target = q.copy()
    for shift in np.linspace(-.08, .08, 14):
        for group in (0, 1):
            state = mpc.query(context(shift), q, q)
            g = np.broadcast_to(q, (mpc.HORIZON, 6)).copy()
            g[:, 0] = -.38 + 2.2 * shift
            g[:, 1] = -1.72 + .30 * group
            g[:, 2] = -1.48 - .57 * group
            g[:, 3] += np.linspace(0., -.23 + .19 * group, mpc.HORIZON)
            fit.append((state, g, (str(shift), group)))
    readout = modes.Readout(fit)
    key, phase, proposals = readout.proposals(context(.006), q, q, deque())
    assert len(proposals) >= 16
    assert np.ptp(np.asarray(proposals)[:, -1, 2]) > .31
    assert proposals[0].shape == (mpc.HORIZON, 6)
    assert all(not np.array_equal(proposals[0], item[1]) for item in fit)
    assert not readout.penalty(key, phase, proposals).any()
    shape = np.broadcast_to(proposals[0][0], (4, 6))
    readout.misses.append((key, shape))
    assert readout.penalty(key, phase, proposals).max() > .3
    assert not readout.penalty(key + 20, phase, proposals).any()
    assert np.all(q == target)


def test_post_miss_track_does_not_displace_reachable_negative_action_history():
    motor = PracticingMotor.__new__(PracticingMotor)
    motor.diag = Counter()
    motor.rescue = True
    motor.readout = modes.Readout.__new__(modes.Readout)
    motor.readout.misses = deque(maxlen=128)
    q = np.array([.17, -1.72, -1.48, 1.36, -.67, .6])
    history = [(i, context(0., -.3 - .06 * i), q, q, q + i * .002) for i in range(38)]
    assert motor.readout.miss(history) == 0
    motor.missed(history)
    assert len(motor.readout.misses) == motor.diag["mode_miss_samples"] > 0
    assert all(shape.shape == (4, 6) for _, shape in motor.readout.misses)
