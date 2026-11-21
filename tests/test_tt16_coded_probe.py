import numpy as np

from scripts.experimental.tt12_motor_recall import predicted, rewire
from scripts.experimental.tt16_coded_probe import decode, excitation, prediction_bank
from scripts.experimental.tt16_transport_calibration import identify
from scripts.experimental.tt6_body_control import CameraBody


def test_signed_port_order_and_unknown_delay_follow_command_clock():
    rng = np.random.default_rng(821742)
    n = 4
    coefficients = rng.normal(scale=.035, size=(1+n+n*(n+1)//2, 4))
    kernels = rng.uniform(.1, .9, size=(n, 4)).cumsum(axis=1)
    model = CameraBody(coefficients, np.zeros(n), kernels/kernels[:, -1:], 0., 100, 100, True)
    tape = excitation(model)
    assert np.max(abs(tape.cumsum(axis=0))) < 3.
    assert np.max(abs(tape)) <= .7
    key = [[3, 0, 1, 2], [-1, 1, 1, -1]]
    ticks = np.arange(len(tape) + 1)
    commands = np.vstack((np.zeros(n), tape))[None]
    seen = predicted(rewire(model, *key), commands, (0, ticks))
    bank = prediction_bank(model, tape)
    np.testing.assert_allclose(bank[1][bank[0].index(key)], seen)
    assert decode(model, tape, ticks, seen)[0] == key
    delayed = seen[np.maximum(ticks-2, 0)] + [.02, -.05, .01, .09]
    delayed[3:6, 3] = np.nan
    res = identify(model, bank, ticks, delayed)
    assert res['timing'] == key
    assert res['inferred_clock'] == [2, 2]
