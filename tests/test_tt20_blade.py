"""Check the optical timing fit and the bounded inverse command state."""

import numpy as np

from scripts.experimental.tt20_blade import PulseCompensator, TimingFit, fit_timing


def test_nominal_compensator_preserves_normalized_actions():
    rng = np.random.default_rng(42)
    controls = rng.uniform(-1., 1., size=(48, 6))
    correction = PulseCompensator(TimingFit.nominal())
    np.testing.assert_allclose([correction.command(row) for row in controls], controls, atol=1e-13)


def test_compensated_clipping_does_not_leave_a_drift_after_returning_to_neutral():
    fit = TimingFit(np.repeat(.87, 5), np.repeat(.36, 5))
    correction = PulseCompensator(fit)
    commands = np.repeat(((.99,) * 5 + (0.,), (-.99,) * 5 + (0.,), (0.,) * 6), 4, axis=0)
    output = np.array([correction.command(row) for row in commands])
    assert np.max(abs(output)) <= 1.
    np.testing.assert_allclose(output.sum(axis=0), 0., atol=1e-13)


def test_fit_tracks_a_known_small_timing_error_on_unfitted_frames():
    rng = np.random.default_rng(17)
    commands = rng.uniform(-.32, .32, size=(72, 5))
    bias = TimingFit(np.array((.89, 1.08, .94, .97, 1.12)), np.array((.20, .16, -.30, .28, .31)))
    nominal = TimingFit.nominal()
    projection = rng.normal(size=(5, 20))

    def simulate(tape):
        control = np.vstack((np.zeros((1, 5)), np.cumsum(tape, axis=0)))
        return (60 * control @ projection).reshape((-1, 10, 2))

    projected = simulate(nominal.model_commands(commands))
    actual = simulate(bias.model_commands(commands))
    learned, metrics = fit_timing(projected + 20., projected, actual + 20., commands, simulate)
    np.testing.assert_allclose(learned.gains, bias.gains, atol=.015)
    np.testing.assert_allclose(learned.delays, bias.delays, atol=.020)
    assert metrics['check']['corrected']['rms_px'] < .03 * metrics['check']['nominal']['rms_px']
