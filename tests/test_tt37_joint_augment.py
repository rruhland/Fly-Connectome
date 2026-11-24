import numpy as np

from scripts.experimental.tt37_joint_augment import CandidateField, MixedField, NativeCorrection


def test_sensor_correction_preserves_encoders_and_fades_outside_training():
    native = np.zeros((48, 11))
    native[:24, 4] = -3.
    native[24:, 4] = 3.
    optical = native.copy()
    optical[:, 0] = .07
    optical[:, 1] = .03
    use = [True, False, False, False, False, False]
    calibration = NativeCorrection(native, optical, use)
    q = native[[0, -1]].copy()
    q[1, 0] = 5.
    expected = q.copy()
    expected[0, 0] += .07
    np.testing.assert_allclose(calibration.predict(q), expected)


def test_native_mix_returns_exact_rgb_when_simulator_has_no_support():
    class Fixture:
        def __init__(self, gain, vector):
            self.gain = gain
            self.vector = vector

        def get(self, optical, q, *, reference=None):
            return self.vector, self.gain

    command, weight = MixedField(Fixture(.7, np.ones(6)), Fixture(0., None)).get(None, None)
    assert weight == .7
    np.testing.assert_equal(command, 1.)
    command, weight = MixedField(Fixture(1., np.ones(6)), Fixture(1., np.zeros(6))).get(None, None)
    assert weight == 1.
    np.testing.assert_allclose(command, 1. - .85 * .35)


def test_coherent_prediction_uses_previous_sent_command_to_avoid_blending_modes():
    states = np.zeros((64, 11))
    states[:, 4] = -3.
    labels = np.repeat([1., -1.], 32)[:, None] * np.ones((64, 6))
    groups = np.repeat([0, 1], 32)[:, None]
    field = CandidateField(dict(states=states, targets_now=labels, group=groups), "local", "coherent")
    for liking in (1., -1.):
        command, confidence = field.get(states[0, :6], states[0, 6:], reference=liking * np.ones(6))
        assert confidence == 1.
        np.testing.assert_allclose(command, liking)
