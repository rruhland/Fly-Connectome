"""The portable motor readout must preserve the native supervisor-free policy on each frame."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
from scripts.experimental.tt24_carried_wrist import CarriedWristPolicy, commands
from scripts.experimental.tt24_donor_trial import Native


def test_portable_feedback_matches_native_across_clipping_and_repeated_commands():
    rng = np.random.default_rng(191703)
    portable = CarriedWristPolicy()
    issued = rng.uniform(-.2, .2, 5)
    reference = issued.copy()
    native = Native()
    try:
        for scale in (.001, .1, 1., 2.5):
            for observation in rng.normal(0., scale, (15, 11)):
                expected = np.empty(5)
                native.command(observation.ctypes.data, portable.weights.ctypes.data,
                               reference.ctypes.data, expected.ctypes.data)
                actual = commands(observation, portable.weights, issued)
                np.testing.assert_array_equal(actual, expected)
                np.testing.assert_array_equal(issued, reference)
    finally:
        native.working.cleanup()
