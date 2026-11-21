import numpy as np
import pytest

from scripts.experimental.tt20_motor_carry import Wiring, identify


def test_independent_sensor_responses_identify_the_signed_permutation():
    rng = np.random.default_rng(1247)
    images = []
    for _ in range(5):
        color = rng.uniform(-.6, .6, (3, 4, 5, 3))
        visible = np.ones((3, 4, 5), dtype=bool)
        images.append([(-color, visible), (color, visible)])
    wiring = Wiring((3, 1, 4, 0, 2), (-1, 1, -1, -1, 1))
    observed = [images[axis][int(sign > 0)] for axis, sign in zip(wiring.order, wiring.signs, strict=True)]
    fit, audit = identify(observed, images)
    assert fit == wiring
    assert audit[-1]['next_assignment_margin'] > .05
    for command in rng.uniform(-1., 1., (6, 6)):
        assert np.array_equal(wiring.physical(fit.inverse(command)), command)


def test_the_estimator_rejects_an_ambiguous_camera_response():
    zero = np.zeros((3, 4, 5, 3)), np.ones((3, 4, 5), dtype=bool)
    with pytest.raises(ValueError, match='unique signed assignment'):
        identify([zero] * 5, [[zero, zero] for _ in range(5)])
