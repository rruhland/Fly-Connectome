"""Check causal ray conditioning of a source-camera trajectory prior."""

import numpy as np

from scripts.experimental.tt18_view import Blob, CameraFit
from scripts.experimental.tt20_arc_carry import ArcPrior, CarryArc, _position_basis, _speed, SOURCE_FRAMES


def trajectory(offset):
    return np.array((offset, -.052 + .13 * offset, .31, -.003 + .04 * offset, -.074, .005))


def test_held_prior_uses_registered_target_pixels():
    priors = []
    for offset in np.linspace(-.11, .11, 9):
        basis, drop = _position_basis(list(SOURCE_FRAMES))
        coordinates = basis @ trajectory(offset) + drop
        priors.append(list(zip(SOURCE_FRAMES, coordinates, [(.012, .022, .009)] * len(coordinates), strict=True)))
    prior = ArcPrior.fit(priors)
    shuffled = ArcPrior.fit(priors, timing_shuffled=True)
    assert prior.arcs == shuffled.arcs == 9
    assert shuffled.covariance[1, 1] > prior.covariance[1, 1] * 10
    held = trajectory(.071)
    receiver = CarryArc(ArcPrior.restore(prior.state()))
    rotation = np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
    camera = CameraFit(np.array((-.65, -1.60, .63)), rotation, 280., 480, 360)
    for tick in range(7, 15):
        basis, drop = _position_basis(tick)
        actual = basis @ held + drop
        pixel = camera.project(actual)[:2]
        blob = Blob(pixel, pixel[None, :], np.ones(1))
        output = receiver.observe(tick, camera, blob, None, None)
        if tick < 12:
            assert output is None
    np.testing.assert_allclose(output[0], actual, atol=.008)
    np.testing.assert_allclose(output[1], _speed(held, tick), atol=.004)
    assert receiver.observe(tick + 1, camera, None, None, None) is None
