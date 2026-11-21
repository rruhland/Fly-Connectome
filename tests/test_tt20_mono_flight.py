"""Check metric observability and a tabletop bounce from one pixel at a time."""

import numpy as np

from scripts.experimental.tt18_view import Blob, CameraFit
from scripts.experimental.tt19_flight import FlightLaw
from scripts.experimental.tt20_mono_flight import MonoFlight


def camera(tick):
    rotation = np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
    eye = np.array((-.24 + .005 * tick, -1.42, .37))
    return CameraFit(eye, rotation, 265., 480, 360)


def sighting(view, position):
    pixel = view.project(position)[:2]
    return Blob(pixel, pixel[None, :], np.ones(1))


def test_gravity_recovers_metric_depth_without_apparent_size():
    law = FlightLaw(drag=0.)
    arc = MonoFlight(law=law)
    start, velocity = np.array((-.04, .50, .14)), np.array((.009, -.065, .059))
    gravity = np.array((0., 0., -law.gravity))
    for tick in range(13):
        position = start + velocity * tick + .5 * gravity * tick**2
        estimate = arc.observe(tick, camera(tick), sighting(camera(tick), position))
        if tick < 5:
            assert estimate is None
    np.testing.assert_allclose(estimate[0], position, atol=.0001)
    np.testing.assert_allclose(arc.velocity, velocity + tick * gravity, atol=.0001)
    assert arc.observe(tick + 1, None, None) is None
    assert arc.velocity is None
    assert arc.observe(tick + 5, camera(tick), sighting(camera(tick), position), position) is None


def test_piecewise_arc_recovers_velocity_after_table_contact():
    law = FlightLaw(drag=0.)
    arc = MonoFlight(law=law)
    gravity = np.array((0., 0., -law.gravity))
    contact = np.array((.05, -.39, .02))
    before = np.array((.008, -.068, -.050))
    after = before * [law.tangent, law.tangent, -law.restitution]
    for tick in range(12):
        # This sub-frame contact lies on the estimator's search grid.
        elapsed = tick - 7.3
        velocity = before if elapsed < 0 else after
        position = contact + velocity * elapsed + .5 * gravity * elapsed**2
        view = camera(tick)
        pixel = sighting(view, position)
        biased_range = view.eye + 1.16 * (position - view.eye)
        estimate = arc.observe(tick, view, pixel, biased_range, size_sigma=.15)
    assert arc.source == 'ballistic_bounce'
    assert arc.bounced
    np.testing.assert_allclose(estimate[0], position, atol=.014)
    np.testing.assert_allclose(arc.velocity, after + elapsed * gravity, atol=.003)
