import importlib.util

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.experimental.tt18_view import CameraFit, LANDMARKS, ViewRegistry, _fit_camera, triangulate_views


def test_nonplanar_points_recover_metric_camera_pose_and_unknown_focal_length():
    points = np.array([mark.position for mark in LANDMARKS])
    rotation = Rotation.from_euler('xyz', (1.4, .2, -.13)).as_matrix()
    original = CameraFit(np.array((-.24, -2.52, 1.02)), rotation, 145., 256, 192)
    pixels = original.project(points)[:, :2]
    assert np.all(original.project(points)[:, 2] > 0.)
    fitted = _fit_camera(points, pixels, (192, 256), None)
    assert fitted is not None
    np.testing.assert_allclose(fitted.eye, original.eye, atol=1e-8)
    np.testing.assert_allclose(fitted.rotation, original.rotation, atol=1e-8)
    assert fitted.focal == pytest.approx(original.focal)
    points[:, 2] = 0.
    assert _fit_camera(points, original.project(points)[:, :2], (192, 256), None) is None


def test_single_view_unknown_depth_and_colocated_cameras_abstain():
    camera = CameraFit(np.array((0., 0., 0.)), np.eye(3), 170., 256, 192)
    second = CameraFit(np.array((.4, .1, 0.)), np.eye(3), 112., 256, 192)
    point = np.array((.29, -.08, 2.51))
    pixel = camera.project(point)[:2]
    np.testing.assert_allclose(camera.project(1.35 * point)[:2], pixel)
    assert triangulate_views({'a': camera}, {'a': pixel}) is None
    assert triangulate_views({'a': camera, 'b': camera}, {'a': pixel, 'b': pixel}) is None
    estimate = triangulate_views({'a': camera, 'b': second},
                                 {'a': pixel, 'b': second.project(point)[:2]})
    np.testing.assert_allclose(estimate, point, atol=1e-10)


@pytest.mark.skipif(importlib.util.find_spec('mujoco') is None, reason='requires native MuJoCo RGB')
def test_hall_registry_fits_new_views_and_refuses_unverifiable_ones():
    from scripts.experimental.tt18_view import object_pixels
    from scripts.experimental.tt18_view_trial import ORIGIN, ViewHall, hall_images, rig_schedule

    arena = ViewHall(seed=7, width=256, height=192, venue='hall', body='rotary', resize=True)
    registry = ViewRegistry()
    schedule = rig_schedule(2)
    try:
        initial = hall_images(arena, schedule['source'])
        fits = registry.update(initial)
        assert set(fits) == {'left', 'right'}
        assert registry.last_events == {'left': 'added', 'right': 'added'}
        moved = hall_images(arena, schedule['added'])
        fits = registry.update(moved)
        assert registry.last_events == {'left': 'relocated', 'right': 'relocated', 'spare': 'added'}
        pixels = object_pixels(moved)
        for role, position in (('ball', arena.ball), ('blade', arena.paddle)):
            estimate = triangulate_views(fits, pixels[role])
            assert estimate is not None
            assert np.linalg.norm(estimate - (position - ORIGIN)) < .030
        mono = {'spare': moved['spare']}
        assert set(registry.update(mono)) == {'spare'}
        assert registry.last_events == {'left': 'removed', 'right': 'removed', 'spare': 'verified'}
        assert triangulate_views(registry.active, object_pixels(mono)['ball']) is None
        assert registry.update({'spare': np.zeros_like(mono['spare'])}) == {}
        assert registry.last_events == {'spare': 'insufficient_landmarks'}
        assert set(registry.update(mono)) == {'spare'}
        assert registry.last_events == {'spare': 'verified'}
    finally:
        arena.close()
