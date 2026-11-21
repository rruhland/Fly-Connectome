"""Check that shaded RGB tracking uses only observable appearance and motion."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.experimental.tt4_camera import cameras
from scripts.experimental.tt4_native_rgb import NativeRegions, native_tracker
from scripts.experimental.tt4_native_temporal import ChangedRegions, modal_image


def shaded_scene(width, height, x, palette, exposure=1.):
    rows, cols = np.indices((height, width))
    surface = np.ones((height, width, 3)) * palette[0]
    regions = ((x, 19, 4), (45, 30, 9))
    for i, (center_x, center_y, radius) in enumerate(regions, 1):
        distance = np.hypot(cols - center_x, rows - center_y) / radius
        shaded = (.68 + .28 * np.sqrt(np.maximum(0, 1 - distance**2)))
        inside = distance <= 1
        surface[inside] = shaded[inside, None] * palette[i]
    return np.clip(surface * exposure, 0, 255).astype(np.uint8)


@pytest.mark.parametrize('palette', [((31, 41, 52), (225, 223, 219), (187, 49, 69)),
                                     ((76, 49, 80), (43, 103, 222), (218, 182, 40))])
def test_separates_moving_and_stationary_shaded_regions_after_recoloring(palette):
    detector = NativeRegions(SimpleNamespace(width=72, height=56))
    for x in (15, 19, 25, 30):
        detections = detector(shaded_scene(72, 56, x, palette))
        for position in ((x, 19), (45, 30)):
            region = min(detections, key=lambda item: np.linalg.norm(np.subtract(item.position, position)))
            assert np.linalg.norm(np.subtract(region.position, position)) < 1
            assert region.area > 25
        assert all(np.linalg.norm(np.subtract(item.position, (x - 6, 19))) > 2 for item in detections)


def test_cross_camera_lightness_change_preserves_region_appearance():
    camera = SimpleNamespace(width=72, height=56)
    palette = ((35, 42, 52), (220, 191, 50), (80, 113, 198))
    left = NativeRegions(camera)(shaded_scene(72, 56, 24, palette))
    right = NativeRegions(camera)(shaded_scene(72, 56, 24, palette, exposure=.71))
    for position in ((24, 19), (45, 30)):
        a, b = [min(regions, key=lambda item: np.linalg.norm(np.subtract(item.position, position)))
                for regions in (left, right)]
        assert np.linalg.norm(np.subtract(a.position, b.position)) < 1
        assert np.linalg.norm(np.subtract(a.appearance, b.appearance)) < .10


def test_detector_rejects_uncalibrated_image():
    detector = NativeRegions(SimpleNamespace(width=72, height=56))
    with pytest.raises(ValueError, match='calibrated resolution'):
        detector(np.zeros((56, 72, 4), dtype=np.uint8))


def test_blank_frames_provide_no_native_tracks():
    tracker = native_tracker(cameras('first_person', 72, 56))
    assert tracker.update(np.zeros((2, 56, 72, 3), dtype=np.uint8)) == ()
    assert tracker.support == {}


def test_probe_reference_suppresses_stationary_rgb_without_object_masks():
    palette = ((35, 42, 52), (223, 218, 211), (80, 113, 198))
    probes = [shaded_scene(72, 56, position, palette) for position in (8, 18, 28, 38, 48)]
    baseline = modal_image(np.array(probes)[:, None])[0]
    detections = ChangedRegions(baseline)(shaded_scene(72, 56, 60, palette))
    assert len(detections) == 1
    assert np.linalg.norm(np.subtract(detections[0].position, (60, 19))) < 1


def test_shaded_stereo_recovers_motion_in_a_calibrated_track():
    rig = cameras('first_person', 128, 96, baseline=.27)
    tracker = native_tracker(rig)
    y, x = np.indices((96, 128))
    nearest = []
    for step in range(7):
        point = np.array((.15 + step * .012, -.9 - step * .05, .40))
        images = []
        for camera, light in zip(rig, (1., .82)):
            u, v, _ = camera.project(point)
            radius = np.hypot(x - u, y - v) / 4.8
            frame = np.full((96, 128, 3), (31, 40, 55), dtype=np.uint8)
            shade = light * (.65 + .35 * np.sqrt(np.maximum(0., 1. - radius**2)))
            inside = radius <= 1.
            frame[inside] = shade[inside, None] * (81, 214, 116)
            images.append(frame)
        tracks = [track for track in tracker.update(images) if track.observed]
        track = min(tracks, key=lambda item: np.linalg.norm(np.subtract(item.position, point)))
        assert np.linalg.norm(np.subtract(track.position, point)) < .06
        nearest.append(track)
    assert len({track.id for track in nearest}) == 1
    assert nearest[-1].velocity[1] < -.02
    tracker.reset()
    assert tracker.tick == -1 and all(detector.previous is None for detector in tracker.detectors)
