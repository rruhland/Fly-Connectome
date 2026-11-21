import numpy as np
import pytest

from scripts.experimental.tt3_arena import TT3Arena
from scripts.experimental.tt4_camera import (ImageDetection, PerspectiveTracker, TT3PerspectiveArena,
                                             cameras, triangulate)


def test_first_person_pair_is_upright_and_calibrated():
    rig = cameras('first_person', 96, 72, baseline=.064)
    assert len(rig) == 2
    left, right = rig
    assert left.eye[1] < -1.37 and right.eye[1] == left.eye[1]
    assert right.eye[0] - left.eye[0] == pytest.approx(.064)
    np.testing.assert_allclose(left.basis, right.basis)
    assert left.project((0, 0, .4))[1] < left.project((0, 0, .1))[1]
    assert left.project((.2, 0, .1))[0] > left.project((0, 0, .1))[0]
    assert len(cameras('mono')) == 1


@pytest.mark.parametrize('view', ('first_person', 'overhead_side'))
def test_triangulation_round_trip(view):
    rig = cameras(view)
    for point in ((-.40, -1.46, .13), (.1, -.12, .58), (.50, 1.20, .22)):
        pixels = [camera.project(point)[:2] for camera in rig]
        assert triangulate(rig, *pixels) == pytest.approx(point, abs=1e-9)


def test_first_person_observation_preserves_tt3_physics():
    original = TT3Arena(25, view_seed=7)
    pov = TT3PerspectiveArena(25, view_seed=7)
    a, b = pov.frame()
    assert a.shape == b.shape == (72, 96, 3)
    assert a.dtype == b.dtype == np.uint8
    assert not np.array_equal(a, b)
    assert pov.ball == original.ball
    for command in ((0., 0., 0.), (.3, -.3, .1), (-.8, .2, -.9)):
        _, a_reward, a_done, a_events = original.step(command, render=False)
        frames, b_reward, b_done, b_events = pov.step(command)
        assert (a_reward, a_done, a_events) == (b_reward, b_done, b_events)
        assert pov.ball == original.ball and pov.paddle == original.paddle
        assert all(frame.shape == (72, 96, 3) for frame in frames)


def test_tracker_pairs_anonymous_pixels_and_resets_detectors():
    rig = cameras()
    points = ((.1, -.95, .39), (-.1, .20, .32))
    made = []

    def detector_factory(camera):
        made.append(camera)

        def detect(frame):
            assert frame.shape == (camera.height, camera.width, 3)
            indices = (0, 1) if camera is rig[0] else (1, 0)
            appearances = ((1., 0., 0.), (0., 1., 0.))
            return [ImageDetection(camera.project(points[i])[:2], appearances[i], 3.) for i in indices]

        return detect

    tracker = PerspectiveTracker(rig, coordinates='world', detector_factory=detector_factory)
    images = np.zeros((2, 72, 96, 3), dtype=np.uint8)
    result = tracker.update(images)
    assert len(result) == 2
    for track, point in zip(result, points):
        assert track.observed and track.age == 1
        assert track.position == pytest.approx(point, abs=1e-9)
        assert tracker.support[track.id] == pytest.approx((3., 3.))
    tracker.reset()
    assert made == [*rig, *rig]
    assert tracker.update(images) == result


def test_default_tracker_reads_real_tt3_rgb_without_identity_masks():
    arena = TT3PerspectiveArena(15, width=192, height=144)
    tracker = PerspectiveTracker(arena.cameras, coordinates='world')
    errors = []
    frame = arena.frame()
    for _ in range(22):
        tracks = tracker.update(frame)
        errors.append(min((np.linalg.norm(np.subtract(track.position, arena.ball))
                           for track in tracks if track.observed), default=3.))
        frame, _, done, _ = arena.step((0., 0., 0.))
        if done:
            break
    assert np.median(errors) < .05
