"""Check that reply continuation requires single-camera evidence of a reversal."""

from types import SimpleNamespace

import numpy as np

from scripts.experimental.tt18_view import Blob, CameraFit
from scripts.experimental.tt19_flight import FlightLaw
from scripts.experimental.tt21_reacquire import ReplyTrack


def frame(camera, position):
    if position is None:
        return SimpleNamespace(fits={'wrist': camera}, detections={'ball': {}},
                               raw_ball=None, raw_ball_sigma=None)
    pixel = camera.project(position)[:2]
    blob = Blob(pixel, pixel[None, :], np.ones(1))
    return SimpleNamespace(fits={'wrist': camera}, detections={'ball': {'wrist': blob}},
                           raw_ball=position, raw_ball_sigma=(.018, .027, .019))


def test_reply_track_bridges_a_short_gap_after_observed_reversal():
    law = FlightLaw(drag=0.)
    tracker = ReplyTrack(law, (.424, .80), coasting_frames=4)
    axes = np.array(((1., 0., 0.), (0., 0., -1.), (0., 1., 0.)))
    camera = CameraFit(np.array((-.19, -1.09, .30)), axes, 210., 480, 360)
    start = np.array((.015, .82, .22))
    speed = np.array((.003, -.083, .043))
    trajectory, _ = law.trajectory(start, speed, 20, bounds=(.424, .80))

    missing = frame(camera, None)
    assert tracker.update(0, 'wrist', missing, None, 0) is None
    for tick in range(1, 12):
        position = start + [0., (tick - 11) * .115, 0.]
        assert tracker.update(tick, 'wrist', frame(camera, position), position, 1) is None
    for tick in range(12, 21):
        position = trajectory[tick - 11]
        tracker.update(tick, 'wrist', frame(camera, position), position, 1)
    assert tracker.turn == 11
    continuation = tracker.update(21, 'wrist', missing, None, 1)
    assert continuation.observed_tick == 20
    np.testing.assert_allclose(continuation.position, trajectory[10], atol=.010)
    assert tracker.update(25, 'wrist', missing, None, 1) is None
    assert tracker.update(26, 'wrist', missing, None, 2) is None
