"""Exercise the table player's view-loss behavior without advancing physics."""
from colorsys import hsv_to_rgb

import numpy as np

from scripts.experimental.tt18_view import CameraFit, OBJECT_HUES
from scripts.experimental.tt18_viewplay import ViewTransport, camera_state


def test_view_loss_clears_recent_metric_tracks_and_frozen_registry_skips_new_ids():
    cameras = {key: CameraFit(np.array([x, -1., -3.]), np.eye(3), 100., 96, 64)
               for key, x in [('left', -.5), ('right', .5)]}
    transport = ViewTransport(False, camera_state(cameras))
    tracker = transport.tracker()
    observations = {}
    for stream, camera in cameras.items():
        rgb = np.zeros((camera.height, camera.width, 3), dtype=np.uint8)
        for role, center in [('ball', (.06, -1.2, .27)), ('blade', (.15, -1.5, .2))]:
            x, y = np.rint(camera.project(center)[:2]).astype(int)
            rgb[y-1:y+2, x-1:x+2] = np.array(hsv_to_rgb(OBJECT_HUES[role], .98, .95)) * 255
        observations[stream] = rgb
    assert len(tracker.update(observations)) == 2
    assert tracker.actor is not None and tracker.ball is not None

    assert tracker.update({'left': observations['left'], 'third': observations['right']}) == ()
    assert tracker.registration[-1]['events'] == {'left': 'frozen', 'third': 'unknown_frozen'}
    assert tracker.sightings[-1] == [1, False, False, False]
    assert tracker.actor is None and tracker.ball is None and tracker.ball_sigma is None
