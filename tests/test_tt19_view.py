"""Pin the measured rim geometry and permitted SO-101 camera feeds."""
from hashlib import sha256

import numpy as np
from scipy.spatial.transform import Rotation

from scripts.experimental.tt18_view import CameraFit
from scripts.experimental.tt19_view import RIM, ViewTransport, _image_rim


def test_wrist_known_rim_recovers_center_and_surface_normal():
    camera = CameraFit(np.array((.3, -.95, .12)), np.eye(3), 135., 320, 240)
    center = np.array((.34, -.91, .26))
    orientation = Rotation.from_euler('xyz', (.37, -.2, .18)).as_matrix()
    pixels = camera.project(center + RIM @ orientation.T)[:, :2]
    point, normal = _image_rim(camera, pixels)
    np.testing.assert_allclose(point, center, atol=1e-9)
    np.testing.assert_allclose(normal, orientation[:, 2], atol=1e-9)


def test_wrist_only_transport_does_not_consume_external_feeds():
    tracker = ViewTransport('wrist').tracker()
    wrist = np.full((48, 64, 3), 120, dtype=np.uint8)
    extra = np.full_like(wrist, 181)
    assert tracker.update({'wrist': wrist, 'left': extra, 'right': extra}) == ()
    expected = sha256(b'wrist')
    expected.update(wrist)
    assert tracker.digest.hexdigest() == expected.hexdigest()
    assert tracker.registration == [{'tick': 0, 'events': {'wrist': 'insufficient_landmarks'}}]
