import numpy as np
from scipy.spatial.transform import Rotation

from scripts.experimental.tt11_remount import RemountPlayer, restore, state
from scripts.experimental.tt11_tracking import orientation


def test_direction_alignment_preserves_handedness():
    rotation = Rotation.from_rotvec([.7, -.4, 2.]).as_matrix()
    rays = np.array([[.1, .3, -.2], [-.2, .1, .2]])
    assert np.allclose(orientation((rays @ rotation.T).T @ rays), rotation)
    assert np.isclose(np.linalg.det(orientation(np.diag([1., 1., -1.]))), 1.)


def test_snapshot_keeps_independent_event_history():
    player = RemountPlayer(np.eye(3), 'transport')
    player.events.append(7)
    saved = state(player)
    player.events.append(12)
    assert restore(saved).events == [7]
