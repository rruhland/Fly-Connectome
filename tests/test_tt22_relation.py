"""Check anonymous pair invariance and causal score labeling."""

from collections import deque

import numpy as np

from scripts.experimental.tt22_relation import _Track, descriptors, targets


def track(position, velocity, size):
    position, velocity = np.asarray(position), np.asarray(velocity)
    return _Track(0, size, deque([(0, position - velocity), (1, position)]))


def test_all_pair_pool_ignores_slot_order_and_camera_rotation():
    objects = [track((.1, .2), (.01, -.03), .014),
               track((-.2, .3), (.015, .002), .045),
               track((.3, -.1), (-.02, .01), .028)]
    bounds = np.array([.5, .5])
    baseline = descriptors(objects, bounds)
    permutation = descriptors(objects[::-1], bounds)
    rotation = np.array(((0., -1.), (1., 0.)))
    turned = descriptors([track(rotation @ item.position, rotation @ item.velocity(), item.radius)
                          for item in objects], bounds)
    np.testing.assert_allclose(baseline[0], permutation[0], atol=1e-12)
    np.testing.assert_allclose(baseline[1], permutation[1], atol=1e-12)
    np.testing.assert_allclose(baseline[0], turned[0], atol=1e-12)
    np.testing.assert_allclose(baseline[1], turned[1], atol=1e-12)


def test_score_labels_start_one_frame_later_when_only_pre_action_frames_are_saved():
    episode = np.zeros((19, 5)), 1.
    _, source_end, source_win, indices = targets(episode, terminal_image=True)
    _, tt_end, tt_win, _ = targets(episode, terminal_image=False)
    assert indices[0] == 6
    assert source_end[0] == 1 and tt_end[0] == 0
    np.testing.assert_array_equal(source_win, tt_win)
