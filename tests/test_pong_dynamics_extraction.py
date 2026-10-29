import numpy as np

from extract_pong_dynamics_data import action_histories, encode
from pong_smdp_adaptive_forest import state_features


def test_history_uses_saved_prefix_and_never_current_or_future_action():
    h = action_histories([2, 0, 1], prefix=[0, 1])
    assert not h[0, :-2].any()
    np.testing.assert_array_equal(h[0, -2:], np.eye(3)[[0, 1]])
    np.testing.assert_array_equal(h[1, -3:], np.eye(3)[[0, 1, 2]])
    np.testing.assert_array_equal(h[2, -4:], np.eye(3)[[0, 1, 2, 0]])
    assert not action_histories([1])[0].any()


def test_encoded_geometry_matches_tail_and_distinguishes_empty_overflow():
    scene = np.array([[.2, .8, 1, 2, 1, 0], [.7, .1, 3, 4, 0, 1]])
    got, valid, tail = encode(scene, 3, True, np.ones(4))
    np.testing.assert_array_equal(got, state_features(scene, 3, True, np.ones(4)))
    assert valid and tail
    empty, valid, tail = encode([], 2, True, np.ones(4))
    assert valid and not tail and not empty[:28].any() and empty[28] == 1 and empty[31] == 1
    _, valid, tail = encode(np.tile(scene[:1], (5, 1)), 2, True, np.ones(4))
    assert not valid and not tail
