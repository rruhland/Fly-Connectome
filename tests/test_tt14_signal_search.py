import numpy as np

from scripts.experimental.tt14_signal_search import SearchPlayer, SignalArena, marker
from scripts.experimental.tt14_signal_change import ChangePlayer


def test_hidden_source_does_not_change_arm_image():
    # A minimal view with a static paddle uses the real frame implementation.
    class View(SignalArena):
        @property
        def paddle(self):
            return np.array([.1, .13, -.05])

    arena = object.__new__(View)
    arena.cursor_axes = np.eye(3)
    arena.origin = np.zeros(3)
    arena.target = np.array([.1,.13,-.05])
    coincident = arena.frame()
    arena.target = np.array([-.22, .27, .14])
    separate = arena.frame()
    np.testing.assert_array_equal(coincident, separate)
    np.testing.assert_allclose(marker(separate), arena.paddle)
    assert np.count_nonzero(separate[...,1:]) == 0


def test_reward_change_resets_search_and_keeps_the_same_motor_learner():
    # A sustained shift at a stationary tool should trigger relearning once.
    rgb = np.zeros((2,192,192,3),dtype=np.uint8)
    rgb[:,95:98,95:98,0] = 240
    player = ChangePlayer(np.eye(3),detect=True,probe=False)
    baseline = SearchPlayer(np.eye(3),probe=False)
    original_motor = player.motor
    for _ in range(65):
        np.testing.assert_array_equal(player.act(rgb,1.),baseline.act(rgb,1.))
    for _ in range(20):
        player.act(rgb,2.)
    assert len(player.switches) == 1
    assert 65 <= player.switches[0] < 70
    assert player.motor is original_motor
    assert player.motor.tick == 85
