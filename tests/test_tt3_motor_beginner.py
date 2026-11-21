"""Check the optional full-court setup and the service scoring boundary."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from tt3_arena import BALL_RADIUS, TT3Arena
from tt3_motor_beginner import PreparedTT3Arena


def test_preparation_moves_the_arm_and_shows_the_ball_before_a_legal_two_bounce_serve():
    arena = PreparedTT3Arena(617)
    initial_joints = arena.joints
    initial_images = arena.frame()
    for index in range(arena.preparation_frames - 1):
        yaw = .15 if index < 10 else -.15 if index < 20 else 0.
        frames, reward, done, info = arena.step((yaw, 0., 0.))
        assert reward == 0 and not done and not info['events']
        if index == 9:
            assert arena.joints != initial_joints
    np.testing.assert_allclose(arena.joints, initial_joints, atol=1e-8)
    assert all(not np.array_equal(a, b) for a, b in zip(initial_images, frames))
    _, reward, done, info = arena.step((0., 0., 0.), render=False)
    assert reward == 0 and not done
    serve, = info['events']
    assert serve['kind'] == 'serve' and serve['frame'] == arena.preparation_frames
    assert serve['ball'][1] > 1.35
    while not arena.done:
        arena.step((0., 0., 0.), render=False)
    serve, first, receiver, terminal = arena.events
    assert first['kind'] == 'serve_bounce' and first['location'][1] > 0.
    assert receiver['kind'] == 'bounce' and receiver['location'][1] < 0.
    assert first['location'][2] == receiver['location'][2] == BALL_RADIUS
    assert terminal['winner'] == -1
    assert arena.point_result()['legal_returns'] == 0
    assert TT3Arena(617).ball[1] < 1.
