import numpy as np

from scripts.experimental.tt24_planar import PlanarBounce
from scripts.experimental.tt24_planar_learning import silhouettes


def _drop(stage, speed):
    arena = PlanarBounce(stage)
    arena.reset(3992)
    # Place the ball immediately above the tested face for a physics unit test.
    arena._x = arena._px
    arena._y = arena._py + arena.radius + arena.paddle_half_height + .02
    arena._vx, arena._vy = 0., -1.6
    _, reward, done = arena.step(np.array((0., speed)))
    while not done:
        _, step_reward, done = arena.step(np.zeros(2))
        reward += step_reward
    return reward, arena.judge()


def test_damped_surface_requires_positive_power_for_verified_rebound():
    stationary, passive = _drop('damped', 0.)
    active, powered = _drop('damped', 1.)
    assert stationary == passive['rebounds'] == 0.
    assert active == powered['rebounds'] >= 1.
    assert powered['events'][0]['paddle_rate'] > 0.
    assert powered['events'][0]['apex_time'] > powered['events'][0]['impact_time']


def test_optical_roles_recover_actual_foreground_components():
    arena = PlanarBounce('lateral', presentation='varied')
    image = arena.reset(3918)
    assert image.dtype == np.uint8 and image.shape == (arena.height, arena.width, 3)
    actor, ball, span = silhouettes(image)
    np.testing.assert_allclose(actor, arena._pixel(arena._px, arena._py), atol=1.)
    np.testing.assert_allclose(ball, arena._pixel(arena._x, arena._y), atol=1.)
    assert 6 <= span <= 12


def test_replaying_actions_reproduces_physics_and_images():
    arena = PlanarBounce('energy', presentation='varied')
    other = PlanarBounce('energy', presentation='varied')
    np.testing.assert_array_equal(arena.reset(731), other.reset(731))
    steps = 0
    while True:
        command = np.array((.31, -.23)) if steps % 2 == 0 else np.array((-.31, .23))
        left, reward, done = arena.step(command)
        right, compared_reward, compared_done = other.step(command)
        np.testing.assert_array_equal(left, right)
        assert (reward, done) == (compared_reward, compared_done)
        steps += 1
        if done:
            break
    assert arena.judge() == other.judge()
