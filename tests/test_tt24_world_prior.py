from contextlib import closing

import numpy as np

from scripts.experimental.tt24_world_prior import LIMITS, WEIGHTS, WorldPrior
from scripts.experimental.tt24_world_prior_train import Worlds, data


def test_task_space_readout_is_bounded_and_rotates_with_horizontal_state():
    rng = np.random.default_rng(74)
    actor = WorldPrior(rng.normal(size=WEIGHTS))
    state = rng.normal(size=11)
    base = actor.target(state)
    rotation = np.array(((0., -1.), (1., 0.)))
    rotated = state.copy()
    for start in (0, 3, 6, 9):
        rotated[start:start + 2] = rotation @ state[start:start + 2]
    np.testing.assert_allclose(actor.target(rotated), np.r_[rotation @ base[:2], base[2], rotation @ base[3:]])
    assert np.linalg.norm(base[:2]) <= LIMITS[0]
    assert abs(base[2]) <= LIMITS[1]
    assert np.linalg.norm(base[3:]) <= LIMITS[2]


def test_planned_source_action_matches_deployed_interface():
    rng = np.random.default_rng(17)
    actor = WorldPrior(rng.normal(size=WEIGHTS))
    with closing(Worlds(workers=1)) as native:
        for state in rng.normal(size=(20, 11)):
            np.testing.assert_allclose(native.target(actor.weights, state), actor.target(state), atol=1.e-12)

        idle = native.play(np.zeros(WEIGHTS), data(86, 20))[0]
        again = native.play(np.zeros(WEIGHTS), data(86, 20))[0]
        np.testing.assert_array_equal(idle, again)
        assert np.any(idle[:, 0] > 0)
        assert np.all(idle[:, 0] <= idle[:, 3])
        assert np.all(idle[idle[:, 0] > 0, 4] >= .045)
