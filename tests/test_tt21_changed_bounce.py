"""Check the tilted collision geometry and causal optical updates."""

import numpy as np

from scripts.experimental.tt4_arena import TABLE_HEIGHT
from scripts.experimental.tt21_changed_bounce import TiltedArena, adapt_offset, crossing


def test_collision_table_and_net_are_rigid_and_arm_is_unchanged():
    level, tilted = TiltedArena(0.), TiltedArena(4.)
    try:
        np.testing.assert_equal(level.model.body('learner_base').pos, tilted.model.body('learner_base').pos)
        table = tilted.model.geom('table').id
        net = tilted.model.geom('net').id
        body = tilted.model.body('tt21_tilted_playing_surface').id
        assert int(tilted.model.geom_bodyid[table]) == int(tilted.model.geom_bodyid[net]) == body
        assert int(tilted.model.body_geomnum[body]) == 10
        normal = tilted.data.geom_xmat[table].reshape((3, 3))[:, 2]
        rad = np.deg2rad(4.)
        np.testing.assert_allclose(normal, [np.sin(rad), 0., np.cos(rad)], atol=1e-10)
        top = tilted.data.geom_xpos[table] + .025*normal
        np.testing.assert_allclose(top, [0., 0., TABLE_HEIGHT], atol=1e-10)
    finally:
        level.close()
        tilted.close()


def test_late_optical_crossing_requires_a_contiguous_incoming_arc():
    tape = [(10, (0., -.80, .15)), (12, (.12, -.96, .2))]
    estimate = crossing(tape)
    np.testing.assert_allclose([estimate['x'], estimate['tick']], [.06, 11.])
    assert crossing([tape[0], (15, tape[1][1])]) is None


def test_training_update_ignores_evaluator_truth():
    rows = [dict(cue=dict(features=[1., .10, 0.]), target=dict(x=.20), evaluator_target=dict(x=-33.)),
            dict(cue=dict(features=[1., .20, 0.]), target=dict(x=.30), evaluator_target=dict(x=100.))]
    offset, _ = adapt_offset(rows, [0., 1., 0.])
    np.testing.assert_allclose(offset, .05)
