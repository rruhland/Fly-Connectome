"""Verify that the extra rotary axes physically move the racket."""

import numpy as np
import pytest

mujoco = pytest.importorskip('mujoco')

from scripts.experimental.tt6_body_arena import BODY_AXES, TT6Arena


def test_rotary_receiver_has_six_active_hinges_and_two_independent_extra_directions():
    arena = TT6Arena(body='rotary', venue='hall')
    try:
        axes, passive, _ = BODY_AXES['rotary']
        assert arena.action_size == len(axes) == 6
        assert len(passive) == 2
        for axis in axes:
            name = f'learner_{axis}'
            joint = arena.model.joint(name)
            actuator = arena.model.actuator(name)
            assert joint.type[0] == mujoco.mjtJoint.mjJNT_HINGE
            assert actuator.trnid[0] == joint.id
        baseline = arena.data.qpos.copy()
        racket = arena.model.geom('learner_racket').id
        center = arena.data.geom_xpos[racket].copy()
        normal = arena.data.geom_xmat[racket].reshape(3, 3)[:, 2].copy()
        displacements = []
        for axis in ('wrist_turn', 'wrist_roll'):
            coordinate = arena.model.joint(f'learner_{axis}').qposadr[0]
            arena.data.qpos[:] = baseline
            arena.data.qpos[coordinate] += .30
            mujoco.mj_forward(arena.model, arena.data)
            movement = arena.data.geom_xpos[racket] - center
            rotation = arena.data.geom_xmat[racket].reshape(3, 3)[:, 2] - normal
            assert np.linalg.norm(movement) > .01
            assert np.linalg.norm(rotation) > .08
            displacements.append(np.r_[movement, rotation])
        assert np.linalg.matrix_rank(displacements) == 2
    finally:
        arena.close()
