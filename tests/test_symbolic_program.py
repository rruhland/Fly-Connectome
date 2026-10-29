"""Policy invariants and actual compositional behavior, without native cameras."""
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
from pong_symbolic_program import (Program, MAX_NODES, N_TERMINALS, ZERO,
    OPS, CONSTANTS, save_program, load_program)


SCENE = np.array([[.12, -.3, .25, .5, 1, 1], [-.2, .35, -.4, .1, 1, 1],
    [.3, -.4, 0, 0, 0, 0]], np.float64)


def custom(nodes=(), outputs=None, pools=(0, 0, 0), gain=1.):
    dormant = list(Program.zero().nodes)
    dormant[:len(nodes)] = nodes
    return Program(tuple(dormant), outputs or (N_TERMINALS, ZERO, ZERO), pools, gain)


def test_zero_and_before_hit_are_bitwise_baseline_for_every_tie_and_sign():
    rng = np.random.default_rng(71)
    programs = [Program.zero(), *(Program.random(rng) for _ in range(20))]
    for q in ([0., 0., 0.], [1., 1., 0.], [-2., -1., -3.], [1e-12, 0., 0.]):
        q = np.array(q)
        for program in programs:
            action, scores = program.choose(SCENE, q, 2, False)
            assert action == np.argmax(q)
            np.testing.assert_array_equal(scores, q)
        for earned in (False, True):
            action, scores = Program.zero().choose(SCENE, q, 2, earned)
            np.testing.assert_array_equal(scores, q)
            assert action == np.argmax(q)


def test_learnable_composition_uses_all_pairs_and_permutation_invariant_reduction():
    # Generic arithmetic: square the difference between the two first coordinates.
    nodes = [(OPS.index('sub'), 0, 6, ZERO),
        (OPS.index('mul'), N_TERMINALS, N_TERMINALS, ZERO)]
    program = custom(nodes, (N_TERMINALS + 1, ZERO, ZERO))
    expected = np.mean((2 * SCENE[:, 0, None] - 2 * SCENE[None, :, 0]) ** 2)
    assert program.residual(SCENE, 1)[0] == pytest.approx(expected)
    q = np.array([-.4, .1, 0.])
    action, scores = program.choose(SCENE, q, 1, True)
    for order in ([2, 0, 1], [1, 2, 0], [2, 1, 0]):
        other_action, other_scores = program.choose(SCENE[order], q, 1, True)
        assert other_action == action
        np.testing.assert_array_equal(other_scores, scores)


def test_division_condition_and_clipping_are_finite_at_zero_and_extremes():
    nodes = [(OPS.index('div'), 2, ZERO, ZERO),
        (OPS.index('lt'), 0, 6, ZERO),
        (OPS.index('select'), N_TERMINALS + 1, N_TERMINALS, 2)]
    program = custom(nodes, (N_TERMINALS + 2, N_TERMINALS, ZERO), (2, 1, 0), 4.)
    scene = SCENE.copy(); scene[:, 2] = [0., 1e300, -1e300]
    residual = program.residual(scene, 4)
    assert np.isfinite(residual).all() and np.max(np.abs(residual)) <= 1
    action, q = program.choose(scene, [-1., 2., .5], 4, True)
    assert np.isfinite(q).all() and action in (0, 1, 2)
    assert q[0] == pytest.approx(-1 + 12 * residual[0])


def test_empty_or_capacity_overflow_preserves_baseline():
    program = Program.random(np.random.default_rng(18))
    for scene in (np.empty((0, 6)), np.tile(SCENE, (3, 1))):
        action, scores = program.choose(scene, [1., 2., 3.], 0, True)
        assert action == 2
        np.testing.assert_array_equal(scores, [1., 2., 3.])


def test_serialization_hash_ignores_inactive_genes_but_rejects_config_change(tmp_path):
    program = custom([(OPS.index('abs'), 0, ZERO, ZERO)])
    nodes = list(program.nodes)
    nodes[-1] = (OPS.index('add'), 0, 1, 2)
    neutral = Program(nodes, program.outputs, program.pools, program.gain)
    assert neutral.active_hash() == program.active_hash()
    path = tmp_path / 'program.json'; save_program(path, program)
    restored = load_program(path)
    assert restored.frozen_parameters() == program.frozen_parameters()
    np.testing.assert_array_equal(restored.choose(SCENE, [.5, -.1, 0.], 3, True)[1],
        program.choose(SCENE, [.5, -.1, 0.], 3, True)[1])
    payload = program.to_dict(); payload['config']['division_floor'] = .1
    with pytest.raises(ValueError, match='config'):
        Program.from_dict(payload)


def test_structural_mutations_are_reproducible_and_every_active_edge_is_legal():
    first, second = Program.zero(), Program.zero()
    a, b = np.random.default_rng(112), np.random.default_rng(112)
    for _ in range(200):
        old = first
        first, second = first.mutate(a), second.mutate(b)
        assert first.to_dict() == second.to_dict()
        assert first.active_hash() != old.active_hash()
        assert first.nodes != old.nodes or first.outputs != old.outputs or first.pools != old.pools
        assert first.active_node_count <= MAX_NODES
        assert first.active_node_count == len(first.active_indices())
        action, q = first.choose(SCENE, [.1, .2, -.1], 4, True)
        assert np.isfinite(q).all() and 0 <= action < 3


def test_signature_is_action_dedup_only_and_contains_no_trajectory_rewards():
    observations = [(SCENE, np.array([0., 1., 0.]), phase, hit)
        for phase in range(5) for hit in (False, True)]
    program = Program.zero()
    assert program.action_signature(observations) == program.action_signature(observations)
    changed = [(scene, [2., 0., 0.], phase, hit) for scene, _, phase, hit in observations]
    assert program.action_signature(changed) != program.action_signature(observations)


@pytest.mark.parametrize('scene', [np.array([[0., 1.]]), np.full((1, 6), np.nan),
    np.array([[0., 0., 0., 0., .5, 1.]])])
def test_rejects_invalid_legal_vision(scene):
    with pytest.raises(ValueError):
        Program.random(np.random.default_rng(0)).choose(scene, [0., 1., 0.], 0, True)
