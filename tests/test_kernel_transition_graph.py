"""Numerical and support checks for the offline anonymous transition graph."""
from pathlib import Path
import sys
import time

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'
sys.path.insert(0, str(SCRIPTS))
from pong_kernel_transition_graph import (KernelGraph, Neighbors, SceneMetric,
    TransitionIndex, backup, elapsed_discount, reconstruct_timing, scheduled_ticks, solve)


def test_public_scheduler_and_report_reconstruction():
    assert scheduled_ticks(np.arange(10)).tolist() == [2, 2, 3, 2, 3, 2, 2, 3, 2, 3]
    elapsed, ticks = reconstruct_timing(4, 8 / 120)
    assert ticks.tolist() == [2, 2, 3, 1]
    assert elapsed.sum() == pytest.approx(8 / 120)
    with pytest.raises(ValueError):
        reconstruct_timing(4, 6 / 120)
    with pytest.raises(ValueError):
        reconstruct_timing(4, 10 / 120)
    with pytest.raises(ValueError):
        reconstruct_timing(4, 8.1 / 120)


def test_elapsed_discount_uses_seconds_and_composes():
    a, b = elapsed_discount(np.array([2, 3]) / 120)
    assert 0 < b < a < 1
    assert a * b == pytest.approx(float(elapsed_discount(5 / 120)))
    assert float(elapsed_discount(3.2)) == pytest.approx(.5)
    with pytest.raises(ValueError):
        elapsed_discount(0.)


def test_sort_is_horizontal_and_masks_phase_history_are_exact():
    metric = SceneMetric(np.ones(4), 1.)
    scene = np.array([[.3, -.4, 0, 0, 1, 1], [-.3, .4, 0, 0, 1, 1]])
    vector, key = metric.encode(scene, 0, False)
    assert vector.reshape(4, 4)[0, 0] > 0
    assert np.array_equal(vector, metric.encode(scene[::-1], 0, False)[0])
    assert metric.encode(scene, 1, False)[1] != key
    assert metric.encode(scene, 0, True)[1] != key
    missing = scene.copy(); missing[0, 4] = 0
    assert metric.encode(missing, 0, False)[1] != key
    assert metric.encode(np.tile(scene, (3, 1)), 0, False)[1] == -1


def fixture_index():
    vectors = np.zeros((5, 16)); vectors[:, 0] = [0., .1, 0., .1, 0.]
    # Action0 has repeated observations from one start; action1 has two starts.
    return TransitionIndex(vectors, np.array([7, 7, 7, 7, 8]),
        np.array([0, 0, 1, 1, 2]), np.array([0, 0, 0, 1, 2]), .2)


def test_support_is_action_conditional_and_requires_distinct_starts():
    index = fixture_index()
    found = index.query(np.zeros((2, 16)), np.array([7, 9]))
    assert found.supported.tolist() == [[False, True, False], [False, False, False]]
    assert np.all(found.weights >= 0)
    assert np.allclose(found.weights.sum(2), found.supported)
    outside = np.ones((1, 16))
    assert not index.query(outside, np.array([7])).supported.any()


def test_overflow_rows_are_retained_but_never_supply_support():
    index = TransitionIndex(np.zeros((4, 16)), np.array([-1, -1, 7, 7]),
        np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1]), .2)
    assert len(index.vectors) == 4
    assert not index.query(np.zeros((1, 16)), np.array([-1])).supported.any()
    assert index.query(np.zeros((1, 16)), np.array([7])).supported.tolist() == [[False, True, False]]


def test_convex_backup_contraction_hit_terminal_and_unknown_boundary():
    indices = np.zeros((4, 3, 8), dtype=np.int32)
    weights = np.zeros((4, 3, 8))
    indices[:, 0, :2] = [0, 1]
    weights[:3, 0, :2] = [.25, .75]
    neighbors = Neighbors(indices, weights, weights.sum(2) > 0)
    hit = np.array([False, True, False, False])
    terminal = np.array([False, False, True, False])
    discounts = elapsed_discount(np.array([2, 3, 2, 3]) / 120)
    v = np.array([.2, .8, .1, .6]); w = np.array([.9, .1, .7, .3])
    bv = backup(v, neighbors, hit, terminal, discounts)
    bw = backup(w, neighbors, hit, terminal, discounts)
    assert np.max(abs(bv - bw)) <= discounts.max() * np.max(abs(v - w)) + 1e-14
    assert bv[1] == bw[1] == 1
    assert bv[2] == bw[2] == 0  # Actual natural terminal.
    assert bv[3] == bw[3] == 0  # Explicit model boundary; not a terminal label.
    values, report = solve(neighbors, hit, terminal, discounts, time.monotonic() + 2)
    assert report['converged'] and np.all((values >= 0) & (values <= 1))
    assert values[0] > .9


def test_serialized_query_and_opening_fallback(tmp_path):
    scene = np.array([[0., 0., 0., 0., 1., 1.]])
    metric = SceneMetric(np.ones(4), .2)
    vector, key = metric.encode(scene, 2, True)
    vectors = np.tile(vector, (4, 1))
    index = TransitionIndex(vectors, np.full(4, key), np.array([0, 0, 1, 1]),
        np.array([0, 1, 0, 1]), .2)
    graph = KernelGraph(metric, index, np.array([.2, .2, .8, .8]))
    before = graph.inspect(scene, 2, True, 0)
    assert before['action'] == 1
    assert graph.inspect(scene, 2, True, 2)['action'] == 2  # Unsupported baseline.
    assert graph.inspect(scene, 2, False, 0)['action'] == 0
    path = tmp_path / 'graph.npz'; graph.save(path)
    assert KernelGraph.load(path).inspect(scene, 2, True, 0) == before
