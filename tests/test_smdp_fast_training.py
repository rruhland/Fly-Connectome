"""Exact query optimization parity without fitting any model."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
from pong_smdp_adaptive_forest import AdaptiveForest, STATE_DIM, option_features, sklearn_q_values, leaf_independence
from pong_smdp_fast_training import (readonly_c_forest, training_leaf_cache,
    fast_q_values, independent_counts_from_leaves)


def synthetic_query_model():
    shape = (32, 3)
    feature = np.full(shape, -2, np.int32); feature[:, 0] = 0
    threshold = np.zeros(shape)
    left = np.full(shape, -1, np.int32); right = left.copy()
    left[:, 0] = 1; right[:, 0] = 2
    values = np.zeros(shape)
    values[:, 1] = .03 + np.arange(32) / 137
    values[:, 2] = .11 + np.arange(32) / 117
    portable = AdaptiveForest(np.ones(4), feature, threshold, left, right, values,
        np.zeros((*shape, 4), np.int32))
    return readonly_c_forest(portable), portable


def test_apply_leaf_values_and_presence_are_bit_exact_without_any_fit():
    model, portable = synthetic_query_model()
    states = np.zeros((16, STATE_DIM), np.float32)
    states[:8, 0] = -1; states[8:, 0] = 1
    options = np.tile(np.arange(4), 4)
    starts = np.tile([0, 1], 8)
    F = option_features(states, options)
    leaves, presence, prediction = training_leaf_cache(model, F, options)
    distinct = leaf_independence(model, F, options, starts)
    assert np.array_equal(distinct, independent_counts_from_leaves(model, leaves, options, starts))
    assert np.array_equal(presence, distinct > 0)
    assert np.array_equal(prediction, model.predict(F))
    assert np.array_equal(prediction, portable.predict(F, options)[0])
    valid = np.ones(len(states), bool); valid[-1] = False
    reference_q, reference_support, _ = sklearn_q_values(model, distinct, states, valid)
    fast_q, fast_support = fast_q_values(model, presence, states, valid)
    assert np.array_equal(reference_q, fast_q)
    assert np.array_equal(reference_support, fast_support)
    assert not fast_support[-1].any()


def test_initial_mc_presence_uses_only_actual_fitted_rows():
    model, _ = synthetic_query_model()
    states = np.zeros((8, STATE_DIM), np.float32); states[:, 0] = -1
    options = np.tile(np.arange(4), 2)
    selected = options == 0
    leaves, presence, prediction = training_leaf_cache(model, option_features(states, options), options, selected)
    q, support = fast_q_values(model, presence, states, np.ones(8, bool))
    assert support[:, 0].all()
    assert not support[:, 1:].any()
    assert np.isfinite(q).all() and np.isfinite(prediction).all()
