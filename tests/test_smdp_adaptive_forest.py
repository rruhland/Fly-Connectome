"""Actual option semantics, physical-time credit and portable export checks."""
from pathlib import Path
import sys

import numpy as np
import pytest
from sklearn.ensemble import ExtraTreesRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
from pong_smdp_adaptive_forest import (AdaptiveForest, SMDPController, STATE_DIM,
    balanced_weights, discount, observed_mc_returns, observed_windows, option_features,
    state_features, leaf_independence, sklearn_q_values)


def test_macro_and_baseline_grouping_have_same_physical_hit_value():
    elapsed = np.tile(np.array([2, 2, 3, 2, 3]) / 120, 4)
    hits = np.zeros(20, bool); hits[15] = True
    rows, _ = observed_windows(np.zeros(20), hits, np.zeros(20, bool),
        np.zeros(20), np.ones(20, bool))
    assert (0, 0, 15, 1) in rows
    mc = observed_mc_returns(hits, np.zeros(20, bool), elapsed)
    assert mc[0] == pytest.approx(float(discount(elapsed[:16].sum())))
    assert mc[0] == pytest.approx(float(np.prod(discount(elapsed[:16]))))
    assert mc[15] == pytest.approx(float(discount(elapsed[15])))
    assert np.isnan(mc[16:]).all()


def test_first_hit_and_natural_end_stop_macros_while_short_censor_does_not():
    rows, rejected = observed_windows(np.array([1, 1, 1, 2]), np.array([0, 0, 1, 0]),
        np.zeros(4, bool), np.zeros(4), np.ones(4, bool))
    assert (0, 1, 2, 1) in rows
    assert not any(row[0] == 3 and row[1] != 3 for row in rows)
    rows, _ = observed_windows(np.array([1, 1]), np.zeros(2), np.array([0, 1]),
        np.zeros(2), np.array([1, 0]))
    assert (0, 1, 1, 2) in rows
    rows, rejected = observed_windows(np.array([1, 2] * 10), np.zeros(20), np.zeros(20),
        np.full(20, 2), np.ones(20, bool))
    assert not any(row[1] != 3 for row in rows)
    assert all(row[0] % 2 for row in rows)  # Only real baseline actions.
    assert rejected['changing_action_before_boundary'] > 0


def test_mc_natural_failure_zero_and_unknown_censor_stays_unknown():
    elapsed = np.full(5, .02)
    assert np.all(observed_mc_returns(np.zeros(5), [0, 0, 0, 0, 1], elapsed) == 0)
    assert np.isnan(observed_mc_returns(np.zeros(5), np.zeros(5), elapsed)).all()


def test_weight_mass_is_equal_by_option_then_start():
    options = np.array([0, 0, 0, 0, 3, 3]); starts = np.array([1, 1, 1, 2, 1, 2])
    weights = balanced_weights(options, starts)
    assert weights[options == 0].sum() == pytest.approx(.5)
    assert weights[(options == 0) & (starts == 1)].sum() == pytest.approx(.25)
    assert weights[(options == 0) & (starts == 2)].sum() == pytest.approx(.25)


def test_empty_overflow_and_onehot_categories():
    assert state_features(np.empty((0, 6)), 0, True, np.ones(4)) is None
    assert state_features(np.ones((5, 6)), 0, True, np.ones(4)) is None
    state = state_features([[.2, -.3, 0, 0, 1, 1]], 2, True, np.ones(4))
    assert state.shape == (STATE_DIM,)
    assert np.array_equal(option_features(state[None], [2])[0, -4:], [0, 0, 1, 0])


def test_c_prediction_export_support_and_serialization(tmp_path):
    rng = np.random.default_rng(17)
    states = rng.normal(size=(128, STATE_DIM)).astype(np.float32)
    options = np.tile(np.arange(4), 32)
    starts = np.arange(128) % 9
    F = option_features(states, options)
    y = .2 + .5 * (options == 0)
    model = ExtraTreesRegressor(n_estimators=32, min_samples_leaf=4, max_depth=None,
        bootstrap=False, random_state=2026102903, n_jobs=1).fit(F, y)
    counts = leaf_independence(model, F, options, starts)
    forest = AdaptiveForest.export(model, np.ones(4), F, options, starts, counts)
    assert np.array_equal(model.predict(F), forest.predict(F, options)[0])
    expected_q, expected_support, _ = sklearn_q_values(model, counts, states, np.ones(128, bool))
    actual_q, actual_support, _ = forest.q_values(states)
    assert np.array_equal(expected_q, actual_q)
    assert np.array_equal(expected_support, actual_support)
    path = tmp_path / 'forest.npz'; forest.save(path)
    reloaded = AdaptiveForest.load(path)
    assert np.array_equal(reloaded.q_values(states)[0], expected_q)
    assert not reloaded.feature.flags.writeable


def test_controller_ends_hold_on_hit_and_preserves_opening():
    class Stub:
        def inspect(self, scene, phase, past_hit):
            return dict(option=1)
    controller = SMDPController(Stub())
    assert controller.choose(None, 2)[0] == 2
    controller.observe(1)
    action, decision = controller.choose(None, 2)
    assert action == 1 and decision['hold_after'] == 15 and decision['phase'] == 1
    controller.observe(0)
    assert controller.choose(None, 2)[1]['mode'] == 'earned_hold'
    controller.observe(1)
    assert controller.remaining == 0
    assert controller.choose(None, 2)[1]['mode'] == 'earned_start'
