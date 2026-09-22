import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from prediction_timing import best_window, window_mask


def test_window_retains_existing_event_amplitudes_without_quiet_alarms():
    state = np.array([2., 2., 1., 3.])
    y = np.array([-1., 1., 0., 0.])
    p = np.array([-.8, .9, .5, .5])
    fitted = best_window(state, p, y, np.full(4, 2))
    keep = window_mask(state, fitted)
    np.testing.assert_array_equal(keep, [True, True, False, False])
    assert fitted['minimum_retention'] == 1
    assert fitted['lower'] == 1.5
    assert fitted['upper'] == 2.5


def test_no_window_can_distinguish_identical_event_and_quiet_state():
    state = np.ones(4)
    y = np.array([-1., 1., 0., 0.])
    p = np.array([-.8, .9, .5, .5])
    fitted = best_window(state, p, y, np.full(4, 2))
    assert fitted['minimum_retention'] == 0
    assert not window_mask(state, fitted).any()


def test_window_applies_fixed_bounds_to_unseen_samples():
    fitted = dict(lower=1., upper=2.)
    np.testing.assert_array_equal(window_mask(np.array([0., 1., 1.5, 2., 3.]), fitted),
                                  [False, True, True, True, False])


def test_quiet_budget_is_enforced_per_tempo_not_pooled():
    y = np.r_[-1., 1., np.zeros(20), -1., 1., np.zeros(100)]
    p = np.r_[-.8, .8, np.full(20, .5), -.8, .8, np.full(100, .5)]
    value = np.r_[2., 2., 2., 2., np.ones(18), 2., 2., np.ones(100)]
    tempo = np.r_[np.full(22, 2), np.full(102, 4)]
    fitted = best_window(value, p, y, tempo)
    assert fitted['minimum_retention'] == 0
    assert not window_mask(np.array([2.]), fitted)[0]
