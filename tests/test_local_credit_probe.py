import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from local_credit_probe import step, replay, issue_design


def test_credit_uses_issued_context_and_clips_signed_prediction():
    x = np.array([[-2., 1.], [-2., 1.]])
    state = np.array([-1., 1.])
    v = issue_design(x, state, split=True)
    np.testing.assert_array_equal(v, [[-2., 1., 0., 0.], [0., 0., -2., 1.]])
    weights = np.ones(4)
    updated, prediction, gain = step(weights, v[0], -1., 1., True, 1., 1.)
    assert prediction == -1.
    assert gain == 1.
    np.testing.assert_array_equal(updated, weights)


def test_closed_gate_makes_no_amplitude_update_even_on_surprise():
    weights = np.array([.2, .3])
    updated, prediction, _ = step(weights, np.array([1., -1.]), 1., False, True, 1., 1.)
    assert prediction == 0.
    np.testing.assert_array_equal(updated, weights)


def test_causal_counts_use_only_previous_confirmations():
    x = np.ones((3, 1))
    y = np.array([0., 0., 1.])
    result = replay(x, y, np.ones(3, bool), np.zeros(1), eta=.1, balanced=True)
    np.testing.assert_array_equal(result['gains'], [1., 1., 3.])
    assert result['weights'][0] == pytest.approx(.3)


def test_ungated_rule_updates_on_a_closed_gate():
    w, p, _ = step(np.zeros(1), np.ones(1), 1., False, False, .1, 1.)
    assert p == 0.
    assert w[0] == .1


def test_first_pass_records_each_forecast_before_its_confirmation_update():
    result = replay(np.ones((2, 1)), np.ones(2), np.ones(2, bool),
                    np.zeros(1), eta=.1)
    np.testing.assert_allclose(result['first_issued'], [0., .1])
