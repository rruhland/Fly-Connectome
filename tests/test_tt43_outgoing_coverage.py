import numpy as np

from scripts.experimental.tt43_outgoing_coverage import choice, estimate, ridge


def test_risk_never_accepts_less_robust_or_unsupported_candidate():
    quality = np.array([-4., -3.9, -3.9, -3.9])
    risk = np.array([.50, .25, 0., 0.])
    agreement = np.array([2, 2, 1, 2])
    support = np.array([True, True, True, False])
    assert choice(agreement, quality, risk, support) == (1, 0)
    support[0] = False
    assert choice(agreement, quality, risk, support) == (0, 0)


def test_material_gain_and_current_legal_quality_are_both_required():
    agreement = np.array([2, 2, 2])
    quality = np.array([-3., -2.3, -2.98])
    support = np.ones(3, dtype=bool)
    # The second candidate is too much worse on the present stroke. The third
    # looks safer by only 0.01, too little to overrule the stock comparison.
    assert choice(agreement, quality, np.array([.45, 0., .44]), support) == (0, 0)


def test_regression_learns_an_observed_risk_mechanism_without_gradient_fitting():
    x = np.tile([.06, .65, .34, .05, 1.28, 1.18, 2.30], (20, 1))
    x[:10, 6] = 1.5
    y = np.r_[np.ones((10, 2)), np.zeros((10, 2))]
    predicted = estimate(ridge(x, y), x[[0, 19]])
    assert (predicted[0] > predicted[1]).all()
    assert ((0 <= predicted) & (predicted <= 1)).all()
