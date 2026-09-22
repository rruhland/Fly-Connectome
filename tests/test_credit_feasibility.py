import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))
from credit_feasibility import feasibility


def test_distinguishable_events_and_quiet_are_feasible():
    x = np.array([[1.], [-1.], [0.], [0.]])
    y = np.array([1., -1., 0., 0.])
    w, report = feasibility(x, y, np.ones(4), 10.)
    assert report['status'] == 0
    assert w[0] >= .1-1e-6


def test_identical_event_and_quiet_features_can_be_infeasible():
    x = np.array([[1.], [-1.], [1.], [-1.]])
    y = np.array([1., -1., 0., 0.])
    w, report = feasibility(x, y, np.ones(4), 10., anticipation=.2)
    assert report['status'] == 2
    assert w is None


def test_lower_clipping_is_included_in_capacity_constraints():
    # Unclipped event mean is always negative; clipped mean can exceed .1.
    x = np.array([[1.], [1.], [-100.], [-1.], [0.]])
    y = np.array([1., 1., 1., -1., 0.])
    w, report = feasibility(x, y, np.ones(5), 10.)
    assert report['status'] == 0
    assert np.clip(x[:3, 0]*w[0], -1, 1).mean() >= .1-1e-6


def test_removing_quiet_limit_makes_conflicting_examples_feasible():
    x = np.array([[1.], [-1.], [1.], [-1.]])
    y = np.array([1., -1., 0., 0.])
    weights, report = feasibility(x, y, np.ones(4), 10., anticipation=.2, quiet_limit=1.)
    assert report['status'] == 0
    assert weights[0] >= .2-1e-6


def test_recorded_multiscale_features_admit_the_known_feasible_control():
    a = np.load(Path(__file__).parent/'fixtures/credit_capacity_feasible.npz')
    p = np.clip(a['x']@a['known_weights'], -1, 1)
    for t in [2, 4, 6]:
        for sign in [-1, 1]:
            mask = (a['tempo'] == t) & (a['y'] == sign)
            assert (p[mask]*sign).mean() >= .1
    weights, report = feasibility(a['x'], a['y'], a['tempo'], 10., quiet_limit=1.)
    assert report['status'] == 0
    assert weights is not None
