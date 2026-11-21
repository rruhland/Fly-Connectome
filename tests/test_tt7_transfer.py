"""Checks that the catching benchmark preserves its pixel boundary and paired trials."""

from collections import Counter

import numpy as np

from scripts.experimental.tt7_transfer_catch import CatchPlayer, FruitCourt, gaze
from scripts.experimental.tt7_transfer_timing import expected_counts, train_dynamics
from scripts.experimental.tt7_transfer_trial import learned_prior, play, schedule, verify


def test_complete_blocks_balance_delay_pairs_and_visibility():
    matches = schedule(7, 4, 18, occlusions=True)
    counts = Counter((row['delays'][0][1], row['delays'][1][1], row['occlusion']) for row in matches)
    assert len(counts) == 72
    assert set(counts.values()) == {1}
    assert matches == schedule(7, 4, 18, occlusions=True)


def test_hidden_basket_is_missing_from_pixels_and_action_tapes_replay():
    court = FruitCourt(occlusion=True)
    x, y, basket = gaze(court.reset(seed=171))
    assert y == 7.
    assert np.isfinite([x, basket]).all()
    rgb = court.frame()
    while gaze(rgb)[1] < 46.:
        rgb, _, _, _, _ = court.step(np.zeros(1))
    assert gaze(rgb)[2] is None

    specification = dict(fall=171, noise=95, delays=((0, 1), (7, 4)), occlusion=True)
    results = {name: play(CatchPlayer(dynamics=dynamics), specification)
               for name, dynamics in [('visual', False), ('dynamics', True)]}
    receipt = dict(rows=[dict(specification=specification, players=results)])
    assert verify(receipt) == 2
    assert all(len(row['commands']) == court.flight for row in results.values())


def test_expected_counts_use_temporal_order_of_optical_evidence():
    initial = np.ones(2) / 2.
    transition = np.array([[.9, .1], [.1, .9]])
    grouped = np.array([[1., .01], [1., .01], [.01, 1.], [.01, 1.]])
    alternating = grouped[[0, 2, 1, 3]]
    first, _ = expected_counts(grouped, initial, transition)
    second, _ = expected_counts(alternating, initial, transition)
    np.testing.assert_allclose(first.sum(), 3.)
    assert second.trace() < first.trace()


def test_donors_use_only_visual_timing_evidence():
    first = dict(timing=dict(estimated=2., informative=10), optical_likelihoods=[[1., .2, .1, .1, .1, .1]] * 6)
    original = dict(visits=[dict(points=[first])])
    unrelated = dict(visits=[dict(points=[dict(**first, evaluator={'delay': 5}, phase='unused')])])
    assert learned_prior(original) == learned_prior(unrelated)
    assert train_dynamics(original) == train_dynamics(unrelated)
