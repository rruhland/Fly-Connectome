"""The return test transfers an adjustment rate without leaking a new wall value."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from hockey_adaptation_rate import RateLearner, select_rate
from occluded_pong_transfer import upper_cover
from registered_arm_preferences import register


def test_same_initial_wall_and_faster_revision_only_after_an_observed_transition():
    worlds = [1.] * 5 + [.3] * 6 + [.8] * 5
    source = [{'lives': [{'bounces': [{'coefficient': value} for value in worlds]}]}]
    stable = [{'lives': [{'bounces': [{'coefficient': 1.} for _ in worlds]}]}]
    fast = RateLearner(1., select_rate(source, 1.)['rate'])
    slow = RateLearner(1., select_rate(stable, 1.)['rate'])
    assert fast.coefficient == slow.coefficient == 1.
    for learner in (fast, slow):
        learner.observe_hint(.4, .3)
        assert learner.coefficient == 1.
        learner.observe(.4, .03)
    assert abs(fast.coefficient - .4) < abs(slow.coefficient - .4)


def test_the_same_fixed_cover_leaves_paddles_and_lower_half_of_each_lane_visible():
    pixels = torch.ones(2, 64, 64, dtype=torch.bool)
    image = upper_cover(pixels)
    assert not image[:, :32, 5:43].any()
    assert image[:, 32:, :].all()
    assert image[:, :, :5].all()
    assert image[:, :, 43:].all()


def test_registration_recovers_an_unseen_point_from_pixel_correspondences():
    seen = np.array([[20., 10.], [70., 55.], [32., 87.]])
    original = np.array([[.8, -.3], [.2, .1], [.55, .5]])
    affine = register(seen, original)
    mix = [.2, .3, .5]
    interior = np.r_[np.average(seen, axis=0, weights=mix), 1.]
    assert np.allclose(interior @ affine, np.average(original, axis=0, weights=mix))
