"""Verify the detector responds only to consecutive confident contradictions."""

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from body_utility_surprise import SurpriseUtility
from body_visual_utility import SCHEMA, UtilityLearner


def test_three_confident_contradictions_reset_and_replay_utility_evidence():
    axes = np.eye(5)
    detector = SurpriseUtility({'schema': SCHEMA, 'weights': [1.] * 5})
    for axis in axes[:2]:
        detector.observe(axis, -1)
    assert detector.resets == 0
    detector.observe(axes[2], -1)
    assert detector.resets == 1
    restarted = UtilityLearner()
    for axis in axes[:3]:
        restarted.observe(axis, -1)
    assert detector.checkpoint() == restarted.checkpoint()


def test_agreement_breaks_a_streak_and_uncertain_errors_do_not_reset():
    e0, e1, e2, e3, e4 = np.eye(5)
    confident = SurpriseUtility({'schema': SCHEMA, 'weights': [1.] * 5})
    for axis, label in ((e0, -1), (e1, -1), (e2, 1), (e3, -1), (e4, -1)):
        confident.observe(axis, label)
    assert confident.resets == 0

    uncertain = SurpriseUtility({'schema': SCHEMA, 'weights': [.05] * 5})
    for axis in (e0, e1, e2, e3, e4):
        uncertain.observe(axis, -1)
    assert uncertain.resets == 0
