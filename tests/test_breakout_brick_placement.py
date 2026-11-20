"""Verify score-only shot credit and use of canonical public pixels."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from breakout_brick_placement import BrickPlacement, BrickShotPolicy, visible_targets
from pong_timing_controller import ArrivalContext


def camera():
    pixels = np.zeros((64, 64), dtype=bool)
    pixels[27:39, 2:4] = True
    pixels[22:24, 36:38] = True
    pixels[15:20, 44:47] = True
    pixels[35:40, 54:57] = True
    return pixels


def test_visible_targets_exclude_paddle_and_ball_from_public_pixels():
    targets = visible_targets(camera(), -.45)
    assert targets.shape == (2, 3)
    np.testing.assert_array_equal(targets[:, 2], [2.5 / 64, 2.5 / 64])


def test_only_native_score_changes_the_rebound_model_and_frozen_branch_retains_it():
    positive = BrickShotPolicy()
    negative = BrickShotPolicy()
    rng = np.random.default_rng(32)
    eligible = np.arange(9)
    _, scores, _ = positive.choose(.12, -.45, visible_targets(camera(), -.45), eligible, rng)
    assert np.ptp(scores) > 0
    positive.observe_score(scores, 1)
    negative.observe_score(scores, 0)
    assert positive.scored == 1 and negative.scored == 0
    assert positive.probabilities @ scores > negative.probabilities @ scores
    frozen = BrickShotPolicy.from_state(json.loads(json.dumps(positive.to_state())), updates_enabled=False)
    frozen.observe_score(scores, 0)
    assert frozen.to_state() == positive.to_state()


def test_credit_is_once_per_flight_and_excludes_unfinished_terminal_zero():
    timing = SimpleNamespace(seconds=0., delay_commands=0, camera_seconds=.02)
    policy = BrickShotPolicy()
    placement = BrickPlacement(timing, policy, 32)
    placement.camera(camera(), -.45)
    approach = ArrivalContext(.08, .55, .04, -.4, .4, .04)
    placement.observe_score(1.)
    assert policy.updates == 0
    timing.contact_offset(approach)
    assert placement.shot is not None
    placement.observe_score(1.)
    timing.seconds = .85
    placement.observe_score(1.)
    placement.observe_score(1.)
    timing.seconds = 1.5
    timing.contact_offset(approach)
    assert policy.updates == 1
    assert placement.shots[0]['score'] == 2.
    placement.observe_score(0., truncated=True)
    assert len(placement.shots) == 2
    assert placement.shots[0]['credited'] and not placement.shots[1]['credited']
    assert policy.updates == 1


def test_center_forecast_requires_reachable_center_and_uses_the_same_braking_hook():
    timing = SimpleNamespace(seconds=0., delay_commands=0, camera_seconds=.02)
    placement = BrickPlacement(timing, None, 32)
    placement.camera(camera(), -.45)
    unreachable = ArrivalContext(.39, .40, .20, -.40, .40, .19)
    assert timing.contact_offset(unreachable) == 0.
    assert placement.shot is None
    eligible = ArrivalContext(.08, .55, .04, -.40, .40, .04)
    assert 0. < timing.contact_offset(eligible) < 1e-8
    assert placement.shot.offset == 0.
    placement.observe_score(0., truncated=True)
    assert placement.shots[0]['predicted_value'] is None
