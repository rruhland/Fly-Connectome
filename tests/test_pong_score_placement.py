"""Terminal credit and interception fallback for the public scoring layer."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from pong_score_placement import ScorePlacement, ScorePlacementController


def test_frozen_learning_depends_on_terminal_reward_and_generalizes_to_mirror():
    policy = ScorePlacement()
    before = policy.frozen_branch()
    policy.observe_score(4, .24, 1.)
    policy.observe_score(3, .24, -1.)
    after = ScorePlacement(**json.loads(json.dumps(policy.to_state())),
                           updates_enabled=False)
    assert before.episode(1).offset(1, .25, .50) == 0.
    assert after.episode(1).offset(1, .25, .50) == -.091
    mirrored = after.episode(1)
    assert mirrored.offset(1, -.25, .50) == .091
    mirrored.finish(-1.)
    assert after.to_state() == policy.to_state()


def test_reward_without_an_eligible_approach_never_trains_placement():
    policy = ScorePlacement()
    episode = policy.episode(1)
    assert episode.offset(1, .10, .10) == 0.
    episode.finish(1.)
    assert not policy.visits.any()
    with pytest.raises(ValueError):
        episode.finish(-1.)


def test_score_updates_last_commitment_once_and_tracks_terminal_zero():
    policy = ScorePlacement()
    policy.observe_score(1, .24, 1.)
    episode = policy.episode(4)
    assert episode.offset(1, .24, .60) == .079
    assert episode.offset(1, .30, .03) == .079
    assert episode.offset(2, .40, .50) == 0.
    episode.finish(0.)
    assert policy.visits.tolist() == [1, 1, 0, 0, 0, 0, 0]
    assert policy.totals[0].sum() == 0.


def test_timing_hook_preserves_zero_offset_and_limits_nonzero_paddle_target():
    timing = SimpleNamespace(seconds=.30, delay_commands=0, camera_seconds=.02)
    policy = ScorePlacement()
    policy.observe_score(1, .25, 1.)
    controller = ScorePlacementController(timing, policy.episode(1))
    context = SimpleNamespace(intercept=.25, seconds=.6, paddle=.15, lower=-.4, upper=.4)
    assert timing.contact_offset(None) == 0.
    assert timing.contact_offset(context) == pytest.approx(.079)
    context.intercept = .35
    assert timing.contact_offset(context) == pytest.approx(.05)
    assert controller.placement.choices[0].offset == .079
