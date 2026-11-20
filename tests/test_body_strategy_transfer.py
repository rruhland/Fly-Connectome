"""Public sensing, reward feedback and matching for the reaching relation task."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))

from body_strategy_transfer import (
    CueLayout,
    HELD_EVENT,
    RelationArena,
    RewardAssociation,
    SUCCESS_TICKS,
    VISUAL_HOLD_TICKS,
    decode_relation_frame,
    run_experiment,
    run_trial,
    sample_layout,
)


def test_camera_and_scalar_reward_hide_unpaid_contact():
    layout = CueLayout((.40, 0.), ((.38, -.32), (.42, 0.)), (.7, .1))
    near = RelationArena('carriage', layout, 0, 8)
    far = RelationArena('carriage', layout, 1, 8)
    scaffold = RelationArena('carriage', layout, 1, 8, feedback=HELD_EVENT)

    np.testing.assert_array_equal(near.frame(), far.frame())
    for tick in range(1, SUCCESS_TICKS + 1):
        near_frame, near_reward = near.step((0., 0.))
        far_frame, far_reward = far.step((0., 0.))
        _, scaffold_reward = scaffold.step((0., 0.))
        np.testing.assert_array_equal(near_frame, far_frame)
        assert far_reward == 0
        assert near_reward == int(tick == SUCCESS_TICKS)
        assert scaffold_reward == (0 if tick == SUCCESS_TICKS else None)
    assert near.success and not far.success


class StillMotor:
    def reset_episode(self):
        pass

    def act(self, position, target):
        return np.zeros(2)

    def observe(self, before, action, after):
        pass


def test_rejection_occurs_after_visible_dwell_and_is_audited_afterward():
    layout = CueLayout((.40, 0.), ((.42, 0.), (.38, -.32)), (.7, .1))
    arena = RelationArena('carriage', layout, 1, 8)
    memory = RewardAssociation(preferred=0)

    result = run_trial(arena, StillMotor(), memory, first_guess=1)

    assert not result['success'] and memory.preferred is None
    assert result['contacts'] == [{'tick': SUCCESS_TICKS, 'relation': 'near', 'reward': 0}]
    assert result['rejections'] == [{
        'tick': VISUAL_HOLD_TICKS, 'relation': 'near', 'reward': 0,
        'physical_hold_verified': True, 'rejected_rewarded_relation': False,
    }]


def test_relation_is_computed_from_reference_rather_than_screen_order():
    rng = np.random.default_rng(91011)
    near_is_left = []
    for _ in range(30):
        layout = sample_layout(rng)
        arena = RelationArena('carriage', layout, 1, 8)
        _, ordered_markers = decode_relation_frame(arena.frame())
        np.testing.assert_allclose(ordered_markers, layout.by_distance(), atol=1 / 96)
        near_is_left.append(ordered_markers[0, 0] < ordered_markers[1, 0])
    assert any(near_is_left) and not all(near_is_left)


def test_matched_online_courses_hide_reversal_and_mask_source_rewards():
    report = run_experiment([13, 14, 15, 16], source_trials=3, target_trials=6, reverse_after=3)

    assert report['feedback'] == 'scalar'
    assert report['target_body'] == 'arm_remapped'
    for stream in report['streams']:
        rewarded = stream['source_rewarded_relation']
        assert stream['initial_associations'] == {
            'carried': rewarded, 'reset': None, 'source_reward_masked': None,
        }
        assert stream['association_checkpoints']['carried']['preferred_relation'] == rewarded
        assert stream['reset'] == stream['source_reward_masked']
        assert stream['target_rewarded_relations'][3] != rewarded
        for condition in ('carried', 'reset'):
            assert stream[condition][3]['success']
            assert stream[condition][4]['first_choice'] == stream['target_rewarded_relations'][3]
            for result in stream[condition]:
                assert all(item['physical_hold_verified'] and not item['rejected_rewarded_relation']
                           for item in result['rejections'])


def test_reward_association_is_an_independent_json_checkpoint():
    source = RewardAssociation()
    source.observe(1, 1)
    encoded = json.dumps(source.export_prior())
    assert json.loads(encoded) == {
        'schema': 'visual-reward-association-v1', 'preferred_relation': 'far',
    }
    target = RewardAssociation.from_prior(json.loads(encoded))
    assert target.preferred == 1
    target.observe(1, 0)
    assert target.preferred is None and source.preferred == 1
    with pytest.raises(ValueError, match='near, far, or null'):
        RewardAssociation.from_prior(dict(source.export_prior(), preferred_relation='right'))
