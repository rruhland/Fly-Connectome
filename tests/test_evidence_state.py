"""Behavioral contracts for opt-in causal evidence association."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from evidence_state import EvidenceTracker


def observation(positions):
    state = torch.zeros(12, 32, 64)
    for y, x in positions:
        state[1, y, x] = 1.
        state[6, y, x] = 1.
    return state


@pytest.mark.parametrize('representation', ['events', 'surface'])
def test_one_flash_does_not_become_a_confirmed_moving_entity(representation):
    tracker = EvidenceTracker(representation=representation)
    tracker.step(observation([(12, 10)]))
    for _ in range(4):
        tracker.step(observation([]))
    assert not any(slot['confirmed'] for slot in tracker.slots)


@pytest.mark.parametrize('representation', ['events', 'surface'])
def test_two_streams_remain_separate_and_continue_across_missing_input(representation):
    tracker = EvidenceTracker(representation=representation)
    for frame in range(5):
        tracker.step(observation([(8, 8+frame), (24, 40-frame)]))
    live = [slot for slot in tracker.slots if slot['confirmed']]
    assert len(live) == 2
    for _ in range(2):
        tracker.step(observation([]))
    centers = sorted((slot['position'] +
                      (tracker.frame-slot['last_seen'])*slot['velocity']).tolist()
                     for slot in live)
    assert centers[0] == pytest.approx([8., 14.], abs=.3)
    assert centers[1] == pytest.approx([24., 34.], abs=.3)
    tracker.reset_state()
    tracker.step(observation([]))
    assert tracker.slots == []
