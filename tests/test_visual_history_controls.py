"""History controls use only permitted observations and retain causal memory."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_observation_model import LocalObservationModel
from visual_history_benchmark import render_case, render_two_mover_case
from visual_history_controls import (FixedLeakyMemory, GenericMultiTracker,
                                     GenericTracker, sensory_sequence)
from visual_history_scoring import hidden_rank


def test_event_only_sensory_excludes_periodic_intensity_refresh():
    case = render_case(direction=1, speed=1, y=16, shape='dot')
    observer = LocalObservationModel()
    event_only = sensory_sequence(observer, case, hybrid=False)
    hybrid = sensory_sequence(observer, case, hybrid=True)
    assert len(event_only) == len(case['events'])
    assert event_only[0].shape == (12, 32, 64)
    assert not torch.equal(event_only[8], hybrid[8])


def test_tracker_carries_observed_motion_into_hidden_interval():
    case = render_case(direction=1, speed=1, y=16, shape='dot')
    sensory = sensory_sequence(LocalObservationModel(), case, hybrid=False)
    tracker = GenericTracker()
    for frame, value in enumerate(sensory[:case['decision']+1]):
        estimate = tracker.step(value)
    y, x = torch.nonzero(case['hidden'][frame], as_tuple=False)[0]
    peak_y, peak_x = torch.nonzero(estimate == estimate.max(), as_tuple=False)[0]
    assert abs(int(peak_y-y)) <= 2
    assert abs(int(peak_x-x)) <= 2


def test_leaky_control_preserves_input_after_blank():
    memory = FixedLeakyMemory(height=8, width=10)
    sample = torch.zeros((12, 8, 10))
    sample[0, 4, 5] = 1.
    memory.step(sample)
    later = memory.step(torch.zeros_like(sample))
    assert later[12:, 4, 5].sum() > 0


def test_acceleration_control_projects_recent_change_in_speed():
    tracker = GenericTracker(height=8, width=16, acceleration=True)
    for x in (2, 4, 7):
        sample = torch.zeros((12, 8, 16))
        sample[0, 4, x] = 1.
        tracker.step(sample)
    field = tracker.step(torch.zeros((12, 8, 16)))
    assert field[4, 11] > field[4, 10]


def test_multi_tracker_retains_two_separate_hidden_movers():
    case = render_two_mover_case(speed=1, y=16,
                                 shapes=('dot', 'plus'))
    sensory = sensory_sequence(LocalObservationModel(), case, hybrid=False)
    tracker = GenericMultiTracker()
    for frame, value in enumerate(sensory[:case['decision']+1]):
        field = tracker.step(value)
    assert len(tracker.slots) >= 2
    for hidden in case['hidden_by_entity']:
        assert hidden_rank(field, hidden[frame], k=64) == 1.


def test_multi_tracker_ignores_weak_filtered_event():
    tracker = GenericMultiTracker(height=8, width=16,
                                  minimum_strength=.4)
    weak = torch.zeros((12, 8, 16))
    weak[0, 4, 4] = .2
    tracker.step(weak)
    assert tracker.slots == []
    strong = torch.zeros_like(weak)
    strong[0, 4, 5] = .8
    tracker.step(strong)
    assert len(tracker.slots) == 1
