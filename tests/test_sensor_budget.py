"""Frame anchors may use only current declared visible sensor evidence."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_observation_model import LocalObservationModel
from sensor_budget import anchored_sequence
from visual_history_benchmark import render_context_case


def test_anchors_ignore_hidden_targets_and_background_metadata():
    case = render_context_case(direction=1, speed=1, y=16,
                               shape='dot', cue_sign=1)
    altered = {**case, 'background': True,
               'hidden': [torch.ones_like(x) for x in case['hidden']],
               'intensity': [torch.randn_like(x) for x in case['intensity']]}
    first = anchored_sequence(LocalObservationModel(), case, period=8)
    second = anchored_sequence(LocalObservationModel(), altered, period=8)
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_future_frames_do_not_change_current_state():
    case = render_context_case(direction=1, speed=1, y=16,
                               shape='dot', cue_sign=1)
    altered = {**case, 'visible': [x.clone() for x in case['visible']]}
    for frame in range(9, len(altered['visible'])):
        altered['visible'][frame].logical_not_()
    first = anchored_sequence(LocalObservationModel(), case, period=8)
    second = anchored_sequence(LocalObservationModel(), altered, period=8)
    assert all(torch.equal(a, b) for a, b in zip(first[:9], second[:9]))
    assert not torch.equal(first[16], second[16])
