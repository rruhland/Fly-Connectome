"""The history benchmark hides true state without leaking it through frames."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from visual_history_benchmark import render_case


def test_opposite_histories_have_same_current_observation_but_different_state():
    right = render_case(direction=1, speed=1, y=16, shape='dot')
    left = render_case(direction=-1, speed=1, y=16, shape='dot')
    frame = right['decision']
    assert frame == left['decision']
    assert torch.equal(right['visible'][frame], left['visible'][frame])
    assert not bool(right['events'][frame].any())
    assert not bool(left['events'][frame].any())
    assert not torch.equal(right['hidden'][frame], left['hidden'][frame])


def test_disappearance_is_unknowable_until_after_hidden_interval():
    continuing = render_case(direction=1, speed=2, y=12, shape='square')
    vanished = render_case(direction=1, speed=2, y=12, shape='square',
                           disappear=True)
    frame = continuing['decision']
    assert all(torch.equal(a, b) for a, b in zip(
        continuing['events'][:frame+1], vanished['events'][:frame+1]))
    assert not bool(vanished['hidden'][frame].any())
    assert bool(continuing['hidden'][frame].any())
    assert continuing['reveal'] is not None
    assert vanished['reveal'] is None


def test_sparse_intensity_contains_only_visible_scene():
    case = render_case(direction=-1, speed=1, y=20, shape='plus')
    frame = case['decision']
    assert torch.equal(case['intensity'][frame],
                       case['visible'][frame].float())
    assert torch.equal(case['intensity'][frame],
                       render_case(direction=1, speed=1, y=20,
                                   shape='dot')['intensity'][frame])
