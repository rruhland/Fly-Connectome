"""The history benchmark hides true state without leaking it through frames."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from visual_history_benchmark import (periodic_steps, render_case,
                                      render_context_case,
                                      render_two_mover_case)


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


def test_two_movers_are_hidden_together_and_reappear():
    case = render_two_mover_case(speed=1, y=16,
                                 shapes=('dot', 'plus'))
    frame = case['decision']
    assert not bool(case['events'][frame].any())
    assert all(bool(mask[frame].any())
               for mask in case['hidden_by_entity'])
    assert not bool(case['visible_objects'][frame].any())
    assert all(reveal is not None and reveal > frame
               for reveal in case['reveal_by_entity'])


def test_two_mover_disappearance_has_same_pre_reveal_observation():
    continuing = render_two_mover_case(speed=1, y=16,
                                       shapes=('dot', 'plus'))
    vanished = render_two_mover_case(speed=1, y=16,
                                     shapes=('dot', 'plus'),
                                     disappear=(True, False))
    frame = continuing['decision']
    assert all(torch.equal(a, b) for a, b in zip(
        continuing['events'][:frame+1], vanished['events'][:frame+1]))
    assert not bool(vanished['hidden_by_entity'][0][frame].any())
    assert bool(vanished['hidden_by_entity'][1][frame].any())


def test_periodic_motion_changes_displacement_and_preserves_occlusion():
    steps = periodic_steps(64, cycle=(1, 2, 3))
    assert steps[:5] == [0, 1, 3, 6, 7]
    case = render_case(direction=1, speed=1, y=16, shape='dot',
                       steps=steps)
    assert bool(case['hidden'][case['decision']].any())
    assert not bool(case['events'][case['decision']].any())
    late = render_case(direction=1, speed=1, y=16, shape='dot',
                       steps=steps, decision_index=-2)
    assert late['decision'] > case['decision']


def test_context_changes_hidden_path_without_exposing_hidden_pixels():
    up = render_context_case(direction=1, speed=1, y=16,
                             shape='dot', cue_sign=-1)
    down = render_context_case(direction=1, speed=1, y=16,
                               shape='dot', cue_sign=1)
    frame = up['decision']
    assert frame == down['decision']
    assert torch.equal(up['visible_objects'][frame],
                       down['visible_objects'][frame])
    assert not torch.equal(up['hidden'][frame], down['hidden'][frame])
    assert not bool(up['events'][frame].any())
    assert not bool(down['events'][frame].any())
    assert bool((up['cue_mask'] & up['visible'][frame]).any())


def test_context_disappearance_is_unknown_before_reveal():
    continuing = render_context_case(direction=-1, speed=1, y=16,
                                     shape='plus', cue_sign=1)
    vanished = render_context_case(direction=-1, speed=1, y=16,
                                   shape='plus', cue_sign=1,
                                   disappear=True)
    frame = continuing['decision']
    assert all(torch.equal(a, b) for a, b in zip(
        continuing['events'][:frame+1], vanished['events'][:frame+1]))
    assert not bool(vanished['hidden'][frame].any())
    assert bool(continuing['hidden'][frame].any())


def test_context_mark_can_be_shuffled_independently_of_outcome():
    aligned = render_context_case(direction=1, speed=1, y=16,
                                  shape='dot', cue_sign=1)
    shuffled = render_context_case(direction=1, speed=1, y=16,
                                   shape='dot', cue_sign=1,
                                   turn_sign=-1)
    first_hidden = next(frame for frame, mask in enumerate(aligned['hidden'])
                        if bool(mask.any()))
    assert all(torch.equal(a, b) for a, b in zip(
        aligned['events'][:first_hidden],
        shuffled['events'][:first_hidden]))
    assert not torch.equal(aligned['hidden'][aligned['decision']],
                           shuffled['hidden'][shuffled['decision']])
