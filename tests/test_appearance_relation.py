"""Visual appearance carries the relation; hidden state remains evaluation-only."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from appearance_relation import appearance_case
from visual_history_benchmark import render_context_case


def test_mark_changes_without_revealing_hidden_trajectory():
    base = render_context_case(direction=1, speed=1, y=16,
                               shape='dot', cue_sign=-1)
    result = appearance_case(base)
    assert result['cue_mask'].sum() == 4
    assert all(torch.equal(a, b) for a, b in zip(base['hidden'], result['hidden']))
    assert not torch.equal(base['events'][0], result['events'][0])
    assert torch.equal(result['visible'][result['decision']], result['visible'][result['decision']-1])


def test_same_mark_and_prefix_cannot_reveal_hidden_disappearance():
    arguments = dict(direction=1, speed=1, y=16, shape='square', cue_sign=1)
    continuing = appearance_case(render_context_case(**arguments))
    vanished = appearance_case(render_context_case(**arguments, disappear=True))
    for frame in range(continuing['reveal']):
        assert torch.equal(continuing['events'][frame], vanished['events'][frame])
