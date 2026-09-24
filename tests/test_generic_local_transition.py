import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

from generic_local_transition import LocalTripletLearner


def test_local_competition_learns_both_tempos_from_unlabeled_events():
    from generic_local_transition import event_sequence, item, evaluate

    learner = LocalTripletLearner(eta=.3, local_competition=True)
    slow = event_sequence([item('square', (16, 32), 'right', 1)], background=True)
    fast = event_sequence([item('square', (16, 32), 'right', 2)], background=True)
    for _ in range(4):
        for sequence in (slow, fast):
            learner.reset_state()
            for event in sequence:
                learner.step(event)
    outcome = evaluate(learner, [('slow', slow), ('fast', fast)])['groups']
    assert outcome['slow']['learned']['f1'] > .8
    assert outcome['fast']['learned']['f1'] > .8
    assert 'unit_competitive' in outcome['slow']


def event(x):
    value = torch.zeros((2, 32, 64))
    value[0, 16, x] = 1
    return value


def test_local_visual_error_strengthens_a_repeated_event_triplet():
    learner = LocalTripletLearner(eta=.3)
    learner.step(event(10))
    learner.step(event(11))
    learner.step(event(12))
    assert learner.weights[0, 1] > 0
    prediction = learner.step(event(13))
    assert prediction[0, 16, 14] > 0
    assert learner.weights.min() >= 0 and learner.weights.max() <= 1


def test_isolated_static_event_cannot_strengthen_transition():
    learner = LocalTripletLearner(eta=.3)
    learner.step(event(10))
    learner.step(torch.zeros((2, 32, 64)))
    learner.step(torch.zeros((2, 32, 64)))
    assert torch.count_nonzero(learner.weights) == 0
