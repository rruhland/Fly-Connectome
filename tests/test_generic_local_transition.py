import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parents[1]/'scripts'))

from generic_local_transition import LocalTripletLearner


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
