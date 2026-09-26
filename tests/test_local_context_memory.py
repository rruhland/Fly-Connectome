"""Visible context and later reacquisition supply local self-supervision."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from local_context_memory import LocalContextMemory
from local_observation_model import LocalObservationModel
from visual_history_benchmark import render_context_case
from visual_history_controls import sensory_sequence


def test_context_effect_is_learned_only_after_observed_reacquisition():
    observer = LocalObservationModel()
    model = LocalContextMemory()
    for sign in (-1, 1):
        case = render_context_case(direction=1, speed=1, y=16,
                                   shape='dot', cue_sign=sign)
        model.reset_state()
        for state in sensory_sequence(observer, case, hybrid=False):
            model.step(state, learn=True)
    assert bool((model.counts > 0).all())
    assert model.effects[0] < 0 < model.effects[1]


def test_hidden_context_is_causal_and_does_not_train_on_target():
    case = render_context_case(direction=1, speed=1, y=16,
                               shape='dot', cue_sign=1)
    model = LocalContextMemory()
    sensory = sensory_sequence(LocalObservationModel(), case, hybrid=False)
    for state in sensory[:case['decision']+1]:
        model.step(state, learn=True)
    assert model.counts.sum() == 0


def test_contradictory_visible_absence_reduces_false_persistence():
    observer = LocalObservationModel()
    model = LocalContextMemory()
    training = render_context_case(direction=1, speed=1, y=16,
                                   shape='dot', cue_sign=1)
    for state in sensory_sequence(observer, training, hybrid=False):
        model.step(state, learn=True)
    continuing = render_context_case(direction=1, speed=1, y=16,
                                     shape='dot', cue_sign=1)
    vanished = render_context_case(direction=1, speed=1, y=16,
                                   shape='dot', cue_sign=1, disappear=True)
    scores = []
    frame = continuing['reveal']+2
    target = continuing['visible_objects'][frame]
    for case in (continuing, vanished):
        model.reset_state()
        for state in sensory_sequence(observer, case, hybrid=False)[:frame+1]:
            field = model.step(state)
        scores.append(float(field[target].max()))
    assert scores[0] > .5
    assert scores[1] < .5
