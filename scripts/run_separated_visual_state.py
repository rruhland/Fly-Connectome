"""Matched opt-in test of local latent continuity and event emission."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases, scene_sequence, train_models
from diverse_visual_experience import (TRAIN_SHAPES,
                                       diverse_training_sequences,
                                       unseen_shape_sequences)
from generic_local_transition import (LocalTripletLearner, accumulate,
                                      empty_score, finish)
from learned_latent_probe import DIRECTIONS
from separated_visual_state import SeparatedVisualState


OUT = Path('docs/experiments/2026-09-24-separated-visual-state-results.json')
VELOCITY = {'up': (-1, 0), 'down': (1, 0),
            'left': (0, -1), 'right': (0, 1)}
NAMES = ('archived', 'matched_primitive', 'recurrent',
         'recurrent_no_imagined', 'no_recurrence', 'fixed')


def make_training_cases():
    cases = [('clean', shape, events)
             for episodes in diverse_training_sequences()
             for shape, _, _, _, events in episodes]
    for shape_index, shape in enumerate(TRAIN_SHAPES):
        for direction_index, direction in enumerate(DIRECTIONS):
            dy, dx = VELOCITY[direction]
            for speed in (1, 2):
                interrupted = (shape_index+direction_index+speed) % 2 == 0
                obj = dict(shape=shape, center=(16, 32),
                           before=(dy*speed, dx*speed),
                           after=(dy*speed, dx*speed))
                if interrupted:
                    obj['hidden'] = (7, 8, 9)
                background = (shape_index+speed) % 2 == 0
                events = scene_sequence(
                    [obj], background=background,
                    noise=0. if interrupted else .002,
                    seed=100+shape_index*16+direction_index*2+speed)
                cases.append(('interruption' if interrupted else 'noise',
                              shape, events))
    return cases


@torch.no_grad()
def train_candidates(cases):
    archived = train_models()['learned']
    matched = copy.deepcopy(archived)
    matched.predictive.zero_()
    recurrent_encoder = copy.deepcopy(archived)
    no_recurrence_encoder = copy.deepcopy(archived)
    recurrent_encoder.predictive.zero_()
    no_recurrence_encoder.predictive.zero_()
    recurrent = SeparatedVisualState(recurrent_encoder, recurrence=True)
    no_recurrence = SeparatedVisualState(no_recurrence_encoder,
                                          recurrence=False)
    encoded = [(kind, shape, events, correlation_sequence(events))
               for kind, shape, events in cases]
    rng = random.Random(3)
    rng.shuffle(encoded)
    for _, _, events, codes in encoded:
        matched.reset_state()
        recurrent.reset_state()
        no_recurrence.reset_state()
        for event, code in zip(events, codes):
            matched.step(code, learn_prediction=True)
            recurrent.step(code, event, learn=True)
            no_recurrence.step(code, event, learn=True)
    no_imagined = copy.deepcopy(recurrent)
    no_imagined.emission_imagined.zero_()
    return dict(archived=archived, matched_primitive=matched,
                recurrent=recurrent, recurrent_no_imagined=no_imagined,
                no_recurrence=no_recurrence)


@torch.no_grad()
def evaluate(models, cases):
    groups = {family: {name: empty_score() for name in NAMES}
              for family, _, _ in cases}
    overall = {name: empty_score() for name in NAMES}
    quiet = {name: dict(frames=0, false_alarm_pixels=0) for name in NAMES}
    state = dict(blank_inputs=0, recurrent_active=0,
                 no_recurrence_active=0)
    fixed = LocalTripletLearner(eta=.3, local_competition=True)
    fixed.weights.fill_(1)
    for family, _, events in cases:
        codes = correlation_sequence(events)
        for model in models.values():
            model.reset_state()
        fixed.reset_state()
        for t, (event, code) in enumerate(zip(events, codes)):
            predictions = dict(
                archived=primitive_to_events(models['archived'].step(code)),
                matched_primitive=primitive_to_events(
                    models['matched_primitive'].step(code)),
                recurrent=models['recurrent'].step(code, event),
                recurrent_no_imagined=models['recurrent_no_imagined'].step(
                    code, event),
                no_recurrence=models['no_recurrence'].step(code, event),
                fixed=fixed.step(event, learn=False))
            if code.sum() == 0:
                state['blank_inputs'] += 1
                state['recurrent_active'] += int(
                    models['recurrent'].imagined.sum() > 0)
                state['no_recurrence_active'] += int(
                    models['no_recurrence'].imagined.sum() > 0)
            if 3 <= t < 14:
                target = events[t+1]
                for name, prediction in predictions.items():
                    accumulate(groups[family][name], prediction, target)
                    accumulate(overall[name], prediction, target)
                    if target.sum() == 0:
                        quiet[name]['frames'] += 1
                        quiet[name]['false_alarm_pixels'] += int(
                            (prediction >= .5).sum())
    return dict(overall={name: finish(row) for name, row in overall.items()},
                groups={family: {name: finish(row)
                                 for name, row in methods.items()}
                        for family, methods in groups.items()},
                quiet=quiet, state=state)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = make_training_cases()
    models = train_candidates(training)
    training_scores = evaluate(models, training)
    heldout = [('single', shape, events)
               for shape, _, _, _, events in unseen_shape_sequences()]
    heldout += [(family, 'unseen', events)
                for family, _, events in make_robust_cases()]
    heldout_scores = evaluate(models, heldout)
    result = dict(training_episodes=len(training),
                  training=training_scores, heldout=heldout_scores,
                  weights=dict(state_sum=float(
                      models['recurrent'].state_weights.sum()),
                      emission_observed_sum=float(
                          models['recurrent'].emission_observed.sum()),
                      emission_imagined_sum=float(
                          models['recurrent'].emission_imagined.sum())),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        training={name: round(row['f1'], 3)
                  for name, row in training_scores['overall'].items()},
        heldout={family: {name: round(row['f1'], 3)
                          for name, row in methods.items()}
                 for family, methods in heldout_scores['groups'].items()},
        quiet=heldout_scores['quiet'], state=heldout_scores['state'],
        weights=result['weights'], elapsed_seconds=result['elapsed_seconds'])),
        flush=True)


if __name__ == '__main__':
    main()
