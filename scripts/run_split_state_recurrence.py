"""Test local hidden continuity with independent raw and motion encoders."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases, scene_sequence, train_models
from generic_local_transition import accumulate, empty_score, finish
from raw_correlation_latent import raw_correlation_sequence
from run_separated_visual_state import make_training_cases
from separated_visual_state import SeparatedVisualState
from split_sensory_pathway import SplitPopulation, train_raw_branch
from diverse_visual_experience import unseen_shape_sequences


OUT = Path('docs/experiments/2026-09-24-split-state-recurrence-results.json')
NAMES = ('archived', 'recurrent', 'no_recurrence', 'state_reset')


@torch.no_grad()
def train_candidates(cases):
    archived = train_models()['learned']
    raw, _, _ = train_raw_branch()
    encoder = SplitPopulation(copy.deepcopy(archived), raw)
    recurrent = SeparatedVisualState(copy.deepcopy(encoder),
                                      recurrence=True, separate_credit=True)
    no_recurrence = SeparatedVisualState(copy.deepcopy(encoder),
                                         recurrence=False, separate_credit=True)
    encoded = [(events, raw_correlation_sequence(events))
               for _, _, events in cases]
    random.Random(3).shuffle(encoded)
    for events, codes in encoded:
        recurrent.reset_state()
        no_recurrence.reset_state()
        for event, code in zip(events, codes):
            recurrent.step(code, event, learn=True)
            no_recurrence.step(code, event, learn=True)
    return dict(archived=archived, recurrent=recurrent,
                no_recurrence=no_recurrence,
                state_reset=copy.deepcopy(recurrent))


@torch.no_grad()
def evaluate(models, cases):
    groups = {family: {name: empty_score() for name in NAMES}
              for family, _, _ in cases}
    overall = {name: empty_score() for name in NAMES}
    exact = {name: empty_score() for name in NAMES}
    quiet = {name: dict(frames=0, false_alarm_pixels=0) for name in NAMES}
    blank = dict(frames=0, recurrent_imagined=0,
                 no_recurrence_imagined=0)
    for family, _, events in cases:
        codes = raw_correlation_sequence(events)
        correlations = correlation_sequence(events)
        for model in models.values():
            model.reset_state()
        for t, (event, code, correlation) in enumerate(
                zip(events, codes, correlations)):
            if family == 'occlusion' and t == 10:
                models['state_reset'].reset_state()
            predictions = dict(
                archived=primitive_to_events(models['archived'].step(correlation)),
                recurrent=models['recurrent'].step(code, event),
                no_recurrence=models['no_recurrence'].step(code, event),
                state_reset=models['state_reset'].step(code, event))
            if code.sum() == 0:
                blank['frames'] += 1
                blank['recurrent_imagined'] += int(
                    models['recurrent'].imagined.sum() > 0)
                blank['no_recurrence_imagined'] += int(
                    models['no_recurrence'].imagined.sum() > 0)
            if 3 <= t < 14:
                target = events[t+1]
                for name, prediction in predictions.items():
                    accumulate(groups[family][name], prediction, target)
                    accumulate(overall[name], prediction, target)
                    if family == 'occlusion' and t == 10:
                        accumulate(exact[name], prediction, target)
                    if target.sum() == 0:
                        quiet[name]['frames'] += 1
                        quiet[name]['false_alarm_pixels'] += int(
                            (prediction >= .5).sum())
    return dict(
        overall={name: finish(score) for name, score in overall.items()},
        groups={family: {name: finish(score) for name, score in rows.items()}
                for family, rows in groups.items()},
        exact_reappearance={name: finish(score) for name, score in exact.items()},
        quiet=quiet, blank_state=blank)


@torch.no_grad()
def matched_pair(models):
    pairs = []
    for direction in (-1, 1):
        obj = dict(shape='diamond', center=(16, 32-2*direction),
                   before=(0, direction), after=(0, direction),
                   hidden=(7, 8, 9))
        events = scene_sequence([obj], background=True)
        codes = raw_correlation_sequence(events)
        for name in ('recurrent', 'no_recurrence', 'state_reset'):
            models[name].reset_state()
        forecasts = {}
        for t, (event, code) in enumerate(zip(events[:11], codes[:11])):
            if t == 10:
                models['state_reset'].reset_state()
            for name in ('recurrent', 'no_recurrence', 'state_reset'):
                forecasts[name] = models[name].step(code, event)
        pairs.append((forecasts, events[11]))
    scores = {}
    for name in ('recurrent', 'no_recurrence', 'state_reset'):
        score = empty_score()
        for forecasts, target in pairs:
            accumulate(score, forecasts[name], target)
        scores[name] = dict(
            **finish(score),
            opposite_forecast_l1=float((pairs[0][0][name]-pairs[1][0][name])
                                       .abs().sum()))
    return scores


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
    pair = matched_pair(models)
    result = dict(training_episodes=len(training),
                  training=training_scores, heldout=heldout_scores,
                  matched_pair=pair,
                  state_weight_sum=float(models['recurrent'].state_weights.sum()),
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        heldout={family: {name: round(score['f1'], 3)
                          for name, score in rows.items()}
                 for family, rows in heldout_scores['groups'].items()},
        reappearance={name: round(score['f1'], 3)
                      for name, score in
                      heldout_scores['exact_reappearance'].items()},
        quiet=heldout_scores['quiet'],
        blank=heldout_scores['blank_state'],
        pair={name: dict(f1=round(score['f1'], 3),
                         opposite_l1=round(score['opposite_forecast_l1'], 3))
              for name, score in pair.items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
