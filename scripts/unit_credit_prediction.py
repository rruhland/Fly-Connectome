"""Compare frame-global and per-unit local predictive credit."""

import copy
import json
import random
import time
from pathlib import Path

import torch

from correlation_input_latent import correlation_sequence, primitive_to_events
from correlation_latent_robustness import make_robust_cases
from diverse_visual_experience import unseen_shape_sequences
from generic_local_transition import accumulate, empty_score, finish
from history_gated_code import train_models_for_history
from history_gated_prediction import (LocalEventReadout,
                                      expanded_interruption_cases,
                                      matched_pair)
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-24-unit-credit-prediction-results.json')
NAMES = ('archived', 'global', 'unit', 'random_unit', 'trace_reset')


@torch.no_grad()
def train_candidates(cases):
    codes = train_models_for_history(cases)
    archived = copy.deepcopy(codes['learned'].motion)
    models = dict(
        global_=LocalEventReadout(copy.deepcopy(codes['learned'])),
        unit=LocalEventReadout(copy.deepcopy(codes['learned']),
                               unit_normalized=True),
        random_unit=LocalEventReadout(copy.deepcopy(codes['random']),
                                      unit_normalized=True))
    models['global'] = models.pop('global_')
    encoded = [(events, correlation_sequence(events))
               for _, _, events in cases]
    random.Random(3).shuffle(encoded)
    for events, correlations in encoded:
        for model in models.values():
            model.reset_state()
        for event, correlation in zip(events, correlations):
            for model in models.values():
                model.step(event, correlation, learn=True)
    models['trace_reset'] = copy.deepcopy(models['unit'])
    return archived, models


@torch.no_grad()
def evaluate(archived, models, cases):
    groups = {family: {name: empty_score() for name in NAMES}
              for family, _, _ in cases}
    exact = {name: empty_score() for name in NAMES}
    exact_groups = {family: {name: empty_score() for name in NAMES}
                    for family, _, _ in cases
                    if family == 'interruption' or family.startswith('occlusion')}
    quiet = {name: dict(frames=0, false_alarm_pixels=0) for name in NAMES}
    for family, _, events in cases:
        correlations = correlation_sequence(events)
        archived.reset_state()
        for model in models.values():
            model.reset_state()
        for t, (event, correlation) in enumerate(zip(events, correlations)):
            selected = (family == 'interruption'
                        or family.startswith('occlusion')) and t == 10
            if selected:
                models['trace_reset'].code.trace.zero_()
            baseline = primitive_to_events(archived.step(correlation))
            predictions = {'archived': baseline}
            for name, model in models.items():
                predictions[name] = torch.maximum(
                    baseline, model.step(event, correlation))
            if 3 <= t < 14:
                target = events[t+1]
                for name, prediction in predictions.items():
                    accumulate(groups[family][name], prediction, target)
                    if selected:
                        accumulate(exact[name], prediction, target)
                        accumulate(exact_groups[family][name],
                                   prediction, target)
                    if target.sum() == 0:
                        quiet[name]['frames'] += 1
                        quiet[name]['false_alarm_pixels'] += int(
                            (prediction >= .5).sum())
    return dict(groups={family: {name: finish(row) for name, row in rows.items()}
                        for family, rows in groups.items()},
                exact={name: finish(row) for name, row in exact.items()},
                exact_by_family={family: {name: finish(row)
                                          for name, row in rows.items()}
                                 for family, rows in exact_groups.items()},
                quiet=quiet)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    training = make_training_cases()
    archived, models = train_candidates(training)
    training_scores = evaluate(archived, models, training)
    heldout = [('single', shape, events)
               for shape, _, _, _, events in unseen_shape_sequences()]
    heldout += [(family, 'unseen', events)
                for family, _, events in make_robust_cases()]
    heldout_scores = evaluate(archived, models, heldout)
    expanded_scores = evaluate(archived, models,
                               expanded_interruption_cases())
    pair = matched_pair(archived, models)
    result = dict(training_episodes=len(training),
                  training=training_scores, heldout=heldout_scores,
                  expanded=expanded_scores, matched_pair=pair,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        training_exact={name: round(row['f1'], 3)
                        for name, row in training_scores['exact'].items()},
        single={name: round(row['f1'], 3)
                for name, row in heldout_scores['groups']['single'].items()},
        expanded_exact={name: round(row['f1'], 3)
                        for name, row in expanded_scores['exact'].items()},
        expanded_by_direction={family: {name: dict(
            f1=round(row['f1'], 3), tp=row['tp'])
            for name, row in rows.items()}
            for family, rows in expanded_scores['exact_by_family'].items()},
        quiet=heldout_scores['quiet'],
        pair={name: dict(f1=round(row['f1'], 3),
                         opposite_l1=round(row['opposite_forecast_l1'], 3))
              for name, row in pair.items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
