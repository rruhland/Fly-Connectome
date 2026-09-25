"""Evaluation-only future-event capacity of frozen history-gated units."""

import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
import torch

from context_trace_capacity import readout_training_cases
from correlation_latent_robustness import scene_sequence
from frozen_code_capacity import local_features, scatter_local_decoder
from generic_local_transition import accumulate, empty_score, finish
from history_gated_code import sources_at_reappearance, train_models_for_history
from history_gated_prediction import expanded_interruption_cases
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-25-history-event-capacity-results.json')


def readout_sources(sources, current, *, units, polarity_split):
    if not polarity_split:
        return sources
    return [(y, x, unit+units*int(current[1, y, x] > 0))
            for y, x, unit in sources]


def selected_sources(model, events, *, polarity_split, reset_trace=False):
    sources = sources_at_reappearance(model, events,
                                      reset_trace=reset_trace)
    return readout_sources(sources, events[10], units=model.history.units,
                           polarity_split=polarity_split)


@torch.no_grad()
def fit_selected(model, cases, *, polarity_split):
    units = model.history.units*(2 if polarity_split else 1)
    rows, columns, values, targets = [], [], [], []
    total_events = reachable_events = 0
    for _, events in cases:
        sources = selected_sources(model, events,
                                   polarity_split=polarity_split)
        features = local_features(sources, units=units)
        future = events[11].reshape(2, -1)
        total_events += int(future.sum())
        for pixel, entries in features.items():
            row = len(targets)
            for feature, value in entries.items():
                rows.append(row)
                columns.append(feature)
                values.append(value)
            target = future[:, pixel].tolist()
            targets.append(target)
            reachable_events += int(sum(target))
    matrix = sparse.csr_matrix((values, (rows, columns)),
                               shape=(len(targets), units*25),
                               dtype=np.float64)
    labels = np.asarray(targets, dtype=np.float64)
    weights = np.stack([lsqr(matrix, labels[:, polarity], damp=1.,
                              atol=1e-5, btol=1e-5, iter_lim=200)[0]
                        for polarity in range(2)])
    return torch.from_numpy(weights).float(), dict(
        rows=len(targets), nonzero_features=matrix.nnz,
        target_events=total_events, reachable_events=reachable_events,
        reachable_fraction=reachable_events/total_events)


@torch.no_grad()
def evaluate(models, weights, cases):
    scores = {name: empty_score() for name in models}
    groups = {}
    reach = {name: dict(target_events=0, reachable_events=0)
             for name in models}
    for label, _, events in cases:
        group = groups.setdefault(label,
                                  {name: empty_score() for name in models})
        target = events[11]
        for name, model in models.items():
            polarity_split = name.endswith('_polarity')
            sources = selected_sources(
                model, events, polarity_split=polarity_split,
                reset_trace=name == 'reset_polarity')
            units = model.history.units*(2 if polarity_split else 1)
            prediction, reachable = scatter_local_decoder(
                weights[name], sources, units=units)
            accumulate(scores[name], prediction, target)
            accumulate(group[name], prediction, target)
            reach[name]['target_events'] += int(target.sum())
            reach[name]['reachable_events'] += int(target[:, reachable].sum())
    return dict(overall={name: finish(score) for name, score in scores.items()},
                by_direction={label: {name: finish(score)
                                      for name, score in methods.items()}
                              for label, methods in groups.items()},
                coverage={name: dict(**row,
                                     reachable_fraction=row['reachable_events']
                                     /row['target_events'])
                          for name, row in reach.items()})


@torch.no_grad()
def matched_pair(models, weights):
    pair = []
    for direction in (-1, 1):
        obj = dict(shape='diamond', center=(16, 32-2*direction),
                   before=(0, direction), after=(0, direction),
                   hidden=(7, 8, 9))
        events = scene_sequence([obj], background=True)
        predictions = {}
        for name, model in models.items():
            polarity_split = name.endswith('_polarity')
            sources = selected_sources(
                model, events, polarity_split=polarity_split,
                reset_trace=name == 'reset_polarity')
            units = model.history.units*(2 if polarity_split else 1)
            predictions[name], _ = scatter_local_decoder(
                weights[name], sources, units=units)
        pair.append((predictions, events[11]))
    result = {}
    for name in models:
        score = empty_score()
        for predictions, target in pair:
            accumulate(score, predictions[name], target)
        result[name] = dict(**finish(score),
            opposite_forecast_l1=float((pair[0][0][name]-pair[1][0][name])
                                       .abs().sum()))
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    codes = train_models_for_history(make_training_cases())
    training = readout_training_cases()
    heldout = expanded_interruption_cases()
    models = dict(learned_unit=codes['learned'],
                  random_unit=codes['random'],
                  learned_polarity=codes['learned'],
                  random_polarity=codes['random'],
                  reset_polarity=codes['learned'])
    weights, fit = {}, {}
    for name in ('learned_unit', 'random_unit',
                 'learned_polarity', 'random_polarity'):
        weights[name], fit[name] = fit_selected(
            models[name], training, polarity_split=name.endswith('_polarity'))
    weights['reset_polarity'] = weights['learned_polarity']
    training_cases = [('training', shape, events)
                      for shape, events in training]
    training_scores = evaluate(models, weights, training_cases)
    heldout_scores = evaluate(models, weights, heldout)
    pair = matched_pair(models, weights)
    result = dict(training_cases=len(training), heldout_cases=len(heldout),
                  fit=fit, training=training_scores, heldout=heldout_scores,
                  matched_pair=pair,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        training={name: round(row['f1'], 3)
                  for name, row in training_scores['overall'].items()},
        heldout={name: round(row['f1'], 3)
                 for name, row in heldout_scores['overall'].items()},
        by_direction={direction: {name: dict(
            f1=round(row['f1'], 3), tp=row['tp'])
            for name, row in methods.items()}
            for direction, methods in
            heldout_scores['by_direction'].items()},
        coverage={name: round(row['reachable_fraction'], 3)
                  for name, row in heldout_scores['coverage'].items()},
        pair={name: dict(f1=round(row['f1'], 3),
                         opposite_l1=round(row['opposite_forecast_l1'], 3))
              for name, row in pair.items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
