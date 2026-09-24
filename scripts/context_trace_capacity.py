"""Evaluation-only local readout of frozen reappearance context codes."""

import copy
import json
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
import torch

from correlation_input_latent import correlation_sequence
from correlation_latent_robustness import make_robust_cases, scene_sequence
from diverse_visual_experience import TRAIN_SHAPES
from frozen_code_capacity import local_features, scatter_local_decoder
from generic_local_transition import accumulate, empty_score, finish
from local_context_trace import train_candidates
from run_separated_visual_state import make_training_cases


OUT = Path('docs/experiments/2026-09-24-context-trace-capacity-results.json')


def readout_training_cases():
    directions = ((-1, 0), (1, 0), (0, -1), (0, 1))
    cases = []
    for shape in TRAIN_SHAPES:
        for dy, dx in directions:
            for speed in (1, 2):
                for background in (True, False):
                    obj = dict(shape=shape, center=(16, 32),
                               before=(dy*speed, dx*speed),
                               after=(dy*speed, dx*speed),
                               hidden=(7, 8, 9))
                    cases.append((shape, scene_sequence([obj],
                                                        background=background)))
    return cases


@torch.no_grad()
def sources_at_reappearance(model, events, *, reset_trace=False):
    model.reset_state()
    codes = correlation_sequence(events)
    for t in range(11):
        if t == 10 and reset_trace:
            model.trace.zero_()
        model.step(events[t], codes[t])
    return model.context.previous_sources.copy()


@torch.no_grad()
def fit_selected_decoder(model, cases):
    rows = []
    columns = []
    values = []
    targets = []
    total_events = reachable_events = 0
    for _, events in cases:
        sources = sources_at_reappearance(model, events)
        features = local_features(sources, units=model.context.units)
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
                               shape=(len(targets), model.context.units*25),
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
    reach = {name: dict(target_events=0, reachable_events=0)
             for name in models}
    by_shape = {}
    for shape, events in cases:
        group = by_shape.setdefault(shape, {name: empty_score() for name in models})
        for name, model in models.items():
            sources = sources_at_reappearance(
                model, events, reset_trace=name == 'trace_reset')
            prediction, reachable = scatter_local_decoder(
                weights[name], sources, units=model.context.units)
            target = events[11]
            accumulate(scores[name], prediction, target)
            accumulate(group[name], prediction, target)
            reach[name]['target_events'] += int(target.sum())
            reach[name]['reachable_events'] += int(target[:, reachable].sum())
    return dict(overall={name: finish(value) for name, value in scores.items()},
                by_shape={shape: {name: finish(value)
                                  for name, value in rows.items()}
                          for shape, rows in by_shape.items()},
                coverage={name: dict(**row,
                                     reachable_fraction=row['reachable_events']
                                     /row['target_events'])
                          for name, row in reach.items()})


@torch.no_grad()
def matched_pair(models, weights):
    pairs = []
    for direction in (-1, 1):
        obj = dict(shape='diamond', center=(16, 32-2*direction),
                   before=(0, direction), after=(0, direction),
                   hidden=(7, 8, 9))
        events = scene_sequence([obj], background=True)
        predictions = {}
        source_lists = {}
        for name, model in models.items():
            sources = sources_at_reappearance(
                model, events, reset_trace=name == 'trace_reset')
            predictions[name], _ = scatter_local_decoder(
                weights[name], sources, units=model.context.units)
            source_lists[name] = sources
        pairs.append((predictions, source_lists, events[11]))
    result = {}
    for name in models:
        score = empty_score()
        for predictions, _, target in pairs:
            accumulate(score, predictions[name], target)
        result[name] = dict(**finish(score),
            opposite_forecast_l1=float((pairs[0][0][name]-pairs[1][0][name])
                                       .abs().sum()),
            different_sources=len(set(pairs[0][1][name])
                                  .symmetric_difference(pairs[1][1][name])))
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    _, trained = train_candidates(make_training_cases())
    models = dict(trace=trained['trace'],
                  zero_trace=trained['zero_trace'],
                  trace_reset=copy.deepcopy(trained['trace']))
    training = readout_training_cases()
    weights = {}
    fit = {}
    for name in ('trace', 'zero_trace'):
        weights[name], fit[name] = fit_selected_decoder(models[name], training)
    weights['trace_reset'] = weights['trace']
    heldout = [(meta['shape'], events)
               for family, meta, events in make_robust_cases()
               if family == 'occlusion']
    training_scores = evaluate(models, weights, training)
    heldout_scores = evaluate(models, weights, heldout)
    pair = matched_pair(models, weights)
    result = dict(training_cases=len(training), fit=fit,
                  training=training_scores, heldout=heldout_scores,
                  matched_pair=pair,
                  elapsed_seconds=time.perf_counter()-started)
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(
        training={name: round(row['f1'], 3)
                  for name, row in training_scores['overall'].items()},
        heldout={name: round(row['f1'], 3)
                 for name, row in heldout_scores['overall'].items()},
        coverage={name: round(row['reachable_fraction'], 3)
                  for name, row in heldout_scores['coverage'].items()},
        pair={name: dict(f1=round(row['f1'], 3),
                         forecast_l1=round(row['opposite_forecast_l1'], 3),
                         different_sources=row['different_sources'])
              for name, row in pair.items()},
        elapsed_seconds=result['elapsed_seconds'])), flush=True)


if __name__ == '__main__':
    main()
